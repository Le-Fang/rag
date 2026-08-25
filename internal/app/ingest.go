package app

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/spf13/cobra"
	"go.uber.org/fx"

	"poc-rag/internal/application/documents"
	"poc-rag/internal/application/ingestion"
	docdomain "poc-rag/internal/domain/document"
	"poc-rag/internal/domain/variant"
)

// corpusLine mirrors the POST /v1/documents body (§6.2): one document per line.
type corpusLine struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	Content string `json:"content"`
}

func newIngestCmd() *cobra.Command {
	var file, variantName string
	cmd := &cobra.Command{
		Use:   "ingest",
		Short: "Bulk-load a JSONL corpus and/or (re-)index documents into variants",
		RunE: func(cmd *cobra.Command, args []string) error {
			// fx only CONSTRUCTS here; the work runs between Start and Stop
			// under cmd.Context(), so Ctrl+C (SIGINT/SIGTERM via Execute's
			// signal context) aborts the ingest instead of being ignored.
			var deps struct {
				fx.In
				DocSvc  *documents.Service
				IngSvc  *ingestion.Service
				DocRepo docdomain.DocumentRepository
				Reg     *variant.Registry
				Log     *slog.Logger
			}
			app := fx.New(modules(configPath), fx.Populate(&deps))
			if err := app.Err(); err != nil {
				return err
			}
			if err := app.Start(cmd.Context()); err != nil {
				return err
			}
			runErr := runIngest(cmd.Context(), deps.DocSvc, deps.IngSvc, deps.DocRepo, deps.Reg,
				deps.Log, file, variantName)
			// Stop on a fresh context: cleanup (closing the Qdrant client)
			// must still run after Ctrl+C canceled cmd.Context(). fx applies
			// its own StopTimeout internally.
			if err := app.Stop(context.Background()); err != nil {
				return errors.Join(runErr, err)
			}
			return runErr
		},
	}
	cmd.Flags().StringVar(&file, "file", "", "JSONL corpus to import (one document per line)")
	cmd.Flags().StringVar(&variantName, "variant", "", "index into this variant only (default: all declared)")
	return cmd
}

func runIngest(ctx context.Context, docSvc *documents.Service, ingSvc *ingestion.Service,
	docRepo docdomain.DocumentRepository, reg *variant.Registry, log *slog.Logger,
	file, variantName string) error {

	targets, err := targetVariants(reg, variantName)
	if err != nil {
		return err
	}

	index := func(doc *docdomain.Document) (failed bool) {
		for _, v := range targets {
			if err := ingSvc.Index(ctx, doc, v); err != nil {
				log.ErrorContext(ctx, "indexing failed", "document_id", doc.ID, "variant", v, "error", err)
				failed = true
			}
		}
		return failed
	}

	skips := 0
	if file != "" {
		f, err := os.Open(file)
		if err != nil {
			return fmt.Errorf("opening corpus: %w", err)
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 1024), 32<<20) // very long documents fit one line
		lineNo := 0
		for scanner.Scan() {
			// A dead context aborts the run; skip-and-count is for line-level
			// faults, not for burning through the file after Ctrl+C.
			if err := ctx.Err(); err != nil {
				return fmt.Errorf("ingest aborted: %w", err)
			}
			lineNo++
			var line corpusLine
			if err := json.Unmarshal(scanner.Bytes(), &line); err != nil {
				log.ErrorContext(ctx, "skipping unparseable line", "line", lineNo, "error", err)
				skips++
				continue
			}
			doc := &docdomain.Document{ID: line.ID, Title: line.Title, Content: line.Content}
			saved, _, err := docSvc.Save(ctx, doc, "") // upsert by id; purges stale chunks (§6.1)
			if err != nil {
				log.ErrorContext(ctx, "skipping document", "line", lineNo, "error", err)
				skips++
				continue
			}
			if index(saved) {
				skips++
			} else {
				log.InfoContext(ctx, "ingested", "line", lineNo, "document_id", saved.ID)
			}
		}
		// Deliberate: stream errors (e.g. a line over the 32MB buffer) abort
		// the whole run; line-level errors above skip-and-count instead (§6.2).
		if err := scanner.Err(); err != nil {
			return fmt.Errorf("reading corpus: %w", err)
		}
	} else {
		// re-index mode: stream what is already stored (§6.2)
		err := docRepo.Each(ctx, func(doc *docdomain.Document) error {
			if err := ctx.Err(); err != nil {
				return fmt.Errorf("ingest aborted: %w", err)
			}
			if index(doc) {
				skips++
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	if skips > 0 {
		return fmt.Errorf("ingest finished with %d skipped documents", skips)
	}
	return nil
}

func targetVariants(reg *variant.Registry, name string) ([]string, error) {
	if name != "" {
		if _, err := reg.Get(name); err != nil {
			return nil, err
		}
		return []string{name}, nil
	}
	var out []string
	for _, v := range reg.List() {
		out = append(out, v.Name)
	}
	return out, nil
}
