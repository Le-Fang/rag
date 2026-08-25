// Package ingestion implements the per-variant indexing pipeline (§6):
// chunk → header → embed → encode — fully in memory — then one
// ReplaceForDocument write.
package ingestion

import (
	"context"
	"fmt"
	"log/slog"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/lexical"
	"poc-rag/internal/domain/variant"
)

type Service struct {
	chunker   chunk.Chunker
	embedders map[string]embedding.EmbedderPort // profile name → adapter (§9.1)
	chunks    chunk.ChunkRepository
	registry  *variant.Registry
	log       *slog.Logger
}

func New(chunker chunk.Chunker, embedders map[string]embedding.EmbedderPort,
	chunks chunk.ChunkRepository, registry *variant.Registry, log *slog.Logger) *Service {
	return &Service{chunker: chunker, embedders: embedders, chunks: chunks, registry: registry, log: log}
}

// Index re-chunks and re-embeds doc into the named variant. Idempotent:
// backed by ReplaceForDocument (§6.1).
func (s *Service) Index(ctx context.Context, doc *document.Document, variantName string) error {
	v, err := s.registry.Get(variantName)
	if err != nil {
		return err
	}
	emb, ok := s.embedders[v.EmbedderProfile]
	if !ok {
		// startup builds one adapter per referenced profile, so this is a wiring bug
		return fmt.Errorf("variant %s: no adapter for embedder profile %q", v.Name, v.EmbedderProfile)
	}
	enc, err := lexical.NewEncoder(lexical.Params{
		Tokenizer:         v.Params.LexicalTokenizer,
		NormalizeNumerals: v.Params.NormalizeNumerals,
		K1:                v.Params.BM25K1,
		B:                 v.Params.BM25B,
		AvgLen:            v.Params.BM25AvgLen,
	})
	if err != nil {
		return fmt.Errorf("variant %s: %w", v.Name, err)
	}

	parents, err := s.chunker.Chunk(ctx, doc, chunk.Config{
		ParentTokens:   v.Params.ParentTokens,
		ChildTokens:    v.Params.ChildTokens,
		ChildOverlap:   v.Params.ChildOverlap,
		HeaderStrategy: v.HeaderStrategy,
	})
	if err != nil {
		return fmt.Errorf("variant %s: %w", v.Name, err)
	}

	// Embed everything before any write (§6.1): a failure here aborts with
	// nothing persisted.
	var children []*chunk.ChildChunk
	var texts []string
	for _, p := range parents {
		for _, c := range p.Children {
			children = append(children, c)
			texts = append(texts, c.EmbedText)
		}
	}
	if len(children) > 0 {
		vecs, err := emb.Embed(ctx, texts, embedding.KindDocument)
		if err != nil {
			return fmt.Errorf("embedding document %s for variant %s: %w", doc.ID, v.Name, err)
		}
		if len(vecs) != len(children) {
			return fmt.Errorf("embedding document %s: got %d vectors for %d children", doc.ID, len(vecs), len(children))
		}
		for i, c := range children {
			c.Dense = embedding.Normalize(vecs[i])
			c.Sparse = enc.EncodeDocument(c.Text) // header-free text only (§7.1)
		}
	}

	if err := s.chunks.ReplaceForDocument(ctx, v.Name, doc.ID, parents); err != nil {
		return err
	}
	s.log.InfoContext(ctx, "indexed document",
		"document_id", doc.ID, "variant", v.Name, "parents", len(parents), "children", len(children))
	return nil
}
