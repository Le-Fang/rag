package documents

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/google/uuid"

	"poc-rag/internal/application/apperror"
	"poc-rag/internal/application/ingestion"
	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/variant"
)

type Service struct {
	docs     document.DocumentRepository
	chunks   chunk.ChunkRepository
	ingest   *ingestion.Service
	registry *variant.Registry
	log      *slog.Logger
}

func New(docs document.DocumentRepository, chunks chunk.ChunkRepository,
	ingest *ingestion.Service, registry *variant.Registry, log *slog.Logger) *Service {
	return &Service{docs: docs, chunks: chunks, ingest: ingest, registry: registry, log: log}
}

// Save upserts by id with replace semantics (§6.1): purge the document's
// chunks in EVERY variant first, then save, then optionally index inline.
// Chunks may be missing (heals via ingest), never stale. The bool reports
// whether the save created a new document (true) or replaced an existing one
// (false) — the handler answers 201 vs 200 with it.
func (s *Service) Save(ctx context.Context, doc *document.Document, variantName string) (*document.Document, bool, error) {
	if err := doc.Validate(); err != nil {
		return nil, false, err
	}
	if doc.ID == "" {
		doc.ID = uuid.NewString()
	} else if _, err := uuid.Parse(doc.ID); err != nil {
		return nil, false, apperror.Invalid("id must be a UUID")
	}
	if variantName != "" {
		if _, err := s.registry.Get(variantName); err != nil {
			return nil, false, err // fail before any write on a typo'd variant
		}
	}

	// 1. purge everywhere — no variant may serve chunks of a superseded body
	if err := s.chunks.DeleteByDocument(ctx, doc.ID); err != nil {
		return nil, false, err
	}

	// 2. save the document point, preserving created_at on replace
	now := time.Now().UTC()
	doc.CreatedAt, doc.UpdatedAt = now, now
	created := false
	existing, err := s.docs.Load(ctx, doc.ID)
	switch {
	case err == nil:
		doc.CreatedAt = existing.CreatedAt
	case errors.Is(err, document.ErrNotFound):
		created = true
	default:
		return nil, false, err // a transient load failure must not silently reset created_at
	}
	if err := s.docs.Save(ctx, doc); err != nil {
		return nil, false, err
	}

	// 3. inline indexing when requested; other variants heal on next ingest
	if variantName != "" {
		if err := s.ingest.Index(ctx, doc, variantName); err != nil {
			s.log.WarnContext(ctx, "inline index failed", "document_id", doc.ID, "variant", variantName, "error", err)
			return nil, false, err
		}
	}
	return doc, created, nil
}

func (s *Service) Get(ctx context.Context, id string) (*document.Document, error) {
	if _, err := uuid.Parse(id); err != nil {
		return nil, apperror.Invalid("id must be a UUID")
	}
	return s.docs.Load(ctx, id)
}

// Delete removes chunks in every variant first, then the document (§6.1):
// a crash leaves a chunkless document — harmless — never orphan chunks.
func (s *Service) Delete(ctx context.Context, id string) error {
	if _, err := uuid.Parse(id); err != nil {
		return apperror.Invalid("id must be a UUID")
	}
	if err := s.chunks.DeleteByDocument(ctx, id); err != nil {
		return err
	}
	return s.docs.Delete(ctx, id)
}
