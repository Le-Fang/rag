package qdrant

import (
	"context"
	"fmt"
	"time"

	"github.com/qdrant/go-client/qdrant"

	"poc-rag/internal/domain/document"
)

// documentsPageSize is deliberately small: unlike the variant registry, a
// documents point can carry a very long content body, so a 64-point page
// (the variant registry's page size) can approach the 64 MB gRPC recv cap.
const documentsPageSize = 8

type DocumentRepository struct {
	c *Client
}

func NewDocumentRepository(c *Client) *DocumentRepository { return &DocumentRepository{c: c} }

var _ document.DocumentRepository = (*DocumentRepository)(nil)

func (r *DocumentRepository) Save(ctx context.Context, doc *document.Document) error {
	payload := map[string]any{
		"title":   doc.Title,
		"content": doc.Content,
	}
	// Omit zero timestamps entirely rather than writing out year 0001 (mirrors
	// variant_repository's CreatedAt handling).
	if !doc.CreatedAt.IsZero() {
		payload["created_at"] = doc.CreatedAt.Format(time.RFC3339)
	}
	if !doc.UpdatedAt.IsZero() {
		payload["updated_at"] = doc.UpdatedAt.Format(time.RFC3339)
	}

	_, err := r.c.q.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: documentsCollection,
		Wait:           qdrant.PtrOf(true),
		Points: []*qdrant.PointStruct{{
			Id: qdrant.NewID(doc.ID),
			// Payload-only point: Vectors must be an empty map, not nil — a nil
			// Vectors field is rejected server-side even in a vector-less
			// collection (Task 2 finding; see claims_integration_test.go).
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{}),
			Payload: qdrant.NewValueMap(payload),
		}},
	})
	if err != nil {
		return fmt.Errorf("saving document %s: %w", doc.ID, err)
	}
	return nil
}

func (r *DocumentRepository) Load(ctx context.Context, id string) (*document.Document, error) {
	points, err := r.c.q.Get(ctx, &qdrant.GetPoints{
		CollectionName: documentsCollection,
		Ids:            []*qdrant.PointId{qdrant.NewID(id)},
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		// Qdrant returns codes.NotFound only when the collection itself is
		// missing (an infrastructure fault), never for a missing point ID —
		// that case is err=nil, points=[], handled below. Do not map this
		// branch to document.ErrNotFound: that would make an outage look
		// like "safe to create" to replace flows.
		return nil, fmt.Errorf("loading document %s: %w", id, err)
	}
	if len(points) == 0 {
		return nil, fmt.Errorf("document %s: %w", id, document.ErrNotFound)
	}
	return documentFromPayload(id, points[0].Payload), nil
}

func (r *DocumentRepository) Delete(ctx context.Context, id string) error {
	_, err := r.c.q.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: documentsCollection,
		Wait:           qdrant.PtrOf(true),
		Points:         qdrant.NewPointsSelector(qdrant.NewID(id)),
	})
	if err != nil {
		return fmt.Errorf("deleting document %s: %w", id, err)
	}
	return nil // deleting a missing point succeeds — idempotent (§6.1)
}

func (r *DocumentRepository) Each(ctx context.Context, fn func(*document.Document) error) error {
	// scrollAll (client.go) paginates via NextPageOffset — using the last-seen
	// point ID as the offset would re-index one document per page boundary.
	// documentsPageSize is small: a documents point can carry a very long
	// content body, so a larger page can approach the 64MB gRPC recv cap.
	return r.c.scrollAll(ctx, documentsCollection, documentsPageSize, qdrant.NewWithPayload(true),
		func(p *qdrant.RetrievedPoint) error {
			return fn(documentFromPayload(p.Id.GetUuid(), p.Payload))
		})
}

func documentFromPayload(id string, p map[string]*qdrant.Value) *document.Document {
	doc := &document.Document{
		ID:      id,
		Title:   pStr(p, "title"),
		Content: pStr(p, "content"),
	}
	if t := pTimePtr(p, "created_at"); t != nil {
		doc.CreatedAt = *t
	}
	if t := pTimePtr(p, "updated_at"); t != nil {
		doc.UpdatedAt = *t
	}
	return doc
}
