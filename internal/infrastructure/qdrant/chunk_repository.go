package qdrant

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/variant"
)

const (
	upsertBatchSize = 256 // keeps upserts clear of gRPC message limits (§6.1)
)

type ChunkRepository struct {
	c *Client
}

func NewChunkRepository(c *Client) *ChunkRepository { return &ChunkRepository{c: c} }

var _ chunk.ChunkRepository = (*ChunkRepository)(nil)

// ReplaceForDocument: delete by document_id filter, then batched upserts, all
// wait=true. Crash-safe and idempotent, not atomic (§6.1). An empty parents
// slice is valid and means delete-only — nothing is re-inserted.
func (r *ChunkRepository) ReplaceForDocument(ctx context.Context, variantName, documentID string, parents []*chunk.ParentChunk) error {
	for _, p := range parents {
		if p.DocumentID != documentID {
			return fmt.Errorf("chunk repository: parent %s has document_id %q, want %q (never stale)", p.ID, p.DocumentID, documentID)
		}
	}
	collection := variant.CollectionPrefix + variantName
	if err := r.deleteByDocumentIn(ctx, collection, documentID); err != nil {
		return err
	}
	points := flattenParents(parents)
	for start := 0; start < len(points); start += upsertBatchSize {
		batch := points[start:min(start+upsertBatchSize, len(points))]
		_, err := r.c.q.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: collection,
			Wait:           qdrant.PtrOf(true),
			Points:         batch,
		})
		if err != nil {
			return fmt.Errorf("upserting chunks for document %s into %s: %w", documentID, collection, err)
		}
	}
	return nil
}

// DeleteByDocument purges the document's chunks in EVERY variant collection,
// discovered by prefix so undeclared leftovers are purged too (§6.1).
//
// A collection can vanish between the ListCollections snapshot above and the
// per-collection delete below (e.g. a concurrent DeleteByVariant) — that is a
// benign race, not a purge failure, so a NotFound from deleteByDocumentIn is
// skipped rather than treated as an error. Any other error is accumulated via
// errors.Join rather than returned immediately, so one broken variant
// collection does not shield the rest of the purge from running.
func (r *ChunkRepository) DeleteByDocument(ctx context.Context, documentID string) error {
	collections, err := r.c.q.ListCollections(ctx)
	if err != nil {
		return fmt.Errorf("listing collections: %w", err)
	}
	var errs []error
	for _, name := range collections {
		if !strings.HasPrefix(name, variant.CollectionPrefix) {
			continue
		}
		if err := r.deleteByDocumentIn(ctx, name, documentID); err != nil {
			if status.Code(err) == codes.NotFound {
				continue
			}
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (r *ChunkRepository) deleteByDocumentIn(ctx context.Context, collection, documentID string) error {
	_, err := r.c.q.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: collection,
		Wait:           qdrant.PtrOf(true),
		Points: qdrant.NewPointsSelectorFilter(&qdrant.Filter{
			Must: []*qdrant.Condition{qdrant.NewMatch("document_id", documentID)},
		}),
	})
	if err != nil {
		return fmt.Errorf("deleting chunks of document %s in %s: %w", documentID, collection, err)
	}
	return nil
}

// DeleteByVariant drops the collection — instant and total (§5.1). Idempotent:
// go-client's DeleteCollection errors ("failed to delete collection") on an
// already-absent collection instead of treating it as a no-op, so existence
// is checked first.
func (r *ChunkRepository) DeleteByVariant(ctx context.Context, variantName string) error {
	name := variant.CollectionPrefix + variantName
	exists, err := r.c.q.CollectionExists(ctx, name)
	if err != nil {
		return fmt.Errorf("checking collection %s: %w", name, err)
	}
	if !exists {
		return nil
	}
	if err := r.c.q.DeleteCollection(ctx, name); err != nil {
		return fmt.Errorf("dropping collection for variant %s: %w", variantName, err)
	}
	return nil
}

// SearchChildren queries the variant's collection directly by name. A
// declared variant whose backing collection has been dropped (a setup
// fault — e.g. DeleteByVariant ran without updating the registry, or the
// registry disagrees with what Setup provisioned) surfaces as an error from
// the underlying Query call, not as an empty-but-successful result set.
func (r *ChunkRepository) SearchChildren(ctx context.Context, q chunk.SearchQuery) ([]*chunk.ChildHit, error) {
	if err := q.Validate(); err != nil {
		return nil, err
	}
	req := &qdrant.QueryPoints{
		CollectionName: variant.CollectionPrefix + q.Variant,
		Limit:          qdrant.PtrOf(uint64(q.Limit)),
		WithPayload:    qdrant.NewWithPayload(true),
	}
	switch {
	case q.Dense != nil:
		req.Query = qdrant.NewQueryDense(q.Dense.Vector)
		req.Using = qdrant.PtrOf("dense")
		req.Params = &qdrant.SearchParams{HnswEf: qdrant.PtrOf(uint64(q.Dense.Ef))}
	case q.Sparse != nil:
		req.Query = qdrant.NewQuerySparse(q.Sparse.Vector.Indices, q.Sparse.Vector.Weights)
		req.Using = qdrant.PtrOf("lexical")
	}

	points, err := r.c.q.Query(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("searching %s: %w", req.CollectionName, err)
	}
	hits := make([]*chunk.ChildHit, 0, len(points))
	for _, p := range points {
		hits = append(hits, childHitFromPoint(p))
	}
	return hits, nil
}

func flattenParents(parents []*chunk.ParentChunk) []*qdrant.PointStruct {
	var points []*qdrant.PointStruct
	for _, p := range parents {
		for _, c := range p.Children {
			payload := map[string]any{
				"document_id":    p.DocumentID,
				"ordinal":        c.Ordinal,
				"text":           c.Text,
				"embed_text":     c.EmbedText,
				"token_count":    c.TokenCount,
				"title":          p.Title,
				"parent_id":      p.ID,
				"parent_ordinal": p.Ordinal,
				"parent_text":    p.Text,
			}
			points = append(points, &qdrant.PointStruct{
				Id: qdrant.NewID(c.ID),
				Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
					"dense":   qdrant.NewVectorDense(c.Dense),
					"lexical": qdrant.NewVectorSparse(c.Sparse.Indices, c.Sparse.Weights),
				}),
				Payload: qdrant.NewValueMap(payload),
			})
		}
	}
	return points
}

func childHitFromPoint(p *qdrant.ScoredPoint) *chunk.ChildHit {
	pl := p.Payload
	return &chunk.ChildHit{
		ChildChunkID:  p.Id.GetUuid(),
		Score:         float64(p.Score),
		Text:          pStr(pl, "text"),
		ParentChunkID: pStr(pl, "parent_id"),
		ParentText:    pStr(pl, "parent_text"),
		DocumentID:    pStr(pl, "document_id"),
		Title:         pStr(pl, "title"),
	}
}
