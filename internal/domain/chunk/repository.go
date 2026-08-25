package chunk

import (
	"context"
	"errors"
	"fmt"

	"poc-rag/internal/domain/lexical"
)

type ChunkRepository interface {
	// ReplaceForDocument deletes and rewrites all chunks for (variant, documentID).
	// Idempotent re-ingest; the adapter flattens parents into child points (§5.1).
	ReplaceForDocument(ctx context.Context, variantName, documentID string, parents []*ParentChunk) error
	DeleteByDocument(ctx context.Context, documentID string) error // all variants
	DeleteByVariant(ctx context.Context, variantName string) error // drops the collection
	SearchChildren(ctx context.Context, q SearchQuery) ([]*ChildHit, error)
}

// SearchQuery expresses one channel per call — exactly one of Dense/Sparse set.
type SearchQuery struct {
	Variant string
	Dense   *DenseQuery
	Sparse  *SparseQuery
	Limit   int
}

// Validate reports an error unless exactly one of Dense/Sparse is set.
func (q SearchQuery) Validate() error {
	if q.Dense == nil && q.Sparse == nil {
		return errors.New("search query: exactly one of Dense or Sparse must be set, got neither")
	}
	if q.Dense != nil && q.Sparse != nil {
		return errors.New("search query: exactly one of Dense or Sparse must be set, got both")
	}
	if q.Limit < 1 {
		return fmt.Errorf("search query: limit must be >= 1, got %d", q.Limit)
	}
	return nil
}

type DenseQuery struct {
	Vector []float32
	Ef     int
}

type SparseQuery struct {
	Vector lexical.SparseVector
}

// ChildHit is self-contained: everything downstream stages need comes straight
// from the point payload — no second fetch (§4.2).
type ChildHit struct {
	ChildChunkID  string
	Score         float64 // channel-native
	Text          string
	ParentChunkID string
	ParentText    string
	DocumentID    string
	Title         string
}
