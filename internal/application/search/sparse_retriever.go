package search

import (
	"context"
	"fmt"
	"time"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/lexical"
	"poc-rag/internal/domain/retrieval"
)

// SparseRetriever encodes the query with the variant's own index-time encoder
// settings — symmetry is structural (§7.1).
type SparseRetriever struct {
	chunks chunk.ChunkRepository
}

func NewSparseRetriever(chunks chunk.ChunkRepository) *SparseRetriever {
	return &SparseRetriever{chunks: chunks}
}

var _ retrieval.Retriever = (*SparseRetriever)(nil)

func (r *SparseRetriever) Channel() string { return "lexical" }

func (r *SparseRetriever) Retrieve(ctx context.Context, q retrieval.Query) ([]retrieval.Candidate, error) {
	enc, err := lexical.NewEncoder(lexical.Params{
		Tokenizer:         q.Variant.Params.LexicalTokenizer,
		NormalizeNumerals: q.Variant.Params.NormalizeNumerals,
		K1:                q.Variant.Params.BM25K1,
		B:                 q.Variant.Params.BM25B,
		AvgLen:            q.Variant.Params.BM25AvgLen,
	})
	if err != nil {
		return nil, fmt.Errorf("variant %s: %w", q.Variant.Name, err)
	}
	sv := enc.EncodeQuery(q.Text)
	if sv.Empty() {
		// a query that tokenizes to nothing short-circuits (§7.1) — no store call
		return nil, nil
	}
	start := time.Now()
	hits, err := r.chunks.SearchChildren(ctx, chunk.SearchQuery{
		Variant: q.Variant.Name,
		Sparse:  &chunk.SparseQuery{Vector: sv},
		Limit:   q.LexicalTopK,
	})
	if err != nil {
		return nil, err
	}
	q.Timings.Add("sparse", time.Since(start))
	return toCandidates(hits), nil
}
