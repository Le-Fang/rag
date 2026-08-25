package fake

import (
	"context"

	"poc-rag/internal/domain/reranking"
)

// Reranker reverses the input order — a rerank effect tests can assert on.
type Reranker struct{}

func New() *Reranker { return &Reranker{} }

var _ reranking.RerankerPort = (*Reranker)(nil)

func (r *Reranker) Rerank(_ context.Context, _ string, docs []string, topN int) ([]reranking.Scored, error) {
	n := min(topN, len(docs))
	out := make([]reranking.Scored, 0, n)
	for i := 0; i < n; i++ {
		idx := len(docs) - 1 - i
		out = append(out, reranking.Scored{Index: idx, Score: 1 - float64(i)/float64(len(docs))})
	}
	return out, nil
}
