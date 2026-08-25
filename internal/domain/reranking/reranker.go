package reranking

import "context"

type Scored struct {
	Index int // index into the input docs slice
	Score float64
}

type RerankerPort interface {
	Rerank(ctx context.Context, query string, docs []string, topN int) ([]Scored, error)
}
