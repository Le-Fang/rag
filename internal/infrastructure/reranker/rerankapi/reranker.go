// Package rerankapi adapts rerank HTTP APIs (§9.1): cohere_style covers
// Cohere, Jina, and Voyage; tei covers text-embeddings-inference.
package rerankapi

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"time"

	"poc-rag/internal/domain/reranking"
	"poc-rag/internal/infrastructure/httpx"
)

type Options struct {
	Style   string // cohere_style | tei
	BaseURL string
	Model   string
	APIKey  string
	Timeout time.Duration
}

type Reranker struct {
	opts   Options
	client *httpx.Client
}

func New(opts Options) *Reranker {
	if opts.Timeout <= 0 {
		opts.Timeout = 30 * time.Second
	}
	return &Reranker{
		opts:   opts,
		client: httpx.New(&http.Client{Timeout: opts.Timeout}, 3, 200*time.Millisecond),
	}
}

var _ reranking.RerankerPort = (*Reranker)(nil)

func (r *Reranker) Rerank(ctx context.Context, query string, docs []string, topN int) ([]reranking.Scored, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	headers := map[string]string{}
	if r.opts.APIKey != "" {
		headers["Authorization"] = "Bearer " + r.opts.APIKey
	}
	url := r.opts.BaseURL + "/rerank"

	switch r.opts.Style {
	case "cohere_style":
		req := map[string]any{"model": r.opts.Model, "query": query, "documents": docs, "top_n": topN}
		var resp struct {
			Results []struct {
				Index          int     `json:"index"`
				RelevanceScore float64 `json:"relevance_score"`
			} `json:"results"`
		}
		if err := r.client.PostJSON(ctx, url, headers, req, &resp); err != nil {
			return nil, fmt.Errorf("rerank: %w", err)
		}
		seen := make([]bool, len(docs))
		out := make([]reranking.Scored, 0, len(resp.Results))
		for _, res := range resp.Results {
			if res.Index < 0 || res.Index >= len(docs) {
				return nil, fmt.Errorf("rerank: response index %d out of range for %d documents", res.Index, len(docs))
			}
			if seen[res.Index] {
				return nil, fmt.Errorf("rerank: duplicate index %d", res.Index)
			}
			seen[res.Index] = true
			out = append(out, reranking.Scored{Index: res.Index, Score: res.RelevanceScore})
		}
		return out, nil

	case "tei":
		req := map[string]any{"query": query, "texts": docs}
		var resp []struct {
			Index int     `json:"index"`
			Score float64 `json:"score"`
		}
		if err := r.client.PostJSON(ctx, url, headers, req, &resp); err != nil {
			return nil, fmt.Errorf("rerank: %w", err)
		}
		seen := make([]bool, len(docs))
		out := make([]reranking.Scored, 0, len(resp))
		for _, res := range resp {
			if res.Index < 0 || res.Index >= len(docs) {
				return nil, fmt.Errorf("rerank: response index %d out of range for %d documents", res.Index, len(docs))
			}
			if seen[res.Index] {
				return nil, fmt.Errorf("rerank: duplicate index %d", res.Index)
			}
			seen[res.Index] = true
			out = append(out, reranking.Scored{Index: res.Index, Score: res.Score})
		}
		sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
		if topN < 0 {
			topN = 0
		}
		if len(out) > topN {
			out = out[:topN]
		}
		return out, nil

	default:
		return nil, fmt.Errorf("rerank: unknown style %q", r.opts.Style)
	}
}
