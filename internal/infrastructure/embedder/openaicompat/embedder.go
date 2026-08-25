// Package openaicompat covers OpenAI, Azure, Jina, Voyage, Together, and
// self-hosted OpenAI-compatible routes: POST {base_url}/embeddings (§9.1).
package openaicompat

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/infrastructure/httpx"
)

type Options struct {
	BaseURL    string
	Model      string
	APIKey     string
	Dimensions int
	BatchSize  int
	Timeout    time.Duration
}

type Embedder struct {
	opts   Options
	client *httpx.Client
}

func New(opts Options) *Embedder {
	if opts.BatchSize <= 0 {
		opts.BatchSize = 96
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 30 * time.Second
	}
	return &Embedder{
		opts:   opts,
		client: httpx.New(&http.Client{Timeout: opts.Timeout}, 3, 200*time.Millisecond),
	}
}

var _ embedding.EmbedderPort = (*Embedder)(nil)

func (e *Embedder) Dimensions() int { return e.opts.Dimensions }
func (e *Embedder) Model() string   { return e.opts.Model }

// Embed ignores kind: OpenAI-compatible embedding APIs are symmetric.
func (e *Embedder) Embed(ctx context.Context, texts []string, _ embedding.Kind) ([][]float32, error) {
	// An empty api_key means the route needs no auth (self-hosted vLLM/TEI, or a
	// proxy that injects credentials). Omit the header rather than sending a
	// bare "Bearer " — some gateways reject the malformed value. Mirrors
	// rerankapi.Reranker.
	headers := map[string]string{}
	if e.opts.APIKey != "" {
		headers["Authorization"] = "Bearer " + e.opts.APIKey
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.opts.BatchSize {
		batch := texts[start:min(start+e.opts.BatchSize, len(texts))]
		req := map[string]any{
			"model":      e.opts.Model,
			"input":      batch,
			"dimensions": e.opts.Dimensions,
		}
		var resp struct {
			Data []struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		}
		if err := e.client.PostJSON(ctx, e.opts.BaseURL+"/embeddings", headers, req, &resp); err != nil {
			return nil, fmt.Errorf("openaicompat embed: %w", err)
		}
		if len(resp.Data) != len(batch) {
			return nil, fmt.Errorf("openaicompat embed: got %d embeddings for %d inputs", len(resp.Data), len(batch))
		}
		// Don't trust response array order: self-hosted continuous-batching
		// servers can complete requests out of order. Place each vector by
		// its reported index instead of its position in the array.
		batchOut := make([][]float32, len(batch))
		seen := make([]bool, len(batch))
		for _, d := range resp.Data {
			if d.Index < 0 || d.Index >= len(batch) {
				return nil, fmt.Errorf("openaicompat embed: index %d out of range for %d inputs", d.Index, len(batch))
			}
			if seen[d.Index] {
				return nil, fmt.Errorf("openaicompat embed: duplicate index %d in response", d.Index)
			}
			seen[d.Index] = true
			if len(d.Embedding) != e.opts.Dimensions {
				return nil, fmt.Errorf("openaicompat embed: model %s returned width %d, profile expects %d: %w",
					e.opts.Model, len(d.Embedding), e.opts.Dimensions, chunk.ErrDimensionMismatch)
			}
			batchOut[d.Index] = d.Embedding
		}
		out = append(out, batchOut...)
	}
	return out, nil
}
