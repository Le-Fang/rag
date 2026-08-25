package fake

import (
	"context"
	"hash/fnv"

	"poc-rag/internal/domain/embedding"
)

// Embedder derives deterministic vectors from a text hash — pipeline tests
// run offline and repeatably (§11).
type Embedder struct {
	dim int
}

func New(dim int) *Embedder {
	if dim <= 0 {
		panic("fake embedder: dimensions must be positive")
	}
	return &Embedder{dim: dim}
}

var _ embedding.EmbedderPort = (*Embedder)(nil)

func (e *Embedder) Dimensions() int { return e.dim }
func (e *Embedder) Model() string   { return "fake" }

func (e *Embedder) Embed(_ context.Context, texts []string, _ embedding.Kind) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		h := fnv.New64a()
		h.Write([]byte(t))
		state := h.Sum64()
		v := make([]float32, e.dim)
		for j := range v {
			state = state*6364136223846793005 + 1442695040888963407
			v[j] = float32(int32(state>>32)) / float32(1<<31)
		}
		out[i] = embedding.Normalize(v)
	}
	return out, nil
}
