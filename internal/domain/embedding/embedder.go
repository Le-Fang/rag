package embedding

import (
	"context"
	"math"
)

// Kind distinguishes document from query embedding. Not decoration: Cohere
// requires input_type, and asymmetric models degrade without it (§4.2).
type Kind int

const (
	KindDocument Kind = iota
	KindQuery
)

type EmbedderPort interface {
	Embed(ctx context.Context, texts []string, kind Kind) ([][]float32, error)
	Dimensions() int
	Model() string
}

// Normalize L2-normalizes in place and returns v. Zero vectors, and vectors
// whose sum of squares is non-finite (NaN/+Inf), pass through unchanged.
func Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 || math.IsNaN(sum) || math.IsInf(sum, 1) {
		return v
	}
	inv := 1 / math.Sqrt(sum)
	for i := range v {
		v[i] = float32(float64(v[i]) * inv)
	}
	return v
}
