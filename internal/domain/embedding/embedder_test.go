package embedding_test

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/embedding"
)

func TestNormalize(t *testing.T) {
	v := embedding.Normalize([]float32{3, 4})
	require.InDelta(t, 0.6, v[0], 1e-6)
	require.InDelta(t, 0.8, v[1], 1e-6)

	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	require.InDelta(t, 1.0, math.Sqrt(norm), 1e-6)

	require.Equal(t, []float32{0, 0}, embedding.Normalize([]float32{0, 0}), "zero vector unchanged")
}

func TestNormalizeNonFiniteSumPassesThrough(t *testing.T) {
	// reflect-based equality can't assert NaN == NaN, so check components
	// individually: the NaN component stays NaN, the rest stay unchanged.
	in := []float32{float32(math.NaN()), 1}
	out := embedding.Normalize(in)
	require.True(t, math.IsNaN(float64(out[0])), "NaN-bearing vector unchanged")
	require.Equal(t, float32(1), out[1])
}
