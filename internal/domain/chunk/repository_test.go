package chunk_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/lexical"
)

func TestSearchQueryValidate(t *testing.T) {
	require.Error(t, chunk.SearchQuery{}.Validate(), "neither Dense nor Sparse set")

	both := chunk.SearchQuery{
		Dense:  &chunk.DenseQuery{Vector: []float32{1}},
		Sparse: &chunk.SparseQuery{Vector: lexical.SparseVector{}},
	}
	require.Error(t, both.Validate(), "both Dense and Sparse set")

	denseOnly := chunk.SearchQuery{Dense: &chunk.DenseQuery{Vector: []float32{1}}, Limit: 10}
	require.NoError(t, denseOnly.Validate())

	sparseOnly := chunk.SearchQuery{Sparse: &chunk.SparseQuery{Vector: lexical.SparseVector{}}, Limit: 10}
	require.NoError(t, sparseOnly.Validate())
}

func TestSearchQueryValidateLimit(t *testing.T) {
	base := func(limit int) chunk.SearchQuery {
		return chunk.SearchQuery{Dense: &chunk.DenseQuery{Vector: []float32{1}}, Limit: limit}
	}
	require.Error(t, base(0).Validate(), "zero limit")
	require.Error(t, base(-1).Validate(), "negative limit must not silently wrap via uint64")
	require.NoError(t, base(1).Validate())
}
