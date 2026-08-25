package fake_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/infrastructure/embedder/fake"
)

func TestFakeEmbedderDeterministic(t *testing.T) {
	e := fake.New(8)
	a, err := e.Embed(context.Background(), []string{"hello", "world"}, embedding.KindDocument)
	require.NoError(t, err)
	b, err := e.Embed(context.Background(), []string{"hello"}, embedding.KindQuery)
	require.NoError(t, err)
	require.Equal(t, a[0], b[0], "same text → same vector, regardless of kind")
	require.NotEqual(t, a[0], a[1], "different text → different vector")
	require.Len(t, a[0], 8)
	require.Equal(t, 8, e.Dimensions())
}

func TestFakeEmbedderProducesNegativeComponents(t *testing.T) {
	// state>>32 must span the full int32 range (not just the positive half),
	// or every component lands in [0,1) and cosines cluster ~0.75.
	e := fake.New(64)
	v, err := e.Embed(context.Background(), []string{"hello"}, embedding.KindDocument)
	require.NoError(t, err)
	hasNegative := false
	for _, x := range v[0] {
		if x < 0 {
			hasNegative = true
			break
		}
	}
	require.True(t, hasNegative, "expected at least one negative component")
}

func TestFakeEmbedderNewPanicsOnNonPositiveDim(t *testing.T) {
	require.Panics(t, func() { fake.New(0) })
}
