package search_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/search"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/retrieval"
)

// emptyEmbedder simulates a misbehaving vendor adapter: success with zero
// vectors for a one-text batch. Vendor-returned shapes are untrusted (§10) —
// ingestion already checks len(vecs) against its batch, and the reranker path
// bounds-checks indices; the query path must not be the one consumer that
// trusts vecs[0] blindly.
type emptyEmbedder struct{}

func (emptyEmbedder) Dimensions() int { return 8 }
func (emptyEmbedder) Model() string   { return "empty" }
func (emptyEmbedder) Embed(context.Context, []string, embedding.Kind) ([][]float32, error) {
	return [][]float32{}, nil
}

func TestDenseRetrieverRejectsEmptyEmbedResponse(t *testing.T) {
	r := search.NewDenseRetriever(
		map[string]embedding.EmbedderPort{"fake_8": emptyEmbedder{}}, &scriptedChunkRepo{})
	v, err := testVariantReg().Get("fake_hdr")
	require.NoError(t, err)

	_, err = r.Retrieve(context.Background(), retrieval.Query{
		Text: "q", Variant: v, AnnTopK: 5, Timings: retrieval.NewTimings(),
	})
	require.Error(t, err, "an empty embed response must surface as an error, never a panic")
	require.ErrorContains(t, err, "expected 1 vector, got 0")
}
