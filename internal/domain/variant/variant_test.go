package variant_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/variant"
)

func valid() *variant.IndexVariant {
	return &variant.IndexVariant{
		Name: "fake_hdr", EmbedderProfile: "fake_1024",
		EmbedderModel: "fake", EmbeddingDim: 1024,
		Chunker: "small_to_big_v1", HeaderStrategy: "title_context",
		Params: variant.Params{
			ParentTokens: 1200, ChildTokens: 250, ChildOverlap: 50,
			LexicalTokenizer: "bigram_v1", NormalizeNumerals: true,
			BM25K1: 1.2, BM25B: 0.75, BM25AvgLen: 250,
		},
	}
}

func TestValidateName(t *testing.T) {
	v := valid()
	require.NoError(t, v.Validate())
	v.Name = "Bad-Name"
	require.Error(t, v.Validate())
}

func TestCollectionName(t *testing.T) {
	require.Equal(t, "chunks_fake_hdr", valid().CollectionName())
}

func TestSameConfigIgnoresCreatedAt(t *testing.T) {
	a, b := valid(), valid()
	b.CreatedAt = time.Now()
	require.True(t, a.SameConfig(b))
}

func TestSameConfigDetectsResolvedModelDrift(t *testing.T) {
	// editing a profile's model behind an unchanged variant must trip (§9.1.1)
	a, b := valid(), valid()
	b.EmbedderModel = "other-model"
	require.False(t, a.SameConfig(b))
}

func TestSameConfigDetectsSparseParamDrift(t *testing.T) {
	a, b := valid(), valid()
	b.Params.NormalizeNumerals = false
	require.False(t, a.SameConfig(b))
}

func TestRegistry(t *testing.T) {
	r := variant.NewRegistry([]*variant.IndexVariant{valid()})
	got, err := r.Get("fake_hdr")
	require.NoError(t, err)
	require.Equal(t, "fake_1024", got.EmbedderProfile)

	_, err = r.Get("nope")
	require.ErrorIs(t, err, variant.ErrUnknown)
	require.Len(t, r.List(), 1)
}
