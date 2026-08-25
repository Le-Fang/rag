//go:build integration

package app

import (
	"bytes"
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/variant"
	infraqdrant "poc-rag/internal/infrastructure/qdrant"
)

func dropTestVariant(name string) *variant.IndexVariant {
	return &variant.IndexVariant{
		Name: name, EmbedderProfile: "fake_8", EmbedderModel: "fake", EmbeddingDim: 8,
		Chunker: "small_to_big_v1", HeaderStrategy: "none",
		Params: variant.Params{
			ParentTokens: 40, ChildTokens: 12, ChildOverlap: 4,
			LexicalTokenizer: "bigram_v1", NormalizeNumerals: true,
			BM25K1: 1.2, BM25B: 0.75, BM25AvgLen: 12,
		},
	}
}

// TestVariantDropRemovesBothArtifacts covers the drift-recovery path end to
// end via the extracted run function (no shelling out): dropping a variant
// removes its chunks_* collection AND its registry point, leaves a sibling
// variant untouched, and a second drop of the same name is a quiet no-op.
// runVariantDrop deliberately never runs Setup, so this recovery works even
// when startup would refuse with ErrConfigDrift.
func TestVariantDropRemovesBothArtifacts(t *testing.T) {
	ctx := context.Background()
	c, err := infraqdrant.NewClient(infraqdrant.Options{Host: "localhost", GRPCPort: 6334})
	require.NoError(t, err)
	t.Cleanup(func() { c.Close() })

	va, vb := dropTestVariant("it_vdrop_gone"), dropTestVariant("it_vdrop_kept")
	repo := infraqdrant.NewVariantRepository(c)
	// The registry is a SHARED collection: clean only this test's artifacts —
	// own chunk collections dropped whole, own registry points by derived id —
	// at start AND mirrored in t.Cleanup (integration-test convention).
	clean := func() {
		for _, v := range []*variant.IndexVariant{va, vb} {
			_ = c.Qdrant().DeleteCollection(ctx, v.CollectionName())
			_ = repo.Delete(ctx, v.Name)
		}
	}
	clean()
	t.Cleanup(clean)

	require.NoError(t, infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{va, vb}).Run(ctx))

	var out bytes.Buffer
	require.NoError(t, runVariantDrop(ctx, c, va.Name, &out))
	require.Contains(t, out.String(), va.Name)

	exists, err := c.Qdrant().CollectionExists(ctx, va.CollectionName())
	require.NoError(t, err)
	require.False(t, exists, "dropped variant's chunk collection must be gone")
	_, err = repo.Load(ctx, va.Name)
	require.ErrorIs(t, err, variant.ErrUnknown, "dropped variant's registry point must be gone")

	exists, err = c.Qdrant().CollectionExists(ctx, vb.CollectionName())
	require.NoError(t, err)
	require.True(t, exists, "the other variant's chunk collection must be untouched")
	_, err = repo.Load(ctx, vb.Name)
	require.NoError(t, err, "the other variant's registry point must be untouched")

	require.NoError(t, runVariantDrop(ctx, c, va.Name, &out),
		"dropping an already-dropped variant must succeed quietly")
}
