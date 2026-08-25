//go:build integration

package qdrant_test

import (
	"context"
	"testing"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/variant"
	infraqdrant "poc-rag/internal/infrastructure/qdrant"
)

func testVariant(name string) *variant.IndexVariant {
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

func setupClient(t *testing.T) *infraqdrant.Client {
	t.Helper()
	c, err := infraqdrant.NewClient(infraqdrant.Options{Host: "localhost", GRPCPort: 6334})
	require.NoError(t, err)
	t.Cleanup(func() { c.Close() })
	return c
}

func cleanCollections(t *testing.T, c *infraqdrant.Client, names ...string) {
	t.Helper()
	ctx := context.Background()
	for _, n := range names {
		_ = c.Qdrant().DeleteCollection(ctx, n)
	}
}

func TestSetupIsIdempotentAndRegisters(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	v := testVariant("it_setup")
	names := []string{"documents", "variants", v.CollectionName()}
	cleanCollections(t, c, names...)
	t.Cleanup(func() { cleanCollections(t, c, names...) })

	repo := infraqdrant.NewVariantRepository(c)
	s := infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{v})
	require.NoError(t, s.Run(ctx))
	require.NoError(t, s.Run(ctx), "running setup twice is a no-op (§5.2)")

	stored, err := repo.Load(ctx, "it_setup")
	require.NoError(t, err)
	require.True(t, stored.SameConfig(v), "registry payload round-trips the full config")
	require.False(t, stored.CreatedAt.IsZero())
}

// TestSetupRefusesDrift covers the §9.1.1 scenario across every field of
// variant.Params (plus the top-level EmbedderModel), each in its own subtest
// so a regression in one comparison doesn't hide behind another passing.
// The final subtest is the mirror image: an UNCHANGED re-run must NOT be
// reported as drift, guarding against pFloat mis-reading a value that comes
// back from Qdrant as an integer-kind payload (see variant.Params doc comment).
func TestSetupRefusesDrift(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(v *variant.IndexVariant)
	}{
		{"embedder_model", func(v *variant.IndexVariant) { v.EmbedderModel = "other-model" }},
		{"bm25_b", func(v *variant.IndexVariant) { v.Params.BM25B = 0.8 }},
		{"normalize_numerals", func(v *variant.IndexVariant) { v.Params.NormalizeNumerals = false }},
		{"header_strategy", func(v *variant.IndexVariant) { v.HeaderStrategy = "title_context" }},
		{"child_overlap", func(v *variant.IndexVariant) { v.Params.ChildOverlap = 0 }},
		// BM25AvgLen is an integral-valued float (12 here); Qdrant round-trips
		// integral-valued floats as integer-kind payload values (see
		// variant.Params doc comment), so this exercises the positive-drift
		// direction of that decode path, not just the "unmodified" case below.
		{"bm25_avg_len", func(v *variant.IndexVariant) { v.Params.BM25AvgLen = 260 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			c := setupClient(t)
			v := testVariant("it_drift_" + tc.name)
			names := []string{"documents", "variants", v.CollectionName()}
			cleanCollections(t, c, names...)
			t.Cleanup(func() { cleanCollections(t, c, names...) })

			repo := infraqdrant.NewVariantRepository(c)
			require.NoError(t, infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{v}).Run(ctx))

			drifted := testVariant(v.Name)
			tc.mutate(drifted)
			err := infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{drifted}).Run(ctx)
			require.ErrorIs(t, err, variant.ErrConfigDrift)
		})
	}

	t.Run("unmodified rerun is not drift", func(t *testing.T) {
		ctx := context.Background()
		c := setupClient(t)
		v := testVariant("it_drift_unmodified")
		names := []string{"documents", "variants", v.CollectionName()}
		cleanCollections(t, c, names...)
		t.Cleanup(func() { cleanCollections(t, c, names...) })

		repo := infraqdrant.NewVariantRepository(c)
		require.NoError(t, infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{v}).Run(ctx))

		same := testVariant(v.Name)
		require.NoError(t, infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{same}).Run(ctx),
			"a re-run with an unchanged config must not be reported as drift")
	})
}

// TestSetupRefusesVectorShapeDrift reproduces the scenario the registry drift
// check alone cannot catch: the variants collection (the registry) is reset
// while the chunks_* collection — expensive to rebuild, so plausibly left in
// place on purpose — is not. Setup must still refuse to silently adopt the
// stale collection under a wider declared embedding dimension.
func TestSetupRefusesVectorShapeDrift(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	v := testVariant("it_shape_drift")
	names := []string{"documents", "variants", v.CollectionName()}
	cleanCollections(t, c, names...)
	t.Cleanup(func() { cleanCollections(t, c, names...) })

	repo := infraqdrant.NewVariantRepository(c)
	require.NoError(t, infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{v}).Run(ctx))

	// Reset only the registry — chunks_it_shape_drift stays behind at dim 8.
	require.NoError(t, c.Qdrant().DeleteCollection(ctx, "variants"))

	widened := testVariant(v.Name)
	widened.EmbeddingDim = 16
	err := infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{widened}).Run(ctx)
	require.ErrorIs(t, err, variant.ErrConfigDrift,
		"a pre-existing chunks_* collection at the wrong dimension must not pass setup silently")
}

// TestSetupDriftRefusalCreatesNoCollection pins the recovery-friendly ordering
// of Setup.Run: the registry drift check must run BEFORE the chunk collection
// is created. Otherwise this wedge exists: registry entry at dim 8, chunks
// collection absent (manual drop / DeleteByVariant), config edited to dim 16 —
// a create-first Setup manufactures an empty dim-16 collection and THEN
// refuses on registry drift, so reverting the config now fails the physical
// shape check against the collection the failed startup itself created.
func TestSetupDriftRefusalCreatesNoCollection(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	v := testVariant("it_drift_no_create")
	names := []string{"documents", "variants", v.CollectionName()}
	cleanCollections(t, c, names...)
	t.Cleanup(func() { cleanCollections(t, c, names...) })

	repo := infraqdrant.NewVariantRepository(c)
	require.NoError(t, infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{v}).Run(ctx))

	// Leave the registry entry (dim 8) behind but drop the chunks collection.
	require.NoError(t, c.Qdrant().DeleteCollection(ctx, v.CollectionName()))

	widened := testVariant(v.Name)
	widened.EmbeddingDim = 16
	err := infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{widened}).Run(ctx)
	require.ErrorIs(t, err, variant.ErrConfigDrift)

	exists, err := c.Qdrant().CollectionExists(ctx, v.CollectionName())
	require.NoError(t, err)
	require.False(t, exists,
		"a drift-refused startup must not have created the chunk collection")
}

// TestSetupRefusesMissingIDFModifier hand-creates the chunks_* collection
// with a "lexical" sparse vector that has no IDF modifier — the shape that
// would otherwise pass the "key exists" check while silently degrading
// lexical scoring to raw term frequency (§14.6).
func TestSetupRefusesMissingIDFModifier(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	v := testVariant("it_idf_drift")
	names := []string{"documents", "variants", v.CollectionName()}
	cleanCollections(t, c, names...)
	t.Cleanup(func() { cleanCollections(t, c, names...) })

	require.NoError(t, c.Qdrant().CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: v.CollectionName(),
		VectorsConfig: qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
			"dense": {Size: uint64(v.EmbeddingDim), Distance: qdrant.Distance_Cosine},
		}),
		SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
			"lexical": {}, // no Modifier set — Qdrant defaults to Modifier_None
		}),
	}))

	repo := infraqdrant.NewVariantRepository(c)
	err := infraqdrant.NewSetup(c, repo, []*variant.IndexVariant{v}).Run(ctx)
	require.ErrorIs(t, err, variant.ErrConfigDrift,
		"a lexical sparse vector without the IDF modifier must not pass setup silently")
}

func TestVariantRepositoryUnknown(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	repo := infraqdrant.NewVariantRepository(c)
	_ = infraqdrant.NewSetup(c, repo, nil).Run(ctx) // ensures base collections
	_, err := repo.Load(ctx, "never_registered")
	require.ErrorIs(t, err, variant.ErrUnknown)
}
