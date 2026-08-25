//go:build integration

package qdrant_test

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/lexical"
	"poc-rag/internal/domain/variant"
	infraqdrant "poc-rag/internal/infrastructure/qdrant"
)

// buildParent makes one parent with two children carrying real vectors.
func buildParent(docID string, denseSeed float32) *chunk.ParentChunk {
	enc, _ := lexical.NewEncoder(lexical.Params{Tokenizer: "bigram_v1", NormalizeNumerals: true, K1: 1.2, B: 0.75, AvgLen: 12})
	mkChild := func(ordinal int, text string) *chunk.ChildChunk {
		dense := make([]float32, 8)
		dense[ordinal%8] = denseSeed
		return &chunk.ChildChunk{
			ID: uuid.NewString(), Ordinal: ordinal, Text: text, EmbedText: text,
			TokenCount: chunk.EstimateTokens(text),
			Dense:      dense, Sparse: enc.EncodeDocument(text),
		}
	}
	return &chunk.ParentChunk{
		ID: uuid.NewString(), Ordinal: 0, Text: "新的番茄食譜 recipe instructions",
		DocumentID: docID, Title: "Recipe",
		Children: []*chunk.ChildChunk{
			mkChild(0, "新的番茄食譜"),
			mkChild(1, "recipe instructions"),
		},
	}
}

func chunkTestEnv(t *testing.T, variantName string) (*infraqdrant.ChunkRepository, *variant.IndexVariant) {
	t.Helper()
	ctx := context.Background()
	c := setupClient(t)
	v := testVariant(variantName)
	names := []string{"documents", "variants", v.CollectionName()}
	cleanCollections(t, c, names...)
	t.Cleanup(func() { cleanCollections(t, c, names...) })
	require.NoError(t, infraqdrant.NewSetup(c, infraqdrant.NewVariantRepository(c), []*variant.IndexVariant{v}).Run(ctx))
	return infraqdrant.NewChunkRepository(c), v
}

func TestReplaceAndSearchDense(t *testing.T) {
	ctx := context.Background()
	repo, v := chunkTestEnv(t, "it_chunks")
	docID := uuid.NewString()
	parent := buildParent(docID, 1.0)

	require.NoError(t, repo.ReplaceForDocument(ctx, v.Name, docID, []*chunk.ParentChunk{parent}))
	// idempotent: replacing again does not duplicate
	require.NoError(t, repo.ReplaceForDocument(ctx, v.Name, docID, []*chunk.ParentChunk{parent}))

	queryVec := make([]float32, 8)
	queryVec[0] = 1.0
	hits, err := repo.SearchChildren(ctx, chunk.SearchQuery{
		Variant: v.Name,
		Dense:   &chunk.DenseQuery{Vector: queryVec, Ef: 100},
		Limit:   10,
	})
	require.NoError(t, err)
	require.Len(t, hits, 2, "replace must not duplicate points")

	top := hits[0]
	require.Greater(t, top.Score, 0.0, "dense hit must carry a real channel-native score")
	require.Equal(t, parent.Children[0].ID, top.ChildChunkID)
	require.Equal(t, "新的番茄食譜", top.Text)
	require.Equal(t, parent.ID, top.ParentChunkID)
	require.Equal(t, parent.Text, top.ParentText, "parent denormalized into the child payload (§5.1)")
	require.Equal(t, docID, top.DocumentID)
	require.Equal(t, "Recipe", top.Title)
}

func TestSearchSparseLexicalE2E(t *testing.T) {
	// index a Chinese chunk, query a substring through real Qdrant, get a hit.
	// The IDF-ranking claim itself is covered by claims_integration_test.go's
	// TestSparseIDFModifier, not here.
	ctx := context.Background()
	repo, v := chunkTestEnv(t, "it_lex")
	enc, _ := lexical.NewEncoder(lexical.Params{Tokenizer: "bigram_v1", NormalizeNumerals: true, K1: 1.2, B: 0.75, AvgLen: 12})

	docID := uuid.NewString()
	parent := buildParent(docID, 1.0)
	require.NoError(t, repo.ReplaceForDocument(ctx, v.Name, docID, []*chunk.ParentChunk{parent}))

	hits, err := repo.SearchChildren(ctx, chunk.SearchQuery{
		Variant: v.Name,
		Sparse:  &chunk.SparseQuery{Vector: enc.EncodeQuery("番茄")},
		Limit:   10,
	})
	require.NoError(t, err)
	require.NotEmpty(t, hits, "a substring of an indexed Chinese chunk must hit")
	require.Greater(t, hits[0].Score, 0.0, "sparse hit must carry a real channel-native score")
	require.Equal(t, "新的番茄食譜", hits[0].Text)
}

func TestReplaceAndSearchMinimalParent(t *testing.T) {
	// Round-trip through real Qdrant: a parent with an empty Title must not
	// surface stale or synthesized values in the returned ChildHit.
	ctx := context.Background()
	repo, v := chunkTestEnv(t, "it_min_fields")
	enc, _ := lexical.NewEncoder(lexical.Params{Tokenizer: "bigram_v1", NormalizeNumerals: true, K1: 1.2, B: 0.75, AvgLen: 12})

	docID := uuid.NewString()
	text := "minimal chunk"
	dense := make([]float32, 8)
	dense[0] = 1.0
	parent := &chunk.ParentChunk{
		ID: uuid.NewString(), Ordinal: 0, Text: "minimal parent text",
		DocumentID: docID, // Title deliberately left empty
		Children: []*chunk.ChildChunk{{
			ID: uuid.NewString(), Ordinal: 0, Text: text, EmbedText: text,
			TokenCount: chunk.EstimateTokens(text), Dense: dense, Sparse: enc.EncodeDocument(text),
		}},
	}
	require.NoError(t, repo.ReplaceForDocument(ctx, v.Name, docID, []*chunk.ParentChunk{parent}))

	queryVec := make([]float32, 8)
	queryVec[0] = 1.0
	hits, err := repo.SearchChildren(ctx, chunk.SearchQuery{
		Variant: v.Name, Dense: &chunk.DenseQuery{Vector: queryVec, Ef: 100}, Limit: 10,
	})
	require.NoError(t, err)
	require.Len(t, hits, 1)
	require.Empty(t, hits[0].Title)
	require.Equal(t, "minimal parent text", hits[0].ParentText)
}

func TestReplaceForDocumentEmptyParentsIsDeleteOnly(t *testing.T) {
	ctx := context.Background()
	repo, v := chunkTestEnv(t, "it_empty_replace")
	docID := uuid.NewString()
	require.NoError(t, repo.ReplaceForDocument(ctx, v.Name, docID, []*chunk.ParentChunk{buildParent(docID, 1.0)}))

	require.NoError(t, repo.ReplaceForDocument(ctx, v.Name, docID, nil), "empty parents = delete-only, no error")

	queryVec := make([]float32, 8)
	queryVec[0] = 1.0
	hits, err := repo.SearchChildren(ctx, chunk.SearchQuery{
		Variant: v.Name, Dense: &chunk.DenseQuery{Vector: queryVec, Ef: 100}, Limit: 10,
	})
	require.NoError(t, err)
	require.Empty(t, hits)
}

func TestReplaceForDocumentRejectsMismatchedDocumentID(t *testing.T) {
	ctx := context.Background()
	repo, v := chunkTestEnv(t, "it_mismatch")
	docID := uuid.NewString()
	parent := buildParent(uuid.NewString() /* different from docID below */, 1.0)

	err := repo.ReplaceForDocument(ctx, v.Name, docID, []*chunk.ParentChunk{parent})
	require.Error(t, err, "a parent whose DocumentID disagrees with the documentID argument must be rejected")
}

func TestDeleteByDocumentAcrossVariants(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	v1, v2 := testVariant("it_del_a"), testVariant("it_del_b")
	names := []string{"documents", "variants", v1.CollectionName(), v2.CollectionName()}
	cleanCollections(t, c, names...)
	t.Cleanup(func() { cleanCollections(t, c, names...) })
	require.NoError(t, infraqdrant.NewSetup(c, infraqdrant.NewVariantRepository(c), []*variant.IndexVariant{v1, v2}).Run(ctx))
	repo := infraqdrant.NewChunkRepository(c)

	docID := uuid.NewString()
	require.NoError(t, repo.ReplaceForDocument(ctx, v1.Name, docID, []*chunk.ParentChunk{buildParent(docID, 1.0)}))
	require.NoError(t, repo.ReplaceForDocument(ctx, v2.Name, docID, []*chunk.ParentChunk{buildParent(docID, 1.0)}))

	// keptDocID must survive the purge of docID — DeleteByDocument must only
	// remove the targeted document's points, in every variant.
	keptDocID := uuid.NewString()
	require.NoError(t, repo.ReplaceForDocument(ctx, v1.Name, keptDocID, []*chunk.ParentChunk{buildParent(keptDocID, 1.0)}))

	require.NoError(t, repo.DeleteByDocument(ctx, docID))

	queryVec := make([]float32, 8)
	queryVec[0] = 1.0
	for _, name := range []string{v1.Name, v2.Name} {
		hits, err := repo.SearchChildren(ctx, chunk.SearchQuery{
			Variant: name, Dense: &chunk.DenseQuery{Vector: queryVec, Ef: 100}, Limit: 10,
		})
		require.NoError(t, err)
		for _, h := range hits {
			require.NotEqual(t, docID, h.DocumentID, "delete must purge every variant (§6.1)")
		}
	}

	survivorHits, err := repo.SearchChildren(ctx, chunk.SearchQuery{
		Variant: v1.Name, Dense: &chunk.DenseQuery{Vector: queryVec, Ef: 100}, Limit: 10,
	})
	require.NoError(t, err)
	require.NotEmpty(t, survivorHits, "an unrelated document must survive the targeted purge")
	for _, h := range survivorHits {
		require.Equal(t, keptDocID, h.DocumentID)
	}
}

func TestDeleteByVariantDropsCollection(t *testing.T) {
	ctx := context.Background()
	repo, v := chunkTestEnv(t, "it_drop")
	require.NoError(t, repo.DeleteByVariant(ctx, v.Name))
	c := setupClient(t)
	exists, err := c.Qdrant().CollectionExists(ctx, v.CollectionName())
	require.NoError(t, err)
	require.False(t, exists)
}

func TestDeleteByVariantIsIdempotent(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	v := testVariant("it_del_variant_idem")
	cleanCollections(t, c, v.CollectionName())
	t.Cleanup(func() { cleanCollections(t, c, v.CollectionName()) })
	repo := infraqdrant.NewChunkRepository(c)

	// Never created in the first place.
	require.NoError(t, repo.DeleteByVariant(ctx, v.Name))
	// Calling again — still absent — must also be a no-op, not an error.
	require.NoError(t, repo.DeleteByVariant(ctx, v.Name))
}
