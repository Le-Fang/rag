//go:build integration

package app_test

import (
	"context"
	"log/slog"
	"testing"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/documents"
	"poc-rag/internal/application/ingestion"
	"poc-rag/internal/application/search"
	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/variant"
	fakeembed "poc-rag/internal/infrastructure/embedder/fake"
	infraqdrant "poc-rag/internal/infrastructure/qdrant"
	fakererank "poc-rag/internal/infrastructure/reranker/fake"
)

type stack struct {
	docs   *documents.Service
	search *search.Service
}

// save persists doc via s.docs and registers a t.Cleanup that removes it
// (idempotent — a later explicit s.docs.Delete in the test body is fine).
// The shared `documents` collection is not dropped by newStack's t.Cleanup
// like the per-variant chunks_* collections are, so every document an e2e
// test saves must be deleted individually or it leaks across test runs.
func (s *stack) save(t *testing.T, ctx context.Context, doc *document.Document, variantName string) (*document.Document, error) {
	t.Helper()
	saved, _, err := s.docs.Save(ctx, doc, variantName)
	t.Cleanup(func() { _ = s.docs.Delete(context.Background(), doc.ID) })
	return saved, err
}

func e2eVariant(name string) *variant.IndexVariant {
	return &variant.IndexVariant{
		Name: name, EmbedderProfile: "fake_8", EmbedderModel: "fake", EmbeddingDim: 8,
		Chunker: "small_to_big_v1", HeaderStrategy: "title_context",
		Params: variant.Params{
			ParentTokens: 200, ChildTokens: 60, ChildOverlap: 10,
			LexicalTokenizer: "bigram_v1", NormalizeNumerals: true,
			BM25K1: 1.2, BM25B: 0.75, BM25AvgLen: 60,
		},
	}
}

// dropVariantArtifacts removes everything Setup.Run provisioned for these
// variants: the chunks_<name> collection (dropped whole) and the registry point
// in `variants` (deleted by name filter, so the point-id derivation stays
// private to the infrastructure package).
func dropVariantArtifacts(ctx context.Context, c *infraqdrant.Client, variants []*variant.IndexVariant) {
	for _, v := range variants {
		_ = c.Qdrant().DeleteCollection(ctx, v.CollectionName())
		// The registry is a SHARED, payload-only collection: unlike chunks_*,
		// this suite must never drop it — only remove the points it wrote itself.
		_, _ = c.Qdrant().Delete(ctx, &qdrant.DeletePoints{
			CollectionName: infraqdrant.VariantsCollection,
			Wait:           qdrant.PtrOf(true),
			Points: qdrant.NewPointsSelectorFilter(&qdrant.Filter{
				Must: []*qdrant.Condition{qdrant.NewMatch("name", v.Name)},
			}),
		})
	}
}

func newStack(t *testing.T, variants ...*variant.IndexVariant) *stack {
	t.Helper()
	ctx := context.Background()
	c, err := infraqdrant.NewClient(infraqdrant.Options{Host: "localhost", GRPCPort: 6334})
	require.NoError(t, err)
	t.Cleanup(func() { c.Close() })
	// Clean at start AND mirror in t.Cleanup: Setup.Run writes a registry point
	// per variant, so without the cleanup half this suite leaves e2e_* entries
	// behind in the shared `variants` collection when run on its own.
	dropVariantArtifacts(ctx, c, variants)
	t.Cleanup(func() { dropVariantArtifacts(ctx, c, variants) })
	require.NoError(t, infraqdrant.NewSetup(c, infraqdrant.NewVariantRepository(c), variants).Run(ctx))

	log := slog.Default()
	reg := variant.NewRegistry(variants)
	embedders := map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)}
	chunkRepo := infraqdrant.NewChunkRepository(c)
	docRepo := infraqdrant.NewDocumentRepository(c)
	ing := ingestion.New(chunk.NewSmallToBig(), embedders, chunkRepo, reg, log)
	searchSvc := search.New(reg,
		[]search.RetrieverEntry{
			{Retriever: search.NewDenseRetriever(embedders, chunkRepo), Modes: []string{"dense", "hybrid"}},
			{Retriever: search.NewSparseRetriever(chunkRepo), Modes: []string{"lexical", "hybrid"}},
		},
		fakererank.New(),
		search.Options{
			DefaultMode: "hybrid", AnnTopK: 50, LexicalTopK: 50, HnswEf: 100, RRFK: 60,
			Weights:    map[string]float64{"dense": 1.0, "lexical": 1.0},
			RerankTopN: 10, RerankMaxCandidates: 20,
		}, log)
	return &stack{
		docs:   documents.New(docRepo, chunkRepo, ing, reg, log),
		search: searchSvc,
	}
}

func textDoc(id, title, content string) *document.Document {
	return &document.Document{ID: id, Title: title, Content: content}
}

func TestE2EHybridSearchAndDelete(t *testing.T) {
	ctx := context.Background()
	v := e2eVariant("e2e_main")
	s := newStack(t, v)

	recipeID := "7c9e6679-7425-40de-944b-e07fc1f90a01"
	_, err := s.save(t, ctx, textDoc(recipeID, "Tomato recipe",
		"這份新的番茄食譜使用一點三湯匙橄欖油。最後加入新鮮羅勒。"), v.Name)
	require.NoError(t, err)
	_, err = s.save(t, ctx, textDoc("7c9e6679-7425-40de-944b-e07fc1f90a02", "Bread notes",
		"The loaf needs steam for a crisp crust and an open crumb."), v.Name)
	require.NoError(t, err)
	measurementID := "7c9e6679-7425-40de-944b-e07fc1f90a04"
	_, err = s.save(t, ctx, textDoc(measurementID, "Sauce ratio",
		"Use 1.3 tablespoons of olive oil in the final sauce."), v.Name)
	require.NoError(t, err)

	// lexical: exact CJK term reaches the right document
	resp, err := s.search.Search(ctx, search.Request{
		Query: "番茄食譜", Variant: v.Name, Mode: "lexical", TopK: 5, Rerank: boolPtr(false),
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Results)
	require.Equal(t, recipeID, resp.Results[0].DocumentID)

	// numeral normalization end to end: digits find the spelled-out form, and
	// the literal decimal survives sentence splitting on the content path — a
	// splitter that cut "1.3" into "1." + "3" would index "1" and "3" and miss.
	resp, err = s.search.Search(ctx, search.Request{
		Query: "1.3", Variant: v.Name, Mode: "lexical", TopK: 5, Rerank: boolPtr(false),
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Results, "「一點三」 must be reachable via '1.3' (§7.1)")
	hitIDs := make([]string, 0, len(resp.Results))
	for _, r := range resp.Results {
		hitIDs = append(hitIDs, r.DocumentID)
	}
	require.Contains(t, hitIDs, recipeID, "「一點三」 must be reachable via '1.3' (§7.1)")
	require.Contains(t, hitIDs, measurementID, "the literal '1.3' must survive chunking and match itself")

	// hybrid with rerank runs both channels and reports diagnostics
	resp, err = s.search.Search(ctx, search.Request{
		Query: "番茄食譜", Variant: v.Name, Mode: "hybrid", TopK: 5,
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Results)
	require.True(t, resp.Stats.Reranked)
	require.Contains(t, resp.Stats.TimingsMs, "total")

	// delete removes the document and its chunks
	require.NoError(t, s.docs.Delete(ctx, recipeID))
	resp, err = s.search.Search(ctx, search.Request{
		Query: "番茄食譜", Variant: v.Name, Mode: "lexical", TopK: 5, Rerank: boolPtr(false),
	})
	require.NoError(t, err)
	for _, r := range resp.Results {
		require.NotEqual(t, recipeID, r.DocumentID, "deleted document must not surface")
	}
}

func TestE2EVariantIsolation(t *testing.T) {
	// §11 isolation: two variants over the same corpus never leak into each other
	ctx := context.Background()
	va, vb := e2eVariant("e2e_iso_a"), e2eVariant("e2e_iso_b")
	s := newStack(t, va, vb)

	docID := "7c9e6679-7425-40de-944b-e07fc1f90a03"
	_, err := s.save(t, ctx, textDoc(docID, "Isolation", "unique isolation payload phrase"), va.Name)
	require.NoError(t, err) // indexed into variant A only

	respA, err := s.search.Search(ctx, search.Request{
		Query: "isolation payload", Variant: va.Name, Mode: "dense", TopK: 5, Rerank: boolPtr(false)})
	require.NoError(t, err)
	require.NotEmpty(t, respA.Results)

	respB, err := s.search.Search(ctx, search.Request{
		Query: "isolation payload", Variant: vb.Name, Mode: "dense", TopK: 5, Rerank: boolPtr(false)})
	require.NoError(t, err)
	require.Empty(t, respB.Results, "variant B was never ingested — separate collections (§5.1)")
}

func boolPtr(b bool) *bool { return &b }
