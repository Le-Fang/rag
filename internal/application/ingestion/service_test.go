package ingestion_test

import (
	"context"
	"log/slog"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/ingestion"
	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/lexical"
	"poc-rag/internal/domain/variant"
	fakeembed "poc-rag/internal/infrastructure/embedder/fake"
)

// memChunkRepo records ReplaceForDocument calls.
type memChunkRepo struct {
	replaced map[string][]*chunk.ParentChunk // key: variant/docID
}

func newMemChunkRepo() *memChunkRepo {
	return &memChunkRepo{replaced: map[string][]*chunk.ParentChunk{}}
}

func (m *memChunkRepo) ReplaceForDocument(_ context.Context, variantName, documentID string, parents []*chunk.ParentChunk) error {
	m.replaced[variantName+"/"+documentID] = parents
	return nil
}
func (m *memChunkRepo) DeleteByDocument(context.Context, string) error { return nil }
func (m *memChunkRepo) DeleteByVariant(context.Context, string) error  { return nil }
func (m *memChunkRepo) SearchChildren(context.Context, chunk.SearchQuery) ([]*chunk.ChildHit, error) {
	return nil, nil
}

func testVariant() *variant.IndexVariant {
	return &variant.IndexVariant{
		Name: "fake_hdr", EmbedderProfile: "fake_8", EmbedderModel: "fake", EmbeddingDim: 8,
		Chunker: "small_to_big_v1", HeaderStrategy: "title_context",
		Params: variant.Params{
			ParentTokens: 40, ChildTokens: 12, ChildOverlap: 4,
			LexicalTokenizer: "bigram_v1", NormalizeNumerals: true,
			BM25K1: 1.2, BM25B: 0.75, BM25AvgLen: 12,
		},
	}
}

func TestIndexFillsVectorsAndPersists(t *testing.T) {
	repo := newMemChunkRepo()
	svc := ingestion.New(
		chunk.NewSmallToBig(),
		map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)},
		repo,
		variant.NewRegistry([]*variant.IndexVariant{testVariant()}),
		slog.Default(),
	)
	doc := &document.Document{
		ID: "doc-1", Title: "zebra",
		Content: "The recipe combines tomatoes and basil. 新的番茄食譜加入新鮮羅勒. Serve with toasted bread.",
	}
	require.NoError(t, svc.Index(context.Background(), doc, "fake_hdr"))

	parents := repo.replaced["fake_hdr/doc-1"]
	require.NotEmpty(t, parents)

	// testVariant uses header_strategy: title_context, so EmbedText carries
	// "[zebra]". The dense vector is built from EmbedText (the header is
	// context for the embedder), but the sparse vector must come from the
	// header-free Text (§7.1) — header terms repeated on every child would be
	// BM25 noise that no query wants. "zebra" (the title) appears only in the
	// header, never in the document body, so its presence would prove the
	// encoder was fed EmbedText.
	headerOnly := lexical.HashTerm("zebra")
	bodyTerm := lexical.HashTerm("recipe")
	foundBody := false
	for _, p := range parents {
		for _, c := range p.Children {
			require.Len(t, c.Dense, 8, "dense vector filled")
			require.False(t, c.Sparse.Empty(), "sparse vector filled from header-free text")
			// Self-guard: the sentinel must actually be live in the header and
			// absent from the body text, or the NotContains below would pass
			// vacuously (e.g. if the fixture's body ever gains the title word).
			require.Contains(t, c.EmbedText, "zebra",
				"sentinel no longer in header — pick a new header-only term")
			require.NotContains(t, c.Text, "zebra",
				"sentinel leaked into body text — pick a new header-only term")
			require.NotContains(t, c.Sparse.Indices, headerOnly,
				"sparse vector must be encoded from Text, not EmbedText — no header terms (§7.1)")
			if slices.Contains(c.Sparse.Indices, bodyTerm) {
				foundBody = true
			}
		}
	}
	// Positive companion: a body-only term must actually be indexed, proving
	// the sparse pipeline ran over the real text (not just "header absent").
	require.True(t, foundBody, `body term "recipe" missing from every child's sparse vector`)
}

// unnormalizedEmbedder returns a fixed vector whose L2 norm is deliberately
// far from 1 (norm of [3,4,0...] is 5). The fake embedder returns unit vectors
// already, so only a stub like this can prove the service normalizes.
type unnormalizedEmbedder struct{ dim int }

func (e unnormalizedEmbedder) Dimensions() int { return e.dim }
func (e unnormalizedEmbedder) Model() string   { return "unnormalized" }
func (e unnormalizedEmbedder) Embed(_ context.Context, texts []string, _ embedding.Kind) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		v := make([]float32, e.dim)
		v[0], v[1] = 3, 4
		out[i] = v
	}
	return out, nil
}

func TestIndexNormalizesVendorVectors(t *testing.T) {
	repo := newMemChunkRepo()
	svc := ingestion.New(
		chunk.NewSmallToBig(),
		map[string]embedding.EmbedderPort{"fake_8": unnormalizedEmbedder{dim: 8}},
		repo,
		variant.NewRegistry([]*variant.IndexVariant{testVariant()}),
		slog.Default(),
	)
	doc := &document.Document{
		ID: "doc-2", Title: "T",
		Content: "The recipe combines tomatoes and basil. Serve with toasted bread.",
	}
	require.NoError(t, svc.Index(context.Background(), doc, "fake_hdr"))

	parents := repo.replaced["fake_hdr/doc-2"]
	require.NotEmpty(t, parents)
	checked := 0
	for _, p := range parents {
		for _, c := range p.Children {
			var norm float64
			for _, x := range c.Dense {
				norm += float64(x) * float64(x)
			}
			// vendor handed us norm 5; cosine collections require unit vectors (§5)
			require.InDelta(t, 1.0, norm, 1e-5, "vendor vector must be L2-normalized before upsert (§5)")
			// direction preserved: 3/5, 4/5
			require.InDelta(t, 0.6, float64(c.Dense[0]), 1e-6)
			require.InDelta(t, 0.8, float64(c.Dense[1]), 1e-6)
			checked++
		}
	}
	require.NotZero(t, checked, "the assertions above must actually have run")
}

func TestIndexEmptyContentIsDeleteOnly(t *testing.T) {
	repo := newMemChunkRepo()
	svc := ingestion.New(
		chunk.NewSmallToBig(),
		map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)},
		repo,
		variant.NewRegistry([]*variant.IndexVariant{testVariant()}),
		slog.Default(),
	)
	doc := &document.Document{ID: "doc-empty", Title: "T"}
	require.NoError(t, svc.Index(context.Background(), doc, "fake_hdr"))

	parents, ok := repo.replaced["fake_hdr/doc-empty"]
	require.True(t, ok, "ReplaceForDocument must still be called with zero parents (delete-only path)")
	require.Empty(t, parents)
}

func TestIndexUnknownVariant(t *testing.T) {
	svc := ingestion.New(chunk.NewSmallToBig(),
		map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)},
		newMemChunkRepo(), variant.NewRegistry(nil), slog.Default())
	err := svc.Index(context.Background(), &document.Document{ID: "d", Content: "x"}, "nope")
	require.ErrorIs(t, err, variant.ErrUnknown)
}
