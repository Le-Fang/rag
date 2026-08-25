package search_test

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/apperror"
	"poc-rag/internal/application/search"
	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/reranking"
	"poc-rag/internal/domain/variant"
	fakeembed "poc-rag/internal/infrastructure/embedder/fake"
	fakererank "poc-rag/internal/infrastructure/reranker/fake"
)

// scriptedChunkRepo returns canned hits (or errors) per channel and records
// calls. Counters are atomic because retrievers run concurrently.
type scriptedChunkRepo struct {
	denseHits, sparseHits   []*chunk.ChildHit
	denseErr, sparseErr     error
	denseBlockUntilCancel   bool          // dense blocks on ctx.Done() and returns ctx.Err()
	denseEntered            chan struct{} // when non-nil, closed once dense is in flight
	sparseBlockThenErr      error         // sparse waits for ctx.Done(), then fails with this
	denseCalls, sparseCalls atomic.Int32
	enteredOnce             sync.Once
}

func (s *scriptedChunkRepo) SearchChildren(ctx context.Context, q chunk.SearchQuery) ([]*chunk.ChildHit, error) {
	if q.Dense != nil {
		s.denseCalls.Add(1)
		if s.denseEntered != nil {
			s.enteredOnce.Do(func() { close(s.denseEntered) })
		}
		if s.denseBlockUntilCancel {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		if s.denseErr != nil {
			return nil, s.denseErr
		}
		return s.denseHits, nil
	}
	s.sparseCalls.Add(1)
	if s.sparseBlockThenErr != nil {
		<-ctx.Done()
		return nil, s.sparseBlockThenErr
	}
	if s.sparseErr != nil {
		return nil, s.sparseErr
	}
	return s.sparseHits, nil
}
func (s *scriptedChunkRepo) ReplaceForDocument(context.Context, string, string, []*chunk.ParentChunk) error {
	return nil
}
func (s *scriptedChunkRepo) DeleteByDocument(context.Context, string) error { return nil }
func (s *scriptedChunkRepo) DeleteByVariant(context.Context, string) error  { return nil }

func hit(childID, parentID, parentText string, score float64) *chunk.ChildHit {
	return &chunk.ChildHit{
		ChildChunkID: childID, ParentChunkID: parentID, ParentText: parentText,
		Text: "child of " + parentID, Score: score, DocumentID: "doc-1", Title: "T",
	}
}

func testVariantReg() *variant.Registry {
	return variant.NewRegistry([]*variant.IndexVariant{{
		Name: "fake_hdr", EmbedderProfile: "fake_8", EmbedderModel: "fake", EmbeddingDim: 8,
		Chunker: "small_to_big_v1", HeaderStrategy: "none",
		Params: variant.Params{
			ParentTokens: 40, ChildTokens: 12, ChildOverlap: 4,
			LexicalTokenizer: "bigram_v1", NormalizeNumerals: true,
			BM25K1: 1.2, BM25B: 0.75, BM25AvgLen: 12,
		},
	}})
}

func newService(repo *scriptedChunkRepo) *search.Service {
	embedders := map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)}
	return search.New(
		testVariantReg(),
		[]search.RetrieverEntry{
			{Retriever: search.NewDenseRetriever(embedders, repo), Modes: []string{"dense", "hybrid"}},
			{Retriever: search.NewSparseRetriever(repo), Modes: []string{"lexical", "hybrid"}},
		},
		fakererank.New(),
		search.Options{
			DefaultMode: "hybrid", AnnTopK: 50, LexicalTopK: 50, HnswEf: 100, RRFK: 60,
			Weights:    map[string]float64{"dense": 1.0, "lexical": 1.0},
			RerankTopN: 10, RerankMaxCandidates: 20,
		},
		slog.Default(),
	)
}

func TestHybridSearchFusesAndExpandsParents(t *testing.T) {
	repo := &scriptedChunkRepo{
		// children c1,c2 share parent p1 — group-by must dedupe to the best
		denseHits:  []*chunk.ChildHit{hit("c1", "p1", "parent one", 0.9), hit("c2", "p1", "parent one", 0.8), hit("c3", "p2", "parent two", 0.7)},
		sparseHits: []*chunk.ChildHit{hit("c3", "p2", "parent two", 12.0)},
	}
	resp, err := newService(repo).Search(context.Background(), search.Request{
		Query: "番茄 recipe", Variant: "fake_hdr", Mode: "hybrid", TopK: 10, Rerank: boolPtr(false),
	})
	require.NoError(t, err)
	require.Equal(t, int32(1), repo.denseCalls.Load())
	require.Equal(t, int32(1), repo.sparseCalls.Load())
	require.Len(t, resp.Results, 2, "two distinct parents")

	// c3 is rank 3 dense + rank 1 lexical → p2 outranks p1 (c1: rank 1 dense only)
	require.Equal(t, "p2", resp.Results[0].ParentChunkID)
	require.Equal(t, "c3", resp.Results[0].ChildChunkID, "winning child reported")
	require.Contains(t, resp.Results[0].Channels, "dense")
	require.Contains(t, resp.Results[0].Channels, "lexical")
	require.NotNil(t, resp.Results[0].LexicalScore)
	require.Equal(t, "parent two", resp.Results[0].Text)
	require.Equal(t, 2, resp.Stats.Parents)
	require.Equal(t, 3, resp.Stats.DenseCandidates)
	require.Equal(t, 1, resp.Stats.LexicalCandidates)
	require.False(t, resp.Stats.Reranked)
}

func TestDenseModeSkipsSparse(t *testing.T) {
	repo := &scriptedChunkRepo{denseHits: []*chunk.ChildHit{hit("c1", "p1", "x", 0.9)}}
	_, err := newService(repo).Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "dense", TopK: 5,
	})
	require.NoError(t, err)
	require.Equal(t, int32(1), repo.denseCalls.Load())
	require.Zero(t, repo.sparseCalls.Load())
}

func TestEmptyTokenizationShortCircuits(t *testing.T) {
	repo := &scriptedChunkRepo{}
	resp, err := newService(repo).Search(context.Background(), search.Request{
		Query: "?!。", Variant: "fake_hdr", Mode: "lexical", TopK: 5, Rerank: boolPtr(false),
	})
	require.NoError(t, err)
	require.Zero(t, repo.sparseCalls.Load(), "no store call for an empty sparse vector (§7.1)")
	require.Empty(t, resp.Results)
}

func TestRerankReordersAndCaps(t *testing.T) {
	// fake reranker reverses order, so the last fused parent comes back first
	repo := &scriptedChunkRepo{denseHits: []*chunk.ChildHit{
		hit("c1", "p1", "parent one", 0.9), hit("c2", "p2", "parent two", 0.8), hit("c3", "p3", "parent three", 0.7),
	}}
	resp, err := newService(repo).Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "dense", TopK: 3, Rerank: boolPtr(true),
	})
	require.NoError(t, err)
	require.True(t, resp.Stats.Reranked)
	require.Equal(t, "p3", resp.Results[0].ParentChunkID, "reranker (order-reversing fake) decides final order")
	require.NotNil(t, resp.Results[0].RerankScore)
	require.Equal(t, 1, resp.Results[0].Rank)
}

func TestRerankCapLimitsCandidates(t *testing.T) {
	// service with max_candidates=2 over 3 parents: the parent beyond the cap
	// is never sent to the reranker and ranks below everything reranked (§7)
	repo := &scriptedChunkRepo{denseHits: []*chunk.ChildHit{
		hit("c1", "p1", "parent one", 0.9), hit("c2", "p2", "parent two", 0.8), hit("c3", "p3", "parent three", 0.7),
	}}
	embedders := map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)}
	svc := search.New(testVariantReg(),
		[]search.RetrieverEntry{{Retriever: search.NewDenseRetriever(embedders, repo), Modes: []string{"dense"}}},
		fakererank.New(),
		search.Options{
			DefaultMode: "dense", AnnTopK: 50, LexicalTopK: 50, HnswEf: 100, RRFK: 60,
			Weights:    map[string]float64{"dense": 1.0},
			RerankTopN: 10, RerankMaxCandidates: 2,
		}, slog.Default())

	resp, err := svc.Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "dense", TopK: 3, Rerank: boolPtr(true),
	})
	require.NoError(t, err)
	require.Len(t, resp.Results, 3)
	// fake reranker reverses the capped pair [p1,p2] → p2, p1; p3 stays last, unscored
	require.Equal(t, "p2", resp.Results[0].ParentChunkID)
	require.Equal(t, "p1", resp.Results[1].ParentChunkID)
	require.Equal(t, "p3", resp.Results[2].ParentChunkID, "beyond-cap parent keeps fused order below reranked ones")
	require.Nil(t, resp.Results[2].RerankScore)
}

// badIndexReranker simulates a misbehaving vendor: it returns an index one
// past the end of the docs slice it was given.
type badIndexReranker struct{}

func (badIndexReranker) Rerank(_ context.Context, _ string, docs []string, _ int) ([]reranking.Scored, error) {
	return []reranking.Scored{{Index: len(docs), Score: 1.0}}, nil
}

func TestRerankRejectsOutOfRangeIndex(t *testing.T) {
	// a malformed vendor response must surface as an error, never a panic (§10)
	repo := &scriptedChunkRepo{denseHits: []*chunk.ChildHit{hit("c1", "p1", "parent one", 0.9)}}
	embedders := map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)}
	svc := search.New(testVariantReg(),
		[]search.RetrieverEntry{{Retriever: search.NewDenseRetriever(embedders, repo), Modes: []string{"dense"}}},
		badIndexReranker{},
		search.Options{
			DefaultMode: "dense", AnnTopK: 50, LexicalTopK: 50, HnswEf: 100, RRFK: 60,
			Weights:    map[string]float64{"dense": 1.0},
			RerankTopN: 10, RerankMaxCandidates: 20,
		}, slog.Default())

	_, err := svc.Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "dense", TopK: 1, Rerank: boolPtr(true),
	})
	require.ErrorContains(t, err, "out of range")
}

// dupIndexReranker simulates a misbehaving RerankerPort that scores the same
// candidate twice. Both HTTP adapters reject this themselves, but the port is
// the contract — a non-adapter implementation must not be able to make a parent
// silently vanish from the ranking.
type dupIndexReranker struct{}

func (dupIndexReranker) Rerank(_ context.Context, _ string, _ []string, _ int) ([]reranking.Scored, error) {
	return []reranking.Scored{{Index: 0, Score: 1.0}, {Index: 0, Score: 0.5}}, nil
}

func TestRerankRejectsDuplicateIndex(t *testing.T) {
	// Two parents, but the reranker names index 0 twice: parent 1 would be left
	// unscored and demoted below the cap while parent 0 keeps the stale score.
	// Mirror the adapters' defense in depth and fail loudly instead (§10).
	repo := &scriptedChunkRepo{denseHits: []*chunk.ChildHit{
		hit("c1", "p1", "parent one", 0.9), hit("c2", "p2", "parent two", 0.8),
	}}
	embedders := map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)}
	svc := search.New(testVariantReg(),
		[]search.RetrieverEntry{{Retriever: search.NewDenseRetriever(embedders, repo), Modes: []string{"dense"}}},
		dupIndexReranker{},
		search.Options{
			DefaultMode: "dense", AnnTopK: 50, LexicalTopK: 50, HnswEf: 100, RRFK: 60,
			Weights:    map[string]float64{"dense": 1.0},
			RerankTopN: 10, RerankMaxCandidates: 20,
		}, slog.Default())

	_, err := svc.Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "dense", TopK: 2, Rerank: boolPtr(true),
	})
	require.ErrorContains(t, err, "duplicate")
}

func TestValidationErrors(t *testing.T) {
	svc := newService(&scriptedChunkRepo{})
	// candidate pools below top_k → 400-class error (§8)
	_, err := svc.Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "dense", TopK: 100,
	})
	var ae *apperror.Error
	require.ErrorAs(t, err, &ae)
	require.Equal(t, 400, ae.Status)
	require.ErrorContains(t, err, "ann_top_k and lexical_top_k must be >= top_k")

	// unknown variant
	_, err = svc.Search(context.Background(), search.Request{Query: "q", Variant: "nope", TopK: 5})
	require.ErrorIs(t, err, variant.ErrUnknown)
}

func TestHybridModeSkipsEmptySparseButRunsDense(t *testing.T) {
	// the empty-sparse-vector short circuit (§7.1) must not also block the
	// healthy dense channel when both are enabled by the mode.
	repo := &scriptedChunkRepo{denseHits: []*chunk.ChildHit{hit("c1", "p1", "parent one", 0.9)}}
	resp, err := newService(repo).Search(context.Background(), search.Request{
		Query: "?!。", Variant: "fake_hdr", Mode: "hybrid", TopK: 5, Rerank: boolPtr(false),
	})
	require.NoError(t, err)
	require.Zero(t, repo.sparseCalls.Load(), "no store call for an empty sparse vector even in hybrid mode (§7.1)")
	require.Equal(t, int32(1), repo.denseCalls.Load())
	require.Len(t, resp.Results, 1)
}

func TestDenseRetrieverErrorFailsTheRequest(t *testing.T) {
	// a failing channel must fail the whole request — never partial results.
	repo := &scriptedChunkRepo{
		denseErr:   errors.New("dense boom"),
		sparseHits: []*chunk.ChildHit{hit("c3", "p2", "parent two", 12.0)},
	}
	resp, err := newService(repo).Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "hybrid", TopK: 5, Rerank: boolPtr(false),
	})
	require.ErrorContains(t, err, "dense boom")
	require.Nil(t, resp)
}

func TestBothRetrieversErrorFirstEnabledOrderWins(t *testing.T) {
	// dense is registered before lexical in newService — its error must win
	// deterministically, and the sparse error must not leak into the message.
	repo := &scriptedChunkRepo{
		denseErr:  errors.New("dense boom"),
		sparseErr: errors.New("sparse boom"),
	}
	_, err := newService(repo).Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "hybrid", TopK: 5, Rerank: boolPtr(false),
	})
	require.ErrorContains(t, err, "dense boom")
	require.NotContains(t, err.Error(), "sparse boom")
}

func TestSelfInflictedCancellationDoesNotMaskRealError(t *testing.T) {
	// sparse fails for a genuine reason; dense is still in flight and only
	// stops because the fan-out cancels it in reaction to sparse's failure.
	// Dense's resulting "context canceled" is a symptom, not the cause, and
	// must not win just because dense is earlier in enabled order.
	repo := &scriptedChunkRepo{
		denseBlockUntilCancel: true,
		sparseErr:             errors.New("sparse boom"),
	}
	_, err := newService(repo).Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "hybrid", TopK: 5, Rerank: boolPtr(false),
	})
	require.ErrorContains(t, err, "sparse boom")
	require.False(t, errors.Is(err, context.Canceled), "the self-inflicted cancellation must not surface as the request's error")
}

// The flip side of TestSelfInflictedCancellationDoesNotMaskRealError: when the
// CALLER hangs up, the resulting context.Canceled is genuine and must be the
// error the request reports. The `caller.Err() == nil` guard is what tells the
// two apart — without it this cancellation would be filtered as self-inflicted
// and the sibling's unrelated failure would be reported in its place (a 500
// where the truth is a 499).
//
// Deterministic by construction: neither channel returns until the caller
// cancels, so no retriever can trip the internal cancel first.
func TestCallerCancellationIsGenuineAndNotMisattributed(t *testing.T) {
	repo := &scriptedChunkRepo{
		denseBlockUntilCancel: true,
		denseEntered:          make(chan struct{}),
		sparseBlockThenErr:    errors.New("sparse boom"),
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-repo.denseEntered // both channels are in flight by now
		cancel()
	}()

	_, err := newService(repo).Search(ctx, search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "hybrid", TopK: 5, Rerank: boolPtr(false),
	})
	require.ErrorIs(t, err, context.Canceled,
		"a cancellation originating with the caller is the real error, not a suppressed symptom")
	require.NotContains(t, err.Error(), "sparse boom",
		"the sibling's failure must not be reported in place of the caller's cancellation")
}

func TestSuppressedSiblingErrorIsLogged(t *testing.T) {
	// the sibling's error isn't just discarded — it's logged so an operator
	// can see both channels failed, not only the one that won the race.
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	embedders := map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)}
	repo := &scriptedChunkRepo{
		denseErr:  errors.New("dense boom"),
		sparseErr: errors.New("sparse boom"),
	}
	svc := search.New(testVariantReg(),
		[]search.RetrieverEntry{
			{Retriever: search.NewDenseRetriever(embedders, repo), Modes: []string{"hybrid"}},
			{Retriever: search.NewSparseRetriever(repo), Modes: []string{"hybrid"}},
		},
		fakererank.New(),
		search.Options{
			DefaultMode: "hybrid", AnnTopK: 50, LexicalTopK: 50, HnswEf: 100, RRFK: 60,
			Weights:    map[string]float64{"dense": 1.0, "lexical": 1.0},
			RerankTopN: 10, RerankMaxCandidates: 20,
		}, logger)

	_, err := svc.Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "hybrid", TopK: 5, Rerank: boolPtr(false),
	})
	require.ErrorContains(t, err, "dense boom")
	require.Contains(t, buf.String(), "sparse boom", "the suppressed sibling error must be logged, not dropped")
}

func TestNoRetrieverEnabledForModeErrors(t *testing.T) {
	// an operator wiring bug (a mode with nothing bound to it) must fail
	// loudly, not silently return zero results with a 200.
	svc := search.New(testVariantReg(), nil, fakererank.New(),
		search.Options{
			DefaultMode: "dense", AnnTopK: 50, LexicalTopK: 50, HnswEf: 100, RRFK: 60,
			Weights:    map[string]float64{"dense": 1.0},
			RerankTopN: 10, RerankMaxCandidates: 20,
		}, slog.Default())

	_, err := svc.Search(context.Background(), search.Request{
		Query: "q", Variant: "fake_hdr", Mode: "dense", TopK: 5,
	})
	require.ErrorContains(t, err, `no retriever enabled for mode "dense"`)
}

func TestStatsRerankedFalseWhenNoParents(t *testing.T) {
	// rerank defaults to true (Rerank left nil), but with zero fused parents
	// there is nothing to rerank — Stats.Reranked must reflect that.
	repo := &scriptedChunkRepo{}
	resp, err := newService(repo).Search(context.Background(), search.Request{
		Query: "?!。", Variant: "fake_hdr", Mode: "lexical", TopK: 5,
	})
	require.NoError(t, err)
	require.False(t, resp.Stats.Reranked, "nothing to rerank when fusion produced zero parents")
}

func boolPtr(b bool) *bool { return &b }
