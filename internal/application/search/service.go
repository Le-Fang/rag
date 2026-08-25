package search

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"time"

	"poc-rag/internal/application/apperror"
	"poc-rag/internal/domain/reranking"
	"poc-rag/internal/domain/retrieval"
	"poc-rag/internal/domain/variant"
)

// RetrieverEntry binds a retriever to the modes that enable it, so mode
// selection is data, not branching (§4.2).
type RetrieverEntry struct {
	Retriever retrieval.Retriever
	Modes     []string
}

type Options struct {
	DefaultMode         string
	AnnTopK             int
	LexicalTopK         int
	HnswEf              int
	RRFK                int
	Weights             map[string]float64
	RerankTopN          int // default top_k when the request omits it
	RerankMaxCandidates int
}

type Request struct {
	Query       string
	Variant     string
	Mode        string
	TopK        int
	AnnTopK     int
	LexicalTopK int
	Rerank      *bool // nil → true (§7: skippable per request)
}

type Result struct {
	Rank          int
	ParentChunkID string
	DocumentID    string
	Text          string // the parent — what the caller reads
	ChildChunkID  string // the winning child
	ChildText     string
	Title         string
	Channels      []string
	DenseScore    *float64
	DenseRank     *int
	LexicalScore  *float64
	LexicalRank   *int
	RRFScore      float64
	RerankScore   *float64
}

type Stats struct {
	Mode              string
	DenseCandidates   int
	LexicalCandidates int
	FusedCandidates   int
	Parents           int
	Reranked          bool
	TimingsMs         map[string]int64
}

type Response struct {
	Results []Result
	Stats   Stats
}

type Service struct {
	registry   *variant.Registry
	retrievers []RetrieverEntry
	reranker   reranking.RerankerPort
	opts       Options
	log        *slog.Logger
}

func New(registry *variant.Registry, retrievers []RetrieverEntry,
	reranker reranking.RerankerPort, opts Options, log *slog.Logger) *Service {
	if log == nil {
		log = slog.Default()
	}
	return &Service{registry: registry, retrievers: retrievers, reranker: reranker, opts: opts, log: log}
}

func (s *Service) Search(ctx context.Context, req Request) (*Response, error) {
	req = s.applyDefaults(req)
	if err := s.validate(req); err != nil {
		return nil, err
	}
	v, err := s.registry.Get(req.Variant) // in-memory set — never the store (§4.2)
	if err != nil {
		return nil, err
	}

	timings := retrieval.NewTimings()
	totalStart := time.Now()
	q := retrieval.Query{
		Text: req.Query, Variant: v,
		AnnTopK: req.AnnTopK, LexicalTopK: req.LexicalTopK, Ef: s.opts.HnswEf,
		Timings: timings,
	}

	// run the enabled retrievers concurrently (§7)
	var enabled []retrieval.Retriever
	for _, e := range s.retrievers {
		if slices.Contains(e.Modes, req.Mode) {
			enabled = append(enabled, e.Retriever)
		}
	}
	if len(enabled) == 0 {
		return nil, fmt.Errorf("no retriever enabled for mode %q", req.Mode)
	}

	// a failing retriever cancels its still-running sibling instead of the
	// caller paying the healthy channel's full latency on the error path
	// (e.g. blocking until an embedder timeout).
	caller := ctx // pre-cancel handle — distinguishes genuine caller cancellation from our own
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	channels := make([]retrieval.ChannelResult, len(enabled))
	errs := make([]error, len(enabled))
	var wg sync.WaitGroup
	for i, r := range enabled {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cands, err := r.Retrieve(ctx, q)
			channels[i] = retrieval.ChannelResult{Channel: r.Channel(), Candidates: cands}
			errs[i] = err
			if err != nil {
				cancel()
			}
		}()
	}
	wg.Wait()

	// deterministic "first error in enabled order wins"; any other error is
	// a suppressed sibling — log it rather than dropping it silently (§8).
	// A cancellation we caused ourselves is a symptom of the sibling's
	// failure, never the cause — it must not outrank the real error just by
	// sitting earlier in enabled order. A cancellation from the CALLER is
	// genuine and falls through as a normal error.
	var firstErr, firstSelfCancel error
	for i, err := range errs {
		if err == nil {
			continue
		}
		if errors.Is(err, context.Canceled) && caller.Err() == nil {
			if firstSelfCancel == nil {
				firstSelfCancel = err
			}
			continue
		}
		if firstErr == nil {
			firstErr = err
			continue
		}
		s.log.WarnContext(ctx, "suppressed sibling retriever error", "channel", enabled[i].Channel(), "error", err)
	}
	if firstErr == nil {
		firstErr = firstSelfCancel
	}
	if firstErr != nil {
		return nil, firstErr
	}

	fuseStart := time.Now()
	fuser := NewRRFFuser(s.opts.RRFK, s.opts.Weights)
	fused := fuser.Fuse(channels)
	parents := groupByParent(fused)
	timings.Add("fuse", time.Since(fuseStart))

	rerank := req.Rerank == nil || *req.Rerank
	reranked := rerank && len(parents) > 0
	if reranked {
		rerankStart := time.Now()
		if err := s.rerankParents(ctx, req.Query, parents); err != nil {
			return nil, err
		}
		timings.Add("rerank", time.Since(rerankStart))
	}

	results := buildResults(parents, req.TopK)
	timings.Add("total", time.Since(totalStart))

	stats := Stats{Mode: req.Mode, FusedCandidates: len(fused), Parents: len(parents),
		Reranked: reranked, TimingsMs: timings.Snapshot()}
	for _, ch := range channels {
		switch ch.Channel {
		case "dense":
			stats.DenseCandidates = len(ch.Candidates)
		case "lexical":
			stats.LexicalCandidates = len(ch.Candidates)
		}
	}
	return &Response{Results: results, Stats: stats}, nil
}

func (s *Service) applyDefaults(req Request) Request {
	if req.Mode == "" {
		req.Mode = s.opts.DefaultMode
	}
	if req.TopK <= 0 {
		req.TopK = s.opts.RerankTopN
	}
	if req.AnnTopK <= 0 {
		req.AnnTopK = s.opts.AnnTopK
	}
	if req.LexicalTopK <= 0 {
		req.LexicalTopK = s.opts.LexicalTopK
	}
	return req
}

func (s *Service) validate(req Request) error {
	if req.Query == "" {
		return apperror.Invalid("query is required")
	}
	if req.Variant == "" {
		return apperror.Invalid("variant is required")
	}
	if req.Mode != "dense" && req.Mode != "lexical" && req.Mode != "hybrid" {
		return apperror.Invalid(fmt.Sprintf("unknown mode %q", req.Mode))
	}
	if req.AnnTopK < req.TopK || req.LexicalTopK < req.TopK {
		return apperror.Invalid("ann_top_k and lexical_top_k must be >= top_k")
	}
	return nil
}

// parentResult is one expanded parent: the winning (best-fused) child speaks
// for it (§7).
type parentResult struct {
	winner      retrieval.FusedCandidate
	rerankScore *float64
}

func groupByParent(fused []retrieval.FusedCandidate) []*parentResult {
	seen := map[string]bool{}
	var out []*parentResult
	for _, fc := range fused { // already sorted by RRF desc — first child wins
		if fc.Hit == nil || seen[fc.Hit.ParentChunkID] {
			continue
		}
		seen[fc.Hit.ParentChunkID] = true
		out = append(out, &parentResult{winner: fc})
	}
	return out
}

// rerankParents scores (query, parent_text) for the top max_candidates
// parents (§7) and stably reorders: reranked parents by score, then the
// rest in fused order.
func (s *Service) rerankParents(ctx context.Context, query string, parents []*parentResult) error {
	capN := min(s.opts.RerankMaxCandidates, len(parents))
	docs := make([]string, capN)
	for i := 0; i < capN; i++ {
		docs[i] = parents[i].winner.Hit.ParentText
	}
	scored, err := s.reranker.Rerank(ctx, query, docs, capN)
	if err != nil {
		return fmt.Errorf("reranking: %w", err)
	}
	// Vendor-returned indices are untrusted. Both HTTP adapters already bounds-
	// and duplicate-check, but RerankerPort is the contract and this is the last
	// place the damage is still visible: a duplicated index would leave some
	// other parent unscored, silently demoting it below the cap while the
	// duplicate overwrites a score that was already assigned.
	seen := make([]bool, capN)
	for _, sc := range scored {
		if sc.Index < 0 || sc.Index >= capN {
			return fmt.Errorf("reranking: response index %d out of range for %d candidates", sc.Index, capN)
		}
		if seen[sc.Index] {
			return fmt.Errorf("reranking: duplicate response index %d for %d candidates", sc.Index, capN)
		}
		seen[sc.Index] = true
		score := sc.Score
		parents[sc.Index].rerankScore = &score
	}
	slices.SortStableFunc(parents, func(a, b *parentResult) int {
		switch {
		case a.rerankScore != nil && b.rerankScore != nil:
			if *a.rerankScore > *b.rerankScore {
				return -1
			} else if *a.rerankScore < *b.rerankScore {
				return 1
			}
			return 0
		case a.rerankScore != nil:
			return -1 // reranked ranks above everything beyond the cap (§7)
		case b.rerankScore != nil:
			return 1
		default:
			return 0
		}
	})
	return nil
}

func buildResults(parents []*parentResult, topK int) []Result {
	n := min(topK, len(parents))
	out := make([]Result, 0, n)
	for i := 0; i < n; i++ {
		p := parents[i]
		h := p.winner.Hit
		res := Result{
			Rank: i + 1, ParentChunkID: h.ParentChunkID, DocumentID: h.DocumentID,
			Text: h.ParentText, ChildChunkID: h.ChildChunkID, ChildText: h.Text,
			Title:    h.Title,
			RRFScore: p.winner.RRFScore, RerankScore: p.rerankScore,
			Channels: []string{}, // serialize as [] rather than null
		}
		for _, ch := range []string{"dense", "lexical"} {
			if cs, ok := p.winner.Channels[ch]; ok {
				res.Channels = append(res.Channels, ch)
				score, rank := cs.Score, cs.Rank
				if ch == "dense" {
					res.DenseScore, res.DenseRank = &score, &rank
				} else {
					res.LexicalScore, res.LexicalRank = &score, &rank
				}
			}
		}
		out = append(out, res)
	}
	return out
}
