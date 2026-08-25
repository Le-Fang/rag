package retrieval

import (
	"context"
	"sync"
	"time"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/variant"
)

type Query struct {
	Text    string
	Variant *variant.IndexVariant // resolved once by SearchService from the registry
	// per-channel candidate pools — each retriever reads its own
	AnnTopK     int
	LexicalTopK int
	Ef          int
	// stage timing collector (deviation 2 — feeds stats.timings_ms, §8)
	Timings *Timings
}

type Candidate struct {
	ChildChunkID  string
	ParentChunkID string
	Score         float64 // channel-native: cosine sim or sparse dot product
	Rank          int     // 1-based within this channel — what RRF consumes
	Hit           *chunk.ChildHit
}

type Retriever interface {
	Retrieve(ctx context.Context, q Query) ([]Candidate, error)
	Channel() string // "dense" | "lexical"
}

// ChannelResult pairs a channel name with its ranked candidates (deviation 1).
type ChannelResult struct {
	Channel    string
	Candidates []Candidate
}

type ChannelScore struct {
	Score float64
	Rank  int
}

// FusedCandidate keeps per-channel diagnostics through fusion (§8).
// ParentChunkID is deliberately not carried here: it is never read off this
// type — callers read Hit.ParentChunkID instead — so keeping a second,
// always-in-sync copy would just be dead weight.
type FusedCandidate struct {
	ChildChunkID string
	RRFScore     float64
	Channels     map[string]ChannelScore
	Hit          *chunk.ChildHit
}

type Fuser interface {
	Fuse(channels []ChannelResult) []FusedCandidate
}

// Timings is a concurrency-safe stage-duration collector.
type Timings struct {
	mu sync.Mutex
	ms map[string]int64
}

func NewTimings() *Timings { return &Timings{ms: map[string]int64{}} }

func (t *Timings) Add(stage string, d time.Duration) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ms[stage] += d.Milliseconds()
}

func (t *Timings) Snapshot() map[string]int64 {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make(map[string]int64, len(t.ms))
	for k, v := range t.ms {
		out[k] = v
	}
	return out
}
