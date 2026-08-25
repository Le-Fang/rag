package search

import (
	"sort"

	"poc-rag/internal/domain/retrieval"
)

// RRFFuser implements weighted Reciprocal Rank Fusion at child level (§7):
// score(child) = Σ_channels weight_c / (k + rank_c). Only ranks enter the
// formula — channel-native scores are incommensurable and kept as diagnostics.
type RRFFuser struct {
	k       int
	weights map[string]float64
}

func NewRRFFuser(k int, weights map[string]float64) *RRFFuser {
	return &RRFFuser{k: k, weights: weights}
}

var _ retrieval.Fuser = (*RRFFuser)(nil)

func (f *RRFFuser) Fuse(channels []retrieval.ChannelResult) []retrieval.FusedCandidate {
	byID := map[string]*retrieval.FusedCandidate{}
	var order []string // first-seen order for deterministic tie-breaking
	for _, ch := range channels {
		weight, ok := f.weights[ch.Channel]
		if !ok {
			weight = 1.0
		}
		for _, c := range ch.Candidates {
			fc, ok := byID[c.ChildChunkID]
			if !ok {
				fc = &retrieval.FusedCandidate{
					ChildChunkID: c.ChildChunkID,
					Channels:     map[string]retrieval.ChannelScore{},
					Hit:          c.Hit,
				}
				byID[c.ChildChunkID] = fc
				order = append(order, c.ChildChunkID)
			}
			fc.RRFScore += weight / float64(f.k+c.Rank)
			fc.Channels[ch.Channel] = retrieval.ChannelScore{Score: c.Score, Rank: c.Rank}
		}
	}
	out := make([]retrieval.FusedCandidate, 0, len(byID))
	for _, id := range order {
		out = append(out, *byID[id])
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].RRFScore > out[j].RRFScore })
	return out
}
