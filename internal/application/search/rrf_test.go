package search_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/search"
	"poc-rag/internal/domain/retrieval"
)

func cand(id string, rank int, score float64) retrieval.Candidate {
	return retrieval.Candidate{ChildChunkID: id, ParentChunkID: "p-" + id, Score: score, Rank: rank}
}

func TestRRFBothChannels(t *testing.T) {
	f := search.NewRRFFuser(60, map[string]float64{"dense": 1.0, "lexical": 1.0})
	fused := f.Fuse([]retrieval.ChannelResult{
		{Channel: "dense", Candidates: []retrieval.Candidate{cand("a", 1, 0.9), cand("b", 2, 0.8)}},
		{Channel: "lexical", Candidates: []retrieval.Candidate{cand("b", 1, 14.0), cand("c", 2, 9.0)}},
	})
	require.Len(t, fused, 3)

	// hand-computed: b = 1/(60+2) + 1/(60+1) = 0.0325230…
	//                a = 1/(60+1) = 0.0163934…, c = 1/(60+2) = 0.0161290…
	require.Equal(t, "b", fused[0].ChildChunkID, "present in both channels wins")
	require.InDelta(t, 1.0/62+1.0/61, fused[0].RRFScore, 1e-9)
	require.Equal(t, "a", fused[1].ChildChunkID)
	require.InDelta(t, 1.0/61, fused[1].RRFScore, 1e-9)
	require.Equal(t, "c", fused[2].ChildChunkID)

	// per-channel diagnostics survive fusion (§8)
	require.Equal(t, 2, fused[0].Channels["dense"].Rank)
	require.Equal(t, 1, fused[0].Channels["lexical"].Rank)
	require.InDelta(t, 14.0, fused[0].Channels["lexical"].Score, 1e-9)
	_, inLexical := fused[1].Channels["lexical"]
	require.False(t, inLexical, "single-channel candidate has no entry for the other channel")
}

func TestRRFWeights(t *testing.T) {
	f := search.NewRRFFuser(60, map[string]float64{"dense": 2.0, "lexical": 1.0})
	fused := f.Fuse([]retrieval.ChannelResult{
		{Channel: "dense", Candidates: []retrieval.Candidate{cand("a", 1, 0.9)}},
		{Channel: "lexical", Candidates: []retrieval.Candidate{cand("b", 1, 9.0)}},
	})
	require.Equal(t, "a", fused[0].ChildChunkID, "weighted channel wins the tie")
	require.InDelta(t, 2.0/61, fused[0].RRFScore, 1e-9)
}

func TestRRFSingleChannelPreservesOrder(t *testing.T) {
	f := search.NewRRFFuser(60, map[string]float64{"dense": 1.0})
	fused := f.Fuse([]retrieval.ChannelResult{
		{Channel: "dense", Candidates: []retrieval.Candidate{cand("a", 1, 0.9), cand("b", 2, 0.8), cand("c", 3, 0.7)}},
	})
	require.Equal(t, []string{"a", "b", "c"},
		[]string{fused[0].ChildChunkID, fused[1].ChildChunkID, fused[2].ChildChunkID})
}
