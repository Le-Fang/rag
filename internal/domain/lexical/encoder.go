package lexical

import (
	"fmt"
	"hash/fnv"
	"sort"
)

// Params are index-time settings and live on the variant (§9): index and query
// paths must encode identically or lexical search silently returns zero hits.
type Params struct {
	Tokenizer         string // must be "bigram_v1"
	NormalizeNumerals bool
	K1, B, AvgLen     float64
}

type SparseVector struct {
	Indices []uint32
	Weights []float32
}

func (v SparseVector) Empty() bool { return len(v.Indices) == 0 }

type Encoder struct {
	p Params
}

func NewEncoder(p Params) (*Encoder, error) {
	if p.Tokenizer != "bigram_v1" {
		return nil, fmt.Errorf("unknown lexical tokenizer %q", p.Tokenizer)
	}
	if !(p.K1 > 0) || !(p.AvgLen > 0) || !(p.B >= 0 && p.B <= 1) {
		return nil, fmt.Errorf("invalid bm25 params: k1=%v b=%v avg_len=%v", p.K1, p.B, p.AvgLen)
	}
	return &Encoder{p: p}, nil
}

// EncodeDocument produces the stored sparse vector: saturated term frequency
// per hashed term (§7.1). Qdrant applies IDF server-side at query time.
func (e *Encoder) EncodeDocument(text string) SparseVector {
	terms := Tokenize(text, e.p.NormalizeNumerals)
	if len(terms) == 0 {
		return SparseVector{}
	}
	counts := make(map[uint32]float64, len(terms))
	for _, t := range terms {
		counts[HashTerm(t)]++ // hash collisions merge counts; accepted risk (§13)
	}
	docLen := float64(len(terms))
	k1, b, avg := e.p.K1, e.p.B, e.p.AvgLen
	indices := sortedIndices(counts)
	weights := make([]float32, len(indices))
	for i, idx := range indices {
		tf := counts[idx]
		weights[i] = float32(tf * (k1 + 1) / (tf + k1*(1-b+b*docLen/avg)))
	}
	return SparseVector{Indices: indices, Weights: weights}
}

// EncodeQuery produces the query vector: weight 1.0 per distinct term (§7.1).
func (e *Encoder) EncodeQuery(text string) SparseVector {
	terms := Tokenize(text, e.p.NormalizeNumerals)
	if len(terms) == 0 {
		return SparseVector{}
	}
	seen := make(map[uint32]float64, len(terms))
	for _, t := range terms {
		seen[HashTerm(t)] = 1.0
	}
	indices := sortedIndices(seen)
	weights := make([]float32, len(indices))
	for i := range weights {
		weights[i] = 1.0
	}
	return SparseVector{Indices: indices, Weights: weights}
}

func HashTerm(term string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(term))
	return h.Sum32()
}

func sortedIndices(m map[uint32]float64) []uint32 {
	out := make([]uint32, 0, len(m))
	for i := range m {
		out = append(out, i)
	}
	sort.Slice(out, func(a, b int) bool { return out[a] < out[b] })
	return out
}
