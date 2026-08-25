package lexical_test

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/lexical"
)

func params() lexical.Params {
	return lexical.Params{Tokenizer: "bigram_v1", NormalizeNumerals: true, K1: 1.2, B: 0.75, AvgLen: 3}
}

func TestEncodeDocumentSaturatedTF(t *testing.T) {
	enc, err := lexical.NewEncoder(params())
	require.NoError(t, err)

	// "alpha alpha beta": 3 terms, len/avg_len = 1, so denom = tf + k1.
	// alpha: 2*2.2/(2+1.2) = 1.375 ; beta: 1*2.2/(1+1.2) = 1.0
	sv := enc.EncodeDocument("alpha alpha beta")
	require.Len(t, sv.Indices, 2)
	require.Len(t, sv.Weights, 2)
	byIndex := map[uint32]float32{}
	for i, idx := range sv.Indices {
		byIndex[idx] = sv.Weights[i]
	}
	alphaIdx := lexical.HashTerm("alpha")
	betaIdx := lexical.HashTerm("beta")
	require.InDelta(t, 1.375, byIndex[alphaIdx], 1e-5)
	require.InDelta(t, 1.0, byIndex[betaIdx], 1e-5)
}

func TestEncodeQueryDistinctTermsWeightOne(t *testing.T) {
	enc, _ := lexical.NewEncoder(params())
	sv := enc.EncodeQuery("alpha alpha beta")
	require.Len(t, sv.Indices, 2)
	for _, w := range sv.Weights {
		require.Equal(t, float32(1.0), w)
	}
}

func TestRoundTripSymmetry(t *testing.T) {
	// text encoded at index time must be matched by the same text as a query (§7.1)
	enc, _ := lexical.NewEncoder(params())
	doc := enc.EncodeDocument("新的番茄食譜 1.3")
	query := enc.EncodeQuery("新的番茄食譜 1.3")
	require.ElementsMatch(t, doc.Indices, query.Indices)
}

func TestHashStability(t *testing.T) {
	require.Equal(t, lexical.HashTerm("食譜"), lexical.HashTerm("食譜"))
	require.NotEqual(t, lexical.HashTerm("食譜"), lexical.HashTerm("食谱"))
}

func TestEncodeEmptyTokenization(t *testing.T) {
	enc, _ := lexical.NewEncoder(params())
	require.Empty(t, enc.EncodeQuery("!?、。").Indices)
	require.Empty(t, enc.EncodeDocument("").Indices)
}

func TestNewEncoderRejectsUnknownTokenizer(t *testing.T) {
	p := params()
	p.Tokenizer = "jieba_v1"
	_, err := lexical.NewEncoder(p)
	require.Error(t, err)
}

func TestNewEncoderRejectsInvalidParams(t *testing.T) {
	pNaN := params()
	pNaN.K1 = math.NaN()
	_, err := lexical.NewEncoder(pNaN)
	require.Error(t, err)

	pBadB := params()
	pBadB.B = -1
	_, err = lexical.NewEncoder(pBadB)
	require.Error(t, err)
}

func TestEncodeDocumentLengthNormalization(t *testing.T) {
	// §IMPORTANT-3 (re-review): the existing saturated-TF fixture has
	// docLen == AvgLen, so the length-normalization (b) term cancels out
	// (1 - b + b*1 = 1). Use AvgLen=2 so the b term actually contributes.
	// Expected weights are hand-computed literals, not re-derived from the
	// BM25 formula in the test body — an oracle that mirrors the
	// implementation would stay green through a copying error in
	// both. With k1=1.2, b=0.75, AvgLen=2, docLen=3:
	//   norm = k1*(1-b+b*docLen/avg) = 1.2*(0.25+0.75*1.5) = 1.2*1.375 = 1.65
	//   alpha (tf=2) = 2*2.2/(2+1.65) = 4.4/3.65 = 1.2054795
	//   beta  (tf=1) = 1*2.2/(1+1.65) = 2.2/2.65 = 0.8301887
	p := params()
	p.AvgLen = 2
	enc, err := lexical.NewEncoder(p)
	require.NoError(t, err)

	sv := enc.EncodeDocument("alpha alpha beta")
	byIndex := map[uint32]float32{}
	for i, idx := range sv.Indices {
		byIndex[idx] = sv.Weights[i]
	}
	require.InDelta(t, 1.2054795, byIndex[lexical.HashTerm("alpha")], 1e-6)
	require.InDelta(t, 0.8301887, byIndex[lexical.HashTerm("beta")], 1e-6)
}

func TestEncodeDocumentIndicesAscendingUnique(t *testing.T) {
	enc, err := lexical.NewEncoder(params())
	require.NoError(t, err)

	sv := enc.EncodeDocument("alpha beta gamma alpha delta beta epsilon alpha")
	for i := 1; i < len(sv.Indices); i++ {
		require.Less(t, sv.Indices[i-1], sv.Indices[i], "indices must be strictly ascending (sorted + unique)")
	}
}

func TestRoundTripSymmetryWithNumerals(t *testing.T) {
	// §MINOR-8 (re-review): symmetry must hold even when the input carries
	// Chinese numerals that flow through ConvertNumeral, not just plain
	// text. The previous fixture ("一點三億粒芝麻 recipe") had ConvertNumeral
	// fail on "一點三億" (億 lands in the fractional part, which is
	// digit-by-digit only), so the numeral path never actually ran and the
	// test asserted symmetry vacuously. This fixture succeeds on both
	// numeral runs — verified below so the test can't go vacuous again.
	enc, err := lexical.NewEncoder(params())
	require.NoError(t, err)

	text := "二零二四年食譜用了三億五千萬粒芝麻"
	doc := enc.EncodeDocument(text)
	query := enc.EncodeQuery(text)
	require.ElementsMatch(t, doc.Indices, query.Indices)

	require.Contains(t, doc.Indices, lexical.HashTerm("2024"))
	require.Contains(t, doc.Indices, lexical.HashTerm("350000000"))
}
