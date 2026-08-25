package chunk

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// Token density differs 4x between Latin and CJK, so a uniform rune stride
// derived from the whole unit's token count blows the budget on the CJK part.
func TestSplitOversizedBoundsEverySliceOnMixedScripts(t *testing.T) {
	const maxTokens = 250
	cjkAlphabet := []rune("我們試做新的番茄食譜香料調整")
	var b strings.Builder
	b.WriteString(strings.Repeat("a", 3000))
	for i := range 1000 {
		b.WriteRune(cjkAlphabet[i%len(cjkAlphabet)])
	}
	text := b.String()

	u := unit{text: text, tokens: EstimateTokens(text)}
	require.Greater(t, u.tokens, maxTokens, "precondition: the unit is oversized")

	slices := splitOversized([]unit{u}, maxTokens)

	var joined strings.Builder
	for i, s := range slices {
		require.LessOrEqual(t, EstimateTokens(s.text), maxTokens,
			"slice %d of %d busts the token budget", i, len(slices))
		require.Equal(t, EstimateTokens(s.text), s.tokens, "slice %d has a stale token count", i)
		joined.WriteString(s.text)
	}
	require.Equal(t, text, joined.String(), "slicing must be lossless")
}

// The decimal guard mirrors the lexical tokenizer's: '.' terminates a sentence
// unless it sits between two digits.
func TestSplitSentencesTreatsDigitFlankedDotAsDecimal(t *testing.T) {
	cases := map[string]struct {
		in   string
		want []string
	}{
		"decimal mid-sentence": {
			in:   "Use 1.3 tablespoons of oil. Add basil after simmering.",
			want: []string{"Use 1.3 tablespoons of oil.", "Add basil after simmering."},
		},
		"digit before a real terminator": {
			in:   "We shipped 5. Then we stopped.",
			want: []string{"We shipped 5.", "Then we stopped."},
		},
		"trailing decimal": {
			in:   "growth of 2.5",
			want: []string{"growth of 2.5"},
		},
		"version-like run": {
			in:   "v1.2.3 shipped.",
			want: []string{"v1.2.3 shipped."},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			require.Equal(t, tc.want, splitSentences(tc.in))
		})
	}
}
