package lexical_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/lexical"
)

func TestConvertNumeral(t *testing.T) {
	cases := []struct {
		in   string
		want string
		ok   bool
	}{
		{"一點三", "1.3", true},
		{"十三", "13", true},
		{"三十五", "35", true},
		{"一百二十", "120", true},
		{"兩千零一", "2001", true},
		{"五萬", "50000", true},
		{"点五", "0.5", true}, // simplified point, no integer part
		{"我們", "", false},   // not a numeral run
		{"", "", false},

		// nested big units (§IMPORTANT-1): scale only the section since the
		// previous big-unit boundary, not the running total.
		{"三億五千萬", "350000000", true},
		{"一億兩千萬", "120000000", true},
		{"一億零五萬", "100050000", true},
		{"三萬五", "30005", true}, // accepted literal reading, unchanged
		{"十萬", "100000", true},
		{"一百萬", "1000000", true},
		{"二十五萬", "250000", true},
		{"三萬五千", "35000", true},
		{"一萬兩千三百四十五", "12345", true},
		{"兩億", "200000000", true},

		// digit-sequence reading (§IMPORTANT-2): no unit runes means read
		// digit-by-digit, e.g. a spoken year.
		{"二零二四", "2024", true},
		{"一九九八", "1998", true},

		// malformed unit sequences (§MINOR-5): units must strictly decrease
		// within a section.
		{"十十", "", false},
		{"二十十", "", false},

		// formal long-form numeral (23 runes): with section-scaling the
		// multiplicative chain is gone, so the overflow guard only needs to
		// reject absurd lengths, not this legitimate one.
		{"九千九百九十九億九千九百九十九萬九千九百九十九", "999999999999", true},
	}
	for _, c := range cases {
		got, ok := lexical.ConvertNumeral(c.in)
		require.Equal(t, c.ok, ok, "input: %q", c.in)
		if c.ok {
			require.Equal(t, c.want, got, "input: %q", c.in)
		}
	}
}

// TestTokenizeBaiFenDianArtifactIsPinned pins the second documented numeral
// artifact (the first, 三萬五 → "30005", is pinned in TestConvertNumeral above).
//
// In 百分點 ("percentage point"), 百 is a unit rune, not a magnitude: the
// tokenizer's numeral run breaks at 分 and hands ConvertNumeral a lone 百,
// which reads positionally as 100. So "一點三個百分點" emits a spurious "100"
// term beside the real "1.3".
//
// This is DELIBERATE and documented in numerals.go — do not "fix" it. The
// symmetry invariant (§7.1) is what makes it harmless: queries run through the
// same Tokenize, so both sides emit the same spurious term and still match.
// Suppressing it on one side only would silently break lexical recall.
func TestTokenizeBaiFenDianArtifactIsPinned(t *testing.T) {
	terms := lexical.Tokenize("一點三個百分點", true)
	require.Contains(t, terms, "1.3", "the real numeral still normalizes")
	require.Contains(t, terms, "100",
		"accepted artifact: 百 in 百分點 reads as the unit 100 (see numerals.go)")

	// The same artifact on the query side — this is the symmetry that makes it
	// harmless, and the reason a one-sided "fix" would be a regression.
	require.Contains(t, lexical.Tokenize("百分點", true), "100")
}
