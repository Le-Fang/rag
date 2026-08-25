package lexical_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/lexical"
)

func TestTokenizeSpecExample(t *testing.T) {
	got := lexical.Tokenize("我們試做新的番茄食譜 recipe", false)
	want := []string{"我們", "們試", "試做", "做新", "新的", "的番", "番茄", "茄食", "食譜", "recipe"}
	require.Equal(t, want, got)
}

func TestTokenizeSingleCJKRune(t *testing.T) {
	require.Equal(t, []string{"用", "recipe"}, lexical.Tokenize("用 recipe", false))
}

func TestTokenizeLowercasesAndDropsPunctuation(t *testing.T) {
	require.Equal(t, []string{"hello", "world"}, lexical.Tokenize("Hello, WORLD!", false))
}

func TestTokenizeDecimalNumberStaysWhole(t *testing.T) {
	require.Equal(t, []string{"1.3"}, lexical.Tokenize("1.3", false))
	require.Equal(t, []string{"v1.3", "beta"}, lexical.Tokenize("v1.3 beta", false))
}

func TestTokenizeNumeralNormalization(t *testing.T) {
	// both surface forms indexed: bigrams of the original run, plus the digit form
	got := lexical.Tokenize("一點三", true)
	require.Equal(t, []string{"一點", "點三", "1.3"}, got)

	// without normalization, only bigrams
	require.Equal(t, []string{"一點", "點三"}, lexical.Tokenize("一點三", false))
}

func TestTokenizeEmpty(t *testing.T) {
	require.Empty(t, lexical.Tokenize("  …!?、。  ", true))
}

// TestTokenizeFullWidthGapIsPinned pins the full-width↔half-width lexical gap
// (see the width-folding note on Tokenize). bigram_v1 does no NFKC-style width
// folding, and full-width digits/Latin are NOT bigrammed — they are outside
// the Han/kana/Hangul scripts, so they take the Latin/digit word path: kept
// whole, lowercased (Ａ→ａ via strings.ToLower), with the decimal-point rule
// firing for full-width digits too (unicode.IsDigit covers them; the rule
// still requires the ASCII '.', so a full-width ．splits the run). The
// resulting terms are byte-distinct from their ASCII forms, so a full-width
// query misses half-width text lexically; the dense channel bridges it. The
// gap is symmetric across index and query, which makes it a recall ceiling,
// not a correctness bug.
//
// If this test fails because width folding was added: that is a stored-bytes
// change — introduce a new lexical_tokenizer value (e.g. bigram_v2) and
// re-ingest; do not silently fold under bigram_v1.
func TestTokenizeFullWidthGapIsPinned(t *testing.T) {
	// full-width digits: word path, ASCII '.' kept between them, but the term
	// bytes differ from the half-width form.
	require.Equal(t, []string{"１.３"}, lexical.Tokenize("１.３", true))
	require.Equal(t, []string{"1.3"}, lexical.Tokenize("1.3", true))

	// a full-width ．is not the ASCII decimal point, so the run splits.
	require.Equal(t, []string{"１", "３"}, lexical.Tokenize("１．３", true))

	// full-width Latin: word path (not bigrams), lowercased to full-width
	// lowercase — still byte-distinct from ASCII.
	require.Equal(t, []string{"ａｂｃ"}, lexical.Tokenize("ＡＢＣ", true))
	require.Equal(t, []string{"abc"}, lexical.Tokenize("abc", true))

	// The load-bearing consequence: the term sets are fully disjoint, so a
	// full-width query cannot match half-width text lexically (and vice
	// versa) — same params on both sides, per the symmetry invariant.
	fullWidth := lexical.Tokenize("ＡＢＣ １.３", true)
	halfWidth := lexical.Tokenize("abc 1.3", true)
	for _, term := range fullWidth {
		require.NotContains(t, halfWidth, term)
	}
}
