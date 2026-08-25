package chunk

import "unicode"

// EstimateTokens approximates XLM-R SentencePiece counts (§6.3): roughly one
// token per CJK rune, roughly four non-space characters per token otherwise.
// Expect 10–15% error; consistent across variants, so comparisons hold.
func EstimateTokens(text string) int {
	cjk, other := 0, 0
	for _, r := range text {
		switch {
		case IsCJK(r):
			cjk++
		case !unicode.IsSpace(r):
			other++
		}
	}
	return cjk + (other+3)/4
}

// IsCJK reports whether r belongs to a CJK script (shared with the lexical
// tokenizer's classification via duplication kept deliberately trivial).
func IsCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}
