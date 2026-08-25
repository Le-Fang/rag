package lexical

import (
	"strings"
	"unicode"
)

// Tokenize implements the bigram_v1 scheme (§7.1): CJK runs become overlapping
// bigrams (a lone CJK rune stays whole), Latin/digit runs become lowercased
// whole words (with '.' kept between digits so "1.3" survives), and — when
// normalizeNumerals is set — each maximal Chinese-numeral run additionally
// emits its digit form, so both surface forms are indexed.
//
// Width folding is deliberately absent: full-width digits and Latin (１.３,
// ＡＢＣ) are outside the CJK scripts, so they take the Latin/digit word path
// and emit terms byte-distinct from their ASCII forms (1.3, abc) — a
// full-width query misses half-width text lexically (the dense channel
// bridges it). The gap is symmetric across index and query, so BM25
// correctness is unaffected; it is a recall ceiling, like Traditional vs
// Simplified (spec §12.2). Any future NFKC-style fold changes stored
// tokenization bytes and MUST ship as a new lexical_tokenizer value (e.g.
// bigram_v2), never a silent change to bigram_v1 — anything that shapes
// stored bytes lives in variant.Params and requires re-ingest (§7.1
// symmetry invariant).
func Tokenize(text string, normalizeNumerals bool) []string {
	runes := []rune(text)
	var terms []string
	var latin, cjk, numeral []rune

	flushLatin := func() {
		if len(latin) > 0 {
			terms = append(terms, strings.ToLower(string(latin)))
			latin = latin[:0]
		}
	}
	flushNumeral := func() {
		if normalizeNumerals && len(numeral) > 0 {
			if digits, ok := ConvertNumeral(string(numeral)); ok {
				terms = append(terms, digits)
			}
		}
		numeral = numeral[:0]
	}
	flushCJK := func() {
		if len(cjk) == 1 {
			terms = append(terms, string(cjk))
		}
		for i := 0; i+1 < len(cjk); i++ {
			terms = append(terms, string(cjk[i:i+2]))
		}
		cjk = cjk[:0]
		flushNumeral()
	}

	for i, r := range runes {
		switch {
		case isCJK(r):
			flushLatin()
			cjk = append(cjk, r)
			if isNumeralRune(r) {
				numeral = append(numeral, r)
			} else {
				flushNumeral()
			}
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			flushCJK()
			latin = append(latin, r)
		case r == '.' && len(latin) > 0 && unicode.IsDigit(latin[len(latin)-1]) &&
			i+1 < len(runes) && unicode.IsDigit(runes[i+1]):
			latin = append(latin, r) // decimal point inside a number: keep "1.3" whole
		default:
			flushCJK()
			flushLatin()
		}
	}
	flushCJK()
	flushLatin()
	return terms
}

func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}
