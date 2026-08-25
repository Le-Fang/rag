package lexical

import (
	"fmt"
	"strings"
)

var numeralDigits = map[rune]int64{
	'零': 0, '〇': 0, '一': 1, '二': 2, '兩': 2, '两': 2, '三': 3, '四': 4,
	'五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}

var numeralUnits = map[rune]int64{'十': 10, '百': 100, '千': 1000}

var numeralBigUnits = map[rune]int64{'萬': 10000, '万': 10000, '億': 100000000, '亿': 100000000}

func isNumeralRune(r rune) bool {
	if _, ok := numeralDigits[r]; ok {
		return true
	}
	if _, ok := numeralUnits[r]; ok {
		return true
	}
	if _, ok := numeralBigUnits[r]; ok {
		return true
	}
	return r == '點' || r == '点'
}

// ConvertNumeral converts a run of Chinese numeral runes to its digit form.
// Three readings apply to the integer part:
//   - positional (contains a unit rune 十/百/千/萬/億/万/亿): place-value
//     arithmetic, e.g. "一百二十" → "120". Colloquial magnitude shorthand is
//     read literally ("三萬五" → "30005", not 35000) — an accepted
//     approximation like §7.1's 百分點 artifact.
//   - digit-sequence (no unit rune, ≥2 digit runes): read digit-by-digit, as
//     in a spoken year, e.g. "二零二四" → "2024".
//   - single digit: both readings agree.
//
// A bare big unit with nothing before it indexes as its own magnitude
// ("萬" alone → 10000).
//
// The fractional part (after 點/点) is always digit-by-digit.
// Returns ok=false when the run is not a valid numeral. This conversion is
// symmetric across index and query, so never "fix" it on one side only.
func ConvertNumeral(s string) (string, bool) {
	runes := []rune(s)
	if len(runes) == 0 {
		return "", false
	}
	pointAt := -1
	for i, r := range runes {
		if r == '點' || r == '点' {
			if pointAt >= 0 {
				return "", false // two decimal points
			}
			pointAt = i
		} else if !isNumeralRune(r) {
			return "", false
		}
	}
	intRunes, fracRunes := runes, []rune(nil)
	if pointAt >= 0 {
		intRunes, fracRunes = runes[:pointAt], runes[pointAt+1:]
	}
	intPart := "0"
	if len(intRunes) > 0 {
		v, ok := convertIntegerRun(intRunes)
		if !ok {
			return "", false
		}
		intPart = v
	}
	if pointAt < 0 {
		return intPart, true
	}
	if len(fracRunes) == 0 {
		return "", false
	}
	var frac strings.Builder
	for _, r := range fracRunes {
		d, ok := numeralDigits[r]
		if !ok {
			return "", false // fraction part is digit-by-digit only
		}
		fmt.Fprintf(&frac, "%d", d)
	}
	return fmt.Sprintf("%s.%s", intPart, frac.String()), true
}

// convertIntegerRun converts a maximal run of integer-numeral runes (no
// decimal point) to its digit string, choosing the positional or
// digit-sequence reading per ConvertNumeral's doc comment.
func convertIntegerRun(runes []rune) (string, bool) {
	if len(runes) < 2 || hasUnitRune(runes) {
		v, ok := convertInteger(runes)
		if !ok || v < 0 {
			return "", false
		}
		return fmt.Sprintf("%d", v), true
	}
	// digit-sequence reading: every rune must be a plain digit.
	var sb strings.Builder
	for _, r := range runes {
		d, ok := numeralDigits[r]
		if !ok {
			return "", false
		}
		fmt.Fprintf(&sb, "%d", d)
	}
	return sb.String(), true
}

func hasUnitRune(runes []rune) bool {
	for _, r := range runes {
		if _, ok := numeralUnits[r]; ok {
			return true
		}
		if _, ok := numeralBigUnits[r]; ok {
			return true
		}
	}
	return false
}

// convertInteger converts a positional-reading run (place-value arithmetic)
// to its integer value. Big-unit boundaries (萬/億/万/亿) scale only the
// section accumulated since the previous boundary, not the running total.
// Unit runes (十/百/千) within a section must strictly decrease, so "十十"
// and "二十十" are rejected rather than silently misparsed.
func convertInteger(runes []rune) (int64, bool) {
	if len(runes) > 32 {
		return 0, false // absurdly long numeral run
	}
	var total, section, current int64
	var prevUnit int64 // 0 means "no unit yet in this section"
	for _, r := range runes {
		if d, ok := numeralDigits[r]; ok {
			current = d
			continue
		}
		if u, ok := numeralUnits[r]; ok {
			if prevUnit != 0 && u >= prevUnit {
				return 0, false // units must strictly decrease within a section
			}
			prevUnit = u
			if current == 0 {
				current = 1 // leading 十 as in 十三
			}
			section += current * u
			current = 0
			continue
		}
		if bu, ok := numeralBigUnits[r]; ok {
			section += current
			if section == 0 {
				section = 1 // bare 萬/億
			}
			total += section * bu
			if total < 0 {
				return 0, false // overflow
			}
			section, current = 0, 0
			prevUnit = 0
			continue
		}
		return 0, false
	}
	total += section + current
	if total < 0 {
		return 0, false // overflow
	}
	return total, true
}
