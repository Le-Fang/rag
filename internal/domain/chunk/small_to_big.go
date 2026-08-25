package chunk

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"github.com/google/uuid"

	"poc-rag/internal/domain/document"
)

// SmallToBig implements chunker strategy "small_to_big_v1" (§4.2): parents of
// ~parent_tokens (never splitting a sentence), children of ~child_tokens with
// overlap.
type SmallToBig struct{}

func NewSmallToBig() *SmallToBig { return &SmallToBig{} }

var _ Chunker = (*SmallToBig)(nil)

// unit is the indivisible windowing element: a sentence.
type unit struct {
	text   string
	tokens int
}

func (c *SmallToBig) Chunk(ctx context.Context, doc *document.Document, cfg Config) ([]*ParentChunk, error) {
	// Config validation covers the YAML path, but variants rehydrated from
	// stored payloads reach us unvalidated. A zero budget divides by zero, and
	// an overlap at or above child_tokens amplifies tokens without bound (the
	// seed alone fills each group). Mirrors internal/config validation.
	if cfg.ParentTokens <= 0 || cfg.ChildTokens <= 0 {
		return nil, fmt.Errorf("chunking document %s: parent_tokens (%d) and child_tokens (%d) must be positive",
			doc.ID, cfg.ParentTokens, cfg.ChildTokens)
	}
	if cfg.ChildOverlap < 0 || cfg.ChildOverlap >= cfg.ChildTokens {
		return nil, fmt.Errorf("chunking document %s: need 0 <= child_overlap (%d) < child_tokens (%d)",
			doc.ID, cfg.ChildOverlap, cfg.ChildTokens)
	}
	var units []unit
	for _, s := range splitContent(doc.Content) {
		units = append(units, unit{text: s, tokens: EstimateTokens(s)})
	}
	if len(units) == 0 {
		return nil, nil
	}

	header := headerFor(doc, cfg.HeaderStrategy)
	var parents []*ParentChunk
	for _, group := range window(units, cfg.ParentTokens, 0) {
		p := &ParentChunk{
			ID: uuid.NewString(), Ordinal: len(parents), Text: joinUnits(group),
			DocumentID: doc.ID, Title: doc.Title,
		}
		childUnits := splitOversized(group, cfg.ChildTokens)
		for _, cg := range window(childUnits, cfg.ChildTokens, cfg.ChildOverlap) {
			text := joinUnits(cg)
			p.Children = append(p.Children, &ChildChunk{
				ID: uuid.NewString(), Ordinal: len(p.Children),
				Text: text, EmbedText: header + text, TokenCount: EstimateTokens(text),
			})
		}
		parents = append(parents, p)
	}
	return parents, nil
}

// window greedily packs units into groups, closing a group before the unit
// that would push it past budget. A single unit over budget forms its own
// group (a sentence is never split into two parents).
//
// With overlap > 0, each new group is seeded with the previous group's
// trailing units. The seed carries WHOLE units only: a preceding unit larger
// than the overlap budget cannot be carried at all and yields an empty seed,
// so documents whose sentences each exceed child_overlap get no overlap
// whatsoever. That is the intended granularity — overlap is measured in
// sentences, not in tokens spliced mid-unit.
//
// A group therefore holds up to budget + overlap tokens, not <= budget: the
// seed is already in cur before the budget check admits the next unit.
func window(units []unit, budget, overlap int) [][]unit {
	var out [][]unit
	var cur []unit
	curTokens := 0
	for _, u := range units {
		if curTokens > 0 && curTokens+u.tokens > budget {
			out = append(out, cur)
			seed, seedTokens := []unit(nil), 0
			for i := len(cur) - 1; i >= 0 && overlap > 0; i-- {
				if seedTokens+cur[i].tokens > overlap {
					break
				}
				seedTokens += cur[i].tokens
				seed = append([]unit{cur[i]}, seed...)
			}
			cur, curTokens = seed, seedTokens
		}
		cur = append(cur, u)
		curTokens += u.tokens
	}
	if len(cur) > 0 {
		out = append(out, cur)
	}
	return out
}

// splitOversized breaks any unit larger than maxTokens into rune slices that
// each fit the budget, so child windowing never receives an unsplittable unit.
// Parent boundaries are formed from whole units before splitting.
func splitOversized(units []unit, maxTokens int) []unit {
	var out []unit
	for _, u := range units {
		out = appendSplit(out, u, maxTokens)
	}
	return out
}

// appendSplit slices u at a uniform rune stride and recurses on any slice that
// still exceeds maxTokens. The stride is derived from the unit's token count,
// but token density varies within a unit (a CJK rune is ~4x a Latin one), so a
// single pass can leave a slice over budget; re-splitting converges because
// every pass at least halves a slice's rune count.
func appendSplit(out []unit, u unit, maxTokens int) []unit {
	if u.tokens <= maxTokens {
		return append(out, u) // fast path: no rune-slice allocation for in-budget units
	}
	runes := []rune(u.text)
	if len(runes) < 2 {
		return append(out, u)
	}
	n := max((u.tokens+maxTokens-1)/maxTokens, 2)
	per := (len(runes) + n - 1) / n
	for start := 0; start < len(runes); start += per {
		end := min(start+per, len(runes))
		nu := unit{text: string(runes[start:end])}
		nu.tokens = EstimateTokens(nu.text)
		out = appendSplit(out, nu, maxTokens)
	}
	return out
}

// joinUnits joins with single spaces. The tokenizer ignores whitespace and
// never forms bigrams across unit boundaries, so this is lossless for the
// sparse channel and harmless for embedding.
func joinUnits(us []unit) string {
	parts := make([]string, len(us))
	for i, u := range us {
		parts[i] = u.text
	}
	return strings.Join(parts, " ")
}

func splitContent(content string) []string {
	var out []string
	for _, para := range strings.Split(content, "\n\n") {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		out = append(out, splitSentences(para)...)
	}
	return out
}

// splitSentences cuts on sentence terminators, with one exception: an ASCII '.'
// flanked by digits is a decimal point, not a terminator. Splitting there would
// yield "1." + "3", which joinUnits rejoins as "1. 3", so documents would index
// "1" and "3" while the query "1.3" tokenizes whole — an end-to-end lexical
// miss. Mirrors the decimal rule in internal/domain/lexical.Tokenize (§7.1).
func splitSentences(s string) []string {
	var out []string
	var b strings.Builder
	runes := []rune(s)
	for i, r := range runes {
		b.WriteRune(r)
		if r == '.' && i > 0 && unicode.IsDigit(runes[i-1]) &&
			i+1 < len(runes) && unicode.IsDigit(runes[i+1]) {
			continue
		}
		if strings.ContainsRune("。．！？!?.", r) {
			if t := strings.TrimSpace(b.String()); t != "" {
				out = append(out, t)
			}
			b.Reset()
		}
	}
	if t := strings.TrimSpace(b.String()); t != "" {
		out = append(out, t)
	}
	return out
}

// headerFor builds the contextual header (§6). The header is part of
// embed_text — stored bytes — so any change to its shape requires re-ingesting
// every title_context variant. A title-less document gets no header at all:
// "[]\n" would be two noise characters embedded on every child.
func headerFor(doc *document.Document, strategy string) string {
	if strategy != "title_context" || doc.Title == "" {
		return ""
	}
	return fmt.Sprintf("[%s]\n", doc.Title)
}
