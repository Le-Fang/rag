package chunk_test

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/document"
)

func cfg() chunk.Config {
	return chunk.Config{
		ParentTokens: 40, ChildTokens: 12, ChildOverlap: 4,
		HeaderStrategy: "title_context",
	}
}

// sentencesDoc builds a document whose content is exactly the given sentences,
// so unit boundaries are exactly texts.
func sentencesDoc(id string, sentences ...string) *document.Document {
	return &document.Document{ID: id, Content: strings.Join(sentences, " ")}
}

// trailingUnit returns the unit text that ends s, given the ordered unit texts
// the chunker was fed. Structural: no byte slicing, no rune cutting.
func trailingUnit(t *testing.T, s string, units []string) string {
	t.Helper()
	for _, u := range units {
		if strings.HasSuffix(s, u) {
			return u
		}
	}
	require.FailNowf(t, "no trailing unit", "text %q ends with none of %q", s, units)
	return ""
}

func TestChunkContent(t *testing.T) {
	doc := &document.Document{
		ID: "doc-2", Title: "Recipe Summary",
		Content: strings.Repeat("The recipe combines tomatoes and fresh basil. ", 20),
	}
	c := chunk.NewSmallToBig()
	parents, err := c.Chunk(context.Background(), doc, cfg())
	require.NoError(t, err)
	require.NotEmpty(t, parents)
	require.Greater(t, len(parents), 1, "40-token parents over ~200 tokens of text must split")

	p := parents[0]
	// document metadata denormalized onto the parent
	require.Equal(t, "doc-2", p.DocumentID)
	require.Equal(t, "Recipe Summary", p.Title)

	require.NotEmpty(t, p.Children)
	ch := p.Children[0]
	require.True(t, strings.HasPrefix(ch.EmbedText, "[Recipe Summary]\n"))
	require.False(t, strings.HasPrefix(ch.Text, "["), "text stays header-free (§7.1)")
	require.NotEmpty(t, ch.ID)
	require.Positive(t, ch.TokenCount)
}

// A decimal point is not a sentence boundary. Splitting "1.3" into "1." + "3"
// would rejoin as "1. 3", so documents index "1" and "3" while the query "1.3"
// tokenizes whole — an end-to-end lexical miss on the §7.1 headline case.
func TestChunkContentKeepsDecimalsWhole(t *testing.T) {
	doc := &document.Document{
		ID: "doc-14", Title: "Sauce Ratio",
		Content: "Use 1.3 tablespoons of olive oil in the final sauce.",
	}
	c := chunk.NewSmallToBig()
	parents, err := c.Chunk(context.Background(), doc, chunk.Config{
		ParentTokens: 100, ChildTokens: 50, ChildOverlap: 10, HeaderStrategy: "none",
	})
	require.NoError(t, err)
	require.Len(t, parents, 1)
	require.Contains(t, parents[0].Text, "1.3", "the decimal must survive sentence splitting")
	require.NotContains(t, parents[0].Text, "1. 3", "a decimal point is not a sentence terminator")
	require.NotEmpty(t, parents[0].Children)
	for _, ch := range parents[0].Children {
		require.Contains(t, ch.Text, "1.3")
		require.NotContains(t, ch.Text, "1. 3")
	}
}

func TestChunkNeverSplitsASentence(t *testing.T) {
	// one sentence far larger than parent_tokens becomes its own parent
	longText := strings.Repeat("我們試做新的番茄食譜", 30) // ~330 tokens, no terminator
	doc := &document.Document{ID: "doc-3", Content: longText + "。 short reply."}
	c := chunk.NewSmallToBig()
	parents, err := c.Chunk(context.Background(), doc, chunk.Config{
		ParentTokens: 100, ChildTokens: 50, ChildOverlap: 10, HeaderStrategy: "none",
	})
	require.NoError(t, err)
	require.Len(t, parents, 2, "oversized sentence is its own parent; it is never split across parents")
	require.Contains(t, parents[0].Text, "我們試做")
	require.Greater(t, len(parents[0].Children), 1, "children DO split inside an oversized parent")
	for _, ch := range parents[0].Children {
		require.LessOrEqual(t, ch.TokenCount, 50+5, "children stay near child_tokens")
	}
}

func TestChildOverlap(t *testing.T) {
	// each sentence is 4 tokens — at or below child_overlap, so every seed
	// carries the whole boundary sentence (overlap is unit-granular).
	sentences := []string{"one two three four.", "five six seven.", "nine ten eleven.", "twelve thirteen."}
	c := chunk.NewSmallToBig()
	parents, err := c.Chunk(context.Background(), sentencesDoc("doc-4", sentences...), chunk.Config{
		ParentTokens: 100, ChildTokens: 8, ChildOverlap: 4, HeaderStrategy: "none",
	})
	require.NoError(t, err)
	require.Len(t, parents, 1)
	children := parents[0].Children
	require.Greater(t, len(children), 1)
	// consecutive children share a whole boundary sentence
	for i := 0; i+1 < len(children); i++ {
		last := trailingUnit(t, children[i].Text, sentences)
		require.True(t, strings.HasPrefix(children[i+1].Text, last),
			"child %d must open with child %d's trailing sentence %q, got %q",
			i+1, i, last, children[i+1].Text)
	}
}

// The flip side of unit-granular overlap, asserted as intended behavior: a
// sentence bigger than child_overlap cannot be carried, so long-sentence
// documents get no overlap at all. Changing this requires changing the seed to
// sub-sentence slices.
func TestChildOverlapIsEmptyWhenSentencesExceedTheOverlapBudget(t *testing.T) {
	sentences := []string{
		"alpha bravo charlie delta.", "echo foxtrot golf hotel.",
		"india juliett kilo lima.", "mike november oscar papa.",
	} // 6-7 tokens each (EstimateTokens: ceil(non-space chars / 4)) — what
	// matters is only that every sentence exceeds ChildOverlap: 4, so no
	// sentence can ever be carried into the next child's seed.
	c := chunk.NewSmallToBig()
	parents, err := c.Chunk(context.Background(), sentencesDoc("doc-6", sentences...), chunk.Config{
		ParentTokens: 100, ChildTokens: 12, ChildOverlap: 4, HeaderStrategy: "none",
	})
	require.NoError(t, err)
	require.Len(t, parents, 1)
	children := parents[0].Children
	require.Greater(t, len(children), 1)
	for i := 0; i+1 < len(children); i++ {
		last := trailingUnit(t, children[i].Text, sentences)
		require.NotContains(t, children[i+1].Text, last,
			"a sentence larger than child_overlap is not carried into the next child (intended)")
	}
}

func TestChunkRejectsInvalidTokenBudgets(t *testing.T) {
	doc := &document.Document{ID: "doc-8", Content: "one two three four."}
	c := chunk.NewSmallToBig()
	cases := map[string]chunk.Config{
		"zero child_tokens":             {ParentTokens: 100, ChildTokens: 0},
		"zero parent_tokens":            {ParentTokens: 0, ChildTokens: 50},
		"negative child_overlap":        {ParentTokens: 100, ChildTokens: 50, ChildOverlap: -1},
		"child_overlap >= child_tokens": {ParentTokens: 100, ChildTokens: 5, ChildOverlap: 50},
		"child_overlap == child_tokens": {ParentTokens: 100, ChildTokens: 50, ChildOverlap: 50},
	}
	for name, cf := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := c.Chunk(context.Background(), doc, cf)
			require.Error(t, err,
				"a variant rehydrated past config validation must error, not divide by zero or amplify tokens")
		})
	}
}

// A title-less document under title_context embeds the bare text — never a
// junk "[]\n" header repeated on every child.
func TestHeaderStrategyTitleContextWithoutTitle(t *testing.T) {
	doc := &document.Document{ID: "doc-15", Content: "one two three four."}
	c := chunk.NewSmallToBig()
	parents, err := c.Chunk(context.Background(), doc, chunk.Config{
		ParentTokens: 100, ChildTokens: 50, ChildOverlap: 10, HeaderStrategy: "title_context",
	})
	require.NoError(t, err)
	require.NotEmpty(t, parents[0].Children)
	for _, ch := range parents[0].Children {
		require.Equal(t, ch.Text, ch.EmbedText, "an empty title yields no header")
	}
}

func TestHeaderStrategyNone(t *testing.T) {
	doc := &document.Document{ID: "doc-9", Title: "T", Content: "one two three four."}
	c := chunk.NewSmallToBig()
	parents, err := c.Chunk(context.Background(), doc, chunk.Config{
		ParentTokens: 100, ChildTokens: 50, ChildOverlap: 10, HeaderStrategy: "none",
	})
	require.NoError(t, err)
	require.NotEmpty(t, parents[0].Children)
	for _, ch := range parents[0].Children {
		require.Equal(t, ch.Text, ch.EmbedText, `strategy "none" embeds the bare text`)
	}
}

func TestChunkEmptyBodies(t *testing.T) {
	c := chunk.NewSmallToBig()
	doc := &document.Document{ID: "doc-11", Content: "   \n\n \t "}
	parents, err := c.Chunk(context.Background(), doc, cfg())
	require.NoError(t, err)
	require.Nil(t, parents)
}
