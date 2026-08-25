package chunk_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/chunk"
)

func TestEstimateTokens(t *testing.T) {
	cases := []struct {
		text string
		want int
	}{
		{"", 0},
		{"我們決定", 4},                    // 1 token per CJK rune
		{"hello world", 3},             // 10 latin chars / 4, rounded up
		{"我們用 recipe", 5},              // 3 CJK + ceil(6/4)=2
		{"   \n\t  ", 0},               // whitespace only
		{"一點三 costs $1.30 now", 3 + 4}, // 3 CJK + ceil(13/4)=4 (non-space latin/symbol chars)
	}
	for _, c := range cases {
		require.Equal(t, c.want, chunk.EstimateTokens(c.text), "text: %q", c.text)
	}
}
