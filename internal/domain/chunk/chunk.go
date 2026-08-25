package chunk

import (
	"errors"

	"poc-rag/internal/domain/lexical"
)

var ErrDimensionMismatch = errors.New("embedding dimension mismatch")

// ParentChunk is the retrieval context unit (§5). It is never stored as its
// own point: the qdrant adapter flattens it into self-contained child points.
// Document metadata is denormalized here by the chunker because the child
// payload needs it and the repository only receives parents.
type ParentChunk struct {
	ID      string
	Ordinal int
	Text    string

	DocumentID string
	Title      string

	Children []*ChildChunk
}

// ChildChunk is the embedded and searched unit. Dense/Sparse are filled by
// the ingestion service after embedding — the chunker leaves them zero.
type ChildChunk struct {
	ID         string
	Ordinal    int
	Text       string // header-free — feeds the sparse encoder (§7.1)
	EmbedText  string // header + text — what gets embedded
	TokenCount int
	Dense      []float32
	Sparse     lexical.SparseVector
}
