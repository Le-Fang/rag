package chunk

import (
	"context"

	"poc-rag/internal/domain/document"
)

// Config is derived from the variant by the ingestion service.
type Config struct {
	ParentTokens   int
	ChildTokens    int
	ChildOverlap   int
	HeaderStrategy string // none | title_context
}

type Chunker interface {
	Chunk(ctx context.Context, doc *document.Document, cfg Config) ([]*ParentChunk, error)
}
