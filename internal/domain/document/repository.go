package document

import "context"

type DocumentRepository interface {
	Save(ctx context.Context, doc *Document) error
	Load(ctx context.Context, id string) (*Document, error) // ErrNotFound when absent
	Delete(ctx context.Context, id string) error            // idempotent
	// Each streams every document (scroll) — serves the ingest command (§6.2).
	Each(ctx context.Context, fn func(*Document) error) error
}
