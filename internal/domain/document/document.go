package document

import (
	"errors"
	"time"
)

var (
	ErrNotFound     = errors.New("document not found")
	ErrEmptyContent = errors.New("document requires content")
)

// Document is variant-free: chunks derived from it live per variant (§5).
type Document struct {
	ID        string
	Title     string
	Content   string
	CreatedAt time.Time
	UpdatedAt time.Time
}

func (d *Document) Validate() error {
	if d.Content == "" {
		return ErrEmptyContent
	}
	return nil
}
