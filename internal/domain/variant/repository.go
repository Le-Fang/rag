package variant

import (
	"context"
	"errors"
)

var (
	ErrUnknown     = errors.New("unknown variant")
	ErrConfigDrift = errors.New("variant config drift: rename the variant, or drop and re-ingest it")
)

type VariantRepository interface {
	Upsert(ctx context.Context, v *IndexVariant) error
	Load(ctx context.Context, name string) (*IndexVariant, error) // ErrUnknown when absent
	List(ctx context.Context) ([]*IndexVariant, error)
}
