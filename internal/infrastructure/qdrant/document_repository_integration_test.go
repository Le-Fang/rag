//go:build integration

package qdrant_test

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/document"
	infraqdrant "poc-rag/internal/infrastructure/qdrant"
)

func TestDocumentRoundTrip(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	require.NoError(t, infraqdrant.NewSetup(c, infraqdrant.NewVariantRepository(c), nil).Run(ctx))
	repo := infraqdrant.NewDocumentRepository(c)

	doc := &document.Document{
		ID:    uuid.NewString(),
		Title: "番茄食譜", Content: "這份番茄食譜使用一點三湯匙橄欖油。 The sauce rests for 10 minutes.",
		CreatedAt: time.Now().UTC().Truncate(time.Second),
		UpdatedAt: time.Now().UTC().Truncate(time.Second),
	}
	require.NoError(t, repo.Save(ctx, doc))

	got, err := repo.Load(ctx, doc.ID)
	require.NoError(t, err)
	require.Equal(t, doc.Title, got.Title)
	require.Equal(t, doc.Content, got.Content, "content stored verbatim")
	require.True(t, doc.CreatedAt.Equal(got.CreatedAt))

	require.NoError(t, repo.Delete(ctx, doc.ID))
	_, err = repo.Load(ctx, doc.ID)
	require.ErrorIs(t, err, document.ErrNotFound)
	require.NoError(t, repo.Delete(ctx, doc.ID), "delete is idempotent")
}

func TestDocumentSaveReplacesFully(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	require.NoError(t, infraqdrant.NewSetup(c, infraqdrant.NewVariantRepository(c), nil).Run(ctx))
	repo := infraqdrant.NewDocumentRepository(c)

	id := uuid.NewString()
	require.NoError(t, repo.Save(ctx, &document.Document{ID: id, Title: "X", Content: "text"}))
	t.Cleanup(func() { _ = repo.Delete(ctx, id) })

	// Re-Save with Title cleared must fully replace the point, not merge —
	// no stale "title" key may linger from the first Save (Tasks 14/16 build
	// on full-replace semantics).
	require.NoError(t, repo.Save(ctx, &document.Document{ID: id, Title: "", Content: "text"}))

	got, err := repo.Load(ctx, id)
	require.NoError(t, err)
	require.Equal(t, "", got.Title, "second Save must fully replace the point, leaving no stale title")
}

func TestDocumentEach(t *testing.T) {
	ctx := context.Background()
	c := setupClient(t)
	require.NoError(t, infraqdrant.NewSetup(c, infraqdrant.NewVariantRepository(c), nil).Run(ctx))
	repo := infraqdrant.NewDocumentRepository(c)

	// 70 documents > scrollAll's page size, so Each must follow
	// NextPageOffset across a page boundary to see them all
	ids := map[string]bool{}
	for i := 0; i < 70; i++ {
		d := &document.Document{ID: uuid.NewString(), Content: "text"}
		require.NoError(t, repo.Save(ctx, d))
		ids[d.ID] = true
		t.Cleanup(func() { _ = repo.Delete(ctx, d.ID) })
	}
	seen := 0
	require.NoError(t, repo.Each(ctx, func(d *document.Document) error {
		if ids[d.ID] {
			seen++
		}
		return nil
	}))
	require.Equal(t, 70, seen)
}
