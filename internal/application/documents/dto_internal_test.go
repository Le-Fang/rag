package documents

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/document"
)

func TestFromDomainOmitsZeroTimestamps(t *testing.T) {
	dto := fromDomain(&document.Document{ID: "x"})
	require.Nil(t, dto.CreatedAt, "a zero CreatedAt must not serialize as a real timestamp")
	require.Nil(t, dto.UpdatedAt)

	now := time.Now()
	dto2 := fromDomain(&document.Document{ID: "y", CreatedAt: now, UpdatedAt: now})
	require.NotNil(t, dto2.CreatedAt)
	require.Equal(t, now, *dto2.CreatedAt)
	require.NotNil(t, dto2.UpdatedAt)
	require.Equal(t, now, *dto2.UpdatedAt)
}
