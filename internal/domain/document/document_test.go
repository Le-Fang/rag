package document_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/document"
)

func TestValidateRequiresContent(t *testing.T) {
	d := &document.Document{ID: "x"}
	require.ErrorIs(t, d.Validate(), document.ErrEmptyContent)

	d.Content = "some text"
	require.NoError(t, d.Validate())
}
