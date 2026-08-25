package apperror_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/apperror"
	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/variant"
)

func TestMap(t *testing.T) {
	cases := []struct {
		err    error
		status int
		code   string
	}{
		{fmt.Errorf("doc x: %w", document.ErrNotFound), 404, "not_found"},
		{fmt.Errorf("variant y: %w", variant.ErrUnknown), 400, "unknown_variant"},
		{fmt.Errorf("save: %w", document.ErrEmptyContent), 400, "invalid_argument"},
		{apperror.Invalid("bad input"), 400, "invalid_argument"},
		{errors.New("boom"), 500, "internal"},
		{context.Canceled, 499, "client_closed"},
		{fmt.Errorf("search: %w", context.Canceled), 499, "client_closed"},
		{context.DeadlineExceeded, 504, "timeout"},
		{fmt.Errorf("search: %w", context.DeadlineExceeded), 504, "timeout"},
	}
	for _, c := range cases {
		e := apperror.Map(c.err)
		require.Equal(t, c.status, e.Status, "err: %v", c.err)
		require.Equal(t, c.code, e.Code, "err: %v", c.err)
	}
	// internal detail must not leak to clients (§10)
	require.Equal(t, "internal error", apperror.Map(errors.New("secret detail")).Message)
}

// A body that trips http.MaxBytesReader surfaces from ShouldBindJSON as a
// *http.MaxBytesError. That is a 413 naming the limit — not a generic 400
// "malformed JSON", which is undiagnosable for large documents that legitimately
// approach the cap. Anything else stays the 400 bind error.
func TestFromBindError(t *testing.T) {
	e := apperror.FromBindError(&http.MaxBytesError{Limit: 64 << 20})
	require.Equal(t, http.StatusRequestEntityTooLarge, e.Status)
	require.Equal(t, "request_too_large", e.Code)
	require.Equal(t, "request body exceeds 64 MiB limit", e.Message)

	// a wrapped chain must still match (errors.As, not a type switch)
	e = apperror.FromBindError(fmt.Errorf("read: %w", &http.MaxBytesError{Limit: 1024}))
	require.Equal(t, http.StatusRequestEntityTooLarge, e.Status)
	require.Equal(t, "request body exceeds 1024 bytes limit", e.Message,
		"a non-whole-MiB limit is reported in bytes, not rounded to 0 MiB")

	e = apperror.FromBindError(errors.New("unexpected EOF"))
	require.Equal(t, http.StatusBadRequest, e.Status)
	require.Equal(t, "invalid_argument", e.Code)
	require.Contains(t, e.Message, "malformed JSON body")

	// the 413 also survives Map, so Respond keeps the one envelope shape
	mapped := apperror.Map(apperror.FromBindError(&http.MaxBytesError{Limit: 64 << 20}))
	require.Equal(t, http.StatusRequestEntityTooLarge, mapped.Status)
	require.Equal(t, "request_too_large", mapped.Code)
}

// A timeout or cancellation arrives wrapped by the call site that hit it, so
// err.Error() names internal hosts and call paths (e.g. an embedder's base URL).
// Both branches must answer with a fixed string instead (§10).
func TestMapSanitizesTimeoutAndCancellation(t *testing.T) {
	timeout := fmt.Errorf("embedding query: post https://api.internal-vendor.example/v1/embeddings: %w",
		context.DeadlineExceeded)
	e := apperror.Map(timeout)
	require.Equal(t, 504, e.Status)
	require.Equal(t, "request timed out", e.Message)
	require.NotContains(t, e.Message, "internal-vendor.example")
	require.NotContains(t, e.Message, "embedding query")

	canceled := fmt.Errorf("searching chunks_openai_3l_hdr: %w", context.Canceled)
	e = apperror.Map(canceled)
	require.Equal(t, 499, e.Status)
	require.Equal(t, "request canceled", e.Message)
	require.NotContains(t, e.Message, "chunks_openai_3l_hdr")
}

// The sanitized 504 message must not cost the operator the detail: 504 is a
// 5xx, so Respond still logs the full wrapped chain (§10).
func TestRespondLogsTimeoutDetailButDoesNotSendIt(t *testing.T) {
	gin.SetMode(gin.TestMode)

	orig := slog.Default()
	t.Cleanup(func() { slog.SetDefault(orig) })
	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/search", nil)
	apperror.Respond(c, fmt.Errorf("embedding query: post https://api.internal-vendor.example/v1/embeddings: %w",
		context.DeadlineExceeded))

	require.Equal(t, 504, w.Code)
	require.Contains(t, buf.String(), "internal-vendor.example", "the operator keeps the detail")
	require.NotContains(t, w.Body.String(), "internal-vendor.example", "the client never sees it")
	require.Contains(t, w.Body.String(), "request timed out")
}

func TestRespondLogsOnlyServerErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	orig := slog.Default()
	t.Cleanup(func() { slog.SetDefault(orig) })

	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))

	// a 500 must be logged with detail (§10)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/documents/x", nil)
	apperror.Respond(c, errors.New("secret detail"))
	require.Contains(t, buf.String(), "request failed")
	require.Contains(t, buf.String(), "secret detail")

	// a 400 must not be logged — it is not a server fault
	buf.Reset()
	w2 := httptest.NewRecorder()
	c2, _ := gin.CreateTestContext(w2)
	c2.Request = httptest.NewRequest(http.MethodGet, "/v1/documents/x", nil)
	apperror.Respond(c2, apperror.Invalid("bad input"))
	require.Empty(t, buf.String())
}
