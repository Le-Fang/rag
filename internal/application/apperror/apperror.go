// Package apperror maps domain sentinels to HTTP statuses behind one JSON
// envelope (§10). Client messages are sanitized; detail goes to slog.
package apperror

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"

	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/variant"
)

type Error struct {
	Status  int
	Code    string
	Message string
}

func (e *Error) Error() string { return e.Message }

func Invalid(msg string) *Error {
	return &Error{Status: http.StatusBadRequest, Code: "invalid_argument", Message: msg}
}

func TooLarge(msg string) *Error {
	return &Error{Status: http.StatusRequestEntityTooLarge, Code: "request_too_large", Message: msg}
}

// FromBindError maps a ShouldBindJSON failure to its client error: a body
// that tripped http.MaxBytesReader answers 413 naming the limit — a generic
// 400 "malformed JSON" would be undiagnosable for large documents that
// legitimately approach the cap — and everything else stays a 400.
func FromBindError(err error) *Error {
	var maxErr *http.MaxBytesError
	if errors.As(err, &maxErr) {
		return TooLarge("request body exceeds " + formatByteLimit(maxErr.Limit) + " limit")
	}
	return Invalid("malformed JSON body: " + err.Error())
}

// formatByteLimit renders whole-MiB limits as "N MiB" (the production cap is
// 64 MiB) and anything else in raw bytes — never a rounded-down "0 MiB".
func formatByteLimit(n int64) string {
	if n >= 1<<20 && n%(1<<20) == 0 {
		return fmt.Sprintf("%d MiB", n>>20)
	}
	return fmt.Sprintf("%d bytes", n)
}

func Map(err error) *Error {
	var appErr *Error
	switch {
	case errors.As(err, &appErr):
		return appErr
	case errors.Is(err, context.Canceled):
		// the caller hung up before we finished — not a server fault (§10).
		// Fixed message: the wrapped chain names internal call sites.
		return &Error{Status: 499, Code: "client_closed", Message: "request canceled"}
	case errors.Is(err, context.DeadlineExceeded):
		// Fixed message: the wrapped chain carries vendor base URLs and other
		// internal detail, which must never reach the client (§10). Respond
		// still logs the full error — 504 is a 5xx.
		return &Error{Status: http.StatusGatewayTimeout, Code: "timeout", Message: "request timed out"}
	case errors.Is(err, document.ErrNotFound):
		return &Error{Status: http.StatusNotFound, Code: "not_found", Message: err.Error()}
	case errors.Is(err, variant.ErrUnknown):
		return &Error{Status: http.StatusBadRequest, Code: "unknown_variant", Message: err.Error()}
	case errors.Is(err, document.ErrEmptyContent):
		return &Error{Status: http.StatusBadRequest, Code: "invalid_argument", Message: err.Error()}
	default:
		return &Error{Status: http.StatusInternalServerError, Code: "internal", Message: "internal error"}
	}
}

// Respond writes the {"error":{"code","message"}} envelope. A 5xx is logged
// with full detail (§10) — the client only ever sees the sanitized message.
func Respond(c *gin.Context, err error) {
	e := Map(err)
	if e.Status >= 500 {
		slog.ErrorContext(c.Request.Context(), "request failed",
			"method", c.Request.Method, "path", c.FullPath(), "error", err)
	}
	c.JSON(e.Status, gin.H{"error": gin.H{"code": e.Code, "message": e.Message}})
}
