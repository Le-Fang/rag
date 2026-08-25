package app

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"
)

// apperror.Respond is a package-level func and logs through the slog default,
// so the injected logger must also BE the default or §10 fault logging lands on
// a different handler than every other log line.
func TestNewLoggerBecomesTheSlogDefault(t *testing.T) {
	prev := slog.Default()
	t.Cleanup(func() { slog.SetDefault(prev) })

	log := newLogger()
	require.Same(t, log, slog.Default())
}
