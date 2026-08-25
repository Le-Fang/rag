package httpx

import (
	"net/http"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/require"
)

func TestNextBackoffCapsAtFiveSeconds(t *testing.T) {
	base := 200 * time.Millisecond
	require.Equal(t, base, nextBackoff(base, 1))
	require.Equal(t, 2*base, nextBackoff(base, 2))
	require.Equal(t, 4*base, nextBackoff(base, 3))
	// 200ms << 24 would be ~3355s without a cap; must clamp to 5s.
	require.Equal(t, 5*time.Second, nextBackoff(base, 25))
	// Large attempt counts must not overflow the shift into a negative duration.
	require.Equal(t, 5*time.Second, nextBackoff(base, 1000))
}

// parseRetryAfter accepts both RFC 9110 header forms (delta-seconds and
// HTTP-date). 0 means "no usable value" — the caller falls back to
// exponential backoff — and every wait is clamped to maxRetryAfter so a
// broken or hostile header cannot hang ingest.
func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)

	require.Equal(t, 5*time.Second, parseRetryAfter("5", now), "delta-seconds")
	require.Equal(t, 5*time.Second, parseRetryAfter(" 5 ", now), "delta-seconds survives whitespace")
	require.Equal(t, 3*time.Second,
		parseRetryAfter(now.Add(3*time.Second).Format(http.TimeFormat), now), "HTTP-date")

	require.Equal(t, time.Duration(0), parseRetryAfter("", now), "absent → exponential backoff")
	require.Equal(t, time.Duration(0), parseRetryAfter("soon", now), "malformed → exponential backoff")
	require.Equal(t, time.Duration(0), parseRetryAfter("-3", now), "negative → exponential backoff")
	require.Equal(t, time.Duration(0), parseRetryAfter("0", now), "zero → exponential backoff")
	require.Equal(t, time.Duration(0),
		parseRetryAfter(now.Add(-time.Minute).Format(http.TimeFormat), now),
		"past date → exponential backoff")

	require.Equal(t, maxRetryAfter, parseRetryAfter("120", now), "delta-seconds clamped to the ceiling")
	require.Equal(t, maxRetryAfter,
		parseRetryAfter(now.Add(10*time.Minute).Format(http.TimeFormat), now), "HTTP-date clamped too")
}

func TestTruncateDoesNotSplitMultiByteRune(t *testing.T) {
	// "é" is 2 bytes in UTF-8; cutting at an odd byte count lands mid-rune.
	s := strings.Repeat("é", 10)
	out := truncate([]byte(s), 5)
	require.True(t, utf8.ValidString(out), "truncate must not emit invalid UTF-8: %q", out)
}
