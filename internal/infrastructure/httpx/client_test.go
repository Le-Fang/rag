package httpx_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/infrastructure/httpx"
)

// The 504 path for a slow vendor rests on a stdlib guarantee: an http.Client
// timeout satisfies errors.Is(err, context.DeadlineExceeded) (net/http marks
// its timeout errors that way since go1.16). apperror.Map turns exactly that
// sentinel into the sanitized 504 (pinned in apperror's own tests); this test
// pins the other half of the chain against stdlib shifts — a REAL client
// timeout, wrapped by PostJSON's "after N attempts: %w", must still match.
// If it stops matching, vendor timeouts silently degrade from 504 to 500.
func TestPostJSONClientTimeoutIsDeadlineExceeded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Drain the body: net/http only watches for client disconnects once
		// the request body is consumed, and the watcher is what cancels
		// r.Context() so this handler (and srv.Close) returns promptly.
		io.Copy(io.Discard, r.Body)
		select {
		case <-r.Context().Done(): // client gave up — return so srv.Close needn't wait
		case <-time.After(5 * time.Second): // generous margin; never reached on a healthy run
		}
	}))
	defer srv.Close()

	hc := &http.Client{Timeout: 30 * time.Millisecond}
	c := httpx.New(hc, 2, time.Millisecond)
	err := c.PostJSON(context.Background(), srv.URL, nil, map[string]string{}, &struct{}{})
	require.Error(t, err)
	require.ErrorIs(t, err, context.DeadlineExceeded,
		"a client-timeout error must keep satisfying errors.Is(err, context.DeadlineExceeded) through the attempt wrapper")
	require.Contains(t, err.Error(), "after 2 attempts", "the wrapping itself is part of the pinned chain")
}

func TestPostJSONRetriesOn5xxThenSucceeds(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	c := httpx.New(srv.Client(), 3, time.Millisecond)
	var resp struct {
		OK bool `json:"ok"`
	}
	err := c.PostJSON(context.Background(), srv.URL, nil, map[string]string{"a": "b"}, &resp)
	require.NoError(t, err)
	require.True(t, resp.OK)
	require.Equal(t, 2, calls)
}

func TestPostJSONNoRetryOn4xx(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	c := httpx.New(srv.Client(), 3, time.Millisecond)
	err := c.PostJSON(context.Background(), srv.URL, nil, map[string]string{}, &struct{}{})
	require.Error(t, err)
	require.Equal(t, 1, calls, "4xx must fail immediately (§10)")
}

// A vendor's Retry-After must win over the (much smaller) first-attempt
// exponential backoff: a 5s-capped backoff under-waits a longer rate-limit
// window during bulk ingest and burns the attempt budget.
func TestPostJSONHonorsRetryAfterOn429(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	c := httpx.New(srv.Client(), 3, time.Millisecond) // exponential backoff alone would wait ~1ms
	start := time.Now()
	var resp struct {
		OK bool `json:"ok"`
	}
	err := c.PostJSON(context.Background(), srv.URL, nil, map[string]string{}, &resp)
	require.NoError(t, err)
	require.True(t, resp.OK)
	require.Equal(t, 2, calls)
	require.GreaterOrEqual(t, time.Since(start), time.Second,
		"the 1s Retry-After must be honored over the ~1ms exponential backoff")
}

func TestPostJSONRetriesOn429(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	c := httpx.New(srv.Client(), 3, time.Millisecond)
	err := c.PostJSON(context.Background(), srv.URL, nil, map[string]string{}, &struct{}{})
	require.Error(t, err)
	require.Equal(t, 3, calls, "429 retries up to max attempts")
}
