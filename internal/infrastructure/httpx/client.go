// Package httpx is the shared JSON-POST client for vendor adapters:
// per-request timeouts (via the http.Client), bounded exponential backoff,
// retry on 429/5xx/transport errors, immediate failure on other 4xx (§10).
// A Retry-After header on a 429/503 overrides the exponential delay for the
// next attempt, clamped to maxRetryAfter.
package httpx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// maxBackoff bounds the exponential backoff delay between retries so a
// large maxAttempts cannot overflow the shift or produce an effectively
// unbounded wait.
const maxBackoff = 5 * time.Second

// maxRetryAfter caps how long a vendor Retry-After header may hold the next
// attempt — a broken or hostile header must not hang a bulk ingest.
const maxRetryAfter = 60 * time.Second

type Client struct {
	http        *http.Client
	maxAttempts int
	baseBackoff time.Duration
}

func New(h *http.Client, maxAttempts int, baseBackoff time.Duration) *Client {
	if maxAttempts <= 0 {
		maxAttempts = 3
	}
	if baseBackoff <= 0 {
		baseBackoff = 200 * time.Millisecond
	}
	return &Client{http: h, maxAttempts: maxAttempts, baseBackoff: baseBackoff}
}

func (c *Client) PostJSON(ctx context.Context, url string, headers map[string]string, req, resp any) error {
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("encoding request: %w", err)
	}
	var lastErr error
	var retryAfter time.Duration // from the previous attempt's 429/503; 0 → exponential backoff
	for attempt := 0; attempt < c.maxAttempts; attempt++ {
		if attempt > 0 {
			wait := nextBackoff(c.baseBackoff, attempt)
			if retryAfter > 0 {
				// the vendor named its rate-limit window; the 5s-capped
				// exponential backoff would under-wait it and burn attempts
				wait = retryAfter
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(wait):
			}
		}
		retryAfter = 0
		r, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("building request: %w", err)
		}
		r.Header.Set("Content-Type", "application/json")
		for k, v := range headers {
			r.Header.Set(k, v)
		}
		res, err := c.http.Do(r)
		if err != nil {
			lastErr = err
			continue
		}
		data, err := io.ReadAll(res.Body)
		res.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		switch {
		case res.StatusCode/100 == 2:
			if err := json.Unmarshal(data, resp); err != nil {
				return fmt.Errorf("decoding response: %w", err)
			}
			return nil
		case res.StatusCode == http.StatusTooManyRequests || res.StatusCode >= 500:
			lastErr = fmt.Errorf("status %d: %s", res.StatusCode, truncate(data, 256))
			// 429 and 503 are the statuses vendors pair with Retry-After
			// (RFC 9110 §10.2.3); other 5xx keep pure exponential backoff.
			if res.StatusCode == http.StatusTooManyRequests || res.StatusCode == http.StatusServiceUnavailable {
				retryAfter = parseRetryAfter(res.Header.Get("Retry-After"), time.Now())
			}
		default:
			return fmt.Errorf("status %d: %s", res.StatusCode, truncate(data, 256))
		}
	}
	return fmt.Errorf("after %d attempts: %w", c.maxAttempts, lastErr)
}

func truncate(b []byte, n int) string {
	if len(b) > n {
		b = b[:n]
	}
	return strings.ToValidUTF8(string(b), "")
}

// parseRetryAfter interprets a Retry-After value in either RFC 9110 §10.2.3
// form — delta-seconds or HTTP-date. It returns 0 (caller falls back to
// exponential backoff) when the value is absent, malformed, or not in the
// future, and clamps to maxRetryAfter so a broken or hostile header cannot
// hang the caller.
func parseRetryAfter(v string, now time.Time) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	var d time.Duration
	if secs, err := strconv.Atoi(v); err == nil {
		d = time.Duration(secs) * time.Second
	} else if at, err := http.ParseTime(v); err == nil {
		d = at.Sub(now)
	} else {
		return 0
	}
	if d <= 0 {
		return 0
	}
	return min(d, maxRetryAfter)
}

// nextBackoff returns the exponential backoff delay for the given attempt
// (1-indexed), clamped to maxBackoff so neither a large attempt count nor
// shift overflow can produce an unbounded or negative duration.
func nextBackoff(base time.Duration, attempt int) time.Duration {
	shift := attempt - 1
	if shift < 0 || shift > 32 {
		return maxBackoff
	}
	d := base << shift
	if d <= 0 || d > maxBackoff {
		return maxBackoff
	}
	return d
}
