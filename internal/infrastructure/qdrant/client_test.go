package qdrant

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// scriptedInvoker returns a grpc.UnaryInvoker that answers each call with the
// next error in script (nil = success), plus a counter of how many calls were
// made. Calls beyond the script's length fail the test — the interceptor's
// attempt budget is exactly what these tests pin.
func scriptedInvoker(t *testing.T, script []error) (grpc.UnaryInvoker, *int) {
	t.Helper()
	calls := 0
	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		if calls >= len(script) {
			t.Fatalf("invoker called %d times, scripted for only %d", calls+1, len(script))
		}
		err := script[calls]
		calls++
		return err
	}
	return invoker, &calls
}

// TestRetryInterceptorPreservesStatusCodeOnDeadContext guards against a bare
// ctx.Err() surfacing as codes.Unknown once it crosses the gRPC boundary —
// callers need the real codes.Canceled/DeadlineExceeded (§10).
func TestRetryInterceptorPreservesStatusCodeOnDeadContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	called := false
	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		called = true
		return nil
	}

	err := retryInterceptor(ctx, "Test", nil, nil, nil, invoker)
	require.Error(t, err)
	require.Equal(t, codes.Canceled, status.Code(err))
	require.False(t, called, "an already-dead context must short-circuit before invoking the call")
}

// TestRetryInterceptorRetriesTransientFailures pins the retry contract for the
// two retryable codes (§10): a transient Unavailable/DeadlineExceeded is
// retried and a subsequent success is surfaced as success, with the exact
// number of invocations accounted for.
func TestRetryInterceptorRetriesTransientFailures(t *testing.T) {
	for _, code := range []codes.Code{codes.Unavailable, codes.DeadlineExceeded} {
		t.Run(code.String(), func(t *testing.T) {
			invoker, calls := scriptedInvoker(t, []error{
				status.Error(code, "transient"),
				nil,
			})
			err := retryInterceptor(context.Background(), "Test", nil, nil, nil, invoker)
			require.NoError(t, err, "a transient %s followed by success must surface as success", code)
			require.Equal(t, 2, *calls, "exactly one retry: fail once, succeed on the second attempt")
		})
	}
}

// TestRetryInterceptorExhaustsAttemptBudget pins the budget: 3 attempts total,
// then the last error is returned as-is.
func TestRetryInterceptorExhaustsAttemptBudget(t *testing.T) {
	unavailable := status.Error(codes.Unavailable, "still down")
	invoker, calls := scriptedInvoker(t, []error{unavailable, unavailable, unavailable})

	err := retryInterceptor(context.Background(), "Test", nil, nil, nil, invoker)
	require.Equal(t, codes.Unavailable, status.Code(err))
	require.Equal(t, 3, *calls, "the budget is exactly 3 attempts (client.go retryInterceptor)")
}

// TestRetryInterceptorDoesNotRetryNonRetryableCodes: anything outside
// Unavailable/DeadlineExceeded — e.g. a caller bug surfacing as
// InvalidArgument — is returned immediately, without burning the budget.
func TestRetryInterceptorDoesNotRetryNonRetryableCodes(t *testing.T) {
	invoker, calls := scriptedInvoker(t, []error{
		status.Error(codes.InvalidArgument, "caller bug"),
	})
	err := retryInterceptor(context.Background(), "Test", nil, nil, nil, invoker)
	require.Equal(t, codes.InvalidArgument, status.Code(err))
	require.Equal(t, 1, *calls, "non-retryable codes must not be retried")
}

// TestRetryInterceptorAbortsBackoffOnContextCancel: a context canceled after a
// failed attempt aborts during the backoff wait — no further invocations, no
// sleeping out the remaining budget. The interceptor deliberately returns the
// LAST ATTEMPT's error (Unavailable here), not the context error: the real
// failure is more diagnostic than the cancellation that interrupted its retry.
func TestRetryInterceptorAbortsBackoffOnContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	invoker := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, opts ...grpc.CallOption) error {
		calls++
		cancel() // die between the first attempt and its backoff
		return status.Error(codes.Unavailable, "down")
	}

	start := time.Now()
	err := retryInterceptor(ctx, "Test", nil, nil, nil, invoker)
	elapsed := time.Since(start)

	require.Equal(t, codes.Unavailable, status.Code(err),
		"the last attempt's error is surfaced, not the context's")
	require.Equal(t, 1, calls, "no further attempts after the context dies")
	require.Less(t, elapsed, 150*time.Millisecond,
		"must abort the backoff immediately (the shortest backoff is 200ms)")
}
