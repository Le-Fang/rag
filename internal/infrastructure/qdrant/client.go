// Package qdrant implements the storage ports over the official Go gRPC
// client. Payload shapes stay in this package; domain entities cross the
// boundary through explicit mappers (§4 conventions).
package qdrant

import (
	"context"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// maxMsgSize raises the 4 MB gRPC default: a documents point can carry a
// very long content body (§6.1).
const maxMsgSize = 64 << 20

type Options struct {
	Host     string
	GRPCPort int
	APIKey   string
	UseTLS   bool
}

type Client struct {
	q *qdrant.Client
}

func NewClient(opts Options) (*Client, error) {
	q, err := qdrant.NewClient(&qdrant.Config{
		Host:   opts.Host,
		Port:   opts.GRPCPort,
		APIKey: opts.APIKey,
		UseTLS: opts.UseTLS,
		GrpcOptions: []grpc.DialOption{
			grpc.WithDefaultCallOptions(
				grpc.MaxCallRecvMsgSize(maxMsgSize),
				grpc.MaxCallSendMsgSize(maxMsgSize),
			),
			grpc.WithChainUnaryInterceptor(retryInterceptor),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("connecting to qdrant: %w", err)
	}
	return &Client{q: q}, nil
}

func (c *Client) Qdrant() *qdrant.Client { return c.q }
func (c *Client) Close() error           { return c.q.Close() }

// retryInterceptor retries Unavailable/DeadlineExceeded with bounded backoff
// (§10) on every Qdrant call. Safe for POINT writes, which are all idempotent
// (§6.1): upserts carry deterministic ids and deletes are by filter, so a retry
// after a lost ack converges on the same state. Collection DDL is NOT
// idempotent — CreateCollection/DeleteCollection error on an
// already-existing/already-absent collection — so a retried DDL call whose ack
// was lost can surface a spurious AlreadyExists/NotFound rather than a clean
// no-op. Setup and DeleteByVariant therefore gate DDL on CollectionExists
// instead of leaning on this interceptor.
// Chain interceptor so it composes with any interceptor the client library
// installs itself; if Task 2's compile shows a conflict, fall back to wrapping
// individual calls in the repositories instead.
func retryInterceptor(ctx context.Context, method string, req, reply any,
	cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	if err := ctx.Err(); err != nil {
		// A bare ctx.Err() surfaces as codes.Unknown once it crosses the gRPC
		// boundary; status.FromContextError maps context.Canceled/
		// DeadlineExceeded to the matching gRPC status code so callers relying
		// on the §10 status-code mapping (e.g. Task 12) see the right code.
		return status.FromContextError(err).Err()
	}
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			// Jitter avoids every retrying caller waking up on the same tick
			// and re-hammering a recovering server in lockstep.
			backoff := 200*time.Millisecond<<(attempt-1) + rand.N(100*time.Millisecond)
			select {
			case <-ctx.Done():
				return err
			case <-time.After(backoff):
			}
		}
		err = invoker(ctx, method, req, reply, cc, opts...)
		switch status.Code(err) {
		case codes.Unavailable, codes.DeadlineExceeded:
			continue
		default:
			return err // includes codes.OK (err == nil)
		}
	}
	return err
}

// scrollAll pages through a collection using the raw points client. The
// go-client wrappers ScrollAndOffset and ScrollAll do exist in v1.19 and do
// carry the cursor — we call the raw points client directly instead so each
// call site controls its own page size and payload selector explicitly (a
// documents point can carry a very long content body, so Task 12's caller
// needs a much smaller page than the variant registry does).
func (c *Client) scrollAll(ctx context.Context, collection string, pageSize uint32,
	payload *qdrant.WithPayloadSelector, fn func(*qdrant.RetrievedPoint) error) error {
	var offset *qdrant.PointId
	for {
		resp, err := c.q.GetPointsClient().Scroll(ctx, &qdrant.ScrollPoints{
			CollectionName: collection,
			Limit:          qdrant.PtrOf(pageSize),
			Offset:         offset,
			WithPayload:    payload,
		})
		if err != nil {
			return fmt.Errorf("scrolling %s: %w", collection, err)
		}
		for _, p := range resp.GetResult() {
			if err := fn(p); err != nil {
				return err
			}
		}
		offset = resp.GetNextPageOffset()
		if offset == nil {
			return nil
		}
	}
}
