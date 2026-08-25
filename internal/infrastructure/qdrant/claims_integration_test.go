//go:build integration

// Package qdrant_test verifies the two Qdrant claims made in §14.6 of the
// design doc, and doubles as the canonical reference for the go-client API
// idioms used by the rest of the Qdrant infrastructure code.
//
// Verified against qdrant/qdrant:v1.19.0 and github.com/qdrant/go-client v1.19.0.
package qdrant_test

import (
	"context"
	"testing"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

func testClient(t *testing.T) *qdrant.Client {
	t.Helper()
	c, err := qdrant.NewClient(&qdrant.Config{Host: "localhost", Port: 6334})
	require.NoError(t, err)
	t.Cleanup(func() { c.Close() })
	return c
}

// Claim 1 (§14.6): collections with no dense vector config are supported,
// and payload-only points round-trip through them.
//
// VERIFIED — the §14.6 fallback (1-dim dummy dense vector) is NOT needed.
//
// One mandatory idiom, though: a payload-only PointStruct must still set
// Vectors to an EMPTY vectors map. Leaving the field nil makes the server
// reject the upsert with `InvalidArgument: Expected some vectors`, which looks
// like the collection shape is unsupported but is really just a nil-vs-empty
// distinction on the wire. Every payload-only write in later tasks must carry
// `Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{})`.
func TestVectorlessCollection(t *testing.T) {
	ctx := context.Background()
	c := testClient(t)
	const name = "claims_vectorless"
	_ = c.DeleteCollection(ctx, name)
	err := c.CreateCollection(ctx, &qdrant.CreateCollection{CollectionName: name})
	require.NoError(t, err, "vector-less collection must be creatable; if this fails, "+
		"switch documents/variants collections to a 1-dim dummy dense vector (§14.6 fallback)")
	t.Cleanup(func() { _ = c.DeleteCollection(ctx, name) })

	id := "0d3f1b1a-9c1e-5a7b-8f2d-111111111111"
	_, err = c.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: name,
		Wait:           qdrant.PtrOf(true),
		Points: []*qdrant.PointStruct{{
			Id: qdrant.NewID(id),
			// Required even though the collection holds no vectors: see doc comment.
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{}),
			Payload: qdrant.NewValueMap(map[string]any{"title": "hello"}),
		}},
	})
	require.NoError(t, err, "payload-only upsert into a vector-less collection must succeed "+
		"when Vectors is an empty map (a nil Vectors field is rejected server-side)")

	got, err := c.Get(ctx, &qdrant.GetPoints{
		CollectionName: name,
		Ids:            []*qdrant.PointId{qdrant.NewID(id)},
		WithPayload:    qdrant.NewWithPayload(true),
	})
	require.NoError(t, err)
	require.Len(t, got, 1)
	require.Equal(t, "hello", got[0].Payload["title"].GetStringValue())
}

// Claim 2 (§14.6): the sparse IDF modifier maintains corpus document frequencies
// and applies IDF to query terms at search time — a rare term must outscore a
// ubiquitous one for the same client-side weight.
func TestSparseIDFModifier(t *testing.T) {
	ctx := context.Background()
	c := testClient(t)
	const name = "claims_sparse_idf"
	_ = c.DeleteCollection(ctx, name)
	err := c.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: name,
		SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
			"lexical": {Modifier: qdrant.Modifier_Idf.Enum()},
		}),
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.DeleteCollection(ctx, name) })

	// term 100 appears in all 4 points; term 200 appears only in point 0.
	const common, rare = uint32(100), uint32(200)
	ids := []string{
		"0d3f1b1a-9c1e-5a7b-8f2d-000000000000", "0d3f1b1a-9c1e-5a7b-8f2d-000000000001",
		"0d3f1b1a-9c1e-5a7b-8f2d-000000000002", "0d3f1b1a-9c1e-5a7b-8f2d-000000000003",
	}
	points := make([]*qdrant.PointStruct, len(ids))
	for i, id := range ids {
		indices, weights := []uint32{common}, []float32{1.0}
		if i == 0 {
			indices, weights = []uint32{common, rare}, []float32{1.0, 1.0}
		}
		points[i] = &qdrant.PointStruct{
			Id: qdrant.NewID(id),
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
				"lexical": qdrant.NewVectorSparse(indices, weights),
			}),
		}
	}
	_, err = c.Upsert(ctx, &qdrant.UpsertPoints{CollectionName: name, Wait: qdrant.PtrOf(true), Points: points})
	require.NoError(t, err)

	score := func(term uint32) float32 {
		res, err := c.Query(ctx, &qdrant.QueryPoints{
			CollectionName: name,
			Query:          qdrant.NewQuerySparse([]uint32{term}, []float32{1.0}),
			Using:          qdrant.PtrOf("lexical"),
			Limit:          qdrant.PtrOf(uint64(1)),
		})
		require.NoError(t, err)
		require.NotEmpty(t, res, "term %d should hit", term)
		return res[0].Score
	}
	rareScore, commonScore := score(rare), score(common)
	t.Logf("IDF scores: rare(term %d)=%f  common(term %d)=%f", rare, rareScore, common, commonScore)
	require.Greater(t, rareScore, commonScore,
		"IDF modifier must weight the rare term above the ubiquitous one")
}

// apiReference is never called. It exists so that `go vet`/`go test` type-check
// the client idioms that later tasks depend on but that the tests above do not
// exercise. If the go-client API drifts, this fails to compile and the drift is
// caught here rather than mid-implementation.
//
//nolint:unused // compile-time API contract check
func apiReference(ctx context.Context) {
	// Full client config surface, including TLS/auth and gRPC interceptors.
	c, err := qdrant.NewClient(&qdrant.Config{
		Host:   "localhost",
		Port:   6334,
		APIKey: "",
		UseTLS: false,
		GrpcOptions: []grpc.DialOption{
			grpc.WithChainUnaryInterceptor(
				func(ctx context.Context, method string, req, reply any,
					cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
					return invoker(ctx, method, req, reply, cc, opts...)
				},
			),
		},
	})
	if err != nil {
		return
	}
	defer c.Close()

	// Dense query with per-request HNSW ef override and a keyword filter.
	_, _ = c.Query(ctx, &qdrant.QueryPoints{
		CollectionName: "documents",
		Query:          qdrant.NewQueryDense([]float32{0.1, 0.2}),
		Using:          qdrant.PtrOf("dense"),
		Limit:          qdrant.PtrOf(uint64(10)),
		Filter:         &qdrant.Filter{Must: []*qdrant.Condition{qdrant.NewMatchKeyword("lang", "en")}},
		Params:         &qdrant.SearchParams{HnswEf: qdrant.PtrOf(uint64(128))},
		WithPayload:    qdrant.NewWithPayload(true),
	})

	// Dense vector upsert.
	_, _ = c.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: "documents",
		Wait:           qdrant.PtrOf(true),
		Points: []*qdrant.PointStruct{{
			Id: qdrant.NewID("0d3f1b1a-9c1e-5a7b-8f2d-222222222222"),
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
				"dense": qdrant.NewVectorDense([]float32{0.1, 0.2}),
			}),
		}},
	})

	// Delete by point IDs.
	_, _ = c.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: "documents",
		Wait:           qdrant.PtrOf(true),
		Points:         qdrant.NewPointsSelector(qdrant.NewID("0d3f1b1a-9c1e-5a7b-8f2d-222222222222")),
	})

	// Paginated scroll. Note: the convenience method c.Scroll() returns only
	// []*RetrievedPoint and drops the cursor, so pagination must go through the
	// raw points client to read GetNextPageOffset().
	var offset *qdrant.PointId
	for {
		resp, err := c.GetPointsClient().Scroll(ctx, &qdrant.ScrollPoints{
			CollectionName: "documents",
			// Type asymmetry: ScrollPoints.Limit is *uint32, QueryPoints.Limit is *uint64.
			Limit:       qdrant.PtrOf(uint32(256)),
			Offset:      offset,
			WithPayload: qdrant.NewWithPayload(true),
		})
		if err != nil {
			return
		}
		_ = resp.GetResult()
		offset = resp.GetNextPageOffset()
		if offset == nil {
			break
		}
	}
}
