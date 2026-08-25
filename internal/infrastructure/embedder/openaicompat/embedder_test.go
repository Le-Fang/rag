package openaicompat_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/infrastructure/embedder/openaicompat"
)

func fakeOpenAI(t *testing.T, dim int) (*httptest.Server, *[]map[string]any) {
	var requests []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/embeddings", r.URL.Path)
		require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		requests = append(requests, req)
		inputs := req["input"].([]any)
		data := make([]map[string]any, len(inputs))
		for i := range inputs {
			vec := make([]float64, dim)
			vec[0] = float64(i + 1)
			data[i] = map[string]any{"index": i, "embedding": vec}
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	return srv, &requests
}

func newEmbedder(url string, dim int) *openaicompat.Embedder {
	return openaicompat.New(openaicompat.Options{
		BaseURL: url, Model: "text-embedding-3-large", APIKey: "test-key",
		Dimensions: dim, BatchSize: 2, Timeout: 5 * time.Second,
	})
}

func TestEmbedRequestShapeAndBatching(t *testing.T) {
	srv, requests := fakeOpenAI(t, 4)
	defer srv.Close()
	e := newEmbedder(srv.URL, 4)

	vecs, err := e.Embed(context.Background(), []string{"a", "b", "c"}, embedding.KindDocument)
	require.NoError(t, err)
	require.Len(t, vecs, 3)
	require.Len(t, *requests, 2, "batch_size=2 over 3 texts → 2 requests")
	first := (*requests)[0]
	require.Equal(t, "text-embedding-3-large", first["model"])
	require.Equal(t, float64(4), first["dimensions"], "dimensions param must be sent (§9.1)")

	// fakeOpenAI marks vec[0] with the 1-based position of the text within
	// its own request batch: batch 1 is ["a","b"] (markers 1,2), batch 2 is
	// ["c"] (marker 1). Each returned vector must land next to its text.
	require.Equal(t, float32(1), vecs[0][0], "vector for \"a\" (batch 1, pos 0)")
	require.Equal(t, float32(2), vecs[1][0], "vector for \"b\" (batch 1, pos 1)")
	require.Equal(t, float32(1), vecs[2][0], "vector for \"c\" (batch 2, pos 0)")
}

func TestEmbedHandlesOutOfOrderResponse(t *testing.T) {
	dim := 4
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		inputs := req["input"].([]any)
		data := make([]map[string]any, len(inputs))
		for i := range inputs {
			vec := make([]float64, dim)
			vec[0] = float64(i + 1)
			data[i] = map[string]any{"index": i, "embedding": vec}
		}
		// Reverse the array order while keeping the index fields correct,
		// simulating a continuous-batching server that completes out of order.
		for l, r := 0, len(data)-1; l < r; l, r = l+1, r-1 {
			data[l], data[r] = data[r], data[l]
		}
		require.NoError(t, json.NewEncoder(w).Encode(map[string]any{"data": data}))
	}))
	defer srv.Close()
	// BatchSize large enough that all texts land in a single request, so
	// the index markers below are unambiguous (not restarted per batch).
	e := openaicompat.New(openaicompat.Options{
		BaseURL: srv.URL, Model: "text-embedding-3-large", APIKey: "test-key",
		Dimensions: dim, BatchSize: 10, Timeout: 5 * time.Second,
	})

	vecs, err := e.Embed(context.Background(), []string{"a", "b", "c"}, embedding.KindDocument)
	require.NoError(t, err)
	require.Len(t, vecs, 3)
	for i, v := range vecs {
		require.Equal(t, float32(i+1), v[0], "vector at position %d must match its text despite out-of-order response", i)
	}
}

func TestEmbedDimensionMismatch(t *testing.T) {
	srv, _ := fakeOpenAI(t, 3) // server returns width 3
	defer srv.Close()
	e := newEmbedder(srv.URL, 4) // adapter expects 4

	_, err := e.Embed(context.Background(), []string{"a"}, embedding.KindQuery)
	require.ErrorIs(t, err, chunk.ErrDimensionMismatch)
}

// An empty api_key means "this route needs no auth" (a self-hosted vLLM/TEI
// route, or a proxy that injects credentials). Sending a bare "Bearer " is not
// neutral: gateways that parse the header reject the malformed value outright.
// Match rerankapi, which has always omitted the header in this case.
func TestEmbedOmitsAuthorizationWithoutAPIKey(t *testing.T) {
	var hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, hadAuth = r.Header["Authorization"]
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[0.1,0.2,0.3,0.4]}]}`)
	}))
	defer srv.Close()

	e := openaicompat.New(openaicompat.Options{
		BaseURL: srv.URL, Model: "text-embedding-3-large", APIKey: "",
		Dimensions: 4, BatchSize: 2, Timeout: 5 * time.Second,
	})
	_, err := e.Embed(context.Background(), []string{"a"}, embedding.KindQuery)
	require.NoError(t, err)
	require.False(t, hadAuth, "no api key configured → no Authorization header at all")
}

func TestEmbedCountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data":[]}`)
	}))
	defer srv.Close()
	e := newEmbedder(srv.URL, 4)
	_, err := e.Embed(context.Background(), []string{"a"}, embedding.KindQuery)
	require.Error(t, err)
}
