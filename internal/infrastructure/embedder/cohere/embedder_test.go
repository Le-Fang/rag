package cohere_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/infrastructure/embedder/cohere"
)

func TestEmbedCohereShape(t *testing.T) {
	var got map[string]any
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/embed", r.URL.Path, "cohere posts to /embed, not /embeddings (§9.1)")
		gotAuth = r.Header.Get("Authorization")
		require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": map[string]any{"float": [][]float64{{0.1, 0.2, 0.3, 0.4}}},
		})
	}))
	defer srv.Close()

	e := cohere.New(cohere.Options{
		BaseURL: srv.URL, Model: "embed-multilingual-v3.0", APIKey: "test-key",
		Dimensions: 4, BatchSize: 96, Timeout: 5 * time.Second,
	})

	_, err := e.Embed(context.Background(), []string{"hello"}, embedding.KindQuery)
	require.NoError(t, err)
	require.Equal(t, "Bearer test-key", gotAuth, "api key must be sent as a bearer token (§9.1)")
	require.Equal(t, "embed-multilingual-v3.0", got["model"], "the profile's model must be sent")
	require.Equal(t, "search_query", got["input_type"], "KindQuery → search_query (§4.2)")
	require.Equal(t, []any{"hello"}, got["texts"], "cohere uses texts, not input")
	require.Equal(t, []any{"float"}, got["embedding_types"])

	_, err = e.Embed(context.Background(), []string{"hello"}, embedding.KindDocument)
	require.NoError(t, err)
	require.Equal(t, "search_document", got["input_type"], "KindDocument → search_document")
}

// An empty api_key means "this route needs no auth" (a self-hosted gateway, or
// a proxy that injects credentials). Sending a bare "Bearer " is not neutral:
// gateways that parse the header reject the malformed value outright. Match
// rerankapi, which has always omitted the header in this case.
func TestEmbedCohereOmitsAuthorizationWithoutAPIKey(t *testing.T) {
	var hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, hadAuth = r.Header["Authorization"]
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": map[string]any{"float": [][]float64{{0.1, 0.2, 0.3, 0.4}}},
		})
	}))
	defer srv.Close()

	e := cohere.New(cohere.Options{
		BaseURL: srv.URL, Model: "embed-multilingual-v3.0", APIKey: "",
		Dimensions: 4, BatchSize: 96, Timeout: 5 * time.Second,
	})
	_, err := e.Embed(context.Background(), []string{"hello"}, embedding.KindQuery)
	require.NoError(t, err)
	require.False(t, hadAuth, "no api key configured → no Authorization header at all")
}

// A vendor returning fewer embeddings than inputs would otherwise slide every
// subsequent vector onto the wrong text — the count check is what stops a
// silent misalignment of the whole batch.
func TestEmbedCohereCountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// two texts in, one embedding out
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": map[string]any{"float": [][]float64{{0.1, 0.2, 0.3, 0.4}}},
		})
	}))
	defer srv.Close()

	e := cohere.New(cohere.Options{
		BaseURL: srv.URL, Model: "embed-multilingual-v3.0", APIKey: "k",
		Dimensions: 4, BatchSize: 96, Timeout: 5 * time.Second,
	})
	_, err := e.Embed(context.Background(), []string{"a", "b"}, embedding.KindDocument)
	require.ErrorContains(t, err, "got 1 embeddings for 2 inputs")
}

func TestEmbedCohereDimensionMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Server returns width-3 vectors; adapter is configured for 4.
		json.NewEncoder(w).Encode(map[string]any{
			"embeddings": map[string]any{"float": [][]float64{{0.1, 0.2, 0.3}}},
		})
	}))
	defer srv.Close()

	e := cohere.New(cohere.Options{
		BaseURL: srv.URL, Model: "embed-multilingual-v3.0", APIKey: "k",
		Dimensions: 4, BatchSize: 96, Timeout: 5 * time.Second,
	})

	_, err := e.Embed(context.Background(), []string{"hello"}, embedding.KindQuery)
	require.ErrorIs(t, err, chunk.ErrDimensionMismatch)
}

func TestEmbedCohereBatching(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		texts := req["texts"].([]any)
		floats := make([][]float64, len(texts))
		for i := range texts {
			floats[i] = []float64{0.1, 0.2, 0.3, 0.4}
		}
		json.NewEncoder(w).Encode(map[string]any{"embeddings": map[string]any{"float": floats}})
	}))
	defer srv.Close()

	e := cohere.New(cohere.Options{
		BaseURL: srv.URL, Model: "embed-multilingual-v3.0", APIKey: "k",
		Dimensions: 4, BatchSize: 2, Timeout: 5 * time.Second,
	})

	vecs, err := e.Embed(context.Background(), []string{"a", "b", "c"}, embedding.KindDocument)
	require.NoError(t, err)
	require.Len(t, vecs, 3)
	require.Equal(t, 2, calls, "batch_size=2 over 3 texts → 2 requests")
}
