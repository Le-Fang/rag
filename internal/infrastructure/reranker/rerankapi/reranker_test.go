package rerankapi_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/infrastructure/reranker/rerankapi"
)

func TestCohereStyle(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/rerank", r.URL.Path)
		require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
		json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{
				{"index": 1, "relevance_score": 0.9},
				{"index": 0, "relevance_score": 0.4},
			},
		})
	}))
	defer srv.Close()

	rr := rerankapi.New(rerankapi.Options{
		Style: "cohere_style", BaseURL: srv.URL, Model: "jina-reranker-v2-base-multilingual",
		APIKey: "k", Timeout: 5 * time.Second,
	})
	scored, err := rr.Rerank(context.Background(), "q", []string{"docA", "docB"}, 2)
	require.NoError(t, err)
	require.Equal(t, "q", got["query"])
	require.Equal(t, []any{"docA", "docB"}, got["documents"])
	require.Equal(t, float64(2), got["top_n"])
	require.Len(t, scored, 2)
	require.Equal(t, 1, scored[0].Index)
	require.InDelta(t, 0.9, scored[0].Score, 1e-9)
}

func TestTEIStyle(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
		// TEI returns a bare array without the results envelope (§9.1)
		json.NewEncoder(w).Encode([]map[string]any{
			{"index": 2, "score": 0.8},
			{"index": 0, "score": 0.5},
			{"index": 1, "score": 0.1},
		})
	}))
	defer srv.Close()

	rr := rerankapi.New(rerankapi.Options{Style: "tei", BaseURL: srv.URL, Timeout: 5 * time.Second})
	scored, err := rr.Rerank(context.Background(), "q", []string{"a", "b", "c"}, 2)
	require.NoError(t, err)
	require.Equal(t, []any{"a", "b", "c"}, got["texts"], "TEI takes texts, not documents")
	require.Len(t, scored, 2, "top_n applied client-side for TEI")
	require.Equal(t, 2, scored[0].Index)
}

func TestTEINegativeTopNClampsToEmptyWithoutPanic(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode([]map[string]any{
			{"index": 0, "score": 0.5},
			{"index": 1, "score": 0.9},
		})
	}))
	defer srv.Close()

	rr := rerankapi.New(rerankapi.Options{Style: "tei", BaseURL: srv.URL, Timeout: 5 * time.Second})
	scored, err := rr.Rerank(context.Background(), "q", []string{"a", "b"}, -1)
	require.NoError(t, err)
	require.Empty(t, scored)
}

func TestTEITopNGreaterThanDocsReturnsAll(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode([]map[string]any{
			{"index": 0, "score": 0.5},
			{"index": 1, "score": 0.9},
		})
	}))
	defer srv.Close()

	rr := rerankapi.New(rerankapi.Options{Style: "tei", BaseURL: srv.URL, Timeout: 5 * time.Second})
	scored, err := rr.Rerank(context.Background(), "q", []string{"a", "b"}, 10)
	require.NoError(t, err)
	require.Len(t, scored, 2)
}

func TestCohereStyleOutOfRangeIndexErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{
				{"index": 5, "relevance_score": 0.9},
			},
		})
	}))
	defer srv.Close()

	rr := rerankapi.New(rerankapi.Options{Style: "cohere_style", BaseURL: srv.URL, Timeout: 5 * time.Second})
	_, err := rr.Rerank(context.Background(), "q", []string{"a", "b"}, 2)
	require.Error(t, err)
	require.Contains(t, err.Error(), "out of range")
}

func TestTEIDuplicateIndexErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode([]map[string]any{
			{"index": 0, "score": 0.9},
			{"index": 0, "score": 0.4},
		})
	}))
	defer srv.Close()

	rr := rerankapi.New(rerankapi.Options{Style: "tei", BaseURL: srv.URL, Timeout: 5 * time.Second})
	_, err := rr.Rerank(context.Background(), "q", []string{"a", "b"}, 2)
	require.Error(t, err)
	require.Contains(t, err.Error(), "duplicate index")
}

func TestUnknownStyleErrors(t *testing.T) {
	rr := rerankapi.New(rerankapi.Options{Style: "bogus", BaseURL: "http://example.invalid", Timeout: 5 * time.Second})
	_, err := rr.Rerank(context.Background(), "q", []string{"a"}, 1)
	require.Error(t, err)
}
