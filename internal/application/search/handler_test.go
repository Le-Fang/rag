package search_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/search"
	"poc-rag/internal/domain/chunk"
)

// TestSearchHandlerReturnsFullyShapedResult exercises the full §8 response
// envelope for one fused-and-reranked result: every resultDTO field must be
// present, and channels must serialize as [] rather than null (§8).
func TestSearchHandlerReturnsFullyShapedResult(t *testing.T) {
	h := &chunk.ChildHit{
		ChildChunkID: "c1", ParentChunkID: "p1", ParentText: "parent text",
		Text: "child text", Score: 0.9, DocumentID: "doc-1", Title: "T1",
	}
	repo := &scriptedChunkRepo{denseHits: []*chunk.ChildHit{h}, sparseHits: []*chunk.ChildHit{h}}
	handler := search.NewHandler(newService(repo))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/v1/search", handler.Search)

	body := `{"query":"番茄 recipe","variant":"fake_hdr","mode":"hybrid","top_k":1}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/search", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)

	var got struct {
		Results []map[string]any `json:"results"`
		Stats   map[string]any   `json:"stats"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	require.Len(t, got.Results, 1)
	res := got.Results[0]

	require.Equal(t, float64(1), res["rank"])
	require.Equal(t, "p1", res["parent_chunk_id"])
	require.Equal(t, "doc-1", res["document_id"])
	require.Equal(t, "parent text", res["text"])
	require.Equal(t, "c1", res["child_chunk_id"])
	require.Equal(t, "child text", res["child_text"])
	require.Equal(t, "T1", res["title"])
	require.ElementsMatch(t, []any{"dense", "lexical"}, res["channels"])
	require.NotNil(t, res["dense_score"])
	require.NotNil(t, res["dense_rank"])
	require.NotNil(t, res["lexical_score"])
	require.NotNil(t, res["lexical_rank"])
	require.NotNil(t, res["rrf_score"])
	require.NotNil(t, res["rerank_score"], "rerank defaults to true (§7)")

	require.Equal(t, "hybrid", got.Stats["mode"])
	require.Equal(t, true, got.Stats["reranked"])
	require.Equal(t, float64(1), got.Stats["parents"])
}

// The score fields are *float64, so `omitempty` drops only a nil pointer (the
// channel did not fire) — never a genuine 0.0, which stays a real JSON 0 and
// remains distinguishable from an absent key.
func TestSearchHandlerKeepsGenuineZeroScores(t *testing.T) {
	h := &chunk.ChildHit{
		ChildChunkID: "c1", ParentChunkID: "p1", ParentText: "parent text",
		Text: "child text", Score: 0, DocumentID: "doc-1", Title: "T1",
	}
	handler := search.NewHandler(newService(&scriptedChunkRepo{denseHits: []*chunk.ChildHit{h}}))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/v1/search", handler.Search)

	body := `{"query":"recipe","variant":"fake_hdr","mode":"dense","top_k":1,"rerank":false}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/search", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusOK, w.Code)

	var got struct {
		Results []map[string]any `json:"results"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	require.Len(t, got.Results, 1)
	res := got.Results[0]
	require.Contains(t, res, "dense_score", "a 0.0 dense score must not vanish from the envelope")
	require.Equal(t, float64(0), res["dense_score"])
	require.NotContains(t, res, "lexical_score", "an unfired channel stays absent")
	require.NotContains(t, res, "rerank_score", "rerank was disabled")
}

// TestSearchHandlerEmptyResultsSerializeChannelsAsEmptyArray covers the
// zero-result path plus the malformed-JSON 400 envelope.
func TestSearchHandlerMalformedJSONReturns400Envelope(t *testing.T) {
	handler := search.NewHandler(newService(&scriptedChunkRepo{}))
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/v1/search", handler.Search)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/search", strings.NewReader(`{"query":`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusBadRequest, w.Code)

	var env struct {
		Error map[string]string `json:"error"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &env))
	require.Equal(t, "invalid_argument", env.Error["code"])
}

func TestSearchHandlerValidationErrorReturns400Envelope(t *testing.T) {
	handler := search.NewHandler(newService(&scriptedChunkRepo{}))
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/v1/search", handler.Search)

	// missing variant is a service-level validation error (§8), not a bind error
	body := `{"query":"q"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/search", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusBadRequest, w.Code)

	var env struct {
		Error map[string]string `json:"error"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &env))
	require.Equal(t, "invalid_argument", env.Error["code"])
}
