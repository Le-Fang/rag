package documents_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/documents"
)

func newTestRouter(h *documents.Handler) *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/v1/documents", h.Save)
	r.GET("/v1/documents/:id", h.Get)
	r.DELETE("/v1/documents/:id", h.Delete)
	return r
}

func errorEnvelope(t *testing.T, body []byte) map[string]string {
	t.Helper()
	var env struct {
		Error map[string]string `json:"error"`
	}
	require.NoError(t, json.Unmarshal(body, &env))
	return env.Error
}

func TestHandlerSaveThenGetRoundTrip(t *testing.T) {
	h := documents.NewHandler(newService(newMemDocRepo(), &memChunkRepo{}))
	r := newTestRouter(h)

	reqBody := `{"content":"hello recipe world","title":"T"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/documents", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusCreated, w.Code)

	var saved map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &saved))
	id, _ := saved["id"].(string)
	require.NotEmpty(t, id)
	require.Equal(t, "T", saved["title"])
	require.NotEmpty(t, saved["created_at"], "a freshly saved document has a real created_at")

	w2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodGet, "/v1/documents/"+id, nil)
	r.ServeHTTP(w2, req2)
	require.Equal(t, http.StatusOK, w2.Code)

	var got map[string]any
	require.NoError(t, json.Unmarshal(w2.Body.Bytes(), &got))
	require.Equal(t, id, got["id"])
	require.Equal(t, "hello recipe world", got["content"])
}

// POST /v1/documents is an upsert-by-id (§8): a new id answers 201 Created,
// re-POSTing an existing id is a replace and answers 200 OK.
func TestHandlerSaveCreateReturns201UpdateReturns200(t *testing.T) {
	h := documents.NewHandler(newService(newMemDocRepo(), &memChunkRepo{}))
	r := newTestRouter(h)

	body := `{"id":"` + uuid.NewString() + `","content":"hello recipe world"}`

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/documents", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusCreated, w.Code, "a new id is a create")

	w2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodPost, "/v1/documents", strings.NewReader(body))
	req2.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w2, req2)
	require.Equal(t, http.StatusOK, w2.Code, "re-POSTing the same id is a replace")
}

func TestHandlerGetUnknownIDReturns404Envelope(t *testing.T) {
	h := documents.NewHandler(newService(newMemDocRepo(), &memChunkRepo{}))
	r := newTestRouter(h)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/documents/"+uuid.NewString(), nil)
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusNotFound, w.Code)

	env := errorEnvelope(t, w.Body.Bytes())
	require.Equal(t, "not_found", env["code"])
	require.NotEmpty(t, env["message"])
}

func TestHandlerDeleteNonUUIDReturns400Envelope(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	h := documents.NewHandler(newService(docRepo, chunkRepo))
	r := newTestRouter(h)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodDelete, "/v1/documents/not-a-uuid", nil)
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusBadRequest, w.Code)

	env := errorEnvelope(t, w.Body.Bytes())
	require.Equal(t, "invalid_argument", env["code"])
	require.Empty(t, chunkRepo.purged, "an invalid id must never reach the chunk store")
}

// DELETE answers 204 with no body — not 200 with an envelope, and not 404 for
// an already-absent id (delete is idempotent, §6.1).
func TestHandlerDeleteReturns204NoContent(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	h := documents.NewHandler(newService(docRepo, chunkRepo))
	r := newTestRouter(h)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/documents",
		strings.NewReader(`{"content":"hello recipe world"}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusCreated, w.Code)
	var saved map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &saved))
	id := saved["id"].(string)

	w2 := httptest.NewRecorder()
	r.ServeHTTP(w2, httptest.NewRequest(http.MethodDelete, "/v1/documents/"+id, nil))
	require.Equal(t, http.StatusNoContent, w2.Code)
	require.Empty(t, w2.Body.String(), "204 carries no body")
	require.Contains(t, chunkRepo.purged, id, "chunks purged before the document (§6.1)")

	w3 := httptest.NewRecorder()
	r.ServeHTTP(w3, httptest.NewRequest(http.MethodGet, "/v1/documents/"+id, nil))
	require.Equal(t, http.StatusNotFound, w3.Code)

	// deleting again is a no-op, still 204
	w4 := httptest.NewRecorder()
	r.ServeHTTP(w4, httptest.NewRequest(http.MethodDelete, "/v1/documents/"+id, nil))
	require.Equal(t, http.StatusNoContent, w4.Code)
}

// ?variant=X is the only trigger for inline indexing (§8): absent, the save is
// body-only and other variants heal on the next ingest.
func TestHandlerSaveVariantQueryParamDrivesInlineIndexing(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	h := documents.NewHandler(newService(docRepo, chunkRepo))
	r := newTestRouter(h)

	body := `{"content":"hello recipe world"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/documents?variant=fake_hdr", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusCreated, w.Code)

	var saved map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &saved))
	require.Equal(t, []string{"fake_hdr/" + saved["id"].(string)}, chunkRepo.replaced,
		"the query parameter must reach the service as the variant to index into")

	// no ?variant → no inline indexing
	w2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodPost, "/v1/documents", strings.NewReader(body))
	req2.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w2, req2)
	require.Equal(t, http.StatusCreated, w2.Code, "no id in the body → a fresh document, still a create")
	require.Len(t, chunkRepo.replaced, 1, "no ?variant means no inline index call")
}

func TestHandlerSaveUnknownVariantReturns400Envelope(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	h := documents.NewHandler(newService(docRepo, chunkRepo))
	r := newTestRouter(h)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/documents?variant=typo_hdr",
		strings.NewReader(`{"content":"hello recipe world"}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusBadRequest, w.Code)

	env := errorEnvelope(t, w.Body.Bytes())
	require.Equal(t, "unknown_variant", env["code"])
	require.Empty(t, chunkRepo.purged, "a typo'd variant must never trigger the purge (§6.1)")
}

func TestHandlerSaveMalformedJSONReturns400Envelope(t *testing.T) {
	h := documents.NewHandler(newService(newMemDocRepo(), &memChunkRepo{}))
	r := newTestRouter(h)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/documents", strings.NewReader(`{"content":`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)
	require.Equal(t, http.StatusBadRequest, w.Code)

	env := errorEnvelope(t, w.Body.Bytes())
	require.Equal(t, "invalid_argument", env["code"])
}
