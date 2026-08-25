package httpserver

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/documents"
	"poc-rag/internal/application/ingestion"
	"poc-rag/internal/application/search"
	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/variant"
)

// memDocRepo / memChunkRepo are the minimal fakes the engine's handler graph
// needs; no test here reaches inline indexing or search retrieval.
type memDocRepo struct{ docs map[string]*document.Document }

func newMemDocRepo() *memDocRepo { return &memDocRepo{docs: map[string]*document.Document{}} }

func (m *memDocRepo) Save(_ context.Context, d *document.Document) error {
	cp := *d
	m.docs[d.ID] = &cp
	return nil
}
func (m *memDocRepo) Load(_ context.Context, id string) (*document.Document, error) {
	d, ok := m.docs[id]
	if !ok {
		return nil, document.ErrNotFound
	}
	return d, nil
}
func (m *memDocRepo) Delete(_ context.Context, id string) error { delete(m.docs, id); return nil }
func (m *memDocRepo) Each(_ context.Context, fn func(*document.Document) error) error {
	for _, d := range m.docs {
		if err := fn(d); err != nil {
			return err
		}
	}
	return nil
}

type memChunkRepo struct{}

func (memChunkRepo) ReplaceForDocument(context.Context, string, string, []*chunk.ParentChunk) error {
	return nil
}
func (memChunkRepo) DeleteByDocument(context.Context, string) error { return nil }
func (memChunkRepo) DeleteByVariant(context.Context, string) error  { return nil }
func (memChunkRepo) SearchChildren(context.Context, chunk.SearchQuery) ([]*chunk.ChildHit, error) {
	return nil, nil
}

// testEngine builds the real engine with an injected body cap; a cap of 0
// means the production NewEngine path (64MiB).
func testEngine(bodyLimitBytes int64) *gin.Engine {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	reg := variant.NewRegistry(nil)
	ing := ingestion.New(chunk.NewSmallToBig(), nil, memChunkRepo{}, reg, log)
	docSvc := documents.New(newMemDocRepo(), memChunkRepo{}, ing, reg, log)
	docsHandler := documents.NewHandler(docSvc)
	// nil retrievers/reranker: no test here gets past the search bind
	srchHandler := search.NewHandler(search.New(reg, nil, nil, search.Options{}, log))
	if bodyLimitBytes == 0 {
		return NewEngine(docsHandler, srchHandler, log)
	}
	return newEngine(docsHandler, srchHandler, log, bodyLimitBytes)
}

func postJSON(e *gin.Engine, path, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	e.ServeHTTP(w, req)
	return w
}

// An oversized body must answer 413 naming the limit — not the generic 400
// "malformed JSON", which is undiagnosable for large documents that
// legitimately approach the cap. Every JSON-binding route is covered;
// production behavior is identical, only the injected cap is smaller than 64MiB.
func TestEngineOversizedBodyReturns413(t *testing.T) {
	const bodyCap = 1024
	e := testEngine(bodyCap)
	big := `{"content":"` + strings.Repeat("a", 4*bodyCap) + `"}`

	for _, path := range []string{"/v1/documents", "/v1/search"} {
		w := postJSON(e, path, big)
		require.Equal(t, http.StatusRequestEntityTooLarge, w.Code, path)

		var env struct {
			Error map[string]string `json:"error"`
		}
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &env), path)
		require.Equal(t, "request_too_large", env.Error["code"], path)
		require.Contains(t, env.Error["message"], "1024 bytes", "the 413 must name the limit")
	}
}

// The happy path through the exported constructor (production 64MiB cap):
// normal-size bodies still bind and the routes answer.
func TestEngineNormalBodyStillBinds(t *testing.T) {
	e := testEngine(0) // NewEngine, production cap

	w := postJSON(e, "/v1/documents", `{"content":"hello recipe world"}`)
	require.Equal(t, http.StatusCreated, w.Code)
	var saved map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &saved))
	require.NotEmpty(t, saved["id"], "the saved document round-trips through the engine")

	w2 := httptest.NewRecorder()
	e.ServeHTTP(w2, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	require.Equal(t, http.StatusOK, w2.Code)
}
