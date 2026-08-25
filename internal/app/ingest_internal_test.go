package app

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/documents"
	"poc-rag/internal/application/ingestion"
	"poc-rag/internal/domain/chunk"
	docdomain "poc-rag/internal/domain/document"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/variant"
	fakeembed "poc-rag/internal/infrastructure/embedder/fake"
)

// Minimal in-memory fakes — the ctx-abort behavior under test lives entirely
// in runIngest's loops, so no store may fail on its own: pre-fix, a canceled
// context against these fakes "succeeds", which is exactly the bug (every
// real call would fail and be skip-and-counted instead of aborting).

type ingestMemDocRepo struct {
	docs map[string]*docdomain.Document
}

func newIngestMemDocRepo() *ingestMemDocRepo {
	return &ingestMemDocRepo{docs: map[string]*docdomain.Document{}}
}

func (m *ingestMemDocRepo) Save(_ context.Context, d *docdomain.Document) error {
	cp := *d
	m.docs[d.ID] = &cp
	return nil
}

func (m *ingestMemDocRepo) Load(_ context.Context, id string) (*docdomain.Document, error) {
	d, ok := m.docs[id]
	if !ok {
		return nil, docdomain.ErrNotFound
	}
	return d, nil
}

func (m *ingestMemDocRepo) Delete(_ context.Context, id string) error {
	delete(m.docs, id)
	return nil
}

func (m *ingestMemDocRepo) Each(_ context.Context, fn func(*docdomain.Document) error) error {
	for _, d := range m.docs {
		if err := fn(d); err != nil {
			return err
		}
	}
	return nil
}

type ingestMemChunkRepo struct {
	replaced []string // "variant/doc" ReplaceForDocument calls, in order
}

func (m *ingestMemChunkRepo) ReplaceForDocument(_ context.Context, v, d string, _ []*chunk.ParentChunk) error {
	m.replaced = append(m.replaced, v+"/"+d)
	return nil
}
func (m *ingestMemChunkRepo) DeleteByDocument(context.Context, string) error { return nil }
func (m *ingestMemChunkRepo) DeleteByVariant(context.Context, string) error  { return nil }
func (m *ingestMemChunkRepo) SearchChildren(context.Context, chunk.SearchQuery) ([]*chunk.ChildHit, error) {
	return nil, nil
}

func newIngestFixture(docRepo *ingestMemDocRepo, chunkRepo *ingestMemChunkRepo,
	log *slog.Logger) (*documents.Service, *ingestion.Service, *variant.Registry) {

	reg := variant.NewRegistry([]*variant.IndexVariant{{
		Name: "fake_hdr", EmbedderProfile: "fake_8", EmbedderModel: "fake", EmbeddingDim: 8,
		Chunker: "small_to_big_v1", HeaderStrategy: "none",
		Params: variant.Params{
			ParentTokens: 40, ChildTokens: 12, ChildOverlap: 4,
			LexicalTokenizer: "bigram_v1", NormalizeNumerals: true,
			BM25K1: 1.2, BM25B: 0.75, BM25AvgLen: 12,
		},
	}})
	ing := ingestion.New(chunk.NewSmallToBig(),
		map[string]embedding.EmbedderPort{"fake_8": fakeembed.New(8)},
		chunkRepo, reg, log)
	return documents.New(docRepo, chunkRepo, ing, reg, log), ing, reg
}

// Ctrl+C during a long file-mode ingest must abort the loop promptly with the
// context error — not burn through every remaining line skip-and-counting
// against a dead context and then report "N skipped documents".
func TestRunIngestFileModeAbortsOnCanceledContext(t *testing.T) {
	var buf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&buf, nil))
	docRepo, chunkRepo := newIngestMemDocRepo(), &ingestMemChunkRepo{}
	docSvc, ingSvc, reg := newIngestFixture(docRepo, chunkRepo, log)

	path := filepath.Join(t.TempDir(), "corpus.jsonl")
	corpus := `{"title":"A","content":"line one"}` + "\n" +
		`{"title":"B","content":"line two"}` + "\n" +
		`{"title":"C","content":"line three"}` + "\n"
	require.NoError(t, os.WriteFile(path, []byte(corpus), 0o644))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := runIngest(ctx, docSvc, ingSvc, docRepo, reg, log, path, "")
	require.ErrorIs(t, err, context.Canceled)
	require.NotContains(t, err.Error(), "skipped",
		"an abort is not the skip-and-count path — no \"ingest finished with N skipped documents\"")
	require.Empty(t, docRepo.docs, "no line may be processed against a dead context")
	require.NotContains(t, buf.String(), "skipping",
		"a dead context must not be logged as per-line skips")
}

// The same contract holds in re-index mode: the per-document loop streams from
// the store, and each iteration must notice the dead context before indexing.
func TestRunIngestReindexModeAbortsOnCanceledContext(t *testing.T) {
	var buf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&buf, nil))
	docRepo, chunkRepo := newIngestMemDocRepo(), &ingestMemChunkRepo{}
	docSvc, ingSvc, reg := newIngestFixture(docRepo, chunkRepo, log)

	stored := &docdomain.Document{
		ID:    "7c9e6679-7425-40de-944b-e07fc1f90ff1",
		Title: "stored", Content: "already stored body",
	}
	require.NoError(t, docRepo.Save(context.Background(), stored))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := runIngest(ctx, docSvc, ingSvc, docRepo, reg, log, "", "")
	require.ErrorIs(t, err, context.Canceled)
	require.Empty(t, chunkRepo.replaced, "no document may be indexed against a dead context")
}
