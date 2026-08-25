package documents_test

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/application/apperror"
	"poc-rag/internal/application/documents"
	"poc-rag/internal/application/ingestion"
	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/variant"
	fakeembed "poc-rag/internal/infrastructure/embedder/fake"
)

type memDocRepo struct {
	docs map[string]*document.Document
	log  *[]string // optional event log shared with memChunkRepo to assert cross-fake ordering
	// loadErr, when set, makes Load fail with a non-ErrNotFound error — the
	// transient-outage case a purely in-memory repo can never produce on its
	// own. Settable mid-test so a document can be stored first, then made
	// unreadable.
	loadErr error
}

func newMemDocRepo() *memDocRepo { return &memDocRepo{docs: map[string]*document.Document{}} }

func (m *memDocRepo) Save(_ context.Context, d *document.Document) error {
	if m.log != nil {
		*m.log = append(*m.log, "save:"+d.ID)
	}
	cp := *d
	m.docs[d.ID] = &cp
	return nil
}
func (m *memDocRepo) Load(_ context.Context, id string) (*document.Document, error) {
	if m.loadErr != nil {
		return nil, m.loadErr
	}
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

type memChunkRepo struct {
	purged     []string  // DeleteByDocument calls, in order
	replaced   []string  // "variant/doc" ReplaceForDocument calls, in order
	log        *[]string // optional event log shared with memDocRepo
	replaceErr error     // when set, ReplaceForDocument fails instead of tracking
}

func (m *memChunkRepo) ReplaceForDocument(_ context.Context, v, d string, _ []*chunk.ParentChunk) error {
	if m.replaceErr != nil {
		return m.replaceErr
	}
	m.replaced = append(m.replaced, v+"/"+d)
	return nil
}
func (m *memChunkRepo) DeleteByDocument(_ context.Context, id string) error {
	if m.log != nil {
		*m.log = append(*m.log, "purge:"+id)
	}
	m.purged = append(m.purged, id)
	return nil
}
func (m *memChunkRepo) DeleteByVariant(context.Context, string) error { return nil }
func (m *memChunkRepo) SearchChildren(context.Context, chunk.SearchQuery) ([]*chunk.ChildHit, error) {
	return nil, nil
}

func newService(docRepo *memDocRepo, chunkRepo *memChunkRepo) *documents.Service {
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
		chunkRepo, reg, slog.Default())
	return documents.New(docRepo, chunkRepo, ing, reg, slog.Default())
}

func TestSaveGeneratesIDAndPurgesBeforeSaving(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	var events []string
	docRepo.log, chunkRepo.log = &events, &events
	svc := newService(docRepo, chunkRepo)

	doc := &document.Document{Content: "hello recipe world"}
	saved, created, err := svc.Save(context.Background(), doc, "")
	require.NoError(t, err)
	require.True(t, created, "a fresh id is a create")
	require.NotEmpty(t, saved.ID, "server generates a UUID when id is absent (§8)")
	require.Equal(t, []string{"purge:" + saved.ID, "save:" + saved.ID}, events,
		"replace purges chunks in every variant BEFORE saving (§6.1: missing heals, stale never)")
	require.False(t, saved.CreatedAt.IsZero())
	require.Empty(t, chunkRepo.replaced, "no inline indexing without ?variant")
}

func TestSaveWithVariantIndexesInline(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	svc := newService(docRepo, chunkRepo)
	saved, _, err := svc.Save(context.Background(),
		&document.Document{Content: "hello recipe world"}, "fake_hdr")
	require.NoError(t, err)
	require.Equal(t, []string{"fake_hdr/" + saved.ID}, chunkRepo.replaced)
}

func TestSaveRejectsInvalid(t *testing.T) {
	svc := newService(newMemDocRepo(), &memChunkRepo{})
	_, _, err := svc.Save(context.Background(), &document.Document{}, "")
	require.Error(t, err, "empty body")
	_, _, err = svc.Save(context.Background(),
		&document.Document{ID: "not-a-uuid", Content: "x"}, "")
	require.Error(t, err, "id must be a UUID")
}

func TestSaveReplacePreservesCreatedAt(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	svc := newService(docRepo, chunkRepo)
	first, created, err := svc.Save(context.Background(), &document.Document{Content: "v1"}, "")
	require.NoError(t, err)
	require.True(t, created, "first save of a fresh id is a create")
	second, createdAgain, err := svc.Save(context.Background(),
		&document.Document{ID: first.ID, Content: "v2"}, "")
	require.NoError(t, err)
	require.False(t, createdAgain, "re-saving an existing id is a replace, not a create")
	require.Equal(t, first.CreatedAt, second.CreatedAt)
	require.Equal(t, []string{first.ID, first.ID}, chunkRepo.purged)
}

func TestDeleteChunksFirst(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	svc := newService(docRepo, chunkRepo)
	saved, _, err := svc.Save(context.Background(), &document.Document{Content: "x"}, "")
	require.NoError(t, err)
	require.NoError(t, svc.Delete(context.Background(), saved.ID))
	_, err = svc.Get(context.Background(), saved.ID)
	require.ErrorIs(t, err, document.ErrNotFound)
	require.NoError(t, svc.Delete(context.Background(), saved.ID), "delete is idempotent (§6.1)")
}

func TestGetRejectsNonUUID(t *testing.T) {
	svc := newService(newMemDocRepo(), &memChunkRepo{})
	_, err := svc.Get(context.Background(), "not-a-uuid")
	require.Error(t, err)
	var appErr *apperror.Error
	require.ErrorAs(t, err, &appErr)
	require.Equal(t, 400, appErr.Status)
}

func TestDeleteRejectsNonUUIDWithoutPurging(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	svc := newService(docRepo, chunkRepo)
	err := svc.Delete(context.Background(), "not-a-uuid")
	require.Error(t, err)
	var appErr *apperror.Error
	require.ErrorAs(t, err, &appErr)
	require.Equal(t, 400, appErr.Status)
	require.Empty(t, chunkRepo.purged, "a rejected id must never reach the chunk store (§10)")
}

// A typo in ?variant= must be caught before the purge: the purge is the
// destructive half of replace semantics (§6.1), so running it and only then
// discovering the variant does not exist would delete a document's chunks in
// every variant and leave the body unchanged — data loss from a typo.
func TestSaveRejectsUnknownVariantBeforeAnyWrite(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	var events []string
	docRepo.log, chunkRepo.log = &events, &events
	svc := newService(docRepo, chunkRepo)

	_, _, err := svc.Save(context.Background(),
		&document.Document{Content: "hello recipe world"}, "typo_hdr")
	require.ErrorIs(t, err, variant.ErrUnknown)
	require.Empty(t, events, "nothing purged or saved when the variant does not exist")
	require.Empty(t, chunkRepo.purged)
	require.Empty(t, chunkRepo.replaced)
}

// created_at is preserved on replace by loading the existing document first.
// A load failure that is NOT ErrNotFound is an outage, not an absence: treating
// it as "no previous document" would silently stamp a fresh created_at onto a
// document that has one, quietly rewriting history on every retry during an
// outage. It must abort instead.
func TestSaveAbortsWhenLoadFailsTransiently(t *testing.T) {
	docRepo, chunkRepo := newMemDocRepo(), &memChunkRepo{}
	svc := newService(docRepo, chunkRepo)

	first, _, err := svc.Save(context.Background(), &document.Document{Content: "v1"}, "")
	require.NoError(t, err)
	require.False(t, first.CreatedAt.IsZero())

	var events []string
	docRepo.log, chunkRepo.log = &events, &events
	outage := errors.New("qdrant unavailable")
	docRepo.loadErr = outage

	_, _, err = svc.Save(context.Background(),
		&document.Document{ID: first.ID, Content: "v2"}, "")
	require.ErrorIs(t, err, outage, "a transient load failure must surface, not be swallowed as ErrNotFound")
	require.Equal(t, []string{"purge:" + first.ID}, events,
		"the purge already ran, but nothing may be saved after an ambiguous load")

	// the stored body is untouched, so created_at was never reset
	docRepo.loadErr = nil
	stored, err := svc.Get(context.Background(), first.ID)
	require.NoError(t, err)
	require.Equal(t, "v1", stored.Content)
	require.Equal(t, first.CreatedAt, stored.CreatedAt)
}

func TestSaveWithVariantLogsWhenInlineIndexFails(t *testing.T) {
	docRepo := newMemDocRepo()
	chunkRepo := &memChunkRepo{replaceErr: errors.New("qdrant unavailable")}
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))

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
		chunkRepo, reg, logger)
	svc := documents.New(docRepo, chunkRepo, ing, reg, logger)

	_, _, err := svc.Save(context.Background(),
		&document.Document{Content: "hello recipe world"}, "fake_hdr")
	require.Error(t, err)
	require.Contains(t, buf.String(), "inline index failed")
	require.Contains(t, buf.String(), "qdrant unavailable")
}
