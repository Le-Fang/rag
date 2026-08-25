package qdrant

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/qdrant/go-client/qdrant"

	"poc-rag/internal/domain/variant"
)

const documentsCollection = "documents"

// Setup is the migration replacement (§5.2): idempotent create-if-missing for
// collections, payload indexes, and registry entries, plus the drift check.
type Setup struct {
	c        *Client
	registry variant.VariantRepository
	variants []*variant.IndexVariant
}

func NewSetup(c *Client, registry variant.VariantRepository, variants []*variant.IndexVariant) *Setup {
	return &Setup{c: c, registry: registry, variants: variants}
}

func (s *Setup) Run(ctx context.Context) error {
	// 1. Base collections: payload-only, no vectors, no payload indexes (§5).
	// Vector-less, so existence is the whole check — there is no shape to drift.
	for _, name := range []string{documentsCollection, VariantsCollection} {
		if _, err := s.ensureCollection(ctx, name, nil, nil); err != nil {
			return err
		}
	}
	// 2. Per-variant registry + chunk collections. The registry drift check
	// runs BEFORE the collection is created: a drift refusal must not leave a
	// freshly manufactured empty collection behind, or reverting the config
	// would then fail the physical shape check against it, wedging startup.
	// Convergence stays sound in this order: a fresh variant writes its
	// registry entry first, and if collection creation then fails transiently,
	// the next startup finds a matching entry and proceeds to create it.
	// Accepted residual: if the registry was reset while a stale collection
	// survived AND the config was changed, this order upserts the new registry
	// entry before the shape check refuses — reverting the config then trips
	// registry drift instead. Recovery is `variant drop <name>` + re-ingest.
	for _, v := range s.variants {
		if err := v.Validate(); err != nil {
			return err
		}
		if err := s.registerVariant(ctx, v); err != nil {
			return err
		}
		if err := s.ensureChunkCollection(ctx, v); err != nil {
			return err
		}
	}
	return nil
}

// ensureCollection creates name if absent, and reports whether it already
// existed so callers whose collections carry a vector shape (chunk
// collections) can validate that shape separately — CollectionExists alone
// says nothing about whether an existing collection still matches.
func (s *Setup) ensureCollection(ctx context.Context, name string,
	vectors *qdrant.VectorsConfig, sparse *qdrant.SparseVectorConfig) (existed bool, err error) {
	exists, err := s.c.q.CollectionExists(ctx, name)
	if err != nil {
		return false, fmt.Errorf("checking collection %s: %w", name, err)
	}
	if exists {
		return true, nil
	}
	err = s.c.q.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName:      name,
		VectorsConfig:       vectors,
		SparseVectorsConfig: sparse,
	})
	if err != nil {
		return false, fmt.Errorf("creating collection %s: %w", name, err)
	}
	return false, nil
}

func (s *Setup) ensureChunkCollection(ctx context.Context, v *variant.IndexVariant) error {
	vectors := qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
		"dense": {
			Size:     uint64(v.EmbeddingDim),
			Distance: qdrant.Distance_Cosine,
			HnswConfig: &qdrant.HnswConfigDiff{
				M:           qdrant.PtrOf(uint64(16)),
				EfConstruct: qdrant.PtrOf(uint64(100)),
			},
		},
	})
	sparse := qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
		"lexical": {Modifier: qdrant.Modifier_Idf.Enum()},
	})
	existed, err := s.ensureCollection(ctx, v.CollectionName(), vectors, sparse)
	if err != nil {
		return err
	}
	if existed {
		// The collection was already there — CollectionExists says nothing
		// about whether it still matches the declared variant. Without this,
		// a chunks_* collection left at the old dimension (e.g. because the
		// registry entry that would normally catch this was reset separately)
		// passes setup silently.
		if err := s.checkChunkCollectionShape(ctx, v); err != nil {
			return err
		}
	}
	// keyword payload index on the purge filter field (§5)
	for _, field := range []string{"document_id"} {
		_, err := s.c.q.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName: v.CollectionName(),
			FieldName:      field,
			FieldType:      qdrant.FieldType_FieldTypeKeyword.Enum(),
			Wait:           qdrant.PtrOf(true),
		})
		if err != nil {
			return fmt.Errorf("indexing %s.%s: %w", v.CollectionName(), field, err)
		}
	}
	return nil
}

// checkChunkCollectionShape validates that an already-existing chunks_*
// collection's dense vector (size, distance) and its "lexical" sparse vector
// still match what v declares. This is a physical-collection check, distinct
// from and complementary to the registry drift check in registerVariant: the
// registry can be reset (or never have this variant registered) while the
// underlying — expensive to rebuild — chunk collection is left behind at an
// old shape, and that combination must still fail loudly.
func (s *Setup) checkChunkCollectionShape(ctx context.Context, v *variant.IndexVariant) error {
	info, err := s.c.q.GetCollectionInfo(ctx, v.CollectionName())
	if err != nil {
		return fmt.Errorf("checking shape of %s: %w", v.CollectionName(), err)
	}
	params := info.GetConfig().GetParams()

	// HNSW M/EfConstruct are deliberately NOT compared here: they're tunable
	// live via UpdateCollection without a reindex, so a difference there is
	// not corrupting drift the way a vector-shape or missing-modifier
	// mismatch is.
	dense := params.GetVectorsConfig().GetParamsMap().GetMap()["dense"]
	wantSize, wantDistance := uint64(v.EmbeddingDim), qdrant.Distance_Cosine
	if dense == nil || dense.GetSize() != wantSize || dense.GetDistance() != wantDistance {
		return fmt.Errorf(
			"variant %q: collection %s dense vector is size=%d distance=%s, variant declares size=%d distance=%s: %w",
			v.Name, v.CollectionName(), dense.GetSize(), dense.GetDistance(), wantSize, wantDistance, variant.ErrConfigDrift)
	}

	// Require the IDF modifier, not just the key's presence: a "lexical"
	// sparse vector without it silently degrades scoring to raw term
	// frequency instead of the IDF-weighted score §14.6 verified (see
	// claims_integration_test.go's TestSparseIDFModifier).
	sp, ok := params.GetSparseVectorsConfig().GetMap()["lexical"]
	if !ok || sp.GetModifier() != qdrant.Modifier_Idf {
		return fmt.Errorf("variant %q: collection %s is missing the %q sparse vector's IDF modifier: %w",
			v.Name, v.CollectionName(), "lexical", variant.ErrConfigDrift)
	}
	return nil
}

func (s *Setup) registerVariant(ctx context.Context, v *variant.IndexVariant) error {
	stored, err := s.registry.Load(ctx, v.Name)
	switch {
	case errors.Is(err, variant.ErrUnknown):
		reg := *v
		reg.CreatedAt = time.Now().UTC()
		return s.registry.Upsert(ctx, &reg)
	case err != nil:
		return err
	}
	if !stored.SameConfig(v) {
		return fmt.Errorf("variant %q: declared config differs from registry (stored model=%s dim=%d): %w",
			v.Name, stored.EmbedderModel, stored.EmbeddingDim, variant.ErrConfigDrift)
	}
	// Match: v itself is left untouched, still carrying whatever CreatedAt the
	// caller passed in (typically zero for a freshly-declared config) — the
	// registry's own state is authoritative and is never read back into v by
	// this function, only compared against.
	return nil
}
