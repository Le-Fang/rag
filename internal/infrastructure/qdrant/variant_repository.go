package qdrant

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"

	"poc-rag/internal/domain/variant"
)

// VariantsCollection is the shared, payload-only registry collection. Exported
// so integration suites in other packages can delete the registry points they
// wrote (by name filter) without hardcoding a copy of this name that could
// drift silently.
const VariantsCollection = "variants"

// variantNamespace fixes UUIDv5 derivation of registry point IDs: Qdrant point
// IDs must be uints or UUIDs, never strings (§5). This is uuid.NameSpaceDNS,
// referenced by name rather than pinned as a literal so the choice reads as
// intentional. Do not change it: every derived point ID depends on it, so
// changing it would orphan every existing registry entry.
var variantNamespace = uuid.NameSpaceDNS

func variantPointID(name string) string {
	return uuid.NewSHA1(variantNamespace, []byte(name)).String()
}

type VariantRepository struct {
	c *Client
}

func NewVariantRepository(c *Client) *VariantRepository { return &VariantRepository{c: c} }

var _ variant.VariantRepository = (*VariantRepository)(nil)

func (r *VariantRepository) Upsert(ctx context.Context, v *variant.IndexVariant) error {
	payload := map[string]any{
		"name":             v.Name,
		"embedder_profile": v.EmbedderProfile,
		"embedder_model":   v.EmbedderModel,
		"embedding_dim":    v.EmbeddingDim,
		"chunker":          v.Chunker,
		"header_strategy":  v.HeaderStrategy,
		"params": map[string]any{
			"parent_tokens":      v.Params.ParentTokens,
			"child_tokens":       v.Params.ChildTokens,
			"child_overlap":      v.Params.ChildOverlap,
			"lexical_tokenizer":  v.Params.LexicalTokenizer,
			"normalize_numerals": v.Params.NormalizeNumerals,
			"bm25_k1":            v.Params.BM25K1,
			"bm25_b":             v.Params.BM25B,
			"bm25_avg_len":       v.Params.BM25AvgLen,
		},
	}
	// Omit created_at entirely when zero rather than writing out year 0001 —
	// callers that never set CreatedAt (e.g. a bare drift-check copy) should
	// round-trip back to a zero time, not a bogus RFC3339 timestamp.
	if !v.CreatedAt.IsZero() {
		payload["created_at"] = v.CreatedAt.Format(time.RFC3339)
	}
	_, err := r.c.q.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: VariantsCollection,
		Wait:           qdrant.PtrOf(true),
		Points: []*qdrant.PointStruct{{
			Id: qdrant.NewID(variantPointID(v.Name)),
			// Payload-only point: Vectors must be an empty map, not nil — a nil
			// Vectors field is rejected server-side even in a vector-less
			// collection (Task 2 finding; see claims_integration_test.go).
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{}),
			Payload: qdrant.NewValueMap(payload),
		}},
	})
	if err != nil {
		return fmt.Errorf("upserting variant %s: %w", v.Name, err)
	}
	return nil
}

func (r *VariantRepository) Load(ctx context.Context, name string) (*variant.IndexVariant, error) {
	points, err := r.c.q.Get(ctx, &qdrant.GetPoints{
		CollectionName: VariantsCollection,
		Ids:            []*qdrant.PointId{qdrant.NewID(variantPointID(name))},
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("loading variant %s: %w", name, err)
	}
	if len(points) == 0 {
		return nil, fmt.Errorf("variant %q: %w", name, variant.ErrUnknown)
	}
	return variantFromPayload(points[0].Payload), nil
}

// Delete removes the variant's registry point by its derived id, wait=true.
// Idempotent at the point level — deleting an absent point is a server-side
// no-op — but NOT if the variants collection itself is missing (a point
// delete against an absent collection is NotFound, per the DDL caveat in
// client.go), so callers on a possibly-unmigrated Qdrant guard with
// CollectionExists first. Deliberately not part of the domain
// variant.VariantRepository port: only the variant-drop recovery command
// needs it, and that command wires the concrete repo directly.
func (r *VariantRepository) Delete(ctx context.Context, name string) error {
	_, err := r.c.q.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: VariantsCollection,
		Wait:           qdrant.PtrOf(true),
		Points:         qdrant.NewPointsSelector(qdrant.NewID(variantPointID(name))),
	})
	if err != nil {
		return fmt.Errorf("deleting variant %s from registry: %w", name, err)
	}
	return nil
}

func (r *VariantRepository) List(ctx context.Context) ([]*variant.IndexVariant, error) {
	var out []*variant.IndexVariant
	err := r.c.scrollAll(ctx, VariantsCollection, 64, qdrant.NewWithPayload(true), func(p *qdrant.RetrievedPoint) error {
		out = append(out, variantFromPayload(p.Payload))
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("listing variants: %w", err)
	}
	return out, nil
}

func variantFromPayload(p map[string]*qdrant.Value) *variant.IndexVariant {
	v := &variant.IndexVariant{
		Name:            pStr(p, "name"),
		EmbedderProfile: pStr(p, "embedder_profile"),
		EmbedderModel:   pStr(p, "embedder_model"),
		EmbeddingDim:    pInt(p, "embedding_dim"),
		Chunker:         pStr(p, "chunker"),
		HeaderStrategy:  pStr(p, "header_strategy"),
	}
	if t := pTimePtr(p, "created_at"); t != nil {
		v.CreatedAt = *t
	}
	if params, ok := p["params"]; ok && params.GetStructValue() != nil {
		m := params.GetStructValue().Fields
		v.Params = variant.Params{
			ParentTokens:      pInt(m, "parent_tokens"),
			ChildTokens:       pInt(m, "child_tokens"),
			ChildOverlap:      pInt(m, "child_overlap"),
			LexicalTokenizer:  pStr(m, "lexical_tokenizer"),
			NormalizeNumerals: pBool(m, "normalize_numerals"),
			BM25K1:            pFloat(m, "bm25_k1"),
			BM25B:             pFloat(m, "bm25_b"),
			BM25AvgLen:        pFloat(m, "bm25_avg_len"),
		}
	}
	return v
}
