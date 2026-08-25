package search

import (
	"context"
	"fmt"
	"time"

	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/retrieval"
)

// DenseRetriever embeds the query with the VARIANT'S OWN embedder profile
// (§9.1) and runs the ANN channel.
type DenseRetriever struct {
	embedders map[string]embedding.EmbedderPort
	chunks    chunk.ChunkRepository
}

func NewDenseRetriever(embedders map[string]embedding.EmbedderPort, chunks chunk.ChunkRepository) *DenseRetriever {
	return &DenseRetriever{embedders: embedders, chunks: chunks}
}

var _ retrieval.Retriever = (*DenseRetriever)(nil)

func (r *DenseRetriever) Channel() string { return "dense" }

func (r *DenseRetriever) Retrieve(ctx context.Context, q retrieval.Query) ([]retrieval.Candidate, error) {
	emb, ok := r.embedders[q.Variant.EmbedderProfile]
	if !ok {
		return nil, fmt.Errorf("variant %s: no adapter for embedder profile %q", q.Variant.Name, q.Variant.EmbedderProfile)
	}
	embStart := time.Now()
	vecs, err := emb.Embed(ctx, []string{q.Text}, embedding.KindQuery)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	// Vendor-returned shapes are untrusted (§10): a nil/empty batch for one
	// text would panic at vecs[0] in the request path.
	if len(vecs) != 1 {
		return nil, fmt.Errorf("embedding query: expected 1 vector, got %d", len(vecs))
	}
	q.Timings.Add("embed_query", time.Since(embStart))

	searchStart := time.Now()
	hits, err := r.chunks.SearchChildren(ctx, chunk.SearchQuery{
		Variant: q.Variant.Name,
		Dense:   &chunk.DenseQuery{Vector: embedding.Normalize(vecs[0]), Ef: q.Ef},
		Limit:   q.AnnTopK,
	})
	if err != nil {
		return nil, err
	}
	q.Timings.Add("dense", time.Since(searchStart))
	return toCandidates(hits), nil
}

func toCandidates(hits []*chunk.ChildHit) []retrieval.Candidate {
	out := make([]retrieval.Candidate, len(hits))
	for i, h := range hits {
		out[i] = retrieval.Candidate{
			ChildChunkID:  h.ChildChunkID,
			ParentChunkID: h.ParentChunkID,
			Score:         h.Score,
			Rank:          i + 1,
			Hit:           h,
		}
	}
	return out
}
