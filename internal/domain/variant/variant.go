package variant

import (
	"fmt"
	"regexp"
	"time"
)

var nameRe = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// CollectionPrefix is prepended to a variant's name to form its Qdrant
// collection name. Exported so adapters (e.g. Task 13) never hardcode a
// second copy of the prefix.
const CollectionPrefix = "chunks_"

// Params spans every index-time setting (§5.3) — anything here shapes stored
// points, so changing any of it is drift (§9.1.1).
//
// Float fields round-trip through Qdrant point payloads and MAY come back
// as integer-kind values (e.g. bm25_avg_len: 250 decodes as an integer, not
// a double), so payload rehydration must read both double and integer kinds.
// SameConfig compares these fields for exact equality, so a rehydration path
// that mis-reads one kind will falsely report drift.
type Params struct {
	ParentTokens      int
	ChildTokens       int
	ChildOverlap      int
	LexicalTokenizer  string
	NormalizeNumerals bool
	BM25K1            float64
	BM25B             float64
	BM25AvgLen        float64
}

// IndexVariant spans the entire indexing configuration (§5). EmbedderModel and
// EmbeddingDim are RESOLVED through the profile at load time and stored in the
// registry, so editing a profile behind an unchanged variant still trips drift.
type IndexVariant struct {
	Name            string
	EmbedderProfile string
	EmbedderModel   string
	EmbeddingDim    int
	Chunker         string
	HeaderStrategy  string
	Params          Params
	CreatedAt       time.Time
}

func (v *IndexVariant) Validate() error {
	if !nameRe.MatchString(v.Name) {
		return fmt.Errorf("variant name %q must match ^[a-z][a-z0-9_]*$", v.Name)
	}
	if v.EmbedderProfile == "" || v.EmbedderModel == "" || v.EmbeddingDim <= 0 {
		return fmt.Errorf("variant %q: embedder profile not resolved", v.Name)
	}
	return nil
}

func (v *IndexVariant) CollectionName() string { return CollectionPrefix + v.Name }

// SameConfig compares every drift-relevant field — everything except CreatedAt.
func (v *IndexVariant) SameConfig(o *IndexVariant) bool {
	return v.Name == o.Name &&
		v.EmbedderProfile == o.EmbedderProfile &&
		v.EmbedderModel == o.EmbedderModel &&
		v.EmbeddingDim == o.EmbeddingDim &&
		v.Chunker == o.Chunker &&
		v.HeaderStrategy == o.HeaderStrategy &&
		v.Params == o.Params
}
