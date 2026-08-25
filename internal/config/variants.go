package config

import (
	"fmt"

	"poc-rag/internal/domain/variant"
)

// DomainVariants resolves each declared variant through its embedder profile
// (§9.1): the returned variants carry the RESOLVED model and dimensions.
func (c *Config) DomainVariants() ([]*variant.IndexVariant, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	out := make([]*variant.IndexVariant, 0, len(c.Variants))
	for _, vc := range c.Variants {
		p, ok := c.Embedders[vc.Embedder]
		if !ok {
			return nil, fmt.Errorf("variant %q references unknown embedder profile %q", vc.Name, vc.Embedder)
		}
		if p.Provider == "fake" && p.Model == "" {
			p.Model = "fake"
		}
		dv := &variant.IndexVariant{
			Name:            vc.Name,
			EmbedderProfile: vc.Embedder,
			EmbedderModel:   p.Model,
			EmbeddingDim:    p.Dimensions,
			Chunker:         vc.Chunker,
			HeaderStrategy:  vc.HeaderStrategy,
			Params: variant.Params{
				ParentTokens:      vc.Params.ParentTokens,
				ChildTokens:       vc.Params.ChildTokens,
				ChildOverlap:      vc.Params.ChildOverlap,
				LexicalTokenizer:  vc.Params.LexicalTokenizer,
				NormalizeNumerals: vc.Params.NormalizeNumerals,
				BM25K1:            vc.Params.BM25K1,
				BM25B:             vc.Params.BM25B,
				BM25AvgLen:        vc.Params.BM25AvgLen,
			},
		}
		if err := dv.Validate(); err != nil {
			return nil, err
		}
		out = append(out, dv)
	}
	return out, nil
}
