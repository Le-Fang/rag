package config

import (
	"fmt"
	"regexp"
	"time"

	validation "github.com/go-ozzo/ozzo-validation/v4"
)

var variantNameRe = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

// checkTimeout rejects a duration that decoded from a bare YAML int (e.g.
// "read_timeout: 10") rather than a duration string (e.g. "10s"). Viper's
// duration hook only fires for string values, so a bare int silently becomes
// nanoseconds; zero (unset) is fine, anything positive but under 1ms is not.
func checkTimeout(field string, d time.Duration) error {
	if d > 0 && d < time.Millisecond {
		return fmt.Errorf(`%s %s is below 1ms — durations must be strings like "10s"`, field, d)
	}
	return nil
}

func (c *Config) Validate() error {
	if err := validation.ValidateStruct(&c.Server,
		validation.Field(&c.Server.HTTPPort, validation.Required, validation.Min(1)),
	); err != nil {
		return fmt.Errorf("server: %w", err)
	}
	if err := checkTimeout("server: read_timeout", c.Server.ReadTimeout); err != nil {
		return err
	}
	if err := checkTimeout("server: write_timeout", c.Server.WriteTimeout); err != nil {
		return err
	}
	if err := validation.ValidateStruct(&c.Qdrant,
		validation.Field(&c.Qdrant.Host, validation.Required),
		validation.Field(&c.Qdrant.GRPCPort, validation.Required, validation.Min(1)),
	); err != nil {
		return fmt.Errorf("qdrant: %w", err)
	}
	for name, e := range c.Embedders {
		if err := validation.ValidateStruct(&e,
			validation.Field(&e.Provider, validation.Required, validation.In("openai_compatible", "cohere", "fake")),
			validation.Field(&e.Dimensions, validation.Required, validation.Min(1)),
		); err != nil {
			return fmt.Errorf("embedders.%s: %w", name, err)
		}
		if e.Provider != "fake" && (e.BaseURL == "" || e.Model == "") {
			return fmt.Errorf("embedders.%s: base_url and model are required for provider %s", name, e.Provider)
		}
		if err := checkTimeout(fmt.Sprintf("embedders.%s: timeout", name), e.Timeout); err != nil {
			return err
		}
	}
	if err := validation.ValidateStruct(&c.Reranker,
		validation.Field(&c.Reranker.Provider, validation.Required, validation.In("cohere_style", "tei", "fake")),
		validation.Field(&c.Reranker.TopN, validation.Required, validation.Min(1)),
		validation.Field(&c.Reranker.MaxCandidates, validation.Required, validation.Min(1)),
	); err != nil {
		return fmt.Errorf("reranker: %w", err)
	}
	// fail at startup, not on the first search request (§9)
	if c.Reranker.Provider != "fake" && c.Reranker.BaseURL == "" {
		return fmt.Errorf("reranker: base_url is required for provider %s", c.Reranker.Provider)
	}
	if c.Reranker.Provider == "cohere_style" && c.Reranker.Model == "" {
		return fmt.Errorf("reranker: model is required for provider cohere_style")
	}
	if err := checkTimeout("reranker: timeout", c.Reranker.Timeout); err != nil {
		return err
	}
	if err := validation.ValidateStruct(&c.Retrieval,
		validation.Field(&c.Retrieval.Mode, validation.Required, validation.In("dense", "lexical", "hybrid")),
		validation.Field(&c.Retrieval.AnnTopK, validation.Required, validation.Min(1)),
		validation.Field(&c.Retrieval.LexicalTopK, validation.Required, validation.Min(1)),
		validation.Field(&c.Retrieval.HnswEf, validation.Required, validation.Min(1)),
		validation.Field(&c.Retrieval.RRFK, validation.Required, validation.Min(1)),
	); err != nil {
		return fmt.Errorf("retrieval: %w", err)
	}
	for key, w := range c.Retrieval.Weights {
		if key != "dense" && key != "lexical" {
			return fmt.Errorf("retrieval.weights: unknown key %q, must be one of dense, lexical", key)
		}
		if w < 0 {
			return fmt.Errorf("retrieval.weights: %q must be >= 0 (0 removes the channel's influence on ranking but still runs it — use mode to skip a channel), got %v", key, w)
		}
	}
	if len(c.Variants) == 0 {
		return fmt.Errorf("variants: at least one variant must be declared")
	}
	seen := map[string]bool{}
	for _, v := range c.Variants {
		if !variantNameRe.MatchString(v.Name) {
			return fmt.Errorf("variants.%s: name must match ^[a-z][a-z0-9_]*$", v.Name)
		}
		if seen[v.Name] {
			return fmt.Errorf("variants.%s: duplicate name", v.Name)
		}
		seen[v.Name] = true
		if _, ok := c.Embedders[v.Embedder]; !ok {
			return fmt.Errorf("variants.%s: references unknown embedder profile %q", v.Name, v.Embedder)
		}
		if v.Chunker != "small_to_big_v1" {
			return fmt.Errorf("variants.%s: unknown chunker %q", v.Name, v.Chunker)
		}
		if v.HeaderStrategy != "none" && v.HeaderStrategy != "title_context" {
			return fmt.Errorf("variants.%s: unknown header_strategy %q", v.Name, v.HeaderStrategy)
		}
		p := v.Params
		if p.ParentTokens <= 0 || p.ChildTokens <= 0 || p.ChildTokens > p.ParentTokens {
			return fmt.Errorf("variants.%s: need 0 < child_tokens <= parent_tokens", v.Name)
		}
		if p.ChildOverlap < 0 || p.ChildOverlap >= p.ChildTokens {
			return fmt.Errorf("variants.%s: need 0 <= child_overlap < child_tokens", v.Name)
		}
		if p.LexicalTokenizer != "bigram_v1" {
			return fmt.Errorf("variants.%s: unknown lexical_tokenizer %q", v.Name, p.LexicalTokenizer)
		}
		if p.BM25K1 <= 0 || p.BM25B < 0 || p.BM25B > 1 || p.BM25AvgLen <= 0 {
			return fmt.Errorf("variants.%s: bm25 params out of range", v.Name)
		}
	}
	return nil
}
