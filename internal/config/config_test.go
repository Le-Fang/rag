package config_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"poc-rag/internal/config"
)

func TestLoadValid(t *testing.T) {
	t.Setenv("TEST_OPENAI_KEY", "sk-test")
	cfg, err := config.Load("testdata/valid.yaml")
	require.NoError(t, err)
	require.Equal(t, 8080, cfg.Server.HTTPPort)
	require.Equal(t, 30*time.Second, cfg.Server.WriteTimeout)
	require.Equal(t, 6334, cfg.Qdrant.GRPCPort)
	require.Equal(t, "sk-test", cfg.Embedders["openai_3_large"].APIKey, "env vars must expand")
	require.Equal(t, 1024, cfg.Embedders["fake_1024"].Dimensions)
	require.Equal(t, "hybrid", cfg.Retrieval.Mode)
	require.Len(t, cfg.Variants, 1)
	require.Equal(t, 1200, cfg.Variants[0].Params.ParentTokens)
	require.Equal(t, 0.75, cfg.Variants[0].Params.BM25B)
}

func TestLoadRejectsBadMode(t *testing.T) {
	cfg := validConfig(t) // helper: testdata/valid.yaml loaded through config.Load
	cfg.Retrieval.Mode = "psychic"
	require.Error(t, cfg.Validate())
}

func TestLoadRejectsUnknownProfileReference(t *testing.T) {
	cfg := validConfig(t)
	cfg.Variants[0].Embedder = "nope"
	require.Error(t, cfg.Validate())
}

func TestLoadRejectsBadVariantName(t *testing.T) {
	cfg := validConfig(t)
	cfg.Variants[0].Name = "Bad-Name"
	require.Error(t, cfg.Validate())
}

func TestLoadRejectsRerankerWithoutBaseURL(t *testing.T) {
	// misconfiguration must fail at startup, not on the first search (§9)
	cfg := validConfig(t)
	cfg.Reranker = config.Reranker{Provider: "cohere_style", Model: "m", TopN: 10, MaxCandidates: 20}
	require.Error(t, cfg.Validate())

	cfg.Reranker = config.Reranker{Provider: "cohere_style", BaseURL: "https://api.jina.ai/v1", TopN: 10, MaxCandidates: 20}
	require.Error(t, cfg.Validate(), "cohere_style also needs model (tei does not — its request carries none)")
}

func TestLoadRejectsUnknownWeightKey(t *testing.T) {
	// an unknown weights key silently defaults to 1.0 in the fuser today —
	// a misconfiguration (typo'd channel name) must fail loudly instead.
	cfg := validConfig(t)
	cfg.Retrieval.Weights = map[string]float64{"dense": 1.0, "lexcial": 1.0}
	require.Error(t, cfg.Validate())
}

func TestLoadRejectsNegativeWeight(t *testing.T) {
	// zero removes a channel's influence on ranking (the channel still runs
	// and its exclusive candidates still surface); negative is always a mistake.
	cfg := validConfig(t)
	cfg.Retrieval.Weights = map[string]float64{"dense": 1.0, "lexical": -0.5}
	err := cfg.Validate()
	require.Error(t, err)
	require.ErrorContains(t, err, "removes the channel's influence on ranking",
		"the message must not claim 0 disables the channel — a zero-weight channel still runs")
}

func TestLoadRejectsDuplicateVariantName(t *testing.T) {
	cfg := validConfig(t)
	cfg.Variants = append(cfg.Variants, cfg.Variants[0])
	require.Error(t, cfg.Validate())
}

func TestLoadRejectsEmbedderWithoutBaseURL(t *testing.T) {
	cfg := validConfig(t)
	cfg.Embedders["openai_missing_url"] = config.Embedder{Provider: "openai_compatible", Dimensions: 1024, BatchSize: 8}
	require.Error(t, cfg.Validate())
}

func TestLoadRejectsSubMillisecondTimeout(t *testing.T) {
	// a bare YAML int (e.g. "read_timeout: 10") decodes as nanoseconds
	// rather than the intended duration string (e.g. "10s"); Validate must
	// catch that instead of silently accepting a near-zero timeout.
	cfg := validConfig(t)
	cfg.Server.ReadTimeout = 10 * time.Nanosecond
	require.Error(t, cfg.Validate())
}

func TestLoadNormalizesEmbedderReferenceCase(t *testing.T) {
	// viper lowercases map keys on unmarshal (Fake_1024 -> fake_1024); Load
	// must lowercase Variant.Embedder references (FAKE_1024) to match.
	cfg, err := config.Load("testdata/mixed_case_embedder.yaml")
	require.NoError(t, err)
	require.Equal(t, "fake_1024", cfg.Variants[0].Embedder)
	_, ok := cfg.Embedders["fake_1024"]
	require.True(t, ok)
}

// validConfig loads testdata/valid.yaml through the package's own loader so
// the helper cannot drift from the fixture as fields are added — tests mutate
// the result to probe one validation rule at a time.
func validConfig(t *testing.T) *config.Config {
	t.Helper()
	t.Setenv("TEST_OPENAI_KEY", "sk-test")
	cfg, err := config.Load("testdata/valid.yaml")
	require.NoError(t, err)
	return cfg
}
