package config_test

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDomainVariantsResolvesProfile(t *testing.T) {
	cfg := validConfig(t) // helper from Task 3
	vs, err := cfg.DomainVariants()
	require.NoError(t, err)
	require.Len(t, vs, 1)
	require.Equal(t, "fake", vs[0].EmbedderModel,
		"fake profiles carry no config model, so resolution defaults it to \"fake\" — the registry stores the RESOLVED model so profile edits trip the drift check (§9.1.1)")
	require.Equal(t, cfg.Embedders[cfg.Variants[0].Embedder].Dimensions, vs[0].EmbeddingDim)
}

func TestDomainVariantsRejectsDuplicateName(t *testing.T) {
	// DomainVariants must run full Validate() — otherwise a duplicate name
	// silently vanishes when the Registry keys variants by name.
	cfg := validConfig(t)
	cfg.Variants = append(cfg.Variants, cfg.Variants[0])
	_, err := cfg.DomainVariants()
	require.Error(t, err)
}
