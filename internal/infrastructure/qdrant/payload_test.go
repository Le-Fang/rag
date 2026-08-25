package qdrant

import (
	"testing"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
)

func TestPFloat(t *testing.T) {
	t.Run("double kind", func(t *testing.T) {
		p := map[string]*qdrant.Value{"f": {Kind: &qdrant.Value_DoubleValue{DoubleValue: 1.5}}}
		require.Equal(t, 1.5, pFloat(p, "f"))
	})

	t.Run("integer kind round-trips as float (§ Params comment)", func(t *testing.T) {
		p := map[string]*qdrant.Value{"f": {Kind: &qdrant.Value_IntegerValue{IntegerValue: 250}}}
		require.Equal(t, float64(250), pFloat(p, "f"))
	})

	t.Run("absent key returns zero", func(t *testing.T) {
		require.Equal(t, float64(0), pFloat(map[string]*qdrant.Value{}, "missing"))
	})

	t.Run("wrong kind returns zero", func(t *testing.T) {
		p := map[string]*qdrant.Value{"f": {Kind: &qdrant.Value_BoolValue{BoolValue: true}}}
		require.Equal(t, float64(0), pFloat(p, "f"))
	})
}
