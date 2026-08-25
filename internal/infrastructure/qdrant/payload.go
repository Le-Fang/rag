package qdrant

import (
	"time"

	"github.com/qdrant/go-client/qdrant"
)

// Read-side payload helpers: tolerant getters over qdrant.Value maps.

func pStr(p map[string]*qdrant.Value, key string) string {
	if v, ok := p[key]; ok {
		return v.GetStringValue()
	}
	return ""
}

func pInt(p map[string]*qdrant.Value, key string) int {
	if v, ok := p[key]; ok {
		return int(v.GetIntegerValue())
	}
	return 0
}

func pFloat(p map[string]*qdrant.Value, key string) float64 {
	if v, ok := p[key]; ok {
		switch v.Kind.(type) {
		case *qdrant.Value_DoubleValue:
			return v.GetDoubleValue()
		case *qdrant.Value_IntegerValue:
			return float64(v.GetIntegerValue())
		}
	}
	return 0
}

func pBool(p map[string]*qdrant.Value, key string) bool {
	if v, ok := p[key]; ok {
		return v.GetBoolValue()
	}
	return false
}

func pTimePtr(p map[string]*qdrant.Value, key string) *time.Time {
	s := pStr(p, key)
	if s == "" {
		return nil
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return nil
	}
	return &t
}
