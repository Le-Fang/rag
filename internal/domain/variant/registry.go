package variant

import (
	"fmt"
	"sort"
)

// Registry is the in-memory variant set loaded at startup. After the drift
// check passes, config equals registry by construction (§9.1.1), so the
// search hot path reads only this — never the store.
type Registry struct {
	byName map[string]*IndexVariant
}

func NewRegistry(vs []*IndexVariant) *Registry {
	m := make(map[string]*IndexVariant, len(vs))
	for _, v := range vs {
		m[v.Name] = v
	}
	return &Registry{byName: m}
}

// Get returns the shared *IndexVariant for name. The returned value is
// immutable, registry-owned state — callers must copy it before mutating.
func (r *Registry) Get(name string) (*IndexVariant, error) {
	v, ok := r.byName[name]
	if !ok {
		return nil, fmt.Errorf("variant %q: %w", name, ErrUnknown)
	}
	return v, nil
}

// List returns every registered variant, sorted by name. The returned
// *IndexVariant values are shared, immutable state — callers must copy
// before mutating.
func (r *Registry) List() []*IndexVariant {
	out := make([]*IndexVariant, 0, len(r.byName))
	for _, v := range r.byName {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
