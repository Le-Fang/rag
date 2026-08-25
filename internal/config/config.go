package config

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/viper"
)

type Server struct {
	HTTPPort     int           `mapstructure:"http_port"`
	ReadTimeout  time.Duration `mapstructure:"read_timeout"`
	WriteTimeout time.Duration `mapstructure:"write_timeout"`
}

type Qdrant struct {
	Host     string `mapstructure:"host"`
	GRPCPort int    `mapstructure:"grpc_port"`
	APIKey   string `mapstructure:"api_key"`
	UseTLS   bool   `mapstructure:"use_tls"`
}

type Embedder struct {
	Provider   string        `mapstructure:"provider"` // openai_compatible | cohere | fake
	BaseURL    string        `mapstructure:"base_url"`
	Model      string        `mapstructure:"model"`
	APIKey     string        `mapstructure:"api_key"`
	Dimensions int           `mapstructure:"dimensions"`
	BatchSize  int           `mapstructure:"batch_size"`
	Timeout    time.Duration `mapstructure:"timeout"`
}

type Reranker struct {
	Provider      string        `mapstructure:"provider"` // cohere_style | tei | fake
	BaseURL       string        `mapstructure:"base_url"`
	Model         string        `mapstructure:"model"`
	APIKey        string        `mapstructure:"api_key"`
	TopN          int           `mapstructure:"top_n"`
	MaxCandidates int           `mapstructure:"max_candidates"`
	Timeout       time.Duration `mapstructure:"timeout"`
}

type Retrieval struct {
	Mode        string             `mapstructure:"mode"` // dense | lexical | hybrid
	AnnTopK     int                `mapstructure:"ann_top_k"`
	LexicalTopK int                `mapstructure:"lexical_top_k"`
	HnswEf      int                `mapstructure:"hnsw_ef"`
	RRFK        int                `mapstructure:"rrf_k"`
	Weights     map[string]float64 `mapstructure:"weights"`
}

type VariantParams struct {
	ParentTokens      int     `mapstructure:"parent_tokens"`
	ChildTokens       int     `mapstructure:"child_tokens"`
	ChildOverlap      int     `mapstructure:"child_overlap"`
	LexicalTokenizer  string  `mapstructure:"lexical_tokenizer"`
	NormalizeNumerals bool    `mapstructure:"normalize_numerals"`
	BM25K1            float64 `mapstructure:"bm25_k1"`
	BM25B             float64 `mapstructure:"bm25_b"`
	BM25AvgLen        float64 `mapstructure:"bm25_avg_len"`
}

type Variant struct {
	Name           string        `mapstructure:"name"`
	Embedder       string        `mapstructure:"embedder"` // profile key
	Chunker        string        `mapstructure:"chunker"`
	HeaderStrategy string        `mapstructure:"header_strategy"` // none | title_context
	Params         VariantParams `mapstructure:"params"`
}

type Config struct {
	Server    Server              `mapstructure:"server"`
	Qdrant    Qdrant              `mapstructure:"qdrant"`
	Embedders map[string]Embedder `mapstructure:"embedders"`
	Reranker  Reranker            `mapstructure:"reranker"`
	Retrieval Retrieval           `mapstructure:"retrieval"`
	Variants  []Variant           `mapstructure:"variants"`
}

func Load(path string) (*Config, error) {
	v := viper.New()
	v.SetConfigFile(path)
	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("reading config %s: %w", path, err)
	}
	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("decoding config: %w", err)
	}
	// Viper lowercases map keys on unmarshal, so Embedders keys come out
	// lowercase while Variant.Embedder reference strings are taken verbatim
	// from YAML. Lowercase the references to match. Two profile keys that
	// differ only by case silently merge inside viper before we ever see
	// them (the second overwrites the first) — undetectable here; keep
	// profile keys lowercase by convention.
	for i := range cfg.Variants {
		cfg.Variants[i].Embedder = strings.ToLower(cfg.Variants[i].Embedder)
	}
	expandEnv(&cfg)
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("validating config: %w", err)
	}
	return &cfg, nil
}

// expandEnv resolves ${VAR} in secret-bearing fields only.
func expandEnv(cfg *Config) {
	cfg.Qdrant.APIKey = os.ExpandEnv(cfg.Qdrant.APIKey)
	cfg.Reranker.APIKey = os.ExpandEnv(cfg.Reranker.APIKey)
	for k, e := range cfg.Embedders {
		e.APIKey = os.ExpandEnv(e.APIKey)
		cfg.Embedders[k] = e
	}
}
