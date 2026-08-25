package app

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"go.uber.org/fx"

	"poc-rag/internal/application/documents"
	"poc-rag/internal/application/ingestion"
	"poc-rag/internal/application/search"
	"poc-rag/internal/config"
	"poc-rag/internal/domain/chunk"
	"poc-rag/internal/domain/document"
	"poc-rag/internal/domain/embedding"
	"poc-rag/internal/domain/reranking"
	"poc-rag/internal/domain/variant"
	cohereembed "poc-rag/internal/infrastructure/embedder/cohere"
	fakeembed "poc-rag/internal/infrastructure/embedder/fake"
	"poc-rag/internal/infrastructure/embedder/openaicompat"
	infraqdrant "poc-rag/internal/infrastructure/qdrant"
	fakererank "poc-rag/internal/infrastructure/reranker/fake"
	"poc-rag/internal/infrastructure/reranker/rerankapi"
)

// modules assembles the full dependency graph for every command. Setup (and
// with it the drift check, §9.1.1) runs inside the Registry constructor, so
// any command that touches variants refuses to start on drift.
func modules(cfgPath string) fx.Option {
	return fx.Options(
		fx.Provide(
			func() (*config.Config, error) { return config.Load(cfgPath) },
			newLogger,
			newQdrantClient,
			newRegistry,
			newEmbedders,
			newReranker,
			func() chunk.Chunker { return chunk.NewSmallToBig() },
			fx.Annotate(infraqdrant.NewDocumentRepository, fx.As(new(document.DocumentRepository))),
			fx.Annotate(infraqdrant.NewChunkRepository, fx.As(new(chunk.ChunkRepository))),
			newIngestionService,
			newSearchService,
			newDocumentsService,
			documents.NewHandler,
			search.NewHandler,
		),
		fx.NopLogger,
	)
}

func newLogger() *slog.Logger {
	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	// apperror.Respond is a package-level func and logs through the slog
	// default; aligning the default with the injected logger keeps §10 fault
	// logging on the same handler as everything else.
	slog.SetDefault(log)
	return log
}

func newQdrantClient(cfg *config.Config, lc fx.Lifecycle) (*infraqdrant.Client, error) {
	c, err := infraqdrant.NewClient(infraqdrant.Options{
		Host: cfg.Qdrant.Host, GRPCPort: cfg.Qdrant.GRPCPort,
		APIKey: cfg.Qdrant.APIKey, UseTLS: cfg.Qdrant.UseTLS,
	})
	if err != nil {
		return nil, err
	}
	lc.Append(fx.Hook{OnStop: func(context.Context) error { return c.Close() }})
	return c, nil
}

// newRegistry resolves declared variants, runs setup + drift check, and
// returns the in-memory variant set the hot path reads (§4.2, §9.1.1).
func newRegistry(cfg *config.Config, c *infraqdrant.Client, log *slog.Logger) (*variant.Registry, error) {
	vs, err := cfg.DomainVariants()
	if err != nil {
		return nil, err
	}
	setup := infraqdrant.NewSetup(c, infraqdrant.NewVariantRepository(c), vs)
	if err := setup.Run(context.Background()); err != nil {
		return nil, fmt.Errorf("qdrant setup: %w", err)
	}
	log.Info("variants registered", "count", len(vs))
	return variant.NewRegistry(vs), nil
}

// newEmbedders builds one adapter per profile referenced by a declared
// variant (§9.1). Unreferenced profiles cost nothing.
func newEmbedders(cfg *config.Config) (map[string]embedding.EmbedderPort, error) {
	referenced := map[string]bool{}
	for _, v := range cfg.Variants {
		referenced[v.Embedder] = true
	}
	out := map[string]embedding.EmbedderPort{}
	for name := range referenced {
		p := cfg.Embedders[name] // existence validated at config load
		switch p.Provider {
		case "openai_compatible":
			out[name] = openaicompat.New(openaicompat.Options{
				BaseURL: p.BaseURL, Model: p.Model, APIKey: p.APIKey,
				Dimensions: p.Dimensions, BatchSize: p.BatchSize, Timeout: p.Timeout,
			})
		case "cohere":
			out[name] = cohereembed.New(cohereembed.Options{
				BaseURL: p.BaseURL, Model: p.Model, APIKey: p.APIKey,
				Dimensions: p.Dimensions, BatchSize: p.BatchSize, Timeout: p.Timeout,
			})
		case "fake":
			out[name] = fakeembed.New(p.Dimensions)
		default:
			return nil, fmt.Errorf("embedder profile %s: unknown provider %q", name, p.Provider)
		}
	}
	return out, nil
}

func newReranker(cfg *config.Config) (reranking.RerankerPort, error) {
	switch cfg.Reranker.Provider {
	case "cohere_style", "tei":
		return rerankapi.New(rerankapi.Options{
			Style: cfg.Reranker.Provider, BaseURL: cfg.Reranker.BaseURL,
			Model: cfg.Reranker.Model, APIKey: cfg.Reranker.APIKey, Timeout: cfg.Reranker.Timeout,
		}), nil
	case "fake":
		return fakererank.New(), nil
	default:
		return nil, fmt.Errorf("reranker: unknown provider %q", cfg.Reranker.Provider)
	}
}

func newIngestionService(chunker chunk.Chunker, embedders map[string]embedding.EmbedderPort,
	chunks chunk.ChunkRepository, registry *variant.Registry, log *slog.Logger) *ingestion.Service {
	return ingestion.New(chunker, embedders, chunks, registry, log)
}

func newSearchService(cfg *config.Config, registry *variant.Registry,
	embedders map[string]embedding.EmbedderPort, chunks chunk.ChunkRepository,
	reranker reranking.RerankerPort, log *slog.Logger) *search.Service {
	return search.New(registry,
		[]search.RetrieverEntry{
			{Retriever: search.NewDenseRetriever(embedders, chunks), Modes: []string{"dense", "hybrid"}},
			{Retriever: search.NewSparseRetriever(chunks), Modes: []string{"lexical", "hybrid"}},
		},
		reranker,
		search.Options{
			DefaultMode: cfg.Retrieval.Mode, AnnTopK: cfg.Retrieval.AnnTopK,
			LexicalTopK: cfg.Retrieval.LexicalTopK, HnswEf: cfg.Retrieval.HnswEf,
			RRFK: cfg.Retrieval.RRFK, Weights: cfg.Retrieval.Weights,
			RerankTopN: cfg.Reranker.TopN, RerankMaxCandidates: cfg.Reranker.MaxCandidates,
		},
		log)
}

func newDocumentsService(docs document.DocumentRepository, chunks chunk.ChunkRepository,
	ingest *ingestion.Service, registry *variant.Registry, log *slog.Logger) *documents.Service {
	return documents.New(docs, chunks, ingest, registry, log)
}
