# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Go POC testing whether a Qdrant vector index + embedder + reranker retrieves well over plain-text documents (multilingual: English + zh-Hant/Cantonese + zh-Hans, code-mixed). Qdrant is the **sole** store — there is no SQL database.

## Commands

```bash
make build              # → bin/poc-rag
make test               # unit tests (no Qdrant needed)
make up                 # docker compose up -d --wait (Qdrant v1.19, healthcheck-gated)
make test-integration   # runs `up` first, then: go test -tags integration -p 1 ./...
```

- Single test: `go test ./internal/domain/lexical/ -run TestConvertNumeral -v`
- Single integration test: `go test -tags integration ./internal/infrastructure/qdrant/ -run TestSetup -v` (Qdrant must be up)
- **Integration tests require `-p 1`** — the two integration packages (`qdrant`, `app`) share one Qdrant and several tests DROP the shared `documents`/`variants` collections. Never point them at a Qdrant holding real data.
- Concurrency-sensitive packages should also pass `go test -race ./internal/application/search/`.
- CLI: `./bin/poc-rag migrate|server|ingest|variant drop <name> --config .apprc.yaml`. For offline/local runs use `provider: fake` embedder/reranker profiles (see the commented stanza in `.apprc.example.yaml`) — no API keys needed. `ingest --file testdata/corpus.jsonl` bulk-loads; `ingest` without `--file` re-indexes from stored documents; non-zero exit means lines were skipped. `variant drop <name>` removes a variant's chunk collection AND registry entry; it deliberately bypasses Setup/the drift check and takes the name raw, so it works for variants the current config no longer declares.
- Repo hygiene expected by review: `gofmt -l .` empty, `go vet ./...` and `go vet -tags integration ./...` clean.

## Architecture

DDD layering with ports in the domain: `internal/domain/*` (entities + interfaces, no infra imports) ← `internal/application/*` (ingestion, search, documents services + gin handlers + apperror) ← `internal/infrastructure/*` (qdrant, embedder/reranker HTTP adapters, httpserver). Wiring is centralized in `internal/app/modules.go` (uber-fx) with cobra commands in `internal/app/`.

**Variants are the experimental unit.** A variant = embedder profile + chunking params + lexical params, declared in config. Each variant gets its own `chunks_<name>` Qdrant collection (named dense vector `dense`, 1024-dim cosine + sparse vector `lexical` with the server-side IDF modifier). Two payload-only collections (`documents`, `variants`) are vector-less. Startup (`migrate` or `server`) runs `infraqdrant.Setup`: it creates missing collections, verifies existing collections' vector shape and IDF modifier, and **refuses to start on config drift** (`variant.ErrConfigDrift`) — the registry stores the *resolved* embedder model so editing a profile behind an unchanged variant is caught; recover with `variant drop <name>` then re-ingest. Anything that shapes stored bytes lives in `variant.Params` (plus `header_strategy: none | title_context` — the header is part of embed_text); changing it requires re-ingest, never silent reuse.

**Approximate BM25 split across client and server.** `internal/domain/lexical` tokenizes (CJK bigrams + lowercased Latin words, decimal points kept between digits, Chinese numerals indexed in both forms — 一點三 ↔ 1.3, 二零二四 ↔ 2024), hashes terms with FNV-1a 32, and encodes documents with saturated TF; Qdrant applies IDF at query time. **The symmetry invariant is load-bearing:** every text transformation must apply identically at index and query time (both sides go through the same `Tokenize`/`ConvertNumeral`/encoder with params read from the variant). Never "fix" a tokenizer/numeral quirk on one side only — known accepted artifacts (百分點 emitting 100, 三萬五 read literally) are documented in `numerals.go`. The chunker's sentence splitter also participates: it must not split decimals (`1.3` stays whole) or the tokenizer's decimal rule never fires.

**Write ordering replaces transactions (§6.1: "chunks may be missing, never stale").** Document replace = purge chunks in *every* variant collection → save document → optionally index inline. Ingestion embeds all children fully in memory before its single `ReplaceForDocument` (delete-then-batch-upsert) write. All Qdrant writes use `wait=true`, and all *point* writes (upsert, delete-by-filter) are idempotent — that is what makes the gRPC retry interceptor (Unavailable/DeadlineExceeded, in `client.go`) safe for them. Collection DDL is **not** idempotent (see `chunk_repository.go`: `DeleteCollection` errors on an absent collection), so a retry after a lost ack can surface a spurious `AlreadyExists`/`NotFound`; `Setup` and `DeleteByVariant` guard with `CollectionExists` rather than relying on the retry.

**Search pipeline** (`internal/application/search`): dense + sparse retrievers run concurrently (a failing channel cancels its sibling; a self-inflicted `context.Canceled` never masks the real error), fuse client-side with weighted RRF at *child* level, group by parent keeping the winning child, then rerank at most `max_candidates` parents (beyond-cap parents keep fused order below reranked ones, `rerank_score` omitted — the DTO field is `json:"rerank_score,omitempty"` over a `*float64`, so an unscored parent has no key at all while a genuine 0.0 still serializes). Query embedding uses the *variant's own* embedder profile — resolving through the variant is what makes multi-embedder comparison valid.

## Qdrant client facts (discovered live, easy to get wrong)

- Payload-only upserts **must** set `Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{})` — nil `Vectors` is rejected ("Expected some vectors") even in vector-less collections.
- `codes.NotFound` from Qdrant means the **collection** is missing (an infra fault), never a missing point — a missing point is `err == nil` + empty result. Don't map `NotFound` to `document.ErrNotFound`.
- Scroll pagination goes through `Client.scrollAll` (raw points client + `NextPageOffset`); never paginate by last-seen ID. `ScrollPoints.Limit` is `*uint32`, `QueryPoints.Limit` is `*uint64`.
- Deleting an absent collection errors (not idempotent) — guard with `CollectionExists` first.
- `internal/infrastructure/qdrant/claims_integration_test.go` is the compile-checked canonical reference for client idioms (pinned: qdrant/qdrant v1.19.0, go-client v1.19.0; the image must stay ≥ 1.10 for the IDF modifier).

## Conventions

- TDD: failing test first, then implementation; hand-computed fixture values in tests are deliberate — verify arithmetic before "correcting" them.
- Integration tests: `//go:build integration` tag, clean collections at start **and** register a mirroring `t.Cleanup`; use unique collection/variant names per test file to avoid cross-test collisions.
- Errors: wrap with context (`fmt.Errorf("variant %s: %w", ...)`); domain sentinels (`document.ErrNotFound`, `document.ErrEmptyContent`, `variant.ErrUnknown`, `variant.ErrConfigDrift`, `chunk.ErrDimensionMismatch`) are matched with `errors.Is`. Client-facing messages are sanitized via `apperror.Map` (unknown errors → 500 "internal error"); 5xx detail goes to slog only.
- Logging: injected `*slog.Logger` with `InfoContext`/`ErrorContext`; `modules.go` sets it as the slog default so `apperror.Respond` uses the same handler.
- HTTP adapters (embedders, reranker) default timeouts to 30s when config omits them, and treat vendor-returned indices as untrusted (bounds + duplicate checks).
- Config: viper lowercases map keys — embedder profile keys must stay lowercase; misconfiguration fails at startup, not on first request (validation lives in `internal/config/validate.go` plus adapter constructors).
