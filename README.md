# poc-rag

A Go proof-of-concept for evaluating retrieval quality over plain-text documents (multilingual: English, zh-Hant/Cantonese, zh-Hans, code-mixed). It tests whether a Qdrant vector index + embedder + reranker retrieves well, and lets you compare **index variants** (embedder × chunking × lexical params) side by side.

Qdrant is the **sole** store — there is no SQL database.

## Requirements

- Go 1.x (see [go.mod](go.mod))
- Docker (for Qdrant and the self-hosted reranker; the compose file pins `qdrant/qdrant:v1.19.0` — the image must stay ≥ 1.10 for the sparse IDF modifier)

## Quick start

```bash
make up               # start Qdrant (REST/dashboard :6333, gRPC :6334) + reranker (:8081), waits for healthy
                      # first run downloads the ~600MB reranker model — one-time, then cached in a volume
make build            # → bin/poc-rag
cp .apprc.example.yaml .apprc.yaml   # then edit (see Configuration)
./bin/poc-rag migrate --config .apprc.yaml
./bin/poc-rag ingest --file testdata/corpus.jsonl --config .apprc.yaml
./bin/poc-rag server --config .apprc.yaml          # or: make run
```

Then search:

```bash
curl -s localhost:8080/v1/search -d '{"query":"mushroom risotto","variant":"openai_3l_hdr"}'
```

### Offline / no API keys

Use the `fake` providers (deterministic hashed vectors, no network). In `.apprc.yaml`, add a fake embedder profile and point a variant at it, and set `reranker.provider: fake` — see the commented stanza in [.apprc.example.yaml](.apprc.example.yaml). Note the fake reranker deliberately **reverses** the order it is handed so a rerank step is visible in tests; never read its output as a relevance signal.

## Configuration

One YAML file, passed with `--config` (default `.apprc.yaml`). [.apprc.example.yaml](.apprc.example.yaml) is the annotated reference. Highlights:

- **`embedders:`** — named profiles (`openai_compatible` | `cohere` | `fake`). API keys are referenced as `${ENV_VAR}` and expanded from the environment — never put secrets in the file. Profile keys must stay lowercase (viper lowercases map keys).
- **`reranker:`** — one global reranker (`cohere_style` | `tei` | `fake`). `top_n` is the default result count; `max_candidates` caps how many parents are sent to the reranker per query (cost ceiling).

  A self-hosted default ships in the compose file: `make up` starts **bge-reranker-v2-m3** (multilingual, Apache-2.0) served by llama.cpp on `:8081`, whose `/v1/rerank` speaks the `cohere_style` wire format (`base_url: http://localhost:8081/v1`, empty `api_key`). Two caveats: scores are raw cross-encoder logits (can be negative), not the 0..1 range API rerankers return — compare ordering, never absolute values across providers; and inference is CPU-only under Docker (~700 tok/s on a 10-core VM — a worst-case batch of 20 × 1200-token parents measured ~48s), so `reranker.timeout` and `server.write_timeout` must budget for it; shrink `max_candidates` (or `parent_tokens`) to trade rerank depth for latency. Each query+parent pair must also fit the compose file's `--ubatch-size` (2048) or the request fails with a 500.
- **`retrieval:`** — default `mode` (`dense` | `lexical` | `hybrid`), channel depths (`ann_top_k`, `lexical_top_k`), `hnsw_ef`, and RRF fusion params (`rrf_k`, `weights`).
- **`variants:`** — the experimental unit. Each variant binds an embedder profile to chunking + lexical params and gets its own `chunks_<name>` Qdrant collection. `header_strategy: title_context` prepends `[<title>]` to each child's embedded text (`none` embeds the bare text). Anything under `params:` shapes stored bytes; changing it requires a re-ingest, never silent reuse.
- Durations must be strings (`"10s"`, not `10`). Misconfiguration fails at startup, not on the first request.

### Config drift

Startup (`migrate` or `server`) verifies every declared variant against the registry stored in Qdrant (including the *resolved* embedder model, so editing a profile behind an unchanged variant is caught) and against the physical collection shape. On mismatch it **refuses to start** with a config-drift error. Recover with:

```bash
./bin/poc-rag variant drop <name> --config .apprc.yaml
```

then re-ingest.

## CLI

```
poc-rag <command> --config .apprc.yaml
```

| Command | What it does |
|---|---|
| `migrate` | Creates missing collections (`documents`, `variants`, `chunks_<name>`), registers variants, runs the drift check. Idempotent. |
| `server` | Runs the same setup, then serves the HTTP API on `server.http_port`. |
| `ingest --file <corpus.jsonl>` | Bulk-loads a JSONL corpus (one document per line, same shape as the `POST /v1/documents` body) and indexes into all variants. |
| `ingest` (no `--file`) | Re-indexes every stored document — use after adding a variant or after `variant drop`. |
| `ingest --variant <name>` | Restricts indexing to one variant. |
| `variant drop <name>` | Removes a variant's chunk collection **and** its registry entry. Deliberately bypasses the drift check and takes the name raw, so it works for variants the current config no longer declares. |

`ingest` skips-and-counts line-level failures and exits non-zero if any document was skipped; stream-level failures (unreadable file, line over 32 MB) abort the run. Ctrl+C aborts promptly.

A synthetic multilingual corpus of 100 food recipes (EN + zh-Hant + zh-Hans, code-mixed) ships in [testdata/corpus.jsonl](testdata/corpus.jsonl) for smoke tests and demos.

## HTTP API

Base URL: `http://localhost:<server.http_port>`. No authentication (POC). All bodies are JSON; request bodies are capped at **64 MiB** (413 beyond that). Unknown JSON fields are **rejected with a 400** rather than silently ignored — a typo'd field must not silently produce a wrong-config comparison.

### Errors

Every error is the same envelope:

```json
{"error": {"code": "not_found", "message": "document 7c9e…: not found"}}
```

| Status | Code | Meaning |
|---|---|---|
| 400 | `invalid_argument` | Malformed JSON, validation failure, bad UUID, empty content |
| 400 | `unknown_variant` | Variant not in the registry |
| 404 | `not_found` | Document does not exist |
| 413 | `request_too_large` | Body over the 64 MiB cap |
| 499 | `client_closed` | Caller disconnected mid-request |
| 504 | `timeout` | Upstream (embedder/reranker/Qdrant) deadline exceeded |
| 500 | `internal` | Anything else — detail goes to the server log only |

### `GET /healthz`

Liveness only (`{"status":"ok"}`). No Qdrant ping — Qdrant reachability fails startup instead.

### `POST /v1/documents[?variant=<name>]`

Create or replace a document (upsert by `id`). Returns **201** when the id is new, **200** when it replaced an existing document.

```json
{
  "id": "7c9e6679-7425-40de-944b-e07fc1f90ae1",
  "title": "Mushroom Risotto",
  "content": "Stir in the mushrooms after the rice has absorbed most of the stock…"
}
```

- `id` — optional; server generates a UUID if absent, rejects non-UUIDs.
- `content` — required, plain text. Paragraphs are separated by blank lines.
- `title` — optional; also feeds the `title_context` header strategy at index time (a title-less document gets no header rather than an empty `[]`).
- `?variant=<name>` — additionally chunk + embed + index the document into that variant inline (the request fails if indexing fails). Without it, the document is only stored; variants pick it up on the next `ingest`.

Replace semantics (§6.1): the document's chunks are purged in **every** variant before the new body is saved — chunks may be missing (they heal on re-ingest), never stale. `created_at` is preserved across replaces.

Response: the stored document, with `created_at` / `updated_at` added.

### `GET /v1/documents/:id`

Returns the stored document. `:id` must be a UUID.

### `DELETE /v1/documents/:id`

Purges the document's chunks in every variant, then the document. Returns **204**.

### `POST /v1/search`

```json
{
  "query": "番茄 recipe 點煮",
  "variant": "openai_3l_hdr",
  "mode": "hybrid",
  "top_k": 10,
  "ann_top_k": 50,
  "lexical_top_k": 50,
  "rerank": true
}
```

- `query`, `variant` — required. The query is embedded with the **variant's own** embedder profile, which is what makes cross-embedder comparison valid.
- `mode` — `dense` | `lexical` | `hybrid`; defaults to `retrieval.mode`.
- `top_k` — results returned; defaults to `reranker.top_n`.
- `ann_top_k` / `lexical_top_k` — per-channel candidate depth; default from `retrieval:`; must each be ≥ `top_k`.
- `rerank` — omitted or `true` runs the reranker; `false` skips it and returns fused order. A reranker failure fails the request (no silent fallback).

Pipeline: the enabled channels run concurrently → weighted RRF fusion at *child* level → group by parent chunk keeping the winning child → rerank at most `reranker.max_candidates` parents on `(query, parent_text)`. Parents beyond the rerank cap keep fused order below the reranked ones and carry no `rerank_score` key at all (a genuine `0.0` still serializes).

Response:

```json
{
  "results": [
    {
      "rank": 1,
      "parent_chunk_id": "…", "document_id": "…",
      "text": "…full parent chunk — what you'd feed an LLM…",
      "child_chunk_id": "…", "child_text": "…the small chunk that actually matched…",
      "title": "Mushroom Risotto",
      "channels": ["dense", "lexical"],
      "dense_score": 0.83, "dense_rank": 1,
      "lexical_score": 7.1, "lexical_rank": 3,
      "rrf_score": 0.0325,
      "rerank_score": 0.97
    }
  ],
  "stats": {
    "mode": "hybrid",
    "dense_candidates": 50, "lexical_candidates": 42,
    "fused_candidates": 71, "parents": 18,
    "reranked": true,
    "timings_ms": {"embed_query": 41, "dense": 6, "sparse": 4, "fuse": 0, "rerank": 180, "total": 232}
  }
}
```

- `channels` lists which channels returned the winning child; the per-channel `*_score`/`*_rank` keys appear only for those channels.
- In `timings_ms`, an **absent key means that stage did not run** for the request (e.g. no `rerank` when `"rerank": false`, no `embed_query` in pure lexical mode); `fuse` and `total` are present on every successful response.

## How retrieval works (short version)

- **Small-to-big chunking**: documents split into paragraphs, then sentences (a decimal point flanked by digits is not a sentence boundary), packed into large *parent* chunks and small overlapping *child* chunks. Children are what gets embedded and matched; the parent is what gets reranked and returned.
- **Dense channel**: 1024-dim cosine ANN over the child embeddings (named vector `dense`).
- **Lexical channel**: approximate BM25 split across client and server — the client tokenizes (CJK bigrams + lowercased Latin words; Chinese numerals indexed both ways, 一點三 ↔ 1.3), FNV-1a-hashes terms, and encodes saturated TF into a sparse vector; Qdrant applies IDF at query time. Every text transformation applies **identically at index and query time** — this symmetry invariant is load-bearing.
- **Fusion**: weighted reciprocal-rank fusion of the channels, client-side.
- **Consistency without transactions** (§6.1): write ordering guarantees chunks may be *missing* but never *stale*; all point writes are idempotent and use `wait=true`.

For conceptual background on the retrieval pipeline, see [docs/notes/rag-understanding.md](docs/notes/rag-understanding.md).

## Development

```bash
make test               # unit tests — no Qdrant needed
make test-integration   # runs `make up` first, then: go test -tags integration -p 1 ./...
```

- **`-p 1` is mandatory for integration tests** — the two integration packages share one Qdrant and several tests **DROP** the shared `documents`/`variants` collections. Never point them at a Qdrant holding real data.
- Single unit test: `go test ./internal/domain/lexical/ -run TestConvertNumeral -v`
- Single integration test: `go test -tags integration ./internal/infrastructure/qdrant/ -run TestSetup -v`
- Race check for the concurrency-sensitive package: `go test -race ./internal/application/search/`
- Hygiene: `gofmt -l .` empty, `go vet ./...` and `go vet -tags integration ./...` clean.

Layout: DDD layering with ports in the domain — `internal/domain/*` (entities + interfaces) ← `internal/application/*` (services, handlers, error mapping) ← `internal/infrastructure/*` (qdrant, embedder/reranker HTTP adapters, httpserver), wired with uber-fx in `internal/app/`. See [CLAUDE.md](CLAUDE.md) for contributor-facing invariants and gotchas.
