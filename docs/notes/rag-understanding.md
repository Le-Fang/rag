# RAG: working notes

## 1. Two pipelines, one index

RAG is two pipelines converging on an index:

- **Ingest (offline):** parse → chunk → encode (dense + sparse) → index. Runs rarely; can afford to be slow and thorough.
- **Query (online):** encode the query the same two ways → retrieve candidates per channel → fuse → rerank → hand context to the model. Latency-bound.

The shape that matters: **wide-and-cheap recall feeding narrow-and-expensive precision.** Stages are ordered by cost per item — pull 50–100 candidates per channel with cheap similarity, rerank ~20 with an expensive model, return ~10. Anything the recall stage misses is unrecoverable downstream, so tune recall generously; the precision stage exists so generous recall doesn't degrade final quality.

The *document* is the unit of ownership: metadata (title, date, source, IDs used for filtering) travels with every chunk derived from it, and updating a document means replacing all of its derived chunks — there is no such thing as patching an individual chunk in place.

## 2. Chunking

The central tension: what you *match on* and what you *read* want different sizes.

- Small chunks embed crisply and match precisely, but carry too little context to be useful on their own.
- Large chunks read well, but their embeddings average many topics into mush, and matching precision drops.

The resolution is **index small, return big** (parent/child, a.k.a. small-to-big): embed and match small child chunks, have each child point at a larger parent, return the parent. Match precision and context quality stop being a single knob. Size each for its job — parents around what the generator wants as context (on the order of a thousand tokens), children around what embeds crisply (a couple hundred). Overlap adjacent children by a fraction of their length so an idea that straddles a boundary appears whole in at least one child instead of being split across two half-matches.

Boundaries matter more than sizes. Splitting mid-sentence — or mid-number, turning "1.3" into "1" and "3" — poisons both the embedding and lexical matching. Sentence-aware splitting is cheap insurance; for structured text, paragraphs and headings are natural seams, and malformed or low-quality input is better dropped than indexed as noise.

A chunk lifted out of its document also loses its context: "bake at 220°C for 20 minutes" embeds better when the chunk carries a title such as *Crispy Roast Potatoes*. Prepending a short contextual header (title, date) to each chunk before encoding is a cheap way to make that context travel with it — at the cost of the header's tokens in every chunk, and of the header terms matching lexically in every chunk of that document.

Chunking parameters are part of the index's identity: change them and the old index isn't comparable, it's stale.

## 3. Ingest: what encoding actually does to a chunk

Both representations are derived from the *same* chunk text — they are two encodings of one thing, stored side by side under the chunk's ID.

**The dense path** hands the chunk text to an embedding model and gets back one fixed-length vector (e.g. 1024 floats). That vector is the model's compression of the chunk's *meaning* into a point in space, where nearby points ≈ similar meaning — across phrasings and, with multilingual models, across languages. Normalize it to unit length so cosine similarity becomes a plain dot product, then store it in an ANN structure (typically an HNSW graph) that can find near neighbors without comparing against every vector. Two properties follow directly from the mechanics:

- The vector is the same size no matter how long the text is — that fixed-size compression is what makes search fast, and it's also exactly *why* long chunks dilute: more topics squeezed into the same number of floats.
- The whole thing is opaque: no individual dimension means anything, so there's nothing to inspect or patch — a bad embedding can only be fixed by re-embedding.

Embedding is the expensive step (an API call per batch), so embed everything for a document first and only then write, keeping a partially-embedded document from ever being partially visible.

**The sparse path** never sees "meaning" — it reduces the chunk to *which surface terms it contains, and how much each counts*:

1. **Tokenize** into terms: lowercase whitespace-split words for Latin scripts; character bigrams for CJK (there are no spaces to split on, and bigrams sidestep needing a segmenter). Keep decimals like "1.3" as one term.
2. **Normalize** so surface variants meet: case-fold; convert spelled-out numerals to digits and index both forms, so "one point three" in any language and "1.3" can find each other.
3. **Map each term to an integer ID** — via a hash (e.g. 32-bit FNV) rather than a maintained vocabulary; collisions are rare enough to accept in exchange for never coordinating a dictionary.
4. **Weight by saturated term frequency** (the BM25 TF curve): the second and third occurrence of a term add less than the first, and longer-than-average chunks get penalized so they can't win by sheer volume.

What's stored is a sparse vector — a map of `{term_id → weight}` with only the nonzero entries. The other half of BM25, **IDF** (rare terms count more than common ones), is applied at *query* time from live collection statistics, so rarity stays current as the corpus grows without rewriting anything already stored.

The symmetry consequence falls straight out of this: at query time the query text must pass through the *identical* tokenize → normalize → hash pipeline, or query terms and stored terms land on different IDs and simply never meet. Same for dense: the query goes through the same embedding model (many models take a "this is a query" vs. "this is a document" instruction — that asymmetry is part of the model, not a violation).

## 4. Retrieval channels fail differently

- **Dense** (embeddings + ANN) finds paraphrases, cross-lingual matches, "the idea of the thing." Weak on exactly the specific stuff: identifiers, product codes, names, numbers, rare jargon — everything the embedding model compresses away as detail.
- **Lexical** (BM25-style sparse) nails exact terms with IDF weighting — IDs, jargon, numbers. Blind to synonyms and paraphrase; brittle across languages.

Because the failure modes are complementary, run both concurrently and fuse. **Reciprocal-rank fusion** scores each candidate as `Σ over channels of w / (k + rank)` — combining by *rank* rather than raw score, which sidesteps the fact that cosine similarity and BM25 scores live on incomparable scales. The constant `k` (≈60) dampens how much the top few ranks dominate; the per-channel weights `w` tilt the balance per corpus. Fuse at the child level, then group by parent keeping each parent's best child — fusing after grouping would throw away the per-child rank information.

ANN search itself trades recall for speed via a search-breadth parameter (`ef` in HNSW): higher explores more of the graph per query. It's a per-query knob, and one of the cheapest recall levers available.

Hybrid isn't free — it's two systems whose text handling must be kept consistent — but single-channel RAG quietly fails on the queries the missing channel would have caught.

## 5. Reranking

Bi-encoders (embedders) compress each text into one vector *independently* — fast and indexable, but lossy: the query never gets to look at the document. Cross-encoders read the (query, document) pair *together*, attending across both texts — far more accurate, far too slow to run over a corpus.

So: retrieve with bi-encoders, rerank the top N with a cross-encoder. Score the (query, *parent*) pair — the parent is what the generator will actually read, so it's what should be judged. Set an explicit **max-candidates cost ceiling** — rerank cost is per query, and unbounded candidate lists are how latency and bills explode. Candidates beyond the cap keep their fused order below the reranked ones, and it should stay visible in the response which results were reranked and which weren't.

The reranker is the biggest single precision lever, but it can only reorder what retrieval surfaced. Recall problems are chunking/retrieval problems; no reranker fixes them.

## 6. Lessons that only show up when you build one

- **Index/query symmetry is load-bearing.** Every text transformation applied at index time must apply identically at query time — tokenization, normalization, numeral handling, casing. An asymmetric "fix" on one side doesn't error; it silently zeroes matches. Corollary: even a genuine tokenizer *bug* can't be fixed on one side only — ship the fix as a new tokenizer version and re-index.
- **Anything that shapes stored bytes requires re-indexing.** Embedder model, dimensions, chunk sizes, tokenizer version — treat the combination as a versioned config, detect drift at startup, and refuse to serve rather than silently reuse an index built under different rules. Comparing results across mismatched configs is worse than no comparison at all.
- **Prefer missing over stale.** Without cross-store transactions, order writes so a crash leaves *gaps* (recoverable by re-ingest) rather than *stale data* (silent wrong answers): purge derived data first, then write the source, re-derive later. Idempotent writes make retries safe; anything non-idempotent (like collection DDL) needs its own existence guard.
- **Measure retrieval separately from generation.** "Does the right chunk appear in the top k?" is checkable with a small hand-built query set, long before an LLM is in the loop. If retrieval is bad, generation quality is noise.
- **Multilingual and code-mixed content breaks quiet assumptions.** Word-boundary tokenization fails on CJK (character n-grams work instead); numbers appear as words in one language and digits in another and should match both ways; full-width vs. half-width forms exist; users query in a different language than the document. Each of these is invisible until a real query misses.
- **Make fakes visible.** Deterministic fake embedders/rerankers let the whole pipeline run offline in tests — but a fake reranker should *visibly* change the order, so a silently skipped rerank step can't masquerade as a working one.
