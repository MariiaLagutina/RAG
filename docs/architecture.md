# Architecture Reference

This document is the detailed map behind the compact diagram in the
[README](../README.md#system-architecture). It separates the mandatory RAG
pipeline from optional engineering bonuses so that each flow can be explained
without reading one oversized diagram.

## Architectural boundaries

The project follows four practical boundaries:

1. **Public boundaries** parse user input and convert expected failures into
   stable CLI or HTTP responses.
2. **Workflows** coordinate already-defined domain operations without owning
   tokenization, ranking, persistence, or generation algorithms.
3. **Domain components** implement one responsibility such as chunking, BM25
   scoring, context construction, or prompt validation.
4. **Persistence adapters** validate stored JSON and bind it to the exact
   corpus and pipeline that produced it.

The public CLI is intentionally thin even though all commands are collected in
one module. Domain logic remains independently testable below that boundary.

| Boundary | Main files | Responsibility |
| --- | --- | --- |
| CLI | [`src/__main__.py`](../src/__main__.py), [`src/cli.py`](../src/cli.py) | Python Fire command registry, argument defaults, progress, concise errors |
| HTTP bonus | [`src/api/app.py`](../src/api/app.py), [`src/api/state.py`](../src/api/state.py) | Long-running process with one loaded index and lazy model loading |
| Assignment schemas | [`src/models.py`](../src/models.py) | Pydantic input/output contracts required by the subject |
| Ingestion | [`src/ingestion/`](../src/ingestion/) | Safe discovery, exact reads, source-aware chunking, invariant audits |
| Retrieval | [`src/retrieval/`](../src/retrieval/) | Tokenization, BM25, reranking, persisted results, caches, optional semantic/hybrid paths |
| Generation | [`src/generation/`](../src/generation/) | Context budgeting, prompt construction, local Qwen execution, validation, answer caching |
| Evaluation | [`src/evaluation/`](../src/evaluation/) | Ground-truth joins, IoU, Recall@K, MRR, error analysis, answer diagnostics |

## Mandatory component flow

```mermaid
flowchart TD
    CLI["Python Fire CLI"]
    Files["Safe file discovery"]
    Reader["Exact UTF-8 documents"]
    Dispatch{"Source type"}
    Py["Python AST chunker"]
    Text["Markdown/text chunker"]
    Terms["Content + metadata terms"]
    BM25["BM25 index"]
    Store["Versioned JSON snapshot"]
    Semantic["Docs-only MiniLM index"]
    Route{"search_dataset mode"}
    Search["Query tokenization + retrieval"]
    Rerank["Auxiliary-path reranker"]
    Fusion["Weighted RRF with semantic ranks"]
    Results["Exact source spans"]
    Context["Deduplicated token budget"]
    Model["Qwen/Qwen3-0.6B"]
    Output["Pydantic output"]
    Eval["IoU, Recall@K, MRR"]

    CLI --> Files --> Reader --> Dispatch
    Dispatch -->|Python| Py --> Terms
    Dispatch -->|Markdown/text| Text --> Terms
    Terms --> BM25 --> Store
    Store -. documentation chunks .-> Semantic
    Store --> Search --> Rerank --> Route
    Route -->|Code, bm25, unknown name| Results
    Route -->|Docs, hybrid| Fusion --> Results
    Semantic -. semantic ranks .-> Fusion
    Results --> Context --> Model --> Output
    Results --> Eval
```

### Why this shape

- Discovery and reading are separate from chunking, so filesystem safety does
  not depend on a parser.
- Python and text use different chunkers because their structural boundaries
  are different, but both return the same immutable `Chunk` contract.
- Exact source text and coordinates remain evidence; synthetic structural
  terms are stored separately for ranking.
- Retrieval produces assignment-compatible locations before generation begins.
  The evaluator therefore measures retrieval independently of Qwen quality.
- Docs datasets add semantic ranks on top of BM25; Code datasets, unknown
  dataset names, and single-query commands use BM25 only. BM25 is therefore
  part of every retrieval path.
- Generation receives a bounded context rather than raw top-k text, keeping
  token budgeting and source traceability deterministic.

## Indexing sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Discovery
    participant Chunker
    participant BM25
    participant Store
    participant MiniLM

    User->>CLI: index --max_chunk_size 2000
    CLI->>Discovery: discover supported corpus files
    Discovery-->>CLI: stable project-relative manifest
    loop Each file
        CLI->>Chunker: exact document + configured limit
        Chunker-->>CLI: exact half-open chunks
    end
    CLI->>BM25: build lexical documents and postings
    CLI->>Store: save corpus/pipeline-bound snapshot
    CLI->>MiniLM: load pinned model (download once, then cache)
    MiniLM-->>CLI: embeddings of documentation chunks only
    CLI->>Store: save semantic index with the same fingerprints
    Store-->>User: counts, fingerprints, semantic status
```

The semantic step runs only after the BM25 snapshot is saved. If the model
cannot be loaded or encoding fails, `index` prints a warning, reports
`semantic_index: unavailable`, and still succeeds with the complete BM25
index. The semantic store writes vectors and metadata to temporary files and
replaces each file atomically, vectors first; the metadata records the vectors'
checksum, so an interrupted write is detected and rejected on load rather than
used.

Incremental indexing changes only the loop: unchanged files may reuse stored
lexical documents after file, pipeline, schema, and snapshot-integrity checks.
The final index must remain byte-for-byte equivalent to a full rebuild of the
same corpus state. The documentation semantic index is re-encoded in full on
every run.

## Search and evaluation sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Corpus
    participant Store
    participant Retriever
    participant Evaluator

    User->>CLI: search_dataset(dataset, k, output, retrieval_mode=auto)
    CLI->>CLI: resolve mode from the dataset file name
    CLI->>Corpus: calculate current fingerprint
    CLI->>Store: load compatible index
    Store-->>CLI: validated BM25 index
    alt Docs dataset or hybrid mode
        CLI->>Store: load semantic index with matching fingerprints
        Store-->>CLI: validated semantic index, or BM25 fallback
        CLI->>Retriever: BM25 top 20 + semantic top 20, weighted RRF
    else Code dataset, bm25 mode, or unknown name
        CLI->>Retriever: BM25 with auxiliary-path reranking
    end
    Retriever-->>CLI: ranked exact source spans per question
    CLI-->>User: StudentSearchResults JSON
    User->>Evaluator: results + answered dataset
    Evaluator-->>User: Recall@1/3/5/10 and MRR
```

Evaluation joins by `question_id`, follows ground-truth order, and rejects
missing, unrelated, or duplicate identifiers and question-text mismatches.
This prevents a reordered result file from changing the meaning of a score.

## Answer-generation sequence

```mermaid
sequenceDiagram
    participant User
    participant Retrieval
    participant Context
    participant Backend
    participant Validator

    User->>Retrieval: answer(question, k)
    Retrieval-->>Context: ranked source locations
    Context->>Context: read, deduplicate, fit token budget
    Context->>Backend: source-labelled prompt
    Backend-->>Validator: generated text
    Validator-->>User: grounded answer or controlled error
```

The CLI `answer`, `answer_dataset`, and the API `/answer` apply the same
deterministic citation policy. A missing `[Source N]` label does not reject an
otherwise grounded answer, because the small model often omits labels. A label
that refers to a source outside the prompt is rejected as a controlled error,
and the exact insufficient-context reply is always accepted. The batch path
preserves the required output schema and records model limitations rather than
treating perfect phrasing as a mandatory pass condition.

## Persistence and compatibility

```mermaid
flowchart LR
    Corpus["Corpus bytes + paths"] --> CorpusFP["Corpus fingerprint"]
    Config["Chunking + BM25 + schema"] --> PipelineFP["Pipeline fingerprint"]
    Docs["Stored lexical documents"] --> Snapshot["BM25 snapshot"]
    CorpusFP --> Snapshot
    PipelineFP --> Snapshot
    Snapshot --> Checksum["Snapshot checksum"]
    Snapshot --> Load{"All identities match?"}
    Checksum --> Load
    Load -->|Yes| Runtime["Runtime BM25 index"]
    Load -->|No| Error["Controlled reindex error / full rebuild"]
```

These identities answer different questions:

- the **corpus fingerprint** proves which paths and bytes were indexed;
- the **pipeline fingerprint** proves which chunking and ranking configuration
  produced the index;
- the **schema version** proves that stored fields have known semantics;
- the **snapshot checksum** detects accidental damage to stored evidence.

The semantic index records the BM25 corpus and pipeline fingerprints together
with the model name and pinned revision, plus a checksum of its embedding
matrix. A hybrid search rejects it when any of these differ from the current
BM25 index and model configuration, and then falls back to BM25.

## Optional bonuses

```mermaid
flowchart TD
    Mandatory["Mandatory BM25 pipeline"]
    Semantic["Pinned CPU MiniLM index"]
    Hybrid["Reciprocal-rank fusion for Docs"]
    Incremental["Incremental BM25 rebuild"]
    SearchCache["Compatible query-result cache"]
    AnswerCache["Validated answer cache"]
    API["Local FastAPI server"]

    Mandatory -. exact chunks .-> Semantic
    Mandatory -. lexical ranks .-> Hybrid
    Semantic -. semantic ranks .-> Hybrid
    Mandatory -. compatible snapshot .-> Incremental
    Mandatory -. repeated query .-> SearchCache
    Mandatory -. validated answer .-> AnswerCache
    Mandatory -. reusable workflows .-> API
```

The bonuses keep the mandatory BM25 route in place:

- semantic-only search remains optional because it scored below BM25;
- hybrid retrieval is selected for Docs datasets by `search_dataset`'s default
  `auto` mode, documented and overridable with `--retrieval_mode bm25`; Code
  always stays on BM25;
- `index` builds the documentation-only semantic index, while
  `index_semantic` still encodes all chunks into the same default directory
  (see the [README](../README.md#optional-bonuses));
- caches use complete compatibility keys and can be omitted;
- the HTTP API reuses domain workflows while the required CLI remains intact;
- all bonus behavior is expected to run on CPU-only campus machines.

## Dependency direction

Dependencies point inward toward domain contracts:

```text
CLI / HTTP
    -> workflows
        -> ingestion, retrieval, generation, evaluation components
            -> immutable dataclasses and Pydantic boundary models
```

Domain modules do not import the CLI or HTTP server. Progress callbacks are
passed into workflows, which prevents `tqdm` from becoming a retrieval or
generation dependency. The HTTP API loads shared runtime state in
[`src/api/state.py`](../src/api/state.py) rather than rebuilding resources per
request.

## Intentional duplication versus refactoring

The audit found a few similar helpers and orchestration patterns, but no safe
production refactor that improves the defense version enough to justify its
risk:

- CLI and HTTP both resolve paths below a configured project root, but they are
  independent public boundaries with different error contracts.
- lexical, semantic, and hybrid workflows share a broad shape, but have
  different persisted resources, timing reports, and compatibility checks.
- generation batch and diagnostic commands both load one backend and expose
  progress, but return different domain results.

`src/cli.py` is large because it collects the stable public command surface.
Algorithms and stateful behavior are already split into component packages.
Moving command functions immediately before evaluation would mostly move code
and invalidate patch-based boundary tests without reducing domain complexity.

## Where to start reading

Use this order when reviewing the implementation:

1. [`src/models.py`](../src/models.py) — assignment-facing contracts.
2. [`src/__main__.py`](../src/__main__.py) — complete public command list.
3. [`src/retrieval/index_store/incremental.py`](../src/retrieval/index_store/incremental.py) — indexing orchestration.
4. [`src/ingestion/chunking/orchestrator.py`](../src/ingestion/chunking/orchestrator.py) — format dispatch.
5. [`src/retrieval/bm25/index.py`](../src/retrieval/bm25/index.py) — ranking math and postings.
6. [`src/retrieval/workflow.py`](../src/retrieval/workflow.py) — persisted retrieval flow.
7. [`src/generation/context/workflow.py`](../src/generation/context/workflow.py) — context boundary.
8. [`src/generation/query/workflow.py`](../src/generation/query/workflow.py) — end-to-end answer composition.
9. [`src/evaluation/retrieval/workflow.py`](../src/evaluation/retrieval/workflow.py) — safe evaluation join.
10. [`src/api/app.py`](../src/api/app.py) — optional non-CLI adapter.

For the reasoning behind changes, use the
[decision log](decision-log.md).
