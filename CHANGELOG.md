# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- ChromaDB backend option alongside FAISS
- LangChain / LlamaIndex adapter layer
- GraphRAG: entity extraction + knowledge graph retrieval
- Multimodal answer generation (images in response)
- RAGAS automated evaluation integration
- Kubernetes Helm chart

---

## [0.3.0] — 2024-Q3

### Added
- **SSE Streaming** (`api/streaming.py`): Real-time token-by-token delivery via Server-Sent Events, with per-stage pipeline events (query_understanding → retrieval → reranking → token → done)
- **Chain-of-Thought generation** (`llm/chain_of_thought.py`): 4-step structured reasoning (UNDERSTAND → EVIDENCE → GAPS → REASONING → FINAL ANSWER) plus optional self-correction pass
- **Query Router** (`retrieval/router.py`): Routes visual/analytical/comparative/factual queries to the optimal retrieval strategy automatically
- **Advanced chunking strategies** (`ingestion/chunking_strategies.py`):
  - `SentenceAwareChunker` — respects sentence boundaries, configurable overlap
  - `MarkdownChunker` — splits on heading levels
  - `SemanticChunker` — embedding similarity-based splits
- **Async pipeline** (`utils/async_pipeline.py`): Concurrent BM25 + vector retrieval using `asyncio` + `ThreadPoolExecutor`
- **Lightweight tracing** (`utils/tracing.py`): Span-based observability compatible with OpenTelemetry
- **MMR diversity ranker** (`fusion/diversity_ranker.py`): Maximal Marginal Relevance ranking to prevent near-duplicate chunks filling context
- **Sparse+Dense unified index** (`retrieval/sparse_dense_index.py`): Single interface for both BM25 and FAISS with incremental updates and save/load
- **NLI hallucination detector** (`validation/hallucination_detector.py`): Sentence-level entailment checking with transformer NLI model, lexical fallback
- **Multi-turn conversation memory** (`memory/conversation_memory.py`): Tracks dialogue history, injects prior context into follow-up queries
- **Smart document parser** (`ingestion/document_parser.py`): Extracts structured text, table, heading, and image blocks from PDF/DOCX/CSV/HTML
- **Admin API** (`api/admin.py`): Token-protected endpoints for index rebuild/purge, trace inspection, and system stats
- **Evaluation runner script** (`scripts/run_eval.py`): Formatted eval reports with `--no-llm` mode for retrieval-only testing
- **Architecture deep-dive** (`ARCHITECTURE.md`): Full diagram + per-component explanation

### Tests added
- `test_chunking.py` — all chunking strategies
- `test_routing.py` — query routing decisions
- `test_tracing.py` — span recording and error handling
- `test_mmr.py` — MMR diversity ranking
- `test_conversation_memory.py` — multi-turn memory
- `test_sparse_dense_index.py` — unified index CRUD + persistence

---

## [0.2.0] — 2024-Q3

### Added
- Docker + docker-compose deployment
- GitHub Actions CI/CD pipeline (test → lint → build)
- API request/response caching layer
- FastAPI middleware (rate limiting, request logging, CORS)
- Table embedding module (`embeddings/table_embedder.py`)
- Metadata filtering for retrieval (`retrieval/metadata_filter.py`)
- Structured prompt templates (`llm/prompt_templates.py`)
- Benchmark script (`scripts/benchmark.py`)
- `pyproject.toml` for modern Python packaging

---

## [0.1.0] — 2024-Q3 (Initial Release)

### Added
- Core RAG pipeline orchestrator (`utils/pipeline.py`)
- BM25 keyword retrieval (`retrieval/bm25.py`)
- FAISS vector retrieval (`retrieval/vector.py`)
- Hybrid RRF fusion (`retrieval/hybrid.py`)
- Text embedding via Sentence-Transformers (`embeddings/text_embedder.py`)
- CLIP image embedding (`embeddings/image_embedder.py`)
- Multi-modal reranking — cross-encoder + CLIP (`ranking/reranker.py`)
- Context fusion with token budgeting (`fusion/context_fusion.py`)
- LLM generation — Anthropic + OpenAI backends (`llm/generator.py`)
- Query understanding + expansion (`llm/query_understanding.py`)
- Answer validation with retry logic (`validation/answer_validator.py`)
- In-memory + Redis feedback memory (`memory/memory_store.py`)
- FastAPI REST interface (`api/app.py`)
- Evaluation framework — Precision@K, Recall@K, MRR, faithfulness (`utils/evaluate.py`)
- Multi-format document ingestion (`ingestion/ingest.py`)
- Recursive text chunker (`utils/chunker.py`)
- Full `requirements.txt`, `.gitignore`, config example
