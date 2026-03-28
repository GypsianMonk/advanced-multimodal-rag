# 🏛️ Architecture Deep-Dive

This document explains every component of the Advanced Multi-Modal RAG system in detail.

---

## Pipeline Flow

```
User Query
    │
    ▼
┌──────────────────────────────────────┐
│         Query Understanding          │
│  intent=factual|visual|analytical    │
│  modality=text|image|table           │
│  rewrite + expand (synonyms + LLM)   │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│            Query Router              │
│  routes to optimal retrieval path    │
│  sets modality_filter + k_multiplier │
└────────────────┬─────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   BM25 Search      Vector Search
   (rank_bm25)       (FAISS IP)
        │                 │
        └────────┬────────┘
                 ▼
┌──────────────────────────────────────┐
│        Hybrid Fusion (RRF)           │
│  score = bm25_w * rrf(rank_bm25)     │
│        + vec_w  * rrf(rank_vec)      │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│         Multi-Stage Reranker         │
│  Stage 1: Broad top-50 candidates    │
│  Stage 2: Cross-encoder → top 10     │
│  CLIP similarity for image docs      │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│         Context Fusion               │
│  token budget enforcement (3k)       │
│  modality diversity (max 3 images)   │
│  recency + source trust scoring      │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│       LLM Generation                 │
│  Standard: RAG prompt template       │
│  Advanced: Chain-of-Thought          │
│  Streaming: SSE token-by-token       │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│       Answer Validation              │
│  lexical grounding score             │
│  LLM self-confidence check           │
│  auto-retry if confidence < threshold│
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│       Memory + Tracing               │
│  query log → Redis / in-memory       │
│  feedback (1-5 rating)               │
│  OpenTelemetry-compatible spans      │
└──────────────────────────────────────┘
```

---

## Component Details

### 1. Query Understanding

**File:** `llm/query_understanding.py`

Classifies intent into four categories:
- `factual` — direct lookups ("What was the Q3 revenue?")
- `visual` — needs images/charts ("Show fraud trend graph")
- `analytical` — requires reasoning ("Why did churn increase?")
- `comparative` — compares entities or time periods ("Q2 vs Q3 performance")

Uses keyword matching for speed; falls back to LLM rewrite for ambiguous queries.

---

### 2. Hybrid Retrieval (BM25 + Vector)

**Files:** `retrieval/bm25.py`, `retrieval/vector.py`, `retrieval/hybrid.py`

**Why hybrid?**

| Method | Strength | Weakness |
|---|---|---|
| BM25 | Exact keywords, names, IDs | Misses paraphrases |
| Vector | Semantic meaning, synonyms | Misses exact terms |
| **Hybrid** | **Both** | Slightly more compute |

**Fusion via RRF (Reciprocal Rank Fusion):**

```
score(doc) = bm25_weight × (1 / (60 + rank_bm25))
           + vec_weight  × (1 / (60 + rank_vec))
```

Default weights: `bm25=0.4`, `vector=0.6`. Tune via `config.yaml`.

---

### 3. Reranking

**File:** `ranking/reranker.py`

Two-stage:
1. **Broad retrieval**: top 50 candidates from hybrid
2. **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, doc) pair
3. **CLIP reranking**: for image documents, computes image-text cosine similarity

Cross-encoders are 10-100x more accurate than bi-encoders for ranking, but too slow for first-stage retrieval — this two-stage design gets the best of both.

---

### 4. Context Fusion

**File:** `fusion/context_fusion.py`

Enforces:
- **Token budget**: max 3,000 tokens in context window
- **Modality diversity**: at most 3 images
- **Source trust**: optional source whitelist for score boosting
- **Recency**: newer documents get slight boost (configurable)

Outputs a formatted context string and a list of source citations.

---

### 5. Answer Validation

**File:** `validation/answer_validator.py`

Two signals combined:
1. **Lexical grounding**: Jaccard similarity between answer words and context words. Low overlap = potential hallucination.
2. **LLM self-scoring**: asks the LLM to rate confidence 0–10.

If combined confidence < `min_confidence` (default 0.6), the pipeline retries with a more restrictive prompt (up to `max_retries=2`).

---

### 6. Chain-of-Thought Generation

**File:** `llm/chain_of_thought.py`

Forces 4-step reasoning:
1. UNDERSTAND — restate the question
2. EVIDENCE — list supporting facts from context
3. GAPS — note missing information
4. REASONING — derive the answer
5. FINAL ANSWER — concise output

Optional second pass: `generate_with_self_correction()` asks the LLM to review its own answer for hallucinations.

---

### 7. Streaming

**File:** `api/streaming.py`

SSE (Server-Sent Events) endpoint at `POST /query/stream`.
Emits pipeline stage events + individual LLM tokens:

```
data: {"stage": "query_understanding", "status": "done", "intent": "factual"}
data: {"stage": "retrieval", "status": "done", "candidates": 30}
data: {"stage": "reranking", "status": "done", "final_docs": 10}
data: {"stage": "generating", "status": "running"}
data: {"stage": "token", "token": "The"}
data: {"stage": "token", "token": " fraud"}
...
data: {"stage": "done", "session_id": "abc123", "confidence": 0.87}
```

---

### 8. Observability

**File:** `utils/tracing.py`

Lightweight span-based tracing (no external dependencies required):
- Each pipeline stage records a `Span` with start/end time and attributes
- `Tracer.summary()` returns full trace breakdown
- Compatible with OpenTelemetry exporters if installed
- Recent traces accessible via `GET /traces` debug endpoint

---

## Deployment

See `docker-compose.yml` for local deployment with:
- FastAPI app container
- Redis (memory backend)
- Optional NGINX reverse proxy

For production, set:
```yaml
memory_backend: redis
min_confidence: 0.7
max_retries: 3
llm:
  provider: anthropic
  model: claude-opus-4-5
```
