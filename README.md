# 🧠 Multi-Modal RAG System

> *Production-grade Retrieval-Augmented Generation with Hybrid Search, Re-ranking, and Answer Validation*

---

## 🚀 Overview

This project implements an **intelligent multi-modal RAG pipeline** that goes far beyond basic "embed → retrieve → generate". It treats retrieval as a quality-control problem — deciding *what deserves to be retrieved* before generation ever happens.

```
User Query
   ↓
Query Understanding Layer    ← Intent classification, modality detection
   ↓
Query Expansion              ← Synonyms, LLM reformulation
   ↓
Hybrid Retrieval             ← BM25 + Vector Search (FAISS / Chroma)
   ↓
Multi-Modal Fusion           ← Text + Images + Tables unified
   ↓
Re-ranking Engine            ← Cross-encoder + CLIP similarity
   ↓
Context Scoring              ← Relevance, recency, reliability
   ↓
LLM Generation               ← Prompt-optimized context injection
   ↓
Answer Validation            ← Self-consistency, hallucination checks
   ↓
Feedback Loop / Memory       ← Personalization, continuous improvement
```

---

## 🏗️ Project Structure

```
advanced-multimodal-rag/
├── ingestion/          # Document loaders, chunking strategies
├── embeddings/         # Text (Sentence Transformers), Image (CLIP), Table
├── retrieval/
│   ├── bm25.py         # BM25 keyword search
│   ├── vector.py       # Dense vector search
│   └── hybrid.py       # Fused hybrid retrieval
├── ranking/
│   └── reranker.py     # Cross-encoder + CLIP reranking
├── fusion/             # Multi-modal context aggregation
├── llm/                # LLM generation + prompt templates
├── validation/         # Answer validation, hallucination checks
├── memory/             # Query history, feedback storage
├── api/
│   └── app.py          # FastAPI REST interface
├── config/             # YAML configs, model registry
├── utils/              # Logging, helpers, metrics
├── tests/              # Unit + integration tests
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/GypsianMonk/advanced-multimodal-rag.git
cd advanced-multimodal-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your API keys and model paths
```

### 3. Ingest documents

```bash
python ingestion/ingest.py --source ./data --type pdf
```

### 4. Run the API

```bash
uvicorn api.app:app --reload --port 8000
```

---

## 🔑 Key Differentiators

| Feature | Basic RAG | This System |
|---|---|---|
| Retrieval | Vector only | BM25 + Vector (Hybrid) |
| Query handling | Raw pass-through | Intent classification + Expansion |
| Ranking | Top-K cosine | Multi-stage cross-encoder |
| Modalities | Text only | Text + Images + Tables |
| Validation | None | Self-consistency + source check |
| Memory | None | Query history + feedback loop |

---

## 📊 Evaluation Metrics

- **Retrieval**: Precision@K, Recall@K, MRR
- **Generation**: Faithfulness, Context Relevance, Hallucination Rate
- **System**: Latency (P50/P95), Throughput (QPS)

Run evals:

```bash
python utils/evaluate.py --config config/eval.yaml
```

---

## 🧩 Tech Stack

- **Embeddings**: `sentence-transformers`, `CLIP` (OpenAI)
- **Vector DB**: `FAISS` / `ChromaDB`
- **BM25**: `rank_bm25`
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: OpenAI GPT-4 / Anthropic Claude
- **API**: `FastAPI` + `Uvicorn`
- **Storage**: Redis (memory/cache), SQLite (feedback)

---

## 🤝 Contributing

PRs welcome. Please run tests before submitting:

```bash
pytest tests/ -v
```

---

## 📄 License

GypsianMonk
