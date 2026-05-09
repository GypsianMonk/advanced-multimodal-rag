<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0c1445,50:1e3a8a,100:0c1445&height=220&section=header&text=Multi-Modal+RAG+System&fontSize=55&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Production-Grade%20Retrieval-Augmented%20Generation%20%7C%20Hybrid%20Search%20%7C%20Re-ranking%20%7C%20Answer%20Validation&descAlignY=60&descSize=14&descColor=93c5fd" width="100%"/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=19&pause=1000&color=93C5FD&center=true&vCenter=true&width=850&lines=BM25+%2B+FAISS+Hybrid+Retrieval+%7C+Cross-Encoder+Re-ranking;Text+%2B+Images+%2B+Tables+%7C+Multi-Modal+Fusion;Self-Consistency+Validation+%7C+Hallucination+Detection+%F0%9F%9A%80)](https://git.io/typing-svg)

<br/>

[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## ◈ What is This?

> *Production-grade RAG that goes far beyond basic "embed → retrieve → generate". It treats retrieval as a quality-control problem — deciding **what deserves to be retrieved** before generation ever happens.*

Most RAG systems do this:

```
Query ──► Embed ──► Top-K Cosine ──► LLM ──► Answer
```

**This system** does this:

```
User Query
   │
   ▼
🔍 Query Understanding     ← Intent classification, modality detection
   │
   ▼
🔄 Query Expansion         ← Synonyms, LLM reformulation
   │
   ▼
⚡ Hybrid Retrieval         ← BM25 + Vector Search (FAISS / Chroma)
   │
   ▼
🖼️  Multi-Modal Fusion      ← Text + Images + Tables unified
   │
   ▼
🏆 Re-ranking Engine       ← Cross-encoder + CLIP similarity
   │
   ▼
📊 Context Scoring         ← Relevance · Recency · Reliability
   │
   ▼
🤖 LLM Generation          ← Prompt-optimized context injection
   │
   ▼
✅ Answer Validation        ← Self-consistency + hallucination checks
   │
   ▼
🧠 Feedback Loop / Memory  ← Personalization, continuous improvement
   │
   ▼
Validated Answer + Source Attribution + Confidence Score
```

---

## ◈ Why This Beats Basic RAG

<div align="center">

| Feature | Basic RAG | This System |
|:---:|:---|:---|
| 🔍 **Retrieval** | Vector only | BM25 + Dense Vector (Hybrid Fusion) |
| 🧠 **Query Handling** | Raw pass-through | Intent classification + Expansion |
| 🏆 **Ranking** | Top-K cosine similarity | Multi-stage cross-encoder reranking |
| 🖼️ **Modalities** | Text only | Text + Images + Tables unified |
| ✅ **Validation** | None | Self-consistency + source checking |
| 💾 **Memory** | Stateless | Query history + personalized feedback loop |

</div>

---

## ◈ Tech Stack

<div align="center">

### ⟡ Embeddings & Retrieval
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-Text_Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://www.sbert.net/)
[![CLIP](https://img.shields.io/badge/CLIP-Image_Embeddings-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/research/clip)
[![FAISS](https://img.shields.io/badge/FAISS-Dense_Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=for-the-badge)](https://www.trychroma.com/)
[![BM25](https://img.shields.io/badge/rank__bm25-Sparse_Search-3ecf8e?style=for-the-badge)](https://github.com/dorianbrown/rank_bm25)

### ⟡ Re-ranking & Generation
[![Cross-Encoder](https://img.shields.io/badge/Cross--Encoder-MS--Marco_MiniLM-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-7c3aed?style=for-the-badge)](https://www.anthropic.com/claude)

### ⟡ API & Storage
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-4B8BBE?style=for-the-badge)](https://www.uvicorn.org/)
[![Redis](https://img.shields.io/badge/Redis-Memory_%26_Cache-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![SQLite](https://img.shields.io/badge/SQLite-Feedback_Store-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

### ⟡ Testing
[![Pytest](https://img.shields.io/badge/Pytest-Unit_%26_Integration-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)

</div>

---

## ◈ Project Structure

```
advanced-multimodal-rag/
│
├── 📥 ingestion/               ← Document loaders, chunking strategies
│
├── 🔢 embeddings/
│   ├── text.py                 ← Sentence Transformers
│   ├── image.py                ← CLIP embeddings
│   └── table.py                ← Tabular data encoding
│
├── 🔍 retrieval/
│   ├── bm25.py                 ← BM25 keyword search
│   ├── vector.py               ← Dense vector search (FAISS / Chroma)
│   └── hybrid.py               ← Fused hybrid retrieval
│
├── 🏆 ranking/
│   └── reranker.py             ← Cross-encoder + CLIP reranking
│
├── 🖼️  fusion/                  ← Multi-modal context aggregation
│
├── 🤖 llm/                     ← LLM generation + prompt templates
│
├── ✅ validation/               ← Answer validation, hallucination checks
│
├── 🧠 memory/                  ← Query history, feedback storage
│
├── 🌐 api/
│   └── app.py                  ← FastAPI REST interface
│
├── ⚙️  config/                  ← YAML configs, model registry
│
├── 🛠️  utils/                   ← Logging, helpers, metrics
│
└── 🧪 tests/                   ← Unit + integration tests
```

---

## ◈ Getting Started

### Installation

```bash
git clone https://github.com/GypsianMonk/advanced-multimodal-rag.git
cd advanced-multimodal-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit with your API keys and model paths
```

### Ingest Documents

```bash
# PDF ingestion
python ingestion/ingest.py --source ./data --type pdf

# Supported types: pdf · txt · html · images · csv/tables
```

### Run the API

```bash
uvicorn api.app:app --reload --port 8000
# Docs available at http://localhost:8000/docs
```

---

## ◈ Evaluation Metrics

<div align="center">

| Category | Metrics |
|:---:|:---|
| 🔍 **Retrieval** | Precision@K · Recall@K · MRR (Mean Reciprocal Rank) |
| 🤖 **Generation** | Faithfulness · Context Relevance · Hallucination Rate |
| ⚡ **System** | Latency P50/P95 · Throughput QPS |

</div>

```bash
# Run full evaluation suite
python utils/evaluate.py --config config/eval.yaml
```

---

## ◈ Contributing

```bash
# Fork → clone → branch
git checkout -b feature/your-feature

# Run tests before submitting
pytest tests/ -v

# Commit + PR
git commit -m "feat: description of what and why"
git push origin feature/your-feature
```

---

## ◈ License

Licensed under the **[MIT License](LICENSE)** — use it, extend it, build on it.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0c1445,50:1e3a8a,100:0c1445&height=120&section=footer" width="100%"/>

*"Retrieval as a quality-control problem — not an afterthought."*

**Built with ❤️ by [GypsianMonk](https://github.com/GypsianMonk)**

</div>
