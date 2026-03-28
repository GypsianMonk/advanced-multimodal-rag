# Contributing to Advanced Multi-Modal RAG

Thanks for your interest in contributing! Here's how to get involved.

## Development Setup

```bash
git clone https://github.com/GypsianMonk/advanced-multimodal-rag.git
cd advanced-multimodal-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install ruff black pytest pytest-asyncio
```

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/short-description` | `feat/chroma-backend` |
| Bug fix | `fix/short-description` | `fix/bm25-tokenizer` |
| Docs | `docs/short-description` | `docs/api-examples` |
| Refactor | `refactor/...` | `refactor/pipeline-cleanup` |

## Commit Style (Conventional Commits)

```
feat: add ChromaDB vector backend
fix: handle empty BM25 query results
docs: add API usage examples
test: add coverage for metadata filter
refactor: simplify context fusion scoring
```

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

## Code Style

```bash
ruff check .        # linting
black .             # formatting
```

## Pull Request Checklist

- [ ] Tests added/updated for all changes
- [ ] `ruff` and `black` pass cleanly
- [ ] Docstrings updated for changed functions
- [ ] `config.example.yaml` updated if new config keys added
- [ ] README updated if new features added

## Areas Where Contributions Are Welcome

- Additional vector DB backends (Qdrant, Weaviate, Pinecone)
- Streaming API responses
- LLM-as-judge validation
- Web scraping ingestion pipeline
- Evaluation dataset creation tooling
- Performance benchmarks
