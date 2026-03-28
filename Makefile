.PHONY: help install dev test lint format clean run docker-build docker-up ingest eval

# ─── Config ───────────────────────────────────────────────────────────────────
PYTHON     := python3
VENV       := venv
PIP        := $(VENV)/bin/pip
PYTEST     := $(VENV)/bin/pytest
UVICORN    := $(VENV)/bin/uvicorn
APP_CONFIG := config/config.yaml

# ─── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Advanced Multi-Modal RAG — Developer Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install       Install dependencies in virtualenv"
	@echo "  make dev           Install + copy example config"
	@echo "  make run           Start FastAPI server (port 8000)"
	@echo "  make test          Run all unit tests"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Auto-format with black + ruff"
	@echo "  make ingest        Ingest documents from ./data/raw"
	@echo "  make eval          Run evaluation suite"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start all services (API + Redis)"
	@echo "  make clean         Remove venv, cache, indexes"
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dev: install
	@if [ ! -f $(APP_CONFIG) ]; then \
		cp config/config.example.yaml $(APP_CONFIG); \
		echo "  ✅ config/config.yaml created — add your API keys"; \
	else \
		echo "  ℹ️  config/config.yaml already exists"; \
	fi
	@mkdir -p data/raw data/indexes logs

# ─── Run ──────────────────────────────────────────────────────────────────────
run:
	$(UVICORN) api.app:app --reload --port 8000

# ─── Testing ──────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short

test-cov:
	$(PYTEST) tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/black .
	$(VENV)/bin/ruff check --fix .

# ─── Data ─────────────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) ingestion/ingest.py --source data/raw --config $(APP_CONFIG)

eval:
	$(PYTHON) utils/evaluate.py --config config/eval.yaml

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t multimodal-rag:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

# ─── Clean ────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(VENV) __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -f data/indexes/*.index data/indexes/*.pkl data/indexes/*.json
