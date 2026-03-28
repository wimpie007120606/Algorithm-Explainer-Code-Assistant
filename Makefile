.PHONY: install setup ingest rebuild-index run test test-cov eval clean lint help

# Default target
help:
	@echo "Algorithm RAG Assistant — available commands:"
	@echo ""
	@echo "  make install       Install Python dependencies"
	@echo "  make setup         Copy .env.example → .env (run once)"
	@echo "  make ingest        Ingest documents from data/raw/"
	@echo "  make rebuild-index Wipe and rebuild the vector index"
	@echo "  make run           Launch the Streamlit app"
	@echo "  make test          Run the test suite"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make eval          Run retrieval evaluation (dry-run)"
	@echo "  make lint          Run ruff linter"
	@echo "  make clean         Remove __pycache__ and .ruff_cache"
	@echo ""

install:
	pip install -r requirements.txt

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example — add your API keys."; \
	else \
		echo ".env already exists — not overwriting."; \
	fi

ingest:
	python scripts/ingest_docs.py

rebuild-index:
	python scripts/rebuild_index.py --yes

run:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --tb=short

eval:
	python scripts/run_eval.py --dry-run

eval-full:
	python scripts/run_eval.py

lint:
	ruff check src/ tests/ app/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
