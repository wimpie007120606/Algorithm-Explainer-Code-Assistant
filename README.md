# Algorithm Explainer & Code Assistant (RAG Pipeline)

A production-quality **Retrieval-Augmented Generation (RAG)** system that answers questions about algorithms and data structures using your own uploaded technical documents as the exclusive knowledge source.

Ask about Ford-Fulkerson, BFS, Dijkstra, Quadtrees — and get grounded answers with source citations, complexity analysis, and Java code **only when supported by your documents**.

---

## Why This Project Matters

Most "AI chat" tools answer from training data, which makes answers unverifiable and potentially wrong for domain-specific material (course notes, research papers, textbooks).

This system solves that by:
1. **Grounding every answer** in retrieved document chunks, not model hallucination
2. **Citing the source** (filename, page number) for every claim
3. **Explicitly refusing** to answer when the retrieved context is insufficient
4. **Separating concerns cleanly** — ingestion, retrieval, generation, and UI are decoupled modules

This is not a notebook tutorial. It is a portfolio-grade engineering project with a real architecture, tests, evaluation, and production-minded design.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│       AnswerService         │
│  retrieve → generate →      │
│  package AnswerResult       │
└────────┬──────────┬─────────┘
         │          │
         ▼          ▼
┌──────────────┐  ┌──────────────────┐
│ VectorStore  │  │    QA Chain      │
│  Retriever   │  │  Prompt + LLM    │
└──────┬───────┘  └──────────────────┘
       │
       ▼
┌──────────────┐
│  ChromaDB    │  (or Pinecone)
│  (local)     │
└──────────────┘
```

**Full architecture notes:** [`docs/architecture.md`](docs/architecture.md)

---

## Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| RAG Orchestration | LangChain |
| LLM | OpenAI (`gpt-4o-mini` default) / Gemini |
| Embeddings | OpenAI `text-embedding-3-small` / Gemini |
| Vector DB | ChromaDB (local) / Pinecone (optional) |
| PDF Parsing | pypdf |
| Text Splitting | langchain-text-splitters |
| Config | pydantic-settings + python-dotenv |
| Testing | pytest |
| Language | Python 3.11+ |

---

## Project Structure

```
algorithm-rag-assistant/
├── app/
│   └── streamlit_app.py          # Streamlit UI — entry point
├── src/
│   ├── config/settings.py        # Centralised config (pydantic-settings)
│   ├── ingestion/
│   │   ├── loaders.py            # PDF / TXT / MD loaders
│   │   ├── parser.py             # Metadata enrichment (headings, code detection)
│   │   ├── chunker.py            # Recursive character + code-aware chunking
│   │   └── pipeline.py           # Orchestration: load → parse → chunk → store
│   ├── embeddings/factory.py     # Provider-agnostic embedding model factory
│   ├── vectordb/
│   │   ├── chroma_store.py       # ChromaDB adapter
│   │   ├── pinecone_store.py     # Pinecone adapter (optional)
│   │   └── retriever.py          # VectorStoreRetriever with threshold filtering
│   ├── llm/
│   │   ├── factory.py            # Chat model factory (OpenAI / Gemini)
│   │   ├── prompts.py            # System prompt, context formatting
│   │   └── qa_chain.py           # LLM invocation with retry
│   ├── services/
│   │   ├── answer_service.py     # End-to-end RAG orchestration
│   │   └── citation_service.py   # Citation deduplication and formatting
│   ├── evaluation/
│   │   ├── metrics.py            # hit@k, precision@k, MRR, recall
│   │   └── eval_runner.py        # Eval suite runner + report
│   └── utils/
│       ├── logging.py            # Structured logging setup
│       ├── files.py              # File system helpers
│       └── ids.py                # Deterministic chunk ID generation
├── data/
│   ├── raw/                      # Drop your PDFs / TXT / MD files here
│   ├── processed/                # Intermediate processing output
│   └── chroma_db/                # ChromaDB persistence (auto-created)
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_chunking.py
│   ├── test_ingestion.py
│   ├── test_prompting.py
│   └── test_retrieval.py
├── scripts/
│   ├── ingest_docs.py            # CLI ingestion tool
│   ├── rebuild_index.py          # Wipe + rebuild vector index
│   └── run_eval.py               # Run evaluation suite
├── docs/architecture.md
├── .env.example
├── requirements.txt
└── Makefile
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd algorithm-rag-assistant

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Minimum required in `.env`:**
```env
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

All other settings have sensible defaults. See `.env.example` for the full list.

---

## Ingesting Documents

Place your PDF, TXT, or MD files in `data/raw/`, then run:

```bash
python scripts/ingest_docs.py
```

**Options:**
```bash
python scripts/ingest_docs.py --dir /path/to/docs   # Custom directory
python scripts/ingest_docs.py --dry-run              # Parse only, don't store
python scripts/ingest_docs.py --reset                # Wipe and re-ingest
```

You can also upload files directly from the Streamlit UI sidebar.

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Running Tests

```bash
pytest                          # All tests
pytest tests/test_chunking.py   # Specific test file
pytest -v --tb=short            # Verbose with short tracebacks
pytest --cov=src                # With coverage report
```

---

## Running Evaluation

```bash
python scripts/run_eval.py
python scripts/run_eval.py --dry-run            # Retrieval only (no LLM cost)
python scripts/run_eval.py --top-k 6
python scripts/run_eval.py --output eval.json   # Save report to file
```

---

## Rebuilding the Index

When you change `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL`:

```bash
python scripts/rebuild_index.py        # Prompts for confirmation
python scripts/rebuild_index.py --yes  # Skip confirmation
```

---

## Example Queries

Once documents are ingested:

- *"Explain how Ford-Fulkerson works."*
- *"What is the Max-Flow Min-Cut theorem?"*
- *"Give the Java implementation for a Quadtree insertion method."*
- *"Compare BFS and DFS using the uploaded lecture notes."*
- *"Why does this recurrence solve to O(n log n) according to the notes?"*
- *"What are the limitations of Bellman-Ford compared to Dijkstra?"*

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTOR_DB` | `chroma` | `chroma` or `pinecone` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `DEFAULT_TOP_K` | `4` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Min relevance score (0 = off) |
| `TEMPERATURE` | `0.0` | LLM temperature (0 = deterministic) |

---

## Strict Grounding Behavior

The assistant is designed to **refuse** to answer when the retrieved context is insufficient:

> *"The provided documents do not contain enough information to answer this question."*

It will not:
- Invent time complexity claims
- Fabricate Java implementations
- Make up proofs or theorems
- Pretend certainty when evidence is weak

This behavior is enforced by the system prompt in `src/llm/prompts.py`.

---

## Limitations

- **No OCR:** Scanned/image PDFs are not supported. Use text-layer PDFs only.
- **No conversation memory:** Each query is answered independently from fresh context — the session history panel is display-only and is not fed back into the RAG pipeline.
- **Embedding cost:** Each ingestion call generates embeddings via the API. Large document sets cost money.
- **Chunking is approximate:** Very short documents may produce only one chunk; very long code blocks may be split at non-ideal boundaries.
- **Evaluation suite is generic:** The default `EvalCase` list uses generic algorithm keywords. Replace with document-specific cases for meaningful metrics.

---

## Future Improvements

- [ ] Reranking layer (cross-encoder reranking after initial retrieval)
- [ ] Hybrid retrieval (BM25 keyword + vector cosine)
- [ ] OCR support via `pytesseract` for scanned PDFs
- [ ] Streaming LLM responses in the Streamlit UI
- [ ] Multi-turn conversation with context history
- [ ] RAGAS integration for automatic faithfulness evaluation
- [ ] Docker compose for one-command deployment
- [ ] Pinecone auto-provisioning script

---

## Screenshots

*Add screenshots here after running the app.*

---

## Resume-Ready Bullet Points

> Copy and adapt these for your resume or portfolio write-up:

- **Built a production-grade RAG system** in Python (LangChain, ChromaDB, OpenAI) that answers algorithm questions grounded exclusively in user-uploaded documents with full source citation.
- **Designed a modular ingestion pipeline** (PDF/TXT/MD → parse → chunk → embed → index) with deterministic chunk IDs for idempotent re-indexing and code-aware chunking for pseudocode-heavy documents.
- **Implemented a strict prompt engineering framework** that enforces citation grounding, refuses hallucinated answers, and structures responses into explanation, Java code, and uncertainty sections.
- **Built provider-agnostic adapter layers** for LLM (OpenAI/Gemini), embeddings, and vector store (ChromaDB/Pinecone) enabling zero-code-change backend swaps.
- **Wrote a lightweight retrieval evaluation module** computing hit@k, precision@k, MRR, and context recall to validate retrieval quality without external evaluation infrastructure.
- **Delivered a polished Streamlit UI** with configurable retrieval settings, real-time document upload, citation display, debug panels, and answer export.

---

## License

MIT
