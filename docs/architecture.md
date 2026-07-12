# Architecture Notes - StudyMate Knowledge Coach

StudyMate is a modular Retrieval-Augmented Generation system for grounded academic study. It is optimized for a data science learning library made from calculus, mathematical statistics, computer science, Python/data science, and course-specific documents.

The system is designed around one core rule: the model should answer from retrieved source material, cite that material, and refuse when the source context is insufficient.

---

## High-Level Flow

```text
                 ┌──────────────────────────┐
                 │     Streamlit UI          │
                 │  learning tracks, modes,  │
                 │  source focus, diagnostics│
                 └────────────┬─────────────┘
                              │
                              v
                 ┌──────────────────────────┐
                 │      AnswerService        │
                 │ validation + orchestration │
                 └────────────┬─────────────┘
                              │
             ┌────────────────┴────────────────┐
             v                                 v
┌──────────────────────────┐      ┌──────────────────────────┐
│  VectorStoreRetriever    │      │        QA Chain           │
│ expanded search + rerank │      │ prompt + LLM + retry      │
└────────────┬─────────────┘      └────────────┬─────────────┘
             │                                 │
             v                                 v
┌──────────────────────────┐      ┌──────────────────────────┐
│ ChromaDB / Pinecone      │      │ OpenAI / Gemini Chat      │
│ persisted source chunks  │      │ grounded response         │
└──────────────────────────┘      └──────────────────────────┘
```

---

## Ingestion Pipeline

```text
PDF / TXT / MD / Image
      |
      v
loaders.py
  - text-layer PDF extraction with pypdf
  - image ingestion through GPT-4o vision
  - low-text PDF detection
  - optional PyMuPDF rendering + GPT-4o vision fallback
      |
      v
parser.py
  - section heading inference
  - code / structured-work detection
      |
      v
chunker.py
  - RecursiveCharacterTextSplitter
  - code-aware separator path
  - deterministic chunk_id
  - char_count metadata
      |
      v
embeddings/factory.py
  - OpenAI or Gemini embeddings
      |
      v
vectordb/chroma_store.py or pinecone_store.py
  - idempotent upsert
  - similarity search
  - collection reset
```

### Important ingestion behavior

The PDF loader does not blindly trust extracted text. If a page only yields a page label such as `Page 1`, it is treated as low-value text. When enabled, the PDF vision fallback renders the page and asks GPT-4o vision to transcribe formulas, diagrams, labels, question numbers, and other study material.

This addresses a common RAG failure mode: a scanned textbook appears to ingest successfully but produces useless vector chunks.

---

## Query Pipeline

```text
User request
      |
      v
AnswerService.answer_study()
  - validates question
  - checks knowledge base count
  - applies study mode and learning track
      |
      v
VectorStoreRetriever.retrieve()
  - embeds query
  - fetches expanded candidates
  - applies optional source filter
  - computes lexical/source-name overlap
  - combines vector + lexical score
  - filters by similarity threshold
      |
      v
prompts.format_context_blocks()
  - formats SOURCE N blocks
  - includes filename and page
      |
      v
qa_chain()
  - builds system + human messages
  - invokes LLM with retry
      |
      v
CitationService
  - deduplicates citations
  - formats source references
      |
      v
AnswerResult
  - answer text
  - chunks
  - metadata
  - source filenames
```

---

## Component Responsibilities

| Component | Responsibility |
|---|---|
| `app/streamlit_app.py` | UI, study modes, learning tracks, upload workflow, source library, debug panels |
| `src/config/settings.py` | Typed settings and environment variable loading |
| `src/ingestion/loaders.py` | PDF/TXT/MD/image loading and vision fallback |
| `src/ingestion/parser.py` | Metadata enrichment |
| `src/ingestion/chunker.py` | Chunking and deterministic chunk IDs |
| `src/ingestion/pipeline.py` | End-to-end ingestion orchestration |
| `src/embeddings/factory.py` | Embedding provider abstraction |
| `src/vectordb/chroma_store.py` | Chroma vector store adapter |
| `src/vectordb/pinecone_store.py` | Pinecone vector store adapter |
| `src/vectordb/retriever.py` | Expanded retrieval, reranking, threshold filtering |
| `src/llm/factory.py` | Chat model provider abstraction |
| `src/llm/prompts.py` | Grounding rules and study-mode prompt instructions |
| `src/llm/qa_chain.py` | Model invocation and retry handling |
| `src/services/answer_service.py` | RAG orchestration boundary |
| `src/services/citation_service.py` | Citation formatting and deduplication |
| `src/evaluation/metrics.py` | Retrieval metric calculations |
| `src/evaluation/eval_runner.py` | Evaluation runner and reporting |

---

## Retrieval Design

The retriever intentionally avoids being a thin wrapper around top-k vector search.

It uses:

1. **Expanded candidate search**: pulls more than the final top-k.
2. **Optional source filtering**: lets the UI restrict results to one textbook or paper.
3. **Lexical overlap scoring**: boosts exact subject/source-name matches.
4. **Combined score**: blends vector relevance with lexical relevance.
5. **Threshold filtering**: prevents weak context from being sent to the LLM.
6. **Debug metadata**: stores vector, lexical, and combined scores for UI inspection.

This is especially important for a multi-textbook library where a data science query may require chunks from calculus, statistics, CS, and Python sources.

---

## Prompt Contract

The prompt is deliberately strict:

- Answer only from retrieved context.
- Refuse when the retrieved context is insufficient.
- Do not invent formulas, proofs, code, dates, definitions, or textbook claims.
- Cite source documents and pages.
- Separate direct source facts from cross-source synthesis.
- Generate practice questions, flashcards, and study plans only from retrieved material.

The goal is not merely to produce fluent answers. The goal is to preserve trust.

---

## Study UX Design

The UI is designed as a study cockpit rather than a generic chat page.

It includes:

- dark high-contrast interface
- data science mastery hero
- advanced programming/math visual system
- learning track selector
- study mode selector
- source focus selector
- available-time input for study plans
- knowledge library health indicators
- low-text PDF warnings
- retrieval debug table
- prompt preview
- Markdown answer export

---

## Configuration Highlights

The most important configuration values for retrieval and textbooks:

| Setting | Purpose |
|---|---|
| `DEFAULT_TOP_K` | Final chunks sent to the model |
| `SIMILARITY_THRESHOLD` | Minimum relevance score |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | Candidate expansion before reranking |
| `RETRIEVAL_MAX_CANDIDATES` | Maximum candidates retrieved |
| `RETRIEVAL_LEXICAL_WEIGHT` | Keyword/source-name scoring weight |
| `PDF_MIN_TEXT_CHARS` | Low-text PDF detection threshold |
| `PDF_VISION_FALLBACK` | Enables scanned-page vision fallback |
| `PDF_VISION_DPI` | PDF page rendering resolution |

---

## Why These Decisions Matter

### Modular boundaries

The UI does not know how retrieval, prompting, or ingestion work. This makes the app easier to test, change, and deploy.

### Provider abstraction

The LLM, embedding model, and vector database are isolated behind factories/adapters. This avoids vendor lock-in and makes OpenAI/Gemini/Chroma/Pinecone switching a configuration issue rather than a rewrite.

### Deterministic chunk IDs

Re-ingesting the same source should not duplicate vectors. Stable IDs keep the vector store consistent.

### Low-text PDF detection

Many academic PDFs are scanned or math-rendered. Detecting low-value extracted text prevents false confidence in broken ingestion.

### Strict refusal behavior

For education, unsupported answers are actively harmful. The model is instructed to refuse rather than invent.

### Retrieval diagnostics

The app exposes scores and retrieved chunks so debugging is possible when an answer is missing or weak.

---

## Quality Gates

The project is validated with:

```bash
ruff check src/ tests/ app/ scripts/
pytest tests/ -q --tb=short
```

Covered areas include:

- settings validation
- ingestion behavior
- low-text PDF detection
- chunking
- retrieval filtering and reranking metadata
- source filtering
- prompt construction
- no-context fallback
- answer orchestration

---

## Extension Points

| Goal | Main Files |
|---|---|
| Add streaming responses | `src/llm/qa_chain.py`, `app/streamlit_app.py` |
| Add conversation memory | `src/services/answer_service.py` |
| Add cross-encoder reranking | `src/vectordb/retriever.py` |
| Add chapter navigation | `src/ingestion/parser.py`, `app/streamlit_app.py` |
| Add hosted vector storage setup | `src/vectordb/pinecone_store.py`, scripts |
| Add learning progress tracking | new service + UI state/storage |
| Add RAGAS/LLM judge evaluation | `src/evaluation/` |

