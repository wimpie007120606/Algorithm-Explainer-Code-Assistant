# Architecture Notes — Algorithm RAG Assistant

## Overview

This system implements a **Retrieval-Augmented Generation (RAG)** pipeline for
answering questions about algorithms and data structures using uploaded technical
documents as the exclusive knowledge source.

```
User Question
     │
     ▼
┌─────────────────┐
│  Streamlit UI   │  ← app/streamlit_app.py
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AnswerService                            │
│                   src/services/answer_service.py                │
│                                                                 │
│  1. Validate KB is non-empty                                    │
│  2. Retrieve top-k chunks from VectorStoreRetriever             │
│  3. Invoke QA chain (prompt + LLM)                              │
│  4. Return AnswerResult (answer + chunks + citations)           │
└─────────────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌────────────────────┐    ┌────────────────────────┐
│ VectorStoreRetriever│   │       QA Chain          │
│ src/vectordb/       │   │  src/llm/qa_chain.py    │
│ retriever.py        │   │                         │
│                     │   │  Prompt construction +  │
│  similarity_search  │   │  LLM call with retries  │
└────────┬───────────┘    └────────────────────────┘
         │                           │
         ▼                           ▼
┌────────────────────┐    ┌────────────────────────┐
│   ChromaVectorStore│   │      LLM Factory        │
│   (or Pinecone)    │   │  src/llm/factory.py     │
│   src/vectordb/    │   │                         │
│   chroma_store.py  │   │  OpenAI / Gemini        │
└────────────────────┘    └────────────────────────┘
```

## Component Responsibilities

### `app/streamlit_app.py`
- Pure UI layer — no business logic
- Caches services with `@st.cache_resource`
- Routes user events to AnswerService
- Renders answers, citations, and debug panels

### `src/config/settings.py`
- Single source of truth for all configuration
- Uses pydantic-settings with .env loading
- Returns a cached singleton via `get_settings()`

### `src/ingestion/`
| Module | Responsibility |
|--------|---------------|
| `loaders.py` | Raw file I/O — PDF, TXT, MD |
| `parser.py` | Text enrichment — section headings, code detection |
| `chunker.py` | Splitting — RecursiveCharacterTextSplitter with code-aware path |
| `pipeline.py` | Orchestration — load → parse → chunk → store |

### `src/embeddings/factory.py`
- Provider-agnostic embedding model factory
- Supports OpenAI and Google Gemini
- Cached singleton

### `src/vectordb/`
| Module | Responsibility |
|--------|---------------|
| `chroma_store.py` | ChromaDB adapter — upsert, similarity search, reset |
| `pinecone_store.py` | Pinecone adapter (optional) — same interface |
| `retriever.py` | High-level retriever — threshold filtering, `RetrievedChunk` type |

### `src/llm/`
| Module | Responsibility |
|--------|---------------|
| `factory.py` | Chat model factory (OpenAI / Gemini) |
| `prompts.py` | System prompt, context formatting, message assembly |
| `qa_chain.py` | LLM invocation with tenacity retry logic |

### `src/services/`
| Module | Responsibility |
|--------|---------------|
| `answer_service.py` | End-to-end RAG orchestration, AnswerResult packaging |
| `citation_service.py` | Citation deduplication and formatting |

### `src/evaluation/`
| Module | Responsibility |
|--------|---------------|
| `metrics.py` | hit@k, precision@k, MRR, recall, keyword coverage |
| `eval_runner.py` | EvalCase runner, EvalReport aggregation |

---

## Data Flow

### Ingestion (offline)
```
PDF / TXT / MD file
       │
       ▼ loaders.py → List[Document] (one per page)
       │
       ▼ parser.py  → enriched metadata (section_heading, is_code_heavy)
       │
       ▼ chunker.py → List[Document] (chunks with chunk_id, char_count)
       │
       ▼ embeddings/factory.py → embedding vectors
       │
       ▼ vectordb/chroma_store.py → persisted in ChromaDB
```

### Query (online, per question)
```
User question
       │
       ▼ VectorStoreRetriever.retrieve()
       │   → embedding of question
       │   → cosine similarity search in ChromaDB
       │   → top-k RetrievedChunk objects (above threshold)
       │
       ▼ prompts.format_context_blocks()
       │   → SOURCE 1: filename (Page N) \n content
       │   → SOURCE 2: ...
       │
       ▼ prompts.build_prompt_messages()
       │   → [SystemMessage(SYSTEM_PROMPT), HumanMessage(context + question)]
       │
       ▼ LLM.invoke()
       │   → grounded answer string
       │
       ▼ CitationService.build_citations()
       │   → deduplicated, scored Citation objects
       │
       ▼ AnswerResult → Streamlit UI
```

---

## Design Decisions

### Why ChromaDB?
ChromaDB runs entirely locally with zero infrastructure overhead, ideal for
portfolio demonstration and local development. The adapter pattern means
switching to Pinecone for production requires only changing `VECTOR_DB=pinecone`.

### Why RecursiveCharacterTextSplitter?
It respects natural language boundaries (paragraphs, sentences) before falling
back to word/character splits. This produces chunks that are coherent reading
units, which significantly improves retrieval relevance for algorithm explanations.

### Why strict grounding in the system prompt?
The core value proposition of a RAG system over a raw LLM call is provenance.
If the model is allowed to use its training knowledge, citations become
meaningless and the system provides no advantage over `gpt-4o` directly.
The strict system prompt is the enforcement mechanism for this guarantee.

### Why tenacity retry in qa_chain.py?
OpenAI API calls can fail transiently (rate limits, network issues). Automatic
exponential-backoff retries make the system more resilient in production.

### Why deterministic chunk IDs (utils/ids.py)?
Re-indexing the same document should not duplicate chunks in the vector store.
Deterministic SHA-256-derived IDs enable idempotent upserts in ChromaDB.

---

## Extension Points

| Extension | Where to change |
|-----------|----------------|
| Add Pinecone backend | `src/vectordb/pinecone_store.py` — already implemented |
| Add Gemini LLM | `src/llm/factory.py` — already implemented |
| Add reranking | `src/vectordb/retriever.py` — add rerank step after similarity search |
| Add hybrid retrieval | `src/vectordb/retriever.py` — add BM25 + combine scores |
| Add OCR for scanned PDFs | `src/ingestion/loaders.py` — add `pytesseract` path |
| Add streaming answers | `src/llm/qa_chain.py` — use `model.stream()` |
| Add conversation history | `src/services/answer_service.py` — inject prior Q/A pairs |
