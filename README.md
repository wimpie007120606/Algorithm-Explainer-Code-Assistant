# StudyMate Knowledge Coach

**An AI-powered data science study platform built with a production-style RAG architecture.**

StudyMate turns uploaded textbooks, lecture notes, question papers, diagrams, screenshots, and code notes into a grounded learning assistant for data science students. It is designed to help a learner master the foundations behind data science: calculus, mathematical statistics, computer science, Python, SQL, algorithms, and modeling workflows.

Unlike a generic chatbot, StudyMate answers only from the learner's uploaded material, cites source pages, refuses unsupported questions, and exposes retrieval diagnostics so the user can understand why an answer was or was not possible.

---

## Why This Project Stands Out

This project is intentionally built like real software, not a notebook demo.

It demonstrates:

- **End-to-end RAG engineering**: ingestion, parsing, chunking, embeddings, vector storage, retrieval, prompt construction, answer generation, citations, evaluation, and UI.
- **Grounded AI behavior**: the assistant is explicitly designed to avoid hallucination and to cite source material.
- **Multi-modal document ingestion**: supports PDFs, markdown, text, images, and scanned/math-heavy pages through GPT-4o vision fallback.
- **Data science education focus**: the interface and prompt strategy are tuned for calculus, mathematical statistics, CS, Python, SQL, and ML foundations.
- **Production-minded structure**: modular packages, provider adapters, deterministic chunk IDs, pydantic settings, tests, linting, and deployment configuration.
- **User-centered UX**: dark high-contrast interface, learning tracks, source focus, study modes, document health checks, debug panels, and large textbook upload support.

---

## Product Vision

Data science students often jump into tools before they truly understand the mathematical and computational foundations. StudyMate is designed to bridge that gap.

The learner can upload:

- a Stewart calculus textbook
- a mathematical statistics textbook
- a computer science or algorithms textbook
- a Python/data science textbook
- lecture notes, memos, slides, diagrams, and question papers

Then StudyMate can help with:

- explaining difficult concepts from the uploaded sources
- generating practice questions from the exact textbook material
- creating flashcards for definitions, formulas, and workflows
- building timed study plans
- connecting calculus to optimization and gradient descent
- connecting probability/statistics to machine learning foundations
- connecting CS fundamentals to efficient data science implementation
- tracing every answer back to source filenames and page numbers

The result is a grounded, source-aware study environment instead of an unverified AI tutor.

---

## Core Capabilities

| Capability | What It Does |
|---|---|
| Grounded Q&A | Answers from retrieved source chunks only |
| Source Citations | Shows filename, page, and relevance score |
| Study Modes | Answer, Explain, Summarize, Practice, Flashcards, Study Plan |
| Learning Tracks | Data Science Core, Calculus, Math Stats, CS, Python/Data Science, General Study |
| Multi-Textbook Retrieval | Searches across a growing source library with reranking |
| Source Focus | Restricts retrieval to one selected document when needed |
| Vision Fallback | Renders low-text PDF pages and transcribes them with GPT-4o vision |
| Document Health | Flags PDFs that look like scanned/page-label-only sources |
| Debug Panel | Shows vector, lexical, and combined retrieval scores |
| Export | Downloads answers as Markdown |
| Local + Cloud Ready | Runs locally or on Streamlit Community Cloud |

---

## What Companies Should Notice

This repository demonstrates practical engineering judgment across product, backend, AI, and UX:

- **Architecture discipline**: UI, services, LLM, retrieval, ingestion, config, evaluation, and utilities are separated into clear modules.
- **Abstraction without overengineering**: vector stores, LLMs, and embeddings use adapter/factory patterns so providers can be swapped without rewriting the app.
- **Reliability thinking**: LLM calls use retry handling for transient API failures.
- **Data integrity**: deterministic chunk IDs prevent duplicate chunks during re-ingestion.
- **Failure transparency**: low-text PDFs are detected instead of silently indexing useless `Page 1` chunks.
- **Responsible AI behavior**: the prompt contract forces the model to refuse unsupported answers rather than pretend.
- **Testing culture**: unit tests cover settings, ingestion, chunking, retrieval, prompting, and answer orchestration.
- **Deployment awareness**: Streamlit secrets, local config, upload limits, and ephemeral cloud storage behavior are documented.

---

## Architecture Overview

```text
Uploaded sources
      |
      v
loaders.py
  - PDF text extraction
  - image ingestion
  - scanned PDF vision fallback
      |
      v
parser.py
  - section metadata
  - code/structured-work detection
      |
      v
chunker.py
  - recursive semantic splitting
  - deterministic chunk IDs
      |
      v
embeddings/factory.py
  - OpenAI or Gemini embeddings
      |
      v
ChromaDB / Pinecone
      |
      v
VectorStoreRetriever
  - expanded candidate retrieval
  - lexical/source-name scoring
  - threshold filtering
      |
      v
AnswerService
  - validates knowledge base
  - retrieves context
  - invokes QA chain
  - packages answer + citations
      |
      v
Streamlit UI
  - study modes
  - source focus
  - diagnostics
  - downloads
```

Detailed architecture notes are in [docs/architecture.md](docs/architecture.md).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| UI | Streamlit |
| RAG Framework | LangChain |
| LLMs | OpenAI / Gemini |
| Embeddings | OpenAI `text-embedding-3-small` / Gemini embeddings |
| Vector Store | ChromaDB local persistence / optional Pinecone |
| PDF Parsing | pypdf |
| Vision Fallback | PyMuPDF page rendering + GPT-4o vision |
| Chunking | langchain-text-splitters |
| Config | pydantic-settings + python-dotenv |
| Evaluation | Custom retrieval metrics |
| Quality | pytest, pytest-cov, ruff |

---

## Project Structure

```text
studymate-rag-assistant/
├── app/
│   └── streamlit_app.py          # High-contrast Streamlit study interface
├── src/
│   ├── config/                   # Typed environment/settings layer
│   ├── embeddings/               # Provider-agnostic embedding factory
│   ├── evaluation/               # Retrieval metrics and eval runner
│   ├── ingestion/                # Load, parse, chunk, and store documents
│   ├── llm/                      # Prompt contract and model invocation
│   ├── services/                 # RAG orchestration and citation services
│   ├── utils/                    # Logging, IDs, file helpers
│   └── vectordb/                 # Chroma/Pinecone adapters and retriever
├── data/
│   ├── raw/                      # Optional local source documents
│   ├── processed/                # Processing output
│   └── chroma_db/                # Local vector database persistence
├── docs/
│   └── architecture.md
├── scripts/
│   ├── ingest_docs.py
│   ├── rebuild_index.py
│   └── run_eval.py
├── tests/
├── .streamlit/
│   ├── config.toml               # Dark theme and large upload limit
│   └── secrets.toml.example
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── Makefile
```

---

## Study Modes

StudyMate supports multiple task modes so it behaves less like a plain chatbot and more like a learning system.

| Mode | Example |
|---|---|
| Answer | "What does the textbook say about Bayes' theorem?" |
| Explain | "Teach me maximum likelihood step by step." |
| Summarize | "Summarize the probability ideas I need before ML." |
| Practice | "Create mixed calculus and statistics practice questions." |
| Flashcards | "Make flashcards for distributions and derivatives." |
| Study Plan | "Build a 2-hour plan for gradient descent foundations." |

---

## Learning Tracks

The UI includes learning tracks that shape the assistant's output while keeping the answer grounded in retrieved context.

- **Data Science Core**: connects calculus, math stats, CS, Python, SQL, and ML foundations when sources support it.
- **Calculus**: emphasizes definitions, intuition, worked examples, and practice.
- **Math Stats**: emphasizes probability, inference, estimators, distributions, and statistical reasoning.
- **Computer Science**: emphasizes algorithms, data structures, complexity, and implementation ideas.
- **Python / Data Science**: emphasizes Python, NumPy, pandas, analysis workflows, and modeling.
- **General Study**: keeps the response focused on the selected uploaded material.

---

## Retrieval Strategy

The retriever does more than a basic top-k vector search.

1. It embeds the user query.
2. It fetches an expanded candidate set from the vector database.
3. It computes a lexical/source-name overlap score.
4. It combines vector relevance with lexical relevance.
5. It filters by the user-configurable similarity threshold.
6. It returns the final chunks to the LLM with metadata and scores.

This makes broad requests such as "make mixed data science practice questions" more likely to pull from the right subject areas in a multi-textbook library.

---

## Scanned PDF and Textbook Handling

Text-layer PDFs are best because they can be extracted quickly and cheaply.

For scanned or math-heavy PDFs, StudyMate includes a fallback path:

1. pypdf tries to extract page text.
2. Pages with too little substantive text are detected.
3. PyMuPDF renders those pages to images.
4. GPT-4o vision transcribes and describes the page.
5. The generated text is chunked and embedded.

This prevents a common RAG failure where a PDF "ingests successfully" but the vector store only contains useless chunks like:

```text
Page 1
Page 2
Page 3
```

---

## Setup

```bash
git clone <your-repo-url>
cd studymate-rag-assistant

python -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt
cp .env.example .env
```

Add your API key to `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DB=chroma
```

Run the app:

```bash
make run
```

Open:

```text
http://127.0.0.1:8501
```

---

## Ingesting Documents

You can upload documents directly in the sidebar, or place files in `data/raw/` and run:

```bash
make ingest
```

Useful commands:

```bash
python scripts/ingest_docs.py --dir /path/to/docs
python scripts/ingest_docs.py --dry-run
python scripts/ingest_docs.py --reset
make rebuild-index
```

For large textbooks:

- Prefer text-layer PDFs.
- Upload one or two books first, verify retrieval, then add more.
- Increase context chunks in the sidebar for cross-book questions.
- Use source focus when studying one textbook chapter or paper.
- Do not commit copyrighted textbooks to Git.

---

## Example Data Science Study Prompts

```text
Teach me how derivatives, optimization, and gradient descent connect.
```

```text
Create mixed practice questions from calculus, math stats, CS, and Python data science.
```

```text
Explain maximum likelihood using only my uploaded statistics textbook.
```

```text
Build a 2-hour study plan for becoming strong at data science foundations.
```

```text
Make flashcards for distributions, integrals, algorithms, and pandas workflows.
```

```text
Compare the textbook explanations of variance, covariance, and correlation.
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---:|---|
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `OPENAI_API_KEY` | empty | Required for OpenAI LLMs, embeddings, and vision fallback |
| `GEMINI_API_KEY` | empty | Required when using Gemini |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTOR_DB` | `chroma` | `chroma` or `pinecone` |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Local Chroma persistence path |
| `DEFAULT_TOP_K` | `4` | Default context chunks |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum relevance score; `0.0` disables filtering |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | `4` | Candidate expansion multiplier before reranking |
| `RETRIEVAL_MAX_CANDIDATES` | `48` | Maximum expanded candidates |
| `RETRIEVAL_LEXICAL_WEIGHT` | `0.18` | Keyword/source-name relevance weight |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Chunk overlap |
| `PDF_MIN_TEXT_CHARS` | `40` | Minimum useful text before fallback/skip |
| `PDF_VISION_FALLBACK` | `true` | Use vision fallback for low-text PDF pages |
| `PDF_VISION_DPI` | `180` | PDF rendering DPI for vision fallback |
| `TEMPERATURE` | `0.0` | Deterministic answer generation |
| `MAX_TOKENS` | `2048` | Maximum answer tokens |

---

## Validation

The project has automated checks for core behavior:

```bash
make lint
make test
```

Current validated test coverage areas:

- settings validation
- PDF/text/markdown ingestion
- low-text PDF detection
- document chunking
- prompt construction
- no-context fallback behavior
- answer service orchestration
- retrieval thresholding, source filtering, reranking metadata
- vector store factory behavior

Recent validation result:

```text
89 passed
Ruff: all checks passed
Streamlit AppTest: exceptions 0
```

---

## Evaluation

Run retrieval-focused evaluation:

```bash
make eval
```

Run full evaluation with LLM answers:

```bash
make eval-full
```

The evaluation module computes:

- hit@k
- precision@k
- reciprocal rank
- context recall
- average retrieval score
- keyword coverage

---

## Deployment Notes

For Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a Streamlit app from the repository.
3. Set the main file path to:

```text
app/streamlit_app.py
```

4. Add secrets in Streamlit's secrets editor:

```toml
LLM_PROVIDER = "openai"
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DB = "chroma"
OPENAI_API_KEY = "sk-..."
```

The app bridges Streamlit secrets into environment variables at startup.

Important cloud caveat:

- Local Chroma storage on Streamlit Community Cloud is ephemeral.
- Uploaded documents may need to be re-ingested after app restarts.
- Pinecone can be enabled for persistent hosted vector storage.

---

## Responsible AI and Limitations

StudyMate is intentionally strict:

- It should not invent facts, formulas, proofs, code, or textbook claims.
- It should refuse when retrieved context is insufficient.
- It should cite the source material used.
- It should not treat page labels as meaningful textbook content.

Known limitations:

- Large scanned textbooks can be slow and expensive to process through vision fallback.
- The app does not yet include multi-turn memory in retrieval.
- Retrieval quality depends on document extraction quality and chunking.
- Cloud-local Chroma persistence is temporary unless a hosted vector database is used.
- A copyrighted textbook should not be committed into the repository.

---

## Roadmap

Completed:

- [x] Modular RAG pipeline
- [x] Streamlit study interface
- [x] Study modes
- [x] Learning tracks
- [x] Source citations
- [x] Chroma vector persistence
- [x] Pinecone adapter
- [x] OpenAI and Gemini provider factories
- [x] Low-text PDF detection
- [x] GPT-4o vision fallback
- [x] Lightweight vector + lexical reranking
- [x] Retrieval evaluation module
- [x] Dark high-contrast data science UI

Planned:

- [ ] Streaming token responses
- [ ] Multi-turn study memory
- [ ] Cross-encoder reranking
- [ ] Per-chapter textbook navigation
- [ ] Assignment-style practice generator
- [ ] Learning progress dashboard
- [ ] Persistent hosted vector database setup script
- [ ] Docker Compose deployment
- [ ] RAGAS or LLM-as-judge evaluation suite

---

## Portfolio Talking Points

Use these as interview/resume talking points:

- Built a production-style RAG application that transforms uploaded academic material into a grounded AI study platform for data science foundations.
- Designed a modular ingestion pipeline for PDFs, markdown, text, images, and scanned textbook pages using pypdf, PyMuPDF, GPT-4o vision, LangChain, and ChromaDB.
- Implemented strict source-grounded prompting so the assistant refuses unsupported answers and cites source filenames/pages.
- Built a hybrid retrieval layer that combines vector similarity with lexical/source-name scoring and threshold filtering.
- Added deterministic chunk IDs to make re-ingestion idempotent and avoid duplicate vector records.
- Created provider-agnostic factories for OpenAI/Gemini chat models and embeddings.
- Developed a high-contrast Streamlit UX with learning tracks, study modes, source focus, document health diagnostics, and debug retrieval scoring.
- Wrote tests across ingestion, chunking, retrieval, prompting, settings, and orchestration to validate the system end to end.

---

## Suggested Project Summary

> StudyMate is a RAG-powered data science study platform that lets students upload textbooks and course material, then receive grounded explanations, practice questions, flashcards, summaries, and study plans with source citations. I built the full pipeline across ingestion, chunking, embeddings, vector search, reranking, prompt design, answer orchestration, evaluation, and Streamlit UX, including scanned-PDF vision fallback and strict no-hallucination behavior.

---

## License

MIT
