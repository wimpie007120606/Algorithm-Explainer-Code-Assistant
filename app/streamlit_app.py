"""
Algorithm Explainer & Code Assistant — Streamlit UI

Entry point:
    streamlit run app/streamlit_app.py

Architecture:
    This module handles only UI concerns.  All business logic lives in src/.
    The app bootstraps the RAG services on first load (cached via st.cache_resource),
    then routes user interactions to AnswerService and CitationService.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run app/streamlit_app.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# ─── page config must be the very first Streamlit call ────────────────────────
st.set_page_config(
    page_title="Algorithm RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── imports after page_config ────────────────────────────────────────────────
from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.services.answer_service import AnswerService, AnswerResult
from src.services.citation_service import CitationService
from src.vectordb.retriever import VectorStoreRetriever
from src.utils.logging import get_logger

log = get_logger(__name__)

# ─── CSS / styling ────────────────────────────────────────────────────────────

_CSS = """
<style>
    /* Answer card */
    .answer-card {
        background: #f8f9fa;
        border-left: 4px solid #4c8bf5;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    /* Citation card */
    .cite-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }
    /* Score badge */
    .score-badge {
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    /* Warning / empty state */
    .empty-state {
        color: #888;
        text-align: center;
        padding: 2rem;
        font-style: italic;
    }
    /* Section labels */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─── cached service initialisation ───────────────────────────────────────────


@st.cache_resource(show_spinner="Connecting to knowledge base…")
def _get_answer_service(top_k: int) -> AnswerService:
    """Initialise the AnswerService once and cache it for the session."""
    try:
        retriever = VectorStoreRetriever(top_k=top_k)
        service = AnswerService(retriever=retriever, top_k=top_k)
        return service
    except Exception as exc:
        st.error(f"Failed to initialise the RAG service: {exc}")
        st.stop()


@st.cache_resource(show_spinner="Loading ingestion pipeline…")
def _get_ingestion_pipeline() -> IngestionPipeline:
    """Return an IngestionPipeline wired to the live vector store."""
    from src.vectordb.retriever import get_vector_store

    store = get_vector_store()
    return IngestionPipeline(vector_store=store)


# ─── sidebar ──────────────────────────────────────────────────────────────────


def _render_sidebar() -> dict:
    """Render sidebar controls and return current config values."""
    settings = get_settings()

    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        st.subheader("Retrieval")
        top_k = st.slider(
            "Top-K chunks",
            min_value=1,
            max_value=10,
            value=settings.default_top_k,
            help="Number of document chunks to retrieve per query.",
        )
        threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=settings.similarity_threshold,
            step=0.05,
            help="Minimum relevance score to include a chunk (0 = disabled).",
        )

        st.divider()
        st.subheader("Display")
        show_chunks = st.checkbox("Show retrieved chunks", value=True)
        show_debug = st.checkbox("Show debug panel", value=False)
        show_prompt = st.checkbox("Show prompt preview", value=False)

        st.divider()
        st.subheader("📄 Ingest Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Files are ingested into the knowledge base immediately.",
        )
        if uploaded_files:
            if st.button("Ingest uploaded files", type="primary"):
                _handle_upload_ingestion(uploaded_files)

        st.divider()
        st.subheader("📊 Knowledge Base")
        if st.button("Check status"):
            _show_kb_status()

        st.divider()
        _render_sidebar_info()

    return {
        "top_k": top_k,
        "threshold": threshold,
        "show_chunks": show_chunks,
        "show_debug": show_debug,
        "show_prompt": show_prompt,
    }


def _render_sidebar_info() -> None:
    settings = get_settings()
    with st.expander("ℹ️ System Info", expanded=False):
        st.markdown(
            f"**LLM:** `{settings.chat_model}` ({settings.llm_provider})\n\n"
            f"**Embeddings:** `{settings.embedding_model}`\n\n"
            f"**Vector DB:** `{settings.vector_db}`\n\n"
            f"**Chunk size:** {settings.chunk_size} chars\n\n"
            f"**Overlap:** {settings.chunk_overlap} chars"
        )


def _handle_upload_ingestion(uploaded_files) -> None:
    """Save uploaded files to a temp location, run ingestion, then clean up."""
    import shutil
    import tempfile

    with st.spinner("Ingesting documents…"):
        pipeline = _get_ingestion_pipeline()
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            paths = []
            for uf in uploaded_files:
                dst = tmp_dir / uf.name
                dst.write_bytes(uf.read())
                paths.append(dst)

            stats = pipeline.ingest_directory(
                directory=tmp_dir,
                file_paths=paths,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if stats.files_succeeded > 0:
        st.success(
            f"Ingested {stats.files_succeeded} file(s) → {stats.total_chunks} chunks."
        )
        # Invalidate the cached service so it picks up new documents
        st.cache_resource.clear()
    if stats.files_failed > 0:
        for err in stats.errors:
            st.error(err)


def _show_kb_status() -> None:
    try:
        retriever = VectorStoreRetriever()
        count = retriever.count_documents()
        if count < 0:
            st.error("Could not connect to the vector store. Check your configuration.")
        elif count == 0:
            st.warning("Knowledge base is empty. Ingest documents to get started.")
        else:
            st.success(f"Knowledge base contains **{count}** chunk(s).")
    except Exception as exc:
        st.error(f"Could not connect to vector store: {exc}")


# ─── main content ─────────────────────────────────────────────────────────────


def _render_header() -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🧠 Algorithm RAG Assistant")
        st.caption(
            "Ask questions about algorithms and data structures — answers are grounded "
            "in your uploaded documents."
        )
    with col2:
        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )


def _render_query_input() -> tuple[str, bool]:
    """Render the question input and return (question, submitted)."""
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder=(
                "e.g. Explain how Ford-Fulkerson works.\n"
                "e.g. Give the Java implementation for BFS.\n"
                "e.g. What is the time complexity of Dijkstra's algorithm?"
            ),
            height=100,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    return question.strip(), submitted


def _render_answer(result: AnswerResult, cfg: dict) -> None:
    """Render the answer, citations, and optional debug panels."""
    st.divider()

    if result.is_error and not result.has_context:
        st.error(result.answer)
        return

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.markdown(result.answer)

    # ── Citations ─────────────────────────────────────────────────────────────
    if result.chunks:
        citations = CitationService.build_citations(result.chunks)

        if cfg["show_chunks"]:
            with st.expander(
                f"📚 Retrieved Context ({result.num_chunks} chunk(s) from {len(result.source_filenames)} source(s))",
                expanded=True,
            ):
                for i, cite in enumerate(citations, start=1):
                    with st.container():
                        score_pct = f"{cite.score * 100:.0f}%"
                        st.markdown(
                            f"**[{i}]** `{cite.filename}` — Page {cite.page} "
                            f"<span class='score-badge'>relevance: {score_pct}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"> {cite.chunk_preview}…",
                        )
                        st.markdown("---")

        # Compact source list under answer
        if result.source_filenames:
            st.caption(
                "Sources: " + " | ".join(
                    f"`{fn}`" for fn in result.source_filenames
                )
            )
    else:
        st.info("No relevant context was retrieved from the knowledge base for this query.")

    # ── Debug panel ───────────────────────────────────────────────────────────
    if cfg["show_debug"] and result.chunks:
        with st.expander("🔍 Debug: Retrieved Chunk Scores", expanded=False):
            import pandas as pd

            rows = []
            for chunk in result.chunks:
                rows.append({
                    "Source": chunk.metadata.get("filename", "?"),
                    "Page": chunk.metadata.get("page", "?"),
                    "Score": round(chunk.score, 4),
                    "Chars": chunk.metadata.get("char_count", len(chunk.content)),
                    "ChunkID": chunk.metadata.get("chunk_id", "?")[:12],
                    "Preview": chunk.content[:120].replace("\n", " "),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Prompt preview ────────────────────────────────────────────────────────
    if cfg["show_prompt"] and result.chunks:
        with st.expander("📝 Prompt Preview", expanded=False):
            from src.llm.prompts import format_context_blocks, SYSTEM_PROMPT

            context_str = format_context_blocks(result.chunks)
            st.markdown("**System Prompt:**")
            st.code(SYSTEM_PROMPT, language="markdown")
            st.markdown("**Context Injected:**")
            st.code(context_str, language="markdown")

    # ── Download ──────────────────────────────────────────────────────────────
    md_export = _build_markdown_export(result)
    st.download_button(
        label="⬇️ Download answer as Markdown",
        data=md_export,
        file_name="rag_answer.md",
        mime="text/markdown",
    )


def _build_markdown_export(result: AnswerResult) -> str:
    """Format the answer result as a downloadable Markdown string."""
    lines = [
        f"# RAG Answer\n",
        f"**Question:** {result.question}\n",
        f"---\n",
        result.answer,
        f"\n---\n",
        f"## Sources\n",
    ]
    for chunk in result.chunks:
        meta = chunk.metadata
        lines.append(
            f"- **{meta.get('filename', '?')}** — Page {meta.get('page', '?')} "
            f"(score: {chunk.score:.3f})\n"
        )
    return "\n".join(lines)


# ─── session history ──────────────────────────────────────────────────────────


def _init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def _save_to_history(result: AnswerResult) -> None:
    st.session_state.history.insert(0, result)
    # Keep last 10 interactions
    st.session_state.history = st.session_state.history[:10]


def _render_history() -> None:
    if not st.session_state.get("history"):
        return

    with st.expander("🕐 Recent Questions", expanded=False):
        for item in st.session_state.history:
            st.markdown(f"**Q:** {item.question}")
            st.caption(f"{item.num_chunks} chunk(s) retrieved")
            st.markdown("---")


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    _init_session_state()

    cfg = _render_sidebar()
    _render_header()

    # Example queries
    with st.expander("💡 Example queries", expanded=False):
        examples = [
            "Explain how Ford-Fulkerson works.",
            "What is the Max-Flow Min-Cut theorem?",
            "Give the Java implementation for a Quadtree insertion method.",
            "Compare BFS and DFS using the uploaded lecture notes.",
            "Why does this recurrence solve to O(n log n) according to the notes?",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")

    st.markdown("#### Ask a question")
    question, submitted = _render_query_input()

    if submitted and question:
        service = _get_answer_service(top_k=cfg["top_k"])

        # Apply the UI threshold to the retriever via its public API
        service._retriever.set_threshold(cfg["threshold"])

        with st.spinner("Searching knowledge base and generating answer…"):
            result = service.answer(question, top_k=cfg["top_k"])

        _save_to_history(result)
        _render_answer(result, cfg)

    elif submitted and not question:
        st.warning("Please enter a question before submitting.")

    _render_history()

    # Empty state on first load
    if not submitted and not st.session_state.history:
        st.markdown(
            "<div class='empty-state'>Upload documents in the sidebar, then ask a question above.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
