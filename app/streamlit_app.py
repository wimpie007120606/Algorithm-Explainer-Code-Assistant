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
    .answer-card {
        background: #f8f9fa;
        border-left: 4px solid #4c8bf5;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .cite-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }
    .score-badge {
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .image-badge {
        background: #fce8ff;
        color: #7b1fa2;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .doc-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.82rem;
    }
    .empty-state {
        color: #888;
        text-align: center;
        padding: 2rem;
        font-style: italic;
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─── cached service initialisation ───────────────────────────────────────────


@st.cache_resource(show_spinner="Connecting to knowledge base…")
def _get_answer_service(top_k: int) -> AnswerService:
    try:
        retriever = VectorStoreRetriever(top_k=top_k)
        return AnswerService(retriever=retriever, top_k=top_k)
    except Exception as exc:
        st.error(f"Failed to initialise the RAG service: {exc}")
        st.stop()


@st.cache_resource(show_spinner="Loading ingestion pipeline…")
def _get_ingestion_pipeline() -> IngestionPipeline:
    from src.vectordb.retriever import get_vector_store
    return IngestionPipeline(vector_store=get_vector_store())


# ─── document library ─────────────────────────────────────────────────────────


def _get_indexed_sources() -> list[dict]:
    """Return a list of dicts describing every indexed source file."""
    try:
        from src.vectordb.retriever import get_vector_store
        store = get_vector_store()
        results = store.underlying_store.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        seen: dict[str, dict] = {}
        for meta in metadatas:
            fname = meta.get("filename", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "file_type": meta.get("file_type", "?"),
                    "chunks": 0,
                }
            seen[fname]["chunks"] += 1
        return sorted(seen.values(), key=lambda x: x["filename"])
    except Exception:
        return []


def _render_document_library() -> None:
    sources = _get_indexed_sources()
    if not sources:
        st.caption("No documents indexed yet.")
        return

    _TYPE_ICON = {
        "pdf": "📄", "text": "📝", "markdown": "📝",
        "image": "🖼️", "?": "📁",
    }
    for s in sources:
        icon = _TYPE_ICON.get(s["file_type"], "📁")
        badge = ""
        if s["file_type"] == "image":
            badge = " <span class='image-badge'>vision</span>"
        st.markdown(
            f"<div class='doc-row'>"
            f"<span>{icon} {s['filename']}{badge}</span>"
            f"<span style='color:#888'>{s['chunks']} chunks</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─── sidebar ──────────────────────────────────────────────────────────────────


def _render_sidebar() -> dict:
    settings = get_settings()

    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        st.subheader("Retrieval")
        top_k = st.slider(
            "Top-K chunks",
            min_value=1, max_value=10,
            value=settings.default_top_k,
            help="Number of document chunks to retrieve per query.",
        )
        threshold = st.slider(
            "Similarity threshold",
            min_value=0.0, max_value=1.0,
            value=settings.similarity_threshold,
            step=0.05,
            help="Minimum relevance score (0 = disabled).",
        )

        st.divider()
        st.subheader("Display")
        show_chunks = st.checkbox("Show retrieved chunks", value=True)
        show_debug = st.checkbox("Show debug panel", value=False)
        show_prompt = st.checkbox("Show prompt preview", value=False)

        st.divider()
        st.subheader("📄 Ingest Documents")
        st.caption("PDFs, text, markdown, and images (PNG/JPG/WEBP)")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg", "webp", "gif", "bmp"],
            accept_multiple_files=True,
            help="Images are described using GPT-4o vision before indexing.",
            label_visibility="collapsed",
        )
        if uploaded_files:
            # Show previews for images before ingesting
            image_files = [f for f in uploaded_files if f.type and f.type.startswith("image/")]
            if image_files:
                with st.expander(f"🖼️ {len(image_files)} image(s) to ingest", expanded=True):
                    cols = st.columns(min(len(image_files), 3))
                    for i, img_file in enumerate(image_files[:3]):
                        with cols[i % 3]:
                            st.image(img_file, caption=img_file.name, use_container_width=True)
                    if len(image_files) > 3:
                        st.caption(f"…and {len(image_files) - 3} more image(s)")

            if st.button("Ingest uploaded files", type="primary", use_container_width=True):
                _handle_upload_ingestion(uploaded_files)

        st.divider()
        st.subheader("📚 Document Library")
        with st.container():
            _render_document_library()
        if st.button("Refresh library", use_container_width=True):
            st.rerun()

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


# ─── ingestion ────────────────────────────────────────────────────────────────


def _handle_upload_ingestion(uploaded_files) -> None:
    """Save uploaded files to temp, ingest each with per-file progress, clean up."""
    import shutil
    import tempfile

    pipeline = _get_ingestion_pipeline()
    tmp_dir = Path(tempfile.mkdtemp())
    n = len(uploaded_files)

    progress_bar = st.progress(0, text="Starting ingestion…")
    status_box = st.empty()

    try:
        paths = []
        for uf in uploaded_files:
            dst = tmp_dir / uf.name
            dst.write_bytes(uf.read())
            paths.append(dst)

        all_stats_list = []
        for i, path in enumerate(paths):
            file_label = path.name
            is_image = path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
            status_box.info(
                f"Processing {i + 1}/{n}: `{file_label}`"
                + (" *(GPT-4o vision…)*" if is_image else "")
            )
            progress_bar.progress((i) / n, text=f"{i}/{n} files processed")

            # Ingest file individually to get per-file feedback
            from src.ingestion.pipeline import IngestionStats
            single_stats = pipeline.ingest_directory(
                directory=tmp_dir,
                file_paths=[path],
            )
            all_stats_list.append((file_label, single_stats))

        progress_bar.progress(1.0, text="Done!")
        status_box.empty()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary
    total_succeeded = sum(s.files_succeeded for _, s in all_stats_list)
    total_failed = sum(s.files_failed for _, s in all_stats_list)
    total_chunks = sum(s.total_chunks for _, s in all_stats_list)
    storage_failures = [s for _, s in all_stats_list if s.storage_failed]

    if storage_failures:
        st.error("Some files were processed but could not be stored in the vector database.")
        for _, s in all_stats_list:
            if s.storage_error:
                st.error(f"Storage error: {s.storage_error}")
    elif total_succeeded > 0:
        st.success(f"✅ Ingested **{total_succeeded}** file(s) → **{total_chunks}** chunks.")
        st.cache_resource.clear()

    for fname, s in all_stats_list:
        if s.files_failed > 0:
            for err in s.errors:
                st.error(f"❌ `{fname}`: {err}")

    if total_failed == 0 and total_succeeded > 0 and not storage_failures:
        st.balloons()


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


# ─── answer rendering ─────────────────────────────────────────────────────────


def _render_header() -> None:
    st.title("🧠 Algorithm RAG Assistant")
    st.caption(
        "Ask questions about algorithms and data structures — answers are grounded "
        "in your uploaded documents, including images described by GPT-4o vision."
    )


def _render_query_input() -> tuple[str, bool]:
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder=(
                "e.g. Explain how Ford-Fulkerson works.\n"
                "e.g. Give the Java implementation for Merge Sort.\n"
                "e.g. What is the time complexity of Dijkstra's algorithm?"
            ),
            height=100,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary")
    return question.strip(), submitted


def _render_answer(result: AnswerResult, cfg: dict) -> None:
    st.divider()

    if result.is_error and not result.has_context:
        st.error(result.answer)
        return

    st.subheader("Answer")
    st.markdown(result.answer)

    if result.chunks:
        citations = CitationService.build_citations(result.chunks)

        if cfg["show_chunks"]:
            with st.expander(
                f"📚 Retrieved Context ({result.num_chunks} chunk(s) from "
                f"{len(result.source_filenames)} source(s))",
                expanded=True,
            ):
                for i, cite in enumerate(citations, start=1):
                    score_pct = f"{cite.score * 100:.0f}%"
                    ftype = ""
                    for chunk in result.chunks:
                        if chunk.metadata.get("filename") == cite.filename:
                            if chunk.metadata.get("file_type") == "image":
                                ftype = " <span class='image-badge'>🖼️ vision</span>"
                            break
                    st.markdown(
                        f"**[{i}]** `{cite.filename}` — Page {cite.page} "
                        f"<span class='score-badge'>relevance: {score_pct}</span>{ftype}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"> {cite.chunk_preview}…")
                    st.markdown("---")

        if result.source_filenames:
            st.caption("Sources: " + " | ".join(f"`{fn}`" for fn in result.source_filenames))
    else:
        st.info("No relevant context was retrieved from the knowledge base for this query.")

    if cfg["show_debug"] and result.chunks:
        with st.expander("🔍 Debug: Chunk Scores", expanded=False):
            import pandas as pd
            rows = [{
                "Source": c.metadata.get("filename", "?"),
                "Page": c.metadata.get("page", "?"),
                "Type": c.metadata.get("file_type", "?"),
                "Score": round(c.score, 4),
                "Chars": c.metadata.get("char_count", len(c.content)),
                "ChunkID": c.metadata.get("chunk_id", "?")[:12],
                "Preview": c.content[:120].replace("\n", " "),
            } for c in result.chunks]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if cfg["show_prompt"] and result.chunks:
        with st.expander("📝 Prompt Preview", expanded=False):
            from src.llm.prompts import format_context_blocks, SYSTEM_PROMPT
            st.markdown("**System Prompt:**")
            st.code(SYSTEM_PROMPT, language="markdown")
            st.markdown("**Context Injected:**")
            st.code(format_context_blocks(result.chunks), language="markdown")

    st.download_button(
        label="⬇️ Download answer as Markdown",
        data=_build_markdown_export(result),
        file_name="rag_answer.md",
        mime="text/markdown",
    )


def _build_markdown_export(result: AnswerResult) -> str:
    lines = [
        f"# RAG Answer\n",
        f"**Question:** {result.question}\n",
        "---\n",
        result.answer,
        "\n---\n",
        "## Sources\n",
    ]
    for chunk in result.chunks:
        meta = chunk.metadata
        lines.append(
            f"- **{meta.get('filename', '?')}** — Page {meta.get('page', '?')} "
            f"(score: {chunk.score:.3f}, type: {meta.get('file_type', '?')})\n"
        )
    return "\n".join(lines)


# ─── session history ──────────────────────────────────────────────────────────


def _init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def _save_to_history(result: AnswerResult) -> None:
    st.session_state.history.insert(0, result)
    st.session_state.history = st.session_state.history[:10]


def _render_history() -> None:
    if not st.session_state.get("history"):
        return
    with st.expander("🕐 Recent Questions", expanded=False):
        for item in st.session_state.history:
            st.markdown(f"**Q:** {item.question}")
            st.caption(f"{item.num_chunks} chunk(s) retrieved | sources: {', '.join(item.source_filenames) or 'none'}")
            st.markdown("---")


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    _init_session_state()
    cfg = _render_sidebar()
    _render_header()

    with st.expander("💡 Example queries", expanded=False):
        for ex in [
            "Explain how Ford-Fulkerson works.",
            "What is the time complexity of Merge Sort vs Quick Sort?",
            "Give the Java implementation for Dijkstra's algorithm.",
            "How does AVL tree rotation work?",
            "Explain the 0/1 Knapsack dynamic programming solution.",
            "What is Kruskal's algorithm and how does Union-Find help?",
            "Compare Bellman-Ford and Dijkstra for negative weights.",
        ]:
            st.markdown(f"- *{ex}*")

    st.markdown("#### Ask a question")
    question, submitted = _render_query_input()

    if submitted and question:
        service = _get_answer_service(top_k=cfg["top_k"])
        service.configure_retrieval(top_k=cfg["top_k"], threshold=cfg["threshold"])

        with st.spinner("Searching knowledge base and generating answer…"):
            result = service.answer(question, top_k=cfg["top_k"])

        _save_to_history(result)
        _render_answer(result, cfg)

    elif submitted and not question:
        st.warning("Please enter a question before submitting.")

    _render_history()

    if not submitted and not st.session_state.history:
        st.markdown(
            "<div class='empty-state'>Upload documents in the sidebar, then ask a question above.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
