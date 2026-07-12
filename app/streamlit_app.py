"""
StudyMate Knowledge Coach — Streamlit UI

Entry point:
    streamlit run app/streamlit_app.py

Architecture:
    This module handles only UI concerns.  All business logic lives in src/.
    The app bootstraps the RAG services on first load (cached via st.cache_resource),
    then routes user interactions to AnswerService and CitationService.
"""

from __future__ import annotations

# Streamlit requires page config before the rest of the app imports.
# ruff: noqa: E402, I001

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running via `streamlit run app/streamlit_app.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# ─── page config must be the very first Streamlit call ────────────────────────
st.set_page_config(
    page_title="StudyMate Knowledge Coach",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _bootstrap_streamlit_secrets() -> None:
    """Expose top-level Streamlit secrets as env vars for the shared settings layer."""
    try:
        secret_values = st.secrets.to_dict()
    except Exception:
        return

    for key, value in secret_values.items():
        if not isinstance(key, str) or not key.isupper():
            continue
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))


_bootstrap_streamlit_secrets()

# ─── imports after page_config ────────────────────────────────────────────────
from src.config.settings import get_settings, missing_secret_message
from src.ingestion.pipeline import IngestionPipeline
from src.services.answer_service import AnswerResult, AnswerService
from src.services.citation_service import CitationService
from src.utils.files import iter_documents
from src.utils.logging import get_logger
from src.vectordb.retriever import VectorStoreRetriever

log = get_logger(__name__)

# ─── CSS / styling ────────────────────────────────────────────────────────────

_CSS = """
<style>
    :root {
        --study-bg: #080a0f;
        --study-panel: #10131b;
        --study-panel-2: #151923;
        --study-text: #f6f7fb;
        --study-muted: #a8afbd;
        --study-dim: #7d8594;
        --study-line: #303642;
        --study-line-soft: #232833;
        --study-accent: #f5f5f7;
    }
    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--study-bg) !important;
        color: var(--study-text) !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1180px;
    }
    section[data-testid="stSidebar"] {
        background: var(--study-panel) !important;
        border-right: 1px solid var(--study-line-soft);
    }
    section[data-testid="stSidebar"] * {
        color: var(--study-text);
    }
    .study-hero {
        border-top: 1px solid var(--study-line);
        border-bottom: 1px solid var(--study-line);
        padding: 1.35rem 0 1.25rem;
        margin-bottom: 1.1rem;
        display: grid;
        grid-template-columns: minmax(0, 1.05fr) minmax(320px, .95fr);
        gap: 2rem;
        align-items: center;
    }
    .study-hero h1 {
        margin: 0 0 .35rem 0;
        font-size: 2.35rem;
        line-height: 1.1;
        letter-spacing: 0;
        color: var(--study-text);
    }
    .study-hero p {
        color: var(--study-muted);
        margin: 0;
        max-width: 780px;
        font-size: 1rem;
    }
    .hero-kicker {
        color: var(--study-dim);
        font-size: .76rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }
    .hero-command {
        border-left: 1px solid var(--study-line);
        color: var(--study-muted);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: .84rem;
        margin-top: 1rem;
        padding-left: .9rem;
        line-height: 1.65;
    }
    .code-visual {
        position: relative;
        min-height: 245px;
        border-top: 1px solid var(--study-line-soft);
        border-bottom: 1px solid var(--study-line-soft);
        overflow: hidden;
    }
    .code-visual::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(var(--study-line-soft) 1px, transparent 1px),
            linear-gradient(90deg, var(--study-line-soft) 1px, transparent 1px);
        background-size: 44px 44px;
        opacity: .42;
    }
    .code-visual::after {
        content: "∇";
        position: absolute;
        right: 1rem;
        top: .2rem;
        color: rgba(246, 247, 251, .09);
        font-size: 8rem;
        line-height: 1;
        font-family: Georgia, serif;
    }
    .code-node {
        position: absolute;
        z-index: 1;
        border: 1px solid var(--study-line);
        background: rgba(8, 10, 15, .78);
        color: var(--study-text);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: .78rem;
        padding: .42rem .58rem;
        min-width: 64px;
        text-align: center;
    }
    .code-node.python { left: 6%; top: 12%; }
    .code-node.sql { right: 8%; top: 18%; }
    .code-node.r { left: 20%; bottom: 13%; }
    .code-node.cpp { right: 27%; bottom: 9%; }
    .code-node.stats { left: 43%; top: 42%; }
    .code-line {
        position: absolute;
        z-index: 1;
        color: var(--study-muted);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: .76rem;
        white-space: nowrap;
    }
    .code-line.one { left: 6%; top: 39%; }
    .code-line.two { right: 6%; top: 56%; }
    .code-line.three { left: 32%; top: 72%; }
    .data-stack {
        border-top: 1px solid var(--study-line-soft);
        border-bottom: 1px solid var(--study-line-soft);
        margin: 1rem 0 1.3rem;
        padding: .9rem 0;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0;
    }
    .stack-item {
        border-right: 1px solid var(--study-line-soft);
        padding: 0 1rem;
    }
    .stack-item:last-child {
        border-right: 0;
    }
    .stack-label {
        color: var(--study-dim);
        font-size: .72rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }
    .stack-text {
        color: var(--study-text);
        font-size: .9rem;
        line-height: 1.35;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0;
        margin: .9rem 0 1.2rem;
        border-top: 1px solid var(--study-line-soft);
        border-bottom: 1px solid var(--study-line-soft);
    }
    .metric-tile {
        background: transparent;
        border-right: 1px solid var(--study-line-soft);
        padding: .9rem 1rem;
        min-height: 82px;
    }
    .metric-tile:last-child {
        border-right: 0;
    }
    .metric-label {
        color: var(--study-dim);
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: .25rem;
    }
    .metric-value {
        color: var(--study-text);
        font-size: 1.55rem;
        line-height: 1.1;
        font-weight: 720;
    }
    .metric-note {
        color: var(--study-muted);
        font-size: .82rem;
        margin-top: .28rem;
    }
    h1, h2, h3, h4, h5, h6,
    p, li, label, span, div {
        color: inherit;
    }
    .stMarkdown,
    .stMarkdown p,
    [data-testid="stMarkdownContainer"] {
        color: var(--study-text);
    }
    .stCaptionContainer,
    [data-testid="stCaptionContainer"] {
        color: var(--study-muted) !important;
    }
    .answer-card {
        background: transparent;
        border-left: 1px solid var(--study-line);
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .cite-card {
        background: transparent;
        border: 1px solid var(--study-line-soft);
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }
    .score-badge {
        background: transparent;
        color: var(--study-text);
        border: 1px solid var(--study-line);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .quality-badge {
        background: transparent;
        color: var(--study-muted);
        border: 1px solid var(--study-line);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .image-badge {
        background: transparent;
        color: var(--study-text);
        border: 1px solid var(--study-line);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .doc-row {
        padding: .55rem 0;
        border-bottom: 1px solid var(--study-line-soft);
        font-size: 0.82rem;
    }
    .doc-title {
        display: block;
        font-weight: 650;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .doc-meta {
        color: var(--study-muted);
        display: flex;
        gap: .45rem;
        flex-wrap: wrap;
        margin-top: .18rem;
    }
    .empty-state {
        color: var(--study-muted);
        text-align: center;
        padding: 2rem;
        font-style: italic;
    }
    .study-tip {
        border-left: 1px solid var(--study-line);
        background: transparent;
        padding: .85rem 1rem;
        color: var(--study-muted);
        margin: .8rem 0;
    }
    div[data-testid="stForm"],
    div[data-testid="stExpander"],
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        border-color: var(--study-line-soft) !important;
        color: var(--study-text) !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea,
    input {
        background: var(--study-panel-2) !important;
        color: var(--study-text) !important;
        border-color: var(--study-line-soft) !important;
    }
    textarea::placeholder,
    input::placeholder {
        color: var(--study-dim) !important;
        opacity: 1 !important;
    }
    button,
    .stButton button,
    .stDownloadButton button,
    [data-testid="stFormSubmitButton"] button {
        background: transparent !important;
        color: var(--study-text) !important;
        border: 1px solid var(--study-line) !important;
        box-shadow: none !important;
    }
    button:hover,
    .stButton button:hover,
    .stDownloadButton button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--study-text) !important;
        color: var(--study-text) !important;
    }
    div[data-testid="stAlert"] {
        background: transparent !important;
        border: 1px solid var(--study-line-soft) !important;
        color: var(--study-text) !important;
    }
    div[data-testid="stAlert"] * {
        color: var(--study-text) !important;
    }
    hr {
        border-color: var(--study-line-soft) !important;
    }
    @media (max-width: 900px) {
        .study-hero {
            grid-template-columns: 1fr;
        }
        .code-visual {
            min-height: 210px;
        }
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .metric-tile:nth-child(2) {
            border-right: 0;
        }
        .metric-tile:nth-child(-n+2) {
            border-bottom: 1px solid var(--study-line-soft);
        }
        .study-hero h1 {
            font-size: 1.65rem;
        }
        .data-stack {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .stack-item:nth-child(2) {
            border-right: 0;
        }
        .stack-item:nth-child(-n+2) {
            padding-bottom: .8rem;
            margin-bottom: .8rem;
            border-bottom: 1px solid var(--study-line-soft);
        }
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
    try:
        from src.vectordb.retriever import get_vector_store

        return IngestionPipeline(vector_store=get_vector_store())
    except Exception as exc:
        st.error(f"Failed to initialise the ingestion pipeline: {exc}")
        st.stop()


# ─── document library ─────────────────────────────────────────────────────────


def _get_indexed_sources() -> list[dict]:
    """Return a list of dicts describing every indexed source file."""
    try:
        from src.vectordb.retriever import get_vector_store
        store = get_vector_store()
        results = store.underlying_store.get(include=["metadatas", "documents"])
        metadatas = results.get("metadatas") or []
        documents = results.get("documents") or []
        seen: dict[str, dict] = {}
        for meta, content in zip(metadatas, documents):
            fname = meta.get("filename", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "file_type": meta.get("file_type", "?"),
                    "chunks": 0,
                    "pages": set(),
                    "chars": 0,
                    "vision_chunks": 0,
                    "low_text_chunks": 0,
                }
            seen[fname]["chunks"] += 1
            seen[fname]["pages"].add(meta.get("page", "?"))
            char_count = meta.get("char_count", len(content or ""))
            seen[fname]["chars"] += int(char_count or 0)
            if meta.get("vision_described"):
                seen[fname]["vision_chunks"] += 1
            if int(char_count or 0) < 80:
                seen[fname]["low_text_chunks"] += 1

        sources = []
        for source in seen.values():
            chunks = max(source["chunks"], 1)
            source["pages"] = len(source["pages"])
            source["avg_chars"] = round(source["chars"] / chunks)
            source["needs_ocr"] = (
                source["file_type"] == "pdf"
                and source["low_text_chunks"] >= max(1, source["chunks"] // 2)
            )
            sources.append(source)

        return sorted(sources, key=lambda x: x["filename"])
    except Exception:
        return []


def _get_library_metrics() -> dict:
    sources = _get_indexed_sources()
    chunks = sum(s["chunks"] for s in sources)
    pages = sum(s["pages"] for s in sources)
    vision_sources = sum(1 for s in sources if s["vision_chunks"])
    needs_ocr = sum(1 for s in sources if s["needs_ocr"])
    return {
        "sources": len(sources),
        "chunks": chunks,
        "pages": pages,
        "vision_sources": vision_sources,
        "needs_ocr": needs_ocr,
        "source_rows": sources,
    }


def _render_document_library() -> None:
    sources = _get_indexed_sources()
    if not sources:
        st.caption("No documents indexed yet.")
        return

    type_icon = {
        "pdf": "📄", "text": "📝", "markdown": "📝",
        "image": "🖼️", "?": "📁",
    }
    for s in sources:
        icon = type_icon.get(s["file_type"], "📁")
        badge = ""
        if s["file_type"] == "image":
            badge = " <span class='image-badge'>vision</span>"
        elif s["vision_chunks"]:
            badge = " <span class='image-badge'>vision OCR</span>"
        if s["needs_ocr"]:
            badge += " <span class='quality-badge'>low text</span>"
        st.markdown(
            f"<div class='doc-row'>"
            f"<span class='doc-title'>{icon} {s['filename']}{badge}</span>"
            f"<span class='doc-meta'>"
            f"<span>{s['chunks']} chunks</span>"
            f"<span>{s['pages']} page(s)</span>"
            f"<span>{s['avg_chars']} avg chars</span>"
            f"</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─── sidebar ──────────────────────────────────────────────────────────────────


def _render_sidebar() -> dict:
    settings = get_settings()
    metrics = _get_library_metrics()
    sources = metrics["source_rows"]
    image_upload_enabled = bool(settings.openai_api_key)
    upload_types = ["pdf", "txt", "md"]
    if image_upload_enabled:
        upload_types.extend(["png", "jpg", "jpeg", "webp", "gif", "bmp"])

    with st.sidebar:
        st.title("Study Console")
        _render_startup_checks()
        st.divider()

        st.subheader("Study Mode")
        track_labels = {
            "data_science": "Data Science Core",
            "calculus": "Calculus",
            "math_stats": "Math Stats",
            "computer_science": "Computer Science",
            "python_ds": "Python / Data Science",
            "general": "General Study",
        }
        track_label = st.selectbox(
            "Learning track",
            options=list(track_labels.values()),
            index=0,
            help="Frames practice, explanations, and plans around a study pathway.",
        )
        learning_track = next(
            key for key, label in track_labels.items() if label == track_label
        )

        study_mode_labels = {
            "answer": "Answer",
            "explain": "Explain",
            "summary": "Summarize",
            "practice": "Practice",
            "flashcards": "Flashcards",
            "study_plan": "Study Plan",
        }
        study_mode_label = st.selectbox(
            "Output style",
            options=list(study_mode_labels.values()),
            index=0,
            help="Changes how the grounded answer is structured.",
            label_visibility="collapsed",
        )
        study_mode = next(
            key for key, label in study_mode_labels.items() if label == study_mode_label
        )

        source_options = ["All indexed sources"] + [s["filename"] for s in sources]
        selected_source = st.selectbox(
            "Focus source",
            options=source_options,
            index=0,
            help="Restrict retrieval to one document when revising a specific paper or chapter.",
        )
        source_filter = None if selected_source == "All indexed sources" else selected_source

        learner_goal = st.text_input(
            "Goal",
            placeholder="e.g. Prepare for Monday's calculus test",
            help="Optional context used when creating plans or practice.",
        )
        study_minutes = st.number_input(
            "Available minutes",
            min_value=5,
            max_value=360,
            value=45,
            step=5,
            help="Used to frame study plans and revision pacing.",
        )

        st.divider()
        st.subheader("Retrieval")
        top_k = st.slider(
            "Context chunks",
            min_value=1, max_value=16,
            value=min(max(settings.default_top_k, 6), 16),
            help="Number of final chunks sent to the model after reranking.",
        )
        threshold = st.slider(
            "Similarity threshold",
            min_value=0.0, max_value=1.0,
            value=settings.similarity_threshold,
            step=0.05,
            help="Minimum combined relevance score. Set to 0.0 while diagnosing missing context.",
        )

        st.divider()
        st.subheader("Display")
        show_chunks = st.checkbox("Show retrieved chunks", value=True)
        show_debug = st.checkbox("Show debug panel", value=False)
        show_prompt = st.checkbox("Show prompt preview", value=False)

        st.divider()
        st.subheader("Add Material")
        if image_upload_enabled:
            st.caption("Upload calculus, math stats, CS, Python, and data science textbooks. Text-layer PDFs work best; scanned pages can use vision fallback.")
        else:
            st.caption("PDFs, text, and markdown. Vision ingestion unlocks after adding OPENAI_API_KEY.")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=upload_types,
            accept_multiple_files=True,
            help="Images and scanned pages are transcribed with GPT-4o vision when available.",
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

            if st.button(
                "Ingest uploaded files",
                type="primary",
                use_container_width=True,
                disabled=not settings.is_ready_for_rag(),
            ):
                _handle_upload_ingestion(uploaded_files)

        bundled_docs = _get_bundled_sample_paths()
        if bundled_docs:
            st.caption(f"{len(bundled_docs)} bundled sample document(s) available in the repo.")
            if st.button(
                "Load bundled sample docs",
                use_container_width=True,
                disabled=not settings.is_ready_for_rag(),
            ):
                _handle_sample_ingestion(bundled_docs)

        st.divider()
        st.subheader("Knowledge Library")
        with st.container():
            _render_document_library()
        if metrics["needs_ocr"]:
            st.warning(
                f"{metrics['needs_ocr']} source(s) look low-text. Re-ingest with OCR/vision before relying on them."
            )
        if st.button("Refresh library", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        _render_sidebar_info()

    return {
        "top_k": top_k,
        "threshold": threshold,
        "study_mode": study_mode,
        "learning_track": learning_track,
        "source_filter": source_filter,
        "learner_goal": learner_goal.strip(),
        "study_minutes": study_minutes,
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
            f"**Chroma Dir:** `{settings.resolved_chroma_dir()}`\n\n"
            f"**Candidate rerank:** {settings.retrieval_candidate_multiplier}× up to "
            f"{settings.retrieval_max_candidates}\n\n"
            f"**Chunk size:** {settings.chunk_size} chars\n\n"
            f"**Overlap:** {settings.chunk_overlap} chars\n\n"
            f"**PDF vision fallback:** `{settings.pdf_vision_fallback}`"
        )


def _render_startup_checks() -> None:
    settings = get_settings()

    if not settings.has_required_llm_api_key():
        st.warning(
            f"{missing_secret_message(settings.required_llm_secret_name())} "
            "Question answering and ingestion stay disabled until you add it."
        )

    if settings.vector_db == "pinecone" and not settings.pinecone_api_key:
        st.warning(
            f"{missing_secret_message('PINECONE_API_KEY')} "
            "Switch VECTOR_DB to `chroma` or add Pinecone secrets."
        )

    if settings.is_streamlit_cloud() and settings.vector_db == "chroma":
        st.info(
            "This deployment uses local Chroma storage in `/tmp`, which is ephemeral on "
            "Streamlit Community Cloud. Uploaded or bundled documents can disappear after "
            "app restarts and may need to be re-ingested."
        )

    if settings.openai_api_key and settings.pdf_vision_fallback:
        import importlib.util

        if importlib.util.find_spec("fitz") is None:
            st.caption(
                "PDF vision fallback is enabled, but PyMuPDF is not installed. "
                "Install `PyMuPDF` before re-ingesting scanned PDFs."
            )

    if not settings.openai_api_key:
        st.caption(
            "Image and scanned-page ingestion require `OPENAI_API_KEY` because visual "
            "content is described with GPT-4o vision."
        )


def _get_bundled_sample_paths() -> list[Path]:
    try:
        raw_dir = get_settings().resolved_raw_dir()
        return [path for path in iter_documents(raw_dir, recursive=True) if path.name != ".gitkeep"]
    except FileNotFoundError:
        return []


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
            is_pdf = path.suffix.lower() == ".pdf"
            status_box.info(
                f"Processing {i + 1}/{n}: `{file_label}`"
                + (" *(GPT-4o vision…)*" if is_image else "")
                + (" *(text extraction + OCR fallback check…)*" if is_pdf else "")
            )
            progress_bar.progress((i) / n, text=f"{i}/{n} files processed")

            # Ingest file individually to get per-file feedback
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


def _handle_sample_ingestion(file_paths: list[Path]) -> None:
    pipeline = _get_ingestion_pipeline()
    with st.spinner("Ingesting bundled sample documents…"):
        stats = pipeline.ingest_directory(
            directory=file_paths[0].parent,
            file_paths=file_paths,
        )

    if stats.storage_failed:
        st.error(f"Bundled sample ingestion failed during storage: {stats.storage_error}")
        return

    if stats.files_failed > 0:
        st.error("Some bundled sample documents failed to ingest.")
        for err in stats.errors:
            st.error(err)
        return

    st.success(f"Loaded **{stats.files_succeeded}** bundled sample document(s).")
    st.cache_resource.clear()
    st.rerun()


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
    metrics = _get_library_metrics()
    st.markdown(
        """
        <div class="study-hero">
            <div>
                <div class="hero-kicker">Data Science Mastery Library</div>
                <h1>StudyMate Knowledge Coach</h1>
                <p>Turn calculus, mathematical statistics, computer science, and Python data science textbooks into a grounded study system for explanations, practice, flashcards, summaries, and timed mastery plans.</p>
                <div class="hero-command">
                    ingest(textbooks) → retrieve(the right pages) → explain with citations<br>
                    calculus + stats + CS + Python → data science fluency
                </div>
            </div>
            <div class="code-visual" aria-hidden="true">
                <div class="code-node python">Python</div>
                <div class="code-node sql">SQL</div>
                <div class="code-node r">R</div>
                <div class="code-node cpp">C++</div>
                <div class="code-node stats">Bayes</div>
                <div class="code-line one">θ ← θ - α∇J(θ)</div>
                <div class="code-line two">SELECT model, AVG(loss)</div>
                <div class="code-line three">p(θ | data) ∝ p(data | θ)p(θ)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="data-stack">
            <div class="stack-item">
                <div class="stack-label">Calculus</div>
                <div class="stack-text">limits, derivatives, integrals, optimization</div>
            </div>
            <div class="stack-item">
                <div class="stack-label">Math Stats</div>
                <div class="stack-text">probability, inference, estimators, distributions</div>
            </div>
            <div class="stack-item">
                <div class="stack-label">Computer Science</div>
                <div class="stack-text">algorithms, data structures, complexity, systems</div>
            </div>
            <div class="stack-item">
                <div class="stack-label">Python DS</div>
                <div class="stack-text">NumPy, pandas, modeling, analysis workflows</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="metric-label">Sources</div>
                <div class="metric-value">{metrics['sources']}</div>
                <div class="metric-note">indexed files</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Context</div>
                <div class="metric-value">{metrics['chunks']}</div>
                <div class="metric-note">searchable chunks</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Coverage</div>
                <div class="metric-value">{metrics['pages']}</div>
                <div class="metric-note">page references</div>
            </div>
            <div class="metric-tile">
                <div class="metric-label">Health</div>
                <div class="metric-value">{metrics['needs_ocr']}</div>
                <div class="metric-note">low-text source(s)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_query_input(can_submit: bool, cfg: dict) -> tuple[str, bool]:
    placeholders = {
        "answer": "Ask across your library, e.g. How do derivatives connect to gradient descent in Python?",
        "explain": "e.g. Teach me maximum likelihood using calculus and math stats from my books.",
        "summary": "e.g. Summarize the probability and calculus ideas I need for machine learning.",
        "practice": "e.g. Create mixed calculus, stats, CS, and Python questions for a data science student.",
        "flashcards": "e.g. Make flashcards for distributions, derivatives, algorithms, and pandas concepts.",
        "study_plan": "e.g. Build a 2-hour study plan to master calculus foundations for data science.",
    }
    placeholder = placeholders.get(cfg["study_mode"], placeholders["answer"])
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Study request",
            placeholder=placeholder,
            height=120,
            label_visibility="collapsed",
        )
        button_label = {
            "answer": "Ask",
            "explain": "Teach Me",
            "summary": "Summarize",
            "practice": "Generate Practice",
            "flashcards": "Make Flashcards",
            "study_plan": "Build Plan",
        }.get(cfg["study_mode"], "Ask")
        submitted = st.form_submit_button(button_label, type="primary", disabled=not can_submit)
    return question.strip(), submitted


def _render_answer(result: AnswerResult, cfg: dict) -> None:
    st.divider()

    if result.is_error and not result.has_context:
        st.error(result.answer)
        return

    st.subheader("Answer")
    st.markdown(result.answer)

    if result.chunks:
        low_text_hits = [
            c for c in result.chunks
            if c.metadata.get("file_type") == "pdf" and len(c.content.strip()) < 80
        ]
        if low_text_hits and len(low_text_hits) == len(result.chunks):
            st.warning(
                "The retrieved PDF chunks are extremely short. This usually means the PDF "
                "needs OCR or vision-based re-ingestion before the assistant can study from it."
            )

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
        st.markdown(
            """
            <div class="study-tip">
                Try lowering the similarity threshold to 0.0, selecting the exact source in the sidebar,
                or re-ingesting scanned/math-heavy PDFs with OCR or vision fallback. If a source is marked
                low text, the vector database may only contain page labels rather than the real content.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if cfg["show_debug"] and result.chunks:
        with st.expander("🔍 Debug: Chunk Scores", expanded=False):
            import pandas as pd
            rows = [{
                "Source": c.metadata.get("filename", "?"),
                "Page": c.metadata.get("page", "?"),
                "Type": c.metadata.get("file_type", "?"),
                "Combined": round(c.score, 4),
                "Vector": round(c.metadata.get("vector_score", c.score), 4),
                "Lexical": round(c.metadata.get("lexical_score", 0.0), 4),
                "Chars": c.metadata.get("char_count", len(c.content)),
                "ChunkID": c.metadata.get("chunk_id", "?")[:12],
                "Preview": c.content[:120].replace("\n", " "),
            } for c in result.chunks]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if cfg["show_prompt"] and result.chunks:
        with st.expander("📝 Prompt Preview", expanded=False):
            from src.llm.prompts import SYSTEM_PROMPT, format_context_blocks
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
        "# StudyMate Answer\n",
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
    settings = get_settings()
    _render_header()

    with st.expander("Example study requests", expanded=False):
        for ex in [
            "Teach me how derivatives, optimization, and gradient descent connect.",
            "Create mixed practice from calculus, math stats, CS, and Python data science.",
            "Explain maximum likelihood using only my uploaded statistics textbook.",
            "Build a 2-hour study plan for becoming strong at data science foundations.",
            "Make flashcards for distributions, integrals, algorithms, and pandas workflows.",
            "Compare the textbook explanations of variance, covariance, and correlation.",
            "What Python concepts should I master before machine learning?",
        ]:
            st.markdown(f"- *{ex}*")

    st.markdown("#### What do you want to accomplish?")
    question, submitted = _render_query_input(settings.is_ready_for_rag(), cfg)

    if submitted and question:
        service = _get_answer_service(top_k=cfg["top_k"])
        service.configure_retrieval(top_k=cfg["top_k"], threshold=cfg["threshold"])

        question_for_model = question
        track_context = {
            "data_science": (
                "Learning track: Data Science Core. Connect calculus, mathematical "
                "statistics, computer science, Python, SQL, algorithms, and modeling "
                "when the uploaded sources support those links."
            ),
            "calculus": "Learning track: Calculus mastery. Emphasize definitions, intuition, worked steps, and practice.",
            "math_stats": "Learning track: Mathematical statistics. Emphasize probability, inference, estimators, and distributions.",
            "computer_science": "Learning track: Computer science. Emphasize algorithms, data structures, complexity, and implementation ideas.",
            "python_ds": "Learning track: Python for data science. Emphasize Python, NumPy, pandas, analysis, and modeling workflows.",
            "general": "Learning track: General study. Keep the answer focused on the uploaded material.",
        }
        if cfg["learner_goal"] or cfg["study_mode"] == "study_plan" or cfg["learning_track"]:
            context_bits = [
                track_context.get(cfg["learning_track"], track_context["general"]),
                f"Available study time: {cfg['study_minutes']} minutes.",
            ]
            if cfg["learner_goal"]:
                context_bits.append(f"Learner goal: {cfg['learner_goal']}.")
            question_for_model = "\n".join(context_bits + [f"Study request: {question}"])

        with st.spinner("Searching knowledge base and generating answer…"):
            result = service.answer_study(
                question_for_model,
                top_k=cfg["top_k"],
                study_mode=cfg["study_mode"],
                source_filter=cfg["source_filter"],
            )
            error_text = f"{result.error or ''} {result.answer}".lower()
            stale_vector_store = (
                "vector store unavailable" in error_text
                or ("collection" in error_text and "does not exist" in error_text)
            )
            if stale_vector_store:
                st.cache_resource.clear()
                service = _get_answer_service(top_k=cfg["top_k"])
                service.configure_retrieval(top_k=cfg["top_k"], threshold=cfg["threshold"])
                result = service.answer_study(
                    question_for_model,
                    top_k=cfg["top_k"],
                    study_mode=cfg["study_mode"],
                    source_filter=cfg["source_filter"],
                )
            result.question = question

        _save_to_history(result)
        _render_answer(result, cfg)

    elif submitted and not question:
        st.warning("Please enter a question before submitting.")

    _render_history()

    if not submitted and not st.session_state.history:
        st.markdown(
            "<div class='empty-state'>Upload readable material in the sidebar, choose a study mode, then ask for answers, practice, flashcards, summaries, or a plan.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
