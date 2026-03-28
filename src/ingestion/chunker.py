"""
Intelligent document chunking.

Strategy:
  • Default: RecursiveCharacterTextSplitter — splits on paragraph, sentence,
    then word boundaries.  This respects semantic units much better than naive
    fixed-size slicing.

  • Code-aware path: when a page is flagged as code-heavy (by the parser), a
    separator list that respects code block boundaries is used.

  • All metadata is propagated to every child chunk, plus:
      - chunk_index: global 0-based index across all input documents (ordering key)
      - chunk_id:    deterministic ID derived from filename+page+local position (see utils/ids.py)
      - char_count:  character length of the chunk

Configuration is driven by settings (chunk_size, chunk_overlap), but callers
can override per-call for experimentation.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import get_settings
from src.utils.ids import make_chunk_id
from src.utils.logging import get_logger

log = get_logger(__name__)

# Separators used for general prose (algorithm explanations, definitions, proofs)
_PROSE_SEPARATORS = [
    "\n\n",   # paragraph break
    "\n",     # line break
    ". ",     # sentence end
    ", ",     # clause
    " ",      # word
    "",       # character fallback
]

# Separators used for code-heavy pages (preserve indentation blocks)
_CODE_SEPARATORS = [
    "\n\n",
    "\n",
    "; ",
    " ",
    "",
]


def _make_splitter(
    chunk_size: int,
    chunk_overlap: int,
    code_mode: bool = False,
) -> RecursiveCharacterTextSplitter:
    separators = _CODE_SEPARATORS if code_mode else _PROSE_SEPARATORS
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    docs: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split a list of Documents into retrieval-ready chunks.

    Args:
        docs:          Parsed Documents (output of parser.parse_documents).
        chunk_size:    Override the configured chunk size (characters).
        chunk_overlap: Override the configured overlap.

    Returns:
        Flat list of chunk Documents, each with full metadata.
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    prose_splitter = _make_splitter(size, overlap, code_mode=False)
    code_splitter = _make_splitter(size, overlap, code_mode=True)

    all_chunks: List[Document] = []
    chunk_index_counter = 0  # global counter across all docs

    for doc in docs:
        is_code_heavy = doc.metadata.get("is_code_heavy", False)
        splitter = code_splitter if is_code_heavy else prose_splitter

        raw_chunks = splitter.split_documents([doc])

        for local_idx, chunk in enumerate(raw_chunks):
            source = chunk.metadata.get("filename", "unknown")
            page = chunk.metadata.get("page", 0)

            chunk.metadata["chunk_index"] = chunk_index_counter
            chunk.metadata["chunk_id"] = make_chunk_id(source, page, local_idx)
            chunk.metadata["char_count"] = len(chunk.page_content)

            all_chunks.append(chunk)
            chunk_index_counter += 1

    log.info(
        "Chunked %d document(s) → %d chunks (size=%d, overlap=%d).",
        len(docs),
        len(all_chunks),
        size,
        overlap,
    )
    return all_chunks
