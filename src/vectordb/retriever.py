"""
Vector store factory and high-level retriever.

`get_vector_store()` returns the correct backend (Chroma / Pinecone) based on
the VECTOR_DB setting.  `VectorStoreRetriever` wraps the store and exposes a
single `retrieve()` method that handles threshold filtering and logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Union

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.embeddings.factory import get_embedding_model
from src.utils.logging import get_logger

log = get_logger(__name__)


# ─── type alias ───────────────────────────────────────────────────────────────

from src.vectordb.chroma_store import ChromaVectorStore
from src.vectordb.pinecone_store import PineconeVectorStore

AnyVectorStore = Union[ChromaVectorStore, PineconeVectorStore]


# ─── factory ──────────────────────────────────────────────────────────────────


def get_vector_store(reset: bool = False) -> AnyVectorStore:
    """
    Instantiate and return the configured vector store.

    Args:
        reset: If True, wipe and recreate the collection/index before returning.

    Raises:
        ValueError: If VECTOR_DB is set to an unsupported value.
    """
    settings = get_settings()
    backend = settings.vector_db
    embedding = get_embedding_model()

    if backend == "chroma":
        store = ChromaVectorStore(embedding_model=embedding)
    elif backend == "pinecone":
        store = PineconeVectorStore(embedding_model=embedding)
    else:
        raise ValueError(f"Unsupported VECTOR_DB value: '{backend}'. Use 'chroma' or 'pinecone'.")

    if reset:
        store.reset_collection()

    return store


# ─── retriever ────────────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its relevance score."""
    document: Document
    score: float

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict:
        return self.document.metadata


class VectorStoreRetriever:
    """
    High-level retriever that wraps a vector store.

    Provides:
      - configurable top_k
      - similarity threshold filtering
      - structured RetrievedChunk output
    """

    def __init__(
        self,
        store: Optional[AnyVectorStore] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._store = store or get_vector_store()
        self._top_k = top_k or settings.default_top_k
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query:   The user's question.
            top_k:   How many chunks to fetch (overrides instance default).
            threshold: Minimum relevance score to include (overrides instance default).
                       Set to 0.0 to disable filtering.

        Returns:
            List of RetrievedChunk objects sorted by relevance (best first).
            May be empty if no chunks meet the threshold.
        """
        k = top_k or self._top_k
        thresh = threshold if threshold is not None else self._threshold

        log.info("Retrieving top-%d chunks for query: '%s'", k, query[:80])

        try:
            raw_results = self._store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            log.error("Vector store retrieval failed: %s", exc)
            raise RuntimeError(f"Retrieval error: {exc}") from exc

        chunks = [RetrievedChunk(document=doc, score=score) for doc, score in raw_results]

        # Filter by threshold
        if thresh > 0.0:
            before = len(chunks)
            chunks = [c for c in chunks if c.score >= thresh]
            log.debug(
                "Threshold filter (%.2f): %d → %d chunks.", thresh, before, len(chunks)
            )

        log.info("Retrieved %d chunk(s) above threshold.", len(chunks))
        return chunks

    def set_threshold(self, threshold: float) -> None:
        """Update the similarity threshold used for filtering retrieved chunks."""
        self._threshold = threshold

    def count_documents(self) -> int:
        """Return the total number of chunks indexed, or -1 on store error."""
        try:
            return self._store.count()
        except Exception as exc:
            log.error("count_documents failed: %s", exc)
            return -1
