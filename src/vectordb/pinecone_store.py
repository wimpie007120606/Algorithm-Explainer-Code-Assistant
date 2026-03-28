"""
Pinecone vector store adapter (optional).

Enable by setting VECTOR_DB=pinecone and supplying PINECONE_API_KEY and
PINECONE_INDEX_NAME in your .env.

Install the extra dependency:
    pip install "pinecone-client>=3.0.0" langchain-pinecone

This adapter mirrors the ChromaVectorStore interface so callers can swap
backends transparently via the factory in retriever.py.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class PineconeVectorStore:
    """
    Pinecone-backed vector store with the same interface as ChromaVectorStore.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        index_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._embedding = embedding_model
        self._index_name = index_name or settings.pinecone_index_name

        if not settings.pinecone_api_key:
            raise EnvironmentError(
                "PINECONE_API_KEY is not set. Cannot initialise Pinecone."
            )

        try:
            from pinecone import Pinecone
            from langchain_pinecone import PineconeVectorStore as _LCPinecone
        except ImportError as exc:
            raise ImportError(
                "pinecone-client and langchain-pinecone are required: "
                "pip install 'pinecone-client>=3.0.0' langchain-pinecone"
            ) from exc

        log.info("Connecting to Pinecone index '%s'.", self._index_name)
        Pinecone(api_key=settings.pinecone_api_key)  # validates key on init

        # PineconeVectorStore.from_existing_index connects to an existing index
        self._store = _LCPinecone.from_existing_index(
            index_name=self._index_name,
            embedding=self._embedding,
        )
        log.info("Pinecone store ready.")

    # ── public interface ───────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(docs)]
        log.info("Upserting %d chunks into Pinecone index '%s'.", len(docs), self._index_name)
        self._store.add_documents(docs, ids=ids)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[tuple[Document, float]]:
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def as_retriever(self, k: int = 4):
        return self._store.as_retriever(search_kwargs={"k": k})

    def reset_collection(self) -> None:
        log.warning("Pinecone index reset is not automatically managed; delete via the Pinecone console.")

    def count(self) -> int:
        """Return total vector count from Pinecone index stats, or -1 on error."""
        try:
            stats = self._store._index.describe_index_stats()
            return stats.get("total_vector_count", -1)
        except Exception as exc:
            log.error("Pinecone count() failed: %s", exc)
            return -1
