"""
ChromaDB vector store adapter.

Wraps LangChain's Chroma integration with:
  - persistent storage by default
  - idempotent add_documents (upsert via deterministic chunk IDs)
  - collection reset for full rebuilds
  - clean initialisation that surfaces errors early
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class ChromaVectorStore:
    """
    Thin wrapper around LangChain's Chroma that adds:
      - clear factory / reset methods
      - metadata-aware upsert using chunk_id
      - informative logging
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._embedding = embedding_model
        self._persist_dir = persist_dir or settings.resolved_chroma_dir()
        self._collection_name = collection_name or settings.chroma_collection_name

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Initialising ChromaDB: collection='%s' persist_dir='%s'",
            self._collection_name,
            self._persist_dir,
        )

        self._store = self._make_chroma_store()

    def _make_chroma_store(self) -> Chroma:
        """Construct the Chroma store with a clamped relevance score function."""

        def _clamped_relevance(distance: float) -> float:
            # Clamp the L2-derived relevance to [0, 1] to avoid spurious warnings
            # when querying against unrelated documents.
            score = 1.0 - distance / 2.0
            return max(0.0, min(1.0, score))

        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embedding,
            persist_directory=str(self._persist_dir),
            relevance_score_fn=_clamped_relevance,
        )

    # ── public interface ───────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add (upsert) a list of chunk Documents into the collection.

        Uses the deterministic `chunk_id` in metadata as the Chroma document ID
        so that re-ingesting the same document does not create duplicates.
        """
        if not docs:
            log.warning("add_documents called with empty list — nothing to do.")
            return

        ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(docs)]

        log.info("Upserting %d chunks into Chroma collection '%s'.", len(docs), self._collection_name)
        self._store.add_documents(documents=docs, ids=ids)
        log.info("Upsert complete.")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[tuple[Document, float]]:
        """
        Return the top-*k* most similar chunks with their similarity scores.

        Scores are L2 distances (lower = more similar for Chroma's default).
        We convert them to a [0, 1] cosine-like score so callers get a uniform
        interface regardless of backend.

        Returns:
            List of (Document, score) tuples sorted by relevance descending.
        """
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        # LangChain returns (doc, relevance_score) where higher = more relevant
        return results

    def as_retriever(self, k: int = 4):
        """Return a LangChain retriever interface for use in chains."""
        return self._store.as_retriever(search_kwargs={"k": k})

    def reset_collection(self) -> None:
        """Delete all documents from the collection (full rebuild workflow)."""
        log.warning(
            "Resetting Chroma collection '%s' — all documents will be deleted.",
            self._collection_name,
        )
        self._store.delete_collection()
        self._store = self._make_chroma_store()
        log.info("Collection '%s' reset and re-created.", self._collection_name)

    def count(self) -> int:
        """Return the number of documents currently stored."""
        return self._store._collection.count()

    @property
    def underlying_store(self) -> Chroma:
        """Access to the raw LangChain Chroma object for advanced use."""
        return self._store
