"""
Vector store factory and high-level retriever.

`get_vector_store()` returns the correct backend (Chroma / Pinecone) based on
the VECTOR_DB setting.  `VectorStoreRetriever` wraps the store and exposes a
single `retrieve()` method that handles threshold filtering and logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from src.config.settings import get_settings
from src.embeddings.factory import get_embedding_model
from src.utils.logging import get_logger
from src.vectordb.chroma_store import ChromaVectorStore
from src.vectordb.pinecone_store import PineconeVectorStore

log = get_logger(__name__)


# ─── type alias ───────────────────────────────────────────────────────────────

AnyVectorStore = ChromaVectorStore | PineconeVectorStore


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
        store: AnyVectorStore | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store or get_vector_store()
        self._top_k = top_k if top_k is not None else settings.default_top_k
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )
        if self._top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self._top_k}.")
        self._validate_threshold(self._threshold)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query:   The user's question.
            top_k:   How many chunks to fetch (overrides instance default).
            threshold: Minimum relevance score to include (overrides instance default).
                       Set to 0.0 to disable filtering.
            source_filter: Optional filename to restrict results to a single source.

        Returns:
            List of RetrievedChunk objects sorted by relevance (best first).
            May be empty if no chunks meet the threshold.
        """
        k = top_k if top_k is not None else self._top_k
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {k}.")

        thresh = threshold if threshold is not None else self._threshold
        self._validate_threshold(thresh)

        if not query.strip():
            log.info("Empty retrieval query received; returning no chunks.")
            return []

        settings = get_settings()
        candidate_k = min(
            max(k, k * settings.retrieval_candidate_multiplier),
            settings.retrieval_max_candidates,
        )
        metadata_filter = {"filename": source_filter} if source_filter else None

        log.info(
            "Retrieving top-%d chunks from %d candidates for query: '%s'",
            k,
            candidate_k,
            query[:80],
        )

        try:
            try:
                raw_results = self._store.similarity_search_with_score(
                    query,
                    k=candidate_k,
                    metadata_filter=metadata_filter,
                )
            except TypeError:
                raw_results = self._store.similarity_search_with_score(query, k=candidate_k)
        except Exception as exc:
            log.error("Vector store retrieval failed: %s", exc)
            raise RuntimeError(f"Retrieval error: {exc}") from exc

        if source_filter:
            raw_results = [
                (doc, score)
                for doc, score in raw_results
                if doc.metadata.get("filename") == source_filter
            ]

        chunks = [
            self._build_scored_chunk(doc, score, query, settings.retrieval_lexical_weight)
            for doc, score in raw_results
        ]
        chunks.sort(key=lambda c: c.score, reverse=True)
        chunks = chunks[:k]

        # Filter by threshold
        if thresh > 0.0:
            before = len(chunks)
            chunks = [c for c in chunks if c.score >= thresh]
            log.debug(
                "Threshold filter (%.2f): %d → %d chunks.", thresh, before, len(chunks)
            )

        log.info("Retrieved %d chunk(s) above threshold.", len(chunks))
        return chunks

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z0-9]{3,}", text.lower())
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "into",
            "show", "give", "some", "what", "when", "where", "which", "how",
            "please", "about", "question", "questions", "answer", "answers",
        }
        return {token for token in tokens if token not in stop_words}

    @classmethod
    def _lexical_score(cls, query: str, doc: Document) -> float:
        query_terms = cls._tokenize(query)
        if not query_terms:
            return 0.0

        metadata_text = " ".join(
            str(doc.metadata.get(key, ""))
            for key in ("filename", "section_heading", "file_type")
        )
        doc_terms = cls._tokenize(f"{metadata_text} {doc.page_content}")
        if not doc_terms:
            return 0.0

        overlap = query_terms & doc_terms
        return len(overlap) / len(query_terms)

    @classmethod
    def _build_scored_chunk(
        cls,
        doc: Document,
        vector_score: float,
        query: str,
        lexical_weight: float,
    ) -> RetrievedChunk:
        lexical_score = cls._lexical_score(query, doc)
        combined_score = (1.0 - lexical_weight) * vector_score + lexical_weight * lexical_score
        combined_score = max(0.0, min(1.0, combined_score))

        doc.metadata["vector_score"] = vector_score
        doc.metadata["lexical_score"] = lexical_score
        doc.metadata["combined_score"] = combined_score

        return RetrievedChunk(document=doc, score=combined_score)

    def set_threshold(self, threshold: float) -> None:
        """Update the similarity threshold used for filtering retrieved chunks."""
        self._validate_threshold(threshold)
        self._threshold = threshold

    def count_documents(self) -> int:
        """Return the total number of chunks indexed, or -1 on store error."""
        try:
            return self._store.count()
        except Exception as exc:
            log.error("count_documents failed: %s", exc)
            return -1

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {threshold}."
            )
