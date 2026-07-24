"""Select the vector store from settings.

Chroma (embedded, default) or FAISS (local per-user index files). Both satisfy the ``VectorStore``
protocol, so nothing downstream changes — switching is a ``VECTOR_STORE`` change with no code edit.
"""

from __future__ import annotations

from multilingual_rag.core.config import Settings
from multilingual_rag.vectorstores.base import VectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    """Return the configured vector store adapter."""
    if settings.vector_store == "faiss":
        from multilingual_rag.vectorstores.faiss_store import FaissVectorStore

        return FaissVectorStore(settings)

    from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

    return ChromaVectorStore(settings)
