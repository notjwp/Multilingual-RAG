"""Build the vector store from settings.

The embedded ChromaDB adapter (cosine space) is the only backend; it satisfies the ``VectorStore``
protocol, so callers depend on the port, not the concrete store. The import is lazy so modules that
never build a store don't pull in ``chromadb`` at import time.
"""

from __future__ import annotations

from multilingual_rag.core.config import Settings
from multilingual_rag.vectorstores.base import VectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    """Return the Chroma vector store adapter."""
    from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

    return ChromaVectorStore(settings)
