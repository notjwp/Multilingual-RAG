"""bge-m3 adapter check. Loads a ~2.2 GB model, so it is opt-in.

Runs only when sentence-transformers is installed AND RUN_MODEL_TESTS=1 is set, keeping the
default suite fast and CI torch-free.
"""

import math
import os
from importlib.util import find_spec

import pytest

_ENABLED = find_spec("sentence_transformers") is not None and os.environ.get("RUN_MODEL_TESTS")

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="set RUN_MODEL_TESTS=1 with sentence-transformers installed to run bge-m3 checks",
)


def test_bge_m3_embeds_normalized_1024d_vectors() -> None:
    from multilingual_rag.embeddings.bge_embeddings import EMBEDDING_DIM, BgeM3EmbeddingProvider

    provider = BgeM3EmbeddingProvider()
    docs = provider.embed_documents(("黑豹队的防守", "The Panthers defense"))
    query = provider.embed_query("What is the defense?")

    assert len(docs) == 2
    assert all(len(vector) == EMBEDDING_DIM for vector in docs)
    assert len(query) == EMBEDDING_DIM
    assert all(isinstance(value, float) for value in query)
    # normalize_embeddings=True -> unit vectors, so dot product is cosine.
    assert math.isclose(math.sqrt(sum(value * value for value in query)), 1.0, abs_tol=1e-3)
    assert provider.embed_documents(()) == []
