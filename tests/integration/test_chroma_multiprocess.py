"""The API-vs-worker multi-process read: a reader whose Chroma client predates a separate
writer process's upsert must still surface the new rows (embedded Chroma's "Error finding id"
staleness, fixed by reload-on-change in ChromaVectorStore)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import DocumentChunk
from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore

# Runs in a SEPARATE OS process (the "worker") and appends one row over the same persist dir.
WRITER = """
import sys
from multilingual_rag.core.config import Settings
from multilingual_rag.core.models import DocumentChunk
from multilingual_rag.vectorstores.chroma_store import ChromaVectorStore
path, coll = sys.argv[1], sys.argv[2]
store = ChromaVectorStore(Settings(chroma_persist_directory=path, chroma_collection_name=coll))
chunk = DocumentChunk(chunk_id="w:0", document_id="w", text="written by a separate process",
    language="en", source="s.txt", chunk_index=0, checksum="ck", token_count=3, metadata={})
store.upsert_chunks((chunk,), ([0.0, 1.0],), user_id="u1")
"""


def _chunk(chunk_id: str, document_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        language="en",
        source="s.txt",
        chunk_index=0,
        checksum="ck",
        token_count=3,
        metadata={},
    )


def test_reader_process_sees_a_writer_process_upsert(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    coll = "docs_mp"
    reader = ChromaVectorStore(
        Settings(chroma_persist_directory=path, chroma_collection_name=coll)
    )
    # Reader opens + loads its collection (the seed) — this is the client that will go stale.
    reader.upsert_chunks((_chunk("seed:0", "seed", "seed row"),), ([1.0, 0.0],), user_id="u1")
    assert [r.document_id for r in reader.search([0.0, 1.0], user_id="u1", top_k=5)] == ["seed"]

    # A SEPARATE process appends a new row over the same directory.
    subprocess.run([sys.executable, "-c", WRITER, str(path), coll], check=True)

    # The reader's originally-stale client must now surface the writer's row (the fix).
    docs = [r.document_id for r in reader.search([0.0, 1.0], user_id="u1", top_k=5)]
    assert "w" in docs, f"stale reader missed the writer's row; saw {docs}"
