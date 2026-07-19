from pathlib import Path

from multilingual_rag.documents.service import sanitize_filename, save_upload_bytes


def test_save_upload_bytes_sanitizes_and_persists_file(tmp_path: Path) -> None:
    saved_path = save_upload_bytes(
        tmp_path,
        filename="../unsafe name.txt",
        content=b"hello",
    )

    assert saved_path.read_bytes() == b"hello"
    assert saved_path.parent == tmp_path
    assert saved_path.name.endswith("unsafe_name.txt")


def test_sanitize_filename_falls_back_for_empty_names() -> None:
    assert sanitize_filename("...") == "document.txt"
