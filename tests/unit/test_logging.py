import json
import logging

import pytest

from multilingual_rag.core.logging import configure_logging


def test_configure_logging_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    logger = logging.getLogger("multilingual_rag.test")
    logger.info("hello")

    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert payload["message"] == "hello"
    assert payload["logger"] == "multilingual_rag.test"


def test_configure_logging_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="Invalid log level"):
        configure_logging("VERBOSE")


def test_json_formatter_outputs_valid_json() -> None:
    configure_logging("INFO")
    formatter = logging.getLogger().handlers[0].formatter
    assert formatter is not None

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="structured",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "structured"
