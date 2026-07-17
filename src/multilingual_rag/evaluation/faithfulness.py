"""Faithfulness judging contract for grounded-answer evaluation.

Faithfulness — is every claim in the answer supported by the retrieved context? — needs an LLM
to judge, so it sits behind a ``Protocol`` like every other external dependency. Phase B ships
the port and the aggregation helper; a concrete (free-tier) judge implementation lands with the
free generation adapter in Phase C. Keep it opt-in: judging costs a model call per example.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class FaithfulnessJudge(Protocol):
    """Decides whether an answer is fully supported by the context it was given."""

    def is_supported(self, *, answer: str, context: str) -> bool:
        """Return True when every claim in ``answer`` is grounded in ``context``."""
        ...


def average_faithfulness(judgements: Sequence[bool]) -> float:
    """Return the share of judged answers found faithful (0.0 when nothing was judged)."""
    if not judgements:
        return 0.0
    return sum(1 for supported in judgements if supported) / len(judgements)
