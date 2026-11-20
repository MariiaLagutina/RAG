"""Choose the dataset retrieval strategy from an explicit mode."""

from pathlib import Path
import re

RETRIEVAL_MODES = ("auto", "bm25", "hybrid")
_NAME_TOKEN_SEPARATOR = re.compile(r"[^a-z0-9]+")


def resolve_retrieval_mode(mode: str, dataset_path: Path) -> str:
    """Return ``bm25`` or ``hybrid`` for one dataset search.

    ``auto`` follows the assignment's dataset naming convention: a file
    whose name contains the token ``docs`` (and not ``code``) holds
    documentation questions and uses hybrid retrieval. Every other name,
    including code datasets and unknown names, uses BM25 only. Only the
    file name is inspected, never its directories or question content.
    """
    if mode not in RETRIEVAL_MODES:
        allowed = ", ".join(RETRIEVAL_MODES)
        raise ValueError(f"Retrieval mode must be one of: {allowed}")
    if mode != "auto":
        return mode
    tokens = set(_NAME_TOKEN_SEPARATOR.split(dataset_path.stem.lower()))
    if "docs" in tokens and "code" not in tokens:
        return "hybrid"
    return "bm25"
