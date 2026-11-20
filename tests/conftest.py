"""Shared test isolation for optional model-backed components."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from src.retrieval.semantic import SemanticBackendLoadError


@pytest.fixture(autouse=True)
def _semantic_model_unavailable() -> Iterator[None]:
    """Keep `index` tests offline: the semantic build fails as if unavailable.

    Tests that need the semantic path patch the builder themselves.
    """
    with patch(
        "src.cli.build_and_store_semantic_index",
        side_effect=SemanticBackendLoadError("model unavailable in tests"),
    ):
        yield
