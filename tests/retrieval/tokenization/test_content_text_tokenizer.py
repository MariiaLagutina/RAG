"""Verify non-Python content analysis matches query identifier analysis."""

import pytest

from src.retrieval.tokenization import (
    ContentTextTokenizer,
    QueryTokenizer,
    TextTokenizer,
)


@pytest.mark.parametrize(
    "lexeme",
    [
        "vLLM",
        "OpenAI",
        "scaled_mm",
        "FP8",
        "HuggingFace",
        "vLLM's",
        "Qwen2.5-Omni",
        "vllm.entrypoints.cli",
        "plain",
    ],
)
def test_content_terms_match_query_terms(lexeme: str) -> None:
    assert set(ContentTextTokenizer().tokenize(lexeme)) == set(
        QueryTokenizer().tokenize(lexeme)
    )


def test_content_keeps_full_token_before_identifier_parts() -> None:
    assert ContentTextTokenizer().tokenize("Use scaled_mm on vLLM.") == [
        "use",
        "scaled_mm",
        "scaled",
        "mm",
        "on",
        "vllm",
        "v",
        "llm",
    ]


def test_content_preserves_term_frequency() -> None:
    assert ContentTextTokenizer().tokenize("OpenAI and OpenAI").count(
        "openai"
    ) == 2


def test_content_extends_text_terms_without_changing_them() -> None:
    text = "The GPU's cache uses `served_model` in vLLM."
    content = ContentTextTokenizer().tokenize(text)
    assert [term for term in content if term in TextTokenizer().tokenize(
        text
    )] == TextTokenizer().tokenize(text)
