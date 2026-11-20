"""Verify BM25 content length counts original words, not derived parts."""

from math import isclose

import pytest

from src.ingestion import Chunk, FileKind, SourceDocument
from src.retrieval.bm25 import BM25Document, BM25Index, build_bm25_documents


def _built(kind: FileKind, path: str, text: str) -> BM25Document:
    document = SourceDocument(file_path=path, kind=kind, text=text)
    chunk = Chunk(path, 0, len(text), text)
    return build_bm25_documents(document, [chunk])[0]


def test_text_length_counts_original_tokens_but_keeps_parts() -> None:
    result = _built(FileKind.TEXT, "docs/a.md", "Use scaled_mm on vLLM.")
    assert result.content_terms == (
        "use", "scaled_mm", "scaled", "mm", "on", "vllm", "v", "llm",
    )
    assert result.content_length == 4
    assert result.scoring_content_length == 4


def test_python_length_still_counts_every_content_term() -> None:
    result = _built(FileKind.PYTHON, "src/a.py", "scaled_mm = 1\n")
    assert result.content_length is None
    assert result.scoring_content_length == len(result.content_terms)


def test_term_frequency_still_counts_derived_parts() -> None:
    result = _built(FileKind.TEXT, "docs/a.md", "vLLM and vLLM")
    assert result.content_terms.count("llm") == 2
    assert result.content_length == 3


def test_index_uses_declared_length_for_normalization_and_average() -> None:
    long_terms = ("cache", "x", "y", "z")
    declared = BM25Document(
        chunk=Chunk("docs/a.md", 0, 5, "cache"),
        content_terms=long_terms,
        content_length=1,
    )
    plain = BM25Document(
        chunk=Chunk("docs/b.md", 0, 5, "cache"),
        content_terms=("cache",),
    )
    index = BM25Index([declared, plain])
    assert isclose(index.statistics.average_content_length, 1.0)
    scores = [hit.content_score for hit in index.search(("cache",), 2)]
    assert isclose(scores[0], scores[1])


@pytest.mark.parametrize("length", [0, 5, True, 1.0])
def test_document_rejects_invalid_declared_length(length: object) -> None:
    with pytest.raises(ValueError):
        BM25Document(
            chunk=Chunk("docs/a.md", 0, 5, "cache"),
            content_terms=("cache", "x", "y", "z"),
            content_length=length,  # type: ignore[arg-type]
        )
