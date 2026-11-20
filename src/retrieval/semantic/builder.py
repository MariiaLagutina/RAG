"""Build a semantic index from the exact documents already used by BM25."""

from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Protocol

from torch import Tensor

from src.ingestion import FileKind
from src.ingestion.files import SUFFIX_TO_KIND
from src.retrieval.bm25 import BM25Document, BM25Index
from src.retrieval.semantic.index import SemanticIndex
from src.retrieval.semantic.models import SemanticDocument


class SemanticTextEncoder(Protocol):
    """Describe the embedding operation required by the index builder."""

    def encode(self, texts: tuple[str, ...]) -> Tensor:
        """Return one normalized vector per input text."""


DocumentFilter = Callable[[BM25Document], bool]


def is_documentation_document(document: BM25Document) -> bool:
    """Return whether a chunk comes from a text (non-Python) corpus file."""
    suffix = PurePosixPath(document.chunk.file_path).suffix.lower()
    return SUFFIX_TO_KIND.get(suffix) is FileKind.TEXT


def build_semantic_index(
    lexical_index: BM25Index,
    encoder: SemanticTextEncoder,
    *,
    include: DocumentFilter | None = None,
) -> SemanticIndex:
    """Encode BM25 chunks without creating a second corpus pipeline."""
    lexical_documents = tuple(
        document
        for document in lexical_index.documents
        if include is None or include(document)
    )
    semantic_documents = tuple(
        SemanticDocument(document.chunk) for document in lexical_documents
    )
    embeddings = encoder.encode(
        tuple(document.chunk.text for document in lexical_documents)
    )
    return SemanticIndex(semantic_documents, embeddings)
