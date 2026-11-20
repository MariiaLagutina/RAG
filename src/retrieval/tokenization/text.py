"""Tokenize documentation without aggressive linguistic normalization."""

from src.retrieval.tokenization.code import expand_code_lexeme
from src.retrieval.tokenization.shared import normalize_lexeme, scan_lexemes


_POSSESSIVE_BASE_MIN_LETTERS = 4


class TextTokenizer:
    """Create stable lexical terms from human-readable documentation."""

    def tokenize(self, text: str) -> list[str]:
        """Return normalized terms while preserving technical lexical units."""
        tokens: list[str] = []
        for lexeme in scan_lexemes(text):
            tokens.extend(expand_text_lexeme(lexeme))
        return tokens


def expand_text_lexeme(lexeme: str) -> list[str]:
    """Add a cautious base signal for likely possessive forms."""
    normalized = normalize_lexeme(lexeme)
    tokens = [normalized]

    if not normalized.endswith("'s"):
        return tokens

    raw_base = lexeme[:-2]
    normalized_base = normalized[:-2]
    base_letter_count = sum(character.isalpha() for character in raw_base)
    if (
        base_letter_count >= _POSSESSIVE_BASE_MIN_LETTERS
        or raw_base.isupper()
    ):
        tokens.append(normalized_base)

    return tokens


class ContentTextTokenizer:
    """Analyze non-Python content with the same per-lexeme terms as queries.

    Queries combine text and code expansion, and Python content and
    metadata already receive identifier components. Applying both expansions
    to each lexeme here keeps term frequency and lets identifiers written in
    documentation (``scaled_mm``, ``OpenAI``) match their query components.
    """

    def tokenize(self, text: str) -> list[str]:
        """Return text terms followed by any additional identifier parts."""
        tokens: list[str] = []
        for lexeme in scan_lexemes(text):
            terms = expand_text_lexeme(lexeme)
            terms.extend(
                term
                for term in expand_code_lexeme(lexeme)
                if term not in terms
            )
            tokens.extend(terms)
        return tokens
