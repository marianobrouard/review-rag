"""Lexical retrieval with Okapi BM25: no external services, deterministic, easy to evaluate."""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Protocol

from review_rag.models import Review, SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# fmt: off
_STOPWORDS = frozenset({
    # Spanish
    "a", "al", "algo", "como", "con", "de", "del", "el", "ella", "en", "es", "esta", "este",
    "esto", "fue", "ha", "la", "las", "le", "lo", "los", "me", "mi", "muy", "no", "o", "para",
    "pero", "por", "que", "se", "si", "sin", "su", "sus", "te", "tiene", "un", "una", "uno",
    "y", "ya",
    # English
    "the", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with", "this", "that",
    "was", "are", "be", "at", "as",
})
# fmt: on


def _stem(token: str) -> str:
    """Very light plural stripping for Spanish and English (alfombras -> alfombra)."""
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, drop stopwords and strip plurals."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return [_stem(t) for t in _TOKEN_RE.findall(ascii_text) if t not in _STOPWORDS]


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, product: str | None = None) -> list[SearchResult]:
        """Return up to ``k`` reviews ranked by relevance, optionally limited to one product."""
        ...


class BM25Retriever:
    """In-memory BM25 over an inverted index. Suitable for up to ~100k reviews."""

    def __init__(self, reviews: Sequence[Review], k1: float = 1.5, b: float = 0.75) -> None:
        if not reviews:
            raise ValueError("BM25Retriever needs at least one review")
        self._reviews = list(reviews)
        self._k1 = k1
        self._b = b

        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len: list[int] = []
        for doc_idx, review in enumerate(self._reviews):
            counts = Counter(tokenize(f"{review.product} {review.title} {review.text}"))
            self._doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((doc_idx, tf))

        n_docs = len(self._reviews)
        self._avg_len = max(sum(self._doc_len) / n_docs, 1.0)
        self._idf = {
            term: math.log(1 + (n_docs - len(posts) + 0.5) / (len(posts) + 0.5))
            for term, posts in self._postings.items()
        }

    def search(self, query: str, k: int = 5, product: str | None = None) -> list[SearchResult]:
        if k < 1:
            raise ValueError("k must be >= 1")

        # Filter before scoring, never after: the same rule applies to access permissions.
        allowed: set[int] | None = None
        if product is not None:
            wanted = product.casefold()
            allowed = {i for i, r in enumerate(self._reviews) if r.product.casefold() == wanted}

        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_idx, tf in self._postings[term]:
                if allowed is not None and doc_idx not in allowed:
                    continue
                norm = self._k1 * (1 - self._b + self._b * self._doc_len[doc_idx] / self._avg_len)
                scores[doc_idx] += idf * tf * (self._k1 + 1) / (tf + norm)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], self._reviews[item[0]].id))
        return [SearchResult(review=self._reviews[i], score=s) for i, s in ranked[:k]]
