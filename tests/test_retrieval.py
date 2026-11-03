import pytest

from review_rag.models import Review
from review_rag.retrieval import BM25Retriever, tokenize


def test_tokenize_strips_accents_stopwords_and_plurals() -> None:
    assert tokenize("¿La batería de los Auriculares?") == ["bateria", "auricular"]


def test_search_ranks_relevant_reviews_first(retriever: BM25Retriever) -> None:
    results = retriever.search("se atasca en las alfombras", k=2)
    assert {r.review.id for r in results} == {"cb-01", "cb-04"}
    assert results[0].score >= results[1].score


def test_search_product_filter_excludes_other_products(retriever: BM25Retriever) -> None:
    results = retriever.search("batería", k=10, product="reloj fitpulse 3")
    assert results
    assert all(r.review.product == "Reloj FitPulse 3" for r in results)


def test_search_without_matching_terms_returns_empty(retriever: BM25Retriever) -> None:
    assert retriever.search("xyzzy quux") == []


def test_search_invalid_k_raises(retriever: BM25Retriever) -> None:
    with pytest.raises(ValueError, match="k must be"):
        retriever.search("batería", k=0)


def test_search_ties_are_ordered_by_id() -> None:
    reviews = [Review(id=i, product="P", rating=5, text="igual texto") for i in ("b", "a")]
    assert [r.review.id for r in BM25Retriever(reviews).search("texto")] == ["a", "b"]


def test_retriever_requires_reviews() -> None:
    with pytest.raises(ValueError, match="at least one review"):
        BM25Retriever([])
