from pathlib import Path

import pytest

from review_rag.ingest import load_reviews
from review_rag.models import LLMAnswer, Review
from review_rag.retrieval import BM25Retriever

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_REVIEWS = ROOT / "data" / "sample_reviews.synthetic.jsonl"
EVAL_CASES = ROOT / "evals" / "retrieval_cases.jsonl"


class FakeLLM:
    """Test double: records the prompts it receives and returns a canned answer."""

    def __init__(self, answer: LLMAnswer | None = None) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMAnswer:
        self.calls.append((system, user))
        if self.answer is None:
            raise AssertionError("FakeLLM was called but no answer was configured")
        return self.answer


@pytest.fixture(scope="session")
def reviews() -> list[Review]:
    return load_reviews(SAMPLE_REVIEWS)


@pytest.fixture(scope="session")
def retriever(reviews: list[Review]) -> BM25Retriever:
    return BM25Retriever(reviews)
