"""Deterministic retrieval evals: recall@k and MRR over labeled questions (no LLM, no cost)."""

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from review_rag.retrieval import Retriever


class RetrievalCase(BaseModel):
    id: str
    question: str
    product: str | None = None
    relevant_ids: list[str] = Field(min_length=1)


class RetrievalReport(BaseModel):
    cases: int
    k: int
    recall_at_k: float
    mrr: float
    failures: list[str] = Field(
        description="Ids of cases that missed at least one relevant review."
    )


def load_cases(path: Path) -> list[RetrievalCase]:
    with path.open(encoding="utf-8") as fh:
        return [RetrievalCase.model_validate(json.loads(line)) for line in fh if line.strip()]


def evaluate_retrieval(
    retriever: Retriever, cases: Sequence[RetrievalCase], k: int = 5
) -> RetrievalReport:
    if not cases:
        raise ValueError("At least one eval case is required")

    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    failures: list[str] = []
    for case in cases:
        ranked_ids = [
            r.review.id for r in retriever.search(case.question, k=k, product=case.product)
        ]
        relevant = set(case.relevant_ids)
        recall = len(relevant.intersection(ranked_ids)) / len(relevant)
        recall_sum += recall
        reciprocal_rank_sum += next(
            (1.0 / rank for rank, rid in enumerate(ranked_ids, start=1) if rid in relevant), 0.0
        )
        if recall < 1.0:
            failures.append(case.id)

    n = len(cases)
    return RetrievalReport(
        cases=n,
        k=k,
        recall_at_k=round(recall_sum / n, 4),
        mrr=round(reciprocal_rank_sum / n, 4),
        failures=failures,
    )
