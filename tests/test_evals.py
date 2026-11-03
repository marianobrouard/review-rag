from review_rag.evals import evaluate_retrieval, load_cases
from review_rag.retrieval import BM25Retriever
from tests.conftest import EVAL_CASES

# Regression guard: if a retrieval change lowers quality below the current baseline, CI fails.
MIN_RECALL_AT_5 = 1.0
MIN_MRR = 0.9


def test_retrieval_eval_meets_baseline(retriever: BM25Retriever) -> None:
    report = evaluate_retrieval(retriever, load_cases(EVAL_CASES), k=5)
    assert report.cases == 10
    assert report.recall_at_k >= MIN_RECALL_AT_5, report.failures
    assert report.mrr >= MIN_MRR
