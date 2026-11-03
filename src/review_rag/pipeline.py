"""Retrieval-augmented answering: retrieve -> generate -> verify citations."""

import html
import re
from collections.abc import Sequence

from review_rag.llm import LLMClient
from review_rag.models import Answer, Citation, SearchResult
from review_rag.prompts import SYSTEM_PROMPT, build_user_message
from review_rag.retrieval import Retriever

MAX_QUESTION_CHARS = 1_000
_MIN_QUOTE_CHARS = 3
_WHITESPACE_RE = re.compile(r"\s+")

NO_RESULTS_MESSAGE = "No relevant reviews were found for this question."


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int = 5) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k

    def ask(self, question: str, product: str | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("The question must not be empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"The question exceeds {MAX_QUESTION_CHARS} characters")

        results = self._retriever.search(question, k=self._top_k, product=product)
        if not results:
            # Nothing to ground an answer on: skip the LLM call (no cost, no hallucination).
            return Answer(
                question=question,
                answer=NO_RESULTS_MESSAGE,
                citations=[],
                insufficient_context=True,
                grounded=True,
            )

        llm_answer = self._llm.generate(SYSTEM_PROMPT, build_user_message(question, results))
        valid, rejected = validate_citations(llm_answer.citations, results)
        has_support = bool(valid) or llm_answer.insufficient_context
        return Answer(
            question=question,
            answer=llm_answer.answer,
            citations=valid,
            insufficient_context=llm_answer.insufficient_context,
            grounded=not rejected and has_support,
            rejected_citations=rejected,
            sources=list(results),
        )


def validate_citations(
    citations: Sequence[Citation], results: Sequence[SearchResult]
) -> tuple[list[Citation], list[Citation]]:
    """Keep citations whose review was retrieved and whose quote appears verbatim in it."""
    texts = {r.review.id: _normalize(f"{r.review.title} {r.review.text}") for r in results}
    valid: list[Citation] = []
    rejected: list[Citation] = []
    for citation in citations:
        source = texts.get(citation.review_id)
        quote = _normalize(html.unescape(citation.quote))
        if source is not None and len(quote) >= _MIN_QUOTE_CHARS and quote in source:
            valid.append(citation)
        else:
            rejected.append(citation)
    return valid, rejected


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.casefold()).strip().strip("\"'«»“”.")
