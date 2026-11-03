"""Command-line interface: ``review-rag search | ask | eval``."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from review_rag.config import Settings
from review_rag.evals import evaluate_retrieval, load_cases
from review_rag.ingest import IngestError, load_reviews
from review_rag.llm import AnthropicLLM, LLMError
from review_rag.pipeline import RAGPipeline
from review_rag.retrieval import BM25Retriever

app = typer.Typer(
    help="Grounded Q&A over product reviews, with verified citations.",
    no_args_is_help=True,
)

DataOption = Annotated[Path | None, typer.Option(help="Reviews file (.jsonl or .csv).")]
ProductOption = Annotated[str | None, typer.Option(help="Only use reviews of this product.")]


def _retriever(settings: Settings, data: Path | None) -> BM25Retriever:
    try:
        return BM25Retriever(load_reviews(data or settings.reviews_path))
    except IngestError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def search(
    query: str,
    k: Annotated[int, typer.Option(min=1, max=50)] = 5,
    product: ProductOption = None,
    data: DataOption = None,
) -> None:
    """Show the most relevant reviews for a query (no LLM call, free)."""
    for result in _retriever(Settings(), data).search(query, k=k, product=product):
        r = result.review
        typer.echo(f"[{result.score:5.2f}] {r.id} · {r.product} · {r.rating}★ · {r.title}")


@app.command()
def ask(question: str, product: ProductOption = None, data: DataOption = None) -> None:
    """Answer a question with Claude, citing and verifying the supporting reviews."""
    settings = Settings()
    pipeline = RAGPipeline(_retriever(settings, data), AnthropicLLM(settings), top_k=settings.top_k)
    try:
        answer = pipeline.ask(question, product=product)
    except (LLMError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(answer.answer)
    for citation in answer.citations:
        typer.echo(f'  · [{citation.review_id}] "{citation.quote}"')
    if not answer.grounded:
        typer.secho(
            f"Warning: {len(answer.rejected_citations)} citation(s) could not be verified.",
            fg=typer.colors.YELLOW,
            err=True,
        )


@app.command(name="eval")
def eval_command(
    cases: Annotated[Path, typer.Option(help="Eval cases (.jsonl).")] = Path(
        "evals/retrieval_cases.jsonl"
    ),
    k: Annotated[int, typer.Option(min=1, max=50)] = 5,
    min_recall: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    data: DataOption = None,
) -> None:
    """Measure retrieval quality (recall@k, MRR). Exits with code 1 if recall < --min-recall."""
    report = evaluate_retrieval(_retriever(Settings(), data), load_cases(cases), k=k)
    typer.echo(report.model_dump_json(indent=2))
    if report.recall_at_k < min_recall:
        raise typer.Exit(code=1)


logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
