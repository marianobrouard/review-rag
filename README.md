# review-rag

Ask questions about products and get answers grounded in customer reviews, with every citation verified.

Example (illustrative output):

```
$ review-rag ask "¿Qué critican los clientes de la cafetera?"
Las críticas más repetidas son las fugas de agua y la limpieza difícil...
  · [bh-05] "Tuve fugas de agua desde la primera semana"
  · [bh-03] "son difíciles de limpiar"
```

## How it works

```
question ──► BM25 search ──► top-k reviews ──► Claude (structured output) ──► citation check ──► answer
                 │                                   ▲
           product filter              reviews wrapped as untrusted data
```

1. **Retrieve**: BM25 over an inverted index (accent-insensitive, light plural stemming, optional product filter).
2. **Generate**: Claude answers using only the retrieved reviews and returns a Pydantic-validated schema (`answer`, `citations`, `insufficient_context`).
3. **Verify**: every citation must point to a retrieved review and quote it verbatim; otherwise the answer is flagged `grounded=false`.

## Quickstart

```bash
uv sync
uv run review-rag search "GPS impreciso" --k 3       # retrieval only, free
uv run review-rag eval                                # retrieval metrics, free
export ANTHROPIC_API_KEY=...                          # or: ant auth login
uv run review-rag ask "¿El reloj tiene defectos?" --product "Reloj FitPulse 3"
```

## Quality and safety

| Area | What is in place |
|---|---|
| Retrieval evals | 10 labeled questions: **recall@5 = 1.0, MRR = 1.0**. Enforced as a CI regression test. |
| Hallucinated citations | Quotes are checked verbatim against the source review. |
| Prompt injection | Review text is HTML-escaped and wrapped in `<review>` tags. The system prompt treats it as data. The sample data includes an injection attempt (`fp-06`), covered by a live test. |
| API robustness | Structured outputs, timeouts, retries with backoff, explicit `max_tokens`, refusal fallbacks, typed error handling. |
| Cost | No LLM call when nothing relevant is retrieved. The system prompt is cache-ready. Model and effort are configurable. |
| Engineering | Python 3.12, `mypy --strict`, ruff (including bandit security rules), pytest, CI with SHA-pinned actions, Dependabot. |

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | API key (or use an `ant auth login` profile) |
| `RAG_LLM_MODEL` | `claude-opus-5-5` | Model. Try `claude-haiku-5-5` for lower cost and compare with the evals. |
| `RAG_EFFORT` | `medium` | Reasoning effort: `low` … `max` |
| `RAG_TOP_K` | `5` | Reviews passed to the model |
| `RAG_REVIEWS_PATH` | `data/sample_reviews.synthetic.jsonl` | Reviews file (`.jsonl` or `.csv`) |

## Design decisions

- **BM25 before embeddings.** It needs no extra API keys, runs deterministically, and its evals are cheap. `Retriever` is a Protocol, so a hybrid retriever can replace it without touching the pipeline.
- **Verify, don't trust.** Structured outputs guarantee the shape of the answer, and the citation check guarantees it is grounded.
- **Provider behind an interface.** `LLMClient` keeps the pipeline testable with a fake model, so no unit test calls the API.

## Roadmap

- Hybrid retrieval (BM25 + embeddings) with a reranker, to fix lexical misses such as "autonomía" vs "batería".
- Answer-quality evals with an LLM judge (faithfulness, completeness).
- Async FastAPI service with rate limiting and per-user budgets.

## Data

`data/sample_reviews.synthetic.jsonl` contains **synthetic reviews written for demonstration and testing**, not real customers. Use your own data with `--data path/to/reviews.csv` (columns: `id, product, rating, title, text`).
