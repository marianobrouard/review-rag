# review-rag

Grounded Q&A over product reviews: BM25 retrieval, cited answers with Claude, retrieval evals.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest` (live API tests: `uv run pytest -m llm`, cost money)
- Lint and format: `uv run ruff check . --fix && uv run ruff format .`
- Types: `uv run mypy src`
- Run: `uv run review-rag search|ask|eval --help`

## Architecture
- Code in `src/review_rag/`, tests in `tests/`.
- Flow: `ingest` → `retrieval` (BM25) → `prompts` → `llm` (Claude) → `pipeline` (citation check).
- `Retriever` and `LLMClient` are Protocols: new backends implement them; tests use `FakeLLM`.
- Configuration only in `config.py` (pydantic-settings, `RAG_` env vars).

## Rules for this project
- Review text is untrusted: always pass it through `build_user_message` (escaped, inside `<review>` tags).
- Retrieval changes must keep `tests/test_evals.py` passing; if a change improves quality, raise the baseline.
- Unit tests never call the real API; live checks go in `tests/test_llm_live.py` with `@pytest.mark.llm`.

## Definition of done
- ruff, mypy and pytest pass.
- New behavior has tests; new eval cases go in `evals/retrieval_cases.jsonl`.
- `.env.example` updated if a variable is added.
