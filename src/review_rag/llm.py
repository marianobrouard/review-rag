"""LLM client abstraction and its Anthropic implementation."""

from typing import Protocol

import anthropic

from review_rag.config import Settings
from review_rag.models import LLMAnswer


class LLMError(RuntimeError):
    """The model could not produce a valid answer."""


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> LLMAnswer:
        """Return a schema-validated answer for the given prompts."""
        ...


class AnthropicLLM:
    """Claude via the official SDK, with structured outputs and server-side refusal fallbacks."""

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self._settings = settings
        api_key = (
            settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        )
        self._client = client or anthropic.Anthropic(
            api_key=api_key,
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,  # retries 408/409/429/5xx with backoff
        )

    def generate(self, system: str, user: str) -> LLMAnswer:
        try:
            response = self._client.beta.messages.parse(
                model=self._settings.llm_model,
                max_tokens=self._settings.max_tokens,
                # Stable prefix: cached once it reaches the model's minimum cacheable length.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self._settings.effort},
                output_format=LLMAnswer,
                # If the model declines on policy grounds, the API retries on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by the API after retries; try again later") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error (HTTP {exc.status_code})") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API (network error or timeout)") from exc
        except TypeError as exc:
            # The SDK raises a plain TypeError, before any request, when no credentials resolve.
            if "authentication" not in str(exc).lower():
                raise
            raise LLMError(
                "No Anthropic credentials found: set ANTHROPIC_API_KEY or run `ant auth login`"
            ) from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request")
        if response.stop_reason == "max_tokens":
            raise LLMError("Answer truncated: increase RAG_MAX_TOKENS")
        if response.parsed_output is None:
            raise LLMError("The model returned no structured answer")
        return response.parsed_output
