"""Runtime configuration, read from environment variables (prefix ``RAG_``) or a ``.env`` file."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Optional: the Anthropic SDK also resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    llm_model: str = "claude-opus-5-5"
    effort: Effort = "medium"
    max_tokens: int = Field(default=16_000, ge=256, le=64_000)
    request_timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    top_k: int = Field(default=5, ge=1, le=20)
    reviews_path: Path = Path("data/sample_reviews.synthetic.jsonl")
