"""Domain models shared across the pipeline."""

from pydantic import BaseModel, ConfigDict, Field


class Review(BaseModel):
    """A single customer review. Its text is untrusted, user-generated content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    product: str = Field(min_length=1, max_length=200)
    rating: int = Field(ge=1, le=5)
    title: str = Field(default="", max_length=300)
    text: str = Field(min_length=1, max_length=10_000)


class SearchResult(BaseModel):
    """A review returned by the retriever, with its relevance score."""

    model_config = ConfigDict(frozen=True)

    review: Review
    score: float


class Citation(BaseModel):
    """Evidence for a claim: the review id and a verbatim quote from that review."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    quote: str


class LLMAnswer(BaseModel):
    """Schema the model must return. Enforced with structured outputs."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation]
    insufficient_context: bool


class Answer(BaseModel):
    """Final answer returned to callers, after citation validation."""

    question: str
    answer: str
    citations: list[Citation]
    insufficient_context: bool
    grounded: bool = Field(
        description="False if the model cited a review or quote that is not in the sources."
    )
    rejected_citations: list[Citation] = Field(default_factory=list)
    sources: list[SearchResult] = Field(default_factory=list)
