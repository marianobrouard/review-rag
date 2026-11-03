"""Load and validate reviews from JSONL or CSV files."""

import csv
import json
from pathlib import Path

from pydantic import ValidationError

from review_rag.models import Review

MAX_FILE_BYTES = 50 * 1024 * 1024


class IngestError(ValueError):
    """Raised when a reviews file cannot be loaded."""


def load_reviews(path: Path) -> list[Review]:
    """Load reviews, validating every row and rejecting duplicate ids."""
    if not path.is_file():
        raise IngestError(f"File not found: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise IngestError(f"File too large (> {MAX_FILE_BYTES} bytes): {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif suffix == ".csv":
        rows = _read_csv(path)
    else:
        raise IngestError(f"Unsupported format {suffix!r}: use .jsonl or .csv")

    reviews: list[Review] = []
    seen: set[str] = set()
    for line_no, row in rows:
        try:
            review = Review.model_validate(row)
        except ValidationError as exc:
            details = exc.errors(include_url=False, include_input=False)
            raise IngestError(f"{path}:{line_no}: invalid review: {details}") from exc
        if review.id in seen:
            raise IngestError(f"{path}:{line_no}: duplicate review id {review.id!r}")
        seen.add(review.id)
        reviews.append(review)
    return reviews


def _read_jsonl(path: Path) -> list[tuple[int, object]]:
    rows: list[tuple[int, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise IngestError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def _read_csv(path: Path) -> list[tuple[int, object]]:
    with path.open(encoding="utf-8", newline="") as fh:
        # Line 1 is the header, so data rows start at line 2.
        return [(line_no, row) for line_no, row in enumerate(csv.DictReader(fh), start=2)]
