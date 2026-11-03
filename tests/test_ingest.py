from pathlib import Path

import pytest

from review_rag.ingest import IngestError, load_reviews
from review_rag.models import Review

VALID_ROW = '{"id": "r1", "product": "P", "rating": 5, "text": "ok"}'


def test_load_reviews_sample_file_has_unique_ids(reviews: list[Review]) -> None:
    assert len(reviews) == 24
    assert len({r.id for r in reviews}) == 24


def test_load_reviews_csv_coerces_rating(tmp_path: Path) -> None:
    path = tmp_path / "reviews.csv"
    path.write_text("id,product,rating,title,text\nr1,Producto,4,Bien,Funciona bien\n")
    [review] = load_reviews(path)
    assert review.rating == 4


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"id": "r1", "product": "P", "rating": 9, "text": "x"}', "invalid review"),
        (f"{VALID_ROW}\n{VALID_ROW}", "duplicate review id"),
        ("{not json", "invalid JSON"),
        ('{"id": "../etc", "product": "P", "rating": 5, "text": "x"}', "invalid review"),
    ],
)
def test_load_reviews_invalid_rows_raise(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "reviews.jsonl"
    path.write_text(content)
    with pytest.raises(IngestError, match=message):
        load_reviews(path)


def test_load_reviews_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "reviews.txt"
    path.write_text(VALID_ROW)
    with pytest.raises(IngestError, match="Unsupported format"):
        load_reviews(path)


def test_load_reviews_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="File not found"):
        load_reviews(tmp_path / "missing.jsonl")
