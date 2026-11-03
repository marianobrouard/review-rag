from typer.testing import CliRunner

from review_rag.cli import app
from tests.conftest import EVAL_CASES, SAMPLE_REVIEWS

runner = CliRunner()


def test_search_command_prints_ranked_reviews() -> None:
    result = runner.invoke(
        app, ["search", "GPS impreciso", "--k", "2", "--data", str(SAMPLE_REVIEWS)]
    )
    assert result.exit_code == 0
    assert "fp-02" in result.output


def test_eval_command_fails_below_min_recall_threshold() -> None:
    args = ["eval", "--cases", str(EVAL_CASES), "--data", str(SAMPLE_REVIEWS), "--k", "1"]
    assert runner.invoke(app, [*args, "--min-recall", "0.0"]).exit_code == 0
    assert runner.invoke(app, [*args, "--min-recall", "1.0"]).exit_code == 1


def test_search_command_with_missing_file_exits_with_error() -> None:
    result = runner.invoke(app, ["search", "x", "--data", "nope.jsonl"])
    assert result.exit_code == 2
