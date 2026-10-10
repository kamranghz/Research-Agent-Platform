import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents import reviewer
from agents.researcher import _research_output_to_state
from core.limits import QuestionLimitReached, QuestionLimiter
from core.router import review_router
from models.schemas import ResearchOutput
from tools import file_reader as file_reader_module
from tools.file_reader import file_reader


def test_file_reader_allows_datasets(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    (root / "a.txt").write_text("hello", encoding="utf-8")
    with patch.object(file_reader_module, "FILE_READER_ROOT", root):
        assert file_reader.invoke({"file_path": str(root / "a.txt")}) == "hello"


@pytest.mark.parametrize(
    "relative", ["../secret.txt", "../../etc/passwd.txt", "sub/../../secret.txt"]
)
def test_file_reader_blocks_escape(tmp_path: Path, relative: str) -> None:
    root = tmp_path / "datasets"
    (root / "sub").mkdir(parents=True)
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    with patch.object(file_reader_module, "FILE_READER_ROOT", root):
        result = file_reader.invoke({"file_path": str(root / relative)})
    assert result.startswith("Error: access denied")
    assert "nope" not in result


def test_file_reader_blocks_absolute_outside(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    outside = tmp_path / "x.txt"
    outside.write_text("nope", encoding="utf-8")
    with patch.object(file_reader_module, "FILE_READER_ROOT", root):
        result = file_reader.invoke({"file_path": str(outside)})
    assert result.startswith("Error: access denied")


def test_disabled_tool_is_not_executed() -> None:
    response = ResearchOutput(
        needs_tool=True,
        tool_name="browser_tool",
        tool_input="https://example.com",
    )
    with patch("agents.researcher.DISABLED_TOOL_NAMES", frozenset({"browser_tool"})):
        state = _research_output_to_state(response, "fetch https://example.com", 1)
    assert state["needs_tool"] is False
    assert "disabled" in state["draft_answer"]


def test_question_limiter() -> None:
    limiter = QuestionLimiter(per_session=2, total=3)
    limiter.acquire("a")
    limiter.acquire("a")
    with pytest.raises(QuestionLimitReached):
        limiter.acquire("a")
    limiter.acquire("b")
    with pytest.raises(QuestionLimitReached):
        limiter.acquire("c")  # global budget (3) exhausted


def test_judge_unavailable_never_fakes_scores() -> None:
    state = {
        "query": "q",
        "draft_answer": "a",
        "tool_result": "",
    }
    with patch.object(reviewer, "_judge_providers", return_value=[]):
        result = reviewer.reviewer_node(state)
    assert result["review_status"] == "unavailable"
    assert result["review_accuracy"] == 0.0
    assert "not reviewed" in result["review_reason"]
    # Router must finalize instead of looping on a dead judge.
    assert review_router({**state, **result, "retry_count": 1}) == "max_retry"


def test_judge_failure_falls_through_providers() -> None:
    good = json.dumps(
        {"accuracy": 0.9, "relevance": 0.9, "clarity": 0.9,
         "completeness": 0.9, "reason": "ok"}
    )
    providers = [{"name": "p1"}, {"name": "p2"}]

    def fake_call(provider, *_args):
        if provider["name"] == "p1":
            raise ValueError("boom")
        return reviewer.ReviewOutput.model_validate_json(good)

    with patch.object(reviewer, "_judge_providers", return_value=providers), \
         patch.object(reviewer, "_call_judge", side_effect=fake_call):
        review, name = reviewer._call_reviewer("q", "d", "e")
    assert name == "p2"
    assert review.accuracy == 0.9
