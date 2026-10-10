import os

import pytest

from models.prompts import REVIEWER_SYSTEM_PROMPT
from models.schemas import ReviewOutput


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="live test: set RUN_LIVE_TESTS=1 (needs the configured LLM)",
)
def test_structured_output() -> None:
    from models.llm import get_structured_llm

    review = get_structured_llm(ReviewOutput).invoke(
        f"""
{REVIEWER_SYSTEM_PROMPT}

User question:
What is MCP?

Draft answer:
MCP stands for Model Context Protocol.
"""
    )

    assert isinstance(review, ReviewOutput)
