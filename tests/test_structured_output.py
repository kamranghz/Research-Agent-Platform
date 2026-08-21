from models.llm import llm
from models.prompts import REVIEWER_SYSTEM_PROMPT
from models.schemas import ReviewOutput


def test_structured_output() -> None:
    structured_reviewer = llm.with_structured_output(ReviewOutput)

    review = structured_reviewer.invoke(
        f"""
{REVIEWER_SYSTEM_PROMPT}

User question:
What is MCP?

Draft answer:
MCP stands for Model Context Protocol.
"""
    )

    print(review)


if __name__ == "__main__":
    test_structured_output()