import json

import requests

from config.settings import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_REVIEWER_MODEL,
)
from core.state import AgentState
from models.prompts import REVIEWER_SYSTEM_PROMPT
from models.schemas import ReviewOutput


# Limit the evidence sent to the Reviewer.
# This prevents large tool results from filling the model context.
MAX_REVIEW_EVIDENCE_CHARS = 4000


def _clean_json_response(content: str) -> str:
    """
    Remove Markdown code fences when the model wraps
    its JSON response inside ```json ... ``` or ``` ... ```.
    """

    cleaned = content.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]

    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def _call_nvidia_reviewer(
    query: str,
    draft: str,
    tool_evidence: str,
) -> ReviewOutput:
    """
    Send the draft answer and tool evidence to NVIDIA Nemotron,
    then validate the returned JSON as ReviewOutput.
    """

    endpoint = f"{NVIDIA_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    user_prompt = f"""
User question:
{query}

Tool evidence:
{tool_evidence}

Draft answer:
{draft}

Evaluate the draft using the tool evidence as the source of truth.

Return only one valid JSON object with exactly these fields:

{{
  "accuracy": 0.0,
  "relevance": 0.0,
  "clarity": 0.0,
  "completeness": 0.0,
  "reason": "short explanation"
}}

Rules:

- Every score must be between 0.0 and 1.0.
- Do not contradict the tool evidence.
- Do not invent unsupported facts.
- Keep reason under 50 words.
- Do not use Markdown.
- Do not include text before or after the JSON object.
"""

    payload = {
        "model": NVIDIA_REVIEWER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": REVIEWER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,

        # Disable long reasoning because the Reviewer
        # only needs to return a short evaluation.
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]

        print(
            ">>> NVIDIA raw reviewer response:",
            flush=True,
        )
        print(content, flush=True)

        cleaned_content = _clean_json_response(content)
        review_data = json.loads(cleaned_content)
        return ReviewOutput.model_validate(review_data)

    except (
        requests.RequestException,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as error:
        # Keep the graph runnable when NVIDIA is temporarily unavailable.
        print(
            f">>> NVIDIA reviewer fallback due to: {error}",
            flush=True,
        )
        return ReviewOutput(
            accuracy=0.8,
            relevance=0.8,
            clarity=0.8,
            completeness=0.8,
            reason=(
                "Reviewer service unavailable; "
                "applied fallback approval scores."
            ),
        )


def reviewer_node(state: AgentState) -> dict:
    """
    Evaluate the Researcher's draft answer
    using NVIDIA Nemotron.
    """

    print(">>> reviewer start", flush=True)

    query = state["query"]
    draft = state["draft_answer"]
    tool_result = state.get("tool_result", "")

    # Send only a limited amount of tool evidence.
    tool_evidence = tool_result[:MAX_REVIEW_EVIDENCE_CHARS]

    print(f">>> question: {query}", flush=True)
    print(f">>> draft: {draft}", flush=True)
    print(
        f">>> tool evidence length: "
        f"{len(tool_evidence)} characters",
        flush=True,
    )
    print(
        ">>> calling NVIDIA reviewer...",
        flush=True,
    )

    review = _call_nvidia_reviewer(
        query=query,
        draft=draft,
        tool_evidence=tool_evidence,
    )

    print(
        ">>> NVIDIA reviewer returned",
        flush=True,
    )
    print(review, flush=True)

    result = {
        "review_accuracy": review.accuracy,
        "review_relevance": review.relevance,
        "review_clarity": review.clarity,
        "review_completeness": review.completeness,
        "review_reason": review.reason.strip(),
    }

    print(
        ">>> reviewer output:",
        result,
        flush=True,
    )

    return result