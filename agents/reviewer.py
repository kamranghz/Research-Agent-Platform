import json

import requests

from config.settings import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
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


class JudgeUnavailable(Exception):
    """Raised when no judge provider returned a valid evaluation."""


def _judge_providers() -> list[dict]:
    """Judge endpoints in preference order: NVIDIA, then Groq."""

    providers = []
    if NVIDIA_API_KEY:
        providers.append(
            {
                "name": "nvidia",
                "base_url": NVIDIA_BASE_URL,
                "key": NVIDIA_API_KEY,
                "model": NVIDIA_REVIEWER_MODEL,
                "extra": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        )
    if GROQ_API_KEY:
        providers.append(
            {
                "name": "groq",
                "base_url": GROQ_BASE_URL,
                "key": GROQ_API_KEY,
                "model": GROQ_MODEL,
                "extra": {},
            }
        )
    return providers


def _call_judge(
    provider: dict,
    query: str,
    draft: str,
    tool_evidence: str,
) -> ReviewOutput:
    """
    Send the draft answer and tool evidence to one judge provider,
    then validate the returned JSON as ReviewOutput.
    Raises on any failure.
    """

    endpoint = f"{provider['base_url']}/chat/completions"

    headers = {
        "Authorization": f"Bearer {provider['key']}",
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
        "model": provider["model"],
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
        # Provider-specific options (e.g. disable long reasoning).
        **provider["extra"],
    }

    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    review_data = json.loads(_clean_json_response(content))
    return ReviewOutput.model_validate(review_data)


def _call_reviewer(
    query: str,
    draft: str,
    tool_evidence: str,
) -> tuple[ReviewOutput, str]:
    """
    Try each judge provider in order.

    Returns (review, provider_name). Raises JudgeUnavailable if all
    fail; callers must NOT invent scores in that case.
    """

    errors: list[str] = []
    for provider in _judge_providers():
        try:
            review = _call_judge(provider, query, draft, tool_evidence)
            return review, provider["name"]
        except (
            requests.RequestException,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            ValueError,
        ) as error:
            errors.append(f"{provider['name']}: {type(error).__name__}")
            print(
                f">>> judge {provider['name']} failed: {error}",
                flush=True,
            )

    raise JudgeUnavailable(
        "; ".join(errors) or "no judge API key configured"
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
        ">>> calling judge...",
        flush=True,
    )

    try:
        review, judge_name = _call_reviewer(
            query=query,
            draft=draft,
            tool_evidence=tool_evidence,
        )
    except JudgeUnavailable as error:
        # Never fabricate scores. The router skips retries and the UI
        # shows that the answer was NOT reviewed.
        print(f">>> judge unavailable: {error}", flush=True)
        return {
            "review_status": "unavailable",
            "review_accuracy": 0.0,
            "review_relevance": 0.0,
            "review_clarity": 0.0,
            "review_completeness": 0.0,
            "review_reason": f"Judge unavailable ({error}); answer not reviewed.",
        }

    print(f">>> judge {judge_name} returned", flush=True)

    result = {
        "review_status": "ok",
        "review_accuracy": review.accuracy,
        "review_relevance": review.relevance,
        "review_clarity": review.clarity,
        "review_completeness": review.completeness,
        "review_reason": review.reason.strip(),
    }

    print(">>> reviewer output:", result, flush=True)

    return result
