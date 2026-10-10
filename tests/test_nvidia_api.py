import os

import pytest
import requests

from config.settings import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_REVIEWER_MODEL,
)


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not NVIDIA_API_KEY,
    reason="live test: set RUN_LIVE_TESTS=1 and NVIDIA_API_KEY",
)
def test_nvidia_chat_completion() -> None:
    # NVIDIA Build exposes an OpenAI-compatible chat endpoint.
    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": NVIDIA_REVIEWER_MODEL,
            "messages": [
                {"role": "user", "content": "Return only the word OK."}
            ],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    response.raise_for_status()
    assert response.json()["choices"][0]["message"]["content"]
