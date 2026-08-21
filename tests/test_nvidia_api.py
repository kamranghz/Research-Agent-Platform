import requests

from config.settings import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_REVIEWER_MODEL,
)


# NVIDIA Build exposes an OpenAI-compatible chat endpoint.
endpoint = f"{NVIDIA_BASE_URL}/chat/completions"


headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json",
}


payload = {
    "model": NVIDIA_REVIEWER_MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Return only the word OK.",
        }
    ],
    "temperature": 0,
    "max_tokens": 32,
    "stream": False,

    # Disable reasoning because the Reviewer only needs
    # a short structured evaluation.
    "chat_template_kwargs": {
        "enable_thinking": False,
    },
}


response = requests.post(
    endpoint,
    headers=headers,
    json=payload,
    timeout=60,
)

# Raise a clear exception for errors such as 401 or 404.
response.raise_for_status()

data = response.json()

print(
    data["choices"][0]["message"]["content"]
)