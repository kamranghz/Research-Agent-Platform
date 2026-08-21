import json

import requests
from langchain_core.tools import tool


# Internal API registry.
# The user never needs to know these URLs.
API_ROUTES = {
    "post": "https://jsonplaceholder.typicode.com/posts/{id}",
    "user": "https://jsonplaceholder.typicode.com/users/{id}",
    "todo": "https://jsonplaceholder.typicode.com/todos/{id}",
}


@tool
def api_tool(request: str) -> str:
    """
    Fetch structured JSON data from a registered API.

    Input must be a JSON string containing:
    - resource: post, user, or todo
    - id: a positive integer

    Example:
    {"resource": "post", "id": 1}
    """

    clean_request = request.strip()

    if not clean_request:
        return "API request cannot be empty."

    try:
        # Convert the model-generated JSON string into a dictionary.
        request_data = json.loads(clean_request)

    except json.JSONDecodeError:
        return (
            "API request must be valid JSON. "
            'Example: {"resource": "post", "id": 1}'
        )

    resource = str(
        request_data.get("resource", "")
    ).strip().lower()

    item_id = request_data.get("id")

    # Only registered API resources are allowed.
    if resource not in API_ROUTES:
        available_resources = ", ".join(API_ROUTES.keys())

        return (
            f"Unknown API resource: {resource}. "
            f"Available resources: {available_resources}."
        )

    # Validate the requested record ID.
    if not isinstance(item_id, int) or item_id < 1:
        return "API id must be a positive integer."

    # Build the endpoint internally.
    endpoint = API_ROUTES[resource].format(id=item_id)

    try:
        response = requests.get(
            endpoint,
            timeout=15,
            headers={
                "Accept": "application/json",
                "User-Agent": "research-agent-platform/1.0",
            },
        )

        response.raise_for_status()

        data = response.json()

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )[:12000]

    except requests.exceptions.Timeout:
        return "API request failed: request timed out."

    except requests.exceptions.RequestException as error:
        return f"API request failed: {error}"

    except ValueError:
        return "API request failed: endpoint returned invalid JSON."