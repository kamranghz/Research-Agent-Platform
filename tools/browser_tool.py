import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def browser_tool(url: str) -> str:
    """
    Fetch a web page and return its readable text content.
    """

    try:
        # Send an HTTP GET request to the target URL.
        response = requests.get(
            url,
            timeout=15,
            headers={
                # Identify our application to the web server.
                "User-Agent": "research-agent-platform/1.0"
            },
        )

        # Raise an exception if the server returns an HTTP error
        # such as 404 or 500.
        response.raise_for_status()

    except requests.RequestException as exc:
        return f"Error: failed to fetch page: {exc}"

    # Parse the HTML document.
    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # Remove elements that are not useful for LLMs.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Extract readable text from the HTML.
    text = soup.get_text(
        separator="\n",
        strip=True,
    )

    # Return an error if no readable content exists.
    if not text:
        return "Error: no readable text found on the page."

    # Limit the returned text to avoid sending excessively
    # large prompts to the LLM.
    return text[:12000]