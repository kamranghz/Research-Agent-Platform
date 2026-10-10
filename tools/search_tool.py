from langchain_core.tools import tool
from config.settings import TAVILY_API_KEY


_client = None


def _get_client():
    """Create the Tavily client lazily so imports work without a key."""

    global _client
    if _client is None:
        from tavily import TavilyClient

        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


@tool
def search_tool(query: str) -> str:
    """Search the web and return the most relevant results."""

    # Reject empty search queries.
    if not query.strip():
        return "Search query cannot be empty."

    if not TAVILY_API_KEY:
        return "Search is disabled: no TAVILY_API_KEY is configured."

    try:
        # Search the web and request up to five relevant results.
        response = _get_client().search(
            query=query.strip(),
            max_results=3,
        )

        results = response.get("results", [])

        if not results:
            return "No search results were found."

        formatted_results = []

        # Convert Tavily's dictionary response into readable text
        # that the Researcher can use as evidence.
        for index, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")[:1200]

            formatted_results.append(
                f"""
Result {index}
Title: {title}
URL: {url}
Content: {content}
""".strip()
            )

        return "\n\n".join(formatted_results)

    except Exception as error:
        # Return the error as tool output instead of crashing the graph.
        return f"Search failed: {error}"