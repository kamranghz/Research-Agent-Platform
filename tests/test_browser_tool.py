from unittest.mock import MagicMock, patch

from tools.browser_tool import browser_tool


@patch("tools.browser_tool.requests.get")
def test_browser_tool_fetches_page(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Example Domain</p></body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = browser_tool.invoke({"url": "https://example.com"})

    mock_get.assert_called_once()
    assert "Example Domain" in result
