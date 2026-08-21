def test_streamlit_ui_imports() -> None:
    import ui.app as ui_app

    assert callable(ui_app.main)


def test_mcp_modules_import() -> None:
    from mcp_integration import client, server
    from tools.mcp_tool import mcp_tool

    assert callable(client.call_mcp_tool)
    assert server.server is not None
    assert mcp_tool.name == "mcp_tool"
