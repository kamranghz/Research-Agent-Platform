from core.state import AgentState
from tools.registry import TOOL_MAP


def tool_executor_node(state: AgentState) -> dict:
    """Execute the tool selected by the Researcher."""

    print(">>> tool executor start", flush=True)

    tool_name = state["tool_name"]
    tool_input = state["tool_input"].strip()
    current_attempts = state.get("tool_attempt_count", 0)

    print(f">>> tool_name: {tool_name}", flush=True)
    print(f">>> tool_input: {tool_input}", flush=True)

    if not tool_input:
        return {
            "tool_result": "Tool execution skipped because tool_input was empty.",
            "needs_tool": False,
            "tool_name": "",
            "tool_input": "",
            "selected_tool_name": tool_name,
            "tool_attempt_count": current_attempts + 1,
        }

    if tool_name not in TOOL_MAP:
        return {
            "tool_result": f"Unknown tool: {tool_name}",
            "needs_tool": False,
            "tool_name": "",
            "tool_input": "",
            "selected_tool_name": tool_name,
            "tool_attempt_count": current_attempts + 1,
        }

    # Each tool expects a different argument name/shape.
    if tool_name == "file_reader":
        tool_args = {"file_path": tool_input}

    elif tool_name == "browser_tool":
        tool_args = {"url": tool_input}

    elif tool_name == "search_tool":
        tool_args = {"query": tool_input}

    elif tool_name == "api_tool":
        tool_args = {"request": tool_input}

    elif tool_name in {"rag_tool", "llamaindex_rag_tool"}:
        tool_args = {"query": tool_input}

    elif tool_name == "mcp_tool":
        # mcp_tool parses a JSON string with tool_name + arguments.
        tool_args = {"request": tool_input}

    else:
        return {
            "tool_result": f"Unsupported tool: {tool_name}",
            "needs_tool": False,
            "tool_name": "",
            "tool_input": "",
            "selected_tool_name": tool_name,
            "tool_attempt_count": current_attempts + 1,
        }

    try:
        print(">>> invoking tool...", flush=True)
        result = TOOL_MAP[tool_name].invoke(tool_args)
        print(">>> tool finished", flush=True)
        print(result, flush=True)

    except Exception as exc:
        print(">>> tool exception", flush=True)
        print(exc, flush=True)

        return {
            "tool_result": f"Tool execution failed: {exc}",
            "needs_tool": False,
            "tool_name": "",
            "tool_input": "",
            "selected_tool_name": tool_name,
            "tool_attempt_count": current_attempts + 1,
            "error_message": str(exc),
        }

    output = {
        "tool_result": str(result),
        "needs_tool": False,
        "tool_name": "",
        "tool_input": "",
        "selected_tool_name": tool_name,
        "tool_attempt_count": current_attempts + 1,
        "error_message": "",
    }

    print(">>> tool executor output:", output, flush=True)
    return output
