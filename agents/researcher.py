import json
import re

from config.settings import ENABLE_LONG_TERM_MEMORY
from core.state import AgentState
from memory.long_term import (
    format_memories_for_prompt,
    search_memories,
)
from memory.short_term import format_conversation_history
from models.llm import llm
from models.prompts import RESEARCHER_SYSTEM_PROMPT
from models.schemas import ResearchOutput


structured_researcher = llm.with_structured_output(
    ResearchOutput
)


def _extract_url(query: str) -> str:
    match = re.search(
        r"https?://[^\s<>\"']+",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(0).rstrip(".,;:!?)]}")


def _build_api_request_from_query(query: str) -> str:
    normalized_query = query.lower()

    resource_aliases = {
        "post": ("post", "posts", "پست"),
        "user": ("user", "users", "کاربر"),
        "todo": (
            "todo",
            "todos",
            "task",
            "tasks",
            "تودو",
            "وظیفه",
        ),
    }

    selected_resource = ""
    for resource_name, aliases in resource_aliases.items():
        if any(alias in normalized_query for alias in aliases):
            selected_resource = resource_name
            break

    id_match = re.search(r"\d+", query)
    if not selected_resource or not id_match:
        return ""

    try:
        item_id = int(id_match.group(0))
    except ValueError:
        return ""

    if item_id < 1:
        return ""

    return json.dumps(
        {"resource": selected_resource, "id": item_id},
        ensure_ascii=False,
    )


def _normalize_api_tool_input(tool_input: str, query: str) -> str:
    clean_input = tool_input.strip()

    if clean_input:
        try:
            parsed_input = json.loads(clean_input)
            if isinstance(parsed_input, dict):
                resource = parsed_input.get("resource")
                item_id = parsed_input.get("id")
                if (
                    resource in {"post", "user", "todo"}
                    and isinstance(item_id, int)
                    and not isinstance(item_id, bool)
                    and item_id >= 1
                ):
                    return json.dumps(
                        {"resource": resource, "id": item_id},
                        ensure_ascii=False,
                    )
        except json.JSONDecodeError:
            pass

    return _build_api_request_from_query(query)


def _normalize_mcp_tool_input(tool_input: str, query: str) -> str:
    """
    Ensure mcp_tool receives stable JSON.

    Expected shape:
    {"tool_name": "...", "arguments": {...}}
    """

    clean_input = tool_input.strip()
    if clean_input:
        try:
            parsed = json.loads(clean_input)
            if (
                isinstance(parsed, dict)
                and isinstance(parsed.get("tool_name"), str)
                and parsed["tool_name"].strip()
            ):
                arguments = parsed.get("arguments", {})
                if arguments is None:
                    arguments = {}
                if isinstance(arguments, dict):
                    return json.dumps(
                        {
                            "tool_name": parsed["tool_name"].strip(),
                            "arguments": arguments,
                        },
                        ensure_ascii=False,
                    )
        except json.JSONDecodeError:
            pass

    lowered = query.lower()
    if "project status" in lowered or "capabilities" in lowered:
        return json.dumps(
            {
                "tool_name": "get_project_status",
                "arguments": {},
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "tool_name": "search_knowledge_base",
            "arguments": {"query": query.strip()},
        },
        ensure_ascii=False,
    )


def _research_output_to_state(
    response: ResearchOutput,
    query: str,
    retry_count: int,
) -> dict:
    tool_input = response.tool_input.strip()

    if response.tool_name == "browser_tool" and not tool_input:
        tool_input = _extract_url(query)

    if response.tool_name == "search_tool" and not tool_input:
        tool_input = query.strip()

    if response.tool_name in {
        "rag_tool",
        "llamaindex_rag_tool",
    } and not tool_input:
        tool_input = query.strip()

    if response.tool_name == "api_tool":
        tool_input = _normalize_api_tool_input(
            tool_input=tool_input,
            query=query,
        )

    if response.tool_name == "mcp_tool":
        tool_input = _normalize_mcp_tool_input(
            tool_input=tool_input,
            query=query,
        )

    # Some models set tool_name=mcp_tool but leave needs_tool=false.
    # Only force the tool path for that known inconsistency.
    wants_tool = bool(response.needs_tool)
    if (
        not wants_tool
        and response.tool_name == "mcp_tool"
        and not response.draft_answer.strip()
    ):
        wants_tool = True

    # Preference / identity statements should be answered directly.
    # Do not send them to document retrieval tools.
    lowered_query = query.lower()
    memory_markers = (
        "remember that",
        "please remember",
        "my preferred",
        "i prefer",
        "my name is",
        "answer style",
    )
    if any(marker in lowered_query for marker in memory_markers):
        if response.tool_name in {
            "rag_tool",
            "llamaindex_rag_tool",
            "search_tool",
            "browser_tool",
        }:
            wants_tool = False
            tool_input = ""

    if wants_tool and response.tool_name and tool_input:
        return {
            "needs_tool": True,
            "draft_answer": "",
            "tool_name": response.tool_name,
            "tool_input": tool_input,
            "selected_tool_name": response.tool_name,
            "retry_count": retry_count,
        }

    draft = response.draft_answer.strip()
    if not draft and any(marker in lowered_query for marker in memory_markers):
        draft = (
            "Understood. I will remember that for future sessions "
            "when long-term memory is enabled."
        )

    return {
        "needs_tool": False,
        "draft_answer": draft,
        "tool_name": "",
        "tool_input": "",
        "retry_count": retry_count,
    }


def _build_context_blocks(state: AgentState) -> str:
    blocks: list[str] = []

    history = format_conversation_history(state.get("messages"))
    if history:
        blocks.append(history)

    memory_context = (state.get("long_term_memory_context") or "").strip()
    if memory_context:
        blocks.append(memory_context)

    return "\n\n".join(blocks)


def researcher_node(state: AgentState) -> dict:
    """
    Decide whether to request a tool or generate an answer.

    Long-term memories are injected as labeled optional context.
    Short-term history comes from checkpointed messages.
    """

    print(">>> researcher start", flush=True)

    query = state["query"]
    tool_result = state.get("tool_result", "")
    retry_count = state.get("retry_count", 0) + 1

    # Load long-term memories once per turn when enabled and empty.
    enable_ltm = state.get(
        "enable_long_term_memory",
        ENABLE_LONG_TERM_MEMORY,
    )
    memory_context = state.get("long_term_memory_context", "")
    memory_update: dict = {}

    if enable_ltm and not memory_context and not tool_result:
        memories = search_memories(
            query=query,
            user_id=state.get("user_id"),
        )
        memory_context = format_memories_for_prompt(memories)
        memory_update["long_term_memory_context"] = memory_context

    print(f">>> query: {query}", flush=True)
    print(f">>> tool_result: {tool_result}", flush=True)

    context_blocks = _build_context_blocks(
        {
            **state,
            "long_term_memory_context": memory_context,
        }
    )

    if tool_result:
        print(">>> generating answer from tool result...", flush=True)

        final_answer_prompt = f"""
You are writing the final answer to the user's question.

{context_blocks}

User question:
{query}

Tool result:
{tool_result}

Instructions:

- Answer the user's question using the tool result.
- Treat the tool result as the source of truth for documents.
- Long-term memory is optional preference/context only.
- Give a clear, concise, and relevant answer.
- When the tool result includes source names and page numbers,
  preserve them in the answer as citations.
- Do not invent information absent from the tool result.
- Do not invent citations, pages, files, or sources.
- Do not request another tool.
- Follow any preferred answer style from memory or conversation.
"""

        response = llm.invoke(final_answer_prompt)
        output = {
            "needs_tool": False,
            "draft_answer": response.content.strip(),
            "tool_name": "",
            "tool_input": "",
            "retry_count": retry_count,
            **memory_update,
        }
        print(">>> researcher output:", output, flush=True)
        return output

    print(">>> calling structured llm...", flush=True)

    user_content = query
    if context_blocks:
        user_content = f"{context_blocks}\n\nUser question:\n{query}"

    response = structured_researcher.invoke(
        [
            {
                "role": "system",
                "content": RESEARCHER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
    )

    print(">>> llm returned", flush=True)
    print(response, flush=True)

    output = _research_output_to_state(
        response=response,
        query=query,
        retry_count=retry_count,
    )
    output.update(memory_update)

    print(">>> researcher output:", output, flush=True)
    return output
