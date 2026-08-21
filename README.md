<div align="center">

# Research Agent Platform

**Multi-Tool LangGraph Research Agent with Hybrid RAG, LlamaIndex, MCP & Memory**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-5C4EE5)](https://www.llamaindex.ai/)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-000000)](https://modelcontextprotocol.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F61)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-ffffff)](https://ollama.com/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron%20Reviewer-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)

*An interview-ready research agent that decides when to use tools, retrieves grounded evidence from documents and the web, reviews answer quality, and remembers useful preferences across sessions.*

[Overview](#overview) · [Architecture](#architecture) · [Repository Structure](#repository-structure) · [Installation](#installation) · [Quick Start](#quick-start) · [Running the Platform](#running-the-platform) · [Troubleshooting](#troubleshooting)

</div>

---

## Overview

Most LLM demos only chat. This platform builds a **tool-using research workflow** where the model must choose the right action, retrieve evidence, and survive a quality gate before an answer is returned.

| Capability | What it solves | Why it matters |
|------------|----------------|----------------|
| **Tool routing** | Prevents invalid / empty tool calls | Stops infinite tool loops |
| **Hybrid RAG** | Dense + lexical retrieval over PDFs | Finds exact tables, pages, and facts |
| **LlamaIndex RAG** | Alternative semantic retrieval backend | Enables retrieval comparison |
| **MCP tools** | Real local Model Context Protocol server | Standardized external tool access |
| **Answer review** | NVIDIA Nemotron scores draft quality | Rejects weak answers and retries |
| **Memory** | Short-term checkpoints + long-term facts | Multi-turn and cross-session continuity |

Typical research failures this system targets:

- Hallucinated citations with no document evidence
- Wrong tool selection or empty tool inputs
- Lost conversation context across turns
- No quality control before returning the final answer

> **Local-first demo:** the Researcher runs on Ollama. Document embeddings use `qwen3-embedding:0.6b`. The Reviewer can use NVIDIA NIM when an API key is available.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Research Agent Platform                              │
│                                                                              │
│   Input: user question (CLI / Streamlit) + thread_id + optional user_id    │
│                                                                              │
│   ┌──────────────────┐                                                      │
│   │    Researcher    │  structured output → answer OR one tool request      │
│   └────────┬─────────┘                                                      │
│            │                                                                 │
│            ▼                                                                 │
│   ┌──────────────────┐     needs_tool + valid input + attempt cap           │
│   │   Tool Router    │─────────────────────────────────────────────┐        │
│   └────────┬─────────┘                                             │        │
│            │ tool                                                  │ review │
│            ▼                                                       ▼        │
│   ┌──────────────────┐                                   ┌────────────────┐ │
│   │  Tool Executor   │  file / browser / search / API /  │    Reviewer    │ │
│   │                  │  rag / llamaindex / mcp           │ NVIDIA scores  │ │
│   └────────┬─────────┘                                   └────────┬───────┘ │
│            │ tool_result                                           │        │
│            ▼                                                       ▼        │
│   ┌──────────────────┐                         approved / max_retry │        │
│   │    Researcher    │  draft answer from evidence                 │        │
│   └────────┬─────────┘                                             │        │
│            └─────────────────── review ────────────────────────────┘        │
│                                            │                                 │
│                                            ▼                                 │
│                                   ┌──────────────────┐                      │
│                                   │    Finalizer     │                      │
│                                   │ final answer +   │                      │
│                                   │ memory updates   │                      │
│                                   └──────────────────┘                      │
│                                                                              │
│   Output: final_answer · selected tool · evidence · review scores           │
└────────────────────────────────────────────────────────────────────────────┘
```

### Memory Flow

```
Current session (thread_id)
        │
        ▼
Short-term checkpoint memory  ──► Researcher prompt context
(LangGraph MemorySaver / SQLite)

Useful durable facts / preferences
        │
        ▼
Long-term memory store  ──► labeled optional context
(Chroma: agent_long_term_memory)
```

### Tooling Surface

| Tool | Input contract | Purpose |
|------|----------------|---------|
| `file_reader` | `{ "file_path": "..." }` | Read local `.txt` / `.md` |
| `browser_tool` | `{ "url": "..." }` | Fetch HTML page text |
| `search_tool` | `{ "query": "..." }` | Web search |
| `api_tool` | `{ "request": "{\"resource\":\"post\",\"id\":1}" }` | Registered REST resources |
| `rag_tool` | `{ "query": "..." }` | Default hybrid PDF retrieval |
| `llamaindex_rag_tool` | `{ "query": "..." }` | LlamaIndex semantic retrieval |
| `mcp_tool` | `{ "request": "{\"tool_name\":\"search_knowledge_base\",\"arguments\":{\"query\":\"...\"}}" }` | Local MCP server tools |

### Component Details

#### Researcher (`agents/researcher.py`)

Uses structured output to choose exactly one path:

1. Answer directly, or
2. Request one registered tool

It also normalizes common model mistakes (missing URL, missing RAG query, MCP JSON shape) before routing.

#### Tool Router + Executor (`core/router.py`, `core/tool_executor.py`)

Routes only when:

- `needs_tool` is true
- tool name is registered
- `tool_input` is non-empty
- tool attempts remain under `MAX_TOOL_ATTEMPTS`

Each tool receives the correct argument shape through a simple `if/elif` dispatcher.

#### Reviewer (`agents/reviewer.py`)

NVIDIA Nemotron evaluates:

- accuracy
- relevance
- clarity
- completeness

Threshold failures send the draft back to the Researcher until approval or max retry.

#### RAG Backends (`rag/`)

- **Hybrid RAG:** Chroma dense retrieval + lexical reranking, with source/page/chunk scores
- **LlamaIndex RAG:** separate Chroma collection for comparative semantic retrieval

#### MCP Integration (`mcp_integration/`)

Local stdio MCP server exposing:

- `search_knowledge_base(query)`
- `get_project_status(detail_level?)`

Wrapped for LangGraph as `mcp_tool`.

---

## Repository Structure

```
Research-Agent-Platform/
├── agents/
│   ├── researcher.py              # Tool selection + grounded drafting
│   └── reviewer.py                # NVIDIA Nemotron quality gate
├── app/
│   └── main.py                    # CLI entrypoint
├── config/
│   └── settings.py                # Env-driven configuration
├── core/
│   ├── graph.py                   # LangGraph compile + checkpointer
│   ├── invoke.py                  # Thread-scoped turn helper
│   ├── router.py                  # Tool / review routing
│   ├── state.py                   # AgentState
│   ├── tool_executor.py           # Tool dispatch
│   └── finalizer.py               # Final answer + memory write
├── mcp_integration/
│   ├── server.py                  # Local MCP stdio server
│   └── client.py                  # MCP client wrapper
├── memory/
│   ├── short_term.py              # LangGraph checkpointer memory
│   └── long_term.py               # Durable preference/fact store
├── models/
│   ├── llm.py                     # Ollama chat model
│   ├── prompts.py                 # System prompts
│   └── schemas.py                 # Structured output schemas
├── rag/
│   ├── retriever_tool.py          # Hybrid RAG tool
│   ├── llamaindex_*.py            # LlamaIndex backend / tool
│   ├── loader.py / splitter.py    # PDF ingestion helpers
│   └── vector_store.py            # Chroma persistence
├── tools/
│   ├── registry.py                # Single source of tool truth
│   ├── file_reader.py
│   ├── browser_tool.py
│   ├── search_tool.py
│   ├── api_tool.py
│   └── mcp_tool.py
├── ui/
│   └── app.py                     # Streamlit chat UI + charts
├── scripts/
│   ├── ingest_documents.py
│   ├── ingest_llamaindex.py
│   ├── compare_rag_backends.py
│   ├── test_mcp.py
│   ├── test_short_term_memory.py
│   └── test_long_term_memory.py
├── tests/
├── datasets/
│   └── sample.pdf                 # Demo knowledge-base PDF
├── requirements.txt
├── README.md
└── .gitignore

# ── NOT committed — generated locally ────────────────────────────
# data/chroma/            vector stores + long-term memory
# data/checkpoints.sqlite short-term memory (optional)
# .env                    API keys / local secrets
```

---

## Installation

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com/) running locally

```bash
# 1. Clone
git clone https://github.com/kamranghz/Research-Agent-Platform.git
cd Research-Agent-Platform

# 2. Create / activate environment
conda create -n jiuwen-agent python=3.11 -y
conda activate jiuwen-agent

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull local models
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3-embedding:0.6b

# 5. Configure environment
cp .env.example .env   # or create .env manually
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` | Researcher LLM |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embeddings (1024-d) |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Vector store root |
| `CHROMA_COLLECTION_NAME` | `research_documents` | Hybrid RAG collection |
| `LLAMAINDEX_CHROMA_COLLECTION_NAME` | `research_documents_llamaindex` | LlamaIndex collection |
| `MEMORY_BACKEND` | `memory` | `memory` or `sqlite` |
| `ENABLE_LONG_TERM_MEMORY` | `true` | Durable memory toggle |
| `NVIDIA_API_KEY` | _(empty)_ | Reviewer via NVIDIA NIM |
| `TAVILY_API_KEY` | _(empty)_ | Optional web search |
| `MCP_CALL_TIMEOUT_SECONDS` | `60` | MCP client timeout |

---

## Quick Start

### 1) Ingest the sample PDF

```bash
python -m scripts.ingest_documents datasets/sample.pdf --reset
python -m scripts.ingest_llamaindex datasets/sample.pdf
```

### 2) Ask a grounded research question

```bash
python -u -m app.main
```

Expected behavior for:

> Based on the indexed PDF, what MCP server rate limitations are listed?

- selects `rag_tool`
- retrieves `sample.pdf` / page 25 / Table 8
- returns a cited answer
- reviewer scores the draft

### 3) Launch the UI

```bash
python -m streamlit run ui/app.py
```

---

## Running the Platform

### Compare retrieval backends

```bash
python -m scripts.compare_rag_backends "MCP server rate limitations"
```

### MCP smoke test

```bash
python -m scripts.test_mcp
```

MCP tools:

- `search_knowledge_base(query)`
- `get_project_status(detail_level?)`

Example `mcp_tool` input:

```json
{
  "tool_name": "search_knowledge_base",
  "arguments": {
    "query": "MCP server rate limitations"
  }
}
```

### Memory demos

```bash
# Short-term conversation continuity (same thread_id)
python -m scripts.test_short_term_memory

# Long-term preference persistence (new session / thread)
python -m scripts.test_long_term_memory
```

### Tests

```bash
python -m compileall .
python -m pytest tests -q --ignore=tests/test_nvidia_api.py --ignore=tests/test_structured_output.py
```

### Example queries

| Query | Expected tool |
|-------|----------------|
| Based on the indexed PDF, what MCP server rate limitations are listed? | `rag_tool` |
| Use LlamaIndex to find the MCP server rate limitations. | `llamaindex_rag_tool` |
| Use the MCP knowledge-base tool to find the MCP server rate limitations. | `mcp_tool` |
| Remember that my preferred answer style is concise and technical. | direct answer + long-term memory |
| Summarize https://example.com | `browser_tool` |

---

## Troubleshooting

**`ModuleNotFoundError` for project packages**

Run commands from the repository root with your conda env activated.

**Ollama connection errors**

```bash
ollama serve
ollama list
```

**Empty RAG results**

```bash
python -m scripts.ingest_documents datasets/sample.pdf --reset
python -m scripts.ingest_llamaindex datasets/sample.pdf
```

**NVIDIA reviewer unavailable / 503**

The graph continues with fallback review scores so demos remain runnable. Set a valid `NVIDIA_API_KEY` for full reviewer behavior.

**MCP tools not discovered**

```bash
python -m scripts.test_mcp
```

The MCP server is launched over stdio by the client; ensure project imports resolve from the repo root.

**Cursor / co-author appearing on GitHub**

This repository history is authored under your GitHub identity only. Recreate/push from a clean local commit if needed.

---

## Design Decisions

1. **LangGraph over a single prompt chain** — explicit control of tool loops, retries, and state
2. **Hybrid RAG as default** — lexical matching helps recover exact tables/IDs that pure dense search can miss
3. **LlamaIndex as an alternative backend** — useful for comparison, not a replacement of the main graph
4. **Real MCP stdio server** — no fake abstraction layer; tools are discoverable and callable
5. **Separated memories** — short-term for conversation continuity, long-term for durable preferences/facts only
6. **Simple tool executor** — readable `if/elif` argument mapping instead of heavy factories

---

<div align="center">
<sub>Research Agent Platform · LangGraph · Hybrid RAG · LlamaIndex · MCP · ChromaDB · Ollama · NVIDIA Nemotron · Streamlit</sub>
</div>
