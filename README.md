<div align="center">

# Research Agent Platform

**A LangGraph research agent with hybrid RAG, multi-tool execution, an independent local reviewer, and a QLoRA / GRPO fine-tuning experiment**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-FF6B35?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[Overview](#overview) · [Architecture](#architecture) · [Installation](#installation) · [Configuration](#configuration) · [Running](#running-the-app) · [Fine-tuning](#fine-tuning-experiments) · [Results](#results) · [Deployment](#deployment) · [Limitations](#limitations)

</div>

---

## Overview

Given a question, the agent runs a LangGraph workflow:

**Researcher → Tool Router → Tool Executor → Reviewer → Finalizer**

- The **Researcher** decides whether a tool is needed and which one (structured output: `needs_tool`, `tool_name`, `tool_input`, `draft_answer`), then writes the answer from the tool result.
- The **Tool Executor** runs one of seven tools and returns the evidence.
- The **Reviewer** is a separate, local, non-Qwen model (default `gemma-reviewer` via Ollama). It scores the draft for accuracy, relevance, clarity and completeness, and can send the draft back for a retry. If the judge is unreachable the answer is labelled **not reviewed**; scores are never fabricated.
- The **Finalizer** produces the final answer and stores the turn in short-term memory.

The repository also contains a small fine-tuning experiment (QLoRA SFT and GRPO on Qwen3-4B-Instruct-2507). It is an experiment: **the production graph does not use the fine-tuned adapter** (see [Results](#results)).

---

## Architecture

```text
User question (Streamlit UI / CLI)
        │
        ▼
 Researcher ──▶ Router ──▶ Tool Executor ──▶ Reviewer ──▶ Finalizer ──▶ Answer
      ▲                                         │
      └──────────── retry (bounded) ◀───────────┘
```

### Tools

| Tool | Purpose |
|---|---|
| `file_reader` | Read local text/PDF files (scoped to `datasets/` in demo mode) |
| `browser_tool` | Fetch and read a given URL (off in demo mode) |
| `search_tool` | Web search via Tavily (needs `TAVILY_API_KEY`) |
| `api_tool` | Fetch a `post` / `user` / `todo` record by id from a registered API; input is a JSON string `{"resource": ..., "id": <positive int>}` |
| `rag_tool` | Retrieval over the indexed documents (ChromaDB) |
| `llamaindex_rag_tool` | The same task through the LlamaIndex backend |
| `mcp_tool` | Call the connected MCP service (`get_project_status`, `search_knowledge_base`, ...) |

### Models

| Role | Default | Notes |
|---|---|---|
| Researcher | Ollama (local) | `LLM_PROVIDER=ollama \| nvidia \| groq`, with optional `LLM_FALLBACK_PROVIDERS`. Hosted defaults: NVIDIA `nvidia/nemotron-3-super-120b-a12b`, Groq `llama-3.3-70b-versatile` |
| Reviewer | `gemma-reviewer` (Ollama, local) | Must be a non-Qwen model on localhost; there is no remote or Qwen fallback |
| Embeddings | Ollama (`EMBEDDING_PROVIDER=ollama`) | Hosted option: NVIDIA embeddings |

### Memory

- **Short-term**: LangGraph checkpoints per `thread_id` (`MEMORY_BACKEND`, SQLite optional).
- **Long-term** (optional, `ENABLE_LONG_TERM_MEMORY`): off in demo mode.

---

## Project structure

```text
agents/        researcher.py, reviewer.py
core/          graph.py, router.py, tool_executor.py, finalizer.py, invoke.py, limits.py, state.py
tools/         file_reader, browser_tool, search_tool, api_tool, mcp_tool, registry
rag/           loaders, splitter, embeddings, vector store, LlamaIndex backend
models/        llm.py (provider factory), prompts.py, schemas.py
config/        settings.py (env-driven configuration)
ui/            app.py (Streamlit)
deploy/        DEPLOY.md, entrypoint.sh (Azure Container Apps guide)
finetuning/    dataset builders, evaluation scripts, GRPO scripts, outputs/
tests/         unit and integration tests
docs/          architecture, experiments, failure modes, roadmap
```

---

## Installation

Prerequisites: Python 3.11+, Git, and [Ollama](https://ollama.ai/) for the local models.

```bash
git clone <this repository>
cd research-agent-platform
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Create the local reviewer:

```bash
ollama create gemma-reviewer -f Modelfile.gemma-reviewer
```

The fine-tuning scripts additionally need a CUDA GPU with `torch`, `transformers`, `peft`, `trl`, `bitsandbytes` and `datasets` (not in `requirements.txt`).

---

## Configuration

```bash
cp .env.example .env      # then fill in what you use
```

Main variables (see `.env.example` for the full list):

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER`, `LLM_FALLBACK_PROVIDERS` | Researcher provider: `ollama`, `nvidia`, `groq` |
| `NVIDIA_API_KEY`, `GROQ_API_KEY` | Keys for hosted providers |
| `EMBEDDING_PROVIDER` | `ollama` or `nvidia` |
| `TAVILY_API_KEY` | Enables `search_tool` |
| `LOCAL_REVIEWER_MODEL`, `LOCAL_REVIEWER_BASE_URL` | Local reviewer (default `gemma-reviewer`, `http://127.0.0.1:11434/v1`) |
| `CHROMA_PERSIST_DIR` | Vector store location |
| `MEMORY_BACKEND`, `ENABLE_LONG_TERM_MEMORY` | Memory settings |
| `DEMO_MODE`, `DEMO_MAX_QUESTIONS_PER_SESSION`, `DEMO_MAX_QUESTIONS_TOTAL` | Public-demo lock-down (browser off, long-term memory off, `file_reader` limited to `datasets/`, question caps) |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional LangSmith tracing |

Never commit `.env`; it is git-ignored.

---

## Running the app

```bash
python -m streamlit run ui/app.py
```

The UI opens at `http://localhost:8501`. Run the tests with:

```bash
python -m pytest tests/ -q
```

Some tests need a running Ollama or network access and are skipped when those are unavailable.

---

## Fine-tuning experiments

All scripts are run from the project root as modules (`python -m finetuning.<name>`).

### 1. Tool-routing SFT (QLoRA)

- Base model `Qwen/Qwen3-4B-Instruct-2507`, 4-bit NF4, LoRA r=16 / alpha=32, bf16.
- Data: **79** human-verified tool-routing records (`tool_routing_verified_sft.jsonl`), split with seed 42 into 71 train / **8 held-out validation**.
- A second round (`final_adapter_v2`) continued from the first adapter for 1 epoch at lr 5e-5 on `tool_routing_sft_v2.jsonl` (79 originals + 10 `llamaindex_rag_tool` records). Those 10 records were written by the author/assistant and were **not independently reviewed**.
- Evaluation: `python -m finetuning.evaluate_tool_routing_pilot`, which compares the base model and the adapter under three prompts (`train`, `contract`, `production`).

> The SFT training scripts (`train_tool_routing_qlora.py`, `retrain_tool_routing_v2.py`) are not present in the current working tree.

### 2. Graph-level routing check

`python -m finetuning.evaluate_langsmith` sends the same 8 held-out queries through the real LangGraph graph (one thread per example, traced to LangSmith). It measures the **configured provider's** routing, not the adapter.

### 3. GRPO

- `grpo_tool_routing.py`: GRPO for tool routing with hard-case mining. Reward is tool correctness (and optionally the graph's reviewer scores).
- `collect_answer_grpo_data.py`, `grpo_answer_quality.py`, `evaluate_answer_grpo.py`: GRPO for answer quality. `answer_grpo_data.jsonl` holds **39** collected examples (31 train / 8 eval); the reward is the reviewer node's weighted score. `use_vllm=False`, and `install_logprob_shim()` computes log-probabilities in plain PyTorch because TRL's fused kernel needs Triton, which is unavailable on Windows.

---

## Results

All numbers come from files under `finetuning/outputs/evaluations/`. Every evaluation set has **8 examples**, so differences of one or two examples are within noise.

### Tool routing: correct tool / exact tool call (8 held-out examples)

| Model, prompt | Tool selection | Exact tool call (name + input) |
|---|---|---|
| Base, training prompt | 0/8 | 0/8 |
| Base, production prompt | 8/8 | 3/8 |
| SFT adapter v1, production prompt | 7/8 | 3/8 |
| SFT adapter v2, production prompt | 8/8 | 5/8 |

The base model with the production prompt already matches the adapter on tool selection, so the original "base 0/8" baseline was unfair (it used a prompt the base model was never given the contract for). The adapter's measured advantage is in argument formatting only, and that is based on 8 examples.

### Graph-level routing (production graph, 8 examples)

8/8 queries routed to the expected tool (`finetuning/outputs/evaluations/langsmith_eval_v1/results.json`). This used the configured provider, not the fine-tuned adapter, so it does **not** measure the adapter.

### GRPO answer quality (8 examples, reviewer-scored)

| Model | Accuracy | Relevance | Clarity | Completeness | Overall |
|---|---|---|---|---|---|
| Baseline Qwen3-4B-Instruct | 0.925 | 1.000 | 1.000 | 1.000 | **0.970** |
| GRPO adapter | 0.900 | 1.000 | 1.000 | 1.000 | **0.960** |

GRPO won 0 of 8 comparisons. The baseline is already near the ceiling, so this experiment shows **no improvement**. The directory `final_adapter_v3_grpo/` in the repo holds only a config, tokenizer and a `manifest.json` from a tool-routing GRPO run that found 0 hard cases and trained 0 epochs; it contains no adapter weights.

---

## Deployment

`deploy/DEPLOY.md` documents the intended zero-fixed-cost demo: a public container image on GHCR run on **Azure Container Apps (Consumption plan)**, no Log Analytics workspace and no paid container registry, with `DEMO_MODE=1`. The first request after idle time is slow (scale-to-zero cold start).

The repository also contains a Hugging Face Space configuration (`Dockerfile`). No public demo URL has been verified yet.

---

## Limitations

- The fine-tuned adapters are experimental and are not wired into the agent.
- Evaluation sets are tiny (8 examples); no confidence intervals are reported.
- The 10 added `llamaindex_rag_tool` SFT records are author-generated.
- Reviewer scores come from a small local model and are noisy.
- `rag_tool` vs `llamaindex_rag_tool` selection is the hardest routing case.

---

## License

[MIT](LICENSE)

## Author

**Kamran Gholizadeh Hamlabadi** · PhD Candidate, University of Ottawa · [github.com/kamranghz](https://github.com/kamranghz)
