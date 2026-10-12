<div align="center">

# Research Agent Platform

**A production-grade autonomous research agent — LangGraph orchestration, hybrid RAG, multi-tool execution, QLoRA fine-tuning, and GRPO reinforcement learning**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-FF6B35?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Eval-1C3A5E?logo=langchain&logoColor=white)](https://smith.langchain.com/)
[![Qwen3-4B](https://img.shields.io/badge/Qwen3--4B-QLoRA%20%2B%20GRPO-7B2FBE)](https://huggingface.co/Qwen/Qwen3-4B)
[![Azure](https://img.shields.io/badge/Azure-Free%20Tier-0078D4?logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Autonomous research · Hybrid RAG · Tool routing · LLM fine-tuning · Reinforcement learning from AI feedback*

[Overview](#overview) · [Architecture](#architecture) · [Installation](#installation) · [Configuration](#configuration) · [Running the App](#running-the-app) · [Fine-tuning](#fine-tuning) · [Eval Results](#evaluation-results) · [Deployment](#deployment)

</div>

---

## Overview

**Research Agent Platform** is an end-to-end autonomous research system built on LangGraph. Given a research query, the agent:

- Routes the query through a **five-node LangGraph pipeline**: Researcher → Router → Tool Executor → Reviewer → Finalizer
- Selects the right tool dynamically — web search, file reading, browser scraping, REST API calls, hybrid RAG over PDFs, or MCP servers
- Cross-checks its own output with a **dual LLM reviewer** (NVIDIA Nemotron + gemma-reviewer via Ollama) before finalising
- Logs every run to **LangSmith** for tracing and evaluation
- Serves a **Streamlit UI** that can be deployed to **Azure App Service (free tier)**

The model powering tool-routing decisions is **Qwen3-4B**, fine-tuned in three phases — SFT, LangSmith-guided evaluation, and GRPO reinforcement learning — all scripts and adapter configs included in this repo.

---

## Architecture

```
User Query (Streamlit UI)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                   │
│                                                        │
│  ┌─────────────┐    ┌──────────┐    ┌───────────────┐  │
│  │  Researcher  │───▶│  Router  │───▶│ Tool Executor │  │
│  └─────────────┘    └──────────┘    └───────┬───────┘  │
│                                             │          │
│                                     ┌───────▼───────┐  │
│                                     │   Reviewer    │  │
│                                     │ (Nemotron +   │  │
│                                     │  gemma-rev.)  │  │
│                                     └───────┬───────┘  │
│                                             │          │
│                                     ┌───────▼───────┐  │
│                                     │   Finalizer   │  │
│                                     └───────────────┘  │
└────────────────────────────────────────────────────────┘
         │
         ▼
   Final Answer + Sources
```

### Nodes

| Node | Role |
|---|---|
| **Researcher** | Decomposes the query, forms a retrieval plan |
| **Router** | Selects the best tool for each sub-task (fine-tuned Qwen3-4B) |
| **Tool Executor** | Runs the selected tool and collects evidence |
| **Reviewer** | Validates evidence quality; loops back if insufficient |
| **Finalizer** | Synthesises evidence into a final grounded answer |

### Tools

| Tool | Description |
|---|---|
| `search_tool` | Web search via SerpAPI / DuckDuckGo |
| `browser_tool` | Headless browser scraping (Playwright) |
| `file_reader` | Read and parse local files (PDF, TXT, DOCX) |
| `api_tool` | Authenticated REST API calls |
| `rag_tool` | Hybrid dense + lexical retrieval over indexed PDFs (ChromaDB) |
| `llamaindex_rag_tool` | Alternative retrieval backend via LlamaIndex |
| `mcp_tool` | Delegate to any connected MCP server |

### Memory

- **Short-term**: LangGraph checkpoints (in-memory or SQLite) — maintains conversation context across turns
- **Long-term** *(optional)*: durable store backend (configurable in `config/settings.py`)

### Reviewer

- **Primary**: NVIDIA Nemotron (cloud-hosted, zero-cost free tier)
- **Secondary**: `gemma-reviewer:latest` via Ollama (local, run with `ollama serve`)

---

## Project Structure

```text
research-agent-platform/
│
├── agents/
│   ├── researcher.py          # Researcher node — query decomposition
│   └── reviewer.py            # Reviewer node — evidence validation
│
├── core/
│   ├── graph.py               # LangGraph graph definition
│   ├── router.py              # Tool routing logic
│   └── finalizer.py           # Answer synthesis
│
├── tools/
│   ├── search_tool.py
│   ├── browser_tool.py
│   ├── file_reader.py
│   ├── api_tool.py
│   └── mcp_tool.py
│
├── rag/
│   ├── embeddings.py          # Embedding pipeline (dense retrieval)
│   ├── hybrid_retriever.py    # Dense + lexical hybrid search
│   └── llamaindex_backend.py  # LlamaIndex alternative backend
│
├── models/
│   └── prompts.py             # All system and few-shot prompts
│
├── config/
│   └── settings.py            # Centralised configuration (env-driven)
│
├── ui/
│   └── app.py                 # Streamlit frontend
│
├── finetuning/
│   ├── train_sft.py                    # Phase 1 — QLoRA SFT
│   ├── evaluate_langsmith.py           # Phase 2 — LangSmith eval
│   ├── grpo_answer_quality.py          # Phase 3 — GRPO RL training
│   ├── evaluate_answer_grpo.py         # GRPO vs baseline comparison
│   ├── data/
│   │   └── real_world/
│   │       └── answer_grpo_data.jsonl  # 39 GRPO training examples
│   └── outputs/
│       ├── evaluations/
│       │   ├── langsmith_eval_v1/
│       │   └── answer_grpo_vs_baseline/
│       │       └── comparison.json
│       └── qwen3_4b_instruct_2507_tool_routing_pilot/
│           └── final_adapter_v3_grpo/  # Adapter config + tokenizer (weights on HF Hub)
│
├── tests/
│   ├── test_graph_execution.py
│   ├── test_browser_integration_fixed.py
│   ├── test_review_evidence_selection.py
│   └── test_tool_routing_scoring.py
│
├── Modelfile.gemma-reviewer    # Ollama Modelfile for gemma reviewer
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local reviewer — optional but recommended)
- Git

### 1 — Clone the repository

```bash
git clone https://github.com/kamranghz/Research-Agent-Platform.git
cd Research-Agent-Platform
```

### 2 — Create a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows note**: if you see `AttributeError: 'NoneType' object has no attribute 'apply'` at import time, this is a Triton unavailability issue on Windows. It is handled automatically by `install_logprob_shim()` in the fine-tuning scripts. No action needed for the main agent.

### 4 — (Optional) Pull the gemma reviewer model

```bash
ollama create gemma-reviewer -f Modelfile.gemma-reviewer
```

To keep the model warm in VRAM during long runs:

```bash
# PowerShell (Windows)
Start-Job { ollama run gemma-reviewer "warm" }

# bash (Linux / macOS)
ollama run gemma-reviewer "warm" &
```

---

## Configuration

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# ── LLM provider ──────────────────────────────────────────────
OPENAI_API_KEY=sk-...            # or any OpenAI-compatible key

# ── Search ────────────────────────────────────────────────────
SERPAPI_KEY=...                  # optional; falls back to DuckDuckGo

# ── LangSmith (evaluation & tracing) ──────────────────────────
LANGSMITH_API_KEY=ls-...
LANGSMITH_PROJECT=research-agent-platform
LANGSMITH_TRACING=true

# ── RAG (ChromaDB) ────────────────────────────────────────────
CHROMA_PERSIST_DIR=./chroma_db   # local directory for vector store
PDF_INDEX_DIR=./datasets          # directory containing PDFs to index

# ── Ollama reviewer (local) ───────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_REVIEWER_MODEL=gemma-reviewer

# ── Memory ────────────────────────────────────────────────────
CHECKPOINT_BACKEND=memory        # "memory" or "sqlite"
SQLITE_PATH=./checkpoints.db     # only used when backend = sqlite
```

> **Security**: never commit `.env`. It is in `.gitignore` by default. All secrets go here or in Azure App Settings — never in code.

---

## Running the App

```bash
python -m streamlit run ui/app.py
```

The UI opens at `http://localhost:8501`.

### Running tests

```bash
pytest tests/ -v
```

---

## Fine-tuning

The `finetuning/` directory contains the full three-phase fine-tuning pipeline for **Qwen3-4B** on tool-routing and answer-quality tasks.

### Phase 1 — Supervised Fine-Tuning (SFT)

QLoRA fine-tuning on curated tool-routing examples.

```bash
python finetuning/train_sft.py
```

Config highlights:
- Base model: `Qwen/Qwen3-4B-Instruct`
- LoRA rank 16, alpha 32
- 4-bit NF4 quantisation, BF16 compute
- ~200 training examples

### Phase 2 — LangSmith Evaluation

Runs the fine-tuned adapter against 8 held-out examples and logs results to LangSmith.

```bash
python finetuning/evaluate_langsmith.py
```

Result: **8/8 correct** on the evaluation set (100% accuracy).

Traces visible at: [smith.langchain.com/projects/research-agent-platform](https://smith.langchain.com/projects/research-agent-platform)

### Phase 3 — GRPO Reinforcement Learning

Group Relative Policy Optimisation (GRPO) on answer quality using AI feedback as the reward signal.

```bash
python finetuning/grpo_answer_quality.py
```

Config highlights:
- Reward model: gemma-reviewer via Ollama
- 93 training steps, 39 examples (31 train / 8 eval)
- `use_vllm=False`, `config.use_liger_kernel=False` (Windows-compatible)
- Pure-PyTorch logprob shim (`install_logprob_shim`) — no Triton required

Evaluate the trained adapter:

```bash
python finetuning/evaluate_answer_grpo.py \
    --grpo-adapter finetuning/outputs/qwen3_4b_instruct_2507_tool_routing_pilot/final_adapter_v3_grpo
```

---

## Evaluation Results

### LangSmith Tool-Routing Eval (Phase 2)

| Metric | Score |
|---|---|
| Accuracy (8 examples) | **8 / 8 (100%)** |
| Correct tool selections | 8 |

### GRPO Answer-Quality Eval (Phase 3)

| Model | Mean Score | Notes |
|---|---|---|
| Baseline (Qwen3-4B-Instruct) | **0.970** | Already high — ceiling effect |
| GRPO adapter (v3) | **0.960** | Noisy signal due to Ollama timeout |

> **Interpretation**: The baseline already scored 0.97/1.0, indicating a ceiling effect on this 8-example set. The next iteration will use harder examples and a stable reward judge to produce a clearer signal. The adapter config and tokenizer are committed to this repo; the model weights (`adapter_model.safetensors`) are hosted on HuggingFace Hub — link to be added when uploaded.

---

## Deployment

### Azure App Service (Free Tier — $0/month)

**Tier**: F1 (Free) — 60 CPU-minutes/day, 1 GB RAM, shared compute.

#### 1 — Build and push Docker image

```bash
# Build
docker build -t research-agent-platform .

# Tag and push to Azure Container Registry (free tier: 10 GB)
az acr login --name <your-acr-name>
docker tag research-agent-platform <your-acr-name>.azurecr.io/research-agent-platform:latest
docker push <your-acr-name>.azurecr.io/research-agent-platform:latest
```

#### 2 — Create App Service

```bash
az webapp create \
  --resource-group <your-rg> \
  --plan <your-plan> \
  --name <your-app-name> \
  --deployment-container-image-name <your-acr-name>.azurecr.io/research-agent-platform:latest
```

#### 3 — Set environment variables

In **Azure Portal → App Service → Configuration → Application Settings**, add all keys from `.env` (never put secrets in code or Docker image).

#### Live URL

`https://<your-app-name>.azurewebsites.net`

---

## Model Weights (HuggingFace Hub)

The GRPO-trained adapter (`final_adapter_v3_grpo`) will be uploaded to:

> 🔗 **`[TO BE ADDED — kamranghz/qwen3-4b-research-agent-grpo]`**

The adapter config and tokenizer are already committed to this repo at:

```
finetuning/outputs/qwen3_4b_instruct_2507_tool_routing_pilot/final_adapter_v3_grpo/
```

---

## Roadmap

- [x] Phase 1 — QLoRA SFT on tool routing
- [x] Phase 2 — LangSmith evaluation (8/8)
- [x] Phase 3 — GRPO answer-quality RL
- [ ] Upload adapter weights to HuggingFace Hub
- [ ] Harder evaluation examples (avoid ceiling effect)
- [ ] Stable reward judge (persistent Ollama VRAM management)
- [ ] Azure deployment (live URL)
- [ ] MCP server integration examples

---

## License

[MIT License](LICENSE)

---

## Author

**Kamran Gholizadeh HamlAbadi**  
PhD Candidate, University of Ottawa · MCRLab  
[github.com/kamranghz](https://github.com/kamranghz) · [LinkedIn](https://www.linkedin.com/in/kamrangh)
