# Research Agent Platform — Architecture

## Purpose

Research Agent Platform is a modular LangGraph-based system for building, tracing, evaluating, and extending AI agents.

The platform is designed to support two related directions:

1. A general AI agent platform with workflows, tool calling, RAG, memory, planning, MCP, evaluation, and multi-agent collaboration.
2. A web-agent research platform for studying browser interaction, web representations, failure modes, reliability gaps, and recovery strategies.

## Current Workflow

```text
START
  |
  v
Researcher
  |
  v
Reviewer
  |
  v
Review Router
  |-----------------------------|
  |                             |
  v                             v
Approved                     Rejected
  |                             |
  v                             v
Finalizer                  Researcher
  |
  v
END
```

## Current Nodes

### Researcher

Responsibilities:

- Receives the user query.
- Generates a draft answer.
- Updates `draft_answer`.
- Increments `retry_count`.

### Reviewer

Responsibilities:

- Receives the query and draft answer.
- Uses structured output.
- Produces:
  - accuracy
  - relevance
  - clarity
  - completeness
  - reason

### Review Router

Responsibilities:

- Compares review scores with configured thresholds.
- Returns:
  - `approved`
  - `rejected`
  - `max_retry`

### Finalizer

Responsibilities:

- Copies the accepted draft into `final_answer`.
- Produces the user-facing response.

## State

The current `AgentState` is working state for a single graph run.

It is not yet persistent memory.

```text
Graph State
    |
    +-- query
    +-- draft_answer
    +-- review scores
    +-- review_reason
    +-- final_answer
    +-- retry_count
```

## Configuration Flow

```text
.env
  |
  v
config/settings.py
  |
  v
models/llm.py
  |
  v
Agents and Graph
```

## Observability

LangSmith tracing records:

- graph runs
- node runs
- model calls
- structured output parsing
- router decisions
- state inputs and outputs
- latency
- token usage
- metadata
- errors

LangSmith currently provides observability, not independent factual verification.

## Target Architecture

```text
START
  |
  v
Request Router
  |-------------------------------|
  |                               |
  v                               v
Research Agent                 Web Agent
  |                               |
  |                               v
  |                         Browser Tools
  |                               |
  |-------------------------------|
  |
  v
Reviewer
  |
  v
Decision Router
  |-----------------------------------------------|
  |                 |                 |           |
  v                 v                 v           v
Approved         Retry          Need Evidence   Human Review
  |
  v
Finalizer
  |
  v
END
```

## Planned Platform Capabilities

- Tool Calling
- Tool Registry
- RAG
- Web Agent Node
- Browser Tools
- Short-term Memory
- Long-term Memory
- Planning
- MCP
- Advanced Evaluation
- Benchmarks
- Multi-Agent Collaboration

## Design Principles

1. Each node has one clear responsibility.
2. Tools execute actions; agents make decisions.
3. Configuration is centralized.
4. Structured output validates shape, not truth.
5. Tracing shows what happened; engineering explains why.
6. New features must be observable and testable.
7. Every capability must include failure-mode analysis.
