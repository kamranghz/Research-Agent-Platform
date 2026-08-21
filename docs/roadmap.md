# Research Agent Platform — Roadmap

## Roadmap Status

This roadmap is the current development plan and should remain stable unless a clear technical reason requires a change.

## Phase 1 — Foundation

Status: Completed

- [x] Project structure
- [x] LangGraph workflow
- [x] Agent state
- [x] Researcher node
- [x] Reviewer node
- [x] Conditional router
- [x] Finalizer node
- [x] Retry policy
- [x] Pydantic structured output
- [x] Central configuration
- [x] `.env` loading
- [x] LangSmith tracing
- [x] Basic structured-output test

## Phase 2 — Tool Calling

Goal: Allow an agent to select and execute external capabilities.

- [ ] Explain tool-calling concepts
- [ ] Implement calculator tool
- [ ] Implement tool registry
- [ ] Bind tools to the model
- [ ] Add a ToolNode or tool-executor node
- [ ] Route between agent and tools
- [ ] Store tool results in state
- [ ] Trace tool calls in LangSmith
- [ ] Handle tool errors
- [ ] Add tests

Target workflow:

```text
Researcher
   |
   +--> Tool Executor
   |        |
   |        v
   +<-- Tool Result
   |
   v
Reviewer
```

## Phase 3 — Web Agent MVP

Goal: Build a Web Agent as a real LangGraph node and use it as a research subject.

- [ ] Add web-specific state fields
- [ ] Implement Web Agent node
- [ ] Implement browser tools
- [ ] Open a page
- [ ] Extract page title and content
- [ ] Collect observations
- [ ] Detect basic failures
- [ ] Record step count and latency
- [ ] Trace browser actions in LangSmith
- [ ] Add recovery behavior
- [ ] Add reproducible web tasks

Initial workflow:

```text
Request Router
   |
   +--> Researcher
   |
   +--> Web Agent <--> Browser Tools
                         |
                         v
                      Observation
```

## Phase 4 — RAG

Goal: Ground answers in trusted documents.

- [ ] Document loader
- [ ] Text splitter
- [ ] Embeddings
- [ ] Vector store
- [ ] Retriever
- [ ] Retriever tool
- [ ] Source metadata
- [ ] Retrieved-context state
- [ ] Grounded Reviewer prompt
- [ ] Context relevance
- [ ] Groundedness
- [ ] Citation checks

## Phase 5 — Memory

Goal: Preserve useful context across turns and sessions.

### Short-Term Memory

- [ ] Add LangGraph checkpointer
- [ ] Add `thread_id`
- [ ] Preserve conversation state
- [ ] Test multiple turns
- [ ] Define state-trimming policy

### Long-Term Memory

- [ ] Define memory schema
- [ ] Extract useful memories
- [ ] Persist memories
- [ ] Retrieve relevant memories
- [ ] Add memory-quality evaluation
- [ ] Prevent stale or incorrect memories

## Phase 6 — Planning

Goal: Allow agents to select multi-step strategies.

- [ ] Planner schema
- [ ] Planner node
- [ ] Step execution
- [ ] Replanning
- [ ] Completion checks
- [ ] Plan-quality evaluation
- [ ] Loop and budget limits

## Phase 7 — MCP

Goal: Expose and consume tools through a standardized protocol.

- [ ] Study MCP architecture
- [ ] Implement a basic MCP client
- [ ] Implement a small MCP server
- [ ] Expose calculator or document search
- [ ] Connect MCP tools to the graph
- [ ] Trace MCP calls
- [ ] Handle discovery and execution failures

## Phase 8 — Advanced Evaluation

Goal: Move from runtime self-review to reproducible evaluation.

- [ ] LangSmith datasets
- [ ] Offline experiments
- [ ] Prompt comparison
- [ ] Model comparison
- [ ] Independent judge model
- [ ] Human feedback
- [ ] Groundedness evaluation
- [ ] Tool-success evaluation
- [ ] Trajectory evaluation
- [ ] Failure-rate reporting
- [ ] Confidence calibration

## Phase 9 — Multi-Agent Collaboration

Goal: Support specialized collaborating agents.

Potential agents:

- Request Router
- Planner
- Research Agent
- Web Agent
- Retrieval Agent
- Tool Executor
- Reviewer
- Failure Analyzer
- Finalizer

## Priority Order

```text
Tool Calling
    |
    v
Web Agent MVP
    |
    v
RAG
    |
    v
Memory
    |
    v
Planning
    |
    v
MCP
    |
    v
Advanced Evaluation
    |
    v
Multi-Agent Collaboration
```
