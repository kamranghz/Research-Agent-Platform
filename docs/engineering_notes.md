# Research Agent Platform — Engineering Notes

## Core Principle

> Tracing shows what happened. Engineering explains why it happened.

LangSmith provides execution evidence. It does not automatically explain architectural weaknesses or guarantee factual correctness.

## Observability, Validation, and Evaluation

### Observability

Observability answers:

- Which node ran?
- What input did it receive?
- What output did it return?
- Which model was called?
- How long did it take?
- How many tokens were used?
- Which route was selected?
- Where did an error occur?

Current tool: LangSmith tracing.

### Validation

Validation checks the structure and type of data.

Examples:

- Is `accuracy` a float?
- Is it between 0 and 1?
- Is `reason` a string?
- Are all required fields present?

Current tool: Pydantic.

Validation does not prove factual correctness.

### Evaluation

Evaluation checks output quality.

Examples:

- Is the answer factually correct?
- Is it relevant?
- Is it grounded in evidence?
- Does it contain hallucinations?
- Are citations valid?
- Was the tool trajectory appropriate?

Current status: The project has an internal LLM Reviewer, but not yet an independent grounded evaluation system.

## Current Trace Interpretation

A current run contains:

```text
LangGraph
├── researcher
│   └── ChatOllama
├── reviewer
│   ├── ChatOllama
│   └── PydanticOutputParser
├── review_router
└── finalizer
```

Interpretation:

1. The Researcher generates a draft.
2. The Reviewer evaluates the draft.
3. Pydantic validates the output structure.
4. The Router applies configured thresholds.
5. The Finalizer returns the approved draft.

Important:

- LangSmith did not create the review scores.
- Qwen produced the review scores.
- LangSmith did not approve the response.
- The project Router returned `approved`.
- LangSmith recorded these events.

## State Is Not Persistent Memory

Current `AgentState` transfers information between nodes during one run.

It does not remember information across separate executions.

```text
Current Graph State = working state for one run
Short-Term Memory    = state across turns in one thread
Long-Term Memory     = persistent information across threads
```

## Configuration Principle

Configuration must load before resources are initialized.

```text
Configuration
    |
    v
Resources
    |
    v
Application
```

In this project:

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
Agent Nodes
```

## Agent and Tool Separation

An agent decides.

A tool executes.

Example:

```text
Web Agent
    |
    v
decides to open a page
    |
    v
Browser Tool
    |
    v
executes the browser action
```

## Research Workflow

Every new capability should follow:

```text
Question
   |
   v
Hypothesis
   |
   v
Implementation
   |
   v
Trace
   |
   v
Experiment
   |
   v
Failure Analysis
   |
   v
Improvement
```

## Questions to Ask for Every Feature

1. What problem does this feature solve?
2. Which layer owns it?
3. What state does it read and update?
4. How will it appear in a trace?
5. What new failure modes can it create?
6. How will we test it?
7. How will we evaluate its quality?
