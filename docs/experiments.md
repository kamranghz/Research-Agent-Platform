# Research Agent Platform — Experiments

## Purpose

Experiments turn architectural ideas into testable claims.

Every experiment should include:

- a question
- a hypothesis
- a controlled change
- a dataset or task set
- metrics
- repeated runs
- trace inspection
- conclusions
- limitations

## Experiment Template

```text
Experiment ID:
Title:
Research question:
Hypothesis:
Independent variable:
Dependent variables:
Controlled variables:
Dataset or task set:
Models:
Prompts:
Tools:
Number of repetitions:
Metrics:
Failure categories:
Results:
Interpretation:
Limitations:
Next experiment:
```

## Planned Experiment 1 — Same-Model vs Independent Judge

### Research Question

Does using the same model for generation and review increase the risk of correlated evaluation errors?

### Variants

```text
Variant A:
Researcher = Qwen
Reviewer   = Qwen

Variant B:
Researcher = Qwen
Reviewer   = different model
```

### Metrics

- answer correctness
- judge agreement
- false approval rate
- false rejection rate
- confidence
- latency
- token usage

---

## Planned Experiment 2 — Ungrounded vs RAG-Grounded Review

### Research Question

Does trusted retrieved context improve factual evaluation?

### Variants

```text
Variant A:
Reviewer sees question + draft

Variant B:
Reviewer sees question + draft + retrieved context
```

### Metrics

- factual correctness
- groundedness
- false approval rate
- citation correctness
- context relevance

---

## Planned Experiment 3 — Threshold Policy Comparison

### Research Question

Is threshold-only routing sufficient?

### Variants

```text
Policy A:
quality scores only

Policy B:
quality scores + confidence

Policy C:
quality scores + confidence + groundedness + evidence sufficiency
```

### Metrics

- approval accuracy
- unnecessary retries
- missed failures
- escalation rate
- final task success

---

## Planned Experiment 4 — Web Representation Comparison

### Research Question

Which web representation is most effective for different tasks?

### Representations

- raw HTML
- rendered DOM
- accessibility tree
- screenshot

### Task Categories

- information extraction
- button identification
- form filling
- navigation
- dynamic-page interaction

### Metrics

- task success
- average steps
- latency
- token usage
- navigation errors
- recovery success
- failure category distribution

---

## Planned Experiment 5 — Web Stack Effects

### Research Question

How do different web application stacks affect agent performance?

### Candidate Site Types

- static HTML
- server-rendered application
- React single-page application
- heavily dynamic application

### Metrics

- task success
- time to first valid observation
- selector stability
- DOM-change failures
- step count
- recovery success

## LangSmith Use

LangSmith will be used to inspect:

- node trajectories
- model calls
- tool calls
- retrieval runs
- browser actions
- latency
- token usage
- errors
- routing decisions

LangSmith traces provide evidence. Experiment conclusions must still be produced through engineering analysis.
