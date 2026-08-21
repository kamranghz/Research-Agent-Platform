# Research Agent Platform — Failure Modes

## Failure Mode 1 — Self-Evaluation Bias

### Observation

The Researcher and Reviewer currently use the same underlying model:

```text
Researcher -> Qwen
Reviewer   -> Qwen
```

### Risk

If the model has incorrect knowledge, the Researcher may generate a wrong answer and the Reviewer may approve the same mistake.

```text
Wrong model belief
    |
    +--> Researcher produces wrong answer
    |
    +--> Reviewer believes answer is correct
    |
    v
Router approves
```

### Planned Mitigations

- Use an independent judge model.
- Use multiple judges.
- Add trusted external context.
- Add human feedback.
- Evaluate on a labeled dataset.

---

## Failure Mode 2 — No External Grounding

### Observation

The Reviewer currently receives the question and draft answer, but no trusted source.

### Risk

The evaluation relies only on the model's internal knowledge.

A plausible answer can be mistaken for a correct answer.

### Planned Mitigations

- Add RAG.
- Provide official documentation to both Researcher and Reviewer.
- Evaluate groundedness.
- Evaluate source quality.
- Add citation validation.

---

## Failure Mode 3 — Threshold-Only Routing

### Observation

The Router currently uses:

- accuracy
- relevance
- clarity
- completeness
- retry count

### Risk

If the Reviewer scores are unreliable, the Router can apply its logic correctly but still make the wrong system-level decision.

### Planned Mitigations

Add signals such as:

- groundedness
- confidence
- evidence sufficiency
- citation validity
- tool success
- retrieval quality
- retry history
- human-review requirement

Add routes such as:

- `need_retrieval`
- `need_tool`
- `need_more_evidence`
- `need_human_review`

---

## Failure Mode 4 — No Uncertainty Estimation

### Observation

The Reviewer produces quality scores but does not explicitly report uncertainty.

### Risk

The system may produce a confident decision when evidence is weak.

### Planned Mitigations

Add fields such as:

```python
confidence: float
evidence_sufficient: bool
needs_retrieval: bool
needs_human_review: bool
```

---

## Failure Mode 5 — Shared Prompt or Model Weakness

### Observation

Different logical agents can still share the same model limitations.

### Risk

Node separation may look like agent diversity without providing actual epistemic diversity.

### Planned Mitigations

- Compare different models.
- Compare different prompts.
- Add adversarial review.
- Use reference-based evaluation.

---

## Failure Mode 6 — Retry Without New Evidence

### Observation

A rejected answer can return to the Researcher, but the Researcher may receive no new information.

### Risk

The model may repeat or slightly rewrite the same weak answer.

### Planned Mitigations

On retry, provide:

- reviewer reason
- failed criteria
- retrieved evidence
- tool results
- explicit revision instructions

---

## Future Web-Agent Failure Modes

These will be investigated after the Web Agent MVP exists.

Potential categories:

- page load timeout
- selector failure
- wrong element selected
- stale DOM
- dynamic content not loaded
- hidden or disabled controls
- login wall
- CAPTCHA
- infinite scrolling
- unexpected navigation
- inaccessible form labels
- screenshot misunderstanding
- accessibility-tree mismatch
- browser state loss
- repeated action loop
- failure to recover
- excessive step count
- token-budget exhaustion

## Failure Analysis Template

For every failure, record:

```text
Task:
Expected behavior:
Observed behavior:
Trace:
Failure category:
Likely root cause:
Evidence:
Recovery attempt:
Outcome:
Proposed improvement:
```
