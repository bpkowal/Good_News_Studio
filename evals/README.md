# Matched-budget ethics comparison

Validate the ten cases, schemas, and command configuration without spending API credits:

```bash
python comparative_ethics_eval.py
```

Run a one-case pilot first:

```bash
python comparative_ethics_eval.py \
  --execute \
  --run-id pilot-01 \
  --case trolley_lever
```

After inspecting `eval_outputs/pilot-01`, run all ten cases:

```bash
python comparative_ethics_eval.py --execute --run-id parliament-vs-o3-01
```

The same command resumes a partially completed run. Add `--force` only when you intend
to replace completed stages.

After changing only the final-answer renderer or judging rubric, reuse the paid contestant
runs and repeat only the judge stage:

```bash
python comparative_ethics_eval.py \
  --execute \
  --run-id parliament-vs-o3-01 \
  --rejudge
```

## Fairness protocol

- Both contestants use the model named by `--model` (default `o3`).
- Every Parliament API response is counted, including retries and hidden reasoning tokens.
- Solo o3 receives one ordinary plain-text call with no Parliament schema, graph,
  action-label middleware, or semantic validators. Its per-case completion-token ceiling
  equals the Parliament's measured completion-token use, up to the model's output limit.
  It uses high reasoning effort by default so the single call can make meaningful use of
  that aggregate budget. The report shows if a cap occurred.
- The Parliament receives the two fixed actions listed in the case fixture. This tests
  deliberation rather than action-planner reliability.
- Each case uses an isolated workspace and episodic-memory file. Cases cannot train later
  cases through Parliament memory.
- A separately accounted judge sees only anonymous, deterministically randomized user-facing
  final answers. It does not see which architecture produced either answer.
- Parliament's internal trace receives a separate process audit for delegate validity,
  landscape semantics, plurality stability, dissent preservation, and synthesis admission.
- Input tokens, actual output tokens, reasoning tokens, cached tokens, estimated costs, and
  judge tokens remain separate in the raw artifacts.

The final-answer rubric measures clarity, scenario fidelity, action/consequence mapping,
ethical coverage, synthesis of deliberation, preservation of objections, uncertainty,
coherence, and reversal conditions. It treats moral disagreement as legitimate and targets
action/consequence inversion,
fabricated decisive facts, omitted explicit catastrophic risk, scope insensitivity,
unjustified certainty, internal contradiction, hypothetical leakage, procedural tie-breaking,
malformed answers, and suppression of grounded dissent.

This is a diagnostic experiment, not a statistically definitive ranking. Review the raw
case artifacts alongside the aggregate report, especially whenever the LLM judge flags a
catastrophic failure.

## Action-label permutation checks

Compare a normal trace with a trace containing the same physical actions under swapped
presentation labels:

```bash
python -m global_workspace.invariance path/to/normal.json path/to/reversed.json
```

The report separates ordinal invariance (selected action and specialist directions),
cardinal invariance (scores, preference strength, epistemic confidence, and aggregate
support), and dynamic invariance (workspace broadcast path). Malformed reversal-review
strings are counted as excluded audit evidence rather than treated as moral responses.
