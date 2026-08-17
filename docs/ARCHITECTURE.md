# Parliament architecture

This document describes the recurrent global-workspace path. It is a map of
responsibilities and invariants, not a complete algorithm specification.

## Entry points

- `parliament.py` is the interactive front door and selects legacy or workspace mode.
- `global_workspace_pipeline.py` assembles dependencies, runs a scenario, saves the
  trace, and renders terminal output.
- `ethics_synthesis_agent.py` is the original, non-workspace pipeline.
- `comparative_ethics_eval.py` compares Parliament with a single-model answer.

## Runtime lifecycle

1. Parse the scenario and establish a small, materially distinct action set.
2. Compile graph-backed action identities and assign presentation-independent IDs.
3. Consult the original RAG-backed ethical agents once to obtain frozen testimony.
4. Run the non-voting visibility audit for structurally produced missing evidence.
5. Let lightweight framework delegates evaluate the same actions recurrently.
6. Validate and symmetrically damp unsupported claims before aggregation.
7. Apply deterministic workspace-access, policy, and halting rules in Python.
8. Test useful counterfactual branches without allowing them to mutate base state.
9. Preserve dissent, reversal conditions, and moral residue in the result.
10. Render a public judgment and record specialist contribution history.

## Responsibility map

| Module | Responsibility |
| --- | --- |
| `engine.py` | Deliberation lifecycle, aggregation, access gates, and halting |
| `models.py` | Domain records exchanged between stages |
| `local_specialists.py` | Delegate, action-planning, synthesis, and reformulation adapters |
| `structured_io.py` | Backend-neutral structured model calls and JSON parsing |
| `framework_retrieval.py` | Shared typed corpus ingestion, evidence-strength classification, provenance, and compatibility loading |
| `evidence_calibration.py` | Non-voting calibration of unstated factual claims |
| `visibility.py` | Non-voting endogenous-observability audit |
| `landscape_validation.py` | Semantic fallback for ambiguous action–consequence mappings |
| `action_identity.py` | Typed action identity graphs and conservative lexical fallback |
| `scenario_semantics.py` | Canonical action ordering and deterministic scenario-fact compilation |
| `semantic_invariants.py` | Typed propositions and checks across transformations |
| `rawls_ledger.py` | Transactional action-to-group Rawlsian position assessments |
| `utilitarian_ledger.py` | Transactional action-to-consequence welfare accounting |
| `deontology_ledger.py` | Transactional duty, right, permission, and conflict assessments |
| `trace_health.py` | Deterministic post-run representation-integrity diagnostics |
| `autonomy_audit.py` | Non-voting coercion tags, evidentiary surcharge, and reversal probe |
| `middleware/` | Lightweight claim damping, reversal audit, and moral-residue tracking |
| `legacy_bridge.py` | Isolated calls into the original ethical agents |
| `openai_backend.py` | OpenAI structured-output adapter |
| `memory.py` | Episodic records and specialist contribution summaries |
| `presentation.py` | Public-facing final judgment |

`global_workspace_pipeline.py` is the composition root: it may wire these pieces
together, but new ethical logic should normally live in the relevant module above.

## Behavioral invariants

- Original testimony is a frozen baseline; recurrence may reconsider it but may not
  silently rewrite it.
- A framework's commitments are distinct from its provisional action. When Care's
  own relational dimensions support opposing actions, the baseline is recorded as
  `NORMATIVELY_CONTESTED`; its relational commitments persist without freezing the
  source model's one-shot recommendation.
- Care delegates compare every action through entrustment, dependency, trust,
  agent-created vulnerability, responsibility, attentiveness, and responsiveness.
  Numerical magnitude may be decisive, secondary, or irrelevant, but decisive
  counting must explain why the relational claims are otherwise comparable.
- Deontological delegates attach a duty verdict to every action and preserve
  unresolved conflicts among autonomy, rescue, universal law, and respect for
  persons. Numerical magnitude is not a substitute for a duty-priority rule.
- Rawlsian delegates record the least-advantaged group's comparative position
  under every action and the relevant liberty or primary good. A generic claim
  to "protect the least advantaged" cannot distinguish actions that leave that
  group in the same stated position.
- Rawlsian directional claims commit as typed action-assessment-group edges. An
  unsupported improvement/preservation/worsening claim is recorded as uncertain
  and damped once; malformed or misbound ledgers leave prior graph state untouched.
- Utilitarian delegates maintain a typed consequence table for every action:
  outcome, affected scope, direction, probability, magnitude, duration,
  reversibility, and evidentiary support. If an unknown comparison is decisive,
  the action remains provisional rather than becoming a fabricated expected value.
- Utilitarian consequence tables commit as action-owned graph nodes. A consequence
  direction contradicted by the action graph becomes unknown; merely unresolved
  scenario grounding stays explicitly unresolved rather than being treated as false.
- Deontological verdicts commit with a norm, relation, duty bearer, protected party,
  competing norm, and evidence basis. Verdict–relation contradictions become
  uncertain; the graph validates internal structure without deciding moral truth.
- Virtue delegates assess every action through the actor's role, virtues, vices,
  circumstances, and practical wisdom. Numerical stakes may inform phronesis but
  cannot replace the character judgment with aggregate maximization.
- These specialists share only the typed envelope (action-indexed commitments,
  numerical role, provisional action, and conflict status). Their substantive
  validators remain framework-specific to avoid identity convergence by schema.
- Public framework summaries consume only committed ledger projections. Rejected
  delegate updates remain inspectable in the trace but cannot enter the final answer.
- Presentation labels and option order never define physical action identity. Action
  identity comes from committed intervention, target, consequence, quantity, and
  modality structure; sparse parses fall back explicitly to lexical identity.
- If two alternatives collapse to one graph signature inside a decision, lexical
  discriminators preserve both branches rather than merging uncertain identities.
- The visibility auditor is epistemic middleware, not a sixth ethical voter.
- A grounded visibility finding receives workspace access as an explicit proposition;
  delegates must state whether it changes their harm estimate and may retain their
  recommendation with a reasoned revision.
- The autonomy auditor never votes or vetoes; coercion raises the evidentiary burden,
  while scenario-established imminent catastrophic third-party harm removes that
  surcharge without deciding the underlying moral question.
- Hypothetical branches do not update the base policy, specialist identity, or memory.
- Preference strength and epistemic confidence remain distinct values.
- Unsupported speculative claims are labeled and damped symmetrically across actions.
- Evidence penalties depend on grounding distance and do not compound with an audit
  that is measuring the same missing support.
- Reformulation arithmetic is restricted to matching measurement family, unit,
  population basis, and time basis. Cross-family tradeoffs remain explicit and are
  never collapsed into an implicit utility score.
- High-risk transformations preserve actor, action, relation, condition polarity,
  consequence, affected party, epistemic status, branch context, and provenance.
  Invalid compression is withheld while its source proposition remains traceable.
- Preference history is compared only across equivalent control contexts. Accepted
  audit propositions persist as explicit specialist commitments in later cycles.
- Substantive dissent survives formal convergence and produces reversal conditions.
- Insufficient valid candidates must not be mislabeled as ordinary non-convergence.
- A hard failure or exhausted budget must not fabricate a settled general rule.

## Extension guidance

- Add normative frameworks as delegates with the same candidate contract.
- Add non-voting audits beside `visibility.py`; expose explicit adjustments and
  evidence rather than hiding policy preferences in them.
- Add cross-cutting guardrails in `middleware/` when they are independently testable
  transformations over domain records.
- Keep backend-specific behavior behind a small adapter and reuse `structured_io.py`.

## Next structural targets

`local_specialists.py` remains the largest compatibility module. Low-risk future
extractions are `action_planning.py`, `synthesis.py`, and `reformulation.py`. The
large `tests/test_global_workspace.py` should then be split along those boundaries.
The preferred sequence is one extraction at a time, with behavior-preserving tests
before changing the underlying reasoning policy.
