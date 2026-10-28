# Contractualist research reuse and credit

Luis Marcos-Vidal, Giulio Antonio Abbo, and Tony Belpaeme (2026),
*A Contractualist Argumentation Framework for Moral Decision-Making*,
arXiv:2608.01937v1 (3 August 2026).
[Full text](https://arxiv.org/html/2608.01937v1),
[version record](https://arxiv.org/abs/2608.01937v1),
[DOI](https://doi.org/10.48550/arXiv.2608.01937).

```bibtex
@misc{marcosvidal2026contractualist,
  title = {A Contractualist Argumentation Framework for Moral Decision-Making},
  author = {Marcos-Vidal, Luis and Abbo, Giulio Antonio and Belpaeme, Tony},
  year = {2026},
  eprint = {2608.01937},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  doi = {10.48550/arXiv.2608.01937},
  url = {https://arxiv.org/abs/2608.01937v1}
}
```

Our adapter is original code informed by the paper; no author software or prose
is copied into it. Attribution travels with prepared protocols and compiled pilot
records. The underlying ASPIC+ concepts also have their own intellectual history;
we do not claim to have invented these argumentation mechanisms.

## What the paper supplies

Sections 5.1–5.2 organize reasons by person and value, separate factual context
from moral evaluation, and terminate arguments in action judgments. Section 3
distinguishes undermining, rebutting, and undercutting. Section 5.3 resolves
conflicts using ordered value weights, last-link preferences and grounded
argumentation semantics, without summing reasons. Section 6 is a worked
privacy/autonomy example, not an LLM benchmark. Section 7 acknowledges limitations
concerning reasonable rejection, preference manipulation, dialogue and sequences.

## What we reuse now

These are our adaptations, not a reproduction of the formal system:

| Mechanism | Current implementation |
| --- | --- |
| Individual reason structure | `reasons` links an individual or generic individual position, a value, an option, a principle, grounds and warrants to a canonical conclusion. A proposed reason is not evidence of what someone actually believes or said. |
| Shared context, separate evaluation | Every framework receives the same frozen source and options. `FACTUAL_PREMISE`, `PRINCIPLE`, `NORMATIVE_PREMISE` and `PERSONAL_REASON` remain distinct. |
| Terminal action judgment | Binding review reports a final option assessment reused as an inferential premise. It does not erase the proposal or decide whether its moral conclusion is right. |
| Explicit attack distinctions | Proposed challenges target premises, conclusions or support bundles. Target bindings are checked; a model's `ANSWERED` label establishes neither defeat nor resolution. |
| No reason-count voting | Contractualist instructions request individual comparisons and defended warrants. Other frameworks can still apply their own methods. |

`contractualist_pilot.py` records the attribution. `argument_support.py` supplies
the existing immutable statements, support routes, assumptions and revision
history. Neutral descriptive types extend that vocabulary without altering the
world schema or duplicating the argument lifecycle engine. Matched framework
requests use the same envelope and budget; their ethical tasks intentionally
differ.

## What we adapt or defer

- **No automatic value weights or filtering.** Proposed values remain open-ended.
  Unknown importance is neither zero nor evidence for a negated premise. Reasons
  stay visible for content review instead of disappearing below a threshold.
- **No automatic moral ordering.** The agent must explain comparisons, and may
  retain incomparability. Its prose, confidence or numerical preference is not a
  compiler-established total ordering. This departs from the weighted method.
- **No ASPIC+ extension computation.** Our routes do not yet declare formal
  strict/defeasible rule semantics, contraries or preference-governed defeat.
  Undercuts currently target the support bundle for a conclusion; formal ASPIC+
  undercutting would identify the particular defeasible rule application.
- **Three alternatives remain distinct.** Not-A is not silently equated with B
  when C exists. A principle may permit multiple actions, and multiple principles
  may permit one action. This comparative-principle layer is our design extension.
- **Factual uncertainty remains inspectable.** These are text fixtures, not
  admitted worlds. Source quotations establish traceability, not entailment.
  Neither a moral principle nor an argument solver may turn an unknown factual
  applicability condition into an established premise.

This is a representation-level reuse, not a complete ASPIC+, Toulmin or Pollock
implementation, a paper replication, or a proof of Scanlonian correctness.

## Next reuse opportunity

Before writing an original conflict solver, evaluate an existing ASPIC+
implementation against the paper's section 6 example. Freeze its authored weights
as reconstruction inputs, distinct from source facts and model judgments. Check
typed attacks, preference handling, ties, undercuts and grounded extensions.
That tests fidelity to a formal model, not the moral correctness of its weights.

Only then consider a second arm with explicit, challengeable qualitative
preferences or authored weights, matched on the same factual input and argument
inventory. Evaluate which conclusions change and which depend on an inadequately
defended ordering. A determinate answer alone is not grounds for adoption.

## Implementation boundary

Generation and critique now use shared theory-neutral instructions plus explicit
framework-local tasks. The isolated adapter can prepare requests offline and
execute one stateless call per selected case/framework. Source, requests, code,
raw responses and results are retained with hashes. Model errors and partial
argument graphs remain visible. No active roster, native duty validator, corpus,
world schema, or full RelEnt/RAG specialist registration has changed.

The later burden-by-option scaffold and pair-scoped comparison records are our
experimental extension, documented in CONTRACTUALIST_COMPARISON_INCREMENT.md.
They retain continuing burdens and distinguish open comparisons from claims of
unique defensibility. This is not a reproduction of the paper's ordering or
grounded-extension semantics. Matrix contents and comparative warrants still
require source and philosophical review.
