# General argument construction and support tracking

Proposal, October 23, 2026. Informed by the [matched live test and adversarial audit](diagnostics/inference_boundary_matched_live_20261023/RESULT.md). This is a proposed architecture and migration sequence, not an implemented replacement or a claim that formal fields solve ethical reasoning.

Implementation update: [the bounded prototype](ARGUMENT_SUPPORT_IMPLEMENTATION.md) now includes canonical revisions/citations, polarity/scope projection, proposal/compiler/critique and a scoped native patch. Two new text cases were exercised live and existing native cases were replayed offline. The optional argument path is nonvoting; a full operative migration and controlled end-to-end quality comparison are not established by these tests.

## What the tests changed

The equal-information pilot did not establish a consistent reasoning advantage from requiring a typed bridge revision. Both formats handled the short contrasts reasonably well and resisted mistaken criticism. Typed output cost more tokens and sometimes diverged from its own prose. Offline adversarial diagnostics also showed that labels can be filled without establishing the claimed relation. More labels, reminders or follow-up rounds are therefore insufficient as the general solution.

The narrow repair cards still help locate and preserve corrections. Their role should become a view over explicit support failures, not an expanding catalog of scenario-specific philosophical exceptions. The system should help construct defensible candidate arguments from the beginning, with bookkeeping performed by code.

## Four records in the argument layer

Keep Parliament's admitted world schema and frozen source stable. Add or evolve only the versioned argument layer:

1. **Statement:** an immutable proposition with text, relevant subject/predicate/object anchors, polarity, modality, hypothesis scope and semantic domain. Physical events, attributed beliefs, conditional maxims and normative judgments remain distinct. Reuse Z10/source-construction/proposition-ledger anchors where available; do not build another parser or scenario ontology.
2. **Support link:** input statement IDs, output statement ID, inference operation, explicit assumptions and supporting source locations. A separate assessment records whether the proposed link is supported, conditional, challenged or unassessed. Having a well-typed link does not establish its validity.
3. **Assessment:** the actor or reviewer, the precise target, support/defect explanation, evidence locations and uncertainty. Source citation validity, semantic support, native admission and normative dispute are separate dimensions. Reviewer attribution is not reviewer infallibility.
4. **Revision event:** explicit assert, qualify, withdraw, supersede or defend operations. Events retain the old record and link replacements. The stable issue ID survives; a semantically different proposition gets a new statement ID. Do not quietly change an INTENDED_USE proposition into BELIEF under the same identity.

Origin, semantic domain, confidence/support, hypothesis scope and lifecycle must be separate fields. CONDITIONAL must not double as UNKNOWN. “Withdrawn” must not be expressed only in a bridge explanation while its conclusion remains active. Citation fields should consistently use `{evidence_id, value_path, quote}` rather than mixing source IDs with informal path suffixes.

Code should allocate IDs, resolve citations, apply revision events, preserve lineage and compute dependency effects. Models should propose substantive propositions, assumptions and inferences. Requiring a model to reproduce every old annotation and manually manage those invariants is an avoidable source of corruption and cost.

## A small set of inference operations

These are argument operations, not dilemma templates or a complete ethical ontology:

| Operation | Support obligation |
| --- | --- |
| Source extraction/paraphrase | Match the particular proposition, polarity and qualifiers to exact text; distinguish excerpt content from effect predicate |
| Calculation | Name inputs, units and assumptions; compute deterministically when possible |
| Physical causal inference | Cite admitted direction/modality or clearly scope a hypothetical mechanism; an absent path in a partial graph is not a negative fact |
| Practical-reasoning interpretation | Identify end, instrumental belief and purposive employment; separate source-stated reasoning from a candidate maxim and actual causal efficacy |
| Norm application | State the governing norm, applicability conditions, relevant exception and case-specific bridge |
| Norm priority/comparison | Identify competing claims and the explicit priority argument; a duty name, rule authority or outcome count does not silently settle it |

Deterministic checks cover source identity, exact spans, arithmetic, compatible units, scope references, lifecycle, graph direction and structural contradictions. Semantic entailment and normative justification remain separate assessments. An independent critic should inspect the exact claim/source pair or the exact proposed bridge, not merely approve that the agent filled all slots.

An assumption is not automatically an error. A conditional argument can be useful and even logically supported while its antecedent remains unknown. The representation must preserve that distinction. Normative premises should be attributed to the framework, not promoted to world facts. Factual support cannot be manufactured by a framework commitment.

## Construct arguments before repairing them

Starting from the admitted world and framework commitment, ask for a small argument with:

- The strongest supported conclusion under a stated candidate maxim or normative application.
- The decisive premises and their exact support, including any interpretive or normative assumptions.
- The strongest competing obligation or objection and the bridge needed to answer it.
- A concrete reversal condition expressed as a hypothetical boundary, not an invented scenario fact.

Where the source underdetermines a material policy reading, permit at most two competing interpretations. They are conditional argument branches sharing the same world, not fabricated alternative factual worlds. If the source explicitly states relevant reasoning, preserve it; do not manufacture ambiguity or universal abstention.

Prefer a useful bounded argument such as “under rescue maxim M and norm N, conclusion C follows subject to unresolved condition Q” over either categorical certainty or a collection of disclaimers. The model should actually apply the framework and explain the priority, rather than merely name the framework. A conditional preference may be reported as such; it must not silently become an unconditional operative vote.

## Repair cards become generated questions

A card can be generated from the support graph's unmet obligation: missing belief support, unsupported exclusivity, an ungrounded normative priority or a contradiction between an active conclusion and a withdrawal. The same renderer can present the exact source, bridge, assumptions, affected conclusions and available revision operations. It need not contain scenario-specific control flow.

Do not depend solely on model-assigned claim kinds to decide whether an obligation is met. Inspect content and source alignment. For example, a BELIEF label attached to causal prose does not provide a belief premise, and a denied belief cannot satisfy an affirmative practical-reasoning requirement.

Keep the existing two-round limit. Target the smallest decisive disputed support link. A follow-up should add evidence, defend an explicit assumption, construct a conditional argument or withdraw the claim. Repeating the preferred label with stronger prose is not progress. A supported defense against a mistaken reviewer is progress.

## Propagation without unnecessary paralysis

Use support sets, not a single indiscriminate dependency chain. If a statement has two independent adequate derivations, withdrawing one should challenge only that support route. Within a required-premise set, uncertainty propagates transitively. Across alternative support sets, surviving support can remain available. Assumptions remain attached to the routes that use them.

Maintain lifecycle independently: an unsupported old conclusion can remain in history without remaining active. A withdrawn premise reopens dependent conclusions unless another adequate support route survives. Independent stipulated facts, other actions and other frameworks are not invalidated. A native validation event cannot revive a withdrawn argument.

## Native checker boundary

An assertion about actual physical success remains subject to the admitted physical graph. A source-stated mistaken belief or conditional candidate maxim is assessed for its stated belief/purpose and internal coherence; it must not invent the successful mechanism as a world effect.

The present native path requirement treats every intended-means classification as demanding an actual in-action path. The characterization test shows the current restriction, not that it should simply be deleted. A later native patch should use the claim's domain and scope to apply the appropriate obligation, preserve physical checks, and keep old saved records readable. Scope the check; do not broadly bypass admission or grant moral authority to inferred intentions.

## Bounded migration sequence

**First: canonical revisions and citations.** Make code manage claim lifecycle and exact source locations. Accept supported alias forms through auditable normalization, preserve their original paths, and compile explicit withdrawal/supersession into active argument views. Test the live stale-conclusion examples and the citation failures. Do not add another live prompt round as a workaround.

**Second: polarity and transitive scope.** Preserve affirmation/denial, modality and hypothesis ancestry independently of claim kind. Reuse existing scoped parser/proposition records. Close the negated-premise and indirect-condition gaps; distinguish UNKNOWN from an asserted scope promotion. Handle malformed input with controlled contract errors. These improvements expose errors; they do not establish semantic entailment.

**Third: operation-specific construction and critique.** Generate at most two small, useful candidate arguments, not a fully regenerated ledger. Assess exact support links with counterexamples and source alignment. Include norm application and competing-duty priority so the change advances ethical reasoning rather than only record hygiene. Use the same compiler and patch mechanism for all frameworks.

**Fourth: scope native checks and evaluate end to end.** Separate physical support from candidate practical reasoning, preserve genuine path checks, and rerun frozen native cases plus held-out matched live cases. Measure admission and argument quality separately. Preserve previous working blueprints and admitted world generation.

Each stage should be independently reviewable. Do not implement this entire proposal as one large schema migration.

## Acceptance and evaluation

Freeze the known failures as red-to-green behavioral cases, retaining the original raw provider records. Required controls include wrong reviewer, explicit source-stated purpose, ineffective intended means, incidental causal contribution, consent without automatic permission/prohibition, conditional maxim analysis and two independent support routes.

Measure separately: source/proposition fidelity; unsupported active claims; valid inference or defended assumptions; quality of norm application and priority argument; preservation of justified original claims; usefulness of conditional conclusions; prose/graph agreement; native admission; token/latency cost. Do not sum these into a single RESOLVED score.

The next live evaluation should use held-out scenarios, more than one sample where feasible, and blinded or independent content review. The current pilot's annotations made its tasks comparatively easy. Normative disagreements cannot be reduced to a universal answer key: evaluate whether the argument defends its premises, respects its framework and answers the objection, while recording contested substantive judgments.

No claim is made that a support graph can guarantee correct ethical conclusions. The intended gain is a system that constructs more useful arguments, localizes uncertainty and contradictions, and applies the right checks to the right claims without contaminating the shared world.
