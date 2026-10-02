# Constrained draft compiler direction

Status: approved architecture. Parliament's world-state schema remains stable.

The compiler composes a small set of general constructions rather than defining
schemas for individual dilemmas:

- event and state identity;
- participants and semantic roles;
- modality, negation, and conditional scope;
- causal and temporal links;
- resource transfer and quantity;
- exclusivity and choice;
- benefits, harms, and other stipulated outcomes;
- unresolved alternatives.

Its pipeline is candidate atoms, compatible alternative drafts, structural
propagation, Parliament invariant checking, and either a valid draft or an
explicit unresolved construction problem. Compilation may normalize topology or
copy source-grounded structure. It must not invent unsupported events, outcomes,
participants, quantities, or moral conclusions.

A construction enters the compiler only when it is expressed in terms of typed
records and source evidence, survives changes of names and domain vocabulary, and
abstains when its required structure is missing or ambiguous. Scenario-specific
lexical patches are not compiler rules.

The first construction is exclusive allocation. It combines direct resource
transfer, distinct recipients, source-supported choice or exclusion, explicit
quantity, and stipulated outcomes. The same rule now accepts both a medicine
`one dose` case and an antivenom `one vial` case. It uses the existing
`ExclusiveAllocationEvidence` structure; `vial` is not added to a resource-type
list. Multiple plausible transfer parents still cause abstention.

Constructed nonreceipt atoms are now evidence-closed. When one resource and one
direct transfer parent are structurally unique, the compiler copies the resource's
singular quantity and combines the resource, choice/exclusion, and parent source
references. Validation independently recomputes the same typed support instead of
trusting the constructed record. Explicit downstream outcomes remain separate and
must survive compilation; the compiler does not derive survival, recovery, harm,
or any other welfare result from nonreceipt.

Clause segmentation protects common person-title abbreviations such as `Dr.` and
`Prof.` so exact evidence spans retain the title. This changes segmentation only;
it does not normalize or invent source wording.

## General construction engine

The Parliament patch now runs the existing deterministic transforms through one
ordered internal construction-rule catalog. The catalog covers:

- event and state identity;
- participants and semantic roles;
- modality, negation, and conditional scope;
- causal and temporal links;
- resource transfer and quantity;
- exclusivity and choice;
- benefits, harms, and stipulated outcomes; and
- unresolved alternatives and abstention.

These tags are diagnostic metadata and are not added to `ScenarioWorldModel`.
The external schema and serialized records are unchanged. A medicine allocation
therefore reaches the same rule program as an antivenom allocation. Conditional
harm and modal rescue records use the same phases without a trolley or rescue
branch in compiler control flow.

Rules run in five phases: actual topology, source binding, topology recheck,
counterfactual derivation, and final normalization. Tests require the catalog to
cover every construction family, forbid domain names in rule identifiers, preserve
conditional causal gates and temporal edges, preserve possible and unresolved
atoms, and avoid adding unsupported effects.

Current regression results: 68 Parliament construction/quantity/status tests,
66 frozen-parser and ellipsis tests, and 12 parser-to-Parliament bridge tests pass.
The broader Parliament module still has the same 11 environment-only import errors
for optional embedding packages; its other 503 tests pass.

Z10 remains evidence and candidate generation. It does not commit a world state.
Parliament remains the authority for admission and invariant validation.

## Z10 candidate-world adapter

[`z10_world_model_adapter.py`](../../z10_world_model_adapter.py) now crosses the
previously missing boundary. Given a Z10 package and Parliament action strings, it:

1. aligns each action with supported proposition alternatives;
2. enumerates distinct reconstructed role readings without combining exclusive
   candidates;
3. closes each selection over Z10 dependencies and validates it;
4. emits an existing-schema `1.3` Parliament world-model payload with exact clause
   evidence; and
5. records unsupported semantics as typed construction problems.

The output is deliberately `PROVISIONAL` and `admission_authorized` is false.
`CONDITIONAL_ON` creates a condition and a gated consequence, but no causal edge.
Direct intervention rows are neutral; recovery or survival remains `UNRESOLVED`
until another supported layer classifies welfare. Quantity stays on its original
mention until reference evidence identifies it with the transferred resource.

The current held-out antivenom scenario produces one concrete two-action candidate
with five parties, four effects, and two conditions. The stripping probe “Lila
gives Omar the medicine, but not Nora” produces three candidates, preserving Nora
as destination, object, or subject. The object and subject readings carry an
`action_role_alignment_unresolved` problem because the supplied action text calls
Nora a destination. All four payloads pass Parliament's real
`parse_world_model(..., require_completeness=False)` boundary. See
[`z10_candidate_worlds_parliament_validation.json`](../../diagnostics/z10_candidate_worlds_parliament_validation.json).
They are structurally valid candidate worlds, not complete or admitted worlds.

## Live admission probe

A fresh serum-allocation scenario was run through Z10 export, candidate-world
enumeration, Parliament completeness validation, and `_admit_action_source_rows`.
The mechanical pipeline returned `COMMITTED` with no admission errors. The full
trace is in
[`z10_candidate_pipeline_live_serum.json`](../../diagnostics/z10_candidate_pipeline_live_serum.json).

This run also exposes the next required boundary fix. The adapter correctly marked
the draft `admission_authorized: false` and retained unresolved causation, welfare,
exclusivity, resource identity, party kind, and Z10 questions. The raw Parliament
admission call does not inspect that metadata and committed the extracted
`world_model` anyway. Therefore this is evidence that candidate models now flow
through the complete mechanical pipeline, but it is not evidence of a semantically
safe commit. A guarded handoff must reject or quarantine drafts with blocking
construction problems before calling Parliament admission.
