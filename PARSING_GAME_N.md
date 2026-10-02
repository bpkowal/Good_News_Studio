# Parsing game N: ATTEMPT and complement entailment

N is a standalone M→N experiment. It leaves M unchanged and uses N-prefixed
diagnostics and `diagnostics/parsing_game_N_user_probes.json` (latest five inputs).

```sh
python parsing_game_N.py
python -m unittest test_parsing_game_N -v
python parsing_game_N.py --sentence "The worker tried to leave."
```

## What N adds

`parse_world_state(text, policy)` uses output schema 3 and returns
`complement_relations` in addition to M's propositions, events, and causal claims.
For the example above:

```text
p0: try(worker, EVENT p1)
    occurrence_status = asserted_in_text
p1: leave(worker)
    occurrence_status = unknown

parliament:proposition:ATTEMPT(p0, p1)
    child_entailment = not_entailed
```

`not_entailed` means the parent does not establish the child's occurrence. It does
**not** mean the child failed or did not occur. A separate sentence asserting that
the worker left receives its own `asserted_in_text` status. A denied attempt gives
the parent `denied_in_text`; possible, conditional, and attributed attempts leave
occurrence unknown. None of these statuses is external truth verification.

Each complement relation records parent/child IDs, namespace, relation type,
entailment, parent assertion status, and source/matching provenance. Unsupported
relations are `UNRESOLVED` with entailment `unknown`; N does not infer that caused,
permitted, or forced events happened. All enriched proposition/event records stay
ineligible for direct world-state commitment.

Local `assertion` annotations preserve linguistic cues. Use `occurrence_status`
for the conservative textual commitment to the whole proposition in context.
An embedded predicate can have affirmative local wording without its occurrence
being asserted. Causal records for embedded predicates are separately blocked by
`embedded_proposition_requires_scope_validation`, including the direct
`parse_claims()` API. Thus trying to cause damage never commits an actual
worker-causes-damage edge.

## Small local VerbNet adapter

[VerbNet 3.3, try-61.1](https://verbs.colorado.edu/verb-index/vn3.3/vn/try-61.1.php)
documents the members try, attempt, and intend, Agent/Theme roles, and an
infinitival complement frame associated with attempt semantics. The checked-in
[adapter](resources/verbnet_attempt_adapter.json) records that evidence and its
version/source. It is manually curated metadata, not a full VerbNet download.

N's application policy is deliberately narrower:

- Activate only `try` and `attempt` with a direct infinitival `xcomp`, a `to`
  marker, no explicit child subject, and a parent subject with no competing
  direct object. Share that subject with the child and record the control source.
- Keep `intend` as documented membership but abstain from ATTEMPT classification.
  Intention is not automatically an attempted action in Parliament's vocabulary.
- Abstain for nominal objects, gerunds, unsupported senses, and unmatched frames.
- Treat the non-entailment rule and subject-control transfer as explicit adapter
  policies, not as a direct transcription of all VerbNet semantics.

This is lexical evidence plus a bounded construction match, not full word-sense
disambiguation, entity-type validation, or a general controller resolver. No
sentence-specific CEM features or holdout-driven weight updates were added.
The causal policy schema remains 2. No runtime network request is needed.

### Complement adapter interface

`ComplementAdapter` is the generic wrapper for namespace, relation type,
entailment, active/documented lexical members, structural matching, controller
recovery, recovered-frame validation, and provenance. Both argument extraction
and complement interpretation use `get_complement_adapters()`. The registry
currently contains only ATTEMPT; its narrow construction policy is unchanged.
`load_attempt_adapter()` and `matches_attempt_frame()` remain available for
compatibility. No permission, prevention, or intention semantics were added.

Interpretation requires exactly one matching, validated adapter. No matches or
multiple matches produce `UNRESOLVED`/`unknown`; registration order cannot choose
a semantic interpretation. Provenance includes per-adapter evaluations alongside
N's existing single-resource fields. Controller transfer likewise requires a
unique structural match. Unsupported constructions may still have draft arguments
from the existing structural extractor; those are not adapter-validated control.

The contrastive suite includes `tried`, `attempted`, denied/modal attempts,
`tried and left`, and `tried to persuade Alice to leave`, with a lexical variant.
For the nested example, ATTEMPT links only `try → persuade`. The lower
`persuade → leave` relation remains unresolved, both embedded occurrences remain
unknown, and the worker is not propagated into the lower event. The current
extractor proposes Alice there; this does not introduce a general object-control
policy for persuasion.

`The worker tried and left.` now recovers the shared subject even though `try`
has no binary argument pair. Its finite coordinated `left` is independently
asserted in text. Promotion is restricted to a direct root coordination with
`and`/`but`/`yet`, an unqualified asserted head, and an asserted local predicate.
Negative, modal, conditional, and disjunctive contrast cases keep leaving unknown.
These draft propositions remain ineligible for world-state commitment.

The repeated spaCy parsing is intentionally unchanged in this increment.

## Freebase's separate role

The [schema sample](resources/freebase_schema_sample.json) contains three property
identifiers from Google's [Basic Concepts](https://developers.google.com/freebase/guide/basic_concepts)
documentation. It is explicitly a documentation-derived sample, **not** sampled
triples or a frequency ranking. Its `freebase:world_entity` namespace is separate
from `parliament:proposition`; it never classifies ATTEMPT.

Freebase uses topics with MIDs, type/property identifiers organized under domains,
and CVTs for multi-field relationships. A dated population measurement is the
documentation's example of why a relationship may need an intermediate node.
These are useful ontology concepts, not evidence about linguistic entailment.

Google's [dump page](https://developers.google.com/freebase) describes the retained,
unmaintained dataset and its licensing. N downloads no dump and calls no Freebase
API. A future bounded importer can inspect local triples separately from semantic
parsing. Counting VerbNet complement predicates for relation discovery is likewise
a future corpus experiment: this tiny adapter cannot support coverage/frequency
claims, and no such counts are fabricated here.

## Regression status

The N suite covers subject-control transfer, resource provenance, negation,
modality, conditionals, attribution, independent successful-event assertions,
abstention, nested causal commitment, serialized probe output, and namespace
separation. It also runs M's unchanged adversarial fixtures against N.
The expanded suite has 29 tests: 25 pass and the four known structural gaps below
remain expected failures. Registry tests also cover replacement policy dispatch,
recovered-frame rejection, empty registration, and ambiguous matches.

Four desired structural behaviors remain explicit expected failures:

- passive controller recovery;
- coordinated event-complement attachment;
- the conditional clause boundary that leaks `refunds` into `close`;
- treating `by the blind man` as a path/proximity adjunct rather than the object
  of `flew`.

Independent ordinary tests require these cases to remain blocked from causal
commitment. The prepositional-role bug is recorded but intentionally not repaired
in this increment.
