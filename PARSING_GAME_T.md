# T2: candidate preservation, compositional scope and document references

`parsing_game_T.export_candidate_graph(text)` exports the draft 0.2 envelope in
[the contract](PARSING_GAME_T_CONTRACT.md). T is a separate candidate-export entry
point; S's selected `parse_world_state` output and training behavior are unchanged.

```sh
.venv/bin/python parsing_game_T.py --sentence "If Maria pulls the lever, the trolley will stop."
.venv/bin/python -m unittest test_scope_semantics test_candidate_validation test_parsing_references test_parsing_game_T test_parsing_game_S -v
```

T reuses S's tokenizer, attachment candidate generator, and controller evidence.
It never calls CEM training. S's small attachment classifier still fits lazily on
first use. T preserves structural attachment alternatives regardless of S's
preferred reading. A C4 decision therefore retains the verb predication, possible
complement/destination links, incompatibilities, and an attachment question.
The destination interpretation is incompatible with the alternative event anchor.
Degree-result semantics remain unresolved in the minimal vocabulary. Infinitival
`advcl` attachments additionally receive a candidate purpose reading.

Explicit modal tokens attach to their own proposition. `will` offers prediction;
`can`/`could` offer ability, permission, and possibility; ambiguous readings are
mutually exclusive candidates with an open question. `must` offers obligation and
an unresolved alternative: epistemic necessity is not mislabeled as possibility.
No rule transfers obligation from `decide` to `act`. Modal readings are candidate
interpretations, not calibrated probabilities. Negation qualifies proposition and
participant/link readings; modal operator polarity is separate. This is a bounded
model of negation versus modal scope.

A direct `if`-marked `advcl` receives a `CONDITIONAL_ON` link from consequence to
condition. The antecedent carries hypothetical context and the consequence carries
conditional context, including its modal candidates. Interrogative `if`, such as
“asked if Anna left,” remains a scope question. These rules depend on spaCy's
attachment and do not establish general discourse scope or conditional entailment.

Each package retains exact source text, document-global offsets, unique IDs, and
coverage questions. Controller candidates supported by S remain separate from
objects and require further validation. No candidate has an actual-occurrence or
world-state eligibility flag. No graph is selected or committed.

T also preserves an explicit child subject as a provisional controller when a
`VB` child has an infinitival `to` marker, including spaCy's `ccomp` attachment.
This recovers the library/workers cases without promoting finite-clause subjects
or filling a parent object's slot.

This increment is partial: no identity-resolving coreference system, complete
quantification/choice/attribution model, or semantic-support checker is implemented.
Every sentence receives a coverage question because source coverage does not imply
semantic completeness. `parse_world_state` compatibility is not supplied by T;
consumers must explicitly adopt the candidate-package interface.

Tests cover alternative survival and exclusivity, modal distinctions, condition
versus interrogative attachment, negation, document offsets, retained purpose and
whether-clause verbs, controller preservation, and dependency acyclicity. S's
attachment suite is run alongside these tests.

T0 validation: 18 tests passed (7 T tests and 11 S tests). A saved four-sentence
[trolley package](diagnostics/parsing_game_T_trolley_candidates.json) retains
`divert` and `act`, and links the conditional killing outcome to pulling the lever.

## T1 document reference candidates

The separate [reference module](parsing_references.py) adds `SAME_REFERENT`
candidates for third-person pronouns, definite descriptions sharing a head lemma,
and repeated proper names. Mention inventory now also includes nominal oblique
arguments. IDs and evidence offsets remain document-global, and earlier mentions
are never merged or erased.

The [versioned policy](resources/T_reference_policy.json) limits antecedents to
non-pronominal mentions within three preceding sentences (and earlier mentions in
the current sentence). Candidate filtering uses observed number, explicit numeral
compatibility, and a small human/nonhuman lexicon plus spaCy PERSON annotations.
Unknown animacy remains open; gender is not guessed from people's names. Singular
they remains possible. Coordinated groups, split antecedents, reflexives, cataphora,
and pronoun chains are not resolved by this increment. Numeral matching is literal,
so semantically equal alternative numeral spellings may be missed.

Ranking uses hand-set sentence recency, animacy agreement, grammatical role, and
description-match weights. Scores are `uncalibrated_score`, never probabilities.
A preferred candidate needs score >= 0.65 and a lead >= 0.25; even then the
reference question remains open for the consumer. No learned classifier or gold
fixture fitting was introduced. These thresholds are application policy, not
empirically calibrated confidence.

Reference questions list ranked candidates and identify affected participant
candidates as blockers. A missing antecedent produces an empty candidate list.
Reference choice sets use `any_subset`: two antecedent mentions may themselves
refer to the same entity, so treating their identity links as mutually exclusive
would incorrectly forbid a consistent identity chain. The selection validator
checks global compatibility with explicit package identity constraints; the
exporter alone makes no such guarantee.

T1 validation: 24 tests passed, comprising six reference tests and the existing
18 T/S tests. Coverage includes ambiguous people, missing antecedents, number and
cardinality contrasts, local binding abstention, distance limits, and preservation
of conditional/modal scope. This is regression evidence, not a measured estimate
of general coreference accuracy. The T0 sample above is retained as historical
output; [T1's sample](diagnostics/parsing_game_T1_trolley_candidates.json) contains
the new reference candidates.

## Selection validation and identity consistency

The pure standard-library [validator](candidate_validation.py) is also exported
from `parsing_game_T`:

```python
from parsing_game_T import export_candidate_graph, empty_selection, validate_candidate_selection

package = export_candidate_graph("Maria can leave.")
selection = empty_selection(package)  # Explicit abstention on every open question.
result = validate_candidate_selection(package, selection)
assert result["contract_valid"]
```

Selections use existing IDs and must include candidate dependencies, argument
nodes, and scope endpoints. Validation checks exact evidence spans, typed
endpoints, schema/ID integrity, acyclic dependencies, symmetric exclusivity,
choice limits, score metadata, scope references, and complete question bookkeeping.
An unresolved question is allowed; affected selections and their dependents remain
provisional. Resolving a question requires selecting one of its supported candidate
records. Extensions remain unverified and cannot satisfy graph dependencies.

Selected `SAME_REFERENT` links produce diagnostic identity components. The optional
`identity_constraints` amendment in the contract records explicit different-referent
constraints; direct and transitive violations fail validation. These constraints
are not inferred from names or generated automatically. Without a constraint,
structural consistency cannot rule out a mistaken identity chain. Quantity and
group-membership semantics are not inferred from identity.

The validator never mutates either input or commits world state. It trusts the
caller's original package: package IDs alone do not authenticate content. Semantic
support, missing interpretations, and external truth remain unassessed.

Validation: 37 tests passed (13 validator tests plus the 24 T/reference/S tests).
Both saved trolley packages also pass with explicit empty selections, retaining
their 11 T0 and 13 T1 open questions. Tests include invalid selections, transitive
identity conflicts, valid reference resolution, provisional propagation, and
input immutability; these results do not measure general semantic accuracy.

## T2 frozen scope semantics

[Hand-authored fixtures](fixtures/T_scope_semantics.json) freeze eleven semantic
projections: exact predicate inventory, polarity, ordered contexts and reporting
sources, and modal readings/polarity. Tests also check participant scope, conditional
link outer scope, dependency-complete individual selections, and input immutability.
These are semantic assertions rather than snapshots generated from current output.
They cover this scope increment, not all seven original contract acceptance fixtures;
attempt, choice-point, and quantity export remain future work.

`export_candidate_graph(text, package_id="fixture_id")` supports reproducible tests.
Production calls still generate a fresh package ID. Never reuse a supplied fixture
ID for different production exports. T2 uses schema 0.2 for `modal_choice` contexts
and attributed `report_proposition_id` references; legacy 0.1 remains readable.

Ambiguous modal qualifications survive even when no reading is selected. The
validator keeps affected selections provisional until the modality question is
resolved; abstention never forces a modal interpretation. Direct clausal complements
of say/report/claim/tell/state retain reporting sources and report anchors, including
nested reports and their own negative/conditional scope. This does not establish
that a report occurred. General quotation, hedges, indirect attribution, and other
discourse operators are still outside these bounded rules.

`must not leave` retains a negative proposition under positive modal candidates.
Other locally negated MD constructions use unresolved polarity and a blocking scope
question. Lexical modality (`not required to`, for example) blocks affected readings
with a specific gap question; it is not silently translated into a duty. Scope on
participant records qualifies the proposition, not the existence of the person or
the grammatical role. Conditional and attribution contexts are ordered by their
governing dependency structure, with the conditional outside a report it governs.

Validation: 41 tests passed, including the eleven frozen examples in subtests and
the existing validator/reference/T/S regressions. This verifies these examples and
contract behavior, not unrestricted semantic accuracy.

## Candidate Parliament worlds

The parser-to-Parliament path now has an executable candidate-world adapter in
[`z10_world_model_adapter.py`](z10_world_model_adapter.py). It enumerates compatible
Z10 proposition readings, validates their selections, and emits Parliament schema
1.3 payloads together with exact evidence bindings and explicit construction
problems. It does not authorize admission.

Two frozen runs provide direct evidence. The held-out antivenom scenario yields a
two-action candidate with conditional recovery effects. A stripping sentence yields
three separate candidates for destination, object, and subject reconstructions;
the two readings inconsistent with the supplied destination wording are explicitly
flagged as unresolved action-role alignments.
All four payloads parse through Parliament's actual schema validator with
completeness disabled; unsupported welfare, causation, resource identity, and
exclusivity remain visible problems. Outputs are in
[`z10_candidate_worlds_antivenom.json`](diagnostics/z10_candidate_worlds_antivenom.json),
[`z10_candidate_worlds_stripping.json`](diagnostics/z10_candidate_worlds_stripping.json),
and [`z10_candidate_worlds_parliament_validation.json`](diagnostics/z10_candidate_worlds_parliament_validation.json).

A subsequent fresh serum-allocation probe reached Parliament's current admission
gate and returned `COMMITTED` with no mechanical errors. The saved
[`live trace`](diagnostics/z10_candidate_pipeline_live_serum.json) also records why
that result is not yet semantically safe: the raw admission function ignored the
adapter's false authorization flag and unresolved construction problems. This
separates demonstrated end-to-end transport from admission readiness.
