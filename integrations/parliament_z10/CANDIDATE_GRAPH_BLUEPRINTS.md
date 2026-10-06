# Candidate graph blueprint ensemble

Status: ten normalized evidence plans; six Parliament 1.3 graph builders.

Parliament keeps one world-state schema. These blueprints are alternative initial
graph plans, not new world schemas and not ethical verdicts. A scenario is matched
to several plans; each plan generates a complete candidate graph from supported
text, records unfilled slots, and is evaluated as a whole.

## Initial blueprint set

1. **Exclusive allocation**
   - Required: resource, bounded quantity or capacity, at least two recipients,
     transfer/intervention alternatives.
   - Proposed edges: resource identity, quantity, transfer, recipient, exclusive
     choice, foregone allocation.
   - Optional: eligibility, priority, conditional outcomes.

2. **Conditional intervention and outcome**
   - Required: action alternatives and consequences explicitly conditional on
     those actions.
   - Proposed edges: action, direct intervention, conditional outcome, causal or
     enabling alternatives, outcome bearer.
   - Optional: probability, temporal delay, mechanism.

3. **Ability or permission choice**
   - Required: modal action propositions and possible recipients or targets.
   - Proposed edges: actor, possible action, modal scope, option membership.
   - Optional: exclusivity, outcomes, duty or prohibition.

4. **Rescue under limited capacity**
   - Required: two rescue actions, distinct rescue targets, explicit “not both”
     capacity evidence, and a copied outcome for each branch.
   - Proposed edges: each rescue intervention, saved party, branch-local survival
     and stated harm, and the exclusive choice.
   - Optional: differential risk, group size, temporal urgency.

5. **Diversion or redirection**
   - Required: controllable process, intervention alternatives, affected parties.
   - Proposed edges: action changes process, process reaches party, harm/outcome,
     counterfactual alternative.
   - Optional: action versus omission, uncertainty, secondary effects.

6. **Action versus omission**
   - Required: available intervention and explicit or recoverable non-action.
   - Proposed edges: act, omit, continuation/default process, contrasting outcomes.
   - Optional: duty, responsibility, temporal deadline.

7. **Rule, duty, permission, or prohibition**
   - Required: deontic modal and governed proposition.
   - Proposed edges: obligation/permission/prohibition scope, bearer, authority,
     compliance and violation alternatives.
   - Optional: exceptions, conflicting duties, sanctions.

8. **Promise, commitment, or reliance**
   - Required: commitment event, promisor, content, and promisee or relying party.
   - Proposed edges: commitment, controller, intended future action, reliance or
     expectation outcome.
   - Optional: breach, changed conditions, competing commitments.

9. **Uncertain risk tradeoff**
   - Required: actions with probabilistic or explicitly uncertain outcomes.
   - Proposed edges: action, possible outcomes, likelihood qualifiers, affected
     parties, mutually exclusive outcome branches.
   - Optional: expected quantities, confidence source, ambiguity alternatives.

10. **Disputed report or belief**
    - Required: attributed proposition or conflicting reports.
    - Proposed edges: speaker/source, reported content, belief/report scope,
      competing factual hypotheses.
    - Optional: evidence reliability, downstream decisions, later confirmation.

## Runtime status

The blank-plan ensemble and cloze chooser now expose all ten families. Their
implementation status is explicit:

- `exclusive_allocation`, `conditional_outcome`, `rescue_contrast`,
  `omission_harm`, `diversion_redirection`, and `uncertain_risk` have
  Parliament 1.3 graph builders;
- `ability_permission`, `deontic_rule`, `promise_reliance`, and
  `disputed_report` emit deep withheld envelopes because schema 1.3 cannot
  preserve their modal, normative, commitment, or attribution semantics.

A withheld plan still records its question assessment, accepted evidence,
typed slot bindings, Z10 selection witness where available, unresolved readings,
full source clauses, and named construction problems. Its `candidate` is `null`
and `admission_authorized` is false.

## Reference depth: water-first question gate

The deepest generation contract is the water-allocation sequence, not the
medicine graph alone. Every normalized proposal first records:

1. the ethical question and scenario options;
2. whether exclusivity is evidenced (`not both`) or merely suggested by `or`;
3. every parsed participant, including outcome participants outside recipient
   heads;
4. accepted copied evidence, unresolved slots/readings, and process warnings;
5. `eligible_for_world_state` plus explicit withholding reasons.

Only then may a candidate world be materialized. The medicine allocation builder
contributes the stronger downstream bookkeeping—Z10 selection closure,
receipt/nonreceipt complements, quantities, likelihoods, and death provenance—
inside this water-first envelope.

The cloze templates use family-specific instructions. Rescue requires the
`either … but not both` structure and keeps both rescue actions and their
outcomes in separate branches. Omission keeps an instrument out of the action
list. Conditional branches retain separate condition, bearer, and
outcome slots. Reports remain attributed content rather than admitted facts.
Only exclusive allocation currently has a warning-only implied-process pass;
its non-copied answer is retained as a note and never becomes a causal edge.

## Cross-family generalization invariants

`blueprint_allocation_invariants.py` is shared by the Z10, ensemble, and cloze
paths. A quantified recipient is typed and counted the same way in each path.
For an exclusive allocation:

- a recipient headcount stays on that recipient, its transfer, and its stated
  outcome;
- the nonrecipient complement carries the indivisible resource quantity, not
  the rival recipient's headcount;
- the complement's local clause evidence is restricted to the resource
  constraint; Parliament attaches the global exclusivity and parent-transfer
  evidence during deterministic compilation;
- compilation, admission, serialization, and frozen replay must be idempotent;
- an authorized allocation envelope is invalid unless its pre-world assessment
  records exclusivity as `evidenced`.

The replay requirement is tested for every implemented family: allocation,
conditional outcome, rescue, omission, diversion, and uncertain risk. The
ability/permission, deontic-rule, promise/reliance, and disputed-report families
must instead remain complete evidence envelopes with `candidate: null` and
`admission_authorized: false`; replay tests must not turn those semantics into
world facts.

## Matching and generation

The matcher scores every blueprint using positive textual evidence rather than
domain names. Signals include Z10 candidate types, predicates, roles, quantities,
modal and conditional scope, scenario-option sets, reconstruction alternatives,
and source spans.

For each scenario:

1. Rank blueprints by supported required slots, supported optional slots, and the
   amount of evidence they explain.
2. Instantiate the best three plans independently.
3. Generate alternative slot bindings when reference, role, scope, or relation
   type remains ambiguous.
4. Materialize a Parliament-schema candidate from each instance.
5. Score the resulting whole graphs on evidence coverage, unsupported atoms,
   unresolved required slots, action distinction, Z10 selection validity, and
   Parliament invariant validity.
6. Present the best candidates and their graphs. Do not convert a low score into
   another admission guard; use it to improve candidate generation or choose a
   different blueprint.

Slots can contain candidate sets. For example, an explicitly conditional action
and recovery statement records `CAUSES | ENABLES` in `unresolved_readings` when
the text does not distinguish them; the world model does not invent a second
relation type.

### Parser-first slot recovery

The cloze model selects and supplements a blueprint; it is no longer the sole
source of slot values. After each cloze pass, the builder reuses the frozen Z10
package and its underlying dependency parse to recover exact source spans for:

- typed quantities and aligned resource mentions;
- active and passive transfer predicates, actors, and destinations;
- conditional propositions in either condition-first or condition-last order;
- positive and negated condition pairs, including passive `by` agents;
- outcome clauses, modal/conditional markers, and branch sentences;
- explicit exclusivity variants such as `cannot ... both`;
- nonreceipt descriptions expressed through negation, `untreated`, or
  `left without`.

Recovered slots carry `parsing_game_Z10` provenance in `z10_recovered_slots`.
They remain exact source spans. Z10 ambiguity is preserved rather than silently
resolved. Derived exclusivity uses its proof record to close allocation branches
without fabricating a `not both` quotation. Parliament's matching constraint
recognizers allow modifiers between a singular marker and resource head, so
`one indivisible dose` retains its exact evidence instead of being normalized to
an unsupported phrase.

## Serum scenario: three initial candidates

Text signals:

- exact quantity: one dose;
- resource mention: serum;
- two destination arguments: Ben and Cara;
- explicit `or ... but not both` choice;
- ability modal on the transfer action;
- two action-conditional recovery propositions.

### Candidate A: exclusive allocation

This should generate the resource and allocation backbone first:

- `one dose SAME_REFERENT serum` as a reference candidate;
- `quantity(serum) = 1 dose`;
- `give serum to Ben` and `give serum to Cara` as transfer alternatives;
- exactly one option can be selected;
- each transfer produces a direct receipt effect and a foregone receipt for the
  other recipient.

Recovery remains an attached optional outcome subgraph.

### Candidate B: conditional intervention and outcome

This should generate the outcome chains first:

- giving serum to Ben `CAUSES | ENABLES` Ben's recovery;
- giving serum to Cara `CAUSES | ENABLES` Cara's recovery;
- recovery is proposed as a beneficial health outcome;
- each outcome remains conditional on its corresponding action.

Quantity and exclusivity remain optional construction slots.

### Candidate C: ability or permission choice

This should preserve the modal reading:

- Ada is controller/actor of both possible transfer actions;
- Ben and Cara are alternative destinations;
- `can` retains ability/permission/possibility alternatives;
- `but not both` proposes an exclusive option set.

Resource transfer and recovery are optional expansions.

For this scenario, Candidate A explains the scarce resource and choice structure,
Candidate B explains the stipulated outcome structure, and Candidate C explains
the modal ambiguity. The generator may select A as the base and compose its
supported optional recovery and modal subgraphs, while retaining B and C as
separate comparison candidates.

## Output requirement

[`blueprint_proposal_contract.py`](../../blueprint_proposal_contract.py)
defines the common envelope used by Z10, ensemble, and cloze generation. Every
proposal records `pre_world_assessment`, `accepted_evidence`, typed
`slot_bindings`, selection validation, unresolved readings, construction
problems, withholding reasons, and admission authorization. Authorized
candidates contain only Parliament 1.3 world fields. Withheld semantic families
carry the same provenance envelope with `candidate: null`.

Every unsuccessful candidate-generation run must include:

- the graph actually generated;
- unresolved or alternative edges drawn explicitly;
- the blueprint and slot bindings used;
- evidence attached to every generated atom;
- unfilled required and optional slots; and
- the competing candidates considered.

For a withheld family, “the graph actually generated” is explicitly `null`.
The withholding reason, accepted evidence, and unresolved slots take its place;
no empty or guessed Parliament graph is presented as an implementation.

This output is diagnostic evidence for improving generation. It is not an
additional rejection policy.

## Functional graph builders

The water-first question gate plus exclusive-allocation bookkeeping form the
reference implementation. Conditional outcome
can now materialize one or two independently copied condition/outcome chains
with branch-local chances and explicit positive/negated condition records.
Non-transfer actions use a source-named process such as the lever between the
direct intervention and a different party's welfare outcome. This preserves
Parliament's required `act -> process state -> outcome` topology without
inventing a process absent from the text. Rescue contrast requires two copied
rescue actions, explicit `not both` evidence, and independent copied outcomes
for each branch. Omission
harm builds separate positive and negated action branches, copies each complete
harm proposition and group count, and retains a brake or other instrument only
as context.

The non-allocation ensemble proposals now use the same outer bookkeeping shape:
source clauses, assignment, slot bindings, Z10 selection and validation,
action-source rows, world model, and unresolved required slots. Cloze proposals
carry the same shape where available; because cloze copies are not Z10 candidate
selections, their selection validation is explicitly `not_assessed`.

## Functional allocation baseline

The first blueprint is implemented in
[`candidate_graph_blueprints.py`](../../candidate_graph_blueprints.py). It matches
the serum scenario and fills every required slot:

- two action options from Z10 `OPTION_OF` candidates;
- one exact resource quantity from `QUANTITY` plus the source phrase
  “one dose of serum”;
- explicit exclusivity from “or ... but not both”;
- a shared actor, shared resource, and distinct destinations;
- conditional recovery propositions and their bearers.

The proposal contains four parties, two actions, six effects, and two explicit
source-stipulated recovery links. Each action has a direct receipt, exclusive
nonreceipt for the other recipient, and recovery outcome. Parliament's construction
engine retains those atoms and adds the two allocation-complement causal links.

Validation result:

- Z10 selection contract: valid;
- Parliament completeness: valid;
- Parliament admission: `COMMITTED`;
- admission errors: none;
- admitted graph: four parties, two actions, six effects, four causal links.

The frozen source package, slot bindings, proposed candidate, validation result,
and admitted world are in
[`exclusive_allocation_blueprint_serum.json`](../../diagnostics/exclusive_allocation_blueprint_serum.json).

## End-to-end Parliament use

The serum blueprint has also passed Parliament's downstream deliberation path.
[`run_blueprint_parliament.py`](../../run_blueprint_parliament.py) performs Z10
export, blueprint filling, strict Parliament parsing, admission, canonical-action
construction, and native frozen-trace validation. The frozen trace lets the normal
Parliament pipeline consume this admitted world without asking another model to
redraw it.

The first OpenAI run used utilitarian and deontological delegates for one cycle.
The trace records `world_generation_calls: 0`, six admitted effects, and four
causal links. Both delegates compared the two recoveries and the complementary
nonreceipt effects. The utilitarian scores were exactly tied. Parliament returned
`UNRESOLVED` because the source supplies no morally relevant difference between
Ben and Cara; it did not invent one. This is useful downstream behavior from a
symmetric graph, rather than a failure to consume the graph.

Artifacts are under
[`diagnostics/blueprint_parliament_serum`](../../diagnostics/blueprint_parliament_serum),
including the source candidate, native frozen trace, semantic-preservation trace,
full Parliament trace, summary, and public answer.

From the parser repository root, the complete run is one command. With no input
argument it uses the development medicine scenario:

```bash
./run_blueprint_parliament.sh
```

Pass arbitrary text directly or from a UTF-8 file:

```bash
./run_blueprint_parliament.sh --text \
  "Maria can divert the trolley toward one worker, and one worker will die."

./run_blueprint_parliament.sh --scenario-file my_scenario.txt
```

The runner first writes `question_assessment.json`, ranks and fills three
blueprints, and writes every graph or withheld attempt to
`candidate_graphs.md` and `candidate_attempts.json`. Only the selected
contract-valid proposal reaches Parliament. A withheld construction still writes
the question, copied evidence, candidate attempts, missing slots,
`blueprint_candidate.json`, and `preparation_manifest.json`; its exit status is
2 and Parliament is not called.

Ranking includes the explicit meta-options `none` and `composite`. If either is
ranked first, construction is withheld while the remaining ranked attempts stay
visible. Every filled proposal now keeps construction provenance outside
Parliament's closed 1.3 world records, using `SOURCE_ASSERTED`,
`STRUCTURALLY_DERIVED`, `WORLD_KNOWLEDGE_HYPOTHESIS`, or `UNRESOLVED`.
Exclusivity has a proof record with status `EXPLICIT`, `DERIVED`,
`HYPOTHESIZED`, or `UNKNOWN`.

Bare conditionals no longer become `CAUSES` automatically. The working
Parliament graph uses `ENABLES` where Parliament requires a downstream parent,
while the proposal retains `ENABLES | CAUSES` as an unresolved relation
alternative. Explicit causal, enabling, and preventing words license their
corresponding relation. A conditional that restates the parent action remains an
explicit condition record and appears as branch scope in the diagnostic graph.
It is not attached as a `condition_id` gate to a `CERTAIN` effect or link: the
selected action and its process-state edge already carry that dependency, and
Parliament rejects a redundant gate that restates its own source. Independent
conditions continue to use attached 1.3 condition structures.

The runner also writes `z10_blueprint_comparison.json`. It compares predicates,
roles, scope structures, and relations for every candidate attempt. Differences
remain diagnostic and never rewrite Z10, change the chosen blueprint, or force
agreement.

An admitted run additionally writes `world_state_topology.md`,
`world_state_topology.json`, and the native `frozen_world_trace.json`, then runs
Parliament and records every result path in `preparation_manifest.json`. The
wrapper loads the existing OpenAI environment without copying secrets into an
artifact. Set `BLUEPRINT_PYTHON` to select a Python executable; otherwise it uses
the repository `.venv` and falls back to `python3` in a clean checkout.

After admission the runner hands the frozen world to RelEnt's
`global_workspace_pipeline.py`. It does not call `parliament.py`, which would
rewrite a new scenario file. Default deliberation is full RelEnt: all five
frameworks, original agents, three cycles, and synthesis/planning/audits on.
The manifest records the Parliament commit, branch, modified source files, and
integration-patch fingerprint. Completed runs also write
`parliament/deliberation_report.md`: original testimony and errors, every cycle
including dissent, framework ledgers, and audit/synthesis records. The native
short answer remains available separately. Admission failures retain their error
and candidate artifact paths in the manifest.
On a TTY it presents RelEnt's own choices (frameworks, max cycles, corpus RAG)
unless those flags were already passed. `--prepare-only` still stops after
graph admission and never starts deliberation.

`--prototype-deliberation` is a pipeline-only cheap smoke: utilitarian and
deontological compact specialists, original agents skipped, one cycle, and no
synthesis, planning, or audits. RelEnt itself has no matching flag.

```bash
# Graph only
./run_blueprint_parliament.sh --prepare-only

# Full RelEnt after cloze admission (TTY prompts unless flags are set)
./run_blueprint_parliament.sh --text "A clinic has one dose of medicine. ..."

# Cheap compact smoke
./run_blueprint_parliament.sh --prototype-deliberation
```
This is the baseline that later blueprints must reproduce: match, fill, show the
graph, validate, and admit before expanding the blueprint ensemble.

## Additive graph amendments (2026-10-04)

The runner now retains the chooser's original candidates and appends separate
amended candidates when Z10 exposes additional supported conditional branches.
The amendment layer uses the existing constructors and Parliament 1.3 schema;
it has no fixed branch/node count. Matching actions and process nodes are reused,
and additional outcomes retain their own source clauses and quantities. Every
candidate core gets native admission; the highest-ranked admitted attempt wins.
An amended candidate is considered immediately before its original, which stays
available if the amendment fails. Unsupported constructions remain visible in
`amendment_inventory.json`.

`flexible_graph` is an additional fallback constructed from the same branch
inventory. It can represent more than two explicit conditional branches, even
when the chooser returned `none` or `composite`. It does not yet represent every
kind of moral dilemma: unresolved references, coordinated consequences, attributed
conditions, and predicates outside the existing outcome interpreter stay recorded
as gaps. No extra model call is made for this expansion.

The supported-core projection preserves arbitrary existing causal, temporal,
and counterfactual links between retained nodes. Hypothetical effects and relations
to their excluded endpoints remain in `hypothesis_overlay.json`. The integration
patch removes Parliament's automatic averted-harm construction step so it cannot
reinsert those hypotheses during admission/replay. Explicit source-supported
benefits continue to be admitted. This is a construction change; validation rules
and the world-state schema are unchanged.

Native example: the two-branch trolley baseline plus “If Maria presses the button,
two workers will survive.” produces admitted amended and flexible graphs with
three actions, nine effects, and six links. See
`diagnostics/blueprint_amendment_three_branches_final/candidate_graphs.md` and
`admission_results.json`. These are deterministic construction/admission checks,
not a fresh live-model generation or deliberation run.

### Explicit process-state outcomes

The additive layer also recognizes positive, unhedged `will` outcomes for
parser predicates `stop`, `start`, `open`, `close`, `fail`, and `activate`, when
the bearer is explicitly a signal, alarm, gate, door, pump, brake, engine, valve,
machine, or switch. It uses `PHYSICAL_STATE` on a `PROCESS` party; no injury,
benefit, or survival consequence is inferred. The source sentence, modality,
parent link, and provenance remain bound to the added node. A conditional remains
a branch association with relation alternatives, not proof of causation.

Example: adding “If Maria pulls the lever, the warning signal will stop.” to the
two-branch trolley yields seven admitted effects while the original six-effect
candidate remains available. See
`diagnostics/blueprint_process_state_amendment_final/candidate_graphs.md`.
Negated and hedged process readings remain visible unresolved constructions;
they are not converted into positive/certain states. Other physical predicates
and nonconditional causal/temporal sentences are not yet generally generated by
this layer. Existing builders and the Parliament schema are unchanged.
## Primitive composition pilot (choice plus conditional outcomes)

The runner now exports `primitive_inventory.json` before macro ranking. It reuses
the Z10-backed conditional extractor and retains scenario-option sets, evidence,
scope, and the candidates needed by those constructions. This is a partial
inventory; the original Z10 package remains the record for other semantics.

After the existing baseline/amended/flexible attempts, the runner appends a
`primitive_composition` attempt built directly from the inventory. It uses the
existing conditional atom builder, graph union, provenance closure and supported
core projection, without a blueprint-family dispatch or new Parliament schema.
It does not change the ranks of existing candidates. The pilot covers at most
two constructed actions; larger inputs still follow the existing blueprint paths.
Additional outcomes on either action do not create another construction family.

Choice alternatives retain their original scope and selection rule in the
inventory. They do not establish exclusive actions, exhaustiveness or occurrence.
This increment does not resolve ambiguous modalities, references or unsupported
outcomes. It tests the already-supported positive conditional outcomes, including
an explicit process-state outcome on an existing branch.

`primitive_comparison.json` compares supported cores (not speculative overlays):
effect readings and ID-independent topology, preserving quantities, parents,
conditions, and relation records. Differences are diagnostics, not admission
requirements. Every composition attempt receives the usual candidate graph and
native admission result, including an explicit pilot-scope problem when withheld.

The reproducible deterministic probes are under
`diagnostics/primitive_composition_pilot/{baseline,signal}`. They use fixed cloze
bindings to isolate construction from model variation and run native Parliament
admission/frozen-trace validation. They do not run live model generation or RelEnt
deliberation. The baseline admits six effects; adding the warning-signal stopping
outcome admits seven, on the same two actions.

### Bounded Care evidence binding

The pinned Parliament integration patch also corrects Care's effect references.
Care selects evidence within the assessed action and matches the full affected
party label (case/spacing/leading articles normalized; quantities preserved).
Keywords in an explanation about rival parties cannot establish party identity.
The live prompt asks for copied local party labels and keeps unsupported
relationships as framework interpretations. There is no inferred entity merge.

An unmatched scenario-grounding claim remains a committed uncertain assessment,
without a false `SUPPORTED_BY` edge; it does not reject or rewrite the world.
`effect_grounding_status` makes matched/unmatched evidence explicit. Matching a
party's outcome is not proof of entrustment, intent, or normative priority.
`test_parliament_care_binding.py` replays the earlier Care failure against the
frozen composition graph and checks wrong-party, wrong-action, quantity, valid
binding, and world preservation. Other frameworks' query behavior is unchanged.

### Entrustment hypothesis containment

Care's structured ENTRUSTED reading now enters the existing proposition ledger
as an unverified hypothesis, even if a model omits it from its empirical-premise
list. ENTRUSTED_RESPONSIBILITY rankings carry it as decision-critical; an outcome
citation cannot verify the relationship. Care assessments separately display
`relationship_evidence_status: UNVERIFIED` for this reading. Other relationship
types remain outside this bounded extension; this is not a general relation verifier.

Shared unresolved dependency records include origin (`introduced_by`), epistemic
type/status, evidence support IDs and derivation dependencies. Existing identity
and status machinery preserves the origin when another agent cites or repeats the
same hypothesis. Adoption retains its uncertainty, and recommendation continuity
cannot silently drop it. Repetition changes attention counts, not truth status.

`test_parliament_hypothesis_containment.py` exercises storage, broadcast data,
explicit citation and repeated-claim adoption, a declaration of ESTABLISHED that
cannot override the ledger, and next-cycle continuity. It confirms the world and
semantic graph remain exactly unchanged. It tests declared adoption of the same
claim, not reliable detection of every hidden or paraphrased premise in prose.


### Quantity atoms cannot verify agent outcomes or relationships

A two-cycle live Care/Deontology replay exposed a binding error: an independent
NEW_HYPOTHESIS audit finding about five workers surviving was rebound to the
established "five workers" cardinality atom. Quantity-only atoms now accept
canonical copies only; they do not borrow outcome/action context from the effect
that seeded them. This rule also covers audit-supplied quantity IDs. Other atoms'
paraphrase matching remains unchanged, and world admission is unchanged.

`test_parliament_hypothesis_containment.py` replays the actual live audit findings,
checks repeated hypothesis identity and Care attribution, checks explicit-ID
bypass attempts, and preserves genuine quantity/effect bindings and the admitted
graph. Evidence and limitations: `diagnostics/hypothesis_containment_live/RESULT.md`.
The live run occurred before this fix; post-fix verification is deterministic.


### Source accounting and promise retention

The pre-ranking inventory now retains all Z10 candidates, source choice sets,
open questions, parser coverage and producer resources. Every candidate has a
construction consumer or a named unconsumed-reading question. This is accounting
of generated readings, not a completeness claim or an additional admission gate.

A bounded `promise` primitive retains Z10's predication, participants, complement
links and scope. It does not infer a semantic promisee from an object, resolve
reference, or establish reliance, breach, duty or content occurrence. Positive,
negative and hypothetical scope survive unchanged. Other unconsumed constructions
remain explicit instead of being recast into conditional outcomes.

All runner attempts (including withheld ones) show retained promise graphs and
missing-construction questions. The proposal envelope uses its existing
`unresolved_readings` field; Parliament's world schema is unchanged. Independent
composition retains these diagnostics even when called without the runner.
`semantic_coverage.json` and the preparation manifest expose accounting separately
from admission. Exact source-aligned conditional records are diagnostics only.
Unconsumed candidates do not necessarily mean omitted world content because
existing builders may recover it directly from text.

The baseline, promise and negated-promise preparation probes admit two actions
and six supported effects with unchanged branch readings. Promise graphs are
retained outside the admitted world: **a typed mapping to Parliament specialists
remains unimplemented**. These are fixed-macro native-admission tests, not live
promise deliberation. Evidence: `diagnostics/primitive_promise_coverage/RESULT.md`.
Tests: `test_blueprint_semantic_coverage.py` plus the existing regression suite.
