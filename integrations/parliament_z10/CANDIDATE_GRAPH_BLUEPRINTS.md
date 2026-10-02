# Candidate graph blueprint ensemble

Status: proposed generation architecture.

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
   - Required: endangered parties, rescue actions, limited rescuer capacity.
   - Proposed edges: rescue intervention, saved party, foregone rescue, survival
     or harm candidates, exclusive choice.
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
and recovery statement can propose both `CAUSES` and `ENABLES` relations when the
text does not distinguish them. The graph keeps both readings rather than omitting
the edge entirely.

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

Every unsuccessful candidate-generation run must include:

- the graph actually generated;
- unresolved or alternative edges drawn explicitly;
- the blueprint and slot bindings used;
- evidence attached to every generated atom;
- unfilled required and optional slots; and
- the competing candidates considered.

This output is diagnostic evidence for improving generation. It is not an
additional rejection policy.

## Functional baseline: exclusive allocation

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

From the parser repository root, the complete default run is now one command:

```bash
./run_blueprint_parliament.sh
```

It creates `diagnostics/blueprint_parliament_runs/<timestamp>/`, prepares and
admits the graph, writes `world_state_topology.md` and
`world_state_topology.json`, runs Parliament, and records every result path in
`preparation_manifest.json`. The wrapper internally applies the frozen-world
test profile and loads the existing OpenAI environment without copying secrets
into an artifact.

To stop after graph admission and topology generation:

```bash
./run_blueprint_parliament.sh --prepare-only
```
This is the baseline that later blueprints must reproduce: match, fill, show the
graph, validate, and admit before expanding the blueprint ensemble.
