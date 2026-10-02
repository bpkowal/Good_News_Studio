# Parsing game M

M is a standalone revision of L. L is unchanged. Run:

```sh
python parsing_game_M.py
python -m unittest test_parsing_game_M -v
```

M uses its own `diagnostics/parsing_game_M_user_probes.json` and M-prefixed
diagnostics. It retains the latest-five probe behavior without migrating or
overwriting older versions' probe files.

## Proposition targets

`parse_world_state(text, policy)` now returns JSON-serializable `propositions`
alongside `causal_claims`, `events`, `entities`, and `alternatives`. Output schema
version is 2. These proposition records describe syntax, not established facts.

For "The flood caused the library to close":

```text
p0: The flood --caused--> [EVENT p1]
    syntactic object/controller: the library
p1: the library --close--> ?
    parent: p0
```

The causal record keeps its original feature scores as diagnostics but clears
`target`, `target_entity`, and `direction`. It exposes
`target_kind="proposition"`, `target_proposition_ids=["p1"]`, and
`proposition_frame_id="p0"`. It is **not eligible for commitment** and carries
`event_complement_requires_proposition_reasoning`. The current CEM policy scores
entity pairs, so these scores cannot validate an event-target edge.

This guard applies to `parse_claims()` and `parse_sentence()` as well as the
combined API. It also blocks the noun edge when the embedded frame cannot be
recovered; in that case the target-ID list is empty. Direct known causal claims
without event complements retain their existing behavior.

Event/state records carry `proposition_frame_id` and
`complement_proposition_ids` to connect the two structural views. Saved user probe
results include the propositions, and the terminal prints their event targets.

## Parent recovery and boundaries

The proposition pass retains predicates with clause complements, including
"allowed" in "The manager allowed the workers to leave." Parent and child links
are bidirectional. The causal training pass keeps its original candidate policy;
permission, compulsion, and reporting do not become causal assertions merely
because their propositions are linked.

`complement_frame_ids` preserves all directly recovered complements. The legacy
singular `event_object_frame_id` is populated only when there is exactly one;
multiple children no longer silently overwrite it.

The existing quantity-change, event/state, and uncertainty representations remain.
General control/coreference, coordinated complement recovery, and proposition-level
semantic inference remain incomplete. Negation/modality on a parent is not proof
that an embedded event occurred. Partial child frames remain conservative.

Policy schema remains 2 because CEM features and fitting are unchanged. The new
guard changes commitment behavior, not the learned classifier. Regression tests
cover public API guards, missing child recovery, permission/force parents, scoped
claims, multiple complements, original causal suites, quantity change, and saved
probe serialization.

## Adversarial sentence regressions

`AdversarialEmbeddingTests` in `test_parsing_game_M.py` adds 12 sentence fixtures
without changing CEM training, its lexicon, or parser behavior. They cover lexical
substitution, negation, modality, permission, denied permission, reporting, passive
control, coordinated complements, independent clauses, nearby non-event nouns,
conditionals, and nested reporting. Assertions check argument identity, bidirectional
links, event-target IDs, scope status, and commitment—not merely predicate presence.

Nine fixtures currently satisfy their structural expectations. Three deliberately
retain the desired behavior as `unittest.expectedFailure` checks:

- Passive control: "The museum was forced to close by the outage." loses the
  subject of `close`.
- Coordinated complements: "The outage caused the museum to close and the workers
  to leave." does not link `leave` to `cause`.
- Clause boundaries: "If the outage caused the museum to close, refunds would
  follow." incorrectly gives `close` the object `refunds`.

These are known defects, not passing extraction coverage. A separate ordinary
test requires all three to remain ineligible for causal commitment; an expected
structural failure cannot hide a failing commitment check. Remove the expected-
failure decorators when the defects are fixed (unexpected success fails the suite).
The existing synthetic multi-complement test remains useful for link bookkeeping;
the new coordinated sentence tests actual end-to-end discovery.
