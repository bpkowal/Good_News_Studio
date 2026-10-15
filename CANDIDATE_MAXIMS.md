# Candidate maxims in deontological reasoning

Each newly generated deontological assessment includes `dp[action].mx`, a
candidate reconstruction of practical reasoning for that option. Omissions need
their own maxim too. The record contains:

- circumstance, policy, end, and chosen means;
- the protected person's proposed role: intended means, foreseeably affected,
  not used, or unresolved;
- the consent constraint and a case-specific universalization argument;
- separate assessments of action permissibility and the maxim's defensibility.

New replies additionally record `instrumental_belief`, `instrumental_use`, and
`practical_reasoning`: in the proposed policy, does the agent believe employing
the person furthers its end, and does it employ the person for that purpose?
`means_effect_ids` and `end_effect_ids` bind physical claims to effects of the
same admitted action. An unknown belief/use stays unknown; these fields do not
establish the scenario actor's actual mental state. Older maxims remain readable
without backfilling those fields.

A candidate maxim is an attributed argument, not evidence of the scenario
actor's intention. The native framework ledger stores it as `candidate_maxim`,
with `maxim_origin: CANDIDATE_INTERPRETATION`. It does not add facts to the frozen
world, authorize consent, or resolve missing intentions. Old saved replies without
maxims remain readable and are not silently backfilled.

The schema and prompt are in the shared native `CompactLocalSpecialist`, so the
compact framework and RelEnt paths using that specialist receive them. The
locked, reproducible integration is
[candidate_maxims.patch](integrations/parliament_z10/candidate_maxims.patch),
followed by
[causal_means_reasoning.patch](integrations/parliament_z10/causal_means_reasoning.patch).

## Targeted challenges

When a deontological objection is scheduled, `logic_puzzles_review.review_jobs`
adds the actual prior maxims and classifications to its localized review packet.
An older argument with no maxim is explicitly asked to supply one. The review
asks the agent to:

1. Identify the inference from purpose and means to the claimed duty priority.
2. Defend or retract instrumentalization claims when the harm was classified as
   a foreseen side effect. External causal necessity alone cannot establish the
   actor's practical reasoning.
3. Explain the consent constraint without inventing consent.
4. Answer the rescue objection to the omission policy.
5. Evaluate the particular maxim and distinguish its quality from the action's
   permissibility. Defend, revise, or preserve uncertainty.

This uses the existing native review route and its two-objection bound. It does
not automatically schedule review for every maxim or add a new debate round.
The model's `RESOLVED` label cannot itself clear a native objection.

Calibration checks explicit inconsistencies between the maxim's person-role,
means classification, action assessment, and duty verdict. A benevolent purpose
does not establish permissibility; an indefensible maxim does not by itself
prohibit the action. These consistency checks cannot prove a philosophical
argument sound. Existing causal-topology checks remain in place; this increment
checks that proposed physical means have an admitted path to their proposed end.
An intervention producing two sibling outcomes does not establish that one
outcome is a means to the other. Conversely, even a supported physical path is
insufficient to establish the proposed instrumental belief and purposive use.

## Causal handoff

The physical-scene producer now exports `CAUSES` for the source-supported
instrument operation → redirection → stated same-process outcomes. Those
relations retain explicit construction provenance, source clauses, modality, and
conditions. Other conditionals are not generally upgraded. An uncertain,
attributed, negated, or unmatched mechanism cannot license the upgrade.

The omission → continuation link stays `ENABLES`; it is a dependency on a choice
branch, not physical causation by the decision-maker. The native doing/allowing
checker is unchanged: the producer now supplies the distinction it already
expects. The reader diagram displays the admitted relation type and retains its
symmetric branches.

Existing frozen worlds are immutable. Rebuild a new candidate to obtain the
corrected relation types; old saved `ENABLES` graphs are not silently rewritten.
These are conditional branch models, not claims that Dana pulled or omitted.

## Reports and verification

The report explanation presents the candidate policy, purpose, means,
justification, and its two assessments, explicitly labeled as an interpretation
rather than established intention. Raw records retain the complete maxim.

Run the focused native tests:

```bash
.runtime/venv/bin/python -m unittest test_candidate_maxims test_means_end_reasoning test_negation_boundary_validation -v
```

Run related application and runtime regressions:

```bash
.venv/bin/python -m unittest test_logic_puzzles_review test_framework_presentation test_parliament_runtime -v
.venv/bin/python -m unittest test_trolley_causal_provenance test_trolley_cloze_recovery test_blueprint_spatial_projection test_branch_choice_handoff -v
```

These tests exercise real native generation, ledger projection, recurrent review,
and objection reconciliation using controlled responses. They establish transport
and validation behavior, not improved live-model argument quality.

The demo can now offer opt-in follow-ups to the revised justification. See
[Optional deeper reasoning review](DEEPER_REASONING_REVIEW.md) for the prompt,
round limits, continuation commands, and remaining evaluation limits.
