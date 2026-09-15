"""Named invariants for the attack suite.

This module is a catalog, not a checker. Production validators must not import
it to decide what the world contains. Tests may import IDs for reporting.
Expected outcomes belong in the test (or a Hypothesis strategy field), stated
from the case's declared topology — not from walking the production graph
with the same function under test. Hypothesis generates, shrinks, and
replays; this catalog only names the rules.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntegrityLayers:
    """Multidimensional coverage for semantic-integrity invariants.

    Distinct from Hypothesis ``coverage`` (story/structural/none). A phenomenon
    can be cataloged and structured-tested while grounding remains untested.
    """

    cataloged: bool = True
    seed_case: bool = False
    structured_property: bool = False
    grounding_property: bool = False
    production_validator: bool = False
    metamorphic_test: bool = False
    end_to_end: bool = False


@dataclass(frozen=True, slots=True)
class Invariant:
    id: str
    layer: str
    description: str
    coverage: str
    tests: tuple[str, ...]
    oracle_risk: str
    notes: str = ""
    production: str = ""
    # Semantic-integrity extras (optional; SymPy/Hypothesis entries leave blank).
    integrity_layers: IntegrityLayers | None = None
    enforcement: str = ""  # experimental | cataloged | enforced
    source_type: str = ""  # fracas | parliament_extension | tooling | ""


# coverage: story = one labeled fixture; structural = labels are incidental;
# none = named here because we need it, not yet tested as an invariant.
# production: pass = Hypothesis currently green; fail = currently falsifies;
# untested = no structural wire. Empty derives from coverage.
# oracle_risk: independent = expected value is in the fixture; coupled = test
# asks production to classify using the same path it uses live.

INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        id="CERTAIN_QUARANTINE",
        layer="epistemic",
        description=(
            "A decision-critical descriptive hypothesis that reopens or denies "
            "an admitted actual CERTAIN row cannot uniquely rank."
        ),
        coverage="structural",
        tests=(
            "invariants.test_epistemic.EpistemicInvariantTests.test_certain_reopen_cannot_uniquely_rank",
            "EpistemicHypothesisBindingTests.test_contradicting_certain_row_quarantines_candidate",
            "EpistemicHypothesisBindingTests.test_agreeing_necessity_claim_does_not_quarantine",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_quarantine is the oracle: CERTAIN row plus "
            "a declared REOPEN claim. RESTATE and AGREE, and POSSIBLE rows, "
            "must not quarantine."
        ),
    ),
    Invariant(
        id="COUNTERFACTUAL_MODALITY_PRESERVATION",
        layer="epistemic",
        description=(
            "An unsettled POSSIBLE or UNKNOWN world row cannot be rebound as a "
            "settled event. Hedged restatements may bind; unhedged ones may not."
        ),
        coverage="structural",
        tests=(
            "invariants.test_epistemic.EpistemicInvariantTests.test_unsettled_row_is_not_rebound_as_settled",
            "EpistemicHypothesisBindingTests.test_possible_row_is_not_rebound_as_settled",
            "EpistemicHypothesisBindingTests.test_hypothesis_about_possible_row_is_not_certain_quarantine",
            "EpistemicHypothesisBindingTests.test_seeded_possible_injury_keeps_modality_and_health_dimension",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_bind is the oracle: UNHEDGED restatements of "
            "POSSIBLE or UNKNOWN rows must not bind; HEDGED and CANONICAL may. "
            "CERTAIN unhedged restatements still bind. Presentation listing "
            "PROBABILISTIC rows as established facts is PRESENTATION_PRESERVES_MODALITY."
        ),
    ),
    Invariant(
        id="PROPOSITION_IDENTITY",
        layer="epistemic",
        description=(
            "A restatement of an admitted world atom rebinds to that atom. "
            "Recurrence increases attention, never status."
        ),
        coverage="structural",
        tests=(
            "invariants.test_epistemic.EpistemicInvariantTests.test_restatement_rebinds_without_raising_status",
            "EpistemicHypothesisBindingTests.test_certain_restatement_rebinds_without_quarantine",
            "EpistemicPropositionLedgerTests.test_recurrence_increases_attention_but_never_status",
            "EpistemicPropositionLedgerTests.test_stronger_paraphrase_of_world_fact_becomes_hypothesis",
            "EpistemicPropositionLedgerTests.test_canonical_component_restatement_remains_established",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_bind is the oracle: CANONICAL restatements "
            "rebind and only raise mention_count; STRONGER and NOVEL mint a "
            "hypothesis. Do not ask resolve_proposition for the expected id."
        ),
    ),
    Invariant(
        id="FRAMEWORK_DERIVED_NOT_HYPOTHESIS",
        layer="epistemic",
        description=(
            "Framework-native relations (doing/allowing, means, duty of care, "
            "least-advantaged priority) are not descriptive hypotheses and do "
            "not take the hypothesis confidence cap."
        ),
        coverage="structural",
        tests=(
            "invariants.test_epistemic.EpistemicInvariantTests.test_framework_derived_is_not_a_descriptive_hypothesis",
            "invariants.test_epistemic.EpistemicInvariantTests.test_audit_reclassifies_normative_new_hypothesis",
            "EpistemicHypothesisBindingTests.test_framework_derived_is_not_a_descriptive_hypothesis",
            "EpistemicHypothesisBindingTests.test_audit_reclassifies_normative_new_hypothesis",
            "PropositionIdentityTests.test_framework_derived_does_not_duplicate_world_facts",
            "FrameworkVoteIntegrityTests.test_direct_copy_of_admitted_world_facts_is_not_a_missing_derivation",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.is_framework_derived is the oracle. NORMATIVE "
            "templates use doing/allowing/duty/least-advantaged cues. EMPIRICAL "
            "claims stay hypotheses and take the confidence cap. Direct copies "
            "of admitted world rows (DIRECT_COPY / PROP:WORLD:*) are premises; "
            "vote integrity must not treat them as a missing derivation."
        ),
    ),
    Invariant(
        id="HYPOTHESIS_NOT_CLOSED_WORLD_INPUT",
        layer="utilitarian",
        description=(
            "An unverified hypothesis may be a reversal boundary. It may not "
            "move closed-world utilitarian scores when admitted numeric nets exist."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantities.ClosedWorldHypothesisTests.test_hypothesis_does_not_move_closed_world_scores",
            "BridgeTests.test_unverified_downstream_hypothesis_does_not_move_utilitarian_scores",
            "EpistemicPropositionLedgerTests.test_decision_critical_hypothesis_does_not_unsettle_closed_world_ranking",
            "EpistemicPropositionLedgerTests.test_unverified_downstream_hypothesis_does_not_move_closed_world_scores",
            "UnadmittedMagnitudeRankingTests.test_admitted_numeric_nets_still_rank_despite_minted_hypothesis",
            "UtilitarianSettledWelfareRankingTests.test_hypothesis_does_not_veto_admitted_five_life_comparison",
            "UtilitarianSettledWelfareRankingTests.test_incommensurable_remainder_attenuates_without_erasing_the_five_saves",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_restore is the oracle: NETS restores the "
            "admitted unique ranking; OPEN leaves the submitted scores. Do not "
            "ask closed_world_utilitarian_leader for the expected winner. "
            "Leftover incommensurable admitted rows and unverified knock-ons "
            "must not veto a unique settled same-unit comparison. They keep "
            "the lean, clear cd, and attenuate via cc=false; they must not "
            "invent a canceling point estimate or force ABSTAIN."
        ),
    ),
    Invariant(
        id="UTIL_ARITHMETIC_IDENTITY",
        layer="utilitarian",
        description=(
            "Admitted Util nets and arithmetic EV rows must match a temporary "
            "SymPy recompute of the same already-parsed operands."
        ),
        coverage="structural",
        tests=(
            "test_sympy_arith.SympyArithTests.test_net_matches_probability_times_magnitude",
            "test_sympy_arith.SympyArithTests.test_wildfire_style_expected_harm_identity",
            "test_sympy_arith.SympyArithTests.test_claimed_mismatch_fails_closed",
            "test_sympy_arith.ClosedWorldSympyGateTests.test_leader_requires_sympy_agreement",
            "test_sympy_arith.ExpectedValueSympyGateTests.test_arithmetic_verified_attaches_identity",
        ),
        oracle_risk="independent",
        notes=(
            "SymPy is a calculator only. Oracle is the declared term list and "
            "claimed float in the test; do not ask production to invent "
            "operands. quantity_magnitude remains the English→number authority."
        ),
        production="pass",
    ),
    Invariant(
        id="UTIL_RANKING_BOUNDARY",
        layer="utilitarian",
        description=(
            "A unique closed-world Util ranking may attach a SymPy-derived "
            "certificate: exact welfare gap, numeric assumptions, and when "
            "exactly one probability is free, a one-variable flip boundary."
        ),
        coverage="structural",
        tests=(
            "test_sympy_arith.RankingCertificateTests.test_gap_and_one_variable_boundary",
            "test_sympy_arith.RankingCertificateTests.test_all_certain_emits_gap_without_boundary",
            "test_sympy_arith.RankingCertificateTests.test_closed_world_helper_attaches_serializable_certificate",
        ),
        oracle_risk="independent",
        notes=(
            "Derived artifact with provenance sympy_ranking_certificate; not "
            "world-state truth. Oracle is declared terms and expected "
            "threshold strings. Multivariable boundaries are out of scope."
        ),
        production="pass",
        source_type="tooling",
        enforcement="cataloged",
    ),
    Invariant(
        id="UTIL_INTERVAL_ROBUSTNESS",
        layer="utilitarian",
        description=(
            "Given a caller-supplied interval for the single free probability, "
            "classify the ranking as ROBUST_LEADER, ROBUST_CHALLENGER, "
            "CONDITIONAL, or INDETERMINATE over that interval."
        ),
        coverage="structural",
        tests=(
            "test_sympy_arith.IntervalRobustnessTests.test_robust_leader_on_interval",
            "test_sympy_arith.IntervalRobustnessTests.test_robust_challenger_on_interval",
            "test_sympy_arith.IntervalRobustnessTests.test_conditional_flip_inside_interval",
        ),
        oracle_risk="independent",
        notes=(
            "Do not invent bounds; only attach robustness when the caller "
            "supplies [lo, hi]. Oracle is declared interval + expected status."
        ),
        production="pass",
        source_type="tooling",
        enforcement="cataloged",
    ),
    Invariant(
        id="UTIL_SUBSTITUTION_INTEGRITY",
        layer="utilitarian",
        description=(
            "Arithmetic EV rows retain a symbolic expression alongside the "
            "evaluated value; substituting declared bindings must recover the "
            "claimed float (catches omitted p and double multipliers)."
        ),
        coverage="structural",
        tests=(
            "test_sympy_arith.SubstitutionIntegrityTests.test_ev_row_keeps_symbolic_expression",
            "test_sympy_arith.SubstitutionIntegrityTests.test_substitution_recovers_claimed",
            "test_sympy_arith.SubstitutionIntegrityTests.test_double_multiplier_fails",
        ),
        oracle_risk="independent",
        notes=(
            "SymPy owns substitution of already-named symbols only. quantity "
            "parsing and effect citation remain semantic authorities."
        ),
        production="pass",
        source_type="tooling",
        enforcement="cataloged",
    ),
    Invariant(
        id="UTIL_DERIVATION_EQUIVALENCE",
        layer="utilitarian",
        description=(
            "Two legal term-lists that claim the same welfare quantity must be "
            "algebraically equivalent under expand/cancel."
        ),
        coverage="structural",
        tests=(
            "test_sympy_arith.DerivationEquivalenceTests.test_equivalent_term_lists",
            "test_sympy_arith.DerivationEquivalenceTests.test_inequivalent_term_lists",
        ),
        oracle_risk="independent",
        notes=(
            "Prefer expand/cancel over heuristic simplify() as the oracle. "
            "Do not ask production to invent alternate derivations."
        ),
        production="pass",
        source_type="tooling",
        enforcement="cataloged",
    ),
    Invariant(
        id="UTIL_RESOURCE_CONSERVATION",
        layer="utilitarian",
        description=(
            "After semantic ownership assigns numeric parts and totals, SymPy "
            "checks exact inequalities such as sum(parts) ≤ total."
        ),
        coverage="structural",
        tests=(
            "test_sympy_arith.ResourceConservationTests.test_sum_leq_total",
            "test_sympy_arith.ResourceConservationTests.test_overallocation_fails",
        ),
        oracle_risk="independent",
        notes=(
            "Ownership/reference is not SymPy's job. Callers pass already-owned "
            "numbers only."
        ),
        production="pass",
        source_type="tooling",
        enforcement="cataloged",
    ),
    Invariant(
        id="ANAPHOR_ENTITY_IDENTITY",
        layer="semantic_integrity",
        description=(
            "Anaphoric reference in scenario clauses must preserve entity "
            "identity: effects that bind an anaphor share the party_id of the "
            "antecedent they resolve to."
        ),
        coverage="structural",
        tests=(
            "invariants.test_anaphor_entity_identity.AnaphorEntityIdentityTests.test_structured_lane_preserves_party_identity",
            "invariants.test_anaphor_entity_identity.AnaphorEntityIdentityTests.test_grounding_lane_segments_discourse_and_preserves_party_identity",
            "invariants.test_anaphor_entity_identity.AnaphorEntityIdentityTests.test_grounding_lane_detects_split_identity",
            "invariants.test_anaphor_negation_hypothesis.AnaphorIdentityHypothesisTests.test_oracle_matches_declared_identity",
        ),
        oracle_risk="independent",
        notes=(
            "FraCaS §3 Anaphora supplies the phenomenon; parliament_expectation "
            "is the oracle, not fracas.gold. Structured lane uses a hand-built "
            "world. Grounding lane runs controlled mini-discourse through "
            "segment_scenario_clauses + parse_world_model. Hypothesis: "
            "strategies.anaphor_identity.anaphor_identity_cases. "
            "Mutations: keep / split_intruder / split_group_kind / "
            "chain_wrong_antecedent. production_validator is not yet wired "
            "into live admission."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=False,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="cataloged",
        source_type="fracas",
    ),
    Invariant(
        id="NEGATION_SCOPE_SIBLINGS",
        layer="semantic_integrity",
        description=(
            "Negating one effect's outcome or condition must not flip "
            "polarity or modality of sibling effects on the same action."
        ),
        coverage="structural",
        tests=(
            "invariants.test_negation_scope_siblings.NegationScopeSiblingsTests.test_structured_lane_preserves_sibling_polarity",
            "invariants.test_negation_scope_siblings.NegationScopeSiblingsTests.test_structured_lane_detects_flipped_sibling",
            "invariants.test_negation_scope_siblings.NegationScopeSiblingsTests.test_grounding_lane_role_attachment_marks_harm",
            "invariants.test_negation_scope_siblings.NegationScopeSiblingsTests.test_grounding_lane_preserves_sibling_polarity",
            "invariants.test_negation_scope_siblings.NegationScopeSiblingsTests.test_grounding_lane_detects_flipped_sibling",
            "invariants.test_anaphor_negation_hypothesis.NegationScopeHypothesisTests.test_oracle_matches_declared_sibling_scope",
        ),
        oracle_risk="independent",
        notes=(
            "Parliament extension (operator_scope), not a FraCaS top-level "
            "section. Classic case: 'receives no treatment and dies' — "
            "negation scopes to treatment; death stays ADVERSE. Oracle is "
            "parliament_expectation. Grounding lane also checks production "
            "relational_role_bindings attachment. Hypothesis: "
            "strategies.negation_scope.negation_scope_cases with corruption "
            "mutations flip_polarity / flip_modality / flip_both / "
            "neutralize / beneficial_possible."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=False,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="cataloged",
        source_type="parliament_extension",
    ),
    Invariant(
        id="QUANTIFIER_PARTY_COUNT",
        layer="semantic_integrity",
        description=(
            "A stated cardinal quantity attaches only to the party whose "
            "noun phrase uniquely owns it; it must not silently leak onto "
            "another party."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantifier_party_count.QuantifierPartyCountTests.test_structured_lane_preserves_count_on_party",
            "invariants.test_quantifier_party_count.QuantifierPartyCountTests.test_structured_lane_detects_leaked_quantity",
            "invariants.test_quantifier_party_count.QuantifierPartyCountTests.test_grounding_lane_preserves_count_on_party",
            "invariants.test_quantifier_party_count.QuantifierPartyCountTests.test_grounding_lane_closed_class_recovers_leaked_quantity",
            "invariants.test_quantifier_party_count.QuantifierPartyCountTests.test_grounding_lane_production_binder_recovers_count",
            "invariants.test_verb_quantifier_hypothesis.QuantifierCountHypothesisTests.test_oracle_matches_declared_count_attachment",
            "invariants.test_repair_ledger.PromotedQuantifierValidatorTests.test_production_rejects_leaked_quantity",
            "invariants.test_repair_ledger.PromotedQuantifierValidatorTests.test_repair_card_for_quantifier_leak",
        ),
        oracle_risk="independent",
        notes=(
            "FraCaS §1 Quantifiers supplies the phenomenon; parliament_expectation "
            "is the oracle. Complements parser QUANTITY_PARTY_UNIQUENESS. "
            "Production validate_world_model rejects non-owner recordings of "
            "assigned_party_quantities spans (QUANTIFIER_PARTY_LEAK). Hypothesis: "
            "strategies.quantifier_count.quantifier_count_cases."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="fracas",
    ),
    Invariant(
        id="PLURAL_MEMBER_DISTINCTNESS",
        layer="semantic_integrity",
        description=(
            "Conjoined plural members remain distinct PERSON parties; a "
            "silent merge into one collective party is not admitted as the "
            "same reading."
        ),
        coverage="structural",
        tests=(
            "invariants.test_plural_member_distinctness.PluralMemberDistinctnessTests.test_structured_lane_keeps_members_distinct",
            "invariants.test_plural_member_distinctness.PluralMemberDistinctnessTests.test_structured_lane_detects_silent_merge",
            "invariants.test_plural_member_distinctness.PluralMemberDistinctnessTests.test_grounding_lane_keeps_members_distinct",
            "invariants.test_plural_member_distinctness.PluralMemberDistinctnessTests.test_grounding_lane_detects_silent_merge",
            "invariants.test_discourse_fracas_hypothesis.PluralMemberHypothesisTests.test_oracle_matches_declared_distinctness",
        ),
        oracle_risk="independent",
        notes=(
            "FraCaS §2 Plurals (conjoined NP family). Oracle is "
            "parliament_expectation member_party_ids. Collective GROUP "
            "readings are a different phenomenon; this invariant rejects "
            "silent merge of named members. Hypothesis: "
            "strategies.plural_members.plural_member_cases with mutations "
            "keep / merge_group / drop_one / single_person."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=False,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="cataloged",
        source_type="fracas",
    ),
    Invariant(
        id="ELLIPSIS_PREDICATE_RESOLUTION",
        layer="semantic_integrity",
        description=(
            "VP-ellipsis resolution must reconstruct the antecedent outcome "
            "predicate on a distinct party; inventing a different predicate "
            "fails."
        ),
        coverage="structural",
        tests=(
            "invariants.test_ellipsis_predicate_resolution.EllipsisPredicateResolutionTests.test_structured_lane_resolves_same_predicate",
            "invariants.test_ellipsis_predicate_resolution.EllipsisPredicateResolutionTests.test_structured_lane_detects_wrong_predicate",
            "invariants.test_ellipsis_predicate_resolution.EllipsisPredicateResolutionTests.test_grounding_lane_resolves_same_predicate",
            "invariants.test_ellipsis_predicate_resolution.EllipsisPredicateResolutionTests.test_grounding_lane_detects_wrong_predicate",
            "invariants.test_ellipsis_predicate_resolution.EllipsisPredicateResolutionTests.test_grounding_lane_production_snaps_ellipsis_span",
            "invariants.test_discourse_fracas_hypothesis.EllipsisPredicateHypothesisTests.test_oracle_matches_declared_predicate",
        ),
        oracle_risk="independent",
        notes=(
            "FraCaS §4 Ellipsis (VP-ellipsis family). Oracle is "
            "parliament_expectation outcome identity after resolution. "
            "Grounding also exercises compile_source_proposition_spans. "
            "Hypothesis: strategies.ellipsis_predicate.ellipsis_predicate_cases."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=False,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="cataloged",
        source_type="fracas",
    ),
    Invariant(
        id="ADJECTIVE_MODIFIER_BINDING",
        layer="semantic_integrity",
        description=(
            "Stacked adjectival modifiers bind only the modified outcome "
            "head; sibling rows must not inherit them. Affirmative Adj+N "
            "preserves the head party's kind and label."
        ),
        coverage="structural",
        tests=(
            "invariants.test_adjective_modifier_binding.AdjectiveModifierBindingTests.test_structured_lane_binds_stacked_modifiers",
            "invariants.test_adjective_modifier_binding.AdjectiveModifierBindingTests.test_structured_lane_detects_leaked_modifiers",
            "invariants.test_adjective_modifier_binding.AdjectiveModifierBindingTests.test_grounding_lane_binds_stacked_modifiers",
            "invariants.test_adjective_modifier_binding.AdjectiveModifierBindingTests.test_grounding_lane_detects_leaked_modifiers",
            "invariants.test_adjective_modifier_binding.AdjectiveModifierBindingTests.test_grounding_lane_production_binds_temporals_to_harm",
            "invariants.test_discourse_fracas_hypothesis.AdjectiveModifierHypothesisTests.test_oracle_matches_declared_binding",
        ),
        oracle_risk="independent",
        notes=(
            "FraCaS §5 Adjectives. Complements QUALIFIER_HEAD_BINDING / "
            "effect_expected_qualifiers. Oracle is parliament_expectation. "
            "Hypothesis: strategies.adjective_modifiers.adjective_modifier_cases."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=False,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="cataloged",
        source_type="fracas",
    ),
    Invariant(
        id="TEMPORAL_ORDER_CONSISTENCY",
        layer="semantic_integrity",
        description=(
            "Temporal before/after markers attach to the later event; the "
            "earlier event must not carry the later-order marker."
        ),
        coverage="structural",
        tests=(
            "invariants.test_temporal_order_consistency.TemporalOrderConsistencyTests.test_structured_lane_keeps_after_on_later_event",
            "invariants.test_temporal_order_consistency.TemporalOrderConsistencyTests.test_structured_lane_detects_swapped_order",
            "invariants.test_temporal_order_consistency.TemporalOrderConsistencyTests.test_grounding_lane_keeps_after_on_later_event",
            "invariants.test_temporal_order_consistency.TemporalOrderConsistencyTests.test_grounding_lane_detects_swapped_order",
            "invariants.test_discourse_fracas_hypothesis.TemporalOrderHypothesisTests.test_oracle_matches_declared_order",
        ),
        oracle_risk="independent",
        notes=(
            "FraCaS §7 Temporal reference (before/after family). Does not "
            "claim full tense/aspect calculus. Hypothesis: "
            "strategies.temporal_order.temporal_order_cases."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=False,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="cataloged",
        source_type="fracas",
    ),
    Invariant(
        id="VERB_ASPECT_CULMINATION",
        layer="semantic_integrity",
        description=(
            "Perfective accomplishments license a CERTAIN culmination; "
            "progressive aspect must not force CERTAIN finish."
        ),
        coverage="structural",
        tests=(
            "invariants.test_verb_aspect_culmination.VerbAspectCulminationTests.test_structured_lane_perfective_licenses_certain_finish",
            "invariants.test_verb_aspect_culmination.VerbAspectCulminationTests.test_structured_lane_detects_uncertain_finish_under_perfective",
            "invariants.test_verb_aspect_culmination.VerbAspectCulminationTests.test_grounding_lane_perfective_licenses_certain_finish",
            "invariants.test_verb_aspect_culmination.VerbAspectCulminationTests.test_grounding_lane_detects_uncertain_finish_under_perfective",
            "invariants.test_verb_aspect_culmination.VerbAspectCulminationTests.test_grounding_lane_progressive_allows_noncertain_finish",
        ),
        oracle_risk="independent",
        notes="FraCaS §8 Verbs (aspectual class culmination).",
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=False,
            metamorphic_test=False,
            end_to_end=False,
        ),
        enforcement="cataloged",
        source_type="fracas",
    ),
    Invariant(
        id="VERB_LEMMA_OUTCOME_BINDING",
        layer="semantic_integrity",
        description=(
            "Outcome and source_proposition must share a verbal lemma so "
            "morphological variants of the same event (purge/purged) bind."
        ),
        coverage="structural",
        tests=(
            "invariants.test_verb_lemma_outcome_binding.VerbLemmaOutcomeBindingTests.test_structured_lane_binds_purge_lemma",
            "invariants.test_verb_lemma_outcome_binding.VerbLemmaOutcomeBindingTests.test_structured_lane_detects_wrong_lemma",
            "invariants.test_verb_lemma_outcome_binding.VerbLemmaOutcomeBindingTests.test_grounding_lane_admits_purge_lemma",
            "invariants.test_verb_quantifier_hypothesis.VerbLemmaHypothesisTests.test_oracle_matches_declared_lemma_binding",
            "invariants.test_verb_quantifier_hypothesis.ConsequenceReassignmentHypothesisTests.test_declared_action_ownership",
            "invariants.test_repair_ledger.PromotedVerbLemmaValidatorTests.test_production_rejects_wrong_lemma",
            "invariants.test_repair_ledger.PromotedVerbLemmaValidatorTests.test_repair_card_for_lemma_mismatch",
        ),
        oracle_risk="independent",
        notes=(
            "FraCaS §8 Verbs morphological binding. Production "
            "_verb_lemma_binding_errors (schema 1.2/1.3) rejects outcomes whose "
            "lemma does not bind source_proposition (VERB_LEMMA_MISMATCH). "
            "Hypothesis: strategies.verb_lemma.verb_lemma_cases; sibling "
            "consequence_reassignment_cases remains deferred."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="fracas",
    ),
    Invariant(
        id="OUTCOME_PREDICATE_COMPLETENESS",
        layer="semantic_integrity",
        description=(
            "Atomic effect outcomes must be finished predicates; dangling "
            "copulas or prepositions (purge is, sent to) are rejected at admit."
        ),
        coverage="structural",
        tests=(
            "invariants.test_outcome_predicate_completeness.OutcomePredicateCompletenessTests.test_structured_lane_complete_outcome_holds",
            "invariants.test_outcome_predicate_completeness.OutcomePredicateCompletenessTests.test_structured_lane_detects_dangling_copula",
            "invariants.test_outcome_predicate_completeness.OutcomePredicateCompletenessTests.test_grounding_lane_rejects_dangling_copula",
            "invariants.test_outcome_predicate_completeness.OutcomePredicateCompletenessTests.test_grounding_lane_admits_finished_predicate",
            "invariants.test_outcome_predicate_completeness.OutcomePredicateCompletenessTests.test_production_validator_rejects_purge_is",
            "invariants.test_outcome_predicate_completeness.OutcomePredicateCompletenessTests.test_repair_card_for_incomplete_outcome",
            "invariants.test_status_conservation_hypothesis.OutcomePredicateHypothesisTests.test_production_matches_declared_predicate_oracle",
        ),
        oracle_risk="independent",
        notes=(
            "Parliament extension under semantic status conservation. "
            "Production validate_world_model rejects fragments; repair cards "
            "offer finished rewrites. Hypothesis: strategies.status_conservation."
            "outcome_predicate_cases. Member of SEMANTIC_STATUS_CONSERVATION."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="SOURCE_STIPULATED_OUTCOME_PRESERVATION",
        layer="semantic_integrity",
        description=(
            "Binary-contrast source clauses that attach life-stake "
            "consequences to each side must admit matching effects (or "
            "explicit UNRESOLVED quarantine); silent omission is rejected."
        ),
        coverage="structural",
        tests=(
            "invariants.test_source_stipulated_outcome_preservation.SourceStipulatedOutcomePreservationTests.test_structured_lane_preserves_both_stakes",
            "invariants.test_source_stipulated_outcome_preservation.SourceStipulatedOutcomePreservationTests.test_structured_lane_detects_omission",
            "invariants.test_source_stipulated_outcome_preservation.SourceStipulatedOutcomePreservationTests.test_grounding_lane_completeness_rejects_omission",
            "invariants.test_source_stipulated_outcome_preservation.SourceStipulatedOutcomePreservationTests.test_grounding_lane_completeness_accepts_both_stakes",
            "invariants.test_source_stipulated_outcome_preservation.SourceStipulatedOutcomePreservationTests.test_production_extractor_reads_binary_choice",
            "invariants.test_source_stipulated_outcome_preservation.SourceStipulatedOutcomePreservationTests.test_repair_card_for_missing_stipulation",
            "invariants.test_source_stipulated_outcome_preservation.SourceStipulatedOutcomePreservationTests.test_unresolved_quarantine_satisfies_preservation",
            "invariants.test_status_conservation_hypothesis.BinaryStipulationHypothesisTests.test_production_matches_declared_stipulation_oracle",
        ),
        oracle_risk="independent",
        notes=(
            "Parliament extension under semantic status conservation. "
            "Production validate_world_completeness rejects omitted binary-"
            "contrast life stakes. Hypothesis: strategies.status_conservation."
            "binary_stipulation_cases. Member of SEMANTIC_STATUS_CONSERVATION."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="QUANTITY_BEARING_CONSEQUENCE_PRESERVATION",
        layer="semantic_integrity",
        description=(
            "When a source consequence carries an explicit quantity "
            "(thousands of lives, decades of research), the matched admitted "
            "effect must record that span."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_structured_lane_preserves_quantities",
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_structured_lane_detects_dropped_quantity",
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_grounding_lane_completeness_rejects_dropped_quantity",
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_grounding_lane_completeness_accepts_recorded_quantities",
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_production_extractor_reads_thousands_and_decades",
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_repair_card_for_missing_quantity",
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_party_quantity_satisfies_preservation",
            "invariants.test_status_conservation_hypothesis.QuantityConsequenceHypothesisTests.test_production_matches_declared_quantity_oracle",
            "invariants.test_status_conservation_hypothesis.QuantityConsequenceHypothesisTests.test_det_provenance_complete_repair_is_minimal_across_topologies",
        ),
        oracle_risk="independent",
        notes=(
            "Parliament extension under semantic status conservation. "
            "Production validate_world_completeness requires explicit_quantity_"
            "spans from risk/erase clauses on the matched effect (or party). "
            "Hypothesis: strategies.status_conservation.quantity_consequence_"
            "cases (placement + licensing topologies). DET repair is "
            "provenance-complete and minimal. Member of "
            "SEMANTIC_STATUS_CONSERVATION."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="REPAIR_PROVENANCE_MINIMALITY",
        layer="semantic_integrity",
        description=(
            "A deterministic repair may add only the minimum provenance edge "
            "required to license the repaired fact; it must not spray the "
            "licensing clause onto unrelated sibling effects that share an "
            "action or party."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_repair_provenance_minimality_does_not_spray_action_siblings",
            "invariants.test_quantity_bearing_consequence_preservation.QuantityBearingConsequencePreservationTests.test_deterministic_patch_adds_quantity_and_licensing_clause",
            "invariants.test_status_conservation_hypothesis.QuantityConsequenceHypothesisTests.test_det_provenance_complete_repair_is_minimal_across_topologies",
        ),
        oracle_risk="independent",
        notes=(
            "Paired with QUANTITY_BEARING_CONSEQUENCE_PRESERVATION. "
            "semantic_patch add_quantity restores meaning; licensing_patch "
            "add_provenance restores the evidence edge only for the target "
            "effect named on the repair card. Non-regression against "
            "provenance inflation."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=False,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="REPAIR_NO_EFFECT",
        layer="semantic_integrity",
        description=(
            "If repair(op) followed by validate yields the same violation on "
            "the same target, the repair was non-effective and must surface "
            "as UNSTABLE_REPAIR / REPAIR_NO_EFFECT rather than looping."
        ),
        coverage="structural",
        tests=(
            "invariants.test_repair_no_effect.RepairNoEffectGuardTests.test_same_target_after_applied_patch_is_unstable",
            "invariants.test_repair_no_effect.RepairNoEffectGuardTests.test_annotate_admit_merges_and_escalates_scope",
            "invariants.test_repair_no_effect.RepairNoEffectHypothesisTests.test_oracle_agrees_with_unstable_detection",
            "invariants.test_repair_no_effect.RepairNoEffectHypothesisTests.test_unstable_annotates_escalates_and_skips_det",
            "invariants.test_repair_no_effect.RepairNoEffectHypothesisTests.test_provenance_complete_det_is_not_unstable",
            "invariants.test_repair_no_effect.RepairNoEffectHypothesisTests.test_quantity_only_det_without_clearance_is_unstable",
        ),
        oracle_risk="independent",
        notes=(
            "Generic unstable-repair guard beside DETERMINISTIC_LOCAL_PATCH. "
            "Fingerprint: (code, entity_id, field). When DET applied patches "
            "targeting that entity and the fingerprint remains after re-admit, "
            "emit REPAIR_NO_EFFECT, escalate to SUBGRAPH_REBUILD, and skip "
            "further DET retries for the session. Hypothesis: "
            "strategies.repair_no_effect + quantity_consequence omit bank."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="QUANTITY_PRECISION_NON_ESCALATION",
        layer="semantic_integrity",
        description=(
            "A vague source quantity (dozens, hundreds, thousands, millions, "
            "several thousand, …) may not be refined to a sharper numeric "
            "literal (10,000, ~10,000, 5–10 000, 100 000×80 life-years, …) "
            "unless that exact numeral is already licensed by source text or "
            "a named derivation."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantity_precision_non_escalation.QuantityPrecisionNonEscalationTests.test_thousands_may_not_become_10000",
            "invariants.test_quantity_precision_non_escalation.QuantityPrecisionNonEscalationTests.test_licensed_exact_numeral_is_not_escalation",
            "invariants.test_quantity_precision_non_escalation.QuantityPrecisionNonEscalationTests.test_life_year_product_from_thousands_is_escalation",
            "invariants.test_quantity_precision_non_escalation.QuantityPrecisionNonEscalationTests.test_spaced_range_headcount_is_escalation",
            "invariants.test_quantity_precision_non_escalation.QuantityPrecisionNonEscalationTests.test_challenge_refined_answer_rejects_life_year_invention",
            "invariants.test_quantity_precision_non_escalation.QuantityPrecisionHypothesisTests.test_oracle_agrees_with_detector",
        ),
        oracle_risk="independent",
        notes=(
            "Paired with QUANTITY_BEARING_CONSEQUENCE_PRESERVATION (retain the "
            "vague span) and AVERTED_ALTERNATIVE_HARM (inherit the same span). "
            "Kernel: relent.precision. Challenge answers that invent life-year "
            "products earn PRECISION_REJECTED rather than VERIFIED_REFINED. "
            "Hypothesis: strategies.quantity_precision."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="QUANTITY_FIELD_TYPING",
        layer="semantic_integrity",
        description=(
            "Recorded effect/party quantities must be magnitude spans "
            "(thousands, thousands of lives, decades, 500+); outcome or "
            "severity phrases (catastrophic loss of life, immediate survival) "
            "are rejected, and incomplete tails (decades of) prefer a "
            "source-licensed longer form or the bare collective head."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantity_field_typing.QuantityFieldTypingTests.test_catastrophic_loss_is_pseudo_outcome",
            "invariants.test_quantity_field_typing.QuantityFieldTypingTests.test_incomplete_decades_of_completes_from_source",
            "invariants.test_quantity_field_typing.QuantityFieldTypingTests.test_bare_thousands_is_retained",
            "invariants.test_quantity_field_typing.QuantityFieldTypingHypothesisTests.test_outcome_phrases_never_survive_sanitize",
            "invariants.test_quantity_field_typing.QuantityFieldTypingHypothesisTests.test_magnitude_phrases_remain_magnitude",
        ),
        oracle_risk="independent",
        notes=(
            "Kernel: relent.quantity_typing. Applied at parse_world_model, DET "
            "add_quantity, and Util magnitude selection. Complements "
            "QUANTITY_PRECISION_NON_ESCALATION (no invented sharpening) and "
            "QUANTITY_BEARING_CONSEQUENCE_PRESERVATION (retain licensed spans)."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="DERIVED_PROPOSITION_STATUS_CONSERVATION",
        layer="semantic_integrity",
        description=(
            "A licensed action-conditioned paraphrase of a world-established "
            "effect (no-purge / refrain / purge not executed ↔ A1→CERTAIN "
            "loss) must inherit that proposition's admitted status rather than "
            "minting a fresh HYPOTHETICAL atom; wrong-action or wrong-polarity "
            "wording must not transfer status."
        ),
        coverage="structural",
        tests=(
            "invariants.test_derived_proposition_status_conservation.DerivedPropositionStatusConservationTests.test_no_purge_paraphrase_binds_established_f1",
            "invariants.test_derived_proposition_status_conservation.DerivedPropositionStatusConservationTests.test_purge_not_executed_paraphrase_binds",
            "invariants.test_derived_proposition_status_conservation.DerivedPropositionStatusConservationTests.test_execute_paraphrase_does_not_bind_refrain_effect",
            "invariants.test_derived_proposition_status_conservation.DerivedPropositionParaphraseHypothesisTests.test_oracle_agrees_with_detector",
            "invariants.test_derived_proposition_status_conservation.DerivedPropositionParaphraseHypothesisTests.test_ledger_resolve_agrees_on_refrain_fixtures",
        ),
        oracle_risk="independent",
        notes=(
            "RelEnt paraphrase binder (PARAPHRASE_OF / EQUIVALENT_TO only). "
            "Kernel: relent.paraphrase. Wired through resolve_proposition / "
            "register_hypothesis. Hypothesis: "
            "strategies.derived_proposition_paraphrase."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="SEMANTIC_STATUS_CONSERVATION",
        layer="semantic_integrity",
        description=(
            "Umbrella for grounding status conservation: source-established "
            "stakes must not silently demote to hypothesis via fragment "
            "outcomes, omitted binary-contrast consequences, or dropped "
            "quantity spans."
        ),
        coverage="structural",
        tests=(
            "invariants.test_semantic_status_conservation.SemanticStatusConservationTests.test_incomplete_evening_world_fails_all_three_gates",
            "invariants.test_semantic_status_conservation.SemanticStatusConservationTests.test_repaired_structured_world_passes_family_oracles",
            "invariants.test_semantic_status_conservation.SemanticStatusConservationTests.test_grounding_lane_admits_repaired_purge_world",
            "invariants.test_semantic_status_conservation.SemanticStatusConservationTests.test_family_repair_cards_cover_all_issue_codes",
            "invariants.test_status_conservation_hypothesis.OutcomePredicateHypothesisTests.test_production_matches_declared_predicate_oracle",
            "invariants.test_status_conservation_hypothesis.BinaryStipulationHypothesisTests.test_production_matches_declared_stipulation_oracle",
            "invariants.test_status_conservation_hypothesis.QuantityConsequenceHypothesisTests.test_production_matches_declared_quantity_oracle",
            "invariants.test_status_conservation_hypothesis.QuantityConsequenceHypothesisTests.test_det_provenance_complete_repair_is_minimal_across_topologies",
        ),
        oracle_risk="independent",
        notes=(
            "Family umbrella. Members: OUTCOME_PREDICATE_COMPLETENESS, "
            "SOURCE_STIPULATED_OUTCOME_PRESERVATION, "
            "QUANTITY_BEARING_CONSEQUENCE_PRESERVATION. Sibling "
            "qualifier-preservation subtypes "
            "(LIKELIHOOD/TEMPORAL/SCOPE_QUALIFIER_PRESERVATION) share the "
            "same admit omit-check and DETERMINISTIC_LOCAL_PATCH path. "
            "SOURCE_INFORMATION_MONOTONICITY remains a deferred umbrella "
            "over status-conservation + qualifier preservation. Issue codes "
            "OUTCOME_PREDICATE_INCOMPLETE, SOURCE_STIPULATED_OUTCOME_MISSING, "
            "QUANTITY_BEARING_CONSEQUENCE_MISSING. Hypothesis generators in "
            "strategies.status_conservation. SymPy Util follows only after "
            "stipulated quantities survive admit."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="LIKELIHOOD_QUALIFIER_PRESERVATION",
        layer="semantic_integrity",
        description=(
            "A source likelihood hedge that binds to an effect's outcome must "
            "survive on that effect's likelihood_qualifiers (or hard-fail); "
            "silent drop is rejected."
        ),
        coverage="structural",
        tests=(
            "invariants.test_qualifier_preservation_hypothesis.QualifierPreservationHypothesisTests.test_production_matches_declared_qualifier_oracle",
            "invariants.test_qualifier_preservation_hypothesis.QualifierPreservationHypothesisTests.test_det_attaches_missing_qualifier_without_sibling_spray",
            "invariants.test_qualifier_preservation_hypothesis.QualifierPreservationRepairTests.test_deterministic_patch_attaches_missing_likelihood",
        ),
        oracle_risk="independent",
        notes=(
            "Subtype under deferred SOURCE_INFORMATION_MONOTONICITY. "
            "Production validate_world_model already omit-checks schema "
            "1.2/1.3 via effect_expected_qualifiers. Issue "
            "LIKELIHOOD_QUALIFIER_MISSING; DETERMINISTIC_LOCAL_PATCH when "
            "effect_id + span are known. Hypothesis placements: effect / "
            "omit / sibling mis-hang. Distinct from "
            "LIKELIHOOD_SPAN_NORMALIZATION (parser) and QUALIFIER_HEAD_BINDING."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="TEMPORAL_QUALIFIER_PRESERVATION",
        layer="semantic_integrity",
        description=(
            "A source temporal qualifier that binds to an effect's outcome "
            "must survive on temporal_qualifiers (or hard-fail); silent drop "
            "is rejected."
        ),
        coverage="structural",
        tests=(
            "invariants.test_qualifier_preservation_hypothesis.QualifierPreservationHypothesisTests.test_production_matches_declared_qualifier_oracle",
            "invariants.test_qualifier_preservation_hypothesis.QualifierPreservationHypothesisTests.test_det_attaches_missing_qualifier_without_sibling_spray",
        ),
        oracle_risk="independent",
        notes=(
            "Subtype under deferred SOURCE_INFORMATION_MONOTONICITY. "
            "Shares SOURCE omit-check and DETERMINISTIC_LOCAL_PATCH with "
            "likelihood/scope. Hypothesis placements: effect / omit / "
            "sibling. Distinct from TEMPORAL_ORDER_CONSISTENCY "
            "(before/after event order)."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="SCOPE_QUALIFIER_PRESERVATION",
        layer="semantic_integrity",
        description=(
            "A source scope qualifier that binds to an effect's outcome must "
            "survive on scope_qualifiers (or hard-fail); silent drop is "
            "rejected."
        ),
        coverage="structural",
        tests=(
            "invariants.test_qualifier_preservation_hypothesis.QualifierPreservationHypothesisTests.test_production_matches_declared_qualifier_oracle",
            "invariants.test_qualifier_preservation_hypothesis.QualifierPreservationHypothesisTests.test_det_attaches_missing_qualifier_without_sibling_spray",
        ),
        oracle_risk="independent",
        notes=(
            "Subtype under deferred SOURCE_INFORMATION_MONOTONICITY. "
            "Shares SOURCE omit-check and DETERMINISTIC_LOCAL_PATCH with "
            "likelihood/temporal. Hypothesis placements: effect / omit / "
            "sibling."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="ATTITUDE_FACTIVITY",
        layer="semantic_integrity",
        description=(
            "Factive attitudes license CERTAIN complements; non-factive "
            "attitudes must not force CERTAIN embedded propositions."
        ),
        coverage="structural",
        tests=(
            "invariants.test_attitude_factivity.AttitudeFactivityTests.test_structured_lane_know_licenses_certain_complement",
            "invariants.test_attitude_factivity.AttitudeFactivityTests.test_structured_lane_detects_noncertain_under_know",
            "invariants.test_attitude_factivity.AttitudeFactivityTests.test_grounding_lane_know_licenses_certain_complement",
            "invariants.test_attitude_factivity.AttitudeFactivityTests.test_grounding_lane_detects_noncertain_under_know",
            "invariants.test_attitude_factivity.AttitudeFactivityTests.test_grounding_lane_believe_allows_noncertain_complement",
            "invariants.test_discourse_fracas_hypothesis.AttitudeFactivityHypothesisTests.test_oracle_matches_declared_factivity",
        ),
        oracle_risk="independent",
        notes=(
            "FraCaS §9 Attitudes (know vs believe factivity). Hypothesis: "
            "strategies.attitude_factivity.attitude_factivity_cases "
            "(repair_stage epistemic_binding)."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=False,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="cataloged",
        source_type="fracas",
    ),
    Invariant(
        id="NO_UNGROUNDED_QUANTITY",
        layer="utilitarian",
        description=(
            "If the world does not admit a numeric magnitude, a specialist may "
            "not mint a threshold or convert a non-mortality outcome into "
            "deaths that uniquely ranks or that bounds a ranking."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantities.QuantityInvariantTests.test_unadmitted_magnitude_cannot_uniquely_rank",
            "UnadmittedMagnitudeRankingTests.test_minted_numeric_threshold_cannot_decide_ranking",
            "UnadmittedMagnitudeRankingTests.test_valuation_reason_cannot_convert_contamination_into_deaths",
            "UnadmittedMagnitudeRankingTests.test_decision_critical_hypothesis_converting_metric_is_bound",
            "UnadmittedMagnitudeRankingTests.test_ordinal_admitted_ranking_may_stand_without_minted_numbers",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_revoke is the oracle (MINTED or MORTALITY). "
            "ORDINAL and ADMITTED numbers must still uniquely rank. "
            "Enforcement remains Util-only."
        ),
    ),
    Invariant(
        id="AGENT_CAUSED_SETTLED_HARM_IS_DOING",
        layer="deontic",
        description=(
            "A DIRECT intervention that produces a settled actual welfare-adverse "
            "row on the protected party (DIRECT, or CAUSES/ACCELERATES/INCREASES "
            "from that DIRECT effect) licenses DOING_HARM. NEUTRAL infrastructure "
            "INTERVENTION still counts as that DIRECT source. ENABLES without "
            "CAUSES is allowing, not doing."
        ),
        coverage="structural",
        tests=(
            "invariants.test_causal.CausalInvariantTests.test_declared_causes_licenses_doing_and_enables_does_not",
            "DeonticGraphRelationTests.test_downstream_caused_harm_licenses_doing",
            "DeonticGraphRelationTests.test_enables_without_causes_is_allowing_not_doing",
            "DeonticGraphRelationTests.test_allowing_conflicts_when_intervention_causes_harm",
            "DeonticGraphRelationTests.test_near_certain_downstream_harm_licenses_doing",
            "DeonticGraphRelationTests.test_possible_downstream_harm_does_not_license_doing",
            "WorkspaceEngineTests.test_doing_harm_without_agent_caused_path_fails_calibration",
            "WorkspaceEngineTests.test_doing_harm_with_direct_adverse_survives_calibration",
            "WorkspaceEngineTests.test_allowing_harm_is_inconsistent_with_direct_adverse",
        ),
        oracle_risk="coupled",
        notes=(
            "Hypothesis case.a0_to_harm is the oracle. Calibration still goes "
            "through harm_relation_conflicts_with_graph. Do not infer CAUSES vs "
            "ENABLES by calling the production path-finder."
        ),
    ),
    Invariant(
        id="CAUSAL_PATH_REQUIRED",
        layer="deontic",
        description=(
            "INTENDED_AS_MEANS requires the protected party's actual ADVERSE "
            "burden to be an intermediate cause of another BENEFICIAL consequence "
            "of the same action. Sibling CAUSES from one INTERVENTION are "
            "FORESEEN_SIDE_EFFECT, not a means path."
        ),
        coverage="structural",
        tests=(
            "invariants.test_causal.CausalInvariantTests.test_declared_means_path_matches_intended_as_means",
            "invariants.test_graph_queries.GraphQueryAdapterTests.test_adapter_means_path_matches_declared_topology",
            "invariants.test_graph_queries.GraphQueryAdapterTests.test_adapter_agrees_with_existing_means_calibration",
            "DeonticGraphRelationTests.test_sibling_outcomes_are_not_a_means_path",
            "DeonticGraphRelationTests.test_intermediate_burden_licenses_means",
            "WorkspaceEngineTests.test_deontological_means_claim_requires_intended_as_means",
        ),
        oracle_risk="coupled",
        notes=(
            "Hypothesis case.harm_to_end is the oracle. Same coupling warning "
            "as AGENT_CAUSED_SETTLED_HARM_IS_DOING. graph_queries.is_intermediate_means "
            "is a NetworkX calculator on the world model, not a replacement for "
            "calibration and not the oracle."
        ),
    ),
    Invariant(
        id="POSTHUMOUS_DIRECTIVE_IS_ANALOGICAL",
        layer="deontic",
        description=(
            "A prior will about bodily remains is not coercion of a living agent's "
            "external freedom and cannot settle a perfect duty of antecedent autonomy."
        ),
        coverage="story",
        tests=(
            "PosthumousDirectiveKantianTests.test_posthumous_directive_is_not_living_agent_coercion",
            "PosthumousDirectiveKantianTests.test_using_organs_as_rescue_means_licenses_intended_as_means",
            "PosthumousDirectiveKantianTests.test_analogical_derivation_cannot_resolve_a_perfect_duty",
            "PosthumousDirectiveKantianTests.test_antecedent_autonomy_is_not_a_settled_kantian_derivation",
        ),
        oracle_risk="independent",
        notes=(
            "The oracle is the fixture's party label (deceased) plus the declared "
            "means path. Kantian analogues (posthumous rights, testament, Formula "
            "of Humanity) stay KANTIAN_ANALOGICAL and CONTESTED. Lying or "
            "falsification may still use a direct UNIVERSAL_LAW or PERFECT_DUTY "
            "derivation. Do not treat bare 'consent' as a posthumous coercion verb."
        ),
    ),
    Invariant(
        id="FOREGONE_NOT_SHARED_EFFECT",
        layer="projection",
        description=(
            "FOREGONE duals are exclusive missed opportunities, not outcomes "
            "shared by both actions."
        ),
        coverage="structural",
        tests=(
            "invariants.test_projection.ProjectionInvariantTests.test_foregone_duals_are_not_shared_outcomes",
            "EpistemicHypothesisBindingTests.test_foregone_duals_are_not_outcome_equivalence",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_equate is the oracle: FOREGONE duals must "
            "not share consequence labels; SHARED actual rows may. Do not infer "
            "overlays from the production projector."
        ),
    ),
    Invariant(
        id="HEALTH_OUTCOME_DIMENSION",
        layer="projection",
        description="HEALTH_OUTCOME rows project as BASIC_SECURITY rather than OTHER.",
        coverage="structural",
        tests=(
            "invariants.test_projection.HealthDimensionTests.test_health_outcome_projects_as_basic_security",
            "EpistemicHypothesisBindingTests.test_seeded_possible_injury_keeps_modality_and_health_dimension",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_be_basic_security is the oracle: declared "
            "HEALTH_OUTCOME maps to BASIC_SECURITY; WELFARE_OUTCOME with novel "
            "labels must not. Do not infer kind from the projector."
        ),
    ),
    Invariant(
        id="NEUTRAL_NONWELFARE_INTERVENTION",
        layer="world_admission",
        description=(
            "DIRECT INTERVENTION on a non-welfare-bearing party (facility, "
            "infrastructure, device) is NEUTRAL."
        ),
        coverage="structural",
        tests=(
            "NeutralDirectInterventionTests.test_beneficial_connect_on_a_facility_is_rejected",
            "NeutralDirectInterventionTests.test_neutral_connect_on_a_facility_is_allowed",
        ),
        oracle_risk="independent",
        notes="Admission loop. Frozen: do not add more admission validators to make this interesting.",
    ),
    Invariant(
        id="BOUND_QUANTITY_NORMALIZATION",
        layer="parser",
        description=(
            "A bound quantity phrase is recognized as one canonical span. "
            "The bound prefix survives, party and effect records match "
            "admission, normalization is idempotent, and the nested-span "
            "check accepts that single representation."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantities.BoundQuantityNormalizationTests.test_bound_phrase_is_one_canonical_span",
            "invariants.test_quantities.OrdinaryQuantitySpanTests.test_plain_quantity_is_one_canonical_span",
        ),
        oracle_risk="independent",
        notes=(
            "Parser contract, not an ethical invariant. Oracle is "
            "BoundQuantityCase.canonical. Do not add an admission validator. "
            "Widen the shared bound-prefix list when a generated hedge is unseen."
        ),
    ),
    Invariant(
        id="LIKELIHOOD_SPAN_NORMALIZATION",
        layer="parser",
        description=(
            "A source likelihood hedge is recognized as one canonical span. "
            "The long phrase survives, it binds only the modified outcome, "
            "an unhedged sibling stays CERTAIN without an invented if, and "
            "normalization is idempotent."
        ),
        coverage="structural",
        tests=(
            "invariants.test_likelihood.LikelihoodSpanNormalizationTests.test_hedge_is_one_canonical_span",
            "QualifierBindingTests.test_closed_class_untypes_certain_when_source_is_almost_certain",
            "QualifierBindingTests.test_duplicate_chance_identity_collapses",
        ),
        oracle_risk="independent",
        notes=(
            "Parser contract, not an ethical invariant. Oracle is "
            "LikelihoodSpanCase.canonical. Do not add an admission validator. "
            "A glued source '20%chance' keeps its literal span and offsets "
            "on QualifierSpan; canonical identity is '20% chance'. "
            "Duplicate chance phrases collapse by that identity. "
            "A chance hedge (almost certain, 20% chance) must not remain "
            "CERTAIN: closed-class untypes it to PROBABILISTIC or POSSIBLE. "
            "Occupying an at-risk situation stays CERTAIN "
            "(RISK_SITUATION_TYPING). Widen the shared likelihood-hedge list "
            "when a generated hedge is unseen. "
            "Distinct from LIKELIHOOD_IS_NOT_A_PARTY_QUANTITY and "
            "QUALIFIER_HEAD_BINDING."
        ),
    ),
    Invariant(
        id="QUANTITY_PARTY_UNIQUENESS",
        layer="parser",
        description=(
            "A stated quantity attaches only to the party whose local noun phrase "
            "uniquely matches it."
        ),
        coverage="structural",
        tests=(
            "invariants.test_quantities.QuantityPartyUniquenessTests.test_quantity_attaches_to_the_unique_party",
            "ExplicitQuantitySpanTests",
        ),
        oracle_risk="independent",
        notes=(
            "Parser contract, not a new admission validator. Oracle is "
            "QuantityPartyCase.expected: each span maps to one declared "
            "party_id; a generic crowd citing the same clause gets nothing. "
            "assigned_party_quantities is shared by parse fill/strip and the "
            "omit-check. Hyphenated cardinals belong in the quantity "
            "strategies, not a separate ID. Admission freeze still applies."
        ),
    ),
    Invariant(
        id="PERSON_ACT_NOT_PARENT_CROWD",
        layer="world_admission",
        description=(
            "A person's framing, execution, or death is not the causal parent of "
            "another party's health or welfare. The parent must be a process, "
            "facility, institution, infrastructure, or resource state."
        ),
        coverage="story",
        tests=("CausalChainCompletenessTests", "WorldModelCompletenessTests"),
        oracle_risk="independent",
        notes=(
            "Admission freeze. Magistrate-shaped. A crowd use/remain/occupy "
            "PHYSICAL_STATE is the mediated process and may hang off another "
            "party's RESOURCE_TRANSFER; trapped, death, and escape still may "
            "not. Do not reopen for Suite A."
        ),
    ),
    Invariant(
        id="CLOSED_ACTION_SET",
        layer="deliberation",
        description=(
            "A third action that temporally shares or splits an explicitly "
            "exclusive pair is not admitted as synthesis."
        ),
        coverage="structural",
        tests=(
            "invariants.test_closed_actions.ClosedActionSetTests.test_exclusive_pair_does_not_admit_share_split_synthesis",
            "BridgeTests.test_synthesis_cannot_add_third_action_to_explicitly_closed_choice",
            "BridgeTests.test_synthesis_withholds_temporal_sharing_that_contradicts_exclusive_choice",
            "WorkspaceEngineTests.test_nonviable_synthesis_does_not_activate_contingency",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_withhold is the oracle: CLOSED templates "
            "use exclusivity cues; OPEN omits them. Do not ask "
            "_scenario_closes_action_set for the expected answer."
        ),
    ),
    Invariant(
        id="ACTION_LABEL_PERMUTATION",
        layer="identity",
        description=(
            "Swapping presentation labels of the same physical actions must not "
            "change ordinal specialist direction."
        ),
        coverage="structural",
        tests=(
            "BridgeTests.test_permutation_evaluator_separates_ordinal_and_cardinal_invariance",
            "BridgeTests.test_canonical_action_order_is_invariant_to_input_permutation",
        ),
        oracle_risk="independent",
        notes="Existing morphism. Extend to party/outcome relabeling in Suite A.",
    ),
    Invariant(
        id="COMPOSITIONAL_HYPOTHESIS_INFLUENCE",
        layer="compositional",
        description=(
            "A decision-critical HYPOTHETICAL premise may shape investigation "
            "or a decision boundary, but if it is necessary for a unique "
            "recommendation that recommendation cannot be governing-eligible "
            "until the premise is established or independently supported."
        ),
        coverage="structural",
        tests=(
            "invariants.test_compositional.CompositionalHypothesisTests.test_hypothetical_necessary_for_unique_ranking_cannot_govern",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_be_governing_eligible is the oracle: a DC "
            "HYPOTHETICAL that is still necessary for a unique recommendation "
            "must not govern. Mention count 0, 1, or 4 must not change that. "
            "A closed-world ranking restored from admitted facts may still "
            "govern; the hypothesis remains a reversal boundary. Recurrence "
            "of an unsupported premise is one manifestation, not the whole rule."
        ),
    ),
    Invariant(
        id="COMPOSITIONAL_UNGROUNDED_THRESHOLD",
        layer="compositional",
        description=(
            "Counterfactual representation plus quantity extraction plus "
            "utilitarian scoring must not invent an expected-value threshold "
            "the world did not admit."
        ),
        coverage="structural",
        tests=(
            "invariants.test_compositional.CompositionalThresholdTests.test_foregone_plus_minted_threshold_cannot_uniquely_rank",
        ),
        oracle_risk="independent",
        notes=(
            "FOREGONE dual plus a minted percent in the decision rule. "
            "Production is only asked whether Util lost its unique ranking."
        ),
    ),
    Invariant(
        id="LIKELIHOOD_IS_NOT_A_PARTY_QUANTITY",
        layer="parser",
        description=(
            "A probability or chance phrase is a likelihood qualifier, not a "
            "party quantity. Percent chances must not occupy quantities[]."
        ),
        coverage="structural",
        tests=(
            "invariants.test_likelihood.LikelihoodIsNotAPartyQuantityTests.test_chance_percent_is_not_a_party_quantity",
        ),
        oracle_risk="independent",
        notes=(
            "Parser contract, not a new admission validator. Oracle is "
            "ChancePercentCase.binds_to_party: CHANCE percents are not "
            "quantity spans and do not assign to a party; COUNT percents and "
            "cardinals do. Score-time conversion of a chance percent into a "
            "death count remains NO_UNGROUNDED_QUANTITY."
        ),
    ),
    Invariant(
        id="QUALIFIER_HEAD_BINDING",
        layer="parser",
        description=(
            "A likelihood, scope, or temporal qualifier binds only to the "
            "effect whose outcome the qualifier grammatically modifies. A "
            "sibling process row citing the same clause must not omit-error "
            "or receive a closed-class copy."
        ),
        coverage="structural",
        tests=(
            "invariants.test_qualifiers.QualifierHeadBindingTests.test_qualifier_binds_only_the_modified_head",
            "QualifierBindingTests.test_percent_chance_that_svo_binds_to_the_verb_not_the_result",
        ),
        oracle_risk="independent",
        notes=(
            "Parser contract, not a new admission validator. Oracle is "
            "QualifierHeadCase.owner_id. effect_expected_qualifiers is the "
            "shared binder: parse fill/strip, omit-check, and the hedged-source "
            "gate all call it. Binding uses the modified head word, not a "
            "shared facility or crowd noun: attributive hedges take the first "
            "content word; chance/that/of/to complements take the clause "
            "predicate so a verb-only outcome still matches. A that/the SVO "
            "complement binds to the verb, not the object or a following "
            "result participle. Graph, ledger, and compact copy the bound "
            "tuples. Distinct from LIKELIHOOD_SPAN_NORMALIZATION."
        ),
    ),
    Invariant(
        id="RISK_SITUATION_TYPING",
        layer="parser",
        description=(
            "An at-risk situation phrase binds to the occupying party or "
            "exposure outcome as a CERTAIN magnitude qualifier. It is a "
            "likelihood hedge only when it modifies a distinct harm event."
        ),
        coverage="structural",
        tests=(
            "invariants.test_risk_situation.RiskSituationTypingTests.test_at_risk_span_types_situation_or_harm",
        ),
        oracle_risk="independent",
        notes=(
            "Parser contract, not a new admission validator. Oracle is "
            "RiskSituationCase.expects_certain_exposure. Do not type "
            "AT_RISK/EXPOSED as PROBABILISTIC merely because 'at moderate "
            "risk' is present. Inverse: a chance hedge on a distinct harm "
            "must not stay CERTAIN (LIKELIHOOD_SPAN_NORMALIZATION). Distinct "
            "from LIKELIHOOD_SPAN_NORMALIZATION."
        ),
    ),
    Invariant(
        id="ACTION_MEDIATED_VS_EXOGENOUS",
        layer="world_admission",
        description=(
            "A human outcome must reach a DIRECT act through an "
            "action-mediated process. Exogeneity is the source's causal "
            "structure, not a likelihood hedge: an action-caused stochastic "
            "process may or must have an inbound intervention path; an "
            "independent background event must not; an ambiguous association "
            "must not invent that link. Conjunction is a condition whose "
            "event_effect_id references the independent event, not a second "
            "ordinary causal parent."
        ),
        coverage="structural",
        tests=(
            "invariants.test_topology.ActionMediatedVersusExogenousTests.test_declared_topology_matches_admission",
            "CopulaChanceWorldTests.test_schema13_conjunctive_copula_outcomes_admit",
            "CopulaChanceWorldTests.test_independent_event_still_cannot_parent_gated_harm",
            "CopulaChanceWorldTests.test_live_floodgate_parented_chance_compiles_and_admits",
        ),
        oracle_risk="independent",
        notes=(
            "Refines existing ancestry; does not loosen it or the "
            "tautological-if gate. Conjunction uses condition.event_effect_id "
            "on the mediated path, not a second causal parent. Oracle is "
            "TopologyCase.should_admit. Foreign-parent repair names a "
            "plausible unused source facility, not an arbitrary existing "
            "process and not a claimed unique intermediary. Admission freeze "
            "still holds for unrelated new validators."
        ),
    ),
    Invariant(
        id="ACTION_ROLE_STABILITY",
        layer="world_admission",
        description=(
            "Repair must keep human recipients of a transfer or assignment, "
            "treat a transferred resource as a party not a recipient, recast "
            "later conduct as DOWNSTREAM rather than adding that party as a "
            "recipient, and retain already-valid FOREGONE overlays."
        ),
        coverage="story",
        tests=(
            "NeutralDirectInterventionTests.test_direct_intervention_on_a_non_recipient_crowd_does_not_add_them",
            "NeutralDirectInterventionTests.test_assignment_recipient_then_downstream_conduct_admits",
            "WorldModelCompletenessTests.test_transferred_resource_is_not_a_recipient",
            "WorldModelValidationTests.test_repair_restores_dropped_counterfactual_overlays",
        ),
        oracle_risk="independent",
        notes=(
            "Refines existing recipient and DIRECT completeness; not a new "
            "gate class and does not loosen ancestry or tautological-if. "
            "Assignment may name the assigned party as recipient of that "
            "DIRECT act; subsequent assist or exposure is DOWNSTREAM. "
            "Admission freeze still holds for unrelated new validators."
        ),
    ),
    Invariant(
        id="CHANCE_IS_NOT_AN_OUTCOME",
        layer="projection",
        description=(
            "A POSSIBLE, PROBABILISTIC, or STIPULATED_CONDITIONAL effect is "
            "not an obtained outcome. BENEFICIAL must not become IMPROVES; "
            "ADVERSE must not become WORSENS. A near-certain conditional "
            "remains conditional. Chance may occupy at_risk, "
            "conditionally_benefited, expected-value reasoning, uncertainty, "
            "or reversal boundaries."
        ),
        coverage="structural",
        tests=(
            "invariants.test_consumption.ChanceOutcomeTests.test_hedged_chance_is_not_an_obtained_outcome",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_count_as_obtained is the oracle: CERTAIN "
            "BENEFICIAL or ADVERSE may count; POSSIBLE, PROBABILISTIC, and "
            "STIPULATED_CONDITIONAL of either polarity must not. Compact roles "
            "send chance-harm and stipulated conditionals to at_risk. Do not "
            "ask project_grounded_action_effects for the expected direction. "
            "CERTAIN rows that only encode an averted unsettled harm are "
            "AVERTED_RISK_IS_NOT_OBTAINED_BENEFIT. Distinct from "
            "RISK_SITUATION_TYPING, which keeps occupying at-risk CERTAIN in "
            "the parser; compact still routes that situation to at_risk."
        ),
    ),
    Invariant(
        id="AVERTED_RISK_IS_NOT_OBTAINED_BENEFIT",
        layer="projection",
        description=(
            "If X is only a POSSIBLE or PROBABILISTIC adverse under the "
            "alternative, certainly preventing X may stand as PREVENTED(X) "
            "CERTAIN. It must not become obtained welfare: not IMPROVES, not "
            "a compact beneficiary, not util CERTAIN BENEFIT, and not an "
            "established welfare gain in public factual listing."
        ),
        coverage="structural",
        tests=(
            "invariants.test_consumption.AvertedRiskTests.test_averted_risk_compact_party_is_not_a_certain_beneficiary",
            "invariants.test_consumption.AvertedRiskTests.test_averted_risk_util_prevention_is_not_certain_benefit",
            "invariants.test_consumption.AvertedRiskTests.test_averted_risk_presentation_is_not_established_welfare_gain",
            "CompactActionRoleTests.test_certain_intermediate_protection_conditionally_benefits_opposed_crowd",
            "CompactActionRoleTests.test_unrelated_intermediate_protection_does_not_project_a_crowd",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_count_as_obtained is the oracle: DIRECT "
            "CERTAIN welfare on the same party may count; AVERTED risk of an "
            "opposed chance-harm must not. Compact, util accounting, and "
            "presentation are separate surfaces of one ID. Distinct from "
            "CHANCE_IS_NOT_AN_OUTCOME and FOREGONE_IS_NOT_OBTAINED. Do not "
            "ask production to classify averted risk. Admission freeze holds. "
            "Distinct from AVERTED_ALTERNATIVE_HARM (CERTAIN opposed harm)."
        ),
    ),
    Invariant(
        id="AVERTED_ALTERNATIVE_HARM",
        layer="semantic_integrity",
        description=(
            "When mutually exclusive alternatives pair CERTAIN survival with "
            "CERTAIN adverse harm that carries a quantity, the magnitude may "
            "be inherited only through an AVERTED_ALTERNATIVE_HARM derivation "
            "bound by a counterfactual edge and source_effect_ids — never by "
            "copying the span onto the survival row as DIRECT_COPY source support."
        ),
        coverage="structural",
        tests=(
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmHypothesisTests.test_oracle_rejects_silent_source_copy",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmHypothesisTests.test_derived_row_licenses_quantity_via_counterfactual",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmHypothesisTests.test_compile_mints_averted_alternative_harm",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmCompileTests.test_chance_gated_world_mints_and_keeps_inherited_quantity",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmCompileTests.test_world_model_from_dict_admits_recompiled_averted_overlays",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmGroundingTests.test_grounding_lane_mints_averted_quantity_without_silent_survival_copy",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmGroundingTests.test_grounding_lane_strips_silent_survival_copy",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmGroundingTests.test_epistemic_marks_averted_claim_established",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis: strategies.averted_alternative_harm. "
            "Compiler compile_averted_alternative_harm_overlays mints the "
            "derived row; compile_grounded_quantities licenses inherited "
            "spans. Completeness does not demand a CAUSES process parent for "
            "CF-licensed AV* rows; provenance is inherited from opposed harm. "
            "world_model_from_dict refreshes admission so overlays are not "
            "WITHHELD. Grounding seed averted_alternative_harm_grounding.yaml "
            "closes the E1/E7 loop. Util scores the derived row instead of "
            "double-counting survival; epistemic treats it as established "
            "certain-alternative aversion, not unsettled averted risk."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=True,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="ADMITTED_PROPOSITION_STATUS",
        layer="epistemic",
        description=(
            "An admitted world proposition remains ESTABLISHED or STIPULATED "
            "in every downstream action comparison. Hypothetical is reserved "
            "for agent-introduced propositions absent from the admitted world."
        ),
        coverage="story",
        tests=(
            "AdmittedWorldGroundingTests.test_admitted_alternative_action_facts_rebind_as_stipulated",
        ),
        oracle_risk="independent",
        notes=(
            "Story fixture. Alternative-action chance and gated rows stay "
            "admitted facts about that alternative. Do not reopen world admission."
        ),
    ),
    Invariant(
        id="OUTCOME_TYPE_PRESERVATION",
        layer="epistemic",
        description=(
            "A derived calculation must preserve the admitted outcome type: "
            "expected trapped is not expected deaths, and a qualitative "
            "near-certain hedge cannot mint an exact expected-death total."
        ),
        coverage="story",
        tests=(
            "AdmittedWorldGroundingTests.test_expected_trapped_is_not_expected_deaths",
            "UnadmittedMagnitudeRankingTests.test_valuation_reason_cannot_convert_contamination_into_deaths",
        ),
        oracle_risk="independent",
        notes=(
            "Story plus existing mortality-conversion test. Ranking revocation "
            "is NO_UNGROUNDED_QUANTITY; public copy must not treat the converted "
            "claim as an unestablished world fact."
        ),
    ),
    Invariant(
        id="LEDGER_ACTION_SCOPE",
        layer="framework_ledger",
        description=(
            "A verified resolved framework ledger must not accept a cross-action "
            "transfer of an effect. An effect that obtains only under one action "
            "cannot be treated as holding equally under the rival plan."
        ),
        coverage="story",
        tests=(
            "WorkspaceEngineTests.test_position_coverage_rejects_cross_action_effect_transfer",
        ),
        oracle_risk="independent",
        notes=(
            "POSITION_COVERAGE verification checks named parties and "
            "action-scoped world effects. Distinct from FOREGONE overlays."
        ),
    ),
    Invariant(
        id="PRESENTATION_PRESERVES_MODALITY",
        layer="presentation",
        description=(
            "Public factual-status copy must not list an unsettled world row "
            "as a bare established fact. Ledger status ESTABLISHED on a "
            "PROBABILISTIC or POSSIBLE atom is matching, not settlement."
        ),
        coverage="structural",
        tests=(
            "invariants.test_consumption.PresentationModalityTests.test_unsettled_row_is_not_listed_as_bare_established_fact",
            "AdmittedWorldGroundingTests.test_factual_status_lists_each_actions_obtained_harm",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_list_as_established_fact is the oracle: "
            "CERTAIN may list; POSSIBLE and PROBABILISTIC must not appear "
            "under Established or derived. Obtained welfare and harm of every "
            "action must appear; a short window that keeps one action's spare "
            "and the other's plant-protect, and drops the latter's certain "
            "farm harm, is incomplete. FOREGONE overlays are "
            "FOREGONE_IS_NOT_OBTAINED even when CERTAIN. Likelihood, scope, "
            "and temporal children are annotation atoms, not independent "
            "world effects; they are not a named invariant until that "
            "contract is decided."
        ),
    ),
    Invariant(
        id="VOTE_REPORT_SPLIT",
        layer="presentation",
        description=(
            "Public judgment must not report an ABSTAIN framework as favoring "
            "the recommendation merely because its directional lean or "
            "recommended_action matches the plurality. Admitted vote "
            "(FULL / ATTENUATED / ABSTAIN) and directional position stay "
            "separate columns and separate prose."
        ),
        coverage="structural",
        tests=(
            "invariants.test_vote_report_split.VoteReportSplitTests.test_abstain_never_reads_as_favors_despite_directional_lean",
            "invariants.test_vote_report_split.VoteReportSplitTests.test_full_vote_still_reported_as_favor",
        ),
        oracle_risk="independent",
        notes=(
            "Intro support clauses use FULL only; Why-favors uses admitted "
            "vote; ABSTAIN leans surface under 'Directional leans without "
            "admitted vote'. Deliberation map headers: Directional position "
            "| Admitted vote."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=False,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="OPERATOR_SCOPE_CONSERVATION",
        layer="semantic_integrity",
        description=(
            "A licensed Conditional or ExceptionRule must conserve operator "
            "scope: A∧¬B→C does not license C under A∧B; A→probably C forbids "
            "probably A→C and bare A→C; A→C unless B licenses C only when the "
            "exception is false."
        ),
        coverage="structural",
        tests=(
            "invariants.test_operator_scope_conservation.OperatorScopeConservationTests.test_a_and_not_b_licenses_only_matching_world",
            "invariants.test_operator_scope_conservation.OperatorScopeConservationTests.test_modal_drop_and_antecedent_move_are_rejected",
            "invariants.test_operator_scope_conservation.OperatorScopeConservationTests.test_unless_exception_blocks_consequent",
            "invariants.test_operator_scope_conservation.OperatorScopeConservationTests.test_parliament_adapter_projects_condition_gates",
            "invariants.test_operator_scope_conservation.OperatorScopeHypothesisTests.test_oracle_agrees_with_detector",
        ),
        oracle_risk="independent",
        notes=(
            "RelEnt kernel: relent.operators + relent.scope. Parliament "
            "adapter project_operator_to_effect_gates maps into "
            "condition_ids/join/modality without owning world state. "
            "Hypothesis: strategies.operator_scope. Temporal feedback "
            "deferred."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=False,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="RELATIONAL_ENTAILMENT",
        layer="semantic_integrity",
        description=(
            "Licensed world edges must respect typed relation algebra: "
            "SAME_ENTITY_AS closes under reflexivity/symmetry/transitivity; "
            "ALTERNATIVE_OF is symmetric but not identity; CAUSES does not "
            "freely transit; quantity/status/harm_under_action transfer only "
            "when RelationSpec allows."
        ),
        coverage="structural",
        tests=(
            "invariants.test_relational_entailment.RelationSpecTableTests.test_every_tag_has_a_spec",
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_identity_closure_a_b_c",
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_alternative_symmetry",
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_causes_does_not_auto_transit",
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_alternative_plus_identity_is_forbidden",
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_quantity_transfer_policies",
            "invariants.test_relational_entailment.RelationalAlgebraHypothesisTests.test_oracle_agrees_with_algebra",
        ),
        oracle_risk="independent",
        notes=(
            "Umbrella for relational closure. Members: RELATIONAL_SYMMETRY, "
            "RELATIONAL_FUNCTION_TRANSFER, RELATIONAL_NON_TRANSFER (AV slice), "
            "RELATIONAL_IDENTITY_CLOSURE (SAME_ENTITY_AS), "
            "RELATIONAL_DIRECTIONALITY / RELATIONAL_TRANSITIVITY "
            "(CAUSES/BEFORE). Kernel: relent.relations RelationSpec + "
            "relent.algebra. Adapter: relent_adapt "
            "project_alternative_of_edges / project_same_entity_edges / "
            "project_causes_edges / project_before_edges. Hypothesis: "
            "strategies.relational_algebra + strategies.identity_closure + "
            "strategies.directionality."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="RELATIONAL_SYMMETRY",
        layer="semantic_integrity",
        description=(
            "ALTERNATIVE_OF projected from counterfactual pairs or opposed "
            "CERTAIN welfare must close under symmetry: ALTERNATIVE_OF(A0,A1) "
            "entails ALTERNATIVE_OF(A1,A0)."
        ),
        coverage="structural",
        tests=(
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_alternative_symmetry",
            "invariants.test_relational_entailment.AvertedRelationalAdapterTests.test_alternative_of_projects_symmetric_closure",
        ),
        oracle_risk="independent",
        notes=(
            "Member of RELATIONAL_ENTAILMENT. Adapter "
            "project_alternative_of_edges; closure via relent.algebra."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="RELATIONAL_FUNCTION_TRANSFER",
        layer="semantic_integrity",
        description=(
            "Quantity may move across ALTERNATIVE_OF only via_derivation_only: "
            "AVERTED_ALTERNATIVE_HARM rows with DERIVED_FROM marks may inherit "
            "CERTAIN opposed-harm spans; unmarked transfer is rejected."
        ),
        coverage="structural",
        tests=(
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_quantity_transfer_policies",
            "invariants.test_relational_entailment.AvertedRelationalAdapterTests.test_derived_averted_quantity_transfer_is_licensed",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmHypothesisTests.test_derived_row_licenses_quantity_via_counterfactual",
        ),
        oracle_risk="independent",
        notes=(
            "Member of RELATIONAL_ENTAILMENT. compile_averted_alternative_harm_overlays "
            "consults function_transfer_errors before mint; "
            "averted_alternative_relational_errors post-checks."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=True,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="RELATIONAL_NON_TRANSFER",
        layer="semantic_integrity",
        description=(
            "ALTERNATIVE_OF must not imply effect equality or silent DIRECT_COPY "
            "of opposed CERTAIN harm quantities onto a survival row; "
            "harm_under_action must not cross action-scoped alternatives."
        ),
        coverage="structural",
        tests=(
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_alternative_plus_identity_is_forbidden",
            "invariants.test_relational_entailment.AvertedRelationalAdapterTests.test_silent_copy_is_relational_non_transfer",
            "invariants.test_averted_alternative_harm_hypothesis.AvertedAlternativeHarmHypothesisTests.test_oracle_rejects_silent_source_copy",
            "invariants.test_relational_entailment.IdentityClosureAdapterTests.test_harm_cross_branch_is_non_transfer",
        ),
        oracle_risk="independent",
        notes=(
            "Member of RELATIONAL_ENTAILMENT. Distinct from AVERTED_ALTERNATIVE_HARM "
            "(compiler + inheritance) by stating the algebra non-transfer rule "
            "explicitly; production validator averted_alternative_relational_errors "
            "+ identity_relational_errors (action-scoped harm via SAME_ENTITY_AS)."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="RELATIONAL_IDENTITY_CLOSURE",
        layer="semantic_integrity",
        description=(
            "SAME_ENTITY_AS / EQUIVALENT_TO form an identity family: licensed "
            "party-alias edges close under reflexivity, symmetry, and "
            "transitivity. Quantity may transfer along identity; "
            "harm_under_action must not cross distinct action branches even "
            "when parties are identity-linked."
        ),
        coverage="structural",
        tests=(
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_identity_closure_a_b_c",
            "invariants.test_relational_entailment.IdentityClosureAdapterTests.test_label_alias_projects_symmetric_identity",
            "invariants.test_relational_entailment.IdentityClosureAdapterTests.test_chain_closes_a_to_c",
            "invariants.test_relational_entailment.IdentityClosureAdapterTests.test_quantity_may_transfer_along_identity",
            "invariants.test_relational_entailment.IdentityClosureAdapterTests.test_harm_cross_branch_is_non_transfer",
            "invariants.test_relational_entailment.IdentityClosureHypothesisTests.test_oracle_agrees_with_adapter",
        ),
        oracle_risk="independent",
        notes=(
            "Member of RELATIONAL_ENTAILMENT. Adapter project_same_entity_edges "
            "(licensed edges + shared party labels) and identity_relational_errors. "
            "Hypothesis: strategies.identity_closure. Bridges anaphor party_id "
            "data without importing discourse parsers into relent/."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="RELATIONAL_DIRECTIONALITY",
        layer="semantic_integrity",
        description=(
            "CAUSES, BEFORE, and DERIVED_FROM are directed: closure must not "
            "invent reverse edges. BEFORE / DERIVED_FROM must not be licensed "
            "in both directions. CAUSES feedback cycles stay host-owned; only "
            "invented reverses are rejected."
        ),
        coverage="structural",
        tests=(
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_causes_does_not_auto_transit",
            "invariants.test_relational_entailment.DirectionalityAdapterTests.test_causes_projects_without_reverse",
            "invariants.test_relational_entailment.DirectionalityAdapterTests.test_before_antisymmetric_base_is_rejected",
            "invariants.test_relational_entailment.DirectionalityHypothesisTests.test_oracle_agrees_with_adapter",
        ),
        oracle_risk="independent",
        notes=(
            "Member of RELATIONAL_ENTAILMENT. Kernel directionality_errors; "
            "adapter project_causes_edges / project_before_edges / "
            "directionality_relational_errors. Hypothesis: "
            "strategies.directionality."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="RELATIONAL_TRANSITIVITY",
        layer="semantic_integrity",
        description=(
            "Transitivity follows RelationSpec: BEFORE closes under "
            "transitivity; CAUSES and DERIVED_FROM are restricted and must "
            "not auto-derive A→C skip-links. Hosts may still license each hop."
        ),
        coverage="structural",
        tests=(
            "invariants.test_relational_entailment.RelationalAlgebraTests.test_causes_does_not_auto_transit",
            "invariants.test_relational_entailment.DirectionalityAdapterTests.test_before_chain_transits",
            "invariants.test_relational_entailment.DirectionalityAdapterTests.test_causes_chain_does_not_skip",
        ),
        oracle_risk="independent",
        notes=(
            "Member of RELATIONAL_ENTAILMENT. Complements RELATIONAL_DIRECTIONALITY. "
            "Hypothesis modes before_transits / causes_no_free_transitivity."
        ),
        production="pass",
        integrity_layers=IntegrityLayers(
            cataloged=True,
            seed_case=False,
            structured_property=True,
            grounding_property=False,
            production_validator=True,
            metamorphic_test=True,
            end_to_end=False,
        ),
        enforcement="enforced",
        source_type="parliament_extension",
    ),
    Invariant(
        id="FOREGONE_IS_NOT_OBTAINED",
        layer="presentation",
        description=(
            "A CERTAIN FOREGONE overlay is a counterfactual relation, not an "
            "established event that occurred in the chosen world."
        ),
        coverage="structural",
        tests=(
            "invariants.test_consumption.ForegoneObtainedTests.test_foregone_overlay_is_not_listed_as_an_obtained_event",
            "CompactActionRoleTests.test_facility_welfare_swap_compiles_foregone_overlays",
            "CompactActionRoleTests.test_opposed_intermediate_supply_compiles_foregone_overlays",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_list_as_established_fact is the oracle: "
            "ACTUAL CERTAIN rows may list; FOREGONE rows must not. Distinct "
            "from FOREGONE_NOT_SHARED_EFFECT and FOREGONE_OVERLAY_UNIQUENESS."
        ),
    ),
    Invariant(
        id="MODALITY_BLIND_COST_SHAPE",
        layer="projection",
        description=(
            "Problem-shape must not treat extra unsettled ADVERSE rows as "
            "more certain cost structure than a smaller set of CERTAIN harms. "
            "Polarity counts are not modality-blind."
        ),
        coverage="structural",
        tests=(
            "invariants.test_consumption.ModalityBlindCostTests.test_polarity_counts_are_not_modality_blind",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_treat_as_more_cost is the oracle: extra "
            "CERTAIN ADVERSE rows may shape cost; extra PROBABILISTIC rows "
            "must not."
        ),
    ),
    Invariant(
        id="FOREGONE_OVERLAY_UNIQUENESS",
        layer="world_admission",
        description=(
            "An action may carry at most one FOREGONE overlay per opposed "
            "actual row on the same party and outcome family."
        ),
        coverage="none",
        tests=(),
        oracle_risk="independent",
        notes=(
            "Habitat A0 E4 DIE_DECOMPRESSION and E6 CERTAIN_DEATH both dual "
            "A1 technician death. Naming only; admission freeze still holds."
        ),
    ),
    Invariant(
        id="NO_PREFERENCE_NO_SUPPORT",
        layer="deliberation",
        description=(
            "Equal action scores and no recommended action cannot uniquely "
            "SUPPORT or govern. The framework may be tied or unresolved, "
            "but not ordinary SUPPORTS with governing eligibility."
        ),
        coverage="structural",
        tests=(
            "invariants.test_authority.PreferenceSupportTests.test_tied_scores_without_recommendation_cannot_uniquely_support",
        ),
        oracle_risk="independent",
        notes=(
            "Hypothesis case.should_uniquely_support is the oracle: PREFERS "
            "may SUPPORT; TIED with recommendation NONE must not. This is "
            "decision state versus authority state, not a quantity rule."
        ),
    ),
    Invariant(
        id="RESIDUAL_RANKING_IS_NOT_NONCOMPARISON",
        layer="deliberation",
        description=(
            "A framework that assessed every live action on a committed "
            "ledger may not have its directional vote zeroed merely because "
            "it reports residual ranking tension. That is ATTENUATED, not "
            "ABSTAIN. A ledger that skipped an action still abstains."
        ),
        coverage="story",
        tests=(
            "FrameworkVoteIntegrityTests.test_incomplete_comparison_with_full_ledger_is_attenuated",
            "FrameworkVoteIntegrityTests.test_incomplete_comparison_without_ledger_row_still_abstains",
            "UtilitarianSettledWelfareRankingTests.test_hypothesis_does_not_veto_admitted_five_life_comparison",
            "UtilitarianSettledWelfareRankingTests.test_incommensurable_remainder_attenuates_without_erasing_the_five_saves",
            "BridgeTests.test_utilitarian_may_cite_admitted_grounded_effect_as_world_proposition",
            "BridgeTests.test_utilitarian_unknown_world_proposition_still_fails_validation",
        ),
        oracle_risk="independent",
        notes=(
            "cc=false after a full ledger comparison is investigative residue. "
            "For Util, that includes incommensurable leftovers and open "
            "hypothesis reversal boundaries after an admitted same-unit lean. "
            "Citing PROP:WORLD of an admitted grounded effect is not an "
            "unknown proposition. More cycles do not complete a ranking "
            "rule that is already a decision boundary."
        ),
    ),
    Invariant(
        id="SOURCE_SUPPORTED_ACTION_PROPOSITIONS",
        layer="action_admission",
        description=(
            "Every proposition added to a canonical action must be attested "
            "in the source or by explicit user-authored provenance. Invented "
            "outcomes, quantities, and participants are rejected even when "
            "the action still preserves required claims. Confirmation of "
            "model prose is not provenance."
        ),
        coverage="story",
        tests=(
            "CanonicalActionCompletenessTests.test_invented_lethal_count_is_unsupported",
            "CanonicalActionCompletenessTests.test_invented_total_departure_is_unsupported",
            "CanonicalActionCompletenessTests.test_invented_quantity_and_participant_are_unsupported",
            "CanonicalActionCompletenessTests.test_user_authored_provenance_attests_new_claims",
        ),
        oracle_risk="independent",
        notes=(
            "Inverse of missing_decision_critical_claims. Does not loosen "
            "source-preservation. Domain-general fixtures; not live-dilemma "
            "nouns in Hypothesis."
        ),
    ),
    Invariant(
        id="CHANCE_PREDICATE_BINDING",
        layer="parser",
        description=(
            "A percent chance binds to its source predicate and polarity. "
            "Blockage and failure risks are not generic survival_chance. "
            "An unbound elliptical percentage is not survival."
        ),
        coverage="story",
        tests=(
            "ScenarioFactExtractionTests.test_process_chance_is_not_survival_chance",
            "ScenarioFactExtractionTests.test_elliptical_survival_chance_still_binds",
            "ScenarioFactExtractionTests.test_unbound_elliptical_chance_is_not_survival",
            "ScenarioFactExtractionTests.test_elliptical_survival_does_not_cross_sentences",
        ),
        oracle_risk="independent",
        notes=(
            "Fact-table contract for contradiction checks. World-model "
            "likelihood_qualifiers remain the bound chance on effects. "
            "Distinct from QUALIFIER_HEAD_BINDING."
        ),
    ),
    Invariant(
        id="EVENT_REFERENCED_CONDITION",
        layer="world_admission",
        description=(
            "An independent stochastic event gates an action-mediated path "
            "through condition.event_effect_id. It is not a second causal "
            "parent, and free-text may not stand in for that event."
        ),
        coverage="structural",
        tests=(
            "invariants.test_topology.ActionMediatedVersusExogenousTests.test_declared_topology_matches_admission",
            "IndependentConditionTests.test_event_referenced_gate_admits",
            "IndependentConditionTests.test_free_text_restating_independent_event_is_rejected",
            "IndependentConditionTests.test_welfare_event_reference_is_rejected",
            "IndependentConditionTests.test_multiple_conditions_require_join",
            "IndependentConditionTests.test_gate_percent_must_not_copy_onto_gated_outcome",
        ),
        oracle_risk="independent",
        notes=(
            "Refines existing exogenous and tautological-if gates. Does not "
            "loosen ancestry. Probability stays on the referenced event. "
            "Oracle for generated worlds is TopologyCase.should_admit."
        ),
    ),
    Invariant(
        id="CONDITIONAL_ROLE_ROUTING",
        layer="projection",
        description=(
            "STIPULATED_CONDITIONAL compact roles are at_risk or "
            "conditionally_benefited, never obtained harm or benefit, even "
            "when the gated outcome is near-certain."
        ),
        coverage="story",
        tests=(
            "CompactActionRoleTests.test_near_certain_stipulated_conditional_is_at_risk",
            "CompactActionRoleTests.test_stipulated_crowd_protection_is_conditionally_benefited",
            "CompactActionRoleTests.test_certain_intermediate_protection_conditionally_benefits_opposed_crowd",
            "CompactActionRoleTests.test_facility_with_welfare_row_counts_physical_harm",
        ),
        oracle_risk="independent",
        notes=(
            "Semantic status is read before confidence. Distinct from "
            "CHANCE_IS_NOT_AN_OUTCOME, which is the Hypothesis graph oracle. "
            "Do not add a new compact slot."
        ),
    ),
    Invariant(
        id="RISK_STATE_IS_NOT_OBTAINED_HARM",
        layer="projection",
        description=(
            "Occupying an at-risk situation is compact at_risk even when the "
            "parser types that situation CERTAIN. Only obtained adverse "
            "welfare outcomes occupy harmed."
        ),
        coverage="story",
        tests=(
            "CompactActionRoleTests.test_certain_at_moderate_risk_is_at_risk_not_harmed",
            "CompactActionRoleTests.test_assignment_quantity_labels_risk_row_without_copy",
            "CompactActionRoleTests.test_underscored_stay_outcome_still_owns_the_subset",
            "CompactActionRoleTests.test_personnel_group_cardinal_is_a_quantity",
            "CompactActionRoleTests.test_assignment_subgroup_is_stripped_from_party_total",
        ),
        oracle_risk="independent",
        notes=(
            "Parser RISK_SITUATION_TYPING stays CERTAIN. This is consumption. "
            "Crowd use/remain process states are NEUTRAL and do not occupy "
            "compact roles. Party totals stay on the party; assigned subsets "
            "are compact labels."
        ),
    ),
    Invariant(
        id="DOES_NOT_INCREASE_IS_BASELINE_CONSTRAINT",
        layer="world_admission",
        description=(
            "DOES_NOT_INCREASE identifies this action's DIRECT act, the "
            "affected independent risk event, and that event's retained "
            "baseline chance. It does not mean the risk cannot occur."
        ),
        coverage="story",
        tests=(
            "DoesNotIncreaseTests.test_does_not_increase_keeps_baseline_chance",
            "DoesNotIncreaseTests.test_does_not_increase_cannot_target_welfare",
            "DoesNotIncreaseTests.test_does_not_increase_cannot_flatten_risk_to_certain",
            "DoesNotIncreaseTests.test_deverbal_noun_chance_stays_on_risk_event",
        ),
        oracle_risk="independent",
        notes=(
            "Negative constraint, not a causal parent. Distinct from PREVENTS."
        ),
    ),
    Invariant(
        id="REPAIR_PRESERVES_VALID_FACTS",
        layer="world_admission",
        description=(
            "A repair cannot resolve an error by deleting previously valid "
            "probabilities, quantities, risk effects, conditions, causal "
            "links, parties, counterfactual overlays, or source citations "
            "unless the listed errors name them."
        ),
        coverage="story",
        tests=(
            "RepairDeltaTests.test_repair_cannot_drop_unimplicated_probability",
            "RepairDeltaTests.test_repair_cannot_empty_unimplicated_causal_links",
            "RepairDeltaTests.test_repair_cannot_drop_cited_party",
        ),
        oracle_risk="independent",
        notes="Observability delta is recorded on each grounding attempt.",
    ),
    Invariant(
        id="SEMANTIC_PRESERVATION_TRACE",
        layer="observability",
        description=(
            "Every run records source proposition, canonical-action "
            "proposition, world node, admitted status, and compact-role "
            "classification, including why each compact role was assigned."
        ),
        coverage="story",
        tests=(
            "SemanticPreservationTraceTests.test_trace_lists_source_world_admission_and_role",
        ),
        oracle_risk="independent",
        notes=(
            "New artifact beside the admitted world. Does not replace the "
            "canonical specialist JSON trace."
        ),
    ),
    Invariant(
        id="OVERALL_AND_CONDITIONAL_LIKELIHOOD_COEXIST",
        layer="world_admission",
        description=(
            "An overall UNLIKELY characterization of remaining deaths must "
            "not overwrite NEAR_CERTAIN if a failure gate occurs, or vice "
            "versa. Protective walls are CERTAIN facility context, not a "
            "second compact harm or benefit."
        ),
        coverage="story",
        tests=(
            "ProtectiveOverallLikelihoodTests.test_overall_unlikely_does_not_overwrite_gated_near_certain",
            "ProtectiveOverallLikelihoodTests.test_separate_unlikely_survival_is_double_counted",
            "ProtectiveOverallLikelihoodTests.test_quantity_payload_keeps_raw_and_canonical",
            "ProtectiveOverallLikelihoodTests.test_source_plan_label_uses_cited_plan_clause",
            "SavedWorldReplayTests.test_committed_sample_stays_committed_with_protective_context",
            "SavedWorldReplayTests.test_rejected_sample_fails_safely",
        ),
        oracle_risk="independent",
        notes="Parser fill, not a second death row.",
    ),
)


def invariant_by_id() -> dict[str, Invariant]:
    return {item.id: item for item in INVARIANTS}


def format_integrity_layers_report() -> str:
    """Report multidimensional coverage for semantic-integrity invariants."""
    headers = (
        "Invariant",
        "cataloged",
        "seed",
        "structured",
        "grounding",
        "validator",
        "metamorphic",
        "e2e",
        "enforcement",
    )
    rows: list[tuple[str, ...]] = []
    for item in INVARIANTS:
        layers = item.integrity_layers
        if layers is None:
            continue
        rows.append((
            item.id,
            "yes" if layers.cataloged else "no",
            "yes" if layers.seed_case else "no",
            "yes" if layers.structured_property else "no",
            "yes" if layers.grounding_property else "no",
            "yes" if layers.production_validator else "no",
            "yes" if layers.metamorphic_test else "no",
            "yes" if layers.end_to_end else "no",
            item.enforcement or "—",
        ))
    if not rows:
        return "No semantic-integrity invariants with integrity_layers."
    widths = [
        max(len(headers[index]), max(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def fmt(cols: tuple[str, ...]) -> str:
        return "  ".join(
            cols[index].ljust(widths[index]) for index in range(len(headers))
        )

    lines = [fmt(headers), "-" * (sum(widths) + 2 * (len(headers) - 1))]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def production_status(item: Invariant) -> str:
    if item.production:
        return item.production
    if item.coverage == "structural":
        return "pass"
    if item.coverage == "story":
        return "story"
    return "untested"


def format_coverage_report() -> str:
    headers = ("Invariant", "Generation", "Production")
    rows = [
        (item.id, item.coverage, production_status(item).upper())
        for item in INVARIANTS
    ]
    widths = [
        max(len(headers[index]), max(len(row[index]) for row in rows))
        for index in range(3)
    ]

    def fmt(cols: tuple[str, ...]) -> str:
        return "  ".join(
            cols[index].ljust(widths[index]) for index in range(3)
        )

    lines = [fmt(headers), "-" * (sum(widths) + 4)]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)
