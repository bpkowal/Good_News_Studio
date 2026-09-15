"""DERIVED_PROPOSITION_STATUS_CONSERVATION over licensed paraphrases."""
from __future__ import annotations

import unittest

from hypothesis import given, settings

from global_workspace.epistemic_ledger import (
    PropositionRecord,
    claim_matches_established,
    register_hypothesis,
    resolve_proposition,
)
from relent import (
    licensed_action_paraphrase,
    status_conserving_paraphrase_errors,
    status_transfers,
)
from strategies.derived_proposition_paraphrase import (
    DerivedPropositionParaphraseCase,
    derived_proposition_paraphrase_cases,
)


def _established_loss_ledger(
    *,
    action_id: str = "A1",
    glosses: tuple[str, ...] = ("refrain from executing the emergency purge",),
) -> dict[str, PropositionRecord]:
    return {
        "PROP:WORLD:F1": PropositionRecord(
            proposition_id="PROP:WORLD:F1",
            claim=(
                "suffers catastrophic loss of life; "
                "affected subject: city residents; "
                "magnitude or qualifier: thousands of lives"
            ),
            proposition_type="DESCRIPTIVE",
            epistemic_status="ESTABLISHED",
            support_ids=["F1"],
            introduced_by="WORLD_MODEL",
            epistemic_type="WORLD_ESTABLISHED",
            aliases=[],
            action_id=action_id,
            outcome="suffers catastrophic loss of life",
            polarity="ADVERSE",
            party_labels=["city residents"],
            quantities=["thousands of lives"],
            effect_kind="HEALTH_OUTCOME",
            modality="CERTAIN",
            context_terms=list(glosses),
        )
    }


class DerivedPropositionStatusConservationTests(unittest.TestCase):
    def test_no_purge_paraphrase_binds_established_f1(self):
        ledger = _established_loss_ledger()
        claim = "Catastrophic loss of life occurs if no purge."
        self.assertEqual(resolve_proposition(ledger, claim), "PROP:WORLD:F1")
        self.assertTrue(claim_matches_established(claim, ledger))
        bound = register_hypothesis(
            ledger, claim, specialist="utilitarian", decision_critical=True,
        )
        self.assertEqual(bound, "PROP:WORLD:F1")
        self.assertEqual(ledger[bound].epistemic_status, "ESTABLISHED")
        self.assertNotEqual(ledger[bound].epistemic_type, "HYPOTHESIS")

    def test_purge_not_executed_paraphrase_binds(self):
        ledger = _established_loss_ledger()
        claim = "Catastrophic loss of life occurs if purge not executed."
        self.assertEqual(resolve_proposition(ledger, claim), "PROP:WORLD:F1")

    def test_execute_paraphrase_does_not_bind_refrain_effect(self):
        ledger = _established_loss_ledger()
        claim = "Catastrophic loss of life occurs if the purge is executed."
        self.assertEqual(resolve_proposition(ledger, claim), "")
        bound = register_hypothesis(ledger, claim, specialist="utilitarian")
        self.assertTrue(bound.startswith("PROP:HYPOTHESIS") or bound == "")
        if bound in ledger:
            self.assertEqual(ledger[bound].epistemic_status, "HYPOTHETICAL")

    def test_paraphrase_relation_transfers_status(self):
        self.assertTrue(status_transfers("PARAPHRASE_OF"))
        self.assertTrue(status_transfers("EQUIVALENT_TO"))

    def test_kernel_oracle_helper_agrees_on_fixture(self):
        errors = status_conserving_paraphrase_errors(
            claim="Catastrophic loss of life occurs if no purge.",
            established_outcome="suffers catastrophic loss of life",
            established_polarity="ADVERSE",
            established_action_id="A1",
            established_action_glosses=(
                "refrain from executing the emergency purge",
            ),
            expect_bind=True,
        )
        self.assertEqual(errors, [])


class DerivedPropositionParaphraseHypothesisTests(unittest.TestCase):
    @settings(max_examples=60, deadline=None)
    @given(derived_proposition_paraphrase_cases())
    def test_oracle_agrees_with_detector(
        self, case: DerivedPropositionParaphraseCase,
    ):
        binds = licensed_action_paraphrase(
            case.claim,
            outcome=case.outcome,
            polarity=case.polarity,
            action_id=case.action_id,
            action_glosses=case.action_glosses,
        )
        self.assertEqual(
            binds,
            case.expect_bind,
            msg=(
                f"mode bind mismatch for {case.claim!r} / {case.outcome!r} "
                f"glosses={case.action_glosses!r}"
            ),
        )
        errors = status_conserving_paraphrase_errors(
            claim=case.claim,
            established_outcome=case.outcome,
            established_polarity=case.polarity,
            established_action_id=case.action_id,
            established_action_glosses=case.action_glosses,
            expect_bind=case.expect_bind,
        )
        self.assertEqual(errors, [])

    @settings(max_examples=40, deadline=None)
    @given(derived_proposition_paraphrase_cases())
    def test_ledger_resolve_agrees_on_refrain_fixtures(
        self, case: DerivedPropositionParaphraseCase,
    ):
        if case.action_id != "A1" or case.polarity != "ADVERSE":
            return
        if "loss" not in case.outcome.casefold() and "die" not in case.outcome.casefold():
            return
        ledger = _established_loss_ledger(
            action_id=case.action_id,
            glosses=case.action_glosses,
        )
        bound = resolve_proposition(ledger, case.claim)
        if case.expect_bind:
            self.assertEqual(bound, "PROP:WORLD:F1", case.claim)
        else:
            self.assertEqual(bound, "", case.claim)


if __name__ == "__main__":
    unittest.main()
