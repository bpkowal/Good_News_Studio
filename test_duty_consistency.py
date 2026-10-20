"""Normative inconsistency remains reviewable without becoming settled truth."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'.runtime/parliament'))
from global_workspace.deontology_ledger import (
    DutyAssessmentProposal, calibrate_deontological_adjudication, perfect_duty_priority_issue,
)
from global_workspace.scenario_semantics import compile_scenario_graph


class DutyConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = json.loads((ROOT/'diagnostics/deontology_rule_attention_20261019/live/control/framework_generation.json').read_text())
        c = cls.trace['cycles'][-1]['candidates'][0]
        cls.rows = c['deontological_ledger_proposal']['assessments']
        grounding = cls.trace['action_source_grounding']
        cls.graph = compile_scenario_graph(cls.trace['scenario'], cls.trace['actions'],
            grounding['actions'], world_model=grounding['world_model'])

    def calibration(self, **changes):
        row = deepcopy(self.rows[1]); row.update(changes)
        item = DutyAssessmentProposal.model_validate(row)
        action = next(n for n in self.graph.nodes.values()
                      if n.kind == 'ACTION' and n.attributes.get('canonical_action_id') == 'A1')
        return calibrate_deontological_adjudication(self.graph, action, item)

    def test_primary_classification_does_not_silently_type_competing_duty(self):
        errors = {perfect_duty_priority_issue(dt, 'COMPETING', 'PERFECT_DUTY')
                  for dt in ('IMPERFECT', 'PERFECT_NEGATIVE')}
        self.assertEqual(len(errors), 1)
        self.assertIn('competing governing duty', errors.pop())
        self.assertIsNone(perfect_duty_priority_issue('PERFECT_NEGATIVE', 'PRIMARY', 'PERFECT_DUTY'))

    def test_inconsistent_primary_priority_is_contested_not_erased(self):
        result = self.calibration(priority_basis='PERFECT_DUTY', duty_type='IMPERFECT')
        self.assertTrue(any('primary governing norm' in error for error in result.errors))
        self.assertEqual(result.assessment.norm, self.rows[1]['norm'])
        self.assertEqual(result.assessment.duty_type, 'IMPERFECT')
        self.assertEqual(result.assessment.verdict, 'CONFLICTED')
        self.assertEqual(result.assessment.priority_basis, 'UNRESOLVED')

    def test_unclassified_competing_priority_stays_unresolved_even_when_contested(self):
        result = self.calibration(governing_norm='COMPETING', priority_basis='PERFECT_DUTY',
                                  resolution_status='CONTESTED')
        self.assertTrue(any('competing governing duty' in error for error in result.errors))
        self.assertEqual(result.assessment.governing_norm, 'UNRESOLVED')

    def test_explained_revision_is_preserved_but_not_certified(self):
        revision = {'previous_duty_type':'IMPERFECT', 'revised_duty_type':'PERFECT_NEGATIVE',
                    'reason':'I now interpret this claim as a negative duty.',
                    'review_status':'UNVERIFIED'}
        result = self.calibration(duty_type='PERFECT_NEGATIVE', priority_basis='PERFECT_DUTY',
                                  classification_revision=revision)
        self.assertEqual(result.assessment.classification_revision, revision)
        self.assertEqual(result.assessment.duty_type, 'PERFECT_NEGATIVE')
        self.assertEqual(result.assessment.verdict, 'CONFLICTED')
        self.assertEqual(result.assessment.resolution_status, 'CONTESTED')
        self.assertTrue(any('independent review' in e for e in result.errors))


if __name__ == '__main__':
    unittest.main()
