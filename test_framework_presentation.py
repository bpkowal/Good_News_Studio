import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from parliament_framework_presentation import (argument_markdown, preserve_model_arguments,
    readable_prose, apply_reviewed_explanations)
from run_parliament_demo import summarize, write_report
from export_parliament_viewer import export, framework_views_for_report
from serve_parliament_viewer import ViewerHandler


WORLD = {
    'actions': [{'action_id': 'A0', 'intervention': 'pull the lever'},
                {'action_id': 'A1', 'intervention': 'leave the lever alone'}],
    'parties': [{'party_id': 'P0', 'label': 'Maria', 'kind': 'PERSON'},
                {'party_id': 'P1', 'label': 'five workers', 'kind': 'HUMAN_GROUP'},
                {'party_id': 'P2', 'label': 'one worker', 'kind': 'PERSON'}],
}


def report(candidate):
    return summarize('A trolley threatens five workers.', {}, {
        'action_source_grounding': {'world_model': copy.deepcopy(WORLD)},
        'cycles': [{'candidates': [candidate]}]})


class FrameworkPresentationTests(unittest.TestCase):
    def test_reviewed_summaries_preserve_authority_and_reject_changed_evidence(self):
        root = Path(__file__).parent
        for scenario, count in [('medicine', 5), ('promise', 3)]:
            r = json.loads((root/'diagnostics/decision_topography_suite_20261009'/scenario/'DEMO_REPORT.json').read_text())
            original = framework_views_for_report(r)
            before = copy.deepcopy(r)
            views = apply_reviewed_explanations(r, copy.deepcopy(original))
            self.assertEqual(len(views), count)
            for old, new in zip(original, views):
                self.assertEqual(new['reason_authority'], 'REVIEWED_PLAIN_LANGUAGE_SUMMARY_OF_SAVED_ARGUMENT')
                for key in ('position', 'preference', 'ledger_status', 'recorded_argument',
                            'live_objections', 'decision_boundaries'):
                    self.assertEqual(old.get(key), new.get(key))
                self.assertTrue(readable_prose(new['reason']))
                for jargon in ('PROP:', 'LEDGER', 'HYPOTHESIS:', 'A0', 'A1'):
                    self.assertNotIn(jargon, new['reason'])
                markdown = argument_markdown(new)
                self.assertIn(new['reason'], markdown)
                self.assertIn('Original presentation and objections', markdown)
            self.assertEqual(r, before)
            changed = copy.deepcopy(r)
            changed['scenario'] += ' Changed source.'
            self.assertEqual(apply_reviewed_explanations(changed, copy.deepcopy(original)), original)
            changed_views = copy.deepcopy(original)
            changed_views[0]['recorded_argument']['rationale'] = 'A different argument.'
            self.assertNotIn('original_presentation', apply_reviewed_explanations(r, changed_views)[0])

    def test_legacy_export_recovers_actor_roles_without_rewriting_saved_report(self):
        c = {'specialist':'deontological', 'recommended_action':'A1',
             'committed_native_ledger': {'ledger_kind':'DEONTOLOGICAL_DUTY_LEDGER',
                 'transaction_status':'COMMITTED_WITH_UNCERTAINTY', 'records':[{
                     'duty_bearer':'Maria', 'protected_party':'one worker', 'norm':'do not kill',
                     'competing_norm':'rescue people', 'competing_protected_party':'five workers',
                     'canonical_action_id':'A1', 'verdict':'PERMISSIBLE'}]}}
        r = report(c)
        view = r['frameworks'][0]
        view.pop('recorded_argument')
        view['reason'] = "Five workers' duty to rescue."
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'framework_generation.json'
            t = {'scenario':r['scenario'], 'action_source_grounding':{'world_model':WORLD},
                 'cycles':[{'candidates':[c]}]}
            p.write_text(json.dumps(t))
            r['artifacts'] = [str(p)]
            before = copy.deepcopy(r)
            fixed = framework_views_for_report(r)[0]
            self.assertIn('Maria’s “rescue people” toward five workers', fixed['reason'])
            self.assertNotIn("workers' duty", fixed['reason'])
            for key in ('position','preference','ledger_status'):
                self.assertEqual(fixed[key], view[key])
            self.assertEqual(r, before)
            self.assertEqual(framework_views_for_report(r), [fixed])
            # Same scenario wording alone cannot authorize substituting another world.
            t['action_source_grounding']['world_model'] = {'actions':[]}
            p.write_text(json.dumps(t))
            self.assertEqual(framework_views_for_report(r), r['frameworks'])

    def test_markdown_link_is_readable_in_browser(self):
        handler = object.__new__(ViewerHandler)
        self.assertEqual(handler.guess_type('argument.md'), 'text/plain; charset=utf-8')

    def test_duty_bearer_and_protected_parties_stay_distinct(self):
        candidate = {'specialist': 'deontological', 'recommended_action': 'A1',
            'framework_vote_status': 'ABSTAIN', 'rationale': "one worker's do not kill",
            'committed_native_ledger': {'ledger_kind': 'DEONTOLOGICAL_DUTY_LEDGER',
                'transaction_status': 'COMMITTED_WITH_UNCERTAINTY', 'records': [{
                    'duty_bearer': 'Maria', 'protected_party': 'one worker',
                    'norm': 'do not kill innocent people', 'competing_norm': 'rescue people',
                    'competing_protected_party': 'five workers', 'canonical_action_id': 'A0',
                    'verdict': 'CONFLICTED', 'reason': 'The action places a worker in danger'}]}}
        before = copy.deepcopy(candidate)
        r = report(candidate); view = r['frameworks'][0]
        self.assertIn('Maria is the duty-bearer', view['reason'])
        self.assertIn('one worker is the protected party', view['reason'])
        self.assertIn('Maria’s “rescue people” toward five workers', view['reason'])
        self.assertNotIn("one worker's do not kill", view['reason'])
        self.assertIn('conflicted', view['reason'])
        self.assertEqual(view['position'], 'ABSTAIN')
        self.assertEqual(candidate, before)
        self.assertEqual(r['world'], WORLD)

    def test_long_explanation_and_original_fields_survive_markdown_export(self):
        reason = 'This choice protects the people on the main track. ' * 35 + 'This is the final complete sentence.'
        c = {'specialist': 'care', 'rationale': reason[:180],
             'model_argument_fields': {'rationale': reason}}
        r = report(c)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_report(r, root/'report')
            self.assertIn(reason, (root/'report/arguments/care.md').read_text())
            self.assertIn('arguments/care.md', (root/'report/DEMO_REPORT.md').read_text())
            export(output=root/'viewer', trolley_report=root/'report/DEMO_REPORT.json')
            payload = json.loads((root/'viewer/trolley.json').read_text())
            self.assertEqual(payload['frameworks'][0]['reason'], reason)
            self.assertEqual(payload['frameworks'][0]['argument_url'], 'data/trolley_care_argument.md')
            self.assertIn(reason, (root/'viewer/trolley_care_argument.md').read_text())

    def test_malformed_or_symbol_only_text_cannot_be_main_reason(self):
        for malformed in ('NONE","fr":true,"bd":"NONE"', 'A0 > A1 ⇒ U(W)'):
            self.assertFalse(readable_prose(malformed))
            view = report({'specialist': 'rawlsian', 'framework_application': malformed,
                           'rationale': 'The claims of the worst-off workers remain contested.'})['frameworks'][0]
            self.assertNotIn(malformed, view['reason'])
            self.assertIn('remain contested', view['reason'])
            self.assertEqual(view['recorded_argument']['framework_application'], malformed)
            self.assertTrue(view['presentation_notices'])

    def test_incomplete_comparisons_are_visible_as_limitations(self):
        for ending in ('toward the at', 'even-', 'but only'):
            view = report({'specialist': 'virtue', 'rationale': 'Practical wisdom favors intervention.',
                           'landscape_cases': {'A1': 'The agent argues '+ending}})['frameworks'][0]
            self.assertNotIn(ending, view['reason'])
            self.assertIn(ending, argument_markdown(view))
            self.assertTrue(any('incomplete' in n for n in view['presentation_notices']))

    def test_assumptions_are_labeled_and_counts_are_given_not_invented(self):
        for name in ('utilitarian', 'deontological', 'care', 'virtue', 'rawlsian'):
            c = {'specialist': name, 'unsupported_assumption': 'NONE',
                 'decision_rule': 'Prefer A0 if this framework treats preventing more deaths as decisive.'}
            v = report(c)['frameworks'][0]
            self.assertEqual([r['kind'] for r in v['assumption_records']],
                             ['RECORDED_INTERPRETIVE_COMMITMENT', 'RUN_ASSUMPTION'])
            self.assertEqual(v['source_basis']['how_known'], 'NOT_SEPARATELY_RECORDED')
            self.assertIn('five workers; one worker', v['source_basis']['text'])
            self.assertEqual(v['assumptions'], [])  # no fabricated model-reported premise

    def test_reversal_names_both_actions_and_labels_authority(self):
        v = report({'specialist': 'utilitarian', 'recommended_action': 'A0',
                    'factual_reversal_threshold': 'If more workers stood on the side track'})['frameworks'][0]
        text = v['reversal_explanations'][0]
        self.assertIn('from “pull the lever” to “leave the lever alone”', text)
        self.assertIn('side track. This is an agent-reported', text)
        self.assertNotIn('A0', text)

    def test_raw_response_is_additive_and_does_not_rewrite_native_judgment(self):
        raw = {'w': 'A complete reason. '*40, 'r': 'A0'}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'care/model_calls.json'; p.parent.mkdir()
            p.write_text(json.dumps([{'response': {'choices': [{'text': json.dumps(raw)}]}},
                                    {'response': {'choices': [{'text': '{"numeric_check": true}'}]}}]))
            before = p.read_bytes()
            c = {'specialist': 'care', 'rationale': 'A truncated record', 'recommended_action': 'A1'}
            preserve_model_arguments([c], folder)
            self.assertEqual(c['model_argument_fields']['rationale'], raw['w'])
            self.assertEqual(c['rationale'], 'A truncated record')
            self.assertEqual(c['recommended_action'], 'A1')
            self.assertEqual(before, p.read_bytes())

    def test_native_formatter_and_normalizers_preserve_roles_and_prose(self):
        from parliament_runtime import DEFAULT_PARLIAMENT_PYTHON, DEFAULT_PARLIAMENT_ROOT
        python = DEFAULT_PARLIAMENT_PYTHON
        root = DEFAULT_PARLIAMENT_ROOT
        if not python.exists() or not root.exists():
            self.skipTest('Pinned native Parliament checkout unavailable')
        code = '''
from global_workspace.deontology_ledger import classify_deontological_authority
from global_workspace.models import CandidateChunk
from global_workspace.specialist_authority import downgrade_claim_authority
reason = 'A complete explanatory sentence. ' * 30
c = CandidateChunk(specialist='care', constraint='CARE', action_scores={}, surprise=0, friction=0, confidence=.5,
    rationale=reason, framework_application=reason, decision_rule=reason,
    factual_reversal_threshold=reason, normative_reversal_threshold=reason, landscape_cases={'A0': reason})
for key in ('rationale', 'framework_application', 'decision_rule', 'factual_reversal_threshold', 'normative_reversal_threshold'):
    assert getattr(c, key) == reason.strip(), key
assert c.landscape_cases['A0'] == reason.strip()
assert downgrade_claim_authority(reason, 'PROVISIONAL_LEANING', limit=None).endswith(reason.strip())
row = {'protected_party':'one worker', 'competing_protected_party':'five workers', 'duty_bearer':'Maria',
       'norm':'do not kill innocent people', 'competing_norm':'rescue people', 'resolution_status':'RESOLVED'}
p = classify_deontological_authority(row)
assert "Maria's duty" in p.rationale and 'protecting one worker' in p.rationale, p.rationale
assert "one worker's do not kill" not in p.rationale
row.update(resolution_status='UNRESOLVED', coercive_actor='NONE', coerced_party='NONE')
p = classify_deontological_authority(row)
assert "NONE's restriction" not in p.rationale
assert 'priority between these duties' in p.rationale
'''
        proc = subprocess.run([str(python), '-c', code], cwd=root, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)


if __name__ == '__main__':
    unittest.main()
