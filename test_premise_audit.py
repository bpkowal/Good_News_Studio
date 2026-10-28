from copy import deepcopy
import json
import unittest

from argument_support import compile_argument, project
from argument_construction import construct
from premise_audit import audit_packet, bind_audit, apply_source_findings


class PremiseAuditTests(unittest.TestCase):
    def setup_argument(self, source_text='Ivo is a volunteer.', claim='Ivo is untrained.'):
        source = dict(id='audit', world_sha256='WORLD', scenario=source_text, framework='Test duty priority',
                      evidence=[{'ref': 'SCENE', 'value': source_text}])
        statement = dict(handle='fact', kind='PURPOSIVE_USE', text=claim, polarity='AFFIRMED',
                         modality='ASSERTED', hypothesis_ids=[],
                         citations=[dict(evidence_id='SCENE', value_path='', quote=source_text)])
        norm = dict(handle='norm', kind='NORMATIVE_PREMISE', text='If untrained, do not operate controls.',
                    polarity='AFFIRMED', modality='ASSERTED', hypothesis_ids=[], citations=[])
        conclusion = dict(handle='verdict', kind='NORMATIVE_JUDGMENT', text='Operating violates the rule.',
                          polarity='AFFIRMED', modality='ASSERTED', hypothesis_ids=['norm'], citations=[])
        routes = [dict(operation='NORM_APPLICATION', inputs=['fact', 'norm'], output='verdict', explanation='Apply rule')]
        state = compile_argument('WORLD', source['evidence'], 'audit', [statement, norm, conclusion], routes)
        return source, state, [statement, norm, conclusion], routes

    def findings(self, packet, status='NOT_ESTABLISHED'):
        return {'findings': [dict(statement_id=c['statement_id'], components=[dict(claim_span=c['text'],
            status=status, evidence=[], reason='Training status is not given.')],
            source_preserving_reading='Training status remains unknown.') for c in packet['claims']]}

    def test_packet_has_full_ancestry_but_no_priority_or_verdict_designation(self):
        source, state, _, _ = self.setup_argument()
        packet = audit_packet(source, state, 'verdict')
        self.assertEqual(len(packet['claims']), 3)
        self.assertNotIn('framework', packet)
        self.assertNotIn('conclusion_handle', packet)
        self.assertNotIn('routes', packet)
        changed = deepcopy(source); changed['evidence'][0]['value'] = 'Other scene'
        with self.assertRaises(ValueError): audit_packet(changed, state, 'verdict')

    def test_unknown_training_challenges_source_binding_without_asserting_opposite(self):
        source, state, _, _ = self.setup_argument()
        packet = audit_packet(source, state, 'verdict')
        audit = bind_audit(packet, self.findings(packet))
        before = deepcopy(state)
        revised = apply_source_findings(state, audit)
        view = project(revised)
        self.assertEqual(state, before)
        self.assertEqual(view['statements'][state['handles']['fact']]['support_status'], 'CHALLENGED')
        self.assertEqual(view['statements'][state['handles']['verdict']]['support_status'], 'REQUIRES_REASSESSMENT')
        self.assertEqual(set(state['statements']), set(revised['statements']))
        self.assertTrue(all(s['polarity']=='AFFIRMED' for s in revised['statements'].values()))

    def test_supported_training_control_is_not_forced_to_abstain(self):
        source, state, _, _ = self.setup_argument('Ivo is trained.', 'Ivo is trained.')
        packet = audit_packet(source, state, 'fact')
        response = self.findings(packet, 'SOURCE_SUPPORTED')
        response['findings'][0]['components'][0]['evidence'] = [dict(evidence_id='SCENE', value_path='', quote='Ivo is trained.')]
        audit = bind_audit(packet, response)
        revised = apply_source_findings(state, audit)
        self.assertEqual(project(revised)['statements'][state['handles']['fact']]['support_status'], 'SUPPORTED')

    def test_missing_targets_unknown_statements_and_fabricated_quotes_fail(self):
        source, state, _, _ = self.setup_argument()
        packet = audit_packet(source, state, 'fact')
        response = self.findings(packet)
        with self.assertRaises(ValueError): bind_audit(packet, {'findings': []})
        bad = deepcopy(response); bad['findings'][0]['statement_id'] = 'Invented statement'
        with self.assertRaises(ValueError): bind_audit(packet, bad)

    def test_paraphrase_is_labeled_and_bad_alias_binds_only_by_unique_exact_quote(self):
        source, state, _, _ = self.setup_argument()
        packet = audit_packet(source, state, 'fact')
        response = self.findings(packet)
        component=response['findings'][0]['components'][0]
        component.update(claim_span='Ivo lacks training', evidence=[dict(evidence_id='EV1',value_path='SCENE',quote='Ivo is a volunteer.')])
        audit=bind_audit(packet,response)
        bound=audit['findings'][0]['components'][0]
        self.assertEqual(bound['claim_binding'],'REVIEWER_PARAPHRASE_NOT_EXACT_SPAN')
        self.assertEqual(bound['citation_bindings'][0]['evidence_id'],'SCENE')
        self.assertEqual(bound['citation_bindings'][0]['raw']['evidence_id'],'EV1')
        duplicate=deepcopy(packet);duplicate['evidence'].append({'ref':'DUPLICATE','value':source['scenario']})
        with self.assertRaises(ValueError):bind_audit(duplicate,response)
        bad = deepcopy(response); bad['findings'][0]['components'][0].update(status='SOURCE_SUPPORTED', evidence=[dict(
            evidence_id='SCENE', value_path='', quote='Ivo is untrained.')])
        with self.assertRaises(ValueError): bind_audit(packet, bad)

    def test_normative_and_hypothetical_conditions_do_not_become_source_facts(self):
        source, state, _, _ = self.setup_argument()
        packet = audit_packet(source, state, 'verdict')
        response = self.findings(packet)
        for finding in response['findings']:
            if finding['statement_id'] == state['handles']['norm']:
                finding['components'] = [dict(claim_span='If untrained', status='CONDITIONAL_PRECONDITION',
                    evidence=[], reason='Hypothetical applicability, not a case fact.'),
                    dict(claim_span='do not operate controls', status='NORMATIVE_ASSUMPTION', evidence=[],
                         reason='Moral authority needs argument.')]
        revised = apply_source_findings(state, bind_audit(packet, response))
        self.assertEqual(project(revised)['statements'][state['handles']['norm']]['support_status'], 'CONDITIONAL')

    def test_stale_audit_cannot_apply_to_different_argument(self):
        source, state, _, _ = self.setup_argument()
        packet = audit_packet(source, state, 'fact')
        audit = bind_audit(packet, self.findings(packet))
        changed = deepcopy(state); changed['construction_problems'] = []
        with self.assertRaises(ValueError): apply_source_findings(changed, audit)

    def test_audit_drives_local_revision_in_constructor(self):
        source, _, statements, routes = self.setup_argument()
        proposal = dict(label='Rule application', statements=statements, supports=routes,
            conclusion_handle='verdict', strongest_objection='Training unknown', answer_to_objection='Reassess',
            reversal_condition='Training established')
        calls = []
        outer = self
        class Backend:
            def complete_json(self, prompt, **kwargs):
                calls.append(prompt)
                packet = json.loads(prompt.split('\n', 1)[1])
                if len(calls) == 1:
                    return outer.findings(packet)
                self_test = packet['premise_audit']
                assert self_test['review_authority'] == 'MODEL_SOURCE_REVIEW_NOT_INDEPENDENT_ORACLE'
                replacement = dict(kind='NORMATIVE_JUDGMENT', text='If Ivo is untrained and the rule has moral authority, operating violates it.',
                    polarity='AFFIRMED', modality='ASSERTED', hypothesis_ids=['norm'], citations=[])
                return dict(assessments=[], revisions=[dict(action='qualify', target='verdict', handle='conditional',
                    statement=replacement, reason='Unknown applicability must remain a condition.'),
                    dict(action='withdraw', target='never_declared', reason='Invalid later operation')],
                    strongest_remaining_conclusion=replacement['text'], remaining_questions=['Is Ivo trained?'])
        before = deepcopy(source)
        result = construct(source, Backend(), saved_generation={'candidates': [proposal]}, premise_audit=True)
        candidate = result['candidates'][0]
        self.assertEqual(len(calls), 2)
        self.assertEqual(source, before)
        self.assertEqual(candidate['status'], 'STAGED_PARTIAL_REVISION')
        self.assertEqual(len(candidate['revision_problems']),1)
        sid = candidate['argument']['handles']['verdict']
        self.assertEqual(candidate['view']['statements'][sid]['lifecycle'], 'SUPERSEDED')
        self.assertIn('If Ivo is untrained', candidate['strongest_remaining_conclusion'])

    def test_incomplete_audit_retains_prior_and_does_not_request_revision(self):
        source, _, statements, routes = self.setup_argument()
        proposal = dict(label='test', statements=statements, supports=routes, conclusion_handle='verdict')
        calls = []
        class Backend:
            def complete_json(self, prompt, **kwargs):
                calls.append(prompt); return {'findings': []}
        result = construct(source, Backend(), saved_generation={'candidates': [proposal]}, premise_audit=True)
        row = result['candidates'][0]
        self.assertEqual(len(calls), 1)
        self.assertEqual(row['status'], 'STAGED_PRIOR_RETAINED')
        self.assertEqual(row['argument'], row['compiled_before'])


if __name__ == '__main__':
    unittest.main()
