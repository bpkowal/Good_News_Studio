from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from argument_frameworks import FRAMEWORK_GUIDANCE
from contractualist_questions import review_questions, render_questions
from contractualist_pilot import compile_response, make_request, source_packet, write_argument_report
from complaint_consequences import consequence_request, bind_consequence_reply, build_consequence_record
from opposing_complaint import bind_assignment
from test_contractualist_pilot import proposal, FIXTURE
from test_complaint_consequences import rich_packet, reviewed_reply
from test_opposing_complaint import assignment


def question():
    """Authored open comparison; does not assert the correct action or priority."""
    return {'id': 'q_priority', 'kind': 'NORMATIVE', 'role': 'DECISION_DEPENDENCY',
        'question': 'Does the greater exposure justify prioritizing one household over others?',
        'person_ids': ['jo'], 'action_ids': ['B', 'C'], 'principle_ids': ['principle'],
        'target_handles': ['cmp'],
        'why_it_matters': 'The proposed allocation needs a comparison of personal claims.',
        'needed_to_resolve': 'A defended priority warrant; counting alone does not supply one.',
        'conditional_implications': [
            {'assumption': 'A stronger individual claim has priority here',
             'implication': 'C gains a reason against B, subject to other affected people.'},
            {'assumption': 'A justified competing priority applies',
             'implication': 'B may remain defensible, conditional on that priority.'}]}


class CriticalQuestionTests(unittest.TestCase):
    def setUp(self):
        self.source = source_packet(json.loads(FIXTURE.read_text())['cases'][0])

    def test_question_focused_contribution_is_valid_without_recommendation(self):
        raw = proposal(self.source); raw['critical_questions'] = [question()]
        raw['conclusion'] = 'The decision turns on an unanswered priority question.'
        original = deepcopy(raw)
        result = compile_response(self.source, 'contractualist', raw)
        self.assertEqual(result['status'], 'STAGED_ARGUMENT')
        self.assertIsNone(result['raw_response']['recommendation_action_id'])
        self.assertEqual(result['critical_question_review']['findings'], [])
        self.assertFalse(result['critical_question_review']['question_is_automatic_blocker'])
        self.assertEqual(result['critical_question_review']['unanswerable_in_principle'], 'NOT_ESTABLISHED')
        self.assertEqual(raw, original)

    def test_conditional_preference_keeps_the_dependency_open_without_being_rejected(self):
        raw = proposal(self.source); raw['critical_questions'] = [question()]
        raw.update(recommendation_action_id='B', recommendation_status='PROVISIONAL_PREFERENCE',
                   conclusion='B only if the stated priority can be defended; the objection remains.')
        result = compile_response(self.source, 'contractualist', raw)
        self.assertEqual(result['critical_question_review']['findings'], [])
        self.assertEqual(result['raw_response']['recommendation_action_id'], 'B')
        self.assertEqual(result['critical_question_review']['questions'][0]['role'], 'DECISION_DEPENDENCY')

    def test_open_dependency_and_unique_verdict_are_visible_without_automatic_rewrite(self):
        raw = proposal(self.source); raw['critical_questions'] = [question()]
        raw.update(recommendation_action_id='B', recommendation_status='UNIQUE_DEFENSIBLE')
        result = compile_response(self.source, 'contractualist', raw)
        self.assertIn('OPEN_DEPENDENCY_WITH_UNIQUENESS_CLAIM',
                      {f['code'] for f in result['critical_question_review']['findings']})
        self.assertEqual(result['raw_response'], raw)
        self.assertNotEqual(result['status'], 'MALFORMED_RESPONSE')

    def test_residual_question_need_not_prevent_an_independently_defended_choice(self):
        raw = proposal(self.source); q = question(); q['role'] = 'RESIDUAL_CONCERN'
        raw['critical_questions'] = [q]
        raw.update(recommendation_action_id='B', recommendation_status='UNIQUE_DEFENSIBLE')
        result = compile_response(self.source, 'contractualist', raw)
        self.assertEqual(result['critical_question_review']['findings'], [])
        self.assertEqual(result['raw_response']['recommendation_action_id'], 'B')
        # Separate comparison findings may still apply; questions do not certify the verdict.
        self.assertFalse(result['critical_question_review']['operative'])

    def test_unknown_references_and_duplicates_preserve_every_question(self):
        raw = proposal(self.source); q = question(); q['person_ids'] = ['not_declared']
        questions = [q, deepcopy(q)]
        result = review_questions(questions, raw, {'A', 'B', 'C'}, 'UNRESOLVED')
        self.assertEqual(result['questions'], questions)
        self.assertEqual({f['code'] for f in result['findings']},
                         {'UNKNOWN_QUESTION_REFERENCE', 'DUPLICATE_QUESTION_ID'})

    def test_unreported_legacy_questions_are_not_reported_as_resolved_or_absent(self):
        raw = proposal(self.source); original = deepcopy(raw)
        result = compile_response(self.source, 'contractualist', raw)
        self.assertEqual(result['critical_question_review']['recording_status'], 'NOT_RECORDED_LEGACY')
        self.assertIn('absence is not evidence', render_questions(result['critical_question_review']))
        self.assertEqual(result['raw_response'], original)
        self.assertNotIn('critical_questions', raw)

    def test_explicit_empty_questions_does_not_certify_semantic_completeness(self):
        raw = proposal(self.source); raw['critical_questions'] = []
        result = compile_response(self.source, 'contractualist', raw)
        review = result['critical_question_review']
        self.assertEqual(review['recording_status'], 'RECORDED')
        self.assertEqual(review['semantic_support'], 'NOT_INDEPENDENTLY_ASSESSED')
        self.assertIn('not independent certification', render_questions(review))

    def test_conditional_answers_do_not_become_world_facts_or_argument_premises(self):
        raw = proposal(self.source)
        before = compile_response(self.source, 'contractualist', raw)
        raw['critical_questions'] = [question()]
        after = compile_response(self.source, 'contractualist', raw)
        self.assertEqual(before['argument'], after['argument'])
        self.assertEqual(before['view'], after['view'])
        self.assertFalse(after['operative'])

    def test_readable_report_leads_with_questions_and_preserves_the_graph(self):
        raw = proposal(self.source); raw['critical_questions'] = [question()]
        result = compile_response(self.source, 'contractualist', raw)
        with TemporaryDirectory() as temp:
            path = Path(temp)/'candidate_arguments.md'
            write_argument_report(result, path)
            report = path.read_text()
            self.assertLess(report.index('Critical unanswered questions'), report.index('```mermaid'))
            self.assertIn('What could help:', report)
            self.assertIn('If A stronger individual claim', report)
            self.assertIn('q_priority', path.with_name('critical_questions.md').read_text())

    def test_new_common_schema_does_not_impose_contractualism_on_other_frameworks(self):
        requests = [make_request(self.source, f) for f in ('contractualist', 'utilitarian')]
        self.assertEqual(requests[0]['response_format'], requests[1]['response_format'])
        self.assertEqual(requests[0]['messages'][0], requests[1]['messages'][0])
        self.assertIn('critical_questions', requests[0]['response_format']['json_schema']['schema']['properties'])
        self.assertIn('action recommendation is NOT required', FRAMEWORK_GUIDANCE['contractualist'])
        self.assertNotIn('actual consent', FRAMEWORK_GUIDANCE['utilitarian'])

    def test_followup_uses_current_question_guidance_without_rewriting_saved_instruction(self):
        packet = rich_packet(); original = deepcopy(packet)
        bound_assignment = bind_assignment(packet, assignment())
        request = consequence_request(packet, bound_assignment, 'consequences_first',
                                      model='o3', reasoning_effort='low')
        user = json.loads(request['messages'][1]['content'])
        self.assertEqual(user['framework_instruction'], original['framework_instruction'])
        self.assertIn('action recommendation is NOT required', request['messages'][0]['content'])
        self.assertEqual(packet, original)
        r = reviewed_reply(); q = question()
        q.update(person_ids=['bea'], action_ids=['A', 'B'], principle_ids=[], target_handles=[])
        r.update(critical_questions=[q], disposition='UNANSWERED',
                 conclusion={'action_id': None, 'status': 'UNRESOLVED',
                             'rationale': 'The priority remains undefended.'})
        result = bind_consequence_reply(packet, bound_assignment, r, build_consequence_record(packet))
        self.assertEqual(result['critical_question_review']['findings'], [])
        self.assertFalse(result['self_report_is_resolution'])
        self.assertFalse(result['operative'])

    def test_factual_and_normative_questions_are_not_silently_conflated(self):
        raw = proposal(self.source); normative = question(); factual = question()
        factual.update(id='q_severity', kind='FACTUAL', question='What health risks remain without cooling?',
                       needed_to_resolve='Individual health and exposure evidence.')
        raw['critical_questions'] = [normative, factual]
        result = compile_response(self.source, 'contractualist', raw)
        self.assertEqual([q['kind'] for q in result['critical_question_review']['questions']], ['NORMATIVE', 'FACTUAL'])
        self.assertEqual(result['critical_question_review']['questions'][0]['needed_to_resolve'], normative['needed_to_resolve'])

    def test_all_frozen_original_pilot_arguments_remain_readable_without_invented_questions(self):
        root = Path(__file__).parent/'diagnostics/contractualist_comparison_v03_live_20261025'
        paths = sorted(root.glob('*/*/response.json'))
        self.assertEqual(len(paths), 16)
        for path in paths:
            original = path.read_bytes(); raw = json.loads(original)
            source = json.loads(path.with_name('source.json').read_text())
            result = compile_response(source, path.parent.name, raw)
            self.assertNotEqual(result['status'], 'MALFORMED_RESPONSE')
            self.assertEqual(result['raw_response'], raw)
            self.assertEqual(result['critical_question_review']['recording_status'], 'NOT_RECORDED_LEGACY')
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
