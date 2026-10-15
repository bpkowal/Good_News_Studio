import json
from pathlib import Path
import unittest

from blueprint_cloze_chooser import _BY_ID, _score_template, choose_by_cloze
from test_blueprint_cloze_chooser import Script


TEXT = ('Maria can pull the lever. If Maria pulls the lever, one worker will die. '
        'If Maria does not pull the lever, five workers will die.')


class TypedOmissionTests(unittest.TestCase):
    def score(self, **extra):
        answers = {'actor':'Maria can pull the lever', 'done':'pulls the lever',
                   'omitted':'does not pull the lever', 'harm_done':'one worker will die',
                   'harm_omitted':'five workers will die'}
        answers.update(extra)
        return _score_template(TEXT, _BY_ID['omission_harm'], answers, 0)

    def test_long_fills_become_separate_copied_roles_and_counts(self):
        row = self.score(done_patient='one worker will die', omitted_patient='five workers will die',
                         done_quantity='one worker', omitted_quantity='five workers',
                         instrument_entity='pulls the lever')
        self.assertEqual({k:row['slots'][k] for k in ['actor','done_patient','omitted_patient',
                         'done_quantity','omitted_quantity','instrument_entity']},
                         {'actor':'Maria','done_patient':'one worker','omitted_patient':'five workers',
                          'done_quantity':'one','omitted_quantity':'five','instrument_entity':'the lever'})
        patient = next(i for i in row['items'] if i['id']=='done_patient')
        self.assertEqual(patient['completion'], 'one worker will die')
        self.assertEqual(patient['hypotheses'][0]['source_span'], 'one worker')

    def test_wrong_branch_patient_and_count_do_not_fill_roles(self):
        row = self.score(done_patient='five workers', omitted_patient='one worker', done_quantity='five')
        self.assertNotIn('done_patient', row['slots'])
        self.assertNotIn('omitted_patient', row['slots'])
        self.assertNotIn('done_quantity', row['slots'])

    def test_action_does_not_absorb_relative_mechanism(self):
        from blueprint_cloze_chooser import _normalize_slot_span
        source='Maria can pull a lever that switches the trolley onto a side track.'
        item=next(i for i in _BY_ID['omission_harm']['items'] if i['id']=='done')
        self.assertEqual(_normalize_slot_span(source,item,'pull a lever that switches the trolley onto a side track',{'actor':'Maria'}),'pull a lever')

    def test_process_fill_does_not_absorb_its_motion_or_victims(self):
        from blueprint_cloze_chooser import _normalize_slot_span, _answers_sentence
        source='The runaway trolley is heading toward five workers.'
        item=next(i for i in _BY_ID['omission_harm']['items'] if i['id']=='controllable_process')
        nominal=_normalize_slot_span(source,item,source,{})
        self.assertEqual(nominal,'The runaway trolley')
        self.assertTrue(_answers_sentence(source,item,nominal,{}))
        self.assertFalse(_answers_sentence(source,item,source,{}))

    def test_harm_fill_keeps_only_outcome_with_original_answer_available(self):
        row=self.score(harm_done='If Maria pulls the lever, one worker will die')
        self.assertEqual(row['slots']['harm_done'],'one worker will die')
        item=next(i for i in row['items'] if i['id']=='harm_done')
        self.assertEqual(item['completion'],'If Maria pulls the lever, one worker will die')

    def test_old_saved_answers_get_typed_branch_bindings(self):
        root=Path(__file__).resolve().parent
        case=json.loads((root/'fixtures/trolley_scene_cloze_cases.json').read_text())['cases'][0]
        result=choose_by_cloze(case['source'], Script(case['ranking'],case['answers']))
        row=next(r for r in result['considered'] if r['blueprint_id']=='omission_harm')
        self.assertEqual(len(row['branch_bindings']),2)
        for binding in row['branch_bindings']:
            self.assertTrue(binding['patient'])
            self.assertNotIn('kill', binding['patient'])
            self.assertTrue(binding['supporting_candidate_ids'])
        self.assertEqual(result['graph']['candidate']['world_model']['schema_version'], '1.4')


if __name__ == '__main__':
    unittest.main()
