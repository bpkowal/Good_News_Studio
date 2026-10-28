from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from run_sol_parliament_comparison import MODEL,prepare,stage_request,run_stage,check_freeze


class FrozenComparisonTests(unittest.TestCase):
    def prepare(self, root):
        scenario=root/'source.txt';scenario.write_text('Ellis chooses one hiker.')
        packet=root/'packet.json';packet.write_text(json.dumps({'source':scenario.read_text(),
            'frameworks':[{'framework':str(i),'reason':'PARLIAMENT_SECRET'} for i in range(5)]}))
        out=root/'experiment';prepare(out,scenario,packet);return out

    def review(self):
        return dict(recommendation={'choice':'Rowan','conclusion':'A provisional rescue choice.'},
            reasoning='SOL_PRIVATE_REPLY',perspectives=[],strongest_objection='Promise',
            answer_to_objection='Consider its force.',assumptions=[],uncertainties=[],
            reversal_conditions=[],assessment_of_supplied_analysis=None)

    def client(self):
        calls=[]
        def create(**kwargs):
            calls.append(deepcopy(kwargs))
            raw={'model':MODEL,'choices':[{'finish_reason':'stop',
                'message':{'content':json.dumps(self.review()),'refusal':None}}],'usage':{}}
            return SimpleNamespace(model_dump=lambda **_:raw)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),calls

    def test_blind_has_only_scenario_and_matched_informed_request_excludes_first_reply(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.prepare(Path(temp));client,calls=self.client()
            blind=stage_request(out,'blind')
            self.assertNotIn('PARLIAMENT_SECRET',json.dumps(blind))
            self.assertEqual(json.loads(blind['messages'][1]['content'])['additional_analysis'],None)
            run_stage(out,'blind',client)
            informed=stage_request(out,'informed')
            self.assertIn('PARLIAMENT_SECRET',json.dumps(informed))
            self.assertNotIn('SOL_PRIVATE_REPLY',json.dumps(informed))
            for key in ('model','reasoning_effort','max_completion_tokens','response_format'):
                self.assertEqual(blind[key],informed[key])
            self.assertEqual(blind['messages'][0],informed['messages'][0])
            self.assertNotIn('tools',informed)
            self.assertEqual(len(calls),1)

    def test_stage_order_and_both_freezes_required_before_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.prepare(Path(temp));client,calls=self.client()
            with self.assertRaises(FileNotFoundError):stage_request(out,'informed')
            with self.assertRaises(FileNotFoundError):stage_request(out,'comparison')
            run_stage(out,'blind',client)
            with self.assertRaises(FileNotFoundError):stage_request(out,'comparison')
            run_stage(out,'informed',client)
            comparison=stage_request(out,'comparison')
            self.assertIn('SOL_PRIVATE_REPLY',json.dumps(comparison))
            self.assertEqual(len(calls),2)

    def test_tampering_and_repeat_calls_fail_before_provider_request(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.prepare(Path(temp));client,calls=self.client()
            run_stage(out,'blind',client)
            with self.assertRaises(FileExistsError):run_stage(out,'blind',client)
            self.assertEqual(len(calls),1)
            (out/'blind.response.json').write_text('{}')
            with self.assertRaises(ValueError):stage_request(out,'informed')
            self.assertEqual(len(calls),1)

    def test_report_source_mismatch_rejected_before_any_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.write_text('New scenario')
            packet=root/'packet';packet.write_text(json.dumps({'source':'Other scenario','frameworks':[{}]*5}))
            with self.assertRaises(ValueError):prepare(root/'out',source,packet)


if __name__=='__main__':unittest.main()
