import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from logic_puzzles_acceptance import SavedResponses


class AcceptancePathTests(unittest.TestCase):
    def test_replay_cannot_fall_back_to_a_live_model(self):
        backend = SavedResponses([{'saved': True}])
        self.assertEqual(backend.complete_json('one'), {'saved': True})
        with self.assertRaises(RuntimeError): backend.complete_json('two')

    def test_same_revision_exposes_native_review_gate(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        root = Path('/tmp/parliament-smoke-614af0c')
        if not native.exists() or not root.exists(): self.skipTest('Native checkout unavailable')
        code = '''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from logic_puzzles_acceptance import compare_paths
result=compare_paths(Path('diagnostics/logic_puzzles_matched_live'),Path(sys.argv[2]))
arms=result['arms']
assert result['fresh_api_calls']==0 and not result['native_validation_modified']
assert all(a['world_unchanged'] for a in arms.values())
assert arms['fresh_targeted']['operative_means']['A0']=='FORESEEN_SIDE_EFFECT'
assert arms['recurrent_assigned_open']['operative_means']['A0']=='INTENDED_AS_MEANS'
assert arms['recurrent_assigned_review']['operative_means']['A0']=='FORESEEN_SIDE_EFFECT'
assert arms['recurrent_assigned_open']['completed_cycles']==2
assert arms['recurrent_assigned_review']['completed_cycles']==2
assert any('principle state changed' in w for w in arms['recurrent_assigned_open']['framework_warnings'])
assert not any('principle state changed' in w for w in arms['recurrent_assigned_review']['framework_warnings'])
for arm in arms:assert (Path(sys.argv[2])/arm/'conflict_graph.md').exists()
audit=json.loads((Path(sys.argv[2])/'recurrent_assigned_review/continuity_audit.json').read_text())
assert audit[-1]['prior_state_present'] and audit[-1]['constraint']=='PROPOSAL_REVIEW'
assert audit[-1]['challenge_response']['issue_id'] in [j['issue_id'] for j in audit[-1]['challenge_agenda']]
changes=arms['recurrent_assigned_review']['all_record_changes']
assert any('competing_protected_party' in r['changes'] for r in changes)
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(native), '-c', code, str(root), folder], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
