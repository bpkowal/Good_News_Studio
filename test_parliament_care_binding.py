"""Native integration regression for the frozen lever composition's Care ledger."""
from pathlib import Path
import subprocess
import unittest

from run_blueprint_parliament import DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON


@unittest.skipUnless(DEFAULT_PARLIAMENT_ROOT.exists() and DEFAULT_PARLIAMENT_PYTHON.exists(),
                     "native Parliament checkout unavailable")
class CareBindingTests(unittest.TestCase):
    def test_party_action_and_quantity_binding_without_world_mutation(self):
        trace = Path("diagnostics/primitive_composition_semantic_probe/probe_manifest.json")
        code = r'''
import json, sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from global_workspace.scenario_semantics import compile_scenario_graph
from global_workspace.care_ledger import _care_effects, apply_care_ledger_transaction, committed_care_assessments
from global_workspace.graph_transactions import SemanticGraphStore
m=json.loads(Path(sys.argv[2]).read_text())
d=json.loads(Path(m['workspace_trace']).read_text())
world=d['action_source_grounding']['world_model']
before=deepcopy(world)
actions=d['actions']
graph=compile_scenario_graph(d['scenario'], actions, world_model=world)
assert not _care_effects(graph, 'A0', 'five workers')
assert [e.effect_id for e in _care_effects(graph, 'A0', 'one worker')] == ['E3']
assert [e.effect_id for e in _care_effects(graph, 'A1', 'five workers')] == ['E6']
assert [e.effect_id for e in _care_effects(graph, 'A1', ' The FIVE workers ')] == ['E6']
assert not _care_effects(graph, 'A1', 'one worker')
# Replay the actual model's mismatched claim. It remains a committed uncertain
# interpretation, with no false support edge or world-state repair.
rows=[]
keys=['verdict','affected_party','relationship_type','dependency_source','responsibility_basis',
      'need_kind','need_urgency','trust_effect','responsiveness','feasibility',
      'competing_care_claim','resolution_status','evidence_basis','reason']
for r in d['care_relationship_ledger']:
    row={k:r[k] for k in keys}
    row['action_id']=r['canonical_action_id']
    row['resolution_status']='RESOLVED'
    rows.append(row)
store=SemanticGraphStore(graph)
record=apply_care_ledger_transaction(store,{'ranking_basis':'ENTRUSTED_RESPONSIBILITY','assessments':rows},
                                   cycle=1,specialist='care',allowed_actions=tuple(actions))
assert record.status=='COMMITTED_WITH_UNCERTAINTY', record
committed=committed_care_assessments(store.graph)
assert committed[0]['grounded_effect_ids']==[]
assert committed[0]['effect_grounding_status']=='UNMATCHED'
assert committed[0]['relationship_evidence_status']=='UNVERIFIED'
assert committed[0]['epistemic_status']=='EXPLICIT_UNCERTAINTY'
assert committed[1]['grounded_effect_ids']==['E6']
assert committed[1]['relationship_evidence_status']=='UNVERIFIED'
assert world==before
print('Care wrong-party claim preserved without false binding; valid binding retained; world unchanged.')
'''
        result = subprocess.run([str(DEFAULT_PARLIAMENT_PYTHON), "-c", code,
                                 str(DEFAULT_PARLIAMENT_ROOT), str(trace.resolve())],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
