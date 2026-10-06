"""Bounded storage/broadcast/adoption test; no live model is needed as oracle."""
from pathlib import Path
import subprocess
import unittest

from run_blueprint_parliament import DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON


@unittest.skipUnless(DEFAULT_PARLIAMENT_ROOT.exists() and DEFAULT_PARLIAMENT_PYTHON.exists(),
                     "native Parliament checkout unavailable")
class HypothesisContainmentTests(unittest.TestCase):
    def test_live_audit_cannot_launder_outcomes_through_quantity_atoms(self):
        code = r'''
import json,sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from global_workspace.models import CandidateChunk
from global_workspace.scenario_semantics import compile_scenario_graph
from global_workspace.epistemic_ledger import seed_proposition_ledger, resolve_proposition, apply_side_premise_audit, ledger_projection
live=json.loads(Path(sys.argv[2]).read_text())
world=live['action_source_grounding']['world_model']
graph=compile_scenario_graph(live['scenario'],live['actions'],world_model=world)
before=deepcopy(graph.to_dict())
ledger=seed_proposition_ledger(graph)
admitted_before={k:asdict(v) for k,v in ledger.items()}
quantity=next(k for k,v in ledger.items() if ':PARTY:' in k and ':QUANTITY:' in k and v.claim=='five workers')
assert resolve_proposition(ledger,'five workers')==quantity
rows=[]
for audit in live['side_premise_audits']:
    care=CandidateChunk(specialist='care',constraint='CARE',action_scores={a:0.5 for a in live['actions']},surprise=0.1,friction=0.2,confidence=0.8)
    apply_side_premise_audit(ledger,[care],audit)
    for finding in care.side_premise_audit_findings:
        record=ledger[finding['proposition_id']]
        assert record.epistemic_type=='HYPOTHESIS',finding
        assert record.epistemic_status=='HYPOTHETICAL',finding
        assert record.introduced_by=='care'
        assert finding['proposition_id']!=quantity
        rows.append({'cycle':audit['cycle'],**finding,'epistemic_status':record.epistemic_status,'introduced_by':record.introduced_by})
assert rows[0]['proposition_id']==rows[1]['proposition_id'],rows
# Direct model/audit citation cannot bypass the same binding rule.
claim='The five workers depend on Maria to operate the lever'
assert not resolve_proposition(ledger,claim,preferred=quantity)
care=CandidateChunk(specialist='care',constraint='CARE',action_scores={},surprise=0.1,friction=0.2,confidence=0.8)
apply_side_premise_audit(ledger,[care],{'status':'FINDINGS','findings':[{'specialist':'care','claim':claim,'binding':quantity,'decision_critical':True}]})
assert ledger[care.side_premise_audit_findings[0]['proposition_id']].epistemic_status=='HYPOTHETICAL'
# Preserve genuine canonical outcome and quantity bindings.
assert resolve_proposition(ledger,ledger['PROP:WORLD:E6'].claim)=='PROP:WORLD:E6'
assert resolve_proposition(ledger,'five workers')==quantity
for key,original in admitted_before.items():
    current=asdict(ledger[key])
    for field in ('claim','aliases','epistemic_status','support_ids','derived_from'):
        assert current[field]==original[field],(key,field)
assert graph.to_dict()==before
out={'saved_live_audit_findings':rows,'world_graph_preserved_exactly':True,'canonical_quantity_binding_preserved':True,'explicit_quantity_citation_cannot_launder_relationship':True,'hypotheses':[r for r in ledger_projection(ledger) if r['epistemic_type']=='HYPOTHESIS']}
target=Path(sys.argv[3]);target.write_text(json.dumps(out,indent=2)+'\n')
print('Saved live survival and dependency findings remain attributed hypotheses; quantity and admitted graph remain intact.')
'''
        live = next(Path("diagnostics/hypothesis_containment_live/parliament").glob("workspace_scenario_*.json")).resolve()
        output = Path("diagnostics/hypothesis_containment_live/audit_replay.json").resolve()
        result = subprocess.run([str(DEFAULT_PARLIAMENT_PYTHON), "-c", code,
                                 str(DEFAULT_PARLIAMENT_ROOT), str(live), str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_entrustment_retains_origin_status_and_dependencies(self):
        code = r'''
import json,sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from global_workspace.models import CandidateChunk, WorkspaceBroadcast
from global_workspace.scenario_semantics import compile_scenario_graph
from global_workspace.epistemic_ledger import seed_proposition_ledger, attach_candidate_dependencies, ledger_projection, shared_unresolved_dependency_projection
from global_workspace.engine import _enforce_dependency_continuity
source=Path(sys.argv[2])
d=json.loads(source.read_text())
world=d['action_source_grounding']['world_model']
before=deepcopy(world)
graph=compile_scenario_graph(d['scenario'],d['actions'],world_model=world)
graph_before=deepcopy(graph.to_dict())
ledger=seed_proposition_ledger(graph)
def chunk(name):
    return CandidateChunk(specialist=name,constraint='CARE' if name=='care' else 'DUTY',
                          action_scores={a:0.5 for a in d['actions']},surprise=0.1,friction=0.2,
                          confidence=0.8,epistemic_confidence=0.8,recommended_action=d['actions'][0],
                          supporting_proposition_ids=[next(iter(ledger))])
care=chunk('care')
care.care_ledger_proposal={'ranking_basis':'ENTRUSTED_RESPONSIBILITY','assessments':[
    {'action_id':'A0','affected_party':'five workers','relationship_type':'ENTRUSTED'}]}
attach_candidate_dependencies(ledger,care)
pid=care.decision_critical_proposition_ids[0]
record=ledger[pid]
assert record.epistemic_status=='HYPOTHETICAL',record
assert record.introduced_by=='care'
assert record.epistemic_type=='HYPOTHESIS'
assert record.support_ids==[]
assert record.derived_from==[]
assert care.schema_valid and care.recommended_action==d['actions'][0]
# A different Care construction is outside this bounded increment.
control=chunk('care')
control.care_ledger_proposal={'ranking_basis':'ACUTE_DEPENDENCY','assessments':[
    {'action_id':'A0','affected_party':'one worker','relationship_type':'DEPENDENCY'}]}
attach_candidate_dependencies(ledger,control)
assert not control.material_empirical_claims
# Both explicit citation and a repeated claim asserting an established basis
# inherit the canonical record's actual status, never the adopter's declaration.
adopter=chunk('deontological')
adopter.material_empirical_claims=[{'claim':record.claim,'declared_basis':'ESTABLISHED','decision_critical':True}]
attach_candidate_dependencies(ledger,adopter)
assert adopter.decision_critical_proposition_ids==[pid]
assert adopter.weakest_decision_critical_status=='HYPOTHETICAL'
assert ledger[pid].introduced_by=='care' and ledger[pid].epistemic_status=='HYPOTHETICAL'
again=chunk('virtue'); again.supporting_proposition_ids.append(pid)
again.decision_critical_proposition_ids=[pid]
attach_candidate_dependencies(ledger,again)
assert ledger[pid].epistemic_status=='HYPOTHETICAL'
assert ledger[pid].mention_count >= 3
# The real broadcast problem_state carries the unresolved projection intact.
shared=shared_unresolved_dependency_projection(ledger,[care,adopter,again])
row=next(r for r in shared if r['proposition_id']==pid)
assert row['introduced_by']=='care' and row['epistemic_status']=='HYPOTHETICAL'
assert row['dependent_specialist_count']==3
broadcast=WorkspaceBroadcast(problem_state={'shared_unresolved_dependencies':shared})
assert asdict(broadcast)['problem_state']['shared_unresolved_dependencies']==shared
projection=next(r for r in ledger_projection(ledger) if r['proposition_id']==pid)
assert projection['introduced_by']=='care' and projection['epistemic_status']=='HYPOTHETICAL'
# Stable recommendations cannot quietly drop an adopted open premise next cycle.
later=chunk('deontological')
_enforce_dependency_continuity(later,adopter,ledger)
assert pid in later.decision_critical_proposition_ids
assert world==before and graph.to_dict()==graph_before
assert not any(n.kind=='RELATIONSHIP' for n in graph.nodes.values())
out={'world_preserved_exactly':True,'semantic_graph_preserved_exactly':True,
     'broadcast_dependency':row,'ledger_record':projection,
     'adopter_weakest_status':adopter.weakest_decision_critical_status,
     'next_cycle_dependency_retained':pid in later.decision_critical_proposition_ids}
target=Path(sys.argv[3]);target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(out,indent=2)+'\n')
print('Entrustment stays attributed and hypothetical across storage, broadcast, adoption, repetition and next-cycle continuity.')
'''
        source = Path("diagnostics/primitive_composition_live_signal/admission_attempts/06_primitive_composition/frozen_world_trace.json").resolve()
        output = Path("diagnostics/hypothesis_containment/entrustment_trace.json").resolve()
        result = subprocess.run([str(DEFAULT_PARLIAMENT_PYTHON), "-c", code,
                                 str(DEFAULT_PARLIAMENT_ROOT), str(source), str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
