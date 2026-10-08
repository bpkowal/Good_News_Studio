from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from parliament_recognition import explicit_probability, supported_records, component_support

FIXTURE = Path('fixtures/recognition_gap_medicine_frozen.json')


class RecognitionTests(unittest.TestCase):
    def test_scene_statement_preserves_inability_and_requires_whole_source(self):
        source = 'Maria cannot stop the trolley.'
        proposition = {'proposition_id':'PR1','predication':source,'status':'ASSERTED','clause_ids':['C0']}
        world = {'propositions':[proposition]}
        cards = supported_records(world, source)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]['claim'], source)
        self.assertEqual(cards[0]['embedded_occurrence'], 'NOT_INFERRED')
        for status in ('GOVERNED','ATTRIBUTED','UNASSERTED'):
            proposition['status'] = status
            self.assertEqual(supported_records(world,source), [])
        proposition.update(status='ASSERTED', predication='stop the trolley.')
        self.assertEqual(supported_records(world,source), [])

    def fixture(self):
        trace = json.loads(FIXTURE.read_text())
        return trace['action_source_grounding']['world_model'], trace['scenario']

    def test_probability_keeps_known_mass_without_claiming_certain_occurrence(self):
        for text, expected in [('95% chance', '0.95'), ('5% chance', '0.05'),
                               ('1% chance', '0.01'), ('0.5% probability', '0.005'),
                               ('0% chance', '0'), ('100 percent chance', '1')]:
            with self.subTest(text=text):
                self.assertEqual(explicit_probability({'modality': 'PROBABILISTIC',
                    'likelihood_qualifiers': [text]}), expected)

    def test_ambiguous_hedged_and_invalid_probabilities_stay_unknown(self):
        for qualifiers in [[], ['may survive'], ['at least 95% chance'],
                           ['not 95% chance'], ['101% chance'], ['-1% chance'],
                           ['95% chance', '5% chance']]:
            self.assertEqual(explicit_probability({'modality': 'PROBABILISTIC',
                'likelihood_qualifiers': qualifiers}), 'UNKNOWN')
        self.assertEqual(explicit_probability({'modality': 'STIPULATED_CONDITIONAL',
            'likelihood_qualifiers': ['95% chance']}), 'UNKNOWN')
        self.assertEqual(explicit_probability({'modality': 'PROBABILISTIC',
            'likelihood_qualifiers': ['95% chance'],
            'overall_likelihood_qualifiers': ['5% chance']}), 'UNKNOWN')

    def test_precision_is_not_silently_rounded_to_fit_the_native_field(self):
        self.assertEqual(explicit_probability({'modality': 'PROBABILISTIC',
            'likelihood_qualifiers': ['0.12345678901234567890123456789% chance']}), 'UNKNOWN')

    def test_quantity_exclusivity_and_rules_have_distinct_proofs(self):
        world, source = self.fixture()
        before = deepcopy(world)
        cards = supported_records(world, source)
        self.assertEqual(world, before)
        self.assertEqual(len(cards), 5)
        quantity = next(c for c in cards if c['kind'] == 'QUANTITY')
        exclusive = next(c for c in cards if c['kind'] == 'EXCLUSIVITY')
        rule = next(c for c in cards if 'patient who' in c['claim'])
        self.assertEqual(quantity['aliases'], ['one dose of medicine'])
        self.assertEqual(quantity['support_ids'], ['P2'])
        self.assertNotEqual(quantity['proposition_id'], exclusive['proposition_id'])
        self.assertEqual(rule['scope'], 'SOURCE_RULE')
        self.assertEqual(rule['support_ids'], ['E5', 'E6', 'E7', 'E8'])
        self.assertEqual(rule['source_evidence'][0]['clause_id'], 'C4')

    def test_quantity_alone_cannot_prove_exclusivity(self):
        world, source = self.fixture()
        for effect in world['effects']:
            effect['provenance'] = [p for p in effect['provenance'] if p['clause_id'] != 'C1']
        cards = supported_records(world, source)
        self.assertTrue(any(c['kind'] == 'QUANTITY' for c in cards))
        self.assertFalse(any(c['kind'] == 'EXCLUSIVITY' for c in cards))

    def test_unbacked_rule_and_changed_source_cannot_be_projected(self):
        world, source = self.fixture()
        world['causal_links'] = []
        self.assertFalse(any(c['kind'] == 'RULE' for c in supported_records(world, source)))
        self.assertEqual(supported_records(world, 'An unrelated scenario.'), [])

    def test_stock_quantity_never_borrows_a_recipient_count_or_modal_possession(self):
        world, source = self.fixture()
        changed = source.replace('A clinic has one dose of medicine.', 'If a clinic has one dose of medicine.')
        for e in world['effects'] + world['parties']:
            for p in e['provenance']:
                if p['clause_id'] == 'C0':p['excerpt'] = 'If a clinic has one dose of medicine.'
        self.assertFalse(any(c['kind'] == 'QUANTITY' for c in supported_records(world, changed)))

    def test_inventory_does_not_require_transfer_or_exclusivity(self):
        world, source = self.fixture()
        world['effects'] = []
        world['actions'] = []
        world['causal_links'] = []
        before = deepcopy(world)
        cards = supported_records(world, source)
        self.assertEqual([c['kind'] for c in cards], ['QUANTITY'])
        self.assertEqual(cards[0]['support_ids'], ['P2'])
        self.assertEqual(world, before)

    def test_resource_identity_and_own_evidence_are_required(self):
        world, source = self.fixture()
        for party in world['parties']:
            if party['party_id'] == 'P2': party['provenance'] = []
        self.assertFalse(any(c['kind'] == 'QUANTITY' for c in supported_records(world, source)))
        world, source = self.fixture()
        world['parties'][1]['kind'] = 'PERSON'
        self.assertFalse(any(c['kind'] == 'QUANTITY' for c in supported_records(world, source)))

    def test_reported_stock_does_not_become_unqualified_inventory(self):
        world, source = self.fixture()
        for text in ['Ada says a clinic has one dose of medicine.',
                     'Perhaps a clinic has one dose of medicine.',
                     'A clinic might have one dose of medicine.',
                     'A clinic has two doses of medicine.']:
            changed = deepcopy(world)
            changed['parties'][1]['provenance'][0]['excerpt'] = text
            self.assertFalse(any(c['kind'] == 'QUANTITY' for c in supported_records(changed, text)))

    def test_compound_reports_support_without_authorizing_the_extra_conclusion(self):
        world, source = self.fixture()
        cards = supported_records(world, source)
        parts = component_support('one dose of medicine (so only one patient can benefit)', cards)
        self.assertEqual([p['status'] for p in parts], ['SUPPORTED_COMPONENT', 'UNRESOLVED_COMPONENT'])
        self.assertFalse(parts[1]['support_ids'])
        self.assertEqual(component_support('Ben is entrusted to Ada', cards)[0]['status'], 'UNRESOLVED_COMPONENT')

    def test_saved_strengthened_quantity_claim_keeps_its_unsupported_words(self):
        world, source = self.fixture()
        cards = supported_records(world, source)
        claim = 'Only one dose of the life-saving medicine is available (so only one patient can be aided)'
        parts = component_support(claim, cards)
        self.assertEqual(parts[0]['status'], 'PARTIALLY_SUPPORTED_COMPONENT')
        self.assertEqual(parts[0]['claim'], 'Only one dose of the life-saving medicine is available')
        self.assertEqual(parts[0]['supported_fragments'][0]['claim'], 'one dose of medicine')
        self.assertFalse(parts[0]['support_ids'])
        self.assertEqual(parts[1]['status'], 'UNRESOLVED_COMPONENT')
        for changed in ['two doses of medicine are available', 'not one dose of medicine is available']:
            self.assertFalse(component_support(changed, cards)[0]['supported_fragments'])

    def test_native_saved_world_projection_and_compound_binding(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        if not native.exists():self.skipTest('Native interpreter unavailable')
        code = r'''
import sys,json
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,'/tmp/parliament-smoke-614af0c')
from global_workspace.scenario_semantics import compile_scenario_graph
from global_workspace.epistemic_ledger import seed_proposition_ledger, resolve_proposition, attach_candidate_dependencies, _semantic_match
from global_workspace.models import CandidateChunk
from global_workspace.utilitarian_ledger import utilitarian_accounting
f=json.loads(Path(sys.argv[1]).read_text()); before=deepcopy(f)
g=compile_scenario_graph(f['scenario'],f['actions'],world_model=f['action_source_grounding']['world_model'])
l=seed_proposition_ledger(g)
cards=g.discourse_collections['recognition_records']
rule=next(c for c in cards if 'patient who' in c['claim'])
assert resolve_proposition(l,rule['claim'])==rule['proposition_id']
assert _semantic_match('Ben will die',l[rule['proposition_id']],ledger=l) is None
assert _semantic_match('All patients will die',l[rule['proposition_id']],ledger=l) is None
for n in g.nodes.values():
 if n.kind=='CONSEQUENCE' and n.attributes.get('world_effect_id') in ['E2','E4']:
  assert utilitarian_accounting(g,n.id)[1]=={'E2':'0.95','E4':'0.05'}[n.attributes['world_effect_id']]
candidate=CandidateChunk('utilitarian','UNCERTAINTY',{'A0':0.8,'A1':0.2},0,0,0.5,
 material_empirical_claims=[{'claim':rule['claim'],'declared_basis':'HYPOTHESIS','decision_critical':True},
 {'claim':'one dose of medicine (so only one patient can benefit)','declared_basis':'HYPOTHESIS','decision_critical':True},
 {'claim':'Ben is entrusted to Ada','declared_basis':'HYPOTHESIS','decision_critical':True}])
attach_candidate_dependencies(l,candidate)
claims=candidate.material_empirical_claims
assert claims[0]['canonical_proposition']==rule['proposition_id']
assert claims[1]['recognized_components'][0]['status']=='SUPPORTED_COMPONENT'
assert l[claims[1]['proposition_id']].epistemic_status=='HYPOTHETICAL'
assert l[claims[2]['proposition_id']].epistemic_status=='HYPOTHETICAL'
assert f==before
print(json.dumps({'cards':cards,'claims':claims,'probabilities':{'E2':'0.95','E4':'0.05'}}))
'''
        result = subprocess.run([str(native), '-c', code, str(FIXTURE)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_imperfect_duty_can_be_represented_without_false_priority(self):
        native = Path('/tmp/parliament-smoke-env/bin/python')
        if not native.exists():self.skipTest('Native interpreter unavailable')
        code = r'''
import sys,json
sys.path.insert(0,'/tmp/parliament-smoke-614af0c')
from global_workspace.local_specialists import _candidate_from_data, extract_scenario_facts, CompactLocalSpecialist
from global_workspace.models import WorkspaceBroadcast
f=json.load(open('diagnostics/logic_puzzles_independent_live/framework_generation.json'))
saved=json.load(open('diagnostics/logic_puzzles_independent_live/deontological/model_calls.json'))[0]['response']
d=json.loads(saved['choices'][0]['text'])
# Synthetic classification probe: no empirical binding or normative conclusion
# is being claimed for the saved scenario.
d.update(ep=[],sps=[],dcp=[])
for row in d['dp'].values():row.update(dt='IMPERFECT',pb='PERFECT_DUTY')
def parse():
 return _candidate_from_data('deontological',f['actions'],d,WorkspaceBroadcast(),
  'NONE',extract_scenario_facts(f['scenario']),scenario_text=f['scenario'])
bad=parse()
assert len([e for e in bad.framework_validation_errors if 'perfect-duty priority' in e])==2
assert not bad.deontological_ledger_proposal
for aid,row in d['dp'].items():
 row.update(pb='UNRESOLVED',v='CONFLICTED',gv='UNRESOLVED',res='CONTESTED')
 d['fm'][aid]='CONFLICTED: imperfect duty priority remains unsettled'
good=parse()
assert not good.framework_validation_errors,good.framework_validation_errors
assert len(good.deontological_ledger_proposal['assessments'])==2
assert all(r['duty_type']=='IMPERFECT' and r['verdict']=='CONFLICTED'
           for r in good.deontological_ledger_proposal['assessments'])
# Exercise the actual prompt rather than only inspecting its source text.
class PromptCaptured(Exception):pass
class Capture:
 def complete_json(self,prompt,**kwargs):
  assert 'Do not change dt merely to pass validation.' in prompt
  assert 'An IMPERFECT rescue duty does not become a' in prompt
  raise PromptCaptured()
try:
 CompactLocalSpecialist('deontological',Capture()).evaluate(f['scenario'],f['actions'],WorkspaceBroadcast())
except PromptCaptured:pass
else:raise AssertionError('Prompt was not captured')
'''
        result = subprocess.run([str(native), '-c', code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':unittest.main()
