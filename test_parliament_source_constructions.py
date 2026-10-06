from copy import deepcopy
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

import parsing_game_Z10 as z10
from blueprint_primitive_composition import extract_inventory, compose
from test_blueprint_semantic_coverage import BASE, PROMISE
from test_blueprint_graph_amendments import blueprint
from parliament_source_constructions import build_packet, render_advisory
from run_blueprint_parliament import DEFAULT_PARLIAMENT_ROOT, DEFAULT_PARLIAMENT_PYTHON


class SourceConstructionTests(unittest.TestCase):
    def packet(self, text):
        package = z10.export_candidate_graph(text)
        return build_packet(package, compose(text, extract_inventory(text, package), blueprint(BASE)["question"]))

    def test_scoped_source_is_available_without_promoting_it(self):
        for prefix, polarity in ((PROMISE, "positive"), ("Maria did not promise Anna to pull the lever. ", "negative")):
            text = prefix + BASE
            packet = self.packet(text)
            before = deepcopy(packet)
            prompt = render_advisory(packet, text)
            self.assertEqual(packet, before)
            self.assertEqual(packet["constructions"][0]["scope"]["polarity"], polarity)
            self.assertIn("ADVISORY, NOT ADMITTED WORLD FACTS", prompt)
            self.assertIn("not authoritative proposition IDs", prompt)
            self.assertIn("no", prompt)

    def test_absent_packet_preserves_legacy_prompt(self):
        self.assertEqual(render_advisory({}, BASE), "")
        self.assertEqual(self.packet(BASE), {})

    def test_mapped_commitment_is_not_an_unselected_advisory(self):
        from blueprint_discourse import build_promise_reliance
        from blueprint_semantic_coverage import attach_coverage
        package = z10.export_candidate_graph(PROMISE)
        inventory = extract_inventory(PROMISE, package)
        world, _ = build_promise_reliance(
            PROMISE,
            {"promisor": "Maria", "promisee": "Anna",
             "commitment_event": "promised",
             "commitment_content": "to pull the lever"},
        )
        proposal = {
            "proposal_id": "promise_map",
            "candidate": {"world_model": world},
            "unresolved_readings": [],
        }
        annotated = attach_coverage(
            inventory, {"candidate_attempts": [{"proposal": proposal}]})
        packet = build_packet(package, annotated["candidate_attempts"][0]["proposal"])
        self.assertEqual(packet, {})

    def test_source_scope_evidence_and_authority_cannot_be_overridden(self):
        text = PROMISE + BASE
        for mutation in ("source", "scope", "evidence", "authority", "candidate"):
            packet = self.packet(text)
            if mutation == "source":
                packet["source_sha256"] = "changed"
            elif mutation == "scope":
                packet["constructions"][0]["scope"]["polarity"] = "negative"
            elif mutation == "evidence":
                packet["constructions"][0]["source_evidence"][0]["text"] = "Invented"
            elif mutation == "authority":
                packet["authority"] = "WORLD_ESTABLISHED"
            else:
                packet["constructions"][0]["source_candidates"][0]["assessment"]["status"] = "accepted"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                render_advisory(packet, text)

    @unittest.skipUnless(DEFAULT_PARLIAMENT_ROOT.exists() and DEFAULT_PARLIAMENT_PYTHON.exists(), "native checkout unavailable")
    def test_native_replay_delivers_advisory_to_specialists_without_world_mutation(self):
        source = Path("diagnostics/primitive_promise_coverage/promise/frozen_world_trace.json").resolve()
        packet = self.packet(PROMISE + BASE)
        code = r'''
import json,sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from global_workspace.frozen_world_replay import load_frozen_world_trace
from global_workspace.local_specialists import CompactLocalSpecialist
from global_workspace.models import WorkspaceBroadcast
from global_workspace.scenario_semantics import compile_scenario_graph
from global_workspace.epistemic_ledger import seed_proposition_ledger
trace=json.loads(Path(sys.argv[2]).read_text()); packet=json.loads(Path(sys.argv[3]).read_text())
scenario=trace['scenario']; original=load_frozen_world_trace(Path(sys.argv[2]),expected_scenario=scenario)
trace['source_construction_advisory']=packet
target=Path(sys.argv[4]);target.write_text(json.dumps(trace))
replay=load_frozen_world_trace(target,expected_scenario=scenario)
assert replay.fingerprint==original.fingerprint
world_before=deepcopy(replay.action_source_grounding)
assert replay.metadata()['source_construction_advisory']==packet
graph=compile_scenario_graph(scenario,list(replay.canonical_actions),world_model=replay.action_source_grounding['world_model'])
graph_before=deepcopy(graph.to_dict());ledger_before=seed_proposition_ledger(graph)
class Fake:
    def __init__(self):self.prompts=[]
    def complete_json(self,prompt,**kwargs):
        self.prompts.append(prompt);return {'choices':[{'text':'{}'}]}
records=[]
for name in ['deontological','care']:
    fake=Fake()
    delegate=CompactLocalSpecialist(name,fake,source_construction_advisory=deepcopy(packet),canonical_action_records=list(replay.canonical_action_records))
    for cycle in (1,2):
        delegate.evaluate(scenario,list(replay.canonical_actions),WorkspaceBroadcast())
    prompts=[p for p in fake.prompts if 'RETAINED SOURCE CONSTRUCTIONS' in p]
    assert len(prompts)>=2,(name,len(fake.prompts))
    assert all(p.count('RETAINED SOURCE CONSTRUCTIONS — ADVISORY')==1 for p in prompts)
    assert all('promised' in p and 'complement' in p for p in prompts)
    assert delegate.source_construction_advisory==packet
    records.append({'specialist':name,'advisory_prompt_count':len(prompts),'scope_preserved':True})
assert replay.action_source_grounding==world_before
assert graph.to_dict()==graph_before and seed_proposition_ledger(graph)==ledger_before
Path(sys.argv[5]).write_text(json.dumps({'world_fingerprint_unchanged':True,'world_and_ledger_unchanged':True,'specialists':records,'model':'capture-only fake; semantic quality not assessed'},indent=2)+'\n')
print('Native replay and two specialist prompt paths preserve source advisory across cycles without world promotion.')
'''
        out = Path("diagnostics/promise_specialist_handoff").resolve()
        out.mkdir(parents=True, exist_ok=True)
        packet_path = out / "source_construction_advisory.json"
        packet_path.write_text(json.dumps(packet, indent=2))
        result = subprocess.run([str(DEFAULT_PARLIAMENT_PYTHON), "-c", code,
                                 str(DEFAULT_PARLIAMENT_ROOT), str(source), str(packet_path),
                                 str(out / "frozen_world_trace.json"), str(out / "handoff_trace.json")],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
