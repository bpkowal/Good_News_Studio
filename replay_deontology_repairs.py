"""Replay allowed local fields of a saved reply; never calls a model."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from deontology_repair_cards import build_cards, apply_repair, fingerprint, get_path, PERSON_PATHS


def replay(source, reply_path, output, parliament_root):
    sys.path.insert(0, str(Path(parliament_root).resolve()))
    source, reply_path, output = Path(source), Path(reply_path), Path(output)
    generation = json.loads(source.read_text())
    candidate = next(c for c in generation['cycles'][-1]['candidates']
                     if c['specialist'] == 'deontological')
    original = candidate['committed_native_ledger']
    world = generation['action_source_grounding']['world_model']
    original_world_hash = fingerprint(world)
    calls = json.loads(reply_path.read_text())
    reply = json.loads(calls[0]['response']['choices'][0]['text'])
    initial = build_cards(world, original)
    current, audits = deepcopy(original), []
    for initial_card in initial:
        # Each accepted repair changes the record hash. Regenerate trusted scope.
        card = next((c for c in build_cards(world, current)
                     if c.family == initial_card.family and c.record_id == initial_card.record_id), None)
        if card is None:
            continue
        row = reply['dp'][card.action_id]
        changes = ({'harm_relation':row['hr']} if card.family == 'HARM_CLASSIFICATION' else
                   {p:(row['mr'] if p == 'means_relation' else get_path(row, p.replace('candidate_maxim.', 'mx.')))
                    for p in PERSON_PATHS})
        authored_refs = {ref for premise in reply.get('ep', []) for ref in premise.get('se', [])}
        authored_refs.update(row.get('mx', {}).get('means_effect_ids', []))
        authored_refs.update(row.get('mx', {}).get('end_effect_ids', []))
        response = {'card_id':card.card_id, 'changes':changes,
                    'evidence_refs':sorted(authored_refs & {e['ref'] for e in card.evidence}),
                    'explanation':reply['qa']['answer'], 'unresolved_questions':[]}
        current, audit = apply_repair(world, current, card, response,
            scenario=generation['scenario'], action_grounding=generation['action_source_grounding']['actions'])
        audit['response_origin'] = 'OFFLINE_PROJECTION_OF_SAVED_FULL_REPLY_NOT_NEW_LOCAL_MODEL_RESPONSE'
        audit['trusted_card'] = card.to_dict()
        audits.append(audit)
    output.mkdir(parents=True, exist_ok=True)
    result = {'api_calls':0, 'operative':False, 'status':'NONVOTING_STAGED_REPAIR',
              'source':str(source), 'source_sha256':fingerprint(generation),
              'saved_reply':str(reply_path), 'saved_reply_sha256':fingerprint(calls),
              'world_sha256':original_world_hash, 'initial_cards':[c.to_dict() for c in initial],
              'audits':audits, 'staged_ledger':current,
              'excluded_full_reply':reply,
              'projection_policy':'Only card-targeted classification fields copied. Norms, verdicts, priority, rankings and other actions excluded.',
              'world_unchanged':fingerprint(world)==original_world_hash,
              'unrelated_action_unchanged':all(
                  next(r for r in current['records'] if r['assessment_node_id']==old['assessment_node_id'])==old
                  for old in original['records']
                  if old['assessment_node_id'] not in {c.record_id for c in initial})}
    (output/'repair_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Local deontology repair replay', '',
             'Offline replay of saved fields; **zero API calls**. This is a nonvoting staged ledger, not an operative revision.', '',
             '| Repair | Result |', '| --- | --- |']
    lines += ['| '+a['family']+' | '+a['status']+' |' for a in audits]
    lines += ['', 'The original world, saved runs, and other action remain unchanged. '
              'The staged ledger has transaction status `STAGED_NONVOTING`; native validation status is recorded separately. '
              'Priority, universalization and permissibility require renewed adjudication. '
              'The saved reply’s changed norm and priority were excluded. '
              'Native validation establishes structural consistency, not the truth of MIXED or the agent’s intentions.', '',
              'See [complete cards, responses, exclusions and audit](repair_replay.json).']
    (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='diagnostics/lab_all_agents_20261019/frameworks/framework_generation.json')
    parser.add_argument('--reply', default='diagnostics/lab_all_agents_20261019/review/deontological/model_calls.json')
    parser.add_argument('--output', default='diagnostics/deontology_repair_cards_20261020')
    parser.add_argument('--parliament-root', default='.runtime/parliament')
    args = parser.parse_args()
    result = replay(args.source, args.reply, args.output, args.parliament_root)
    print(json.dumps({'api_calls':0, 'repairs':[(a['family'],a['status'],a['errors']) for a in result['audits']]}))
