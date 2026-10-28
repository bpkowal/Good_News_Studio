"""Bounded local derivation repairs; no world edits or operative moral votes."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json


VERSION = 'derivation-repair/0.1'
PERSON_PATHS = ('means_relation', 'candidate_maxim.person_role',
                'candidate_maxim.instrumental_belief', 'candidate_maxim.instrumental_use',
                'candidate_maxim.practical_reasoning', 'candidate_maxim.means_effect_ids',
                'candidate_maxim.end_effect_ids')
PERSON_MESSAGES = {
    'candidate maxim denies intended use while proposing instrumental belief and purposive use',
    'candidate maxim person-role conflicts with means classification',
    'candidate maxim intended use lacks a proposed instrumental belief and purposive use',
    'intended-as-means classification lacks an in-action causal path',
}
DEPENDENTS = {'verdict': 'CONFLICTED', 'governing_norm': 'UNRESOLVED',
              'priority_basis': 'UNRESOLVED', 'resolution_status': 'CONTESTED',
              'priority_rule': 'Priority requires renewed adjudication after an upstream repair.',
              'reason': 'Prior adjudication is superseded; the repaired classification does not settle permissibility.',
              'candidate_maxim.action_permissibility': 'UNRESOLVED',
              'candidate_maxim.maxim_assessment': 'UNRESOLVED',
              'candidate_maxim.universalization': 'Prior assessment is superseded; universalization requires renewed adjudication.'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def get_path(row, path):
    value = row
    for part in path.split('.'):
        value = value[part]
    return value


def set_path(row, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        row = row[part]
    row[parts[-1]] = deepcopy(value)


@dataclass(frozen=True)
class RepairCard:
    card_id: str
    version: str
    family: str
    action_id: str
    record_id: str
    world_sha256: str
    record_sha256: str
    issue_records: tuple
    target_paths: tuple
    current_values: dict
    evidence: tuple
    violated_invariant: str
    required_repair: str
    invalidate_if_changed: tuple
    authority: str = 'VALIDATOR_DERIVED_REPAIR_SCOPE_NOT_MORAL_VERDICT'

    def to_dict(self):
        return asdict(self)


def build_cards(world, ledger):
    """Use existing native issue records, not model-authored repair permissions."""
    cards = []
    for row in ledger.get('records', []):
        if not isinstance(row.get('candidate_maxim'), dict):
            continue  # Missing maxim structure is outside these first two families.
        issues = [i for i in row.get('calibration_issues', []) + row.get('pending_repair_issues', [])
                  if i.get('status') == 'ACTIVE']
        for family in ('HARM_CLASSIFICATION', 'PERSON_ROLE_CONSISTENCY'):
            matches = [i for i in issues if (
                i.get('kind') == 'HARM_RELATION_GRAPH_MISALIGN' if family == 'HARM_CLASSIFICATION'
                else i.get('message') in PERSON_MESSAGES)]
            if not matches:
                continue
            paths = ('harm_relation',) if family == 'HARM_CLASSIFICATION' else PERSON_PATHS
            try:
                values = {p: deepcopy(get_path(row, p)) for p in paths}
            except KeyError:
                continue
            action = row['canonical_action_id']
            evidence = []
            for index, effect in enumerate(world.get('effects', [])):
                if effect.get('action_id') == action:
                    evidence.append({'ref': effect['effect_id'], 'path': '/effects/'+str(index),
                                     'kind': 'FROZEN_WORLD_COPY', 'value': deepcopy(effect)})
            for index, relation in enumerate(world.get('causal_links', [])):
                if relation.get('action_id') == action:
                    evidence.append({'ref': 'LINK:'+str(index), 'path': '/causal_links/'+str(index),
                                     'kind': 'FROZEN_WORLD_COPY', 'value': deepcopy(relation)})
            evidence += [{'ref': 'ARGUMENT:'+p, 'path': row['assessment_node_id']+'/'+p,
                          'kind': 'ATTRIBUTED_COMMITTED_ARGUMENT', 'value': deepcopy(v)}
                         for p, v in values.items()]
            harm = family == 'HARM_CLASSIFICATION'
            card = RepairCard('', VERSION, family, action, row['assessment_node_id'],
                fingerprint(world), fingerprint(row), tuple(deepcopy(matches)), paths, values,
                tuple(evidence),
                ('Under the native classification contract, DOING_HARM needs admitted agent-caused '
                 'settled welfare harm; possibility and ENABLES alone do not establish it.' if harm else
                 'Means relation, person role, instrumental belief/use and same-action means/end '
                 'references must be mutually consistent; physical necessity is not proof of intention.'),
                ('Reconsider the harm classification using this action topology, or leave it unresolved. '
                 'Do not infer permissibility or settled injury.' if harm else
                 'Repair the person-role bundle atomically, or leave it unresolved. Do not infer '
                 'actual intentions, consent or a moral verdict.'), tuple(DEPENDENTS))
            data = card.to_dict(); data.pop('card_id')
            cards.append(RepairCard(**{**card.__dict__, 'card_id':'REPAIR:'+fingerprint(data)[:24]}))
    return cards


def response_schema(card):
    enums = {
        'harm_relation': ['DOING_HARM','ALLOWING_HARM','PREVENTING_HARM','WITHHOLDING_BENEFIT','MIXED','NOT_APPLICABLE','UNRESOLVED'],
        'means_relation': ['INTENDED_AS_MEANS','FORESEEN_SIDE_EFFECT','NO_INSTRUMENTALIZATION','NOT_APPLICABLE','UNRESOLVED'],
        'candidate_maxim.person_role': ['INTENDED_MEANS','FORESEEN_AFFECTED','NOT_USED','UNRESOLVED'],
        'candidate_maxim.instrumental_belief': ['RELIES_ON_PERSON','DOES_NOT_RELY','UNRESOLVED'],
        'candidate_maxim.instrumental_use': ['USES_PERSON','DOES_NOT_USE','UNRESOLVED'],
    }
    props = {}
    for path in card.target_paths:
        props[path] = ({'type':'string','enum':enums[path]} if path in enums else
                       {'type':'array','items':{'type':'string'}} if path.endswith('_effect_ids') else
                       {'type':'string','minLength':1})
    fields = {'card_id':{'type':'string','enum':[card.card_id]},
              'changes':{'type':'object','properties':props,'required':list(props),'additionalProperties':False},
              'evidence_refs':{'type':'array','minItems':1,'items':{'type':'string','enum':[e['ref'] for e in card.evidence]}},
              'explanation':{'type':'string','minLength':1},
              'unresolved_questions':{'type':'array','items':{'type':'string'}}}
    return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}


def request_repair(card, backend, *, max_tokens=1200):
    """One small response per card; caller supplies the backend and call budget."""
    prompt = ('LOCAL DERIVATION REPAIR. Return only the requested local fields. '
              'Keep the framework commitments and all other actions fixed. Do not choose a verdict. '
              'Values remain attributed interpretations; abstention is valid.\n' +
              json.dumps(card.to_dict(), ensure_ascii=False))
    response = backend.complete_json(prompt, schema=response_schema(card), max_tokens=max_tokens,
                                     temperature=0.0)
    if isinstance(response, dict) and 'choices' in response:
        return json.loads(response['choices'][0]['text'])
    return response


def _native_proposal(ledger):
    from global_workspace.deontology_ledger import DutyAssessmentProposal
    rows = []
    for record in ledger['records']:
        row = {k:deepcopy(v) for k,v in record.items() if k in DutyAssessmentProposal.model_fields}
        row['action_id'] = record['canonical_action_id']
        rows.append(row)
    return {'assessments':rows}


def apply_repair(world, ledger, card, response, *, scenario='', action_grounding=None):
    """Trusted scope check, dependent invalidation, then whole native revalidation.

    Returns a NONVOTING staged ledger. It never edits an existing run/graph or
    invokes the full-response continuity bypass based on an LLM's own label.
    """
    original = deepcopy(ledger)
    audit = {'version':VERSION,'card_id':card.card_id,'family':card.family,
             'response':deepcopy(response),'status':'REJECTED','errors':[],
             'world_state_authority':'NONE','operative':False,
             'semantic_support':'NOT_INDEPENDENTLY_ASSESSED'}
    try:
        trusted = next((c for c in build_cards(world, ledger) if c.card_id == card.card_id), None)
        if trusted is None or trusted != card:
            raise ValueError('STALE_OR_ALTERED_CARD')
        if not isinstance(response, dict) or set(response) != set(response_schema(card)['required']):
            raise ValueError('INVALID_LOCAL_RESPONSE_FIELDS')
        if response['card_id'] != card.card_id:
            raise ValueError('WRONG_CARD_ID')
        changes = response['changes']
        if not isinstance(changes, dict) or set(changes) != set(card.target_paths):
            raise ValueError('OUT_OF_SCOPE_OR_INCOMPLETE_ATOMIC_REPAIR')
        refs = response['evidence_refs']
        if not isinstance(refs, list) or not refs or any(not isinstance(r,str) for r in refs):
            raise ValueError('MISSING_EVIDENCE_REFERENCES')
        if not set(refs) <= {e['ref'] for e in card.evidence}:
            raise ValueError('UNKNOWN_EVIDENCE_REFERENCE')
        if not isinstance(response['explanation'],str) or not response['explanation'].strip():
            raise ValueError('MISSING_EXPLANATION')
        if not isinstance(response['unresolved_questions'],list) or any(not isinstance(q,str) for q in response['unresolved_questions']):
            raise ValueError('INVALID_QUESTIONS')
        updated = deepcopy(ledger)
        row = next(r for r in updated['records'] if r['assessment_node_id'] == card.record_id)
        if all(get_path(row,p) == v for p,v in changes.items()):
            raise ValueError('NO_LOCAL_CHANGE')
        for path, value in changes.items():
            set_path(row,path,value)
        previous_closure = {p:deepcopy(get_path(row,p)) for p in DEPENDENTS}
        for path,value in DEPENDENTS.items():
            set_path(row,path,value)
        from global_workspace.deontology_ledger import (
            DutyAssessmentProposal, apply_deontological_ledger_transaction,
            calibrate_deontological_adjudication, harm_relation_conflicts_with_graph)
        from global_workspace.scenario_semantics import compile_scenario_graph
        from global_workspace.graph_transactions import SemanticGraphStore
        actions = world['actions']
        graph = compile_scenario_graph(scenario,[a['intervention'] for a in actions],
            action_grounding or {},world_model=deepcopy(world))
        proposal = _native_proposal(updated)
        item = DutyAssessmentProposal.model_validate(next(r for r in proposal['assessments'] if r['action_id']==card.action_id))
        action = next(n for n in graph.nodes.values() if n.kind=='ACTION' and n.attributes.get('canonical_action_id')==card.action_id)
        conflict = harm_relation_conflicts_with_graph(graph,item,action=action)
        if card.family == 'HARM_CLASSIFICATION' and conflict:
            raise ValueError(conflict)
        errors = calibrate_deontological_adjudication(graph,action,item).errors
        if card.family == 'PERSON_ROLE_CONSISTENCY' and any(
            'candidate maxim' in e or 'intended-as-means' in e for e in errors):
            raise ValueError('INCONSISTENT_PERSON_ROLE_BUNDLE: '+'; '.join(errors))
        store = SemanticGraphStore(graph)
        tx = apply_deontological_ledger_transaction(store,proposal,cycle=1,specialist='deontological',
            allowed_actions=tuple(a['action_id'] for a in actions))
        if not tx.status.startswith('COMMITTED'):
            raise ValueError('NATIVE_REVALIDATION_REJECTED: '+'; '.join(tx.errors))
        # Transactions return assessment payloads, without graph-node metadata.
        # Compare every returned payload field before restoring that metadata.
        records = []
        for payload in tx.proposal['committed_assessments']:
            prior_record = next(r for r in ledger['records']
                                if r['assessment_node_id'] == payload['assessment_node_id'])
            if payload['assessment_node_id'] != card.record_id and any(
                    prior_record.get(k) != v for k, v in payload.items()):
                raise ValueError('UNRELATED_RECORD_CHANGED_DURING_REVALIDATION')
            records.append({**deepcopy(prior_record), **deepcopy(payload)})
        new = next(r for r in records if r['assessment_node_id']==card.record_id)
        if any(get_path(new,p) != v for p,v in changes.items()):
            raise ValueError('NATIVE_REVALIDATION_DID_NOT_RETAIN_LOCAL_CHANGE')
        for old in ledger['records']:
            if old['assessment_node_id'] != card.record_id:
                other = next(r for r in records if r['assessment_node_id']==old['assessment_node_id'])
                if other != old:
                    raise ValueError('UNRELATED_RECORD_CHANGED_DURING_REVALIDATION')
        if new['resolution_status'] != 'CONTESTED' or new['governing_norm'] != 'UNRESOLVED' or new['priority_basis'] != 'UNRESOLVED':
            raise ValueError('DOWNSTREAM_CLOSURE_WAS_NOT_INVALIDATED')
        # Target record may acquire fresh native calibration bookkeeping, but
        # normative commitments and identities must not change.
        fixed = ('norm','norm_kind','competing_norm','competing_norm_kind','duty_type',
                 'duty_bearer','protected_party','competing_protected_party','derivation')
        prior = next(r for r in ledger['records'] if r['assessment_node_id']==card.record_id)
        if any(new.get(k) != prior.get(k) for k in fixed):
            raise ValueError('PRINCIPLE_OR_IDENTITY_REVISION_NOT_AUTHORIZED')
        # Native adjudication may stop emitting a classification warning once
        # resolution is CONTESTED. Withdrawing closure does not repair that
        # classification. Retain its previously issued card separately while
        # the explicit graph check still finds the same harm conflict.
        pending = [deepcopy(i) for i in
                   prior.get('calibration_issues', []) + prior.get('pending_repair_issues', [])
                   if conflict and i.get('kind') == 'HARM_RELATION_GRAPH_MISALIGN'
                   and i.get('status') == 'ACTIVE']
        new['pending_repair_issues'] = list({fingerprint(i):i for i in pending}.values())
        result = deepcopy(ledger)
        result['records'] = deepcopy(records)
        # Existing consumers use COMMITTED* as an operative-ledger gate. Native
        # validation of a scratch graph must not accidentally satisfy that gate.
        result['transaction_status'] = 'STAGED_NONVOTING'
        result['native_validation_status'] = tx.status
        result['preference_status'] = 'REQUIRES_REDELIBERATION'
        audit.update(status='STAGED_NATIVE_REVALIDATED',
            revision_category='EVIDENCE_REPAIR' if card.family=='HARM_CLASSIFICATION' and changes['harm_relation']=='UNRESOLVED' else 'APPLICATION_REVISION',
            before_record_sha256=fingerprint(prior), after_record_sha256=fingerprint(new),
            changes=deepcopy(changes), invalidated_paths=list(DEPENDENTS),
            superseded_closure=previous_closure, dependency_rule='CONSERVATIVE_AFFECTED_ROW_INVALIDATION',
            effective_resolution='CONTESTED', preference_status='REQUIRES_REDELIBERATION',
            native_validation_status=tx.status,
            carried_forward_repair_issues=deepcopy(new['pending_repair_issues']),
            native_warnings=list(tx.errors), native_transaction=deepcopy(tx.proposal))
        return result,audit
    except (ValueError, KeyError, TypeError, StopIteration) as error:
        audit['errors'].append(str(error) or type(error).__name__)
        return original,audit
