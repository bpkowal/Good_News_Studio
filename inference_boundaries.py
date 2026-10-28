"""Typed argument bridges and precise evidence locations, never world edits.

Types describe reviewer/model annotations. Findings request review, not rejection
of the whole argument or certification of semantic support.
"""
from copy import deepcopy

KINDS=['PHYSICAL_CONTRIBUTION','PURPOSE','BELIEF','PURPOSIVE_USE','INTENDED_USE',
       'MAXIM','SOURCE_RULE','NORMATIVE_PREMISE','NORMATIVE_JUDGMENT']
MODES=['SOURCE_STATED','ATTRIBUTED_INFERENCE','CONDITIONAL','UNKNOWN']
RULES={
    'SOURCE_TO_INTERPRETATION':
        'Bind the particular predicate and qualifiers to exact source or provenance text. '
        'A citation can support purpose without supporting exclusivity or a complete actual maxim.',
    'CAUSAL_ROLE_TO_INTENDED_USE':
        'Separate actual causal contribution from the proposed end, instrumental belief and purposive use. '
        'Actual efficacy is neither proof of purposive use nor required for a mistaken instrumental belief. '
        'Defend the bridge, express the reading conditionally, or withdraw it. Do not infer a prohibition.',
    'INTENDED_USE_TO_PROHIBITION':
        'Using a person as a means does not alone establish merely-as-means treatment or prohibition. '
        'Identify the additional normative premises, consent facts and their limits.',
    'SOURCE_RULE_TO_NORMATIVE_PRIORITY':
        'A source rule states a requirement; it does not alone establish its moral priority. '
        'Identify the applicable normative premise or retain a conditional argument.'}


def value_at(value,pointer):
    """Resolve a JSON pointer relative to the exact evidence record's value."""
    if pointer=='':return value
    if not isinstance(pointer,str) or not pointer.startswith('/'):
        raise ValueError('Evidence value_path must be a JSON pointer')
    for component in pointer[1:].split('/'):
        component=component.replace('~1','/').replace('~0','~')
        if isinstance(value,list):
            if not component.isdigit():raise ValueError('Invalid evidence list index')
            value=value[int(component)]
        else:value=value[component]
    return value


def reasoning_schema():
    text={'type':'string','minLength':1}
    ids={'type':'array','items':text}
    support={'type':'object','additionalProperties':False,
             'properties':{'evidence_ref':text,'value_path':{'type':'string'},'quote':text},
             'required':['evidence_ref','value_path','quote']}
    claim_fields={'id':text,'kind':{'type':'string','enum':KINDS},
                  'interpretation_mode':{'type':'string','enum':MODES},'text':text,
                  'supports':{'type':'array','items':support},'condition_ids':ids}
    bridge_fields={'id':text,'rule_id':{'type':'string','enum':list(RULES)},
                   'premise_ids':ids,'assumption_ids':ids,'conclusion_id':text,'explanation':text}
    return {'type':'object','additionalProperties':False,
            'properties':{'claims':{'type':'array','minItems':1,'items':{'type':'object',
                'additionalProperties':False,'properties':claim_fields,'required':list(claim_fields)}},
                'bridges':{'type':'array','minItems':1,'items':{'type':'object','additionalProperties':False,
                    'properties':bridge_fields,'required':list(bridge_fields)}}},
            'required':['claims','bridges']}


def _check_reasoning(mapping,evidence):
    """Verify bindings/structure. Valid references are not entailment judgments."""
    if set(mapping)!={'claims','bridges'} or not mapping['claims'] or not mapping['bridges']:
        raise ValueError('Reasoning requires claims and bridges')
    sources={e['ref']:e['value'] for e in evidence}
    if len(sources)!=len(evidence):raise ValueError('Duplicate evidence reference')
    claims={c['id']:c for c in mapping['claims']}
    if len(claims)!=len(mapping['claims']):raise ValueError('Duplicate argument claim')
    dependencies={cid:set() for cid in claims}
    for claim in claims.values():
        required={'id','kind','interpretation_mode','text','supports','condition_ids'}
        if not required<=set(claim) or set(claim)-required-{'polarity','modality','lifecycle'}:
            raise ValueError('Unknown or missing argument claim field')
        if claim.get('polarity','AFFIRMED') not in ('AFFIRMED','DENIED','UNKNOWN'):
            raise ValueError('Unknown claim polarity')
        if claim['kind'] not in KINDS or claim['interpretation_mode'] not in MODES:
            raise ValueError('Unknown claim kind or interpretation mode')
        if not claim['text'].strip():raise ValueError('Empty argument claim')
        dependencies[claim['id']].update(claim['condition_ids'])
        if claim['interpretation_mode']=='CONDITIONAL' and not claim['condition_ids'] and claim['kind']!='MAXIM':
            raise ValueError('Conditional conclusion must identify its condition')
        for support in claim['supports']:
            if set(support)!={'evidence_ref','value_path','quote'}:
                raise ValueError('Support must identify an exact evidence location')
            try: value=value_at(sources[support['evidence_ref']],support['value_path'])
            except (KeyError,IndexError,TypeError):raise ValueError('Unknown evidence location')
            if not isinstance(value,str) or not support['quote'] or support['quote'] not in value:
                raise ValueError('Support quote does not match evidence location')
    bridge_ids=set()
    for bridge in mapping['bridges']:
        if set(bridge)!={'id','rule_id','premise_ids','assumption_ids','conclusion_id','explanation'}:
            raise ValueError('Unknown or missing bridge field')
        if bridge['id'] in bridge_ids:raise ValueError('Duplicate bridge identity')
        bridge_ids.add(bridge['id'])
        if bridge['rule_id'] not in RULES or bridge['conclusion_id'] not in claims:
            raise ValueError('Unknown bridge rule or conclusion')
        target_kind=claims[bridge['conclusion_id']]['kind']
        if bridge['rule_id']=='CAUSAL_ROLE_TO_INTENDED_USE' and target_kind not in ('BELIEF','PURPOSIVE_USE','INTENDED_USE'):
            raise ValueError('Causal/practical bridge must identify a practical-reasoning conclusion')
        if bridge['rule_id'] in ('INTENDED_USE_TO_PROHIBITION','SOURCE_RULE_TO_NORMATIVE_PRIORITY') and target_kind!='NORMATIVE_JUDGMENT':
            raise ValueError('Normative bridge must identify a normative conclusion')
        dependencies[bridge['conclusion_id']].update(bridge['premise_ids']+bridge['assumption_ids'])
    visited,active=set(),set()
    def visit(cid):
        if cid not in claims:raise ValueError('Dangling bridge premise or condition')
        if cid in active:raise ValueError('Cyclic argument bridge')
        if cid in visited:return
        active.add(cid)
        for parent in dependencies[cid]:visit(parent)
        active.remove(cid);visited.add(cid)
    for cid in claims:visit(cid)
    return True


def check_reasoning(mapping,evidence):
    try:return _check_reasoning(mapping,evidence)
    except (KeyError,TypeError,AttributeError,IndexError) as error:
        raise ValueError('Malformed argument record: '+str(error)) from error


def boundary_card(mapping,evidence):
    check_reasoning(mapping,evidence)
    claims={c['id']:c for c in mapping['claims']}
    parents={cid:set(c['condition_ids']) for cid,c in claims.items()}
    for bridge in mapping['bridges']:
        parents[bridge['conclusion_id']].update(bridge['premise_ids']+bridge['assumption_ids'])
    def conditional(cid):
        return claims[cid]['interpretation_mode']=='CONDITIONAL' or any(conditional(p) for p in parents[cid])
    findings=[]
    def note(code,bridge):
        findings.append({'code':code,'bridge_id':bridge['id'],
                         'conclusion_id':bridge['conclusion_id'],'status':'REVIEW_REQUESTED'})
    for bridge in mapping['bridges']:
        inputs=[claims[c] for c in bridge['premise_ids']+bridge['assumption_ids']]
        kinds={c['kind'] for c in inputs if c['interpretation_mode']!='UNKNOWN' and c.get('polarity','AFFIRMED')=='AFFIRMED'}
        conclusion=claims[bridge['conclusion_id']]
        if bridge['rule_id']=='CAUSAL_ROLE_TO_INTENDED_USE':
            if conclusion['interpretation_mode']!='UNKNOWN' and not {'PURPOSE','BELIEF','PURPOSIVE_USE'}<=kinds:
                note('PRACTICAL_REASONING_BRIDGE_INCOMPLETE',bridge)
        elif bridge['rule_id'] in ('INTENDED_USE_TO_PROHIBITION','SOURCE_RULE_TO_NORMATIVE_PRIORITY'):
            if 'NORMATIVE_PREMISE' not in kinds:note('NORMATIVE_BRIDGE_INCOMPLETE',bridge)
        if conclusion['interpretation_mode']=='SOURCE_STATED' and not conclusion['supports']:
            note('SOURCE_STATUS_LACKS_DIRECT_BINDING',bridge)
        if conclusion['interpretation_mode'] not in ('CONDITIONAL','UNKNOWN') and any(
                conditional(c['id']) for c in inputs):
            note('CONDITIONAL_SCOPE_NOT_PRESERVED',bridge)
    return {'version':'inference-boundary/0.1','authority':'INVESTIGATIVE_ARGUMENT_ANNOTATION',
            'operative':False,'reasoning':deepcopy(mapping),'findings':findings,
            'questions':[RULES[rule] for rule in dict.fromkeys(b['rule_id'] for b in mapping['bridges'])],
            'semantic_support':'NOT_INDEPENDENTLY_ASSESSED',
            'physical_path_required_for_belief':False,
            'citation_policy':'Quote matching verifies location, not entailment or exclusivity.'}
