"""Focused source-applicability review, separate from normative argument critique.

Findings are attributed model judgments. Exact quotations establish traceability,
not entailment. Nothing here edits a world or determines moral priority.
"""
from copy import deepcopy

from argument_support import resolve_citation, project, assess_support
from deontology_repair_cards import fingerprint

VERSION = 'premise-audit/0.1'
STATUSES = ('SOURCE_SUPPORTED', 'CONTRADICTED', 'NOT_ESTABLISHED',
            'NORMATIVE_ASSUMPTION', 'CONDITIONAL_PRECONDITION')


def audit_packet(source, state, conclusion_handle):
    """Audit the conclusion's support ancestry, including embedded conditions.

    Hide the framework, route explanations, prior reviews and preferred conclusion
    designation. Statement wording can still reveal an argument's direction.
    """
    if fingerprint(source['evidence']) != fingerprint(state['evidence']):
        raise ValueError('Audit source evidence changed')
    if source['world_sha256'] != state['world_sha256']:
        raise ValueError('Audit world changed')
    view = project(state)
    pending = [state['handles'][conclusion_handle]]
    relevant = set()
    while pending:
        sid = pending.pop()
        if sid in relevant:
            continue
        relevant.add(sid)
        pending.extend(state['statements'][sid]['hypothesis_ids'])
        for rid, route in state['routes'].items():
            if route['output'] == sid and view['support_routes'][rid]['status'] != 'WITHDRAWN':
                pending.extend(route['inputs'])
    claims = []
    for sid in sorted(relevant):
        row = state['statements'][sid]
        claims.append({'statement_id': sid, 'text': row['text'],
                       'polarity': row['polarity'], 'modality': row['modality']})
    return {'version': VERSION, 'source_sha256': fingerprint(source),
            'argument_sha256': fingerprint(state), 'evidence': deepcopy(source['evidence']),
            'claims': claims}


def audit_schema(packet=None):
    from argument_construction import object_schema
    text = {'type': 'string', 'minLength': 1}
    refs = [e['ref'] for e in packet['evidence']] if packet else []
    citation = object_schema({'evidence_id': {'type':'string','enum':refs} if refs else text,
                             'value_path': {'type': 'string'}, 'quote': text})
    component = object_schema({'claim_span': text, 'status': {'type': 'string', 'enum': list(STATUSES)},
        'evidence': {'type': 'array', 'items': citation}, 'reason': text})
    finding = object_schema({'statement_id': text,
        'components': {'type': 'array', 'items': component, 'minItems': 1},
        'source_preserving_reading': text})
    return object_schema({'findings': {'type': 'array', 'items': finding}})


AUDIT_INSTRUCTION = (
    'Audit only source applicability, independently of whether the ethical conclusion is attractive. '
    'For EVERY supplied statement, separate its decisive factual components and conditions from normative '
    'principles. Identify each claim_span component precisely; prefer exact substrings. Check each added qualifier, role, absence claim, mental '
    'state, quantifier, modality and necessity condition against the FULL source. SOURCE_SUPPORTED means '
    'the quoted source supports that precise component, not merely that a quote exists. CONTRADICTED '
    'needs contrary source support. Missing information is NOT_ESTABLISHED, never the opposite fact. '
    'NORMATIVE_ASSUMPTION identifies a moral principle needing argument, not a factual error. '
    'CONDITIONAL_PRECONDITION identifies a genuinely hypothetical factual antecedent not established in '
    'this case; it must remain a condition when the norm is applied. Do not let a moral principle hide '
    'unstated factual applicability. Cite exact source spans for source judgments. Preserve whatever '
    'weaker factual or conditional reading survives. evidence_id must be an existing evidence ref; '
    'value_path is the JSON pointer within its value (empty string for a whole text value). '
    'Do not recommend an action or decide duty priority.\n')


def _bind_quote(citation, evidence):
    """Resolve wrong transport aliases only when the exact quote has one location.

    Preserve the raw citation and how it was resolved. No approximate matching,
    source rewriting or inferred semantic endorsement is permitted.
    """
    try:
        return {**resolve_citation(citation, evidence), 'resolution_method':'DECLARED_LOCATION'}
    except ValueError as error:
        if not isinstance(citation, dict) or set(citation) != {'evidence_id','value_path','quote'}:
            raise
        quote = citation['quote']
        if not isinstance(quote, str) or not quote:
            raise
        matches = []
        def visit(value, path, ref):
            if isinstance(value, str) and quote in value:
                matches.append({'evidence_id':ref,'value_path':path,'quote':quote})
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, path+'/'+str(key).replace('~','~0').replace('/','~1'), ref)
            elif isinstance(value, list):
                for index, item in enumerate(value):visit(item,path+'/'+str(index),ref)
        for entry in evidence:visit(entry['value'],'',entry['ref'])
        if len(matches) != 1:
            raise ValueError('Invalid citation has no unique exact-quote location') from error
        binding=resolve_citation(matches[0],evidence)
        binding.update(raw=deepcopy(citation),resolution_method='UNIQUE_EXACT_QUOTE',
                       transport_issue=str(error))
        return binding


def bind_audit(packet, response):
    """Reject mismatched targets/quotes while leaving entailment explicitly reviewed."""
    if set(response) != {'findings'} or not isinstance(response['findings'], list):
        raise ValueError('Invalid premise audit')
    claims = {c['statement_id']: c for c in packet['claims']}
    seen = set()
    result = deepcopy(response)
    for finding in result['findings']:
        if set(finding) != {'statement_id', 'components', 'source_preserving_reading'}:
            raise ValueError('Invalid premise finding')
        sid = finding['statement_id']
        if sid not in claims or sid in seen:
            raise ValueError('Unknown or duplicate audited statement')
        seen.add(sid)
        if not finding['source_preserving_reading'] or not finding['components']:
            raise ValueError('Empty premise finding')
        for component in finding['components']:
            if set(component) != {'claim_span', 'status', 'evidence', 'reason'}:
                raise ValueError('Invalid premise component')
            if (not isinstance(component['claim_span'],str) or not component['claim_span']
                    or component['status'] not in STATUSES or not component['reason']):
                raise ValueError('Invalid audited claim span or status')
            component['claim_binding'] = ('EXACT_SUBSTRING' if component['claim_span'] in claims[sid]['text']
                                          else 'REVIEWER_PARAPHRASE_NOT_EXACT_SPAN')
            if component['status'] in ('SOURCE_SUPPORTED', 'CONTRADICTED') and not component['evidence']:
                raise ValueError('Source judgment needs exact evidence')
            component['citation_bindings'] = [_bind_quote(c, packet['evidence']) for c in component['evidence']]
    if seen != set(claims):
        raise ValueError('Premise audit omitted a target')
    return {**result, 'version': VERSION, 'source_sha256': packet['source_sha256'],
            'argument_sha256': packet['argument_sha256'],
            'review_authority': 'MODEL_SOURCE_REVIEW_NOT_INDEPENDENT_ORACLE'}


def apply_source_findings(state, audit):
    """Attribute reviews of existing SOURCE routes; do not rewrite any premise.

    Conditional preconditions and normative principles cannot become source facts.
    A later critic may disagree; both reviews remain in the event history.
    """
    if fingerprint(state) != audit['argument_sha256']:
        raise ValueError('Stale premise audit')
    result = deepcopy(state)
    for finding in audit['findings']:
        statuses = {c['status'] for c in finding['components']}
        status = 'SUPPORTED' if statuses == {'SOURCE_SUPPORTED'} else 'CHALLENGED'
        for rid, route in state['routes'].items():
            if route['operation'] == 'SOURCE' and route['output'] == finding['statement_id']:
                result = assess_support(result, rid, status,
                    reviewer=audit['review_authority'],
                    rationale='; '.join(c['claim_span'] + ': ' + c['status'] + ' — ' + c['reason']
                                        for c in finding['components']))
    return result
