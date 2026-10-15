"""Reversible Mermaid-only bundles. Never rewrite the accepted world."""
import re


def action_bundles(world):
    bundles = {}
    for action in world.get('actions', []):
        candidates = [e for e in world.get('effects', [])
                      if e.get('action_id') == action['action_id'] and e.get('directness') == 'DIRECT']
        if len(candidates) != 1:
            continue
        effect = candidates[0]
        if (effect.get('outcome') != action.get('intervention')
                or effect.get('polarity') != 'NEUTRAL' or effect.get('modality') != 'CERTAIN'
                or effect.get('effect_kind') not in {'INTERVENTION', 'RESOURCE_TRANSFER'}
                or any(effect.get(k) for k in ('condition_ids','likelihood_qualifiers',
                    'scope_qualifiers','temporal_qualifiers','overall_likelihood_qualifiers'))):
            continue
        bundles[action['action_id']] = [action['action_id'], effect['effect_id']]
    return bundles


def context_propositions(world):
    def strings(value):
        if isinstance(value,str):
            yield value
        elif isinstance(value,dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value,list):
            for child in value:
                yield from strings(child)
    referenced=set(strings({key:value for key,value in world.items() if key!='propositions'}))
    return [p['proposition_id'] for p in world.get('propositions',[])
            if p.get('status')=='ASSERTED' and p['proposition_id'] not in referenced]


def bundle_mermaid(source, bundles, context_ids=()):
    aliases = {ids[1]: ident for ident, ids in bundles.items()}
    result = []
    for line in source.splitlines():
        if any(re.match(r'^\s*'+re.escape(ident)+r'(?:\[|\(\[)',line) for ident in context_ids):
            continue
        if any(re.match(r'^\s*'+re.escape(ident)+r'\[', line) for ident in aliases):
            continue
        # Only graph identifier positions are changed. Labels and source text
        # containing words resembling an ID stay untouched.
        edge = re.match(r'^(\s*)(\w+)(\s+(?:-->|-\.->)\|.*\|\s+)(\w+)\s*$', line)
        if edge:
            if edge[2] in context_ids or edge[4] in context_ids:
                continue
            left = aliases.get(edge[2], edge[2]); right = aliases.get(edge[4], edge[4])
            if left == right and edge[2] != edge[4]:
                continue
            line = edge[1]+left+edge[3]+right
        result.append(line)
    return '\n'.join(result)
