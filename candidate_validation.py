"""Pure contract checks. Input package is trusted caller-supplied evidence, not LLM output.

No parser, model, network, or world-state writes. Identity consistency means
consistency with explicit package constraints, not proof of semantic identity.
"""
import math


VOCABULARY = {
    'PREDICATION': ({'proposition': 'proposition'}, [None]),
    'PARTICIPANT': ({'proposition': 'proposition', 'mention': 'mention'},
                    ['subject', 'object', 'agent', 'patient', 'destination', 'location', 'controller']),
    'SAME_REFERENT': ({'mention_a': 'mention', 'mention_b': 'mention'}, [None]),
    'EVENT_LINK': ({'parent': 'proposition', 'child': 'proposition'}, ['complement', 'purpose', 'attempt', 'unresolved']),
    'MODALITY': ({'proposition': 'proposition'}, ['prediction', 'possibility', 'ability', 'permission', 'obligation', 'unresolved']),
    'CONDITIONAL_ON': ({'consequence': 'proposition', 'condition': 'proposition'}, [None]),
    'OPTION_OF': ({'proposition': 'proposition', 'choice_point': 'choice_point'}, [None]),
    'QUANTITY': ({'mention': 'mention'}, None),
}


def empty_selection(package):
    """Explicitly leave every question unanswered; no default graph selection."""
    return dict(schema_version=package['schema_version'], package_id=package['package_id'], selected_node_ids=[],
                selected_candidate_ids=[], extensions=[], question_resolutions=[
                    dict(question_id=q['id'], status='unresolved', selected_candidate_ids=[], rationale='')
                    for q in package['open_questions']])


def validate_candidate_selection(package, selection):
    result = dict(contract_valid=False, errors=[], unresolved_question_ids=[],
                  provisional_candidate_ids=[], unverified_extension_ids=[], identity_components=[],
                  semantic_support='not_assessed', world_state_commitment='not_authorized')
    errors = result['errors']

    def fail(code, path):
        errors.append(dict(code=code, path=path))

    def record(obj, required, path, optional=()):
        if not isinstance(obj, dict):
            fail('expected_object', path)
            return False
        if set(required) - obj.keys():
            fail('missing_fields', path)
        if obj.keys() - set(required) - set(optional):
            fail('unexpected_fields', path)
        return not (set(required) - obj.keys())

    def string(value, path):
        if not isinstance(value, str) or not value:
            fail('expected_nonempty_string', path)

    def ids(values, path, nonempty=False):
        if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
            fail('expected_id_list', path)
            return False
        if len(set(values)) != len(values) or (nonempty and not values):
            fail('duplicate_or_missing_ids', path)
        return True

    def evidence_refs(obj, path):
        ids(obj['evidence_ids'], path + '.evidence_ids', True)

    def provenance(values, path):
        if not isinstance(values, list) or not values:
            fail('expected_provenance', path)
            return
        for item in values:
            if record(item, ['producer', 'version', 'method', 'resource_ids'], path):
                for key in ('producer', 'version', 'method'):
                    string(item[key], path + '.' + key)
                ids(item['resource_ids'], path + '.resource_ids')

    envelope = ['schema_version', 'package_id', 'document', 'producer', 'evidence', 'nodes',
                'candidates', 'choice_sets', 'open_questions', 'coverage']
    if not record(package, envelope, 'package', ['identity_constraints', 'condition_contents', 'reconstructions']):
        return result
    if not record(selection, ['schema_version', 'package_id', 'selected_node_ids',
                             'selected_candidate_ids', 'question_resolutions', 'extensions'], 'selection'):
        return result
    for name, obj in [('package', package), ('selection', selection)]:
        if obj['schema_version'] not in ['0.1', '0.2', '0.3', '0.4']:
            fail('unsupported_schema_version', name)
        string(obj['package_id'], name + '.package_id')
    if package['package_id'] != selection['package_id']:
        fail('package_mismatch', 'selection.package_id')
    if package['schema_version'] != selection['schema_version']:
        fail('schema_version_mismatch', 'selection.schema_version')
    if package['schema_version'] in ['0.3', '0.4'] and 'condition_contents' not in package:
        fail('missing_condition_contents', 'package')
    if package['schema_version'] not in ['0.3', '0.4'] and 'condition_contents' in package:
        fail('condition_contents_requires_schema_0.3', 'package')
    if package['schema_version'] == '0.4' and 'reconstructions' not in package:
        fail('missing_reconstructions', 'package')
    if package['schema_version'] != '0.4' and 'reconstructions' in package:
        fail('reconstructions_requires_schema_0.4', 'package')
    if record(package['document'], ['id', 'text'], 'document'):
        string(package['document']['id'], 'document.id')
        if not isinstance(package['document']['text'], str):
            fail('expected_string', 'document.text')
    producer = package['producer']
    if record(producer, ['name', 'version', 'resources'], 'producer'):
        string(producer['name'], 'producer.name')
        string(producer['version'], 'producer.version')
        if not isinstance(producer['resources'], list):
            fail('expected_list', 'producer.resources')
        else:
            for r in producer['resources']:
                if record(r, ['id', 'version', 'description'], 'resource'):
                    for key in r:
                        string(r[key], 'resource.' + key)
    coverage = package['coverage']
    if record(coverage, ['status', 'unrepresented_evidence_ids', 'limitations'], 'coverage'):
        if coverage['status'] != 'partial':
            fail('unsupported_coverage_status', 'coverage.status')
        ids(coverage['unrepresented_evidence_ids'], 'coverage.unrepresented_evidence_ids')
        ids(coverage['limitations'], 'coverage.limitations')
    for name in ['evidence', 'nodes', 'candidates', 'choice_sets', 'open_questions', 'identity_constraints', 'condition_contents', 'reconstructions']:
        if not isinstance(package.get(name, []), list):
            fail('expected_list', name)
    for name in ['question_resolutions', 'extensions']:
        if not isinstance(selection[name], list):
            fail('expected_list', 'selection.' + name)
    for name in ['selected_node_ids', 'selected_candidate_ids']:
        ids(selection[name], 'selection.' + name)
    if errors:
        return result

    tables, seen = {}, set()
    for name in ['evidence', 'nodes', 'candidates', 'choice_sets', 'open_questions', 'identity_constraints', 'condition_contents', 'reconstructions']:
        tables[name] = {}
        for obj in package.get(name, []):
            if not isinstance(obj, dict) or not isinstance(obj.get('id'), str) or not obj['id']:
                fail('missing_id', name)
                continue
            if obj['id'] in seen:
                fail('duplicate_id', obj['id'])
            seen.add(obj['id'])
            tables[name][obj['id']] = obj
    for e in tables['evidence'].values():
        if record(e, ['id', 'start', 'end', 'text'], e['id']):
            start, end = e['start'], e['end']
            if (type(start) is not int or type(end) is not int or
                not 0 <= start < end <= len(package['document']['text']) or
                package['document']['text'][start:end] != e['text']):
                fail('invalid_evidence_span', e['id'])
    for n in tables['nodes'].values():
        required = ['id', 'kind', 'label', 'evidence_ids']
        if n.get('kind') == 'proposition':
            required += ['predicate', 'predicate_evidence_ids']
        if record(n, required, n['id']):
            if n['kind'] not in ['mention', 'proposition', 'choice_point']:
                fail('invalid_node_kind', n['id'])
            string(n['label'], n['id'])
            evidence_refs(n, n['id'])
            if n['kind'] == 'proposition':
                string(n['predicate'], n['id'])
                ids(n['predicate_evidence_ids'], n['id'], True)
    for c in tables['candidates'].values():
        path = c['id']
        if not record(c, ['id', 'type', 'arguments', 'value', 'evidence_ids', 'scope',
                         'provenance', 'assessment', 'requires', 'exclusive_with'], path):
            continue
        evidence_refs(c, path)
        ids(c['requires'], path + '.requires')
        ids(c['exclusive_with'], path + '.exclusive_with')
        provenance(c['provenance'], path)
        kind = c['type']
        if not isinstance(kind, str) or kind not in VOCABULARY:
            fail('unknown_candidate_type', path)
        else:
            roles, values = VOCABULARY[kind]
            if record(c['arguments'], roles, path + '.arguments'):
                for role in roles:
                    string(c['arguments'][role], path + '.' + role)
            if values is not None and c['value'] not in values:
                fail('invalid_candidate_value', path)
            if kind == 'QUANTITY' and record(c['value'], ['operator', 'amount', 'unit'], path + '.value'):
                if c['value']['operator'] != 'exact' or type(c['value']['amount']) is not int or c['value']['amount'] < 0:
                    fail('invalid_quantity', path)
                if c['value']['unit'] is not None:
                    string(c['value']['unit'], path)
        if record(c['scope'], ['polarity', 'contexts'], path + '.scope'):
            if c['scope']['polarity'] not in ['positive', 'negative', 'unresolved']:
                fail('invalid_polarity', path)
            if not isinstance(c['scope']['contexts'], list):
                fail('expected_context_list', path)
            else:
                for ctx in c['scope']['contexts']:
                    kinds = {'conditional': ['condition_proposition_id'], 'hypothetical': [],
                             'attributed': ['source_mention_id'], 'questioned': [], 'modal': ['modality_candidate_id'],
                             'modal_choice': ['modality_choice_set_id']}
                    if not isinstance(ctx, dict) or not isinstance(ctx.get('kind'), str) or ctx['kind'] not in kinds:
                        fail('invalid_context', path)
                        continue
                    if record(ctx, ['kind', 'evidence_ids'] + kinds[ctx['kind']], path,
                              ['report_proposition_id'] if ctx['kind'] == 'attributed' else []):
                        evidence_refs(ctx, path)
                        if package['schema_version'] == '0.1' and (ctx['kind'] == 'modal_choice' or 'report_proposition_id' in ctx):
                            fail('scope_requires_schema_0.2', path)
                        if 'report_proposition_id' in ctx:
                            string(ctx['report_proposition_id'], path)
                        for key in kinds[ctx['kind']]:
                            if key != 'source_mention_id' or ctx[key] is not None:
                                string(ctx[key], path)
        if record(c['assessment'], ['status', 'score'], path + '.assessment'):
            if c['assessment']['status'] not in ['proposed', 'preferred', 'unresolved']:
                fail('invalid_assessment', path)
            score = c['assessment']['score']
            if score is not None and record(score, ['value', 'kind', 'source', 'calibration_id'], path):
                value = score['value']
                if type(value) not in (int, float) or not math.isfinite(value):
                    fail('invalid_score', path)
                elif score['kind'] in ['uncalibrated_probability', 'calibrated_probability'] and not 0 <= value <= 1:
                    fail('invalid_probability', path)
                if score['kind'] not in ['uncalibrated_score', 'uncalibrated_probability', 'calibrated_probability']:
                    fail('invalid_score_kind', path)
                string(score['source'], path)
                if score['calibration_id'] is not None:
                    string(score['calibration_id'], path)
    for group in tables['choice_sets'].values():
        if record(group, ['id', 'kind', 'candidate_ids', 'selection_rule', 'exhaustive'], group['id'], ['evidence_ids']):
            ids(group['candidate_ids'], group['id'])
            if 'evidence_ids' in group:
                evidence_refs(group, group['id'])
            elif group['kind'] == 'scenario_option' and group['selection_rule'] == 'at_most_one':
                fail('missing_exclusivity_evidence', group['id'])
            if group['kind'] not in ['interpretation', 'scenario_option'] or group['selection_rule'] not in ['any_subset', 'at_most_one'] or type(group['exhaustive']) is not bool:
                fail('invalid_choice_set', group['id'])
    for q in tables['open_questions'].values():
        if record(q, ['id', 'kind', 'evidence_ids', 'candidate_ids', 'question', 'blocking_for'], q['id']):
            evidence_refs(q, q['id'])
            ids(q['candidate_ids'], q['id'])
            ids(q['blocking_for'], q['id'])
            string(q['question'], q['id'])
            if q['kind'] not in ['attachment', 'reference', 'scope', 'modality', 'missing_representation', 'unsupported_semantics']:
                fail('invalid_question_kind', q['id'])
    for restriction in tables['identity_constraints'].values():
        if record(restriction, ['id', 'mention_a', 'mention_b', 'evidence_ids', 'provenance'], restriction['id']):
            string(restriction['mention_a'], restriction['id'])
            string(restriction['mention_b'], restriction['id'])
            evidence_refs(restriction, restriction['id'])
            provenance(restriction['provenance'], restriction['id'])
    for bundle in tables['condition_contents'].values():
        if record(bundle, ['id', 'condition_proposition_id', 'required_candidate_ids', 'question_ids', 'evidence_ids'], bundle['id']):
            string(bundle['condition_proposition_id'], bundle['id'])
            ids(bundle['required_candidate_ids'], bundle['id'], True)
            ids(bundle['question_ids'], bundle['id'])
            evidence_refs(bundle, bundle['id'])
    for reconstruction in tables['reconstructions'].values():
        fields = ['id', 'proposition_id', 'antecedent_proposition_id', 'antecedent_candidate_id',
                  'predication_candidate_id', 'participant_candidate_ids', 'evidence_ids', 'method']
        if record(reconstruction, fields, reconstruction['id']):
            for key in ['proposition_id', 'antecedent_proposition_id', 'antecedent_candidate_id', 'predication_candidate_id']:
                string(reconstruction[key], reconstruction['id'])
            ids(reconstruction['participant_candidate_ids'], reconstruction['id'])
            evidence_refs(reconstruction, reconstruction['id'])
            if reconstruction['method'] != 'stripping_not_nominal':
                fail('unknown_reconstruction_method', reconstruction['id'])
    for resolution in selection['question_resolutions']:
        if record(resolution, ['question_id', 'status', 'selected_candidate_ids', 'rationale'], 'resolution'):
            string(resolution['question_id'], 'resolution')
            ids(resolution['selected_candidate_ids'], 'resolution')
            if resolution['status'] not in ['resolved_by_selection', 'unresolved'] or not isinstance(resolution['rationale'], str):
                fail('invalid_resolution', 'resolution')
    for extension in selection['extensions']:
        if record(extension, ['id', 'origin', 'evidence_ids', 'description', 'verification_status'], 'extension'):
            string(extension['id'], 'extension')
            evidence_refs(extension, 'extension')
            string(extension['description'], 'extension')
            if extension['origin'] != 'llm_inferred' or extension['verification_status'] != 'unverified':
                fail('unverified_extension_required', 'extension')
    if errors:
        return result

    nodes, candidates, questions = tables['nodes'], tables['candidates'], tables['open_questions']
    resources = {r['id'] for r in producer['resources']}
    if len(resources) != len(producer['resources']):
        fail('duplicate_resource_id', 'producer.resources')

    def refs(values, table, path):
        for ident in values:
            if ident not in table:
                fail('dangling_reference', path + ':' + ident)

    def node_ref(ident, kind, path):
        if ident not in nodes or nodes[ident]['kind'] != kind:
            fail('invalid_endpoint', path)

    for name, table in tables.items():
        for obj in table.values():
            if 'evidence_ids' in obj:
                refs(obj['evidence_ids'], tables['evidence'], obj['id'])
            for source in obj.get('provenance', []):
                refs(source['resource_ids'], resources, obj['id'])
    refs(coverage['unrepresented_evidence_ids'], tables['evidence'], 'coverage')
    for n in nodes.values():
        if n['kind'] == 'proposition' and not set(n['predicate_evidence_ids']) <= set(n['evidence_ids']):
            fail('predicate_evidence_not_in_node', n['id'])
    anchors = {}
    for c in candidates.values():
        if c['type'] == 'PREDICATION':
            anchors.setdefault(c['arguments']['proposition'], set()).add(c['id'])
    for c in candidates.values():
        ident = c['id']
        for role, kind in VOCABULARY[c['type']][0].items():
            endpoint = c['arguments'][role]
            node_ref(endpoint, kind, ident)
            if kind == 'proposition' and c['type'] != 'PREDICATION' and not anchors.get(endpoint, set()).intersection(c['requires']):
                fail('missing_predication_dependency', ident)
        refs(c['requires'], candidates, ident)
        refs(c['exclusive_with'], candidates, ident)
        for other in c['exclusive_with']:
            if other == ident or (other in candidates and ident not in candidates[other]['exclusive_with']):
                fail('invalid_exclusivity', ident)
        for ctx in c['scope']['contexts']:
            refs(ctx['evidence_ids'], tables['evidence'], ident)
            if ctx.get('report_proposition_id'):
                node_ref(ctx['report_proposition_id'], 'proposition', ident)
            if ctx['kind'] == 'conditional':
                endpoint = ctx['condition_proposition_id']
                node_ref(endpoint, 'proposition', ident)
            elif ctx['kind'] == 'attributed' and ctx['source_mention_id'] is not None:
                node_ref(ctx['source_mention_id'], 'mention', ident)
            elif ctx['kind'] == 'modal':
                mod = candidates.get(ctx['modality_candidate_id'])
                if not mod or mod['type'] != 'MODALITY':
                    fail('invalid_modal_reference', ident)
            elif ctx['kind'] == 'modal_choice':
                group = tables['choice_sets'].get(ctx['modality_choice_set_id'])
                if (not group or group['kind'] != 'interpretation' or
                    group['selection_rule'] != 'at_most_one' or not group['candidate_ids'] or
                    any(x not in candidates or candidates[x]['type'] != 'MODALITY'
                        for x in group['candidate_ids'])):
                    fail('invalid_modal_choice_reference', ident)
                elif not any(q['kind'] == 'modality' and
                             set(group['candidate_ids']) <= set(q['candidate_ids'])
                             for q in questions.values()):
                    fail('modal_choice_without_question', ident)
                elif any(candidates[x]['arguments']['proposition'] not in c['arguments'].values()
                         for x in group['candidate_ids']):
                    fail('modal_choice_proposition_mismatch', ident)
        score = c['assessment']['score']
        if score and score['kind'] == 'calibrated_probability' and score['calibration_id'] not in resources:
            fail('missing_calibration_artifact', ident)
        if (c['assessment']['status'] == 'unresolved' or c['scope']['polarity'] == 'unresolved') and not any(ident in q['candidate_ids'] for q in questions.values()):
            fail('unresolved_without_question', ident)
    for group in tables['choice_sets'].values():
        refs(group['candidate_ids'], candidates, group['id'])
    for q in questions.values():
        refs(q['candidate_ids'] + q['blocking_for'], candidates, q['id'])
    condition_bundles = {}
    for bundle in tables['condition_contents'].values():
        condition = bundle['condition_proposition_id']
        node_ref(condition, 'proposition', bundle['id'])
        refs(bundle['required_candidate_ids'], candidates, bundle['id'])
        refs(bundle['question_ids'], questions, bundle['id'])
        if condition in condition_bundles:
            fail('duplicate_condition_content', bundle['id'])
        condition_bundles[condition] = bundle
        if not anchors.get(condition, set()).intersection(bundle['required_candidate_ids']):
            fail('missing_condition_content_anchor', bundle['id'])
    def carried_conditions(candidate):
        found = {ctx['condition_proposition_id'] for ctx in candidate['scope']['contexts'] if ctx['kind'] == 'conditional'}
        if candidate['type'] == 'CONDITIONAL_ON':
            found.add(candidate['arguments']['condition'])
        return found
    if package['schema_version'] in ['0.3', '0.4']:
        for c in candidates.values():
            if carried_conditions(c) - condition_bundles.keys():
                fail('missing_condition_content', c['id'])
    reconstructed = set()
    antecedents = {}
    for r in tables['reconstructions'].values():
        prop = r['proposition_id']
        source = r['antecedent_proposition_id']
        node_ref(prop, 'proposition', r['id'])
        node_ref(source, 'proposition', r['id'])
        if prop == source or prop in reconstructed:
            fail('invalid_reconstruction_identity', r['id'])
        reconstructed.add(prop)
        antecedents[prop] = source
        for key, endpoint in [('predication_candidate_id', prop), ('antecedent_candidate_id', source)]:
            c = candidates.get(r[key])
            if not c or c['type'] != 'PREDICATION' or c['arguments']['proposition'] != endpoint:
                fail('invalid_reconstruction_anchor', r['id'])
        expected_roles = {c['id'] for c in candidates.values() if c['type'] == 'PARTICIPANT' and c['arguments']['proposition'] == prop}
        if expected_roles != set(r['participant_candidate_ids']):
            fail('invalid_reconstruction_participants', r['id'])
        for ident in expected_roles:
            if r['predication_candidate_id'] not in candidates[ident]['requires']:
                fail('missing_reconstruction_dependency', ident)
    for prop in antecedents:
        visited, current = set(), prop
        while current in antecedents:
            if current in visited:
                fail('reconstruction_provenance_cycle', prop)
                break
            visited.add(current)
            current = antecedents[current]
    for restriction in tables['identity_constraints'].values():
        for key in ['mention_a', 'mention_b']:
            node_ref(restriction[key], 'mention', restriction['id'])
        if restriction['mention_a'] == restriction['mention_b']:
            fail('self_identity_constraint', restriction['id'])
    if errors:
        return result

    # Iterative dependency closure avoids recursion limits on long documents.
    closures = {}
    for ident in candidates:
        found, pending = set(), list(candidates[ident]['requires'])
        while pending:
            other = pending.pop()
            if other == ident:
                fail('dependency_cycle', ident)
                break
            if other not in found:
                found.add(other)
                pending.extend(candidates[other]['requires'])
        closures[ident] = found
    for ident, c in candidates.items():
        for ctx in c['scope']['contexts']:
            if ctx.get('report_proposition_id') and not anchors.get(ctx['report_proposition_id'], set()).intersection(closures[ident]):
                fail('missing_report_dependency', ident)
            if ctx['kind'] == 'conditional' and not anchors.get(ctx['condition_proposition_id'], set()).intersection(closures[ident]):
                fail('missing_condition_dependency', ident)
            if ctx['kind'] == 'modal' and ctx['modality_candidate_id'] not in closures[ident]:
                fail('missing_modal_dependency', ident)

    selected = set(selection['selected_candidate_ids'])
    selected_nodes = set(selection['selected_node_ids'])
    refs(selected, candidates, 'selection.candidates')
    refs(selected_nodes, nodes, 'selection.nodes')
    for ident in selected.intersection(candidates):
        c = candidates[ident]
        if not set(c['requires']) <= selected:
            fail('unselected_dependency', ident)
        if set(c['exclusive_with']).intersection(selected):
            fail('exclusive_selection', ident)
        required_nodes = set(c['arguments'].values())
        for ctx in c['scope']['contexts']:
            for key in ['condition_proposition_id', 'source_mention_id', 'report_proposition_id']:
                if ctx.get(key):
                    required_nodes.add(ctx[key])
        if not required_nodes <= selected_nodes:
            fail('unselected_endpoint', ident)
    for group in tables['choice_sets'].values():
        if group['selection_rule'] == 'at_most_one' and len(selected.intersection(group['candidate_ids'])) > 1:
            fail('choice_selection_limit', group['id'])
    resolutions = {}
    for r in selection['question_resolutions']:
        ident = r['question_id']
        if ident in resolutions or ident not in questions:
            fail('duplicate_or_unknown_question', ident)
            continue
        resolutions[ident] = r
        chosen = set(r['selected_candidate_ids'])
        if r['status'] == 'resolved_by_selection':
            if not chosen or not chosen <= selected or not chosen <= set(questions[ident]['candidate_ids']):
                fail('unsupported_question_resolution', ident)
            elif any(candidates[c]['value'] == 'unresolved' or candidates[c]['scope']['polarity'] == 'unresolved' for c in chosen):
                fail('unresolved_candidate_is_not_resolution', ident)
        elif chosen:
            fail('unresolved_question_has_resolution_ids', ident)
    if set(resolutions) != set(questions):
        fail('missing_question_resolutions', 'selection')
    unresolved = [ident for ident in questions if ident not in resolutions or resolutions[ident]['status'] == 'unresolved']
    result['unresolved_question_ids'] = unresolved
    blocked = {c for ident in unresolved
               for c in questions[ident]['blocking_for'] + questions[ident]['candidate_ids']}
    for ident, c in candidates.items():
        for ctx in c['scope']['contexts']:
            if ctx['kind'] == 'modal_choice':
                members = set(tables['choice_sets'][ctx['modality_choice_set_id']]['candidate_ids'])
                if any(members <= set(questions[q]['candidate_ids']) for q in unresolved):
                    blocked.add(ident)
    # Incomplete condition content is not an occurrence claim and remains open.
    # Iterate because an outer condition can itself depend on a scoped condition.
    changed = True
    while changed:
        changed = False
        for ident, c in candidates.items():
            if ident in blocked:
                continue
            for condition in carried_conditions(c):
                bundle = condition_bundles.get(condition)
                if not bundle:
                    continue  # Legacy schemas retain their existing behavior.
                required = set(bundle['required_candidate_ids'])
                if (not required <= selected or set(bundle['question_ids']).intersection(unresolved) or
                    required.intersection(blocked) or any(closures[r].intersection(blocked) for r in required)):
                    blocked.add(ident)
                    changed = True
                    break
    result['provisional_candidate_ids'] = sorted(ident for ident in selected.intersection(candidates)
        if ident in blocked or closures[ident].intersection(blocked) or any(ident in questions[q]['candidate_ids'] for q in unresolved))
    extension_ids = set()
    for extension in selection['extensions']:
        ident = extension['id']
        if ident in seen or ident in extension_ids:
            fail('duplicate_extension_id', ident)
        extension_ids.add(ident)
        refs(extension['evidence_ids'], tables['evidence'], ident)
    result['unverified_extension_ids'] = sorted(extension_ids)

    # Equivalence closure is diagnostic only; it does not mutate mention nodes.
    parent = {ident: ident for ident in selected_nodes if ident in nodes and nodes[ident]['kind'] == 'mention'}

    def root(ident):
        while parent[ident] != ident:
            parent[ident] = parent[parent[ident]]
            ident = parent[ident]
        return ident

    for ident in selected.intersection(candidates):
        c = candidates[ident]
        if c['type'] == 'SAME_REFERENT':
            a, b = c['arguments']['mention_a'], c['arguments']['mention_b']
            if a in parent and b in parent:
                parent[root(a)] = root(b)
    components = {}
    for ident in parent:
        components.setdefault(root(ident), []).append(ident)
    result['identity_components'] = sorted(sorted(group) for group in components.values())
    for restriction in tables['identity_constraints'].values():
        a, b = restriction['mention_a'], restriction['mention_b']
        if a in parent and b in parent and root(a) == root(b):
            fail('identity_constraint_violated', restriction['id'])
    result['contract_valid'] = not errors
    return result
