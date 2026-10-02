"""Z9: expose modal/content negation ambiguity in stripping reconstructions."""
import parsing_game_Z5 as z5
import parsing_game_Z6 as z6
import parsing_game_Z7 as z7
import parsing_game_Z8 as z8


def mark_reconstruction_scope_gaps(package):
    for reconstruction in package['reconstructions']:
        prop = reconstruction['proposition_id']
        local = [c for c in package['candidates'] if c['arguments'].get('proposition') == prop]
        modals = [c for c in local if c['type'] == 'MODALITY']
        if not modals:
            continue
        affected = [c for c in local if c['type'] in {'PREDICATION', 'PARTICIPANT', 'MODALITY'}]
        evidence_ids = list(dict.fromkeys(reconstruction['evidence_ids'] +
                                         [e for c in modals for e in c['evidence_ids']]))
        for candidate in affected:
            # Neither NOT MODAL(P) nor MODAL(NOT P) is selected by exporting
            # the inherited modal and the negative remnant together.
            candidate['scope']['polarity'] = 'unresolved'
            candidate['assessment']['status'] = 'unresolved'
        package['open_questions'].append(dict(
            id=z6.next_question_id(package), kind='scope', evidence_ids=evidence_ids,
            candidate_ids=[c['id'] for c in affected],
            question='Resolve negation of the modal versus negation of its content in this '
                     'ellipsis reconstruction: NOT MODAL(P) versus MODAL(NOT P). '
                     'For an ability reading, inability to perform the reconstructed action '
                     'differs from ability to refrain from it. No operator order is selected.',
            blocking_for=[c['id'] for c in affected]))


def export_candidate_graph(text, *, package_id=None):
    package = z5.export_candidate_graph(text, package_id=package_id)
    z8.separate_reconstructions(package)
    mark_reconstruction_scope_gaps(package)
    z6.preserve_condition_content(package)
    z7.add_condition_contents(package)
    package['schema_version'] = '0.4'
    package['producer'].update(name='parsing_game_Z9', version='Z9')
    return package
