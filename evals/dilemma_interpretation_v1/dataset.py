"""Local data integrity checks and gold-free inputs; never invokes a parser/model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CONSTRUCTIONS = {'stripping', 'verb_phrase_ellipsis', 'gapping', 'sluicing',
                'noun_phrase_ellipsis', 'comparative_ellipsis', 'fragment_answer',
                'coreference', 'truncation', 'nonellipsis_control'}
CHECK_KINDS = {'scope', 'roles', 'reference', 'quantity', 'occurrence', 'alternatives', 'missing_content'}


def load_cases(split='dev', root=ROOT):
    if split not in {'dev', 'heldout'}:
        raise ValueError('Choose dev or explicitly opt into heldout.')
    return [json.loads(line) for line in (root / (split + '.jsonl')).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cases(cases):
    errors = []
    seen = set()
    groups = {s: {k: set() for k in ('scenario_family', 'template_family', 'text')} for s in ('dev', 'heldout')}
    for case in cases:
        ident = case.get('id', '<missing>')
        def require(condition, message):
            if not condition:
                errors.append(ident + ': ' + message)
        require(ident not in seen, 'duplicate case ID')
        seen.add(ident)
        split = case.get('split')
        require(split in groups, 'invalid split')
        if split not in groups:
            continue
        for field in groups[split]:
            require(isinstance(case.get(field), str) and bool(case.get(field)), 'missing ' + field)
            groups[split][field].add(case.get(field))
        require(case.get('construction') in CONSTRUCTIONS, 'unknown construction')
        require(case.get('annotation_status') in {'draft_review_required', 'adjudicated'}, 'annotation status missing')
        require(case.get('annotation_coverage') == 'targeted_not_exhaustive', 'coverage must be explicit')
        evidence = case.get('evidence', [])
        evidence_ids = {e['id'] for e in evidence}
        require(len(evidence_ids) == len(evidence) and bool(evidence), 'missing/duplicate evidence IDs')
        for e in evidence:
            require(isinstance(e['start'], int) and isinstance(e['end'], int)
                    and 0 <= e['start'] < e['end'] <= len(case['text']), 'invalid evidence offsets')
            require(case['text'][e['start']:e['end']] == e['text'], 'evidence text mismatch')
        gold = case['gold']
        readings = gold['readings']
        reading_ids = {r['id'] for r in readings}
        require(len(reading_ids) == len(readings), 'duplicate reading IDs')
        expected_policy = ('abstain' if not readings else 'alternatives_nonexhaustive'
                           if len(readings) > 1 else 'single_target_reading')
        require(gold['reading_policy'] == expected_policy, 'inconsistent reading policy')
        require(bool(readings) or bool(gold['open_questions']), 'abstention needs a specific question')
        require(bool(gold['checks']) and bool(gold['forbidden_inferences']), 'missing semantic checks')
        require(bool(gold['downstream_probes']), 'missing downstream probe')
        for collection in (readings, gold['checks'], gold['downstream_probes']):
            require(len({item['id'] for item in collection}) == len(collection), 'duplicate annotation IDs')
        for item in readings + gold['checks']:
            require(bool(item['evidence_ids']) and set(item['evidence_ids']) <= evidence_ids, 'dangling evidence reference')
        for check in gold['checks']:
            require(check['kind'] in CHECK_KINDS and bool(check['requirement']), 'invalid semantic check')
        for probe in gold['downstream_probes']:
            require(set(probe['required_reading_ids']) <= reading_ids, 'dangling probe reading reference')
            require(bool(probe['question']) and bool(probe['acceptable_answer']), 'empty downstream probe')
            require(probe['answer_mode'] in {'across_alternatives', 'scope_sensitive'}, 'invalid probe mode')
    for field in ('scenario_family', 'template_family', 'text'):
        overlap = groups['dev'][field] & groups['heldout'][field]
        if overlap:
            errors.append('cross-split ' + field + ' overlap: ' + repr(sorted(overlap)))
    return errors


def validate(root=ROOT, repo=REPO):
    manifest = json.loads((root / 'manifest.json').read_text())
    cases = load_cases('dev', root) + load_cases('heldout', root)
    errors = validate_cases(cases)
    for split in ('dev', 'heldout'):
        actual = sum(c['split'] == split for c in cases)
        if actual != manifest['splits'][split]['cases']:
            errors.append(split + ': case count mismatch')
        if digest(root / (split + '.jsonl')) != manifest['splits'][split]['sha256']:
            errors.append(split + ': dataset digest differs from frozen draft')
    for name, expected in manifest['baseline']['files'].items():
        path = repo / name
        if not path.is_file() or digest(path) != expected:
            errors.append('Z10 baseline changed: ' + name)
    return dict(valid=not errors, errors=errors,
                split_counts=dict(Counter(c['split'] for c in cases)),
                construction_counts={s: dict(Counter(c['construction'] for c in cases if c['split'] == s))
                                     for s in ('dev', 'heldout')},
                semantic_gold_review='required; integrity validation does not adjudicate meaning')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['dev', 'heldout'], default='dev')
    parser.add_argument('--export-inputs', type=Path, help='Write ONLY id/text; no gold or construction labels.')
    args = parser.parse_args()
    result = validate()
    print(json.dumps(result, indent=2))
    if not result['valid']:
        raise SystemExit(1)
    if args.export_inputs:
        protected = {ROOT / name for name in ('dev.jsonl', 'heldout.jsonl', 'manifest.json', 'dataset.py', 'README.md')}
        if args.export_inputs.resolve() in protected:
            parser.error('Cannot overwrite dataset artifacts with exported inputs.')
        args.export_inputs.write_text(''.join(json.dumps({k: c[k] for k in ('id', 'text')}, ensure_ascii=False) + '\n'
                                            for c in load_cases(args.split)))


if __name__ == '__main__':
    main()
