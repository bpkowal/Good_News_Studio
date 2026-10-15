"""Write Decisions-API topography onto a saved demo report. Viewer stays no-API."""
import argparse
import json
from pathlib import Path

from parliament_decisions_topography import (
    JACKKNIFE_FOLDS, apply_to_report, enrich_jackknife_report, topography_from_decisions,
)
from run_blueprint_parliament import DEFAULT_OPENAI_ENV, PROJECT_ROOT, _load_environment


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--answers', type=Path, help='Replay saved answers instead of calling the API')
    parser.add_argument('--openai-env', type=Path, default=DEFAULT_OPENAI_ENV)
    parser.add_argument('--jackknife', action='store_true',
                        help='10-fold packet-text ablation intervals for each recorded action')
    parser.add_argument('--annotate-hinges', action='store_true',
                        help='Store omitted packet tenths and hinging ranks on existing jackknife topography (no API)')
    parser.add_argument('--write', action='store_true', help='Overwrite the saved DEMO_REPORT topography')
    parser.add_argument('--checkpoint-tag', action='append', help='Run label, inherited by child runs; repeatable')
    args = parser.parse_args(argv)
    from parliament_run_provenance import current_provenance, record_checkpoint, record_configuration
    from parliament_decisions_topography import DECISIONS_MODEL
    record_configuration(args, decisions_model=DECISIONS_MODEL)
    report = json.loads(args.report.read_text())
    if args.annotate_hinges:
        report = enrich_jackknife_report(report, PROJECT_ROOT)
    elif args.answers:
        payload = json.loads(args.answers.read_text())
        report['topography'] = topography_from_decisions(
            payload['actions'], payload['per_action_points'], native=report.get('topography'))
    else:
        environment = _load_environment(args.openai_env) if args.openai_env.exists() else dict(__import__('os').environ)
        key = environment.get('OPENAI_API_KEY')
        if not key:
            parser.error('OPENAI_API_KEY is absent; pass --answers or an env file.')
        report = apply_to_report(report, PROJECT_ROOT, key,
                                 jackknife_folds=JACKKNIFE_FOLDS if args.jackknife else 0)
    if args.write:
        report.setdefault('presentation_revisions', []).append(current_provenance())
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if args.write:record_checkpoint('report_revised',artifacts=[args.report])
    topo = report.get('topography') or {}
    print(topo.get('status'), args.report)
    for row in topo.get('actions') or []:
        bits = []
        for axis in ('care', 'welfare', 'obligation'):
            point = row.get(axis) or {}
            if not point.get('present'):
                bits.append(axis + '=omit')
                continue
            bit = axis + '=' + ('%.3f' % point['probability'])
            interval = point.get('interval') or {}
            if isinstance(interval.get('low'), (int, float)) and isinstance(interval.get('high'), (int, float)):
                bit += ' [%.3f, %.3f]' % (interval['low'], interval['high'])
            hinge = interval.get('hinging') or {}
            if hinge.get('present'):
                bit += ' hinge=fold%s Δ=%.3f' % (hinge.get('fold'), hinge.get('abs_delta') or 0)
            bits.append(bit)
        print(row['id'], *bits)
    hinge = ((topo.get('flags') or {}).get('jackknife_hinging') or {})
    if hinge.get('action_id') is not None:
        print('most hinging', hinge.get('action_id'), hinge.get('axis'),
              'fold=%s' % hinge.get('fold'), 'Δ=%.3f' % (hinge.get('abs_delta') or 0))
    return 0 if topo.get('status') == 'RECORDED' else 1


def main(argv=None):
    from parliament_run_provenance import record_execution
    return record_execution(_main, argv, entrypoint='run_parliament_decisions_topography.py', output_flag='--report', in_place=True)

if __name__ == '__main__':
    raise SystemExit(main())
