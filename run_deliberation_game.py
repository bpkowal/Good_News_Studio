"""Play human-scored reasoning rounds over a real admitted Parliament case."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import deliberation_game as game

ROOT=Path(__file__).resolve().parent
DEFAULT_REPORT=ROOT/'diagnostics/contractualist_frontend_20261026/report/DEMO_REPORT.json'


def start(args):
    from parliament_runtime import DEFAULT_PARLIAMENT_ROOT,check_runtime
    check=check_runtime(source_only=True)
    if not check['ready']:raise ValueError('Native source does not match its locked recipe.')
    sys.path.insert(0,str(DEFAULT_PARLIAMENT_ROOT))
    from global_workspace.frozen_world_replay import load_frozen_world_trace
    report=json.loads(args.report.read_text())
    trace=Path(report['artifacts'][0])
    if not trace.is_absolute():trace=ROOT/trace
    frozen=load_frozen_world_trace(trace,expected_scenario=report['scenario'])
    if frozen.action_source_grounding['world_model'] != report['world']:
        raise ValueError('Report and admitted world differ.')
    digest=hashlib.sha256(trace.read_bytes()).hexdigest()
    if digest != report['execution']['frozen_trace_sha256']:raise ValueError('Frozen trace digest differs.')
    policy=json.loads(args.resume_policy.read_text()) if args.resume_policy else None
    case={'scenario':report['scenario'],'world':report['world'],'frameworks':report['frameworks'],
          'hinges':report['hinges'],'source_report_sha256':hashlib.sha256(args.report.read_bytes()).hexdigest(),
          'frozen_trace_sha256':digest,'world_sha256':game.fingerprint(report['world']),
          'source_authority':'NATIVE_REPLAY_VALIDATED_FROZEN_WORLD'}
    game.create(args.session,case,rounds=args.rounds,policy=policy)
    (args.session/'source_report.json').write_bytes(args.report.read_bytes())
    (args.session/'frozen_world_trace.json').write_bytes(trace.read_bytes())
    from parliament_run_provenance import record_resource,record_checkpoint,record_configuration
    record_resource(args.report);record_resource(trace)
    if args.resume_policy:record_resource(args.resume_policy)
    record_configuration(game_version=game.VERSION,world_sha256=case['world_sha256'],round_budget=args.rounds,
                         learning_target='round_allocation',reward_source='human_feedback',new_model_calls=0)
    record_checkpoint('deliberation_game_prepared',artifacts=[args.session/'case.json'])


def live_backend(request,folder):
    from run_blueprint_parliament import _load_environment,DEFAULT_OPENAI_ENV
    from run_opposing_complaint_probe import call_once
    from run_contractualist_pilot import write_new
    from openai import OpenAI
    environment=_load_environment(DEFAULT_OPENAI_ENV)
    key=environment.get('OPENAI_API_KEY')
    if not key:raise ValueError('OPENAI_API_KEY is unavailable.')
    write_new(folder/'request.json',request)
    result=call_once(folder,OpenAI(api_key=key,max_retries=0,timeout=180))
    result['provider_calls']=1
    return result


def choose_framework(state,label):
    names=list(state['framework_emphasis'])
    print(label+': '+', '.join(names))
    value=input('Framework: ').strip()
    if value not in names:raise ValueError('Choose a recorded framework.')
    return value


def play(args):
    print('Live game: up to the remaining round budget in API requests; no automatic retries.')
    print('Rate whether each move was worth its cost. No score is manufactured by the model.')
    while True:
        state,case,_,_=game.read(args.session)
        if state['ended']:break
        print('\nRemaining rounds:',state['remaining_rounds'],'| learned updates:',state['policy']['revision'])
        print('Advisory move values:',game.recommendations(state))
        print('Moves: investigate / self_counterfactual / deliberate / answer / cues / weights / stop / quit')
        kind=input('Move: ').strip()
        if kind=='quit':break
        try:
            if kind=='answer':
                game.observe(args.session,input('New game information: '),author=args.rater);continue
            if kind=='cues':
                cues={}
                for cue in game.CUES:
                    value=input(cue+' [0..1, blank = unassessed]: ').strip()
                    cues[cue]=None if not value else float(value)
                game.configure(args.session,cues=cues,author=args.rater);continue
            if kind=='weights':
                emphasis={name:float(input(name+' emphasis [nonnegative]: ')) for name in state['framework_emphasis']}
                attention=float(input('Counterfactual attention [0..1]: '))
                game.configure(args.session,emphasis=emphasis,counterfactual_attention=attention,author=args.rater);continue
            if kind=='stop':
                print('Available actions:',[(a['action_id'],a['intervention']) for a in case['world']['actions']])
                action=input('Action ID [blank = unresolved]: ').strip() or None
                record=game.move(args.session,'stop',action_id=action,rationale=input('Reason for decision/abstention: '))
            else:
                framework=choose_framework(state,'Active perspective')
                peer=choose_framework(state,'Opposing perspective') if kind=='deliberate' else None
                record=game.move(args.session,kind,framework=framework,peer=peer,
                                 question=input('Specific question or challenged inference: '),backend=live_backend)
            print(json.dumps(record,indent=2,ensure_ascii=False))
            value=input('Was this move worth its cost? [0..1; blank = leave unrated]: ').strip()
            if value:game.rate(args.session,record['id'],float(value),rater=args.rater,reason=input('Why? '))
        except (ValueError,KeyError) as error:
            print('Not recorded:',error)
    print('Game report:',args.session/'GAME_REPORT.md')


def _main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','play','show','rate','answer'))
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=DEFAULT_REPORT)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--resume-policy',type=Path)
    parser.add_argument('--rater',default='local_player')
    parser.add_argument('--move-id',type=int)
    parser.add_argument('--score',type=float)
    parser.add_argument('--reason',default='')
    parser.add_argument('--text',default='')
    parser.add_argument('--db',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.command=='start':start(args)
    elif args.command=='play':play(args)
    elif args.command=='answer':game.observe(args.session,args.text,author=args.rater)
    elif args.command=='rate':game.rate(args.session,args.move_id,args.score,rater=args.rater,reason=args.reason)
    state,case,moves,feedback=game.read(args.session)
    print(json.dumps({'version':game.VERSION,'remaining_rounds':state['remaining_rounds'],
                     'policy_revision':state['policy']['revision'],'moves':len(moves),'ratings':len(feedback),
                     'ended':state['ended'],'world_mutation':False,'native_dispatch':False},indent=2))
    from parliament_run_provenance import record_artifacts,record_checkpoint
    artifacts=[path for path in args.session.rglob('*') if path.is_file() and 'provenance' not in path.parts and path.name!='game.sqlite3']
    record_artifacts(artifacts)
    record_checkpoint('deliberation_game_recorded',artifacts=[args.session/'game_record.json',args.session/'policy_checkpoint.json'])
    return 0


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='show':return _main(argv)
    if '--session' not in argv:return _main(argv)
    folder=Path(argv[argv.index('--session')+1]);folder.mkdir(parents=True,exist_ok=True)
    argv += ['--db',str(folder/'game.sqlite3')]
    from parliament_run_provenance import record_execution
    return record_execution(_main,argv,entrypoint='run_deliberation_game.py',output_flag='--db',in_place=True)


if __name__=='__main__':raise SystemExit(main())
