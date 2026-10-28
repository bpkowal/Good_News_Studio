"""Human-scored investigation game over a frozen Parliament case. No toy reward."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from behavioral_policy import ContextualPolicy, finite
from argument_frameworks import FRAMEWORK_GUIDANCE

VERSION = 'parliament-deliberation-game/0.1'
MOVES = ('investigate', 'self_counterfactual', 'deliberate', 'stop')
CUES = ('factual_uncertainty','normative_conflict','causal_uncertainty',
        'relationship_claims','outcome_uncertainty')
FEATURES = tuple([name for cue in CUES for name in (cue, cue+'_assessed')] + ['remaining_budget'])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def context(state):
    encoded = {}
    for cue in CUES:
        value = state['question_cues'][cue]
        if value is not None and (not finite(value) or not 0 <= value <= 1):
            raise ValueError('Cue annotations must be null or finite values in [0,1].')
        encoded[cue] = 0.0 if value is None else value
        encoded[cue+'_assessed'] = float(value is not None)
    encoded['remaining_budget'] = state['remaining_rounds']/state['round_budget']
    return encoded


def connect(folder):
    db = sqlite3.connect(str(Path(folder)/'game.sqlite3'))
    db.row_factory = sqlite3.Row
    return db


def create(folder, case, *, rounds=2, policy=None):
    folder = Path(folder)
    if type(rounds) is not int or not 1 <= rounds <= 4:
        raise ValueError('Choose one to four additional rounds.')
    if (folder/'game.sqlite3').exists():
        raise ValueError('Use a new game session.')
    if not case.get('world') or not case.get('frameworks'):
        raise ValueError('A frozen world and its recorded framework arguments are required.')
    folder.mkdir(parents=True,exist_ok=True)
    names = [view['framework'] for view in case['frameworks']]
    model = ContextualPolicy.restore(policy) if policy else ContextualPolicy(MOVES,FEATURES,interactions=False)
    if model.actions != MOVES or model.features != FEATURES or model.interactions:
        raise ValueError('Game control weights must match the additive control schema; scenario-action weights cannot be relabeled.')
    case = deepcopy(case)
    case_hash = fingerprint(case)
    (folder/'case.json').write_text(json.dumps(case,indent=2,ensure_ascii=False)+'\n')
    state = {'version':VERSION,'case_sha256':case_hash,'round_budget':rounds,'remaining_rounds':rounds,
             'question_cues':dict.fromkeys(CUES), 'cue_authority':'USER_ANNOTATED_NOT_AUTOMATIC_TRUTH',
             'framework_emphasis':{name:1/len(names) for name in names},
             'counterfactual_attention':0.25, 'policy':model.snapshot(), 'ended':False,
             'terminal_action_id':None,'terminal_rationale':None,
             'observations':[],
             'world_mutation':False,'native_dispatch':False,'reward_source':'EXPLICIT_HUMAN_WORTHWHILE_RATING_ONLY'}
    db = connect(folder)
    try:
        with db:
            db.executescript('''CREATE TABLE state(id INTEGER PRIMARY KEY CHECK(id=1),record TEXT NOT NULL);
            CREATE TABLE moves(id INTEGER PRIMARY KEY,record TEXT NOT NULL);
            CREATE TABLE feedback(move_id INTEGER PRIMARY KEY REFERENCES moves(id),record TEXT NOT NULL);
            CREATE TABLE configuration(id INTEGER PRIMARY KEY,record TEXT NOT NULL);''')
            db.execute('INSERT INTO state VALUES(1,?)',(json.dumps(state),))
    finally:
        db.close()
    export(folder)
    return state


def read(folder):
    db = connect(folder)
    try:
        state = json.loads(db.execute('SELECT record FROM state WHERE id=1').fetchone()[0])
        if state['version'] != VERSION:
            raise ValueError('Unsupported game version.')
        moves = [json.loads(row[0]) for row in db.execute('SELECT record FROM moves ORDER BY id')]
        feedback = [json.loads(row[0]) for row in db.execute('SELECT record FROM feedback ORDER BY move_id')]
    finally:
        db.close()
    case = json.loads((Path(folder)/'case.json').read_text())
    if fingerprint(case) != state['case_sha256']:
        raise ValueError('Frozen game case changed.')
    return state,case,moves,feedback


def configure(folder, *, cues=None, emphasis=None, counterfactual_attention=None, author):
    state,case,_,_ = read(folder)
    if state['ended']:raise ValueError('This episode has ended.')
    before = deepcopy(state)
    if cues is not None:
        if set(cues) != set(CUES):raise ValueError('Supply the declared cue schema.')
        state['question_cues'] = deepcopy(cues)
        context(state)
    if emphasis is not None:
        if set(emphasis) != set(state['framework_emphasis']) or any(not finite(v) or v < 0 for v in emphasis.values()) or sum(emphasis.values()) <= 0:
            raise ValueError('Each recorded perspective needs a nonnegative emphasis; total must be positive.')
        state['framework_emphasis'] = {k:v/sum(emphasis.values()) for k,v in emphasis.items()}
    if counterfactual_attention is not None:
        if not finite(counterfactual_attention) or not 0 <= counterfactual_attention <= 1:
            raise ValueError('Counterfactual attention must be in [0,1].')
        state['counterfactual_attention'] = counterfactual_attention
    db = connect(folder)
    try:
        with db:
            db.execute('INSERT INTO configuration(record) VALUES(?)',(json.dumps({'author':author,'before':before,'after':state}),))
            db.execute('UPDATE state SET record=? WHERE id=1',(json.dumps(state),))
    finally:db.close()
    export(folder)


def recommendations(state):
    policy = ContextualPolicy.restore(state['policy'])
    values = policy.values(context(state))
    if state['remaining_rounds'] == 0:
        values = {'stop':values['stop']}
    return sorted(values.items(),key=lambda item:-item[1])


def observe(folder,text,*,author):
    if not text.strip() or not author.strip():raise ValueError('An attributed observation is required.')
    state,_,_,_ = read(folder)
    if state['ended']:raise ValueError('This episode has ended.')
    state['observations'].append({'text':text,'author':author,
                                  'authority':'USER_PROVIDED_GAME_INFORMATION_NOT_NATIVE_WORLD'})
    db=connect(folder)
    try:
        with db:db.execute('UPDATE state SET record=? WHERE id=1',(json.dumps(state),))
    finally:db.close()
    export(folder)


def request_for(state,case,moves,kind,framework,peer,question):
    history = []
    for move in moves:
        if not move.get('reply'):continue
        if move['kind'] == 'self_counterfactual' and state['counterfactual_attention'] == 0:continue
        history.append({'kind':move['kind'],'framework':move.get('framework'),'peer':move.get('peer'),
                        'reply':move['reply'],'findings':move['findings'],
                        'authority':'UNVERIFIED_GAME_ARGUMENT_NOT_WORLD_EVIDENCE'})
    task = {
        'investigate':'Identify the decision-relevant missing information; distinguish source answers from unanswered factual and normative questions. Ask what would change the comparison.',
        'self_counterfactual':'Challenge your own recorded framework argument with a specific alternative, counterexample or reversal condition. Preserve hypothetical scope and explain which inference is tested.',
        'deliberate':'Address the actual opposing framework argument, not merely your own opening. State the disputed ground or warrant and defend a reply or keep the objection open.',
    }[kind]
    properties = {'summary':{'type':'string'},'source_quotes':{'type':'array','items':{'type':'string'}},
                  'remaining_questions':{'type':'array','items':{'type':'string'}},
                  'counterfactuals':{'type':'array','items':{'type':'string'}},
                  'suggested_action_id':{'anyOf':[{'type':'string'},{'type':'null'}]}}
    schema = {'type':'object','additionalProperties':False,'properties':properties,'required':list(properties)}
    return {'model':'o3','reasoning_effort':'low','max_completion_tokens':4096,
            'messages':[{'role':'system','content':FRAMEWORK_GUIDANCE[framework]+' '+task+
                         ' The frozen world is shared. Peer arguments and new hypotheses are not world facts. '
                         'User-provided game information may inform conditional reasoning with explicit attribution; it does not rewrite the native world. '
                         'Use exact source quotations. Do not invent outcomes, consensus, reliance or authority. '
                         'You may leave the decision unresolved. Emphasis controls requested attention, not moral truth.'},
                        {'role':'user','content':json.dumps({'scenario':case['scenario'],'world':case['world'],
                            'recorded_frameworks':case['frameworks'],'role':framework,'opposing_framework':peer,
                            'focus_question':question,'game_history':history,
                            'user_provided_game_information':state['observations'],
                            'requested_perspective_emphasis':state['framework_emphasis'],
                            'requested_counterfactual_attention':state['counterfactual_attention']},ensure_ascii=False)}],
            'response_format':{'type':'json_schema','json_schema':{'name':'deliberation_game_reply','strict':True,'schema':schema}}}


def move(folder,kind,*,framework=None,peer=None,question='',backend=None,action_id=None,rationale=''):
    state,case,history,_ = read(folder)
    if state['ended']:raise ValueError('This episode has ended.')
    if kind not in MOVES:raise ValueError('Unknown control move.')
    if kind != 'stop' and state['remaining_rounds'] <= 0:raise ValueError('No additional rounds remain.')
    names = set(state['framework_emphasis'])
    if kind != 'stop' and (framework not in names or framework not in FRAMEWORK_GUIDANCE):raise ValueError('Select a recorded framework.')
    if kind == 'deliberate' and (peer not in names or peer == framework):raise ValueError('A peer challenge needs a different recorded perspective.')
    actions = {a['action_id'] for a in case['world']['actions']}
    if kind == 'stop' and action_id is not None and action_id not in actions:raise ValueError('Select an existing action or leave the decision unresolved.')
    if kind != 'stop' and backend is None:raise ValueError('A reasoning backend is required.')
    record = {'kind':kind,'framework':framework,'peer':peer,'question':question,'context':context(state),
              'policy_at_selection':deepcopy(state['policy']),'status':'STOPPED' if kind == 'stop' else 'ATTEMPT_RESERVED',
              'reply':None,'findings':[],'provider_calls':0,'requested_emphasis':state['framework_emphasis'],
              'counterfactual_attention':state['counterfactual_attention']}
    if kind == 'stop':
        state.update(ended=True,terminal_action_id=action_id,terminal_rationale=rationale)
    else:state['remaining_rounds']-=1
    db = connect(folder)
    try:
        with db:
            record['id'] = db.execute('INSERT INTO moves(record) VALUES(?)',(json.dumps(record),)).lastrowid
            db.execute('UPDATE moves SET record=? WHERE id=?',(json.dumps(record),record['id']))
            db.execute('UPDATE state SET record=? WHERE id=1',(json.dumps(state),))
        if kind != 'stop':
            request = request_for(state,case,history,kind,framework,peer,question)
            path = Path(folder)/('move_%03d'%record['id']);path.mkdir()
            started=time.monotonic()
            try:
                result = backend(request,path)
                record['provider_calls'] = result.get('provider_calls',1)
                record['transport_status'] = result['status']
                if result['status'] == 'RETURNED':
                    raw = result['response']
                    record['raw_response'] = deepcopy(raw)
                    from contractualist_pilot import validate_shape
                    validate_shape(raw,request['response_format']['json_schema']['schema'])
                    record['reply'] = deepcopy(raw)
                    for quote in raw['source_quotes']:
                        if not quote or quote not in case['scenario']:record['findings'].append({'kind':'UNSUPPORTED_SOURCE_QUOTE','quote':quote})
                    if raw['suggested_action_id'] is not None and raw['suggested_action_id'] not in actions:
                        record['findings'].append({'kind':'UNKNOWN_ACTION_PROPOSAL','action_id':raw['suggested_action_id']})
                    record['status'] = 'ARGUMENT_RECORDED_NOT_NATIVE_ADMITTED'
                else:record['status']='FAILED_OR_INCOMPLETE'
            except Exception as error:
                record.update(status='FAILED',error_type=type(error).__name__)
            record['elapsed_seconds']=time.monotonic()-started
            with db:db.execute('UPDATE moves SET record=? WHERE id=?',(json.dumps(record),record['id']))
    finally:db.close()
    export(folder)
    return record


def rate(folder,move_id,worthwhile,*,rater,reason):
    if not finite(worthwhile) or not 0 <= worthwhile <= 1 or not rater.strip() or not reason.strip():
        raise ValueError('Human feedback needs a [0,1] rating, identified rater and explanation.')
    state,_,moves,_ = read(folder)
    selected = next((m for m in moves if m['id'] == move_id),None)
    if selected is None or selected['status']=='ATTEMPT_RESERVED':raise ValueError('No completed move to rate.')
    model = ContextualPolicy.restore(state['policy'])
    before = model.snapshot()
    model.update(selected['context'],selected['kind'],worthwhile)
    record = {'move_id':move_id,'worthwhile':worthwhile,'rater':rater,'reason':reason,
              'authority':'HUMAN_SUBJECTIVE_PROCESS_FEEDBACK_NOT_MORAL_TRUTH',
              'policy_before':before,'policy_after':model.snapshot()}
    state['policy']=model.snapshot()
    db = connect(folder)
    try:
        with db:
            db.execute('INSERT INTO feedback VALUES(?,?)',(move_id,json.dumps(record)))
            db.execute('UPDATE state SET record=? WHERE id=1',(json.dumps(state),))
    finally:db.close()
    export(folder)
    return record


def export(folder):
    state,case,moves,feedback = read(folder)
    folder=Path(folder)
    (folder/'policy_checkpoint.json').write_text(json.dumps(state['policy'],indent=2)+'\n')
    (folder/'game_record.json').write_text(json.dumps({'state':state,'moves':moves,'feedback':feedback},indent=2,ensure_ascii=False)+'\n')
    lines=['# Parliament deliberation game', '',case['scenario'],'',
           'Mode: human-scored game over a frozen admitted case. No native vote or world mutation.',
           'Remaining additional rounds: '+str(state['remaining_rounds']),
           'Control-weight revision: '+str(state['policy']['revision']), '', '## Moves','']
    for m in moves:
        lines += ['### Move %s: %s / %s'%(m['id'],m['kind'],m['status']),
                  (m.get('reply') or {}).get('summary','No model argument returned.'),
                  'Findings: '+json.dumps(m['findings']), '']
    lines += ['## Human feedback','']+[str(f['move_id'])+': '+str(f['worthwhile'])+' — '+f['reason'] for f in feedback]
    lines += ['', '## Remaining scope', '',
              'Only round-allocation weights learn in this increment. Perspective emphasis and counterfactual attention are recorded experimental settings, not measured moral authority.',
              'No feedback means no learning. A repeated score cannot update the same move twice.',
              'Unsupported quotes and unknown action proposals remain labeled in the move record. Replies and hypotheses do not enter the admitted world.']
    (folder/'GAME_REPORT.md').write_text('\n'.join(lines)+'\n')
