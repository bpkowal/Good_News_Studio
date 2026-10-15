"""Evidence-bound spatial cells and a typed reader view; never alter world records."""
from html import escape
import re

RELATIONS = {
    'ACTOR': ('entity','process'), 'AFFECTS': ('process','entity'),
    'UNDERGOES': ('entity','process'), 'ONTO': ('process','place'),
    'REMAINS_ON': ('process','place'), 'AWAY_FROM': ('process','place'),
    'TOWARD': ('process','place'),
    'ON': ('entity','place'), 'OCCUPIED_BY': ('place','entity'),
    'HEADING_TOWARD': ('entity','entity'), 'RESULT': ('entity','effect'),
    'ENABLES': ('process','process'), 'CAUSES': ('process','process'), 'BRANCH_OUTCOME': ('gap','effect'),
    'MECHANISM_UNRESOLVED': ('process','gap'),
    'DECISION_MAKER': ('entity','choice'), 'OMITS': ('choice','entity'),
    'BRANCH_FOR': ('choice','process'), 'CONTINUATION_UNRESOLVED': ('choice','gap'),
    'BRANCH_CONTINUATION': ('process','process'),
}


def branch_choice_cells(world):
    """Describe omission relative to a unique paired intervention, not physics."""
    from parsing_game_S import get_nlp
    actions=world.get('actions',[])
    parsed={a['action_id']:get_nlp()(a['intervention']) for a in actions}
    positive=[a for a in actions if not any(t.dep_=='neg' for t in parsed[a['action_id']])]
    cells=[]
    for action in actions:
        aid=action['action_id'];doc=parsed[aid]
        negative=any(t.dep_=='neg' for t in doc)
        root=next((t for t in doc if t.dep_=='ROOT'),None)
        matches=[a for a in positive if a.get('actor_party_id')==action.get('actor_party_id')
                 and a.get('recipient_party_ids')==action.get('recipient_party_ids')
                 and root and (root.lemma_=='do' or any(t.dep_=='ROOT' and t.lemma_==root.lemma_ for t in parsed[a['action_id']]))]
        omission=negative and len(matches)==1
        cells.append({'action_id':aid,'kind':'omission' if omission else 'intervention' if not negative else 'unresolved',
            'decision_maker_id':action.get('actor_party_id'),
            'omitted_intervention_id':matches[0]['action_id'] if omission else None,
            'intervention_description':matches[0]['intervention'] if omission else action['intervention'],
            'instrument_ids':action.get('recipient_party_ids',[]),
            'origin':'STRUCTURALLY_DERIVED' if omission else 'FROZEN_FIELD',
            'clause_ids':list(dict.fromkeys(action.get('clause_ids',[])+(matches[0].get('clause_ids',[]) if omission else []))),
            'meaning':'Choice to omit the paired intervention; not an act causing continued motion.' if omission else 'Candidate intervention; occurrence is not asserted.'})
    return cells


def spatial_cells(text, world):
    """Reuse the parser's noun spans and conservative explicit mechanism matcher."""
    from blueprint_physical_scene import mechanism_from_source, _nominal
    doc, mechanism = mechanism_from_source(text)
    if not mechanism:
        return None
    parties=world.get('parties',[])
    def normalize(value):return re.sub(r'^(?:a|an|the)\s+','', ' '.join(value.lower().split()))
    def resolve(value):
        exact=[p['party_id'] for p in parties if normalize(p['label'])==normalize(value)]
        if len(exact)==1:return exact[0]
        # Unique adjective-expanded mentions, e.g. 'a runaway trolley' / 'the trolley'.
        matches=[p['party_id'] for p in parties if normalize(value).endswith(' '+normalize(p['label']))]
        return matches[0] if len(matches)==1 else None
    mover=resolve(mechanism['process']);instrument=resolve(mechanism['instrument']);destination=resolve(mechanism['destination'])
    if not all((mover,instrument,destination)):return None
    def evidence(sentence):return {'start':sentence.start_char,'end':sentence.end_char,'text':sentence.text}
    readings=[]
    def add(kind,source,target,sentence):
        if source and target and not any(r['type']==kind and r['source']==source and r['target']==target for r in readings):
            readings.append({'id':'SP'+str(len(readings)+1),'type':kind,'source':source,'target':target,
                             'origin':'SOURCE_READING','evidence':evidence(sentence)})
    for token in doc:
        sent=token.sent
        # Scene locations require affirmative, nonconditional local wording.
        if any(t.dep_=='neg' or t.lower_ in {'if','unless','may','might','could','according'} for t in sent):continue
        if '?' in sent.text or any(c in sent.text for c in '\"“”'):continue
        if sent.root.lemma_.lower() in {'say','report','believe','think','claim','deny','false'}:continue
        if token.lower_=='on' and token.dep_=='prep':
            objects=[c for c in token.children if c.dep_=='pobj']
            if len(objects)==1:add('ON',resolve(_nominal(token.head)),resolve(_nominal(objects[0])),sent)
        if token.lemma_=='stand' and token.dep_=='relcl' and any(c.lower_=='where' for c in token.children):
            subjects=[c for c in token.children if c.dep_=='nsubj']
            if len(subjects)==1:add('ON',resolve(_nominal(subjects[0])),resolve(_nominal(token.head)),sent)
        if token.lemma_=='head' and token.dep_=='ROOT':
            subjects=[c for c in token.children if c.dep_=='nsubj']
            targets=[p for c in token.children if c.lower_ in {'toward','towards'} for p in c.children if p.dep_=='pobj']
            if len(subjects)==len(targets)==1:add('HEADING_TOWARD',resolve(_nominal(subjects[0])),resolve(_nominal(targets[0])),sent)
    transitions=[];questions=[]
    for action in world.get('actions',[]):
        effects=[e for e in world.get('effects',[]) if e.get('action_id')==action['action_id']]
        transition=[e for e in effects if e.get('party_id')==mover and e.get('effect_kind')=='PHYSICAL_STATE' and normalize(e.get('outcome',''))==normalize(mechanism['transition'])]
        if len(transition)==1:
            start=text.find(mechanism['transition'])
            transitions.append({'action_id':action['action_id'],'process_id':transition[0]['effect_id'],
                'movement_entity':mover,'instrument':instrument,'destination':destination,'relation':'ONTO',
                'origin':'SOURCE_READING','evidence':{'start':start,'end':start+len(mechanism['transition']),'text':mechanism['transition']}})
        elif any(e.get('party_id')==mover and e.get('polarity')=='NEUTRAL'
                 and e.get('directness')=='DOWNSTREAM' for e in effects):
            from blueprint_physical_scene import omission_continuation
            harms=[e for e in effects if e.get('polarity')=='ADVERSE']
            continuation=omission_continuation(text,mechanism['process'],harms[0]['outcome']) if len(harms)==1 else None
            # Native admission omits the compiler's predicate field. Bind the
            # exact source-supported continuation and mover instead of relying
            # on that optional producer annotation.
            process=[e for e in effects if e.get('party_id')==mover
                     and e.get('polarity')=='NEUTRAL' and e.get('directness')=='DOWNSTREAM'
                     and continuation and e.get('outcome')==continuation['transition']]
            if len(process)==1 and resolve(continuation['destination']):
                start=text.find(continuation['transition'])
                transitions.append({'action_id':action['action_id'],'process_id':process[0]['effect_id'],
                    'movement_entity':mover,'instrument':instrument,'destination':resolve(continuation['destination']),
                    'relation':'TOWARD','origin':'STRUCTURALLY_DERIVED',
                    'evidence':{'start':start,'end':start+len(continuation['transition']),'text':continuation['transition']}})
            else:
                questions.append({'kind':'missing_physical_continuation','action_id':action['action_id'],
                                  'question':'No unique source-backed default route could be bound.'})
        elif effects:
            questions.append({'kind':'missing_physical_continuation','action_id':action['action_id'],
                              'question':'What physical continuation is supported on this branch? No route is supplied by the action label alone.'})
    if not transitions or not any(r['type']=='ON' for r in readings):return None
    return {'schema_version':'spatial-reader-cells/1','observational_only':True,
            'movement_entity':mover,'instrument':instrument,'destination':destination,
            'scene_relations':readings,'branch_transitions':transitions,'open_questions':questions,
            'branch_choices':branch_choice_cells(world)}


def typed_reader_view(text, world):
    cells=spatial_cells(text,world)
    if not cells:return None
    parties={p['party_id']:p for p in world['parties']}
    places={r['target'] for r in cells['scene_relations'] if r['type']=='ON'}|{cells['destination']}
    nodes=[];edges=[];groups=[]
    def node(ident,label,kind,refs,group,**extra):
        nodes.append({'id':ident,'label':label,'kind':kind,'world_ids':refs,'group':group,**extra});return ident
    def edge(source,relation,target,origin='FROZEN_FIELD',**extra):
        edges.append({'id':'TE'+str(len(edges)+1),'source':source,'type':relation,'target':target,'origin':origin,**extra})
    scene={}
    for r in cells['scene_relations']:
        for pid in (r['source'],r['target']):
            if pid not in scene:scene[pid]=node('SC_'+pid,parties[pid]['label'],'place' if pid in places else 'entity',[pid],'scene')
        edge(scene[r['source']],r['type'],scene[r['target']],'SOURCE_READING',evidence=r['evidence'])
    choices={c['action_id']:c for c in cells['branch_choices']}
    positive=[a for a in world['actions'] if choices[a['action_id']]['kind']=='intervention']
    for action in world['actions']:
        aid=action['action_id'];group='branch_'+aid;label=action['intervention'];expanded=False
        if re.fullmatch(r'(?:does|do|did)\s+not',label,re.I) and len(positive)==1:
            from parsing_game_S import get_nlp
            phrase=positive[0]['intervention'];verb=next((t for t in get_nlp()(phrase) if t.dep_=='ROOT'),None)
            base=phrase[:verb.idx]+verb.lemma_+phrase[verb.idx+len(verb.text):] if verb else phrase
            label='Does not '+base
            expanded=True
        clause_ids=set(action.get('clause_ids',[]))|{p.get('clause_id') for p in action.get('provenance',[])}
        conditions=[c for c in world.get('conditions',[]) if clause_ids.intersection(set(c.get('clause_ids',[]))|{p.get('clause_id') for p in c.get('provenance',[])})]
        groups.append((group,conditions[0]['description'] if len(conditions)==1 else 'Option: '+label))
        actor=action.get('actor_party_id');direct=[e['effect_id'] for e in world['effects'] if e['action_id']==aid and e['directness']=='DIRECT']
        omission=choices[aid]['kind']=='omission'
        act=node(aid,label,'choice' if omission else 'process',[aid]+direct,group,
                 display_reconstruction=expanded,choice_semantics=choices[aid])
        if actor:
            actor_id='ACT_'+actor
            if not any(n['id']==actor_id for n in nodes):node(actor_id,parties[actor]['label'],'entity',[actor],'actor')
            edge(actor_id,'DECISION_MAKER' if omission else 'ACTOR',act)
        transition=next((t for t in cells['branch_transitions'] if t['action_id']==aid),None)
        branch_entities={}
        def entity(pid):
            if pid not in branch_entities:branch_entities[pid]=node(aid+'_'+pid,parties[pid]['label'],'place' if pid in places else 'entity',[pid],group)
            return branch_entities[pid]
        outcomes=[e for e in world['effects'] if e['action_id']==aid and e.get('polarity') in {'ADVERSE','BENEFICIAL'}]
        if transition:
            process=next(e for e in world['effects'] if e['effect_id']==transition['process_id'])
            mechanism=node(process['effect_id'],process['outcome'],'process',[process['effect_id']],group)
            for recipient in action.get('recipient_party_ids',[]):
                if recipient in parties:edge(act,'OMITS' if omission else 'AFFECTS',entity(recipient),
                    meaning='Omits operating this instrument; does not physically affect it.' if omission else 'Intervention acts on this instrument.')
            links=[i for i,l in enumerate(world.get('causal_links',[])) if l.get('relation') in {'CAUSES','ENABLES'} and l['source_id'] in direct and l['target_id']==process['effect_id']]
            if omission:
                edge(act,'BRANCH_FOR',mechanism,'STRUCTURALLY_DERIVED',
                     meaning='Physical continuation within this choice branch, not physical causation by omission.')
            elif links:edge(act,world['causal_links'][links[0]]['relation'],mechanism,world_link_ids=links)
            else:edge(act,'BRANCH_CONTINUATION',mechanism,'STRUCTURALLY_DERIVED',
                      meaning='Mechanism represented in this intervention branch; no new causal relation is asserted.')
            edge(entity(cells['movement_entity']),'UNDERGOES',mechanism)
            edge(mechanism,transition['relation'],entity(transition['destination']),transition['origin'],evidence=transition['evidence'])
        else:
            mechanism=node(aid+'_GAP','Physical continuation not recorded','gap',[],group)
            edge(act,'CONTINUATION_UNRESOLVED' if omission else 'MECHANISM_UNRESOLVED',mechanism,'UNRESOLVED')
        for outcome in outcomes:
            pid=outcome['party_id'];person=entity(pid)
            out=node(outcome['effect_id'],outcome['outcome'],'effect',[outcome['effect_id']],group)
            edge(person,'RESULT',out,meaning='The named party has this branch outcome; not a causal arrow from the person.')
            location=[r for r in cells['scene_relations'] if r['type']=='ON' and r['source']==pid]
            for r in location:edge(entity(r['target']),'OCCUPIED_BY',person,'SOURCE_READING',evidence=r['evidence'])
            if not transition:edge(mechanism,'BRANCH_OUTCOME',out,'SOURCE_BRANCH',world_ids=[outcome['effect_id']])
    # Enforce arrow direction by endpoint type, independently of labels/layout.
    kinds={n['id']:n['kind'] for n in nodes}
    for e in edges:
        if (kinds[e['source']],kinds[e['target']])!=RELATIONS[e['type']]:raise ValueError('Invalid typed reader endpoints: '+e['type'])
    lines=['flowchart TB'];all_groups=[('scene','Scene')]+groups+[('actor','Decision maker')]
    def safe(value):return escape(value,quote=True).replace('\n',' ').replace('|','&#124;')
    for group,title in all_groups:
        members=[n for n in nodes if n['group']==group]
        if not members:continue
        lines += [f'  subgraph {group}["{safe(title)}"]','    direction LR']
        for n in members:lines.append(f'    {n["id"]}["{safe(n["label"])}<br/>{n["kind"]}"]')
        lines.append('  end')
    for e in edges:
        arrow='-->' if e['origin']=='FROZEN_FIELD' else '-.->'
        lines.append(f'  {e["source"]} {arrow}|"{e["type"]}"| {e["target"]}')
    lines+=['  classDef sourceReading fill:#eef4fb,stroke:#46739a,color:#233b51','  classDef unresolved fill:#fff3db,stroke:#af7b26,stroke-dasharray:5 3,color:#60461e']
    colors={'entity':('#eef3fa','#64809e'),'place':('#e9f4ec','#60836a'),'process':('#fff6e4','#a18a56'),'choice':('#fff6e4','#a18a56'),'effect':('#f6edf5','#96738d')}
    for kind,(fill,stroke) in colors.items():
        lines.append('  classDef reader_'+kind+' fill:'+fill+',stroke:'+stroke+',color:#26332e')
        members=[n['id'] for n in nodes if n['kind']==kind]
        if members:lines.append('  class '+','.join(members)+' reader_'+kind)
    for n in nodes:
        if n['kind']=='gap':lines.append('  class '+n['id']+' unresolved')
    return {'schema_version':'typed-reader-view/1','mermaid':'\n'.join(lines),'nodes':nodes,'edges':edges,
            'cells':cells,'notice':'Typed reader projection. Dashed arrows are source readings, structural derivations or unresolved branch structure, not additional admitted relations. Omission is a choice: DECISION_MAKER identifies its owner; OMITS identifies the unused instrument; BRANCH_FOR is branch dependence, not causation. Native records remain unchanged. RESULT associates a patient with an outcome; it does not make the patient its cause. Default motion is bound only when the stated scene and omission outcome identify the same patients; no REMAINS_ON or AWAY_FROM route is invented. Repeated entity boxes are display copies, not new identities. Entity/place/process are reader roles; original native kinds are retained in inspection. Full graph retains every original record.'}
