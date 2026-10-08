'use strict';
const $ = id => document.getElementById(id);
const state = {run:null, diagram:0, view:'world', zoom:1, fullLinks:false, version:0, render:0, cache:new Map(), index:[]};
const node = (tag, text, cls) => {const n=document.createElement(tag); if(text!==undefined)n.textContent=String(text); if(cls)n.className=cls; return n;};
const clear = n => n.replaceChildren();
const notice = message => {$('notice').textContent=message||''; $('notice').hidden=!message;};
const list = (target, values, empty='None separately recorded.') => {clear(target); if(!values?.length){target.append(node('p',empty)); return;} const ul=node('ul'); values.forEach(v=>ul.append(node('li',typeof v==='string'?v:JSON.stringify(v)))); target.append(ul);};
const badge = status => node('span',status.replaceAll('_',' '),'badge '+(status.includes('BEAM')?'beam':status.includes('FAILED')?'failure':status==='FULL'||status==='FROZEN'||status==='COMPLETED'?'':'partial'));

function validateRun(run){
  if(!run || run.schema_version!=='parliament-view/1')throw new Error('Unsupported saved-run version. Open a parliament-view/1 export.');
  if(typeof run.source!=='string'||typeof run.id!=='string'||typeof run.title!=='string'||typeof run.status!=='string')throw new Error('Run identity, source or status is missing.');
  for(const field of ['diagrams','frameworks','hinges','unknowns','not_decided'])if(!Array.isArray(run[field]))throw new Error('Missing record array: '+field);
  for(const view of run.frameworks)if(!view||typeof view.framework!=='string'||!['FULL','ATTENUATED','ABSTAIN'].includes(view.position))throw new Error('A framework participation record is incomplete.');
  for(const h of run.hinges)if(!h||typeof h.text!=='string'||typeof h.framework!=='string'||typeof h.record!=='string')throw new Error('A decision hinge lacks its originating record.');
  for(const field of ['unknowns','not_decided'])if(run[field].some(v=>typeof v!=='string'))throw new Error('Expected text records in '+field);
  for(const d of run.diagrams){
    if(!d.world||typeof d.mermaid!=='string'||typeof d.label!=='string'||!['FROZEN','CANDIDATE'].includes(d.authority))throw new Error('A world diagram is incomplete.');
    if(d.authority==='CANDIDATE'&&d.sent_to_agents)throw new Error('A candidate diagram cannot claim agent participation.');
  }
  if(run.status==='INTERPRETIVE_BEAM'&&run.diagrams.some(d=>d.authority==='FROZEN'))throw new Error('An interpretive beam cannot contain a selected frozen world.');
  return run;
}

function renderRail(){
  clear($('run-list'));
  state.index.forEach(entry=>{const b=node('button',undefined,'run-link'+(state.run?.id===entry.id?' selected':'')); b.setAttribute('aria-current',state.run?.id===entry.id?'page':'false'); b.append(node('span',entry.title),node('small',entry.status.replaceAll('_',' '))); b.addEventListener('click',()=>loadRun(entry)); $('run-list').append(b);});
}

async function loadRun(entry){
  const request=++state.version;
  try{
    notice('');
    let data=state.cache.get(entry.id);
    if(!data){if(!/^[a-zA-Z0-9_-]+\.json$/.test(entry.file))throw new Error('Invalid saved-run filename.'); const response=await fetch('data/'+entry.file); if(!response.ok)throw new Error('Saved run could not be loaded.'); data=validateRun(await response.json()); state.cache.set(entry.id,data);}
    if(request!==state.version)return;
    state.run=data; state.diagram=0; state.zoom=1; renderRun();
  }catch(error){if(request===state.version)notice(error.message);}
}

function sourceHighlight(excerpts=[]){
  clear($('source-text'));
  const text=state.run.source, ranges=[];
  excerpts.forEach(excerpt=>{if(!excerpt)return; let offset=0, pos; while((pos=text.indexOf(excerpt,offset))!==-1){ranges.push([pos,pos+excerpt.length]);offset=pos+excerpt.length;}});
  ranges.sort((a,b)=>a[0]-b[0]);const merged=[];
  ranges.forEach(range=>{const prev=merged.at(-1);if(prev&&range[0]<=prev[1])prev[1]=Math.max(prev[1],range[1]);else merged.push([...range]);});
  let offset=0;for(const [start,end] of merged){$('source-text').append(document.createTextNode(text.slice(offset,start)),node('mark',text.slice(start,end)));offset=end;}
  $('source-text').append(document.createTextNode(text.slice(offset)));
}

function setView(view){
  state.view=view;
  for(const name of ['world','frameworks','hinges']){$('view-'+name).hidden=name!==view; $('tab-'+name).setAttribute('aria-selected',String(name===view)); $('tab-'+name).tabIndex=name===view?0:-1;}
}

function renderRun(){
  const r=state.run; renderRail(); $('run-title').textContent=r.title;
  $('run-subtitle').textContent=r.mode==='SAVED_REPLAY_NO_API'?'Saved replay · no new model calls · original records remain unchanged.':'Saved record · inspect attributed outputs and their limits.';
  $('download-run').disabled=false; clear($('status-strip'));
  const stopReason=r.stop?.reason===r.status&&r.status==='FRAMEWORK_PARTIAL'?'Some framework positions remain provisional or abstaining.':r.stop?.reason||'No stop reason recorded.';
  $('status-strip').append(badge(r.status),node('span',stopReason),node('span','Native judgment: '+(r.judgment_status||'not recorded')));
  $('source-hash').textContent='Source '+(r.source_sha256||'unhashed').slice(0,12);
  sourceHighlight(); clear($('stipulations'));
  if(r.stipulated_claims?.length){const details=node('details',undefined,'stipulations');details.append(node('summary','Source stipulations · '+r.stipulated_claims.length));const ul=node('ul');r.stipulated_claims.forEach(p=>ul.append(node('li',p.claim||'Unnamed source claim')));details.append(ul);$('stipulations').append(details);}
  clear($('diagram-tabs'));r.diagrams.forEach((d,i)=>{const button=node('button',d.label,i===state.diagram?'active':'');button.setAttribute('aria-pressed',String(i===state.diagram));button.addEventListener('click',()=>{state.diagram=i;state.zoom=1;renderDiagram();});$('diagram-tabs').append(button);});
  renderDiagram();renderFrameworks();renderHinges();setView(state.view);
}

function recordsFor(d){
  const result=[];
  for(const [array,key,kind] of [['parties','party_id','Party'],['actions','action_id','Action'],['effects','effect_id','Effect'],['conditions','condition_id','Condition'],['propositions','proposition_id','Proposition'],['commitments','commitment_id','Commitment'],['reports','report_id','Report'],['modal_operators','modal_id','Modal operator'],['normative_propositions','normative_proposition_id','Normative proposition']])
    for(const record of d.world[array]||[]){const id=record[key]||record.modal_operator_id||record.id; if(id)result.push({id,kind,record,label:record.label||record.intervention||record.outcome||record.description||record.claim||record.predication||record.force||id});}
  for(const [i,record] of (d.world.causal_links||[]).entries())result.push({id:'relation:'+i,kind:'Relation',record,label:record.source_id+' '+record.relation+' '+record.target_id});
  for(const item of d.source_ledger_nodes||[])result.push({id:item.id,kind:'Source ledger addition',record:item.record,label:item.record.claim});
  return result;
}

function inspect(item){
  const r=item.record;clear($('inspector'));
  const left=node('div');left.append(node('div',item.kind.toUpperCase()+' / '+item.id,'record-kind'),node('h2',item.label));
  const facts=node('div',undefined,'facts');for(const key of ['kind','effect_kind','modality','polarity','directness','relation','condition_join','derivation_operation'])if(r[key])facts.append(node('span',key.replaceAll('_',' ')+': '+r[key]));left.append(facts);
  const excerpts=(r.provenance||[]).filter(p=>p&&typeof p==='object').map(p=>p.excerpt).filter(e=>typeof e==='string'&&e.length);
  const exact=excerpts.filter(e=>state.run.source.includes(e));
  if(item.kind==='Source ledger addition'){
    left.append(node('p','Source-ledger addition, outside the frozen-world topology. Support references are ledger bookkeeping, not causal or duty relations.','ledger-layer-note'));
    if(typeof r.claim==='string'&&state.run.source.includes(r.claim)){excerpts.push(r.claim);exact.push(r.claim);}
  }
  left.append(node('h3','Supporting source'));
  if(excerpts.length)excerpts.forEach(e=>{left.append(node('blockquote',e));if(!state.run.source.includes(e))left.append(node('p','This excerpt does not match the saved source exactly.','error'));});
  else left.append(node('p','No direct source excerpt on this record. Do not infer support from graph proximity.'));
  if(r.derivation_explanation)left.append(node('h3','Recorded derivation'),node('p',r.derivation_explanation));
  const qualifiers={};for(const key of ['condition_ids','scope_qualifiers','likelihood_qualifiers','temporal_qualifiers','quantities','derivation_assumptions'])if(r[key]?.length)qualifiers[key]=r[key];
  if(item.kind==='Source ledger addition')for(const key of ['proposition_id','introduced_by','epistemic_status','epistemic_type','support_ids','derived_from'])if(r[key]!==undefined)qualifiers[key]=r[key];
  const right=node('div');right.append(node('h3','Provenance & scope'),node('p','Saved record only. Its presence in an admitted graph does not establish occurrence of every alternative action.'));
  const pre=node('pre',JSON.stringify({provenance:r.provenance||[],...qualifiers},null,2));right.append(pre);
  const details=node('details');details.append(node('summary','Complete original record'),node('pre',JSON.stringify(r,null,2)));right.append(details);
  $('inspector').append(left,right);sourceHighlight(exact);
}

function emptyInspector(){clear($('inspector'));$('inspector').append(node('div','Select a node or relation to inspect its source, provenance and scope.'),node('p','Use the record list if you prefer keyboard navigation. Source claims and agent arguments remain separate.'));}

function applyZoom(){const svg=$('graph-canvas').querySelector('svg');if(!svg)return; const box=svg.viewBox.baseVal;if(!box.width||!box.height)return;const viewport=$('graph-viewport');const fit=Math.min(Math.max(1,viewport.clientWidth-44)/box.width,Math.max(1,viewport.clientHeight-44)/box.height);svg.style.width=box.width*fit*state.zoom+'px';svg.style.height=box.height*fit*state.zoom+'px';svg.style.maxWidth='none';}

async function renderDiagram(){
  const generation=++state.render, d=state.run.diagrams[state.diagram];clear($('graph-canvas'));clear($('record-list'));emptyInspector();sourceHighlight();
  [...$('diagram-tabs').children].forEach((button,i)=>{button.classList.toggle('active',i===state.diagram);button.setAttribute('aria-pressed',String(i===state.diagram));});
  if(!d){$('graph-authority').textContent='NO FROZEN WORLD';$('graph-caption').textContent='No world diagram was recorded. Missing construction is not replaced by an invented graph.';$('graph-canvas').append(node('p','No recorded diagram available.','empty'));$('mermaid-source').textContent='';return;}
  $('graph-authority').textContent=d.authority==='CANDIDATE'?'CANDIDATE / NOT SENT TO AGENTS':'FROZEN WORLD';
  $('graph-caption').textContent=d.authority==='CANDIDATE'?'An admitted candidate interpretation. No unique world was frozen, and this graph was not sent to agents.':d.sent_to_agents?'This is the recorded frozen world supplied to framework assessment. Action branches are alternatives.':'A frozen world was recorded; framework assessment is not recorded for this run.';
  const mermaidSource=(state.fullLinks?d.mermaid:(d.compact_mermaid||d.mermaid))+(d.source_ledger_mermaid?'\n'+d.source_ledger_mermaid:'');
  $('graph-detail').textContent=state.fullLinks?'Show compact links':'Show full links';$('graph-detail').setAttribute('aria-pressed',String(state.fullLinks));
  const omitted=d.hidden_membership_edges?.length||0;
  $('graph-simplification').textContent=!state.fullLinks&&omitted?`Compact view hides ${omitted} action–effect membership lines already connected through recorded relations. All nodes and original records remain available. Actions and direct effects are separate identities even when their wording matches.`:'Full recorded topology. Actions and direct effects may share wording but have separate identities.';
  $('mermaid-source').textContent=mermaidSource;
  const records=recordsFor(d);for(const item of records){const b=node('button',item.id+' · '+item.label);b.addEventListener('click',()=>inspect(item));$('record-list').append(b);}
  try{
    if(!window.mermaid)throw new Error('The local Mermaid renderer is unavailable. Use the record list and Mermaid source below.');
    const result=await window.mermaid.render('world-render-'+generation,mermaidSource);
    if(generation!==state.render)return;
    $('graph-canvas').innerHTML=result.svg;
    $('graph-canvas').querySelectorAll('g.node').forEach(g=>{const id=g.getAttribute('data-id')||g.id.replace(/^flowchart-/,'').replace(/-\d+$/,'');const item=records.find(r=>r.id===id);if(item){g.setAttribute('tabindex','0');g.setAttribute('role','button');g.setAttribute('aria-label','Inspect '+item.label);g.addEventListener('click',()=>inspect(item));g.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();inspect(item);}});}});
    $('graph-canvas').querySelectorAll('.edgePath, .flowchart-link').forEach(edge=>{const target=edge.getAttribute('id')||edge.querySelector('path')?.id||'';const item=records.find(r=>r.kind==='Relation'&&(target.startsWith('L-'+r.record.source_id+'-'+r.record.target_id+'-')||(edge.classList.contains('LS-'+r.record.source_id)&&edge.classList.contains('LE-'+r.record.target_id))));if(item){edge.style.cursor='pointer';edge.addEventListener('click',()=>inspect(item));}});
    applyZoom();
  }catch(error){if(generation===state.render)$('graph-canvas').replaceChildren(node('p','Diagram rendering failed: '+error.message+' The original records and Mermaid source remain available.','error'));}
}

function field(card,label,value){const p=node('p');p.append(node('span',label,'field-label'),document.createTextNode(value||'Not separately recorded.'));card.append(p);}
function renderFrameworks(){
  const target=$('framework-cards');clear(target);const rows=state.run.frameworks;
  if(!rows.length){target.append(node('div','No agents received a unique frozen world. There are no framework positions to compare.','empty'));return;}
  const grid=node('div',undefined,'framework-grid');
  rows.forEach(view=>{const card=node('article',undefined,'card framework-card');const top=node('div',undefined,'framework-top');top.append(node('h2',view.framework),badge(view.position));card.append(top,node('div','Native ledger: '+view.ledger_status,'ledger'));
    field(card,'Recorded preference',view.preference?'“'+view.preference+'” — not necessarily operative.':'Withheld.');card.lastChild.classList.add('preference');
    field(card,'Main recorded reason',view.reason);field(card,'Important assumption',(view.assumptions||[]).join('; '));
    field(card,'Live objection / uncertainty',(view.live_objections||[]).join('; ')||'None separately recorded; this is not independent verification.');
    field(card,'Reversal condition',(view.reversal_conditions||[]).join('; '));
    const details=node('details');details.append(node('summary','Recorded dependencies'));const holder=node('div');list(holder,view.dependencies||[]);details.append(holder);card.append(details);grid.append(card);});target.append(grid);
}

function renderHinges(){
  const r=state.run;clear($('hinge-cards'));
  if(!r.hinges.length)$('hinge-cards').append(node('div','No explicit decision hinge was recorded. The viewer does not generate one to fill the gap.','empty'));
  r.hinges.forEach((h,i)=>{const card=node('article',undefined,'card hinge-card');const body=node('div');body.append(node('span',h.framework.toUpperCase(),'field-label'),node('p',h.text,'hinge-text'),node('small','Model-reported · not independently verified · '+h.record));card.append(node('div',String(i+1).padStart(2,'0'),'hinge-number'),body);$('hinge-cards').append(card);});
  list($('unknown-list'),r.unknowns,'No additional uncertainty recorded. Missing information is not evidence of completeness.');list($('nondecisions'),r.not_decided);
  clear($('agreement'));$('agreement').append(node('p',r.agreement||'No agreement assessment recorded.'));
  const ul=node('ul');for(const d of r.disagreements||[])ul.append(node('li',d.framework+' · '+d.kind+' dependency: '+d.dependency));$('agreement').append(ul);
  if(r.status==='INTERPRETIVE_BEAM')$('agreement').append(node('p','The unresolved disagreement is between semantic interpretations. No framework agreement has been tested.'));
}

document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.view)));
document.querySelector('.view-nav').addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();const names=['world','frameworks','hinges'],i=names.indexOf(state.view);const next=names[(i+(event.key==='ArrowRight'?1:2))%3];setView(next);$('tab-'+next).focus();});
$('graph-detail').addEventListener('click',()=>{state.fullLinks=!state.fullLinks;state.zoom=1;renderDiagram();});
$('zoom-in').addEventListener('click',()=>{state.zoom=Math.min(5,state.zoom*1.25);applyZoom();});$('zoom-out').addEventListener('click',()=>{state.zoom=Math.max(.3,state.zoom/1.25);applyZoom();});$('zoom-reset').addEventListener('click',()=>{state.zoom=1;applyZoom();$('graph-viewport').scrollTo(0,0);});window.addEventListener('resize',applyZoom);
$('download-run').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(state.run,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=state.run.id+'-parliament-view.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('import-run').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>20*1024*1024)throw new Error('Run JSON must be smaller than 20 MB.');const data=validateRun(JSON.parse(await file.text()));state.cache.set(data.id,data);state.version++;state.index=state.index.filter(e=>e.id!==data.id);state.index.push({id:data.id,title:data.title,status:data.status});state.run=data;state.diagram=0;state.zoom=1;notice('');renderRun();}catch(error){notice('Could not open run: '+error.message);}finally{event.target.value='';}});
(async()=>{try{if(window.mermaid)window.mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{primaryColor:'#edf2e7',primaryTextColor:'#25332e',primaryBorderColor:'#88a081',lineColor:'#758575',fontFamily:'system-ui, sans-serif',fontSize:'12px'},flowchart:{htmlLabels:false,useMaxWidth:false}});const response=await fetch('data/runs.json');if(!response.ok)throw new Error('Saved-run index could not be loaded.');const index=await response.json();if(index.schema_version!=='parliament-view-index/1')throw new Error('Unsupported run index.');state.index=index.runs;renderRail();if(state.index.length)await loadRun(state.index[0]);}catch(error){notice(error.message+' Start the viewer with python serve_parliament_viewer.py.');}})();
