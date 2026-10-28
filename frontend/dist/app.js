'use strict';
const $ = id => document.getElementById(id);
const state = {run:null, diagram:0, view:'world', zoom:1, fullLinks:false, version:0, render:0, cache:new Map(), index:[], topoRot:0, topoPick:null};
const VIEWS = ['world','frameworks','hinges','topography'];
const FRAMEWORK_COLORS = {utilitarian:'#42664e',deontological:'#825917',care:'#675677',virtue:'#3d5a80',rawlsian:'#9b3c32',contractualist:'#167174'};
const svgEl = (name, attrs, text) => {const n=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=String(text);return n;};
const node = (tag, text, cls) => {const n=document.createElement(tag); if(text!==undefined)n.textContent=String(text); if(cls)n.className=cls; return n;};
const clear = n => n.replaceChildren();
const notice = message => {$('notice').textContent=message||''; $('notice').hidden=!message;};
const list = (target, values, empty='None separately recorded.') => {clear(target); if(!values?.length){target.append(node('p',empty)); return;} const ul=node('ul'); values.forEach(v=>ul.append(node('li',typeof v==='string'?v:JSON.stringify(v)))); target.append(ul);};
const badge = status => node('span',status.replaceAll('_',' '),'badge '+(status.includes('BEAM')?'beam':status.includes('FAILED')?'failure':status==='FULL'||status==='FROZEN'||status==='COMPLETED'?'':'partial'));

function validateRun(run){
  if(!run || run.schema_version!=='parliament-view/1')throw new Error('Unsupported saved-run version. Open a parliament-view/1 export.');
  if(typeof run.source!=='string'||typeof run.id!=='string'||typeof run.title!=='string'||typeof run.status!=='string')throw new Error('Run identity, source or status is missing.');
  for(const field of ['diagrams','frameworks','hinges','unknowns','not_decided'])if(!Array.isArray(run[field]))throw new Error('Missing record array: '+field);
  for(const view of run.frameworks){
    if(!view||typeof view.framework!=='string'||!['FULL','ATTENUATED','ABSTAIN','CLARIFICATION'].includes(view.position))throw new Error('A framework participation record is incomplete.');
    if(view.critical_questions!=null&&(!Array.isArray(view.critical_questions)||view.critical_questions.some(q=>!q||['question','kind','role','why_it_matters','needed_to_resolve'].some(k=>typeof q[k]!=='string')||!Array.isArray(q.person_ids)||q.person_ids.some(id=>typeof id!=='string')||!Array.isArray(q.conditional_implications)||q.conditional_implications.some(c=>!c||typeof c.assumption!=='string'||typeof c.implication!=='string'))))throw new Error('A critical question record is incomplete.');
  }
  for(const h of run.hinges)if(!h||typeof h.text!=='string'||typeof h.framework!=='string'||typeof h.record!=='string')throw new Error('A decision hinge lacks its originating record.');
  for(const field of ['unknowns','not_decided'])if(run[field].some(v=>typeof v!=='string'))throw new Error('Expected text records in '+field);
  for(const d of run.diagrams){
    if(!d.world||typeof d.mermaid!=='string'||typeof d.label!=='string'||!['FROZEN','CANDIDATE'].includes(d.authority))throw new Error('A world diagram is incomplete.');
    if(d.authority==='CANDIDATE'&&d.sent_to_agents)throw new Error('A candidate diagram cannot claim agent participation.');
  }
  if(run.status==='INTERPRETIVE_BEAM'&&run.diagrams.some(d=>d.authority==='FROZEN'))throw new Error('An interpretive beam cannot contain a selected frozen world.');
  if(run.topography!=null){
    if(typeof run.topography!=='object'||Array.isArray(run.topography))throw new Error('Topography must be a record object.');
    if(!Array.isArray(run.topography.actions)||!Array.isArray(run.topography.framework_arrows||[]))throw new Error('Topography actions are missing.');
  }
  return run;
}

function renderRail(){
  const count=document.querySelector('.rail-label span');if(count)count.textContent=String(state.index.length).padStart(2,'0');
  clear($('run-list'));
  state.index.forEach(entry=>{const b=node('button',undefined,'run-link'+(state.run?.id===entry.id?' selected':'')); b.setAttribute('aria-current',state.run?.id===entry.id?'page':'false'); b.append(node('span',entry.title),node('small',entry.status.replaceAll('_',' '))); b.addEventListener('click',()=>loadRun(entry)); $('run-list').append(b);});
}

async function loadRun(entry){
  const request=++state.version;
  try{
    notice('');
    let data=state.cache.get(entry.id);
    if(!data){if(!/^[a-zA-Z0-9_-]+\.json$/.test(entry.file))throw new Error('Invalid saved-run filename.'); const response=await fetch('data/'+entry.file); if(!response.ok)throw new Error('Saved run could not be loaded.'); const bytes=await response.arrayBuffer(); data=validateRun(JSON.parse(new TextDecoder().decode(bytes))); try {const {exportReceipt}=await import('./provenance-receipts.mjs'); data.export_receipt=await exportReceipt(data,entry,bytes);} catch(error) {data.export_receipt={status:'UNAVAILABLE',reason:'Receipt verification could not be loaded: '+error.message};} state.cache.set(entry.id,data);}
    if(request!==state.version)return;
    state.run=data; state.diagram=0; state.zoom=1; state.topoRot=0; state.topoPick=null; await renderRun();
  }catch(error){if(request===state.version)notice(error.message);}
}

function renderProvenance(){
  const r=state.run, summary=$('provenance-summary'), list=$('checkpoint-list');clear(summary);clear(list);
  const rows=[['Original generation',r.generation_provenance],['Report or replay',r.run_provenance],...(Array.isArray(r.presentation_revisions)?r.presentation_revisions:[]).map(x=>['Later report revision',x]),['Viewer export',r.export_receipt?.manifest||r.viewer_provenance?.execution]];
  for(const [label,record] of rows){
    const row=node('div',undefined,'provenance-row');row.append(node('strong',label));
    if(!record?.run_id){row.append(node('span','Unknown historical provenance'));}
    else{
      row.append(node('span',record.run_id),node('span',record.mode||'Mode not recorded'));
      const app=record.application||{};row.append(node('small','Commit '+(app.commit||'unknown').slice(0,12)+' · source '+(app.source_sha256||'unknown').slice(0,12)+(app.dirty?' · uncommitted work captured':'')));
      if(Array.isArray(record.checkpoint_tags)&&record.checkpoint_tags.length)row.append(node('span','Tags: '+record.checkpoint_tags.join(', ')));
    }
    summary.append(row);
  }
  const receipt=r.export_receipt||{status:'UNAVAILABLE',reason:'No receipt was loaded for this imported record.'};
  summary.append(node('p',receipt.status+': '+receipt.reason));
  if(receipt.renderer)summary.append(node('p','Served renderer files: '+receipt.renderer.status+(receipt.renderer.changed.length?' · changed: '+receipt.renderer.changed.join(', '):'')));
  if(receipt.url&&receipt.status==='MATCHED'){const a=node('a','Download export manifest');a.href=receipt.url;a.download='';summary.append(a);}
  const events=x=>Array.isArray(x?.checkpoints)?x.checkpoints.filter(c=>c&&typeof c.id==='string'):[];
  const revisions=Array.isArray(r.presentation_revisions)?r.presentation_revisions:[];
  const checkpoints=[...events(r.generation_provenance),...events(r.run_provenance),...revisions.flatMap(events),...events(receipt.manifest||r.viewer_provenance?.execution)];
  const unique=Array.from(new Map(checkpoints.map(x=>[x.id,x])).values());
  list.append(node('h3','Recorded checkpoints'));
  if(!unique.length)list.append(node('p','No checkpoint receipts were recorded for these historical stages.'));
  else {const ul=node('ul');for(const cp of unique)ul.append(node('li',cp.stage+' · '+cp.status+' · '+cp.at+(Array.isArray(cp.tags)&&cp.tags.length?' · '+cp.tags.join(', '):'')+' · '+cp.run_id));list.append(ul);}
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
  for(const name of VIEWS){$('view-'+name).hidden=name!==view; $('tab-'+name).setAttribute('aria-selected',String(name===view)); $('tab-'+name).tabIndex=name===view?0:-1;}
  if(view==='world')scheduleFit();
}

async function renderRun(){
  const r=state.run; renderRail(); $('run-title').textContent=r.title;
  $('run-subtitle').textContent=r.mode==='SAVED_REPLAY_NO_API'?'Saved replay · no new model calls · original records remain unchanged.':'Saved record · inspect attributed outputs and their limits.';
  $('download-run').disabled=false; clear($('status-strip'));
  const stopReason=r.stop?.reason===r.status&&r.status==='FRAMEWORK_PARTIAL'?'Some framework positions remain provisional or abstaining.':r.stop?.reason||'No stop reason recorded.';
  $('status-strip').append(badge(r.status),node('span',stopReason),node('span','Native judgment: '+(r.judgment_status||'not recorded')));
  $('run-provenance-record').textContent=JSON.stringify({original_generation:r.generation_provenance||{status:'UNKNOWN_HISTORICAL'},report_execution:r.run_provenance||{status:'UNKNOWN_HISTORICAL'},later_revisions:r.presentation_revisions||[],viewer:r.viewer_provenance||{status:'UNKNOWN_HISTORICAL'}},null,2);
  renderProvenance();
  $('source-hash').textContent='Source '+(r.source_sha256||'unhashed').slice(0,12);
  sourceHighlight(); clear($('stipulations'));
  if(r.stipulated_claims?.length){const details=node('details',undefined,'stipulations');details.append(node('summary','Source stipulations · '+r.stipulated_claims.length));const ul=node('ul');r.stipulated_claims.forEach(p=>ul.append(node('li',p.claim||'Unnamed source claim')));details.append(ul);$('stipulations').append(details);}
  clear($('diagram-tabs'));r.diagrams.forEach((d,i)=>{const button=node('button',d.label,i===state.diagram?'active':'');button.setAttribute('aria-pressed',String(i===state.diagram));button.addEventListener('click',()=>{state.diagram=i;state.zoom=1;renderDiagram();});$('diagram-tabs').append(button);});
  await renderDiagram();renderFrameworks();renderHinges();renderTopography();setView(state.view);
}

function recordsFor(d){
  const result=[];
  for(const [array,key,kind] of [['parties','party_id','Party'],['actions','action_id','Action'],['effects','effect_id','Effect'],['conditions','condition_id','Condition'],['propositions','proposition_id','Proposition'],['commitments','commitment_id','Commitment'],['reports','report_id','Report'],['modal_operators','modal_id','Modal operator'],['normative_propositions','normative_proposition_id','Normative proposition']])
    for(const record of d.world[array]||[]){const id=record[key]||record.modal_operator_id||record.id; if(id)result.push({id,kind,record,label:record.label||record.intervention||record.outcome||record.description||record.claim||record.predication||record.force||id});}
  for(const [i,record] of (d.world.causal_links||[]).entries())result.push({id:'relation:'+i,kind:'Relation',record,label:record.source_id+' '+record.relation+' '+record.target_id});
  for(const item of d.source_ledger_nodes||[])result.push({id:item.id,kind:'Source ledger addition',record:item.record,label:item.record.claim});
  for(const [id, ids] of Object.entries(d.reader_bundles||{})){
    const item=result.find(r=>r.id===id);
    if(item)item.bundledRecords=ids.map(key=>result.find(r=>r.id===key)).filter(Boolean).map(r=>({id:r.id,kind:r.kind,record:r.record}));
  }
  for(const n of (state.fullLinks?[]:(d.typed_reader_view?.nodes||[]))){
    const originals=n.world_ids.map(id=>result.find(x=>x.id===id)).filter(Boolean);
    result.push({id:n.id,kind:'Reader node',label:n.label,record:n,bundledRecords:originals.map(x=>({id:x.id,kind:x.kind,record:x.record}))});
  }
  for(const edge of (state.fullLinks?[]:(d.typed_reader_view?.edges||[])))result.push({id:edge.id,kind:'Reader relation',record:edge,label:edge.source+' '+edge.type+' '+edge.target});
  return result;
}

function inspect(item){
  const r=item.record;clear($('inspector'));
  const left=node('div');left.append(node('div',item.kind.toUpperCase()+' / '+item.id,'record-kind'),node('h2',item.label));
  const facts=node('div',undefined,'facts');for(const key of ['kind','effect_kind','modality','polarity','directness','relation','condition_join','derivation_operation'])if(r[key])facts.append(node('span',key.replaceAll('_',' ')+': '+r[key]));left.append(facts);
  const excerpts=(r.provenance||[]).filter(p=>p&&typeof p==='object').map(p=>p.excerpt).filter(e=>typeof e==='string'&&e.length);
  const exact=excerpts.filter(e=>state.run.source.includes(e));
  if(item.kind==='Proposition'&&typeof r.predication==='string'&&state.run.source.includes(r.predication)){
    excerpts.push(r.predication);exact.push(r.predication);
  }
  if(item.kind==='Reader node'||item.kind==='Reader relation'){
    left.append(node('p','Reader projection: display roles and source readings do not change the admitted world. Dashed relations and physical gaps have no extra world-state authority.'));
    if(r.evidence?.text)excerpts.push(r.evidence.text);
    for(const original of item.bundledRecords||[])for(const p of original.record.provenance||[])if(typeof p.excerpt==='string'&&!excerpts.includes(p.excerpt))excerpts.push(p.excerpt);
  }
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
  if(item.bundledRecords?.length){const bundle=node('details');bundle.append(node('summary','Bundled action and direct effect · '+item.bundledRecords.map(r=>r.id).join(', ')),node('pre',JSON.stringify(item.bundledRecords,null,2)));right.append(bundle);}
  $('inspector').append(left,right);sourceHighlight(exact);
}

function emptyInspector(){clear($('inspector'));$('inspector').append(node('div','Select a node or relation to inspect its source, provenance and scope.'),node('p','Use the record list if you prefer keyboard navigation. Source claims and agent arguments remain separate.'));}

function svgBox(svg){
  const vb=svg.viewBox&&svg.viewBox.baseVal;
  if(vb&&vb.width&&vb.height)return {width:vb.width,height:vb.height};
  const parts=(svg.getAttribute('viewBox')||'').trim().split(/[\s,]+/).map(Number);
  if(parts.length===4&&parts[2]>0&&parts[3]>0)return {width:parts[2],height:parts[3]};
  return null;
}

function applyZoom(){
  const canvas=$('graph-canvas'), viewport=$('graph-viewport'), panel=$('view-world');
  const svg=canvas&&canvas.querySelector('svg');
  if(!svg||!viewport||(panel&&panel.hidden))return false;
  const box=svgBox(svg), width=viewport.clientWidth, height=viewport.clientHeight;
  if(!box||width<40||height<40)return false;
  const scale=Math.max(Math.min((width-44)/box.width,(height-44)/box.height),0.05)*state.zoom;
  svg.removeAttribute('width');svg.removeAttribute('height');
  svg.style.width=box.width*scale+'px';svg.style.height=box.height*scale+'px';
  svg.style.maxWidth='none';svg.style.maxHeight='none';
  return true;
}

function scheduleFit(){
  let tries=0;
  const attempt=()=>{if(applyZoom()||++tries>24)return;requestAnimationFrame(attempt);};
  requestAnimationFrame(attempt);
}

async function renderDiagram(){
  const generation=++state.render, d=state.run.diagrams[state.diagram];clear($('graph-canvas'));clear($('record-list'));clear($('graph-context'));emptyInspector();sourceHighlight();
  [...$('diagram-tabs').children].forEach((button,i)=>{button.classList.toggle('active',i===state.diagram);button.setAttribute('aria-pressed',String(i===state.diagram));});
  if(!d){$('graph-authority').textContent='NO FROZEN WORLD';$('graph-caption').textContent='No world diagram was recorded. Missing construction is not replaced by an invented graph.';$('graph-canvas').append(node('p','No recorded diagram available.','empty'));$('mermaid-source').textContent='';return;}
  $('graph-authority').textContent=d.authority==='CANDIDATE'?'CANDIDATE / NOT SENT TO AGENTS':'FROZEN WORLD';
  $('graph-caption').textContent=d.authority==='CANDIDATE'?'An admitted candidate interpretation. No unique world was frozen, and this graph was not sent to agents.':d.sent_to_agents?'This is the recorded frozen world supplied to framework assessment. Action branches are alternatives.':'A frozen world was recorded; framework assessment is not recorded for this run.';
  const ledgerLayer=state.fullLinks?d.source_ledger_mermaid:(d.compact_source_ledger_mermaid||d.source_ledger_mermaid);
  const typed=!state.fullLinks&&d.typed_reader_view;
  const mermaidSource=(state.fullLinks?d.mermaid:(typed?.mermaid||d.compact_mermaid||d.mermaid))+(ledgerLayer?'\n'+ledgerLayer:'');
  $('graph-detail').textContent=state.fullLinks?'Show compact graph':'Show full graph';$('graph-detail').setAttribute('aria-pressed',String(state.fullLinks));
  const omitted=d.hidden_membership_edges?.length||0;
  $('graph-simplification').textContent=!state.fullLinks&&omitted?`Compact view hides ${omitted} action–effect membership lines already connected through recorded relations. All nodes and original records remain available. Actions and direct effects are separate identities even when their wording matches.`:'Full recorded topology. Actions and direct effects may share wording but have separate identities.';
  if(!state.fullLinks&&Object.keys(d.reader_bundles||{}).length)$('graph-simplification').textContent+=' Matching action/direct-effect records share a display box; select it to inspect both IDs. Physical mechanisms and human outcomes remain separate.';
  if(typed){$('graph-authority').textContent=d.authority==='CANDIDATE'?'CANDIDATE + READER INTERPRETATION':'FROZEN WORLD + READER INTERPRETATION';$('graph-caption').textContent+=' Typed reader projection; dashed relations are source readings, not additional frozen-world edges.';$('graph-simplification').textContent=typed.notice;}
  const legend=$('graph-legend');if(legend){legend.children[0].textContent=typed?'Solid: saved role/link projected into the reader view':'Solid: recorded relation';legend.children[1].textContent=typed?'Dashed: source reading or unresolved structure':'Dashed: role, effect or scope';}
  $('mermaid-source').textContent=mermaidSource;
  const records=recordsFor(d);for(const item of records){const b=node('button',item.id+' · '+item.label);b.addEventListener('click',()=>inspect(item));$('record-list').append(b);}
  if(!state.fullLinks&&d.reader_context_ids?.length){
    $('graph-context').append(node('h3','Recorded scene & constraints'));
    for(const id of d.reader_context_ids){const item=records.find(r=>!state.fullLinks&&r.id===id&&r.kind==='Reader node')||records.find(r=>r.id===id);if(item){const b=node('button',item.label);b.addEventListener('click',()=>inspect(item));$('graph-context').append(b);}}
  }
  try{
    if(!window.mermaid)throw new Error('The local Mermaid renderer is unavailable. Use the record list and Mermaid source below.');
    const result=await window.mermaid.render('world-render-'+generation,mermaidSource);
    if(generation!==state.render)return;
    $('graph-canvas').innerHTML=result.svg;
    $('graph-canvas').querySelectorAll('g.node').forEach(g=>{const id=g.getAttribute('data-id')||g.id.replace(/^flowchart-/,'').replace(/-\d+$/,'');const item=records.find(r=>!state.fullLinks&&r.id===id&&r.kind==='Reader node')||records.find(r=>r.id===id);if(item){g.setAttribute('tabindex','0');g.setAttribute('role','button');g.setAttribute('aria-label','Inspect '+item.label);g.addEventListener('click',()=>inspect(item));g.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();inspect(item);}});}});
    const aliases=state.fullLinks?{}:Object.fromEntries(Object.entries(d.reader_bundles||{}).map(([id,ids])=>[ids[1],id]));
    const displayedId=id=>aliases[id]||id;
    $('graph-canvas').querySelectorAll('.edgePath, .flowchart-link').forEach(edge=>{const target=edge.getAttribute('id')||edge.querySelector('path')?.id||'';const typedItem=!state.fullLinks&&records.find(r=>r.kind==='Reader relation'&&target.startsWith('L-'+r.record.source+'-'+r.record.target+'-'));const item=typedItem||records.find(r=>r.kind==='Relation'&&(target.startsWith('L-'+displayedId(r.record.source_id)+'-'+displayedId(r.record.target_id)+'-')||(edge.classList.contains('LS-'+displayedId(r.record.source_id))&&edge.classList.contains('LE-'+displayedId(r.record.target_id)))));if(item){edge.style.cursor='pointer';edge.addEventListener('click',()=>inspect(item));}});
    scheduleFit();
  }catch(error){if(generation===state.render)$('graph-canvas').replaceChildren(node('p','Diagram rendering failed: '+error.message+' The original records and Mermaid source remain available.','error'));}
}

function field(card,label,value){const p=node('p');p.append(node('span',label,'field-label'),document.createTextNode(value||'Not separately recorded.'));card.append(p);}
function renderCriticalQuestions(card,view){
  if(!Array.isArray(view.critical_questions)||!view.critical_questions.length)return;
  const box=node('section',undefined,'critical-questions');box.append(node('h3','Critical unanswered questions'));
  for(const q of view.critical_questions){
    const item=node('div',undefined,'critical-question');
    item.append(node('span',q.kind.toLowerCase()+' · '+q.role.replaceAll('_',' ').toLowerCase(),'field-label'),node('p',q.question,'question-text'));
    field(item,'People affected',q.person_ids.map(id=>view.person_labels?.[id]||id).join('; '));
    field(item,'Why it matters',q.why_it_matters);
    field(item,'What could help',q.needed_to_resolve);
    if(q.conditional_implications.length){const details=node('details');details.append(node('summary','How different answers would change the comparison'));const ul=node('ul');for(const c of q.conditional_implications)ul.append(node('li','If '+c.assumption+': '+c.implication));details.append(ul);item.append(details);}
    box.append(item);
  }
  box.append(node('p','Agent-proposed questions and conditional arguments; no answer or vote is inferred.','ledger'));card.append(box);
}
function renderFrameworks(){
  const target=$('framework-cards');clear(target);const rows=state.run.frameworks;
  if(!rows.length){target.append(node('div','No agents received a unique frozen world. There are no framework positions to compare.','empty'));return;}
  if(state.run.agent_roster?.replaced_for_this_run==='rawlsian')target.append(node('p','This run uses Contractualist reasoning in place of Rawls. All five perspectives received the same frozen world in independent passes. Critical questions are a contribution, not a forced vote.','framework-roster'));
  const grid=node('div',undefined,'framework-grid');
  rows.forEach(view=>{const card=node('article',undefined,'card framework-card');card.id='framework-'+view.framework;card.dataset.framework=view.framework;const top=node('div',undefined,'framework-top');top.append(node('h2',view.framework==='contractualist'?'Contractualist':view.framework),badge(view.position));card.append(top,node('div',view.ledger_status==='NOT_APPLICABLE_ARGUMENT_ADAPTER'?'Argument adapter · no native ledger':('Native ledger: '+view.ledger_status),'ledger'));
    field(card,'Recorded preference',view.preference?'“'+view.preference+'” — not necessarily operative.':view.position==='CLARIFICATION'?'No ranking claimed; the agent clarifies what remains unanswered.':'Withheld.');card.lastChild.classList.add('preference');
    field(card,view.reason_authority==='REVIEWED_PLAIN_LANGUAGE_SUMMARY_OF_SAVED_ARGUMENT'?'Explanation of the saved argument':'Main recorded reason',view.reason);
    if(view.reason_authority==='REVIEWED_PLAIN_LANGUAGE_SUMMARY_OF_SAVED_ARGUMENT')card.append(node('p','Reviewed plain-language summary; the original agent wording is preserved in the full record.','ledger'));
    if(view.argument_url&&/^data\/[a-z_]+_argument\.md$/.test(view.argument_url)){const a=node('a','Full recorded explanation (Markdown)');a.href=view.argument_url;a.target='_blank';a.rel='noopener';card.append(a);}
    renderCriticalQuestions(card,view);
    if(view.assumption_records?.length){const box=node('div');box.append(node('span','Assumptions and interpretive commitments','field-label'));const ul=node('ul');view.assumption_records.forEach(r=>ul.append(node('li',r.kind.replaceAll('_',' ').toLowerCase()+': '+r.text)));box.append(ul);card.append(box);}
    else field(card,'Important assumption',(view.assumptions||[]).join('; '));
    if(view.source_basis?.text)field(card,'What is given, and how it is known',view.source_basis.text);
    field(card,'Live objection / uncertainty',(view.display_uncertainties||view.live_objections||[]).join('; ')||'None separately recorded; this is not independent verification.');
    if(view.preference){const box=node('div');box.append(node('span','Would reverse the recorded preference','field-label'));const values=view.display_reversals||view.reversal_explanations||view.reversal_conditions||[];if(values.length){const ul=node('ul');values.forEach(text=>ul.append(node('li',text)));box.append(ul);}else box.append(node('p','Not separately recorded.'));card.append(box);}
    else field(card,view.position==='CLARIFICATION'?'What could resolve the comparison':'Abstention boundary',(view.decision_boundaries||[]).join('; ')||'No recorded condition that would resolve the withheld ranking. This is not the reversal of a preference that was never admitted.');
    for(const note of view.presentation_notices||[])card.append(node('p',note,'ledger'));
    const details=node('details');details.append(node('summary','Recorded dependencies'));const holder=node('div');list(holder,view.dependencies||[]);details.append(holder);card.append(details);grid.append(card);});target.append(grid);
}

function renderHinges(){
  const r=state.run;clear($('hinge-cards'));
  if(!r.hinges.length)$('hinge-cards').append(node('div','No explicit decision hinge was recorded. The viewer does not generate one to fill the gap.','empty'));
  r.hinges.forEach((h,i)=>{const card=node('article',undefined,'card hinge-card');const body=node('div');const role=h.role==='OPEN_QUESTION'?'Critical unanswered question · no vote inferred':h.role==='ABSTENTION_BOUNDARY'?'Abstention boundary · no ranking to reverse':'Would reverse a recorded preference';body.append(node('span',h.framework.toUpperCase(),'field-label'),node('p',h.text,'hinge-text'),node('small',role+' · model-reported · not independently verified · '+h.record));card.append(node('div',String(i+1).padStart(2,'0'),'hinge-number'),body);$('hinge-cards').append(card);});
  list($('unknown-list'),r.unknowns,'No additional uncertainty recorded. Missing information is not evidence of completeness.');list($('nondecisions'),r.not_decided);
  clear($('agreement'));$('agreement').append(node('p',r.agreement||'No agreement assessment recorded.'));
  const ul=node('ul');for(const d of r.disagreements||[])ul.append(node('li',d.framework+' · '+d.kind+' dependency: '+d.dependency));$('agreement').append(ul);
  if(r.status==='INTERPRETIVE_BEAM')$('agreement').append(node('p','The unresolved disagreement is between semantic interpretations. No framework agreement has been tested.'));
}

function welfareRank(point){return (point.direction==='HARM'||point.unit==='DEATHS')?-point.value:point.value;}
function axisCoordinate(point, fallback){
  if(!point||!point.present)return null;
  if(typeof point.probability==='number')return point.probability;
  return fallback(point);
}
function axisInterval(point){
  const interval=point&&point.interval;
  if(interval&&typeof interval.low==='number'&&typeof interval.high==='number')return [interval.low, interval.high];
  return null;
}
function drawErrorCross(svg, x, y, x0, x1, y0, y1){
  const g=svgEl('g',{class:'topo-interval'});
  if(Math.abs(x1-x0)>1e-9){
    g.append(svgEl('line',{x1:x0,y1:y,x2:x1,y2:y,stroke:'#8a9584','stroke-width':'1.2'}));
    g.append(svgEl('line',{x1:x0,y1:y-4,x2:x0,y2:y+4,stroke:'#8a9584','stroke-width':'1.2'}));
    g.append(svgEl('line',{x1:x1,y1:y-4,x2:x1,y2:y+4,stroke:'#8a9584','stroke-width':'1.2'}));
  }
  if(Math.abs(y1-y0)>1e-9){
    g.append(svgEl('line',{x1:x,y1:y0,x2:x,y2:y1,stroke:'#8a9584','stroke-width':'1.2'}));
    g.append(svgEl('line',{x1:x-4,y1:y0,x2:x+4,y2:y0,stroke:'#8a9584','stroke-width':'1.2'}));
    g.append(svgEl('line',{x1:x-4,y1:y1,x2:x+4,y2:y1,stroke:'#8a9584','stroke-width':'1.2'}));
  }
  if(g.childElementCount)svg.append(g);
}
function drawIntervalBox(svg, item, ox, oy, scale, rot){
  const welfare=axisInterval(item.action.welfare), care=axisInterval(item.action.care), obligation=axisInterval(item.action.obligation);
  if(!welfare||!care||!obligation)return;
  if(welfare[1]-welfare[0]<1e-9&&care[1]-care[0]<1e-9&&obligation[1]-obligation[0]<1e-9)return;
  const corners=[];
  for(const w of welfare)for(const c of care)for(const o of obligation){
    const [x,y,z]=topoTriple({welfare:w,care:c,obligation:o},rot);
    corners.push(iso(x,y,z,ox,oy,scale));
  }
  const g=svgEl('g',{class:'topo-interval'});
  for(let i=0;i<8;i++){
    for(const bit of [1,2,4]){
      if(i&bit)continue;
      const j=i|bit;
      g.append(svgEl('line',{x1:corners[i][0],y1:corners[i][1],x2:corners[j][0],y2:corners[j][1],stroke:'#8a9584','stroke-width':'1',opacity:'0.7'}));
    }
  }
  svg.append(g);
}
function topoNorms(topo){
  return (topo.actions||[]).map(action=>{
    const care=axisCoordinate(action.care, point=>typeof point.ordinal==='number'?point.ordinal/2:null);
    const obligation=axisCoordinate(action.obligation, point=>typeof point.ordinal==='number'?point.ordinal/3:null);
    let welfare=axisCoordinate(action.welfare, point=>typeof point.value==='number'?point.value:null);
    if(welfare!=null&&action.welfare&&typeof action.welfare.probability!=='number'&&typeof action.welfare.value==='number'){
      const ranks=(topo.actions||[]).filter(row=>row.welfare?.present&&typeof row.welfare.probability!=='number').map(row=>welfareRank(row.welfare));
      const min=Math.min(...ranks), max=Math.max(...ranks);
      welfare=max===min?0.5:(welfareRank(action.welfare)-min)/(max-min);
    }
    return {action, care, welfare, obligation, hole:care==null||welfare==null||obligation==null};
  });
}
function interpolateVertical(x, z, samples){
  let num=0, den=0;
  for(const sample of samples){
    const d2=(x-sample.x)*(x-sample.x)+(z-sample.z)*(z-sample.z);
    if(d2<1e-12)return sample.y;
    const w=1/d2;
    num+=w*sample.y; den+=w;
  }
  return den?num/den:0;
}
function hasLandscape(topo){return topoNorms(topo||{}).filter(item=>!item.hole).length>=2;}
function drawLandscape(svg, items, ox, oy, scale, rot){
  const samples=items.filter(item=>!item.hole).map(item=>{
    const [x,y,z]=topoTriple(item, rot);
    return {x,y,z};
  });
  if(samples.length<2)return;
  const n=16;
  const grid=[];
  let ymin=Infinity, ymax=-Infinity;
  for(let j=0;j<=n;j++){
    const row=[];
    for(let i=0;i<=n;i++){
      const x=i/n, z=j/n;
      const y=Math.max(0, Math.min(1, interpolateVertical(x,z,samples)));
      row.push({x,y,z});
      if(y<ymin)ymin=y; if(y>ymax)ymax=y;
    }
    grid.push(row);
  }
  const quads=[];
  for(let j=0;j<n;j++)for(let i=0;i<n;i++){
    const cells=[grid[j][i], grid[j][i+1], grid[j+1][i+1], grid[j+1][i]];
    quads.push({cells, depth:cells.reduce((sum,cell)=>sum+cell.x+cell.z,0)/4, height:cells.reduce((sum,cell)=>sum+cell.y,0)/4});
  }
  quads.sort((a,b)=>b.depth-a.depth);
  const g=svgEl('g',{class:'topo-landscape','aria-hidden':'true'});
  const span=Math.max(1e-6, ymax-ymin);
  quads.forEach(quad=>{
    const t=(quad.height-ymin)/span;
    const fill='rgba('+(88+Math.round(t*40))+','+(118+Math.round(t*20))+','+(96-Math.round(t*18))+',0.42)';
    g.append(svgEl('polygon',{points:quad.cells.map(cell=>iso(cell.x,cell.y,cell.z,ox,oy,scale).join(',')).join(' '),fill,stroke:'#7d9276','stroke-width':'0.45'}));
  });
  svg.append(g);
}
function iso(x,y,z,ox,oy,scale){return [ox+(x-z)*scale, oy-y*scale*1.15-(x+z)*scale*0.5];}
function topoTriple(coords, rot){
  const x=coords.welfare, y=coords.care, z=coords.obligation;
  return [[x,y,z],[z,y,x],[y,x,z],[x,z,y]][rot%4];
}
function topoLabels(rot){return [{x:'Welfare',y:'Care',z:'Obligation'},{x:'Obligation',y:'Care',z:'Welfare'},{x:'Care',y:'Welfare',z:'Obligation'},{x:'Welfare',y:'Obligation',z:'Care'}][rot%4];}
function fillMissing(triple){return triple.map(v=>v==null?0:v);}

function inspectTopo(item){
  const target=$('topo-inspector');clear(target);state.topoPick=item.action.id;
  const left=node('div');left.append(node('div','ACTION LOCUS / '+item.action.id,'record-kind'),node('h2',item.action.label));
  const facts=node('div',undefined,'facts');
  for(const axis of ['care','welfare','obligation']){
    const point=item.action[axis]||{};
    const interval=point.interval;
    const span=interval&&typeof interval.low==='number'?' ['+interval.low.toFixed(3)+', '+interval.high.toFixed(3)+']':'';
    const label=point.present?(typeof point.probability==='number'?'P='+point.probability.toFixed(3)+span:(point.verdict||String(point.value))):'omitted';
    facts.append(node('span',axis+': '+label));
  }
  left.append(facts);
  for(const axis of ['care','welfare','obligation']){
    const point=item.action[axis]||{};
    left.append(node('h3',axis[0].toUpperCase()+axis.slice(1)));
    if(!point.present)left.append(node('p',point.omit_reason||'This axis was not recorded.'));
    else{
      if(typeof point.probability==='number')left.append(node('p','Decisions probability: '+point.probability.toFixed(3)+'. This is P(the action satisfies the axis criterion), given the Parliament packet.'));
      if(point.interval&&point.interval.method==='jackknife')left.append(node('p','Jackknife 95% CI ['+point.interval.low.toFixed(3)+', '+point.interval.high.toFixed(3)+'] from '+(point.interval.folds_used||0)+'/'+(point.interval.folds||10)+' packet-text ablations. Missing folds are omitted, not zero. The marker is the jackknife mean.'));
      const hinge=point.interval&&point.interval.hinging;
      if(hinge&&hinge.present){
        const delta=Number(hinge.delta_from_mean);
        const signed=(delta>=0?'+':'')+delta.toFixed(3);
        const present=hinge.effect_when_present==='raises_estimate'?'raises':'lowers';
        left.append(node('p','Most hinging tenth: fold '+hinge.fold+' (chars '+hinge.omitted_start+'–'+hinge.omitted_end+'). Leave-one-out P='+Number(hinge.loo_probability).toFixed(3)+' vs mean '+Number(hinge.mean).toFixed(3)+' (Δ='+signed+'). When present, this slice '+present+' the point estimate.'));
        const excerpt=hinge.omitted_text||(state.run.topography?.provenance?.jackknife?.packet_blocks||[]).find(block=>block.fold===hinge.fold)?.omitted_text;
        if(excerpt)left.append(node('pre',excerpt));
      }else if(hinge&&hinge.omit_reason)left.append(node('p',hinge.omit_reason));
      if(point.question)left.append(node('p',point.question));
      if(point.reason)left.append(node('p',point.reason));
      if(point.calculation)left.append(node('p',point.calculation));
      if(point.total_welfare==='UNKNOWN')left.append(node('p','Total welfare remains UNKNOWN. The plotted number is a restricted count endpoint, not a welfare unit.'));
      if(point.unresolved)left.append(node('p','Obligation is CONFLICTED: plotted as a mid-band, not treated as a unique maximum.'));
      const native=(item.action.native||{})[axis];
      if(native&&native.present)left.append(node('p','Native record (not the plotted coordinate): '+(native.verdict||String(native.value??native.ordinal))));
      const pre=node('pre',JSON.stringify(point,null,2));left.append(pre);
    }
  }
  const right=node('div');
  const flags=state.run.topography?.flags||{};
  if((flags.degenerate||[]).includes('care_welfare_survival_copy')||(flags.degenerate||[]).includes('care_welfare_probability_copy')){
    right.append(node('h3','Degenerate Care/welfare coincidence'));
    right.append(node('p','Care ranking matches the welfare ranking. This is occupancy of the same order under another name, not an attractor.'));
    const care=state.run.frameworks.find(v=>v.framework==='care');
    (care?.live_objections||[]).forEach(text=>right.append(node('p',text)));
  }
  if((flags.attractors||[]).includes(item.action.id))right.append(node('h3','Attractor occupancy'),node('p','This action is a recorded maximum on at least two non-copy axes.'));
  const arrows=(state.run.topography?.framework_arrows||[]).filter(a=>a.action_id===item.action.id);
  if(arrows.length){right.append(node('h3','Framework arrows here'));arrows.forEach(a=>right.append(node('p',a.framework+' → '+a.preference+' ('+(a.position||'unspecified')+')')));}
  right.append(node('p',hasLandscape(state.run?.topography)
    ?'The 3D composite interpolates a surface through the recorded action loci. That surface is display-only, not additional evidence. 2D projections and jackknife boxes stay recorded coordinates.'
    :'The scatter uses recorded coordinates only. It does not interpolate terrain between actions. Jackknife boxes are ablation intervals, not interpolated terrain.'));
  target.append(left,right);
  renderTopoDrawings();
}

function emptyTopoInspector(){clear($('topo-inspector'));$('topo-inspector').append(node('div','Select an action locus to inspect Care, welfare and obligation coordinates.'),node('p','Hollow markers sit on a face because at least one axis is omitted. A missing or refused Decisions probability is never drawn as zero.'));}

function drawCube(svg, ox, oy, scale, labels){
  const corners=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]].map(([x,y,z])=>iso(x,y,z,ox,oy,scale));
  const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  edges.forEach(([a,b])=>svg.append(svgEl('line',{x1:corners[a][0],y1:corners[a][1],x2:corners[b][0],y2:corners[b][1],stroke:'#c5cfbe','stroke-width':'1'})));
  const lx=iso(1.08,0,0,ox,oy,scale), ly=iso(0,1.12,0,ox,oy,scale), lz=iso(0,0,1.08,ox,oy,scale);
  svg.append(svgEl('text',{x:lx[0],y:lx[1],'font-size':'11',fill:'#42664e'},labels.x));
  svg.append(svgEl('text',{x:ly[0],y:ly[1],'font-size':'11',fill:'#675677'},labels.y));
  svg.append(svgEl('text',{x:lz[0],y:lz[1],'font-size':'11',fill:'#825917'},labels.z));
}

function drawLocus(svg, x, y, item, selected){
  const g=svgEl('g',{class:'topo-point',tabindex:'0',role:'button','aria-label':'Inspect '+item.action.label});
  const mark={cx:x,cy:y,r:selected?9:7,fill:item.hole?'#fff':'#25332e',stroke:selected?'#a8782f':'#25332e','stroke-width':selected?3:1.5};
  if(item.hole)mark['stroke-dasharray']='3 2';
  g.append(svgEl('circle',mark));
  g.append(svgEl('text',{x:x+12,y:y-8,'font-size':'11',fill:'#25332e'},item.action.id));
  g.addEventListener('click',()=>inspectTopo(item));
  g.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();inspectTopo(item);}});
  svg.append(g);
}

function renderTopo3d(target, topo){
  clear(target);
  const svg=svgEl('svg',{viewBox:'0 0 640 420',role:'group','aria-label':hasLandscape(topo)?'Isometric interpolated landscape of recorded action loci':'Isometric scatter of recorded action loci'});
  const ox=320, oy=330, scale=150;
  drawCube(svg,ox,oy,scale,topoLabels(state.topoRot));
  const items=topoNorms(topo);
  if(hasLandscape(topo))drawLandscape(svg,items,ox,oy,scale,state.topoRot);
  const arrows=topo.framework_arrows||[];
  arrows.forEach((arrow,i)=>{
    const item=items.find(row=>row.action.id===arrow.action_id); if(!item)return;
    const [x,y,z]=fillMissing(topoTriple(item,state.topoRot));
    const end=iso(x,y,z,ox,oy,scale);
    const start=iso(0.08+(i%3)*0.04,0.08,0.08+(Math.floor(i/3)*0.04),ox,oy,scale);
    svg.append(svgEl('line',{x1:start[0],y1:start[1],x2:end[0],y2:end[1],stroke:FRAMEWORK_COLORS[arrow.framework]||'#6c7770','stroke-width':'1.6',opacity:'0.85'}));
  });
  items.forEach(item=>drawIntervalBox(svg,item,ox,oy,scale,state.topoRot));
  items.forEach(item=>{
    const [x,y,z]=fillMissing(topoTriple(item,state.topoRot));
    const [sx,sy]=iso(x,y,z,ox,oy,scale);
    drawLocus(svg,sx,sy,item,state.topoPick===item.action.id);
  });
  target.append(svg);
}

function renderTopo2d(target, topo, xAxis, yAxis){
  clear(target);
  const svg=svgEl('svg',{viewBox:'0 0 320 260',role:'group','aria-label':xAxis+' versus '+yAxis});
  const left=46, top=18, width=250, height=190;
  svg.append(svgEl('line',{x1:left,y1:top+height,x2:left+width,y2:top+height,stroke:'#9aad91'}));
  svg.append(svgEl('line',{x1:left,y1:top,x2:left,y2:top+height,stroke:'#9aad91'}));
  svg.append(svgEl('text',{x:left+width/2,y:248,'text-anchor':'middle','font-size':'11',fill:'#6c7770'},xAxis));
  svg.append(svgEl('text',{x:14,y:top+height/2,transform:'rotate(-90 14 '+(top+height/2)+')','font-size':'11',fill:'#6c7770'},yAxis));
  const items=topoNorms(topo);
  const xPresent=items.some(item=>item[xAxis]!=null), yPresent=items.some(item=>item[yAxis]!=null);
  if(!xPresent||!yPresent){
    svg.append(svgEl('text',{x:160,y:120,'text-anchor':'middle','font-size':'12',fill:'#6c7770'},'Axis omitted — no interpolated plane.'));
    target.append(svg);return;
  }
  items.forEach(item=>{
    if(item[xAxis]==null||item[yAxis]==null){
      const x=item[xAxis]==null?left:left+item[xAxis]*width;
      const y=item[yAxis]==null?top+height:top+height-item[yAxis]*height;
      const g=svgEl('g',{class:'topo-point',tabindex:'0',role:'button'});
      g.append(svgEl('circle',{cx:x,cy:y,r:6,fill:'#fff',stroke:'#25332e','stroke-dasharray':'3 2'}));
      g.append(svgEl('text',{x:x+10,y:y-8,'font-size':'11'},item.action.id+' hole'));
      g.addEventListener('click',()=>inspectTopo(item));
      svg.append(g);
      return;
    }
    const x=left+item[xAxis]*width, y=top+height-item[yAxis]*height;
    const xr=axisInterval(item.action[xAxis]), yr=axisInterval(item.action[yAxis]);
    drawErrorCross(svg,x,y, xr?left+xr[0]*width:x, xr?left+xr[1]*width:x, yr?top+height-yr[1]*height:y, yr?top+height-yr[0]*height:y);
    drawLocus(svg,x,y,item,state.topoPick===item.action.id);
  });
  target.append(svg);
}

function renderTopoDrawings(){
  const topo=state.run?.topography; if(!topo||topo.status!=='RECORDED')return;
  renderTopo3d($('topo-3d'),topo);
  renderTopo2d($('topo-p-care-welfare'),topo,'welfare','care');
  renderTopo2d($('topo-p-care-obligation'),topo,'obligation','care');
  renderTopo2d($('topo-p-welfare-obligation'),topo,'obligation','welfare');
}

function renderTopography(){
  const topo=state.run?.topography;
  const unavailable=!topo||topo.status!=='RECORDED'||!(topo.actions||[]).length;
  $('topo-empty').hidden=!unavailable; $('topo-body').hidden=unavailable;
  if(unavailable){
    $('topo-empty').textContent=topo?.unavailable_reason||'No native Care, welfare or obligation coordinates were recorded. The viewer does not invent an attractor.';
    return;
  }
  $('topo-caption').textContent=(topo.caption||'')+(hasLandscape(topo)?' The 3D composite interpolates a surface through the recorded loci (Shepard, p=2). That surface is display-only, not additional evidence. 2D projections remain recorded points.':'');
  $('topo-axis-legend').textContent=topoLabels(state.topoRot).y+' up · '+topoLabels(state.topoRot).x+' · '+topoLabels(state.topoRot).z;
  clear($('topo-loci'));
  topoNorms(topo).forEach(item=>{const b=node('button',item.action.id+' · '+item.action.label);b.addEventListener('click',()=>inspectTopo(item));$('topo-loci').append(b);});
  clear($('topo-flags'));
  const flags=topo.flags||{};
  (flags.attractors||[]).forEach(id=>$('topo-flags').append(node('span','Attractor occupancy: '+id,'attractor')));
  (flags.degenerate||[]).forEach(name=>$('topo-flags').append(node('span','Degenerate: '+name.replaceAll('_',' '),'degenerate')));
  (flags.conflicts||[]).forEach(row=>$('topo-flags').append(node('span','Conflict '+(row.axes||[]).join('/')+' ','conflict')));
  (flags.holes||[]).forEach(name=>$('topo-flags').append(node('span','Hole: '+name,'hole')));
  if(flags.jackknife_hinging&&flags.jackknife_hinging.action_id!=null){
    const h=flags.jackknife_hinging;
    $('topo-flags').append(node('span','Most hinging tenth: '+h.action_id+' '+h.axis+' fold '+h.fold,'degenerate'));
  }
  if(hasLandscape(topo))$('topo-flags').append(node('span','3D interpolated landscape','attractor'));
  (topo.framework_arrows||[]).forEach(arrow=>{
    const mark=node('span',undefined,'');
    const swatch=node('span',undefined,'topo-swatch');swatch.style.background=FRAMEWORK_COLORS[arrow.framework]||'#6c7770';
    mark.append(swatch,document.createTextNode(arrow.framework+(arrow.action_id?' → '+arrow.action_id:'')));
    $('topo-flags').append(mark);
  });
  if(state.topoPick){
    const item=topoNorms(topo).find(row=>row.action.id===state.topoPick);
    if(item)inspectTopo(item);
    else {state.topoPick=null;emptyTopoInspector();renderTopoDrawings();}
  }else {emptyTopoInspector();renderTopoDrawings();}
}

document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.view)));
document.querySelector('.view-nav').addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();const i=VIEWS.indexOf(state.view);const next=VIEWS[(i+(event.key==='ArrowRight'?1:VIEWS.length-1))%VIEWS.length];setView(next);$('tab-'+next).focus();});
$('topo-rotate').addEventListener('click',()=>{state.topoRot=(state.topoRot+1)%4;renderTopography();});
$('topo-reset').addEventListener('click',()=>{state.topoRot=0;renderTopography();});
$('graph-detail').addEventListener('click',()=>{state.fullLinks=!state.fullLinks;state.zoom=1;renderDiagram();});
$('zoom-in').addEventListener('click',()=>{state.zoom=Math.min(5,state.zoom*1.25);applyZoom();});$('zoom-out').addEventListener('click',()=>{state.zoom=Math.max(.3,state.zoom/1.25);applyZoom();});$('zoom-reset').addEventListener('click',()=>{state.zoom=1;applyZoom();$('graph-viewport').scrollTo(0,0);});window.addEventListener('resize',applyZoom);
new ResizeObserver(()=>{if(state.view==='world')applyZoom();}).observe($('graph-viewport'));
$('download-run').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(state.run,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download=state.run.id+'-parliament-view.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('import-run').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{if(file.size>20*1024*1024)throw new Error('Run JSON must be smaller than 20 MB.');const data=validateRun(JSON.parse(await file.text()));data.export_receipt={status:'UNAVAILABLE',reason:'Imported JSON has no independently loaded export receipt.'};state.cache.set(data.id,data);state.version++;state.index=state.index.filter(e=>e.id!==data.id);state.index.push({id:data.id,title:data.title,status:data.status});state.run=data;state.diagram=0;state.zoom=1;state.topoRot=0;state.topoPick=null;notice('');await renderRun();}catch(error){notice('Could not open run: '+error.message);}finally{event.target.value='';}});
(async()=>{try{if(window.mermaid)window.mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{primaryColor:'#edf2e7',primaryTextColor:'#25332e',primaryBorderColor:'#88a081',lineColor:'#758575',fontFamily:'system-ui, sans-serif',fontSize:'12px'},flowchart:{htmlLabels:false,useMaxWidth:false}});const response=await fetch('data/runs.json');if(!response.ok)throw new Error('Saved-run index could not be loaded.');const index=await response.json();if(index.schema_version!=='parliament-view-index/1')throw new Error('Unsupported run index.');state.index=index.runs;renderRail();if(state.index.length){const params=new URLSearchParams(location.search);const requested=params.get('run');await loadRun(state.index.find(r=>r.id===requested)||state.index[0]);const view=params.get('view');if(VIEWS.includes(view))setView(view);}}catch(error){notice(error.message+' Start the viewer with python serve_parliament_viewer.py.');}})();
