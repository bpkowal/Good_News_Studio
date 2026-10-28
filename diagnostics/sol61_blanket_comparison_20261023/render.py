"""Render frozen comparison artifacts; never edits any frozen request or response."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from run_sol_parliament_comparison import check_freeze,render_review

for stage in ('protocol','blind','informed','comparison'):check_freeze(HERE,stage)
blind=json.loads((HERE/'blind.response.json').read_text())
informed=json.loads((HERE/'informed.response.json').read_text())
comparison=json.loads((HERE/'comparison.response.json').read_text())
parliament=json.loads((HERE/'parliament_packet.json').read_text())


def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for item in value.values():yield from strings(item)
    elif isinstance(value,list):
        for item in value:yield from strings(item)


sources={'BLIND':blind,'INFORMED':informed,'PARLIAMENT':parliament}
bindings=[]
for finding in comparison['findings']:
    for quote in finding['evidence']:
        matched=bool(quote['quote']) and any(quote['quote'] in text for text in strings(sources[quote['source']]))
        bindings.append({'dimension':finding['dimension'],**quote,'exact_substring_match':matched})
(HERE/'comparison_quote_checks.json').write_text(json.dumps(bindings,indent=2)+'\n')
lines=['# Frozen Sol / Parliament comparison','',
    'Both Sol reviews were frozen before the comparison call. The second used a fresh request and did not receive the first answer.',
    '', '## Recommendation comparison','',comparison['recommendation_comparison'],
    '', '## Model comparative assessment','',comparison['overall_assessment']]
for finding in comparison['findings']:
    lines+=['','### '+finding['dimension'],'',finding['assessment']]
    for evidence in finding['evidence']:
        matched=any(b['dimension']==finding['dimension'] and b['source']==evidence['source']
                    and b['quote']==evidence['quote'] and b['exact_substring_match'] for b in bindings)
        lines+=['',evidence['source']+(' — exact source substring:' if matched else ' — citation NOT EXACTLY VERIFIED:'),
                '', '> '+evidence['quote'].replace('\n','\n> ')]
for key in ('what_parliament_added','regressions_or_contamination','unresolved_issues','limits'):
    lines+=['','## '+key.replace('_',' ').capitalize(),'']+['- '+s for s in comparison[key]]
lines+=['','## Audit trail','',
    '- [Blind review](blind.md) and [its freeze receipt](blind.freeze.json).',
    '- [Parliament-informed review](informed.md) and [its freeze receipt](informed.freeze.json).',
    '- [Frozen model comparison](comparison.response.json) and [its freeze receipt](comparison.freeze.json).',
    '- [Exact Parliament packet](parliament_packet.json) and [preregistered protocol/rubric](protocol.json).',
    '- [Quote checks](comparison_quote_checks.json). Matching a quote does not establish a normative judgment.',
    '', 'Single scenario, one response per condition, different input lengths. This cannot establish a dependable treatment effect or a budget-matched superiority of Sol or Parliament.']
(HERE/'MODEL_COMPARISON.md').write_text('\n'.join(lines)+'\n')

esc=html.escape
def paragraphs(text):return ''.join('<p>'+esc(p)+'</p>' for p in text.split('\n\n'))
def review_html(reply):
    result='<p class="choice">'+esc(reply['recommendation']['choice'])+'</p>'+paragraphs(reply['recommendation']['conclusion'])
    result+=paragraphs(reply['reasoning'])
    for row in reply['perspectives']:
        result+='<h3>'+esc(row['perspective'])+'</h3>'+paragraphs('For Rowan: '+row['case_for_rowan'])+paragraphs('For Sasha: '+row['case_for_sasha'])+paragraphs(row['assessment'])
    result+='<h3>Strongest objection and reply</h3>'+paragraphs(reply['strongest_objection'])+paragraphs(reply['answer_to_objection'])
    for key in ('assumptions','uncertainties','reversal_conditions'):
        result+='<h3>'+esc(key.replace('_',' ').capitalize())+'</h3><ul>'+''.join('<li>'+esc(t)+'</li>' for t in reply[key])+'</ul>'
    if reply['assessment_of_supplied_analysis']:
        result+='<h3>Assessment of supplied analysis</h3>'+paragraphs(reply['assessment_of_supplied_analysis'])
    return result

findings=''.join('<h3>'+esc(f['dimension'])+'</h3>'+paragraphs(f['assessment']) for f in comparison['findings'])
table='<table><tr><th>Parliament framework</th><th>Recorded preference</th><th>Participation</th></tr>'+''.join('<tr><td>'+esc(f['framework'])+'</td><td>'+esc(f.get('preference') or 'No full ranking')+'</td><td>'+esc(f['position'])+'</td></tr>' for f in parliament['frameworks'])+'</table>'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frozen Sol / Parliament comparison</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#24372f;background:#f6f5f0;margin:0}main{max-width:1450px;margin:auto;padding:40px 28px}h1{font:42px Georgia,serif}h2{font:29px Georgia,serif}h3{font-size:18px;margin-top:28px}p{margin:12px 0}.note{background:#e9eee7;padding:18px 24px;border-radius:8px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}.card{background:white;border:1px solid #ccd5c9;border-radius:9px;padding:25px;margin:22px 0}.choice{font-weight:700;font-size:20px}a{color:#315d55}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #dde2d8;padding:10px}.badge{font-size:13px;color:#56715e}li{margin:7px 0}details{margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:850px){.columns{grid-template-columns:1fr}}</style><main>
<p class="badge">FROZEN EXPERIMENT · GPT-6.1 SOL · NO WORLD OR LEDGER CHANGES</p>
<h1>Does Parliament improve a Sol review?</h1>
<p><a href="/?run=thermal_blanket_promise&view=frameworks">Inspect Parliament’s original framework records</a> · <a href="MODEL_COMPARISON.md">Full comparative audit</a> · <a href="RESULT.md">Assistant assessment and limits</a> · <a href="verification.json">Isolation verification</a> · <a href="protocol.json">Protocol</a></p>
<div class="note">The first review received the scenario only. The second received the same scenario plus Parliament’s exact frozen viewer packet, without the first review. Both responses were frozen before a third Sol call compared them. This is a single-case, non-independent model comparison.</div>
'''+ '<section class="card"><h2>The exact scenario</h2>'+paragraphs((HERE/'scenario.txt').read_text())+'</section>'+\
    '<section class="card"><h2>Comparative assessment</h2>'+paragraphs(comparison['overall_assessment'])+paragraphs(comparison['recommendation_comparison'])+findings+'</section>'+\
    '<section class="card"><h2>Parliament’s recorded positions</h2>'+table+'<p>Recorded preference and operative participation remain separate. Parliament’s status is '+esc(parliament['status'])+'.</p></section>'+\
    '<div class="columns"><section class="card"><h2>1 · Blind review</h2><p class="badge">Scenario only · frozen before Parliament-informed call</p>'+review_html(blind)+'<p><a href="blind.request.json">Exact request</a> · <a href="blind.freeze.json">Freeze receipt</a></p></section>'+\
    '<section class="card"><h2>2 · Parliament-informed review</h2><p class="badge">Fresh context · no blind response supplied</p>'+review_html(informed)+'<p><a href="informed.request.json">Exact request</a> · <a href="informed.freeze.json">Freeze receipt</a></p></section></div>'+\
    '<section class="card"><h2>Limits</h2><ul>'+''.join('<li>'+esc(t)+'</li>' for t in comparison['limits'])+'</ul><p>Exact quotations and freeze hashes permit inspection; they do not turn a model’s moral judgment into an independent oracle.</p></section></main></html>'
(HERE/'index.html').write_text(page)
public=ROOT/'frontend/dist/data/sol61_blanket_comparison'
public.mkdir(parents=True,exist_ok=True)
for path in HERE.iterdir():
    if path.suffix in ('.json','.md','.txt','.html'):shutil.copyfile(path,public/path.name)
print('Rendered frozen comparison at http://127.0.0.1:7860/data/sol61_blanket_comparison/index.html')
print('Exact comparison quotations:',sum(b['exact_substring_match'] for b in bindings),'/',len(bindings))
