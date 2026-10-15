"""Exercise the actual prepare runner with saved chooser answers; no API calls."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from blueprint_cloze_chooser import choose_by_cloze
from test_blueprint_cloze_chooser import Script
from run_blueprint_parliament import main

case=json.loads((ROOT/'fixtures/trolley_scene_cloze_cases.json').read_text())['cases'][0]
result=choose_by_cloze(case['source'],Script(case['ranking'],case['answers']))
destination=Path(__file__).parent/(sys.argv[1] if len(sys.argv)>1 else 'production_prepare')
with patch('run_blueprint_parliament.choose_by_cloze',return_value=result):
    raise SystemExit(main(['--text',case['source'],'--prepare-only','--output-dir',
        str(destination),'--checkpoint-tag','omission-agent-handoff']))
