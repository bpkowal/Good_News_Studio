"""Optional direct analysis through the same native OpenAI backend."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parliament-root', type=Path, required=True)
    p.add_argument('--model', default='o3')
    a = p.parse_args()
    sys.path.insert(0, str(a.parliament_root))
    from dotenv import load_dotenv
    from global_workspace.openai_backend import OpenAIWorkspaceLLM
    load_dotenv(Path('/Users/benjaminkowal/Documents/Python/Python Coding/RAGAIMODEL/.env'))
    prompt = ('Give a strong direct moral analysis of this scenario. Explain your recommendation, '
        'relevant uncertainty, competing considerations, and what could change your judgment. '
        'Return your analysis in the analysis field.\n\n'+a.scenario.read_text().strip())
    schema = {'type':'object', 'properties':{'analysis':{'type':'string'}},
              'required':['analysis'], 'additionalProperties':False}
    result = OpenAIWorkspaceLLM(a.model, timeout=120).complete_json(prompt, schema=schema, max_tokens=3072, temperature=0)
    (a.output/'baseline_response.json').write_text(json.dumps(result, indent=2)+'\n')
    text = json.loads(result['choices'][0]['text'])['analysis']
    (a.output/'BASELINE.md').write_text(text+'\n')


if __name__ == '__main__':
    main()
