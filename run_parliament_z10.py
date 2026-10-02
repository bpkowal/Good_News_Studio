"""Run Parliament world grounding with an exported Z10 advisory packet.

Use Parliament's Python 3.11 environment. This runs grounding, not deliberation.
An explicit response fixture or live model choice is required.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from parliament_z10_bridge import render_advisory


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--parliament-root', type=Path, required=True)
    cli.add_argument('--packet', type=Path, required=True)
    cli.add_argument('--actions', nargs='+', required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--max-attempts', type=int, default=2)
    backend = cli.add_mutually_exclusive_group(required=True)
    backend.add_argument('--response-file', type=Path, help='Offline JSON response fixture; no model calls')
    backend.add_argument('--openai-model', help='Explicitly enable live calls using Parliament\'s backend')
    args = cli.parse_args()
    if args.max_attempts < 1:
        cli.error('--max-attempts must be positive')
    packet = json.loads(args.packet.read_text(encoding='utf-8'))
    source = packet.get('package', {}).get('document', {}).get('text')
    if not isinstance(source, str):
        cli.error('packet is missing source text')
    render_advisory(packet, source)  # Reject before creating any provider client.
    sys.path.insert(0, str(args.parliament_root.resolve()))
    from global_workspace.local_specialists import ground_actions_in_scenario
    import inspect
    if 'parser_evidence_packet' not in inspect.signature(ground_actions_in_scenario).parameters:
        cli.error('Parliament needs integrations/parliament_z10/grounding_hook.patch')
    prompts = []
    if args.response_file:
        response = args.response_file.read_text(encoding='utf-8')
        json.loads(response)

        class FixtureBackend:
            def complete_json(self, prompt, **kwargs):
                prompts.append(prompt)
                return {'choices': [{'text': response}]}

        llm = FixtureBackend()
    else:
        from global_workspace.openai_backend import OpenAIWorkspaceLLM
        llm = OpenAIWorkspaceLLM(model=args.openai_model)
    result = ground_actions_in_scenario(
        llm, source, args.actions, max_attempts=args.max_attempts,
        stage_one_guidance_mode='EVIDENCE_ONLY', parser_evidence_packet=packet)
    advisory = result.get('parser_advisory') or {}
    output = {
        'mode': 'offline_fixture' if args.response_file else 'live_grounding',
        'parliament_commit': subprocess.check_output(
            ['git', '-C', str(args.parliament_root), 'rev-parse', 'HEAD'], text=True).strip(),
        'parser_packet': packet, 'actions': args.actions, 'grounding': result,
        'parser_advisory_model_call_attempts': advisory.get('model_call_attempts', 0),
        'note': 'A deterministic or early-return path can bypass model grounding. '
                'Prompt delivery does not demonstrate semantic use. No deliberation is run.',
    }
    if args.response_file:
        output['captured_prompts'] = prompts
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result.get('status'),
                      'advisory_model_call_attempts': output['parser_advisory_model_call_attempts'],
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
