# Source-rule attention and rejected-argument review

This increment applies to independent framework generation and Logic_Puzzles targeted review. It leaves the frozen-world schema, native duty schema, native admission validators and pinned Parliament runtime unchanged. It does not modify the separate full RelEnt entrypoint's native prompt builder.

`deontology_rule_attention.py` projects existing normative_propositions and their governed propositions into a read-only rule context. It retains source IDs, force, exact governed content, bearer, authority and clause references; missing references remain missing. The moral authority of a rule is explicitly unassessed. This is prompt context, not a new world assertion or a compliance judgment.

`logic_puzzles_frameworks.generate` supplies this context to deontology by default. Its `rule_attention=False` keyword is the controlled comparison switch. Other frameworks receive no additional rule instruction. The existing l/dr/rs/pr/foq response fields carry the requested applicability, institutional-authority and emergency-conflict analysis. The candidate's informational source_rule_attention assessment remains NOT_INDEPENDENTLY_ASSESSED; mentioning a rule cannot certify meaningful consideration.

A rejected deontology reply retains its exact authored duty/maxim proposal and a hash of the model response. `run_logic_puzzles.project_conflicts` displays missing-ledger arguments as nonoperative PROPOSED_ARGUMENT nodes and routes their warnings without granting world-support edges. `logic_puzzles_review` can review such a proposal in a fresh native assessment with the original reply as context. It does not pretend the prior ledger was admitted. A previously committed ledger still uses the existing recurrent review path. Standard queue/call bounds remain in effect.

This removes a review blind spot; it does not guarantee every rejected argument is scheduled within a bounded run. Native admission and semantic adequacy remain separate. Open duty questions or failure to justify a priority can survive a committed ledger.

## Reproduce the bounded live pilot

Use a new output directory to preserve existing evidence. This command makes up to six adapter calls, with provider retry behavior unchanged:

```bash
.runtime/venv/bin/python run_deontology_rule_attention_pilot.py \
  --trace diagnostics/deontic_choice_composition_20261016/live_final/frozen_world_trace.json \
  --saved-assessment diagnostics/deontic_choice_composition_20261016/deliberation/frameworks/framework_generation.json \
  --output-dir diagnostics/deontology_rule_attention_new_run \
  --model o3 --execute
```

The two initial arms have the same world, core excerpt, model, requested response ceiling and maximum call allowance. The saved-reply review is a separate probe. Interpret results by reading the arguments, not by counting admitted ledgers or changed preferences.

The [October 19 results](diagnostics/deontology_rule_attention_20261019/RESULT.md) show a successful recovery and partial substantive correction, but no evidence of better initial reasoning from the rule-context prompt. All original failed proposals remain saved.
