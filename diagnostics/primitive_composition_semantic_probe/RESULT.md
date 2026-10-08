# Limited semantic survival test

Input was the independent primitive-composition lever graph, not the selected
omission blueprint. OpenAI o3 ran the original deontological and Care agents for
one cycle, with normal audits and semantic ledgers enabled (not prototype mode).

The graph survived: downstream world_model equals the input world_model exactly.
All seven effects remain, status COMMITTED, no world validation issues, no trace
health errors, and zero world-generation calls. The frozen replay validated
scenario and action identity. Evidence is in semantic_survival.json.

The interpretation layers did not produce a governing recommendation:

- Deontology's A0 assessment has three calibration errors: doing-harm lacks
  graph support, intended-as-means lacks a path, and resolved adjudication uses
  unsupported decisive premises. The sentence does not establish intent or
  instrumental use of the worker. A0's working links remain ENABLES; do not
  strengthen them to CAUSES merely to satisfy this assessment.
- Care labels the workers ENTRUSTED and describes a track controller despite
  neither role being stipulated. Its A0 claim about five workers cites E3,
  whose actual outcome is one worker's death. The graph remains intact, but
  this claim-to-effect binding is questionable and needs separate diagnosis.
- Final judgment and selected action are UNRESOLVED. This is a downstream
  reasoning/grounding limitation, not rejection or deletion of the candidate.

No parser, blueprint, world schema or semantic validator was changed for this
test. The initial sandboxed API connection failed; the authorized network-enabled
retry completed. Full log: run.log. Full trace and report: parliament/.
