# Live API pipeline results

Executed the current working tree, including uncommitted blueprint chooser/runner changes. Live generation used gpt-4o-mini; full medicine deliberation used o3, five frameworks, up to three cycles, original agents and native audits, without RAG retrieval. No native validation was bypassed and no implementation files were changed for this test.

| Scenario | Construction/admission | Deliberation | Final judgment |
| --- | --- | --- | --- |
| Medicine allocation | Baseline exclusive_allocation admitted; interpretation separated | Completed all three cycles | UNRESOLVED; plurality favors Ben |
| Trolley omission | Six attempts admitted; three retained beam members | NOT_RUN: unresolved world interpretation | No normative judgment produced |

Medicine admitted eight effects and six links. Amended allocation was rejected for indistinguishable action grounding; primitive composition was rejected for conditional links lacking conditions. The baseline candidate reached Parliament. Completion is not settled normative acceptance: utilitarian, deontological and Rawlsian final votes abstained, while virtue and care support was attenuated. The original Rawlsian testimony baseline also recorded a parse failure.

A notable recognition gap remains: the source explicitly states one dose and death of the non-recipient, yet paraphrases of these premises appear as hypothetical dependencies in the final answer. The world also retains the 95% and 5% survival qualifications, but the utilitarian ledger reports insufficient evidence for ranking. These are observations from this run, not a diagnosis proving which component is responsible. The entrusted-care relationship is an additional agent-introduced premise and should be assessed separately.

Trolley's omission baseline was admitted in this live run, unlike the deterministic fixture that failed on AV3/AV6. The live generator produced a different candidate; this does not fix or invalidate that regression failure. The runner retained omission_harm, conditional_outcome and an amended ability_permission interpretation without selecting a unique frozen world for deliberation.

Inspect all successful, rejected and withheld [medicine candidate graphs](medicine/candidate_graphs.md), the [medicine world topology](medicine/world_state_topology.md), the [full Parliament report](medicine/parliament/deliberation_report.md), and [trolley candidate graphs](trolley/candidate_graphs.md). The [summary](summary.json) preserves statuses, attempt errors and final artifact paths.

Commands used:

```bash
./run_blueprint_parliament.sh --output-dir diagnostics/live_pipeline_latest_20261006/medicine
./run_blueprint_parliament.sh --scenario-file diagnostics/live_pipeline_latest_20261006/trolley.txt --output-dir diagnostics/live_pipeline_latest_20261006/trolley
```
