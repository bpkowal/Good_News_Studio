# Contractualist agent in the five-perspective viewer

Open [the framework comparison](http://127.0.0.1:7860/?run=thermal_blanket_contractualist&view=frameworks),
[the considered world](http://127.0.0.1:7860/?run=thermal_blanket_contractualist&view=world),
or [decision hinges](http://127.0.0.1:7860/?run=thermal_blanket_contractualist&view=hinges).
The Contractualist card is shown first, followed by the other four perspectives.

The [3D Topography view](http://127.0.0.1:7860/?run=thermal_blanket_contractualist&view=topography)
now includes the interpolated surface and sensitivity boxes. A separate
twenty-request probability pass completed its coordinates without changing the
world or five arguments. See [the figure update](topography_update/RESULT.md).

This run made five fresh recorded adapter calls: four native independent passes
and one Contractualist generation. All five received the same exact blanket
scenario and unchanged, previously accepted world. World construction was not
rerun. Full sequential RelEnt/RAG deliberation, peer debate and collective
adjudication were not run.

| Perspective | Recorded preference | Participation |
| --- | --- | --- |
| Utilitarian | No comparative ranking | ABSTAIN; native ledger committed |
| Deontological | Sasha | ABSTAIN; native ledger committed with uncertainty |
| Care | Rowan | ABSTAIN; no operative Care ledger |
| Virtue | Rowan | ATTENUATED; native ledger committed |
| Contractualist | Rowan, provisionally | ATTENUATED argument; no native ledger or vote |

The Contractualist output asks whether Sasha relied on the promise. It also admits
residual disagreement about promise versus survival advantage, but **does not
clearly formulate that central normative priority as a critical question**. Its
provisional Rowan conclusion still rests on an insufficiently defended priority.
This is a limitation visible in the new run, not a successful replication of the
earlier question-only reply. The output was not replaced or repaired to make it
appear more unresolved or more persuasive.

The original Rawlsian run remains separately available as **Thermal blanket
promise**. No earlier Rawls output, score or native ledger was relabeled. The
Contractualist questions are available in Frameworks and Hinges; no coordinate is
invented for them in the existing three-axis topography.

Inspect the [full report](report/DEMO_REPORT.md),
[original Contractualist argument](contractualist/response.json),
[source and world supplied to it](contractualist/source.json),
[content review](CONTENT_REVIEW.md), [browser screenshot](browser_overview.png)
and [verification](verification.json).

99 relevant adapter/report/viewer/presentation/topography tests passed before
the run; a subsequent focused subset of 38 passed. The native four candidates
were structurally VALID, which does not establish sound reasoning. The new
Contractualist graph remains partial with 25 binding and 11 comparison findings.

To repeat with fresh outputs:

```bash
.venv/bin/python run_contractualist_demo.py \
  --trace diagnostics/thermal_blanket_end_to_end_20261023/report_input/frozen_world_trace.json \
  --output diagnostics/contractualist_frontend_next
```

The viewer is saved output and browsing makes no API calls. See
[runner design](../../CONTRACTUALIST_DEMO.md) for export/catalog instructions.
