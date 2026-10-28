# Thermal blanket: live entry-to-browser inspection

Open [the considered world and arguments](http://127.0.0.1:7860/?run=thermal_blanket_promise). The viewer is read-only saved output of fresh live model calls. [The first attempt](http://127.0.0.1:7860/?run=thermal_blanket_first_attempt) remains alongside it.

## New scenario

A mountain shelter has one thermal blanket. Ellis can give it to Rowan or Sasha, but not both. Rowan has an 80% survival chance with it; Sasha has a 60% chance. The person without it dies. Before learning about Rowan, Ellis promised Sasha the blanket. No timely alternative rescue or warming equipment exists. Whom should Ellis choose?

The [exact entered text](../../demo_scenarios/thermal_blanket_promise.txt) stayed unchanged throughout. This is a new author-written example of allocation plus promise, not a new blueprint family or an independent benchmark.

## What ran

1. Entry, Z10, primitive inventory, cloze/macros, candidate construction and native admission ran live.
2. The default small-model chooser left an interpretive beam. It admitted an allocation graph and thinner/inferior alternatives, including a promise-only graph. No framework ran in that attempt.
3. Retrying the same source with o3 initially exposed an adapter temperature error. After correcting the transport, o3 selected the allocation graph but two terminal-ellipsis citations failed exact quotation checks.
4. A narrow binding correction now accepts a terminal ellipsis only when the preceding text is an exact substring of the referenced clause. Internal omissions, invented words, wrong clauses and unaddressed differences remain invalid. The original raw model response is unchanged.
5. [resume.py](resume.py) replayed that exact saved response against the identical source/comparison. Existing proposals passed native admission again; the saved review selected its original candidate. No new chooser calls or manual world choice were used in the continuation.
6. All five independent framework agents ran live on that frozen world, one recorded adapter call each. Two frameworks received the existing bounded targeted-review round. [finish.py](finish.py) assembled a separate report input without editing the frozen trace, initial generation or review.
7. The report was exported into the existing viewer catalog. [verify.py](verify.py) checked the served source and world against every stage, checked all five argument links over HTTP, and confirmed no world mutation. [browser.png](browser.png) records a headless Chrome render; the normal browser was also opened.

This is an end-to-end continuation with preserved failed attempts, **not a clean first-pass success**. It exercises compact framework generation and targeted native review. Full sequential RelEnt/RAG deliberation and the experimental premise-audit argument constructor were not run for this scenario. A subsequent [Decisions topography update](topography_update/RESULT.md) adds the existing interpolated surface and ten-fold sensitivity intervals.

## Recorded results

Final status: **FRAMEWORK_PARTIAL**, with no governing moral verdict.

| Framework | Recorded preference | Operative/report status after review |
| --- | --- | --- |
| Utilitarian | No full ranking | ABSTAIN; ledger committed |
| Deontological | Give to Sasha | ATTENUATED; committed with uncertainty |
| Care | Give to Rowan | ABSTAIN; no committed Care ledger |
| Virtue | Give to Rowan | FULL; ledger committed |
| Rawlsian | Give to Rowan | ABSTAIN; committed with uncertainty |

All four targeted issues remain OPEN. The reviewer declaring a reply reconciled did not retire them. The initial report used incomplete native topography. The current viewer uses a subsequent Decisions probability pass with ten-fold sensitivity intervals; native coordinates are retained in that update's before snapshot. The initial Markdown report remains a historical account of the native-coordinate stage; the report JSON and viewer contain the new topography.

The [full report](report/DEMO_REPORT.md), [targeted review](review/targeted_review.md), [frozen topology](pipeline/world_state_topology.md) and [verification](verification.json) retain the relevant records.

## Content defects worth inspecting

- The frozen graph treats “Rescuer Ellis” and “Ellis” as separate parties; the promise is attached to the latter. The agents still received the original scenario, but graph identity remains imperfect.
- Two effects carry the DIE relation while their outcome label says “does not get the blanket.” The source says death; that discrepancy is preserved for inspection.
- The deontological review prompt includes an irrelevant staff-exposure sentence inherited from earlier lab work. This is context contamination, not a source fact; its influence has not been isolated here.
- Duty priority remains contested. Structural validity and source-bound graphs have not established better moral conclusions.

## Reproduce a fresh run

Regression verification: **82 passed, four skipped** across chooser, selection review,
demo orchestration and viewer tests. The [regression log](regression_final.log)
includes intentional failure-fixture messages; the unittest result is successful.

Use a new output directory:

```bash
.venv/bin/python run_parliament_demo.py \
  --scenario demo_scenarios/thermal_blanket_promise.txt \
  --output diagnostics/thermal_blanket_fresh_run \
  --chooser-model o3 --review --review-depth standard
```

That makes new API calls; stochastic results may differ. To reopen the saved browser result with no model calls:

```bash
.venv/bin/python serve_parliament_viewer.py --refresh
```

Then open `http://127.0.0.1:7860/?run=thermal_blanket_promise`. Export supports additional catalog entries while preserving the original three demos. Public provenance retains IDs/hashes and relative paths instead of machine-specific absolute paths.
