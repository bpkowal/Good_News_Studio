# Optional follow-ups to revised arguments

The compact demo can offer deeper review after its ordinary targeted review.
The offer concerns a decision-relevant question in the **revised** argument,
including the case where a deontological agent accepts foreseen side-effect harm
but applies a prohibition against intentional killing without explaining the
application.

The question distinguishes intending an intervention, intending harm as an end,
intending harm as a means, and foreseeing harm. The physical world does not
establish an actor's actual intentions. The follow-up can ask the agent to defend
a different prohibition, revise its conclusion, or acknowledge uncertainty.
It does not require the agent to recommend pulling the lever.

## User choice

The existing demo command remains usable:

```bash
.venv/bin/python run_parliament_demo.py --scenario scenario.txt --output diagnostics/new_demo --review
```

`--review-depth ask` is the default. When a supported question remains, an
interactive terminal displays it and asks whether to run up to two additional
review calls. An empty answer or anything other than yes declines. Noninteractive
runs retain the offer in the report and do not start additional calls.

Use `--review-depth standard` for the ordinary review without a prompt. Use
`--review-depth deep` to opt in in advance; that flag also enables ordinary review.
`--followup-rounds 1` reduces the additional-round limit to one.

A saved review can be inspected without calling the provider:

```bash
.venv/bin/python logic_puzzles_followup.py --trace saved_run/review/targeted_review.json --output-dir diagnostics/review_offer
```

To opt in for a saved review, use the pinned native interpreter and add `--execute`:

```bash
.runtime/venv/bin/python logic_puzzles_followup.py --trace saved_run/review/targeted_review.json --output-dir diagnostics/deeper_review --execute
```

The saved-run browser viewer remains read-only. The report shows the offer;
it does not introduce a browser endpoint that can spend API credits.

## Bounds, state, and evidence

- At most two additional rounds per invocation, with one adapter call per round.
  Existing provider transport/retry behavior is unchanged.
- Each round uses the latest admitted revision and its preceding challenge reply.
  It can return to the same framework. It does not restart from the original opening.
- Accepted response snapshots and their hashes are retained. The native replay
  must reproduce the relevant committed duty state before a follow-up call.
- A rejected revision preserves the previous argument and stops this invocation.
  An unchanged structured justification stops further rounds; new wording alone
  is not independently assessed as progress.
- Earlier traces are immutable. New output folders preserve all questions,
  attempted replies, native decisions, remaining questions, and stopping reasons.
- The admitted world, other frameworks' arguments, and the two-action limit remain
  unchanged. A follow-up question has investigative authority, not world-state or
  governing moral authority.

`reasoning_followup.json` and `.md` contain the history. Each individual native
review has its own trace. The aggregate retains resumable input so a user can
explicitly opt in again in a new output folder; the system never loops indefinitely.

## Scope and interpretation

This first increment detects the deontological duty-application gap above and
retained deontological premise errors. It is not a detector of every significant
ethical issue, nor a semantic oracle. A question may remain after a strong answer
because its philosophical adequacy is not independently assessed. A model's
`RESOLVED` label cannot certify it, and an open advisory question does not itself
invalidate an admitted recommendation.

The feature is available in the compact demo and saved framework-review path.
Full RelEnt deliberation is unchanged.

## Verification

```bash
.venv/bin/python -m unittest test_logic_puzzles_followup test_logic_puzzles_review test_logic_puzzles_partial_review test_framework_presentation test_parliament_demo test_parliament_run_provenance test_parliament_runtime -v
```

The focused tests use controlled backends through real native replay/review.
They verify sequential revisions, explicit opt-in, budgets, unchanged-answer and
rejection stops, state verification, report offers, and source-world preservation.
They do not demonstrate improved live-model philosophical reasoning.
