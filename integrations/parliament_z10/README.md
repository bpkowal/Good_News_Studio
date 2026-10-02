# Z10 advisory bridge: first Parliament integration

Z10 remains frozen. The bridge gives Parliament its complete candidate package
alongside the original scenario, plus readable reconstruction role frames.
It does not replace the input with a guessed corrected sentence.

For `Maria can save the child, but not the dog.`, the additional reading identifies
the omitted `save` predicate, subject `Maria`, and object `the dog`. Its polarity
remains unresolved, and the original question about `NOT MODAL(P)` versus
`MODAL(NOT P)` stays in the packet. Parliament can consider inability to save the
dog versus ability to refrain, without this adapter choosing one. Named-role
alternatives remain separate propositions. Unimplemented verb-phrase ellipsis,
gapping, and sluicing reconstructions remain missing-content questions.

## Files and boundary

- `parliament_z10_bridge.py` in the project root validates schema 0.4 packages,
  matches exact source text, preserves every package field, and renders advisory
  instructions and reconstruction notes. No candidates are selected by default.
- `grounding_hook.patch` is the full diff of the live Parliament checkout against
  `614af0c1601dd7169842f130f1c26a5dd0b30564`. Apply it once to that commit.
  It adds the optional `parser_evidence_packet` argument to
  Parliament's `ground_actions_in_scenario`. It requires `EVIDENCE_ONLY`, validates
  before backend calls, and includes the packet in primary and repair grounding
  prompts. Existing world admission checks still run. Auxiliary calls do not
  receive this packet. The result records advisory model-call attempts;
  deterministic/early-return paths can bypass the model and have no such record.
  The patch also begins a constrained draft compiler for exclusive allocations:
  when a nonreceipt row has exactly one source-grounded same-action transfer,
  Parliament deterministically supplies its quantity, parent edge, and derivation
  type. Ambiguous transfer parents remain untouched. A transfer to the affected
  person may directly parent a stipulated welfare outcome; mentioning the
  transferred resource alone no longer licenses an invented process-state node.
- `run_parliament_z10.py` runs that grounding API using either a response fixture
  or an explicitly requested live OpenAI backend. This first integration stops at
  world grounding: it does not wire the interactive launcher, its framing cache,
  or downstream ethical deliberation to Z10.

The separate packet version is `z10-advisory/1`; this is not Parliament's native
Stage-1 packet schema. IDs retain their Z10 package scope. Parser evidence IDs
must not be used as Parliament clause citations. Source hash checks detect
mismatches; they are not authenticity or semantic-support checks. Template markers
inside data strings are escaped without changing JSON values.

## Use

The hook was applied and tested in `/tmp/parliament-smoke-614af0c`, based on
`bpkowal/Good_News_Studio`, branch `relent-framework`, commit
`614af0c1601dd7169842f130f1c26a5dd0b30564`. The patch is saved here for persistence.
On another checkout at that commit, apply it with `git apply` using its absolute
path. Do not apply it again to the already patched temporary checkout.

From this parser project's root, prepare evidence using its frozen environment:

```sh
.venv/bin/python parliament_z10_bridge.py \
  --text-file integrations/parliament_z10/example.txt \
  --output diagnostics/z10_parliament_example_packet.json \
  --notes-output diagnostics/z10_parliament_example_notes.txt
```

Run the offline grounding probe in Parliament's isolated Python 3.11 environment:

```sh
/tmp/parliament-smoke-env/bin/python run_parliament_z10.py \
  --parliament-root /tmp/parliament-smoke-614af0c \
  --packet diagnostics/z10_parliament_example_packet.json \
  --actions 'save the child' 'save the dog' \
  --response-file integrations/parliament_z10/rejected_response.json \
  --output diagnostics/z10_parliament_grounding_smoke.json
```

Expected outcome: `REJECTED`, with two advisory model-call attempts. The fixture
deliberately returns `{}` so the test proves prompt delivery and continued
rejection of invalid output; it does not measure the model's understanding.
The runner exits successfully when the probe completes; inspect `grounding.status`
for admission. It saves original evidence, actions, results, and fixture prompts.
For a live experiment, replace `--response-file ...` with `--openai-model MODEL`
and use an environment with Parliament's provider dependencies and credentials.
No live calls were made during this implementation.

## Validation and remaining work

Nine bridge tests passed, including the actual patched Parliament function with
a fake backend: initial and repair delivery, baseline isolation, invalid-input
rejection before model calls, role alternatives, conditions, modal scope, lossless
transport, source immutability, and JSON/template escaping.
The saved medicine first draft and a cleaner explicit either/or variant both now
commit without another model call. Their nonrecipient rows inherit `one`, point to
the selected recipient transfer, and are typed
`EXCLUSIVE_ALLOCATION_COMPLEMENT`. A deliberately ambiguous two-parent draft is
left unchanged. Existing process-intermediate tests continue to reject shortcuts
when a separate process bearer is actually required.
The held-out antivenom variant changes the actor, recipients, transfer verb,
resource, container word, and outcome. Its first replay exposed a lexical
`dose`/`vial` dependency. The compiler now uses typed exclusive-allocation evidence
and source-bound singular resource structure; it commits the antivenom replay while
preserving `one vial`.
Two later live `o3` runs are recorded in
`diagnostics/PARLIAMENT_Z10_LIVE_ALLOCATION_TESTS.md`. Antivenom committed after a
repair pass; an epinephrine-injector variant remained rejected after its repair
added allocation branches without their complete factual provenance. These runs
show that topology construction generalizes, while evidence and quantity closure
still need to become deterministic compiler responsibilities.
Z10's 13 focused tests passed; dataset integrity and all frozen baseline hashes
passed. Parliament's same 71-test regression sample remained at 64 passes,
six existing compiler errors, and one existing presentation assertion failure;
the failing test identities are unchanged. See
`diagnostics/parliament_medicine_controller_regression.log` and the prior smoke
report.

```sh
PARLIAMENT_SMOKE_ROOT=/tmp/parliament-smoke-614af0c \
PARLIAMENT_SMOKE_PYTHON=/tmp/parliament-smoke-env/bin/python \
.venv/bin/python -m unittest test_parliament_z10_bridge -v
```

The advisory bridge establishes transport. The constrained compiler separately
provides the first deterministic semantic construction step. Next, compare raw-source
grounding with advisory grounding on reviewed development gold, keeping model,
actions, and generation settings fixed. Measure ellipsis roles, operator scope,
unsupported additions, abstention, and admission separately. Preserve the original
scenario in both arms. Full packages increase prompt size; no silent truncation or
lossy compacting has been added. The held-out set has not been used for inference.

A fluent rewritten version is a later, explicitly labelled interpretation view:
generate it only from a validated selected bundle, retain blockers and unresolved
readings, and map every inserted phrase to its reconstruction evidence. A rewrite
must never become replacement source evidence for Parliament's admission checks.
