# Parliament / frozen Z10 smoke test — 2026-09-30

The components run offline, but they are not integrated. Parliament's selected
regression suites are not green. No parser or Parliament implementation was changed.

Pinned Parliament: `bpkowal/Good_News_Studio`, branch `relent-framework`, commit
`614af0c1601dd7169842f130f1c26a5dd0b30564`. Isolated checkout:
`/tmp/parliament-smoke-614af0c` (clean after testing).

| Check | Result |
| --- | --- |
| Z10/Z9/ellipsis coverage focused tests | 13 passed |
| Development exports | 16 processed; exact evidence spans and package immutability preserved |
| Empty selections and individual candidate dependency closures | 171/171 contract-valid |
| Native Stage-1 evidence positive control | Passed: observation, unresolved finding, advisory authority, deep-copy isolation |
| Raw Z10 packages passed directly to native evidence builder | All 16 yielded zero observations and zero hypotheses |
| Parliament launcher and pipeline `--help` | Both passed in isolated Python 3.11 environment |
| Parliament launcher tests | 56 passed, 1 assertion failure |
| Hospital malware compiler regression tests | 8 passed, 6 errors |
| Dataset integrity and frozen baseline hashes | Passed |

## Failures and implications

The launcher assertion expects `FOREGONE E0_4 -> A1 E1_3`; current presentation
prints `COUNTERFACTUAL E0_4 -> A1 E1_3`. This is an expectation/output mismatch,
not evidence of a parser failure.

Six compiler tests raise `WorldModelValidationError` on their shared hospital
fixture before completing their intended checks. Diagnostics cite missing
`8 years` quantity fields, a missing intervening process node, conditions that
restate causal parents, and condition/causal-parent conflicts. This suggests
fixture/validator drift; it does not establish which side should change.
Review the fixture against the current contract before changing expectations
or relaxing validators. Full tracebacks: `parliament_offline_smoke.log`.

The native evidence builder reads Parliament's `neutral_skeleton`, `skeleton`,
and related Stage-1 structures. It does not consume Z10's candidate package
schema. The negative probe is a deliberate wrong-schema call: silent empty
output demonstrates that direct interchange is unsuitable. The positive control
uses the native schema. These builder probes load the standalone module through
`runpy`; they do not exercise an integrated grounding pipeline.

The next integration step is an explicit, version-checked adapter and injection
point that retain evidence, alternatives, dependencies, operator scope, and
unresolved questions as advisory input. Unsupported mappings must remain visible.
Do not treat an adapted candidate as an admitted world fact. Test rejected schemas
and preservation of scope before comparing raw-text and parser-assisted grounding.

## Environment and reproduction

Z10 uses the existing Python 3.9.6 environment, unchanged. Parliament's dataclass
`slots` usage prevents startup there. A clean temporary Python 3.11 environment
was created with pydantic 2.13.5, sympy 1.14.0, numpy 2.4.6, PyYAML 6.0.3, and
python-dotenv 1.2.3. This is a minimal smoke environment, not a fully installed
or locked Parliament production environment.

From the parser project:

```sh
.venv/bin/python -m unittest test_parsing_game_Z10 test_parsing_game_Z9 test_ellipsis_coverage -q
.venv/bin/python smoke_parser_parliament.py --parliament-root /tmp/parliament-smoke-614af0c
.venv/bin/python evals/dilemma_interpretation_v1/dataset.py
```

From the pinned Parliament checkout:

```sh
/tmp/parliament-smoke-env/bin/python parliament.py --help
/tmp/parliament-smoke-env/bin/python global_workspace_pipeline.py --help
/tmp/parliament-smoke-env/bin/python -m unittest tests.test_parliament tests.test_world_compiler_regression -v
```

Machine-readable summary: `parliament_Z10_smoke.json`; repeatable boundary output:
`parliament_Z10_boundary_smoke.json`. The boundary script records the checkout's
actual commit. Selection probes cover individual dependency closures, not every
possible combination. Contract validity does not measure semantic accuracy.
No live model calls or held-out inference were performed. Dataset integrity checks
read both splits structurally; semantic gold remains awaiting review.
