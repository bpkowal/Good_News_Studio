# Separate RelEnt/RAG runtime profile

The `full_relent` profile is installed and passes readiness checks. It shares the
locked native source with `compact_frameworks`, but uses its own Python 3.11 venv
(`.runtime/relent-venv`) and exact 116-distribution dependency lock. The compact
17-distribution environment was not modified and still passes its check.

Full profile scope: OpenAI-backed native RelEnt, five original framework modules,
and native passage RAG. Local GGUF/llama.cpp is outside this profile. No world
schema, admission validator, evidence thresholds, or retrieval algorithm changed.

## Verified

- [Full runtime check](full_check.json): exact dependencies, native imports, source,
  79 corpus files and 11 embedding-cache records verified.
- [Compact runtime check](compact_check.json): ready independently.
- [Idempotent setup](idempotent_setup.json): ready without reinstalling or changing source.
- [Native entrypoint](native_help.txt): full CLI imports and `--help` exit successfully.
- [Dependency consistency](pip_check.txt): no broken requirements.
- [Real offline retrieval](retrieval_smoke.json): local CPU MiniLM embeddings,
  native approved corpus filtering, native framework lenses/identity tags/thresholds,
  and native evidence ranking. No fake embeddings, credentials, or LLM calls.

| Framework | Available passages | Retrieved |
| --- | ---: | ---: |
| Utilitarian | 55 | 3 CORE |
| Deontological | 35 | 3 CORE |
| Virtue | 32 | 3 CORE |
| Care | 32 | 3 CORE |
| Rawlsian | 26 | 3 CORE |

The recorded MiniLM revision is `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`.
Its existing local snapshot was copied into a dedicated hashed cache. Full setup
can fetch that exact revision on a new machine; normal runtime retrieval is offline.
This run verified installation and idempotence, not a second clean full dependency
installation. Resource hash checks do not certify the normative adequacy of retrieved
passages, and this smoke is not an end-to-end live deliberation.

[Regression log](regression.log): **108 tests ran; 104 passed, four skipped**.
The targeted suite covers runtime isolation/resource drift, runner/report behavior,
Cloze, trolley recovery, and admission core. The separately documented preexisting
primitive-composition outcome omission was not addressed by this dependency change.

## Commands

```bash
.venv/bin/python parliament_runtime.py setup --profile full_relent --python python3.11
.venv/bin/python parliament_runtime.py check --profile full_relent
.venv/bin/python parliament_rag_smoke.py --output diagnostics/relent_rag_smoke.json
```

The full blueprint runner now selects this interpreter automatically; `--prepare-only`
and the compact demo retain the compact interpreter. An explicit interpreter override
must still satisfy the chosen profile. The runner passes pinned retrieval settings
after loading its environment so `.env` cannot silently redirect the checked cache.
