# Pinned Parliament runtime

The current live-demo path uses a persistent native checkout and Python environment
under `.runtime/`. This directory is ignored by Git. Historical diagnostics that
refer to `/tmp/parliament-smoke-614af0c` are not rewritten.

Check readiness without credentials or model calls:

```bash
.venv/bin/python parliament_runtime.py check
```

Reconstruct on a new machine with Python 3.11, Git, and network access:

```bash
.venv/bin/python parliament_runtime.py setup --python python3.11
```

For source reconstruction/checking without dependency installation:

```bash
.venv/bin/python parliament_runtime.py setup --source-only
.venv/bin/python parliament_runtime.py check --source-only
```

`source_only` readiness is not executable readiness. The normal check verifies
the exact base commit and origin, all tracked source inputs plus patch-added
Python modules, required files, patch/requirements hashes, Python major/minor,
installed dependency versions, and native imports. Its JSON includes the lock
hash and source-tree hash. It never loads `.env`, constructs a model client, or
sends an API request.

## What is locked

`pipeline.lock.json` pins the repository, exact commit, ordered patches, final
patched-tree hash, Python version, package versions, requirements hash, required
native files, and currently used world/report/viewer contract labels. The six
patches are applied in their listed order with `git apply --check` before every
application. No branch name or commit date substitutes for the exact SHA.

The old `discourse_workspace.patch` has malformed hunk counts/insufficient trailing
context. The lock uses `runtime_patches/discourse_workspace.patch`: a canonical
diff generated using unique exact original contexts against the pinned base plus
grounding patch. Its semantic additions are unchanged. The historical patch was
preserved. All six patches have been applied to a second fresh base checkout and
reproduce the locked source-tree hash.

Two dependency profiles share the same pinned source checkout:

| Profile | Environment | Purpose |
| --- | --- | --- |
| `compact_frameworks` (default) | `.runtime/venv` | Admission and independent compact frameworks |
| `full_relent` | `.runtime/relent-venv` | OpenAI RelEnt, all five original agents, native MiniLM RAG |

```bash
.venv/bin/python parliament_runtime.py setup --profile full_relent --python python3.11
.venv/bin/python parliament_runtime.py check --profile full_relent
.venv/bin/python parliament_rag_smoke.py --output diagnostics/relent_rag_smoke.json
```

Full setup installs its own exact dependency lock and restores the embedding model
at the exact recorded Hugging Face revision. The dedicated `.runtime/rag/hub`
cache includes a pinned `refs/main`; runtime retrieval is offline. Model/tokenizer
file hashes and corpus hashes are checked. Corpora come from the pinned native
checkout and retain native approval/retraction filtering and evidence thresholds.
The smoke command uses real CPU embeddings and native retrieval for all five
frameworks, without credentials, an LLM, or API calls.

This profile supports the OpenAI backend, not local GGUF/llama.cpp. Chroma's legacy
compatibility loader is installed, although the five original agents use passage
retrieval. It does not install CUDA helpers or replace native retrieval behavior.
Package versions are pinned; this is not a cross-platform wheel/build-artifact lock.
The verified platform is Python 3.11 on macOS ARM64.

## Startup and failure behavior

`run_parliament_demo.py` and `run_blueprint_parliament.py --prepare-only` use these
defaults and run preflight before loading API credentials or invoking the chooser.
Full `run_blueprint_parliament.py` selects the separate RelEnt interpreter automatically;
`--prepare-only` selects compact. Explicit `--parliament-python` overrides remain
subject to the selected profile’s checks. They write `runtime_check.json` in the run directory. A missing/drifted runtime is
a software setup problem, not an epistemic refusal to construct a world. The
blueprint runner retains the source and an explicit `RUNTIME_UNAVAILABLE` manifest;
the demo writes its failure report. Independent framework CLI startup checks the
same runtime before constructing its backend. Saved-report replay bypasses runtime
checks and remains offline.

Setup is idempotent for an already matching runtime. It never deletes or resets
an existing source checkout to hide local changes, nor does it apply patches twice.
After an interrupted clone/patch operation, preserve or move the incomplete folder
aside and rerun setup into the default empty location. Existing local changes need
review rather than automatic repair.

## Updating the pin

Do not regenerate hashes at startup to accept whatever happens to be installed.
Use a separate checkout to review the base/patch/dependency change, reconstruct the
result, run runtime and native regressions, then update the lock and requirement
pins together in a reviewed commit. Until the lock changes, altered patch files,
source, added Python modules, or dependency versions fail readiness checks.

Application run manifests, source snapshots, parent/child lineage, report revisions,
and viewer provenance are described in [Run versioning](RUN_VERSIONING.md).
This change records runtime identity, not yet every prompt, resource, or local
parser-code change contributing to an experiment.
