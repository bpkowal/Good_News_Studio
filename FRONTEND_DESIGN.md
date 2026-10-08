# Parliament: saved-run inspection interface

## Purpose and first increment

A researcher should be able to identify the original dilemma, distinguish the
world considered by agents from competing proposals, follow evidence behind a
graph record, and see the dependencies behind framework positions. Success is
inspectability, not an appearance of consensus or full admission.

This first increment is a local read-only viewer for medicine, promise and trolley
saved runs. It renders Mermaid, offers source/record inspection, compares
frameworks, and exposes recorded hinges. No provider calls, graph edits, world
selection, moral adjudication, or native ledger updates are available here.

## Layout and interactions

- A run rail switches among the three cases with status visible before selection.
- The header names the run, replay provenance, stop status and native judgment.
- The World tab places the complete source beside a graph and record inspector.
  Candidate tabs remain labeled candidate/not sent to agents. Zoom and reset work
  per graph. Clicking a node or causal relation opens its original record,
  provenance, qualifiers and exact source excerpts. A record list is a keyboard
  alternative to diagram inspection. Matching excerpts highlight in the source.
- The Frameworks tab separates operative participation, recorded preference and
  ledger admission. Reasons, assumptions, dependencies and live objections are
  visible. No committed ledger or native abstention cannot appear as FULL.
- The Hinges tab displays recorded reversal conditions and their original record
  paths, current unknowns, disagreement and nondecisions. Model-reported hinges
  are attributed and unverified, not new source propositions.
- Run JSON can be downloaded or an exported presentation record opened locally.
  Import accepts only `parliament-view/1`. Source/world objects are not edited.

## Presentation contract

`export_parliament_viewer.py` adapts `parliament-demo/1` to `parliament-view/1`.
The presentation record contains:

| Field | Meaning |
| --- | --- |
| `id`, `title`, `schema_version` | UI identity; unsupported versions are rejected |
| `source`, `source_sha256` | Exact source text and export hash |
| `status`, `stop`, `judgment_status`, `mode` | Recorded pipeline outcome and replay context |
| `diagrams` | Each original world, hash, Mermaid source, authority, and whether framework assessment was recorded |
| `frameworks` | Native participation plus attributed reasons/preferences/dependencies |
| `hinges`, `unknowns`, `disagreements`, `not_decided` | Recorded analytical limits, never synthesized extra arguments |
| `provenance` | Saved-report hash and zero new model calls |

The adapter does not choose a beam member or promote a framework claim into a
world effect. It omits local diagnostic paths, execution commands and credentials
from the static bundle. World/source hashes track export identity; they are not
semantic correctness certificates. Promise occurrence can be in the source ledger
without being a world event: source stipulations are displayed separately.

## Implementation

Buildless HTML/CSS/JavaScript with a pinned local Mermaid 10.9.3 renderer. The
first increment deliberately avoids a frontend toolchain: this checkout has no
Node/npm available. A Python standard-library server serves only `frontend/dist`
on loopback. The packaged viewer works without a runtime CDN or API key. No
credentials are loaded by the server. UI content uses DOM text nodes; Mermaid
runs with strict security. Imported JSON is data, not executable HTML.

Typography, generous graph space and muted colors fit research inspection. Amber
marks incomplete participation, red/purple distinguishes retained interpretations,
and source support remains textual rather than a misleading green truth badge.
The three views collapse to one column on smaller screens. Controls have labels,
keyboard focus and graph-record alternatives; graph interaction is optional.

## Next increment

Add a job-based Python API around the existing runner only after reviewing this
viewer. Model credentials stay server-side. Jobs need progress, cancellation,
immutable artifact snapshots and separate software/epistemic outcomes. New-run
controls must not imply they exist in this read-only version. React/TypeScript
can be introduced if state complexity grows; the presentation contract should
remain independent of the implementation. No deployment is performed in this
local research increment.

## Acceptance checks

Medicine/promise display their frozen worlds; trolley displays three candidate
worlds and no agents. Source excerpts and raw records agree with saved payloads.
Framework abstention is not disguised as a selected recommendation. Switching
runs, diagrams, views, zoom and inspector must not mutate data. Unknown versions
and malformed imports show readable errors. Export tests verify source hashes,
original world identity and absence of local paths/commands. Browser QA checks
rendering and interactions separately from these data checks.

### Source-ledger display layer

Frozen diagrams may include purple, dashed-border `SOURCE LEDGER` additions for
saved stipulated records whose `introduced_by` is `SOURCE_TEXT`. These additions
are separate presentation nodes, outside `world`; they do not create commitments
or duties in the frozen schema. Their dashed connections are explicitly labeled
`ledger support reference` and use only existing support IDs resolving to world
nodes. The inspector preserves original proposition IDs, support IDs, epistemic
status, and origin. Exact source matches are highlighted without fabricating a
provenance record. Candidate worlds receive no such overlay, since their ledger
alignment has not been established. Ordinary world-derived ledger claims are not
duplicated in this layer.
