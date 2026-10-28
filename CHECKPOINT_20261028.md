# Parliament checkpoint before the deliberation-learning game

October 28, 2026. Application branch: `Parliament_Frontend`.

This checkpoint preserves the non-simulation Parliament application as it stood
before beginning `Learning_Deliberation_Game`: parser/blueprint refinements,
argument support and inference review, repair cards, shared neutral framework
instructions, Contractualist pilots and question-capable adapter, the five-view
demo, source/argument provenance, the viewer and interpolated 3D presentation.
Saved live requests, responses, rejected/partial arguments, reviews and diagrams
are included with their original limitations. No model call was needed to save it.

Rawlsian historical runs retain their identities. The newer Contractualist demo
uses four native specialists and a separate attributed argument adapter over
the same frozen world; it does not fabricate a Contractualist native ledger or
a collective verdict. Topography probabilities and interpolation are display
assessments, not calibrated moral truth or authorizations to act.

The numeric contextual-weight simulations, their source/tests and diagnostic
databases are retained in the working tree for a separate commit on the new
learning branch. They are not presented as features of this Parliament baseline.
The only unstaged edit to the shared provenance module adds simulation execution
labels; its previously committed non-simulation behavior remains the baseline.

## Native runtime backup

The live native checkout remains on `relent-framework` at pinned base
`614af0c1601dd7169842f130f1c26a5dd0b30564`, with its applied changes intact.
The exact modified native source is separately published at
[backup/parliament-native-20261028](https://github.com/bpkowal/Good_News_Studio/tree/backup/parliament-native-20261028),
commit `3eb9d2ea331eb936e522d5a7132a14bd98ef2217`.

All 1,398 native source inventory entries match the runtime lock's patched tree
fingerprint `45e8451584de716b8c4e814f8b4bafbfdc3795ff2780cff805b66b2b68bfcc94`.
An isolated checkout was used; the live native HEAD and runtime lock were not
advanced to the backup commit. The application retains the ordered patch recipe,
including `argument_support_scope.patch`.

## Verification

- 210 application tests ran in the Python 3.9 report/parser environment: OK,
  four skipped.
- 40 native reasoning tests passed in the pinned Python 3.11 environment.
- Source-only runtime verification reports ready, with the exact locked tree.
- Reviewed checkpoint paths and source ZIP text had no credential-pattern alerts.
- The application remote matched its existing local HEAD before the checkpoint;
  backups do not rewrite main or force-push existing history.

An initial combined test invocation used Python 3.9 for native imports requiring
Python 3.11; six checks failed at that environment boundary. The separated runs
above use the appropriate profiles. No unrelated native or application behavior
was changed to hide those results.

See [backup inventory](diagnostics/backup_20261028/staging_plan.json),
[native audit](diagnostics/backup_20261028/native_audit.json) and
[application test log](diagnostics/backup_20261028/application_tests.log).
These checks establish code/test and source-byte preservation, not semantic
completeness or philosophical correctness of generated arguments.
