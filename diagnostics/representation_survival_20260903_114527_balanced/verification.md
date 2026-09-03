# Balanced Broadcast Verification

The saved five-framework trace was replayed through the new transport projection;
no specialist or model was rerun.

- All five framework capsules are present in the delivered `problem_state`.
- All five assigned challenge targets remain visible.
- The projected `problem_state` is complete, parseable JSON.
- Current action leans use canonical IDs rather than repeated long action prose.
- Every framework capsule uses the same schema and a 2,200-character maximum.
- The complete in-memory ProblemState is unchanged by projection.
- Projection fields remain explicitly framework-attributed and informational only.

The historical trace predates the new `action_cases` contribution field, so its
replayed Care and Rawlsian capsules cannot recover rival-action prose that was never
stored in that trace. New runs will carry each framework's strongest rival case in
the balanced `counterclaims` field.

Regression results:

- `test_global_workspace.py`: 372 passed
- `test_trace_health.py`: 8 passed
- `test_virtue_ledger.py`: 5 passed
- `test_parliament.py`: 25 passed

The repository-wide discovery run also exposes unrelated outstanding failures in
world-state and framework-ledger test modules. The global-workspace projection test
set itself is green.
