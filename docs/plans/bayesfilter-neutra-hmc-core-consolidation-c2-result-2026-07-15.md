# NeuTra HMC Core Consolidation Phase C2 Result

Date: 2026-07-15  
Decision: `PASS_C2_ACTIVE_DEFAULT_MIGRATION`

## Outcome

The active LGSSM gap-closure route now delegates batched and sequential HMC to
`bayesfilter.inference.neutra_hmc`. The campaign retains only target loading,
transport application, LGSSM telemetry summarization, archive serialization,
comparison, and scientific result assembly.

The old local controller was removed, not merely shadowed. The route policy now
rejects unledgered/stale/duplicate routes, missing shared bindings, fixed-budget
active paths, and reachable local TFP sampler bypasses.

## Evidence And Nonclaims

- Shared core, campaign compatibility, and policy suite: `27 passed` before
  the final bypass/seed-separation additions; the final focused suite is
  recorded in the S1 entry check.
- Existing archive compatibility assertions passed.
- Historical routes and evidence files were not rewritten.
- This closes engineering consolidation only; robustness remains untested
  beyond the prior two-seed fixture until S1 and F2 run.

## Handoff

Proceed to the S1 subplan for seed `(20260715, 1203)`. A seed-specific failure
rejects that candidate but does not stop the new-fixture lane unless it reveals
a common target or harness defect.
