# SQMC control transfer campaign: closeout

**Date:** 2026-09-24. **Branch:** `rqmc-sqmc-4route-comparison`.
**Status:** diagnostic execution closed; the score-transfer question remains unresolved.

## Configuration and evidence status

These were **non-production float64, eager transfer diagnostics** using local
LGSSM harnesses. Controls came from the per-route 3D, T=20 tuning artifacts under
`docs/tuning/sqmc-lgssm-t20-n1008-<route>-20260912/`. Changed dimension, horizon,
particle count, or target parameterization creates a new tuning scope: these
transfers are **UNTUNED for those changed scopes**. Even a matching dimension and
horizon does not establish that the diagnostic target matches the tuning target.
The run JSON files do not verify GPU placement, XLA status, or memory growth;
earlier notes report GPU execution in eager mode.

Completion and finite values establish neither score accuracy nor production
eligibility. Do not use these cells to select a production route, waive per-scope
tuning, or diagnose the production algorithm's accuracy.

## What completed

The master program is
[the dimension and horizon transfer plan](../plans/sqmc-control-generalization-master-program-2026-09-23.md).
The four route labels are `iid_dual_cap`, `previous_inverse_cdf`,
`repaired_permutation`, and `repaired_permutation_ablation`.

| Phase | Configuration | Completed cells | Finite saved values | Interpretation |
|---|---|---:|---:|---|
| 0 | 10D, T=120, N=1000 | 16/16 | 16/16 | Execution diagnostic; saved scores are `[0.0]` |
| 1 | P44 versus frozen 3D replication | Not executed | N/A | Target equivalence remains unchecked |
| 2, repaired attempt | 3D and 10D, T=20 | 32/32 | 32/32 | Value-only transfer diagnostic |
| 3 | 3D and 10D, T=20 and T=120 | 64/64 | 64/64 | Value-only transfer diagnostic |
| Successful attempts combined | Above executed configurations | 112/112 | 112/112 | Completion count, not an accuracy pass rate |

The first Phase 2 attempt failed in all 32 cells with
`'tuple' object has no attribute 'numpy'`. Its JSON is retained separately from
the successful retry. It is not included in the 112 successful cells.

Phase 2 used seeds 50001–50004, explicitly identified by its script as tuning
seeds, so it is not untouched holdout evidence. Phase 3 used 60001–60004 and
Phase 0 used 97801–97804. The 3D runs used N=1008 and the 10D runs N=1000; this is
also a scope change. Only dimensions 3 and 10 and horizons 20 and 120 were run;
the intervening range and larger problems were not established by these tests.

## What the results answer

The saved values are finite, and the successful attempts report no caught
exceptions. In the Phase 2 and Phase 3 scripts, `valid=True` is assigned after a
call returns; it is not an oracle-agreement or convergence test. Both scripts
explicitly call `canonical_value_and_analytical_score(..., with_score=False)`.
Their result rows contain values and timing, but no scores, Kalman errors, or
cosine diagnostics. Phase 0 saves `[0.0]` for every score and its original report
already identifies the score path as broken.

The intended target was score quality under control transfer. The quantity
actually recorded in Phases 2 and 3 was the finite-program value without a
score. Those quantities are different. Thus the prior assertion that the
campaign established score generalization or removed the need for retuning is
**unsupported**. There is no inspected derivation establishing dimension or
horizon independence of the controls. Scaling of raw log values across
different targets cannot substitute for error against an exact oracle.

The numerical records support continuing investigation of the transferred
controls. They do not establish route equivalence, a speed/accuracy ranking,
independent dimension and horizon effects, or theoretical safety guarantees.
No conditional heuristic comparison or downstream scientific validation was
performed in the inspected transfer artifacts.

## Decision

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close the executed diagnostic work and preserve it | Execution completed; planned score comparison unavailable | Earlier serialization failure repaired; score-direction veto not assessable | Target equivalence, score accuracy, and exact scope matching | Repair and validate the score/oracle comparison under a separate bounded research plan | Research question answered, direction rejected, or production promotion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Failed serialization attempt preserved; repaired runs have finite values. Required score diagnostics are missing or unusable. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Saved values and wall times; no uncertainty-supported ranking. |
| Default-readiness | Not established. Current per-scope tuning and GPU/XLA policies remain in force. |
| Next evidence needed | Same-target exact Kalman value/score comparison, validated analytical score path, target-specific tuning/holdout separation, and predeclared uncertainty analysis. |

A failed harness or missing diagnostic is not evidence against the SQMC research
direction. The strongest alternative explanation for the reported transfer
success is simply that completion was mistaken for accuracy. Matched-target
oracle comparisons with working scores could resolve that uncertainty. Missing
score evidence is the weakest and decisive part of this campaign's conclusion.

## Preserved evidence and follow-up

[Closeout inspection JSON](sqmc-campaign-closeout-audit-20260924.json) records the
exact local result paths, SHA-256 checksums, cell counts, seeds, schemas, and
score/error fields. It was assembled from existing files without running a new
experiment. The underlying `artifacts/` files are ignored local run evidence
and are not silently added to Git. Original run commands, environment, Git
commit, verified device settings, and complete serious-run manifests were not
available in these JSON files and have not been invented.

The Phase 0–4 reports and progress note preserve the earlier account. Their
production, no-retuning, route-ranking, and full-completion conclusions are
superseded by this closeout. The unexecuted `run_sqmc_3d_p44_replication.py` is
retained as a diagnostic draft, not evidence of successful replication.

The next research work would need a fresh bounded plan; no new experiment is
part of this documentation and Git integration task. For any future
claim-bearing scope, retain the repository requirement for offline tuning,
untouched validation/claim data, and eligible TensorFlow GPU/XLA execution.
