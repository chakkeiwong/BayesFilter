# SGQF Whole High-Dimensional Leaderboard Active Campaign Note

Date: 2026-07-22

Status: `EXECUTION_AUTHORIZED; GENERALIZED_SV_LEVEL3_CANDIDATE_SELECTED`

Governing program:
`docs/plans/bayesfilter-sgqf-whole-highdim-leaderboard-repair-master-program-2026-07-22.md`

## Skeptical Audit Update

The program remains sensible after two corrections:

1. The canonical scalar-SV seed is `81101`, as fixed by the June dataset
   manifest and live runner. The Phase 0 crosswalk's `81102` entry was stale.
2. Generalized-SV level 2 is finite but is not the best cheap default. On the
   full T=1008 fixture, the level-2/level-5 value gap was `4.7830381e-3`, while
   the level-3/level-5 gap was `6.0507077e-5`. Level 3 costs the same order of
   wall time and is the active candidate. Level 2 remains a baseline only.

No wrong baseline, proxy promotion criterion, environment mismatch, or missing
stop condition remains for the next CPU/GPU engineering run. GPU execution is
an engineering gate, not scientific evidence.

## Research Intent Ledger

| Field | Active definition |
| --- | --- |
| Main question | Does the scalar `GeneralizedSVPriorMeanSSM` raw-observation SGQF Gaussian-projection likelihood and its manual same-scalar score execute correctly at T=1008? |
| Candidate | Fixed level-3 standard-normal quadrature, transition before every observation, manual forward sensitivities in `(z_gamma,log_tau,mu_over_tau)`. |
| Expected failure mode | Wrong timing; KSC/native-target substitution; incomplete parameter sensitivity; covariance collapse; XLA/device failure. |
| Promotion criterion | Repository identity passes; T1/T2 transition tests pass; independent primal value equals score scalar; manual score matches central FD; level 3 agrees with level 5 and 41-point dense Gaussian-projection reference within `1e-4`; CPU/GPU XLA parity passes. |
| Promotion veto | Any target substitution, runtime autodiff, nonfinite value/score, variance below `1e-12`, identity mismatch, or CPU/GPU parity failure. |
| Continuation veto | Source target/data cannot be reconstructed, recurrence is mathematically undefined, or bounded attempt budget is exhausted. |
| Repair trigger | A localized implementation, serialization, XLA, or threshold failure with target and math intact. |
| Explanatory only | Runtime, allocator bytes, level-2 gap, Zhao-Cui/UKF values, and cross-method differences. |
| Must not be concluded | Exact nonlinear likelihood, exact posterior, SGQF superiority, statistically supported algorithm ranking, default readiness, or HMC readiness. |

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Scalar prior-mean model | June source contract and checked `svmodels`; reviewed target | Silent use of two-state `NativeGeneralizedSVSSM` | Manifest family and raw-y test |
| Seed `81105`, T=1008 | Existing frozen generalized-SV dataset | Cross-dataset comparison | Serialized tensor hashes |
| Transition before y1 | Checked author `ssmodel.complete` | Initial-observation target drift | T1/T2 transition count |
| Level 3 | Full-horizon 2/3/5 ladder; active hypothesis | Quadrature under-resolution | Level 5 and 41-point dense reference |
| Central FD `h=1e-5` | Diagnostic convenience choice | Cancellation or truncation error | Tiny/asymmetric and full-horizon comparisons; not primary implementation |
| GPU XLA | Engineering requirement | CPU fallback or hidden allocator behavior | Trusted device placement, memory growth, allocator, parity |

## Evidence Contract

The exact comparator for implementation consistency is the independently coded
primal recurrence at the same level. The dense 41-point recurrence checks the
quadrature/refinement choice for the same sequential Gaussian-projection
quantity. Neither is an exact nonlinear filtering oracle.

Hard vetoes are nonfinite results, failed data/route identity, incorrect
transition count, variance at or below `1e-12`, score/FD failure, level/dense
gap above `1e-4`, wrong device placement, failed memory-growth configuration,
or CPU/GPU parity above `1e-9` relative for value and `1e-8` relative for score.

Artifacts are written without overwrite under
`docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/attempt03/generalized-sv/`.

## Pre-Mortem

- A successful command could still compute KSC or a two-state target. The
  route manifest and raw-y likelihood tests reject that substitution.
- Value/score equality could be tautological. The primal recurrence is separate
  and central FD differentiates it independently.
- Levels could agree because of shared algebra. The T1 direct formula and
  checked source equations provide separate mechanics anchors; broad exactness
  remains a nonclaim.
- GPU could silently fall back to CPU. Soft placement is disabled and result
  devices are hard-gated.
