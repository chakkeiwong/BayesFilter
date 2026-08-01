# SIR Remaining-Gap Closure Phase 2 Result

Date: 2026-07-16

Plan: `docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md`

Artifact:
`docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase2_j1_n64_128_256_r16_attempt01/`

Status: `NO_TEACHER_J1_DISAGREEMENT_DETECTED_AT_CURRENT_PRECISION`

## Result

The independent `J=1` online `O(N^2)` teacher was screened against the dense
split-Gauss--Legendre reference at observation counts `T=1,2`. The artifact
binds the convention `dense time_steps=T-1`, uses `R=16` independent seeds and
the planned `N=64,128,256` ladder, and computes Bonferroni-adjusted 95% Student
intervals for teacher minus dense reference.

At the primary `N=256` rung, every expanded interval contained zero. Therefore
the declared teacher-disagreement continuation veto did not fire. This is not a
teacher certification or equivalence result: the value and observation-noise
score intervals remain wide enough that material finite-particle bias could be
undetected.

## Primary Rung

| Horizon | Quantity | Mean teacher minus reference | Simultaneous half-width | Expanded interval contains zero |
| ---: | --- | ---: | ---: | --- |
| `T=1` | value | `-2.4071e-3` | `2.1090e-2` | yes |
| `T=1` | score log observation scale | `+7.6647e-3` | `3.1114e-2` | yes |
| `T=2` | value | `-2.6669e-3` | `4.1078e-2` | yes |
| `T=2` | score log kappa scale | `+2.7606e-6` | `2.7484e-5` | yes |
| `T=2` | score log nu scale | `+1.6552e-4` | `2.4216e-4` | yes |
| `T=2` | score log observation scale | `+6.1911e-3` | `5.2666e-2` | yes |

The `T=1` kappa and nu scores are structurally zero because no transition has
occurred. They are not evidence about transition-score accuracy.

## Validity Checks

- All `6 x 16` replicate outputs were finite.
- Minimum ESS was positive at every rung; at `N=256` it was about `218.9` for
  `T=1` and `195.3` for `T=2`.
- Maximum backward-kernel row-sum error was `0` for `T=1` and
  `8.88e-16` for `T=2`.
- The frozen dense values, scores, and refinement diagnostics reproduced.
- Artifact SHA-256 entries reproduce for `result.json` and
  `run_manifest.json`.
- The run was deliberately CPU-only with `CUDA_VISIBLE_DEVICES=-1`, float64,
  and XLA JIT enabled.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue to Phase 3 | No `J=1` teacher disagreement detected at current precision | No nonfinite, normalization, target, or interval-exclusion veto | Intervals may be too wide to detect material bias; only `J=1` is externally checked | Run exact-current-source GPU/XLA certificate before claim-bearing comparisons | Teacher unbiasedness/convergence, practical equivalence, LEDH accuracy, HMC readiness, leaderboard readiness |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | passed |
| Statistically supported ranking | none; no ranking was attempted |
| Descriptive-only differences | all particle-count trends, means, ESS, runtime, and interval widths |
| Default readiness | not established |
| Next evidence needed | exact-source GPU/XLA and route identity, followed by explicitly limited LEDH--teacher disagreement screens |

## Post-Run Red Team

The strongest alternative explanation is low power: interval containment can
occur even when a biased teacher is noisy. A reproduced largest-rung interval
excluding the dense reference, or a newly found local score-term defect, would
overturn the continuation decision. The weakest evidence is the lack of an
external oracle beyond `J=1` and the absence of a justified practical
equivalence margin.

