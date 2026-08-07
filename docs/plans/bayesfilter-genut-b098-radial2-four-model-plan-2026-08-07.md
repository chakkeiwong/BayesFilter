# GenUT b=.98 + Radial-2 Four-Model Comparison Plan

Date: 2026-08-07
Status: `AUTHORIZED_BOUNDED_FOUR_MODEL_COMPARISON`
## Research Intent

Test whether the GenUT pairwise correction with coordinate cap `b=0.98,
p=8` and pairwise-direction radial RMS cap `2.0` is a viable and descriptively
favorable choice across LGSSM, KSC transformed SV, predator-prey, and Austria
SIR. The incremental contribution of the radial cap is identified with common
random numbers rather than inferred from separate campaigns.

## Evidence Contract

| Field | Frozen decision |
|---|---|
| Models | `lgssm_T50`, `ksc_sv_T10`, `predator_prey_T20`, `austria_sir_T20` |
| Particle count | `N=1008` |
| Claim seeds | `98201..98216`, common within every model/arm comparison |
| Calibration | Existing disjoint model-specific calibration trajectories with seeds `98401,98402` for the first three models and `98301,98302` for Austria |
| Arm 1 | Scope-specific diagonal-only baseline |
| Arm 2 | Same baseline plus four pairwise steps; strength `.02` for LGSSM/KSC/Austria and `.05` for predator-prey |
| Arm 3 | Pairwise arm plus coordinate cap `b=.98,p=8`, radial cap off |
| Arm 4 | Pairwise arm plus coordinate cap `b=.98,p=8` and radial RMS cap `2.0` |
| Runtime | TensorFlow FP32, TF32 enabled, GPU/XLA, verified memory growth |
| Hard validity | finite/program-valid GPU rows; mean/row/column and score-additivity residuals `<=5e-4`; normalized displacement `<=2`; capped arms require post-cap maximum `<1.000001` |
| Primary comparison | Candidate-minus-coordinate-only paired value and score summaries, per-arm value/score seed SDs, and cap activity/displacement |
| Reference roles | LGSSM exact Kalman; KSC dense transformed-mixture diagnostic; predator-prey prior same-target SGQF/Zhao-Cui diagnostics; Austria same-target SGQF/UKF diagnostics |
| Derivative check | Same-program centered value FD at `h=1e-3` on the first claim seed for the coordinate-only and dual-cap arms; explanatory because inherited FP32/TF32 routes can fail this step size |
| Artifact | `docs/benchmarks/artifacts/genut_b098_radial2_four_model_20260807/attempt01/` |

No approximate comparator is an exact nonlinear oracle. A closer score does
not prove lower bias. Sixteen seeds support descriptive paired comparisons and
MCSE estimates, not a universal ranking.

## Research-Question Guardian

| Role | Diagnostic |
|---|---|
| Promotion criterion | None; this campaign does not change a default |
| Promotion veto | Calibration or claim hard-validity failure for that arm |
| Continuation veto | Wrong target hash/event order, invalid baseline, GPU/memory-policy failure, corrupted artifact, or budget exhaustion |
| Repair trigger | Harness or serialization failure under unchanged target and controls |
| Explanatory | Reference gaps, FD residuals, cap activity, value/score SDs, paired mean changes |
| Must not be concluded | Exact score, lower score bias, Zhao-Cui faithfulness, posterior correctness, HMC/NeuTra readiness, or universal superiority |

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Early diagnostic |
|---|---|---|---|---|
| `b=.98,p=8` | Austria T2 least-distorting valid cap | transferred hypothesis | cap is not tail-only in another chart | active fraction and displacement |
| radial cap `2` | prior pairwise direction-cap experiments | hypothesis | improves one target by chance or raises variance | direct paired coordinate-only comparison |
| pairwise strengths | prior per-model pairwise campaigns | warm-start controls | stale or over-strong outside prior scope | disjoint calibration validity |
| baseline controls | model-specific July leaderboard tuning | historical scoped baseline | not re-tuned under current code | exact target hash/event-order checks and baseline validity |
| TF32 FP32 | repository target backend | execution choice | small-step FD sensitivity | record FD as explanatory and retain exact-reference comparisons |

## Skeptical Audit

- Wrong baselines are controlled by reading the prior scope-bound GenUT
  controls and checking target hashes/event orders. Austria uses its separate
  T20 controls, not the T2 bounded-teacher settings.
- The radial effect is identified against the identical `b=.98` arm. A
  diagonal and pairwise-only ladder prevents comparison against only a weak
  baseline.
- Approximate-reference proximity remains explanatory. It cannot become a
  score-accuracy promotion criterion.
- Candidate failures do not invalidate the harness or stop other models. An
  invalid baseline or target mismatch does.
- KSC has state dimension one, so the pairwise correction/radial cap is a
  structural no-op; only the coordinate cap can change that route.
- The artifacts contain raw per-seed rows, manifests, controls, target hashes,
  and checkpoints, so successful commands answer the stated question.

Audit decision: `PASS_FOR_BOUNDED_EXECUTION`.

## Budget And Stop Conditions

Budget one four-model calibration/claim campaign plus one localized
infrastructure retry, at most 45 GPU minutes. Use a fresh attempt directory and
do not expand controls after viewing claim results. Stop on a continuation
veto; record arm-specific scientific/numerical failures and continue.
