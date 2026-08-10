# Zhao-Cui Bounded GenUT Austria T2 Joint-Control Repair Plan

Date: 2026-08-06

Status: `EXECUTED_STOPPED_CALIBRATION_VETO`

Terminal result:
`docs/plans/bayesfilter-zhao-cui-bounded-genut-austria-t2-crossed-validation-result-2026-08-06.md`

## Trigger and question

The crossed-validation plan exhausted its two attempts at calibration. All 18
rows were invalid because the corrected bounded coordinate left `(-1,1)`;
maximum `|u|` was `1.026--1.186`. The normalized physical affine restoration
errors were only `6.2e-8--4.6e-6`, so restoration was not the failure.

The material plan flaw is that diagonal strength `0.2` was inherited and held
fixed while only pairwise strength/cap were tuned. That did not execute the
earlier recommendation to tune diagonal and pairwise controls jointly. This
repair asks whether a joint target-specific grid yields a numerically valid T2
candidate whose teacher-seed variation is no more than half its particle-seed
Monte Carlo variation.

## Evidence contract and research ledger

| Field | Frozen decision |
|---|---|
| Baseline | Exact no-shape finite program under common validation particle seeds |
| Calibration grid | diagonal strength `{0,.02,.05,.1}`; pairwise strength `{.01,.02,.04}`; cap `{disabled,2,4}`; four steps for each correction |
| Selection | Among candidates passing boundary, affine, finite-program and FD gates, minimize mean `(diagonal objective + pairwise objective)` over calibration seeds; ties by lower diagonal strength, pairwise strength, then cap |
| Validation | The three already-built but untouched validation teachers crossed with particle seeds `98801..98806` |
| Promotion criterion | All validation/FD gates pass and between-teacher SD divided by pooled within-teacher particle SD is `<=0.5` for value and each score coordinate |
| Hard veto | Non-finite/invalid/non-GPU row; corrected `|u|>=1`; normalized affine mean/covariance residual `>2e-4`; FD absolute `>0.08` or normalized `>0.03` |
| Continuation veto | No joint-grid candidate passes; validation artifact/teacher identity failure; campaign attempt budget exhausted |
| Explanatory only | Candidate-baseline displacements, residual values, cap activity, sign counts and runtime |
| Nonclaims | No exact physical moments, true-score accuracy, improvement, ranking, T20, HMC/NeuTra, posterior, or default readiness |

The candidate score remains the total JVP of its own finite scalar and FD tests
only that identity. There is no exact T2 Austria score authority.

## Default audit

| Choice | Status and reason | Failure mode | Diagnostic |
|---|---|---|---|
| Reuse calibration teacher/seeds | repair tuning data; no claim data are reused | local overfit | untouched crossed validation |
| Reuse validation teachers | constructed but never loaded/evaluated by stopped runs | none from result peeking | manifest hashes stay frozen |
| Joint grid bounds | repair hypotheses centered below failed diagonal `0.2` and around pairwise `0.02` | viable point outside grid | no-candidate continuation veto |
| Combined residual | declared bounded-moment objective | not score accuracy | explicit nonclaim and FD consistency only |
| `0.5` sensitivity ratio | user-approved adequacy threshold | only three teacher seeds | no ranking/default claim |
| FP32/no-TF32/GPU/XLA | unchanged derivative-parity scope | near-boundary sensitivity | boundary and same-program FD vetoes |

## Skeptical audit

Verdict: `PASS_AFTER_REPAIR`.

- The wrong fixed diagonal baseline is removed from tuning; no failed candidate
  is treated as evidence against the full teacher direction.
- Calibration residual is not promoted to score accuracy. Extension beyond T2
  still requires the untouched teacher-sensitivity and numerical screens.
- Common particles and identical non-candidate controls preserve fairness.
- Validation teachers and particle seeds remain untouched by the failed runs.
- Stop conditions and nonclaims are unchanged and explicit.
- The existing full crossed artifact answers the question if calibration
  succeeds; a calibration ledger is written before any stop.

## Budget and execution

One GPU/XLA launch plus one localized infrastructure retry, at most 25 aggregate
GPU minutes. Reuse the four strict 128-sample teacher artifacts. Evaluate all
36 candidates on calibration seeds `98701,98702`, FD-check candidates under the
frozen gates, freeze the combined-residual selection, then run the 3-by-6
validation and teacher/particle variance decomposition. Preserve a fresh output
directory and write a terminal result note. T20 remains out of scope until
teachers T3 through T20 exist and are independently validated.
