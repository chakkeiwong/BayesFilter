# BayesFilter V1 Model B/C BC0 Baseline Matrix Result

## Date

2026-05-15

## Governing Plan

BC0 executes:

```text
docs/plans/bayesfilter-v1-model-bc-bc0-baseline-reconciliation-plan-2026-05-14.md
```

under the master program:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Phase Intent

BC0 is an evidence-indexing phase.  It does not add new numerical evidence.
It reconciles the current Model B/C value, score, branch, reference, HMC,
GPU/XLA, and Hessian status before the wider BC1-BC8 testing campaign.

## Plan Tightening

No phase-scope ambiguity required a plan rewrite.  The only operational
tightening is interpretive:

- `dense_one_step_projection_only` is a diagnostic comparator for Models B-C,
  not an exact full nonlinear likelihood reference.
- the tiny Model B SVD-CUT4 HMC smoke is diagnostic only, not convergence.
- the existing GPU/XLA row is one fixed Model B SVD-CUT4 shape, not broad
  speedup evidence.
- default Model C score evidence is valid only on the structural fixed-support
  branch with `allow_fixed_null_support=True`.

## Independent Audit

As a second-developer audit, BC0 is safe to execute because it only reads
existing artifacts and writes V1-lane documentation.  It stays out of
MacroFinance, DSGE, Chapter 18b, structural plans, and shared reset memos.
The audit found no drift from the master goal: every row below keeps value and
score claims separate from reference, HMC, GPU/XLA, and Hessian claims.

## Evidence Sources

- `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.json`
- `docs/benchmarks/bayesfilter-v1-model-b-nonlinear-hmc-smoke-2026-05-14.json`
- `docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.json`
- `docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-result-2026-05-14.md`
- `docs/plans/bayesfilter-v1-p2-branch-diagnostics-result-2026-05-14.md`
- `docs/plans/bayesfilter-v1-p3-benchmark-refresh-result-2026-05-14.md`
- `docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-result-2026-05-14.md`
- `docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-result-2026-05-14.md`
- `docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-result-2026-05-14.md`
- `docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-result-2026-05-14.md`

## Baseline Matrix

| Model | Filter | Value status | Score status | Branch | Reference status | HMC status | GPU/XLA status | Hessian status | Current claim label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model B nonlinear accumulation | SVD cubature | `finite_implemented_filter`; value branch `3/3`; no active floors, weak gaps, or nonfinite rows | `finite_analytic_score_branch`; score branch `3/3` | `smooth_simple_spectrum_no_active_floor`; `allow_fixed_null_support=False`; structural null count `0`; fixed-null residual `0.0` | `dense_one_step_projection_only`; first-step projection error `1.680660838937209e-02`; not exact full nonlinear likelihood | `deferred`; no HMC smoke yet for this filter | `deferred`; no GPU/XLA row yet for this filter | `deferred_no_consumer`; result Hessian is `None` | `certified` for first-rung value/score only |
| Model B nonlinear accumulation | SVD-UKF | `finite_implemented_filter`; value branch `3/3`; no active floors, weak gaps, or nonfinite rows | `finite_analytic_score_branch`; score branch `3/3` | `smooth_simple_spectrum_no_active_floor`; `allow_fixed_null_support=False`; structural null count `0`; fixed-null residual `0.0` | `dense_one_step_projection_only`; first-step projection error `1.680660838937209e-02`; not exact full nonlinear likelihood | `deferred`; no HMC smoke yet for this filter | `deferred`; no GPU/XLA row yet for this filter | `deferred_no_consumer`; result Hessian is `None` | `certified` for first-rung value/score only |
| Model B nonlinear accumulation | SVD-CUT4 | `finite_implemented_filter`; value branch `3/3`; no active floors, weak gaps, or nonfinite rows | `finite_analytic_score_branch`; score branch `3/3` | `smooth_simple_spectrum_no_active_floor`; `allow_fixed_null_support=False`; structural null count `0`; fixed-null residual `0.0` | `dense_one_step_projection_only`; first-step projection error `3.6430863839773675e-03`; not exact full nonlinear likelihood | `diagnostic`; tiny CPU smoke only, 8 finite samples, acceptance `1.0`, no convergence claim | `diagnostic`; one fixed Model B SVD-CUT4 shape only | `deferred_no_consumer`; result Hessian is `None` | `certified` for first-rung value/score only |
| Model C autonomous nonlinear growth | SVD cubature | `finite_implemented_filter`; value branch `3/3`; no active floors, weak gaps, or nonfinite rows | `finite_analytic_score_branch`; score branch `3/3` | `structural_fixed_support_no_active_floor`; `allow_fixed_null_support=True`; structural null count `1`; structural-null covariance residual `4.930380657631324e-32`; fixed-null residual `1.178621711759372e-31` | `dense_one_step_projection_only`; first-step projection error `4.9264906908805806e-02`; not exact full nonlinear likelihood | `deferred`; no Model C HMC until structural score gates pass in BC1-BC3 | `deferred`; no Model C GPU/XLA row yet | `deferred_no_consumer`; result Hessian is `None` | `certified` for first-rung structural fixed-support value/score only |
| Model C autonomous nonlinear growth | SVD-UKF | `finite_implemented_filter`; value branch `3/3`; no active floors, weak gaps, or nonfinite rows | `finite_analytic_score_branch`; score branch `3/3` | `structural_fixed_support_no_active_floor`; `allow_fixed_null_support=True`; structural null count `1`; structural-null covariance residual `1.0585463634721146e-31`; fixed-null residual `1.3138756722640583e-16` | `dense_one_step_projection_only`; first-step projection error `1.4899570336240098e-01`; not exact full nonlinear likelihood | `deferred`; no Model C HMC until structural score gates pass in BC1-BC3 | `deferred`; no Model C GPU/XLA row yet | `deferred_no_consumer`; result Hessian is `None` | `certified` for first-rung structural fixed-support value/score only |
| Model C autonomous nonlinear growth | SVD-CUT4 | `finite_implemented_filter`; value branch `3/3`; no active floors, weak gaps, or nonfinite rows | `finite_analytic_score_branch`; score branch `3/3` | `structural_fixed_support_no_active_floor`; `allow_fixed_null_support=True`; structural null count `1`; structural-null covariance residual `0.0`; fixed-null residual `0.0` | `dense_one_step_projection_only`; first-step projection error `3.3911269316548864e-02`; not exact full nonlinear likelihood | `deferred`; no Model C HMC until structural score gates pass in BC1-BC3 | `deferred`; no Model C GPU/XLA row yet | `deferred_no_consumer`; result Hessian is `None` | `certified` for first-rung structural fixed-support value/score only |

## Gate Result

BC0 primary gate passes.  No Model B/C/filter cell has unknown status:

- value and score are first-rung certified for all six rows;
- default Model C score rows explicitly cite structural fixed support and
  `allow_fixed_null_support=True`;
- HMC, GPU/XLA, reference, and Hessian gaps are labeled as `diagnostic` or
  `deferred` rather than left unknown.

## Veto Diagnostics

| Veto | Status |
| --- | --- |
| Default Model C score claim omits structural fixed-support branch | Clear |
| Dense one-step projection described as exact full nonlinear likelihood | Clear |
| Tiny HMC smoke described as convergence evidence | Clear |
| Tiny GPU/XLA shape described as broad speedup evidence | Clear |
| Nonlinear Hessians promoted without a named consumer | Clear |

## Interpretation

The current evidence supports first-rung value and analytic-score checks for
Models B-C across SVD cubature, SVD-UKF, and SVD-CUT4.  It does not yet support
wider parameter-box stability, stronger score residual stress claims,
horizon/noise robustness, exact nonlinear likelihood, HMC convergence, broad
GPU/XLA speedup, or nonlinear Hessian production support.

## Continuation Decision

BC1 is justified.  The exact accepted starting branches are:

- Model B: smooth simple spectrum with no active floor.
- Model C: structural fixed support with no active floor and
  `allow_fixed_null_support=True`.

BC1 must keep row-level failure labels and may not run score stress tests
across unresolved branch transitions.
