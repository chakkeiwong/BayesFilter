# Phase IE3 plan: linear-Gaussian recovery tests

## Date

2026-05-16

## Purpose

Execute controlled PF and EDH recovery diagnostics against an analytic
linear-Gaussian reference.  This is the first executable check of the monograph
claim that classical filtering and affine particle-flow constructions recover
known special cases under stated assumptions.

## Allowed Write Set

- `experiments/dpf_monograph_evidence/fixtures/`;
- `experiments/dpf_monograph_evidence/diagnostics/`;
- `experiments/dpf_monograph_evidence/runners/`;
- `experiments/dpf_monograph_evidence/reports/`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie3-linear-gaussian-recovery-result-{YYYY-MM-DD}.md`;
- reviewer-grade reset memo continuity section.

## Prerequisites

- IE0 accepted;
- IE2 harness ready;
- IE1 source status recorded or explicitly deferred.

## Tasks

1. Implement a tiny linear-Gaussian fixture with analytic Kalman mean,
   covariance, and log-likelihood.
   The phase-owned fixture artifact must expose:
   - transition coefficient `a`;
   - observation coefficient `h`;
   - process variance `q`;
   - observation variance `r`;
   - prior mean and variance;
   - one fixed observation;
   - analytic predictive mean/variance;
   - analytic posterior mean/variance;
   - analytic one-step log likelihood.
2. Define the baseline ladder:
   - analytic Kalman reference as the trusted comparator;
   - optional naive importance or prior-sampling baseline, if used, as an
     explanatory comparator;
   - bootstrap/SIR PF as stochastic engineering evidence unless replication and
     uncertainty support a stronger statement;
   - EDH affine recovery as deterministic special-case algebraic evidence.
3. Implement bootstrap/SIR PF checks as replicated descriptive engineering
   evidence with:
   - fixed seed list `[101, 102, 103, 104, 105]`;
   - particle-count ladder `[64, 256]`;
   - reported mean error, variance error, log-likelihood error, MCSE or
     interval fields, and descriptive-only interpretation.
4. Implement EDH affine recovery checks as deterministic special-case evidence:
   - integrate or directly evaluate the one-step affine recovery against the
     analytic posterior mean and variance;
   - report deterministic absolute tolerances for posterior mean and variance;
   - keep EDH exactness restricted to the linear-Gaussian special case.
5. Record convergence slopes or deterministic tolerances separately from Monte
   Carlo variability.
6. Produce JSON and Markdown results using the IE2 schema, including source-
   support class and uncertainty status.
7. Set `CUDA_VISIBLE_DEVICES=-1` before scientific imports in the runner and
   record `cpu_only=true`, `gpu_hidden_before_import=true`,
   `pre_import_cuda_visible_devices=-1`, and
   `pre_import_gpu_hiding_assertion=true` in the run manifest.
8. Populate canonical cap keys:
   - `max_particles=256`;
   - `max_time_steps=1`;
   - `max_replications=5`;
   - `max_wall_clock_seconds=30`;
   - `max_sinkhorn_iterations=null` with a non-applicability reason;
   - `max_finite_difference_evaluations=null` with a non-applicability reason.

## Required Phase Artifacts

- fixture module:
  `experiments/dpf_monograph_evidence/fixtures/linear_gaussian.py`;
- diagnostic runner:
  `experiments/dpf_monograph_evidence/runners/run_linear_gaussian_recovery.py`;
- Markdown report:
  `experiments/dpf_monograph_evidence/reports/linear-gaussian-recovery-result.md`;
- JSON result:
  `experiments/dpf_monograph_evidence/reports/outputs/linear_gaussian_recovery.json`;
- phase result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie3-linear-gaussian-recovery-result-{YYYY-MM-DD}.md`.

The JSON result must contain one schema-valid row with diagnostic id
`linear_gaussian_recovery` and phase id `IE3`.  It may summarize PF and EDH
subdiagnostics inside `finite_checks`, `shape_checks`, `tolerance`,
`uncertainty_status`, and `explanatory_only_diagnostics`; it must not emit
diagnostic IDs owned by other phases.

## Primary Criterion

PF and EDH diagnostics either recover the analytic reference within stated
tolerances or produce a structured blocker identifying the failed layer.

## Veto Diagnostics

- likelihood, mean, and covariance references are not analytic;
- Monte Carlo noise is treated as deterministic failure or success;
- EDH exactness is exported beyond the linear-Gaussian case;
- seed policy or tolerance is missing.
- PF summaries omit replication count, MCSE/interval logic, or explicit
  descriptive-only status.
- runner omits CPU-only pre-import GPU hiding proof;
- JSON result cannot be validated by the IE2 harness;
- EDH deterministic recovery is described outside the linear-Gaussian special
  case.

## Outcome Classification

- Promotion/pass criterion: EDH deterministic special-case recovery and PF
  stochastic summaries meet their predeclared comparator/tolerance contracts.
- Promotion veto: analytic Kalman reference is missing, source-support class is
  missing, or PF uncertainty is not reported.
- Continuation veto: the IE2 schema cannot represent PF uncertainty or EDH
  deterministic tolerance.
- Repair trigger: finite/shape failures, unstable seed behavior, or unexpected
  convergence slope.
- Explanatory-only diagnostics: naive baseline behavior and smoke-only PF
  records without replication.

## Expected Artifacts

- `experiments/dpf_monograph_evidence/reports/linear-gaussian-recovery-result.md`;
- `experiments/dpf_monograph_evidence/reports/outputs/linear_gaussian_recovery.json`;
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie3-linear-gaussian-recovery-result-{YYYY-MM-DD}.md`.

## Exit Labels

- `ie3_linear_gaussian_recovery_passed`;
- `ie3_linear_gaussian_recovery_passed_with_caveats`;
- `ie3_linear_gaussian_recovery_blocked`.
