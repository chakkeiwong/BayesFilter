# Audit: structural SVD 12-phase execution pass

## Date

2026-05-06

## Reviewed plan

- `docs/plans/bayesfilter-structural-svd-12-phase-implementation-plan-2026-05-05.md`

## Audit stance

Pretending to be another developer, I audited the plan and this execution pass
for two failure modes:

1. under-execution, where existing implemented gates are ignored and the same
   planning work is repeated;
2. over-claiming, where passing toy tests are treated as DSGE, derivative, JIT,
   or HMC readiness.

The correct reading is in the middle: BayesFilter has already implemented and
tested the early structural contracts, exact Kalman spine, generic fixtures,
and adapter gates; it has not closed the model-specific DSGE, derivative, JIT,
or HMC promotion gates.

## Plan completeness

The twelve phases are complete as a roadmap.  They cover:

1. mathematical source audit;
2. code reuse and migration audit;
3. structural sigma-point core;
4. exact Kalman and degenerate linear spine;
5. generic structural fixtures;
6. MacroFinance adapter and derivative spine;
7. DSGE adapter integration;
8. model-specific DSGE completion evidence;
9. derivative and Hessian safety;
10. JIT/static-shape production readiness;
11. HMC validation ladder;
12. documentation and release provenance.

No new phase is required before execution.  The key audit requirement is to
respect the gates exactly as written.

## Evidence reviewed

BayesFilter code:

- `bayesfilter/structural.py`;
- `bayesfilter/filters/sigma_points.py`;
- `bayesfilter/filters/kalman.py`;
- `bayesfilter/filters/particles.py`;
- `bayesfilter/adapters/dsge.py`;
- `bayesfilter/adapters/macrofinance.py`;
- `bayesfilter/backends.py`;
- `bayesfilter/testing/structural_fixtures.py`.

BayesFilter tests:

- `tests/test_structural_partition.py`;
- `tests/test_structural_sigma_points.py`;
- `tests/test_structural_ar_p.py`;
- `tests/test_filter_metadata.py`;
- `tests/test_degenerate_kalman.py`;
- `tests/test_macrofinance_adapter.py`;
- `tests/test_dsge_adapter_gate.py`;
- `tests/test_derivative_validation_smoke.py`;
- `tests/test_backend_readiness.py`;
- full `pytest -q`.

Source and plan evidence:

- `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`;
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex`;
- `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex`.

## Phase-by-phase audit decision

| Phase | Audit decision | Reason |
| --- | --- | --- |
| 1. Mathematical source audit | Satisfied for BayesFilter-local contracts | Existing audit classifies exact LGSSM, structural nonlinear, and labeled approximations. |
| 2. Code reuse/migration audit | Satisfied for current BayesFilter pass | Existing audit rejects blind migration of DSGE SVD and MacroFinance derivatives. |
| 3. Structural sigma-point core | Passed as eager reference | Structural tests pass; metadata is honest: eager NumPy and finite-difference smoke only. |
| 4. Exact Kalman spine | Passed as value reference | Degenerate Kalman tests pass; exact path remains distinct. |
| 5. Generic fixtures | Passed for toy/generic coverage | AR(2), nonlinear accumulation, and structural sigma-point fixtures pass. |
| 6. MacroFinance adapter | Passed for value adapter gates | MacroFinance economics and derivative recursions remain client-owned. |
| 7. DSGE adapter | Passed for metadata gate | Mixed metadata without completion map fails closed. |
| 8. DSGE completion evidence | Blocked | No Rotemberg/SGU/EZ/NAWM residual evidence in this pass. |
| 9. Derivative/Hessian safety | Guardrails pass, certification blocked | Tests validate blocking logic, not analytic SVD filter derivatives. |
| 10. JIT/static-shape gate | Blocked | Current structural sigma-point and Kalman references are eager. |
| 11. HMC ladder | Blocked | Value, residual, derivative, and JIT gates are not all closed. |
| 12. Release gate | Documentation for this pass complete; production release blocked | Evidence docs exist, but production/HMC claims are not justified. |

## Test results

Commands run:

```bash
pytest -q tests/test_structural_partition.py tests/test_structural_sigma_points.py tests/test_structural_ar_p.py tests/test_filter_metadata.py
pytest -q tests/test_degenerate_kalman.py tests/test_filter_metadata.py
pytest -q tests/test_structural_ar_p.py tests/test_structural_sigma_points.py
pytest -q tests/test_macrofinance_adapter.py tests/test_degenerate_kalman.py
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
pytest -q tests/test_derivative_validation_smoke.py
pytest -q tests/test_backend_readiness.py
pytest -q
git diff --check
```

Results:

- structural core subset: `14 passed`;
- degenerate Kalman subset: `5 passed`;
- generic fixture subset: `6 passed`;
- MacroFinance adapter subset: `34 passed`;
- DSGE adapter subset: `5 passed`;
- derivative smoke: `1 passed`;
- backend readiness: `4 passed`;
- full suite: `63 passed`;
- `git diff --check`: passed.

Observed warnings:

- TensorFlow Probability deprecation warnings in the optional MacroFinance
  comparison test;
- pytest-cache write warnings because `.pytest_cache` under BayesFilter is
  read-only in this sandbox.

Neither warning changes the audit decision.

## Issues found

### 1. Phase 8 remains the central modeling blocker

Rotemberg, SGU, EZ, and NAWM-scale DSGE filtering cannot be promoted from the
BayesFilter adapter gate alone.  The missing evidence is model-specific:

- exact state role metadata;
- deterministic completion maps;
- sigma-point or particle residual tests;
- second-order/pruned residual checks where relevant;
- EZ timing classification.

### 2. Phase 9 is only a guardrail today

The backend gate correctly blocks small-gap spectral derivative claims without
finite-difference and JVP/VJP evidence.  That is necessary but not sufficient.
It does not derive or certify analytic gradients/Hessians for the structural
SVD sigma-point filter.

### 3. Phase 10 has no production backend yet

The current reference paths declare `eager_numpy` or `eager`.  That is the right
metadata.  It also means HMC speed and XLA/static-shape claims are still open.

### 4. HMC should not be run as a shortcut

Running HMC before Phase 8 residual evidence and Phase 9/10 derivative/JIT
evidence would recreate the original debugging pattern: sampler symptoms would
be used to diagnose a target whose structural semantics and gradients were not
yet certified.

## Required modification to the plan

No structural modification is required.  Add one operational note for the next
coding agent:

> Treat Phases 8--11 as promotion gates.  Passing Phases 3--7 allows toy and
> adapter validation, but it does not authorize DSGE nonlinear HMC benchmarks.

This note is reflected in the execution reset memo.

## Next implementation hypotheses

H1: A DSGE client-side residual-test harness will expose whether Rotemberg and
SGU deterministic completion maps are sufficient for structural sigma-point
filtering.

H2: If spectral-gap telemetry frequently approaches zero in realistic SVD
filter paths, a non-spectral custom-gradient or alternative factor backend will
be required for HMC.

H3: A production JAX/TF backend can be built only after the integration space,
state dimension, innovation dimension, and observation mask conventions are
made static enough for compilation.

H4: The exact Kalman and structural AR(2) fixtures should remain mandatory
regression tests before every DSGE/HMC promotion.

## Audit decision

Proceed with committing this execution documentation.  Do not implement or
stage backend code in this pass.  The next coding session should begin with the
Phase 8 client residual harness or the Phase 9 derivative-certification work,
depending on whether the priority is DSGE structural correctness or HMC
gradient safety.
