# Reset memo: structural SVD 12-phase execution pass

## Date

2026-05-06

## Scope

This reset memo records the execution pass for
`docs/plans/bayesfilter-structural-svd-12-phase-implementation-plan-2026-05-05.md`.
The pass was run from `/home/chakwong/BayesFilter`.

The important interpretation is conservative:

- execute the phases that already have enough mathematical/source evidence and
  existing BayesFilter tests;
- do not rewrite backend code merely because the plan contains implementation
  phases;
- do not promote DSGE nonlinear filtering, SVD/eigen derivatives, JIT targets,
  or HMC convergence beyond the evidence actually present;
- do not stage unrelated dirty files from another agent.

## Baseline repository state

Observed BayesFilter status before this pass:

```text
## main...origin/main [ahead 2]
 M docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
 M docs/references.bib
?? "docs/A general method for approximating non-linear transformations of probability distributions Julier(96).pdf"
?? docs/plans/ch18b-restructuring-and-literature-strengthening-plan-2026-05-05.md
?? docs/plans/templates/
```

These files were treated as pre-existing unrelated work and were not staged by
this pass.

Observed `/home/chakwong/python` status:

```text
## main...origin/main
?? .codex
?? .serena/
```

Those untracked files were not touched.

## Plan audit

The plan remains correctly ordered.  Its key safety property is that it keeps
three paths separate:

1. exact collapsed LGSSM/Kalman filtering;
2. structural nonlinear filtering over the declared stochastic or innovation
   variables, with deterministic completion pointwise;
3. explicitly labeled approximations, especially mixed full-state nonlinear
   approximations.

No blocking omission was found in the plan.  The required operational
clarification is that Phases 8--12 are promotion gates, not a license to claim
DSGE/HMC readiness from existing smoke tests.

See
`docs/plans/bayesfilter-structural-svd-12-phase-execution-audit-2026-05-06.md`
for the detailed independent audit.

## Phase 1: mathematical source audit

### Plan

Confirm that the mathematical source audit exists and that it separates exact
LGSSM, structural nonlinear, and labeled-approximation paths.

### Execute

Reviewed:

- `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`;
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex`;
- `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex`;
- BayesFilter structural contracts and tests.

The existing source audit supplies the required classification table and states
that DSGE Rotemberg, SGU, EZ, and NAWM adapters remain blocked until
model-specific structural maps and residual tests are supplied.

### Test

Ran:

```bash
rg -n "exact|structural nonlinear|labeled approximation|deterministic completion" docs/plans
```

Result: passed with many expected hits in structural plans and reset memos.

### Audit

The audit does not treat zero rows of a shock-impact matrix as the source of
truth for state roles.  It treats explicit structural metadata and completion
maps as the source of truth.

### Interpretation

Phase 1 is satisfied for BayesFilter-local structural contracts and toy
fixtures.  It is not satisfied for promoting Rotemberg, SGU, EZ, or NAWM
nonlinear filters.

### Next phase justified?

Yes.  Phase 2 is justified as a code reuse and migration audit.

## Phase 2: code reuse and migration audit

### Plan

Classify candidate code in BayesFilter, DSGE, and MacroFinance as reuse,
localized-fix, labeled approximation, or blocked.

### Execute

Reviewed:

- `bayesfilter/structural.py`;
- `bayesfilter/filters/sigma_points.py`;
- `bayesfilter/filters/kalman.py`;
- `bayesfilter/filters/particles.py`;
- `bayesfilter/adapters/dsge.py`;
- `bayesfilter/adapters/macrofinance.py`;
- `bayesfilter/backends.py`;
- tests under `tests/`.

The existing source/code audit classifies:

- covariance Kalman as exact LGSSM value reference;
- BayesFilter structural sigma points as approximate Gaussian structural
  reference with pointwise completion;
- DSGE default SVD sigma adapter as not reusable as the structural backend for
  mixed nonlinear DSGE;
- MacroFinance differentiated Kalman as a later derivative-provider candidate;
- particle filtering as value/Monte Carlo only unless separately audited.

### Test

Ran file/code searches over BayesFilter contracts and tests.  The search found
the expected metadata, approximation-label, differentiability-status, and
compiled-status gates.

### Audit

Reuse decisions are adequate for BayesFilter-local execution.  Client economic
logic remains in DSGE and MacroFinance.

### Interpretation

Phase 2 is satisfied enough to validate the existing BayesFilter core tests.

### Next phase justified?

Yes.  Phase 3 is justified as validation of the existing structural sigma-point
core, not as an unscoped rewrite.

## Phase 3: BayesFilter structural sigma-point core

### Plan

Validate that the structural sigma-point backend uses declared innovation
integration, deterministic completion, and honest metadata.

### Execute

Reviewed and tested:

- `bayesfilter/filters/sigma_points.py`;
- `bayesfilter/structural.py`;
- `bayesfilter/testing/structural_fixtures.py`;
- structural sigma-point tests.

### Test

Ran:

```bash
pytest -q tests/test_structural_partition.py tests/test_structural_sigma_points.py tests/test_structural_ar_p.py tests/test_filter_metadata.py
```

Result:

```text
14 passed, 1 warning
```

The warning was a pytest-cache write warning caused by the read-only sandboxed
BayesFilter `.pytest_cache`, not a test failure.

### Audit

The existing reference backend supports `innovation` integration and blocks
other integration spaces with `NotImplementedError`.  Metadata labels the value
path as `structural_svd_sigma_point`, `finite_difference_smoke_only`, and
`eager_numpy`.  The default approximation label is
`sigma_point_gaussian_closure`, so it does not claim exact nonlinear likelihood.

### Interpretation

Phase 3 is passed for the eager NumPy reference backend.  It is not a compiled
or HMC-ready production backend.

### Next phase justified?

Yes.  Exact Kalman separation can be validated independently.

## Phase 4: exact Kalman and degenerate linear spine

### Plan

Validate that exact covariance-form Kalman filtering remains separate from
nonlinear structural sigma-point filtering and tolerates singular process
covariance where mathematically valid.

### Execute

Reviewed and tested:

- `bayesfilter/filters/kalman.py`;
- degenerate Kalman tests;
- metadata tests.

### Test

Ran:

```bash
pytest -q tests/test_degenerate_kalman.py tests/test_filter_metadata.py
```

Result:

```text
5 passed, 1 warning
```

### Audit

Exact Kalman metadata uses `filter_name="covariance_kalman"`,
`integration_space="full_state"`, `deterministic_completion="none"`,
`approximation_label=None`, `differentiability_status="value_only"`, and
`compiled_status="eager"`.

### Interpretation

Phase 4 is passed for the value-side exact linear Gaussian reference.  It is
not a derivative/HMC gate.

### Next phase justified?

Yes.  Generic structural fixtures can be validated.

## Phase 5: generic structural fixtures

### Plan

Validate that the endogenous/exogenous split is treated as a generic
state-space issue, not merely a DSGE special case.

### Execute

Reviewed:

- AR(2) lag shift fixture;
- nonlinear accumulation fixture;
- worked structural UKF fixture.

### Test

Ran:

```bash
pytest -q tests/test_structural_ar_p.py tests/test_structural_sigma_points.py
```

Result:

```text
6 passed, 1 warning
```

### Audit

The AR(2) model explicitly declares one stochastic coordinate and one
deterministic lag coordinate.  The nonlinear accumulation model declares a
stochastic `m` block and deterministic `k` block.  These fixtures exercise the
generic structural contract and are not DSGE-specific.

### Interpretation

Phase 5 is passed for current toy/generic fixtures.  Dense quadrature coverage
for richer nonlinear cases remains a useful follow-up.

### Next phase justified?

Yes.  MacroFinance adapter readiness can be tested as value/derivative-provider
gates.

## Phase 6: MacroFinance adapter and analytic derivative spine

### Plan

Validate that BayesFilter can normalize MacroFinance-shaped LGSSM objects
without importing MacroFinance economics, and that derivative providers remain
gated.

### Execute

Reviewed:

- `bayesfilter/adapters/macrofinance.py`;
- MacroFinance derivative source note;
- MacroFinance adapter tests.

### Test

Ran:

```bash
pytest -q tests/test_macrofinance_adapter.py tests/test_degenerate_kalman.py
```

Result:

```text
34 passed, 3 warnings
```

Warnings were TensorFlow Probability deprecation warnings and the pytest-cache
write warning.

### Audit

The adapter keeps model construction and derivative recursions client-owned.
BayesFilter records value-side Kalman results and readiness metadata; it does
not claim that MacroFinance analytic score/Hessian code has been migrated or
certified for BayesFilter production HMC.

### Interpretation

Phase 6 is passed for value-side adapter readiness and metadata gates.  Full
analytic derivative/Hessian migration remains future work.

### Next phase justified?

Yes.  DSGE metadata adapter gates can be validated.

## Phase 7: DSGE adapter integration

### Plan

Validate fail-closed DSGE metadata handling without importing DSGE economics
into BayesFilter.

### Execute

Reviewed:

- `bayesfilter/adapters/dsge.py`;
- `tests/test_dsge_adapter_gate.py`.

### Test

Ran:

```bash
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

Result:

```text
5 passed, 1 warning
```

### Audit

Mixed DSGE metadata without a deterministic completion map is blocked.  Passing
adapter metadata is correctly treated as adapter readiness, not nonlinear
filtering correctness.

### Interpretation

Phase 7 is passed for BayesFilter adapter gates.  It does not certify Rotemberg,
SGU, EZ, or NAWM nonlinear structural filtering.

### Next phase justified?

Only as model-specific evidence work in the owning DSGE project.  BayesFilter
should not promote these models from adapter readiness alone.

## Phase 8: model-specific DSGE completion evidence

### Plan

Check whether the current BayesFilter pass can promote Rotemberg, SGU, EZ, or
NAWM-style nonlinear filtering.

### Execute

Reviewed the DSGE gate semantics and Chapter 18b warnings.

### Test

No model-specific Rotemberg/SGU/EZ residual test was run in this BayesFilter
pass.  The available BayesFilter test is the adapter gate from Phase 7.

### Audit

The plan explicitly says first-order bridge readiness is not enough.  The
BayesFilter adapter gate does not prove second-order/pruned residual identities
or EZ timing.

### Interpretation

Phase 8 is not closed.  It remains blocked pending client-owned residual tests
and timing audits.

### Next phase justified?

Derivative and backend safety gates can be tested as generic metadata gates,
but no DSGE nonlinear filtering promotion is justified.

## Phase 9: derivative and Hessian safety gate

### Plan

Validate that derivative/Hessian claims are blocked unless finite-difference,
JVP/VJP, Hessian, and spectral-gap evidence exists.

### Execute

Reviewed:

- `bayesfilter/backends.py`;
- `tests/test_backend_readiness.py`;
- `tests/test_derivative_validation_smoke.py`.

### Test

Ran:

```bash
pytest -q tests/test_derivative_validation_smoke.py
pytest -q tests/test_backend_readiness.py
```

Results:

```text
1 passed, 1 warning
4 passed, 1 warning
```

### Audit

The tests validate the gate logic, including blocking spectral derivatives near
small gaps when JVP/VJP checks are missing.  They do not certify the SVD
sigma-point filter's analytic gradient or Hessian.

### Interpretation

Phase 9 guardrails pass.  Certified SVD sigma-point gradient/Hessian support
remains open.

### Next phase justified?

Only metadata/JIT-readiness gate validation is justified.  Production HMC
promotion is not justified.

## Phase 10: JIT and static-shape production gate

### Plan

Check whether existing backends claim compiled target readiness.

### Execute

Reviewed backend metadata and readiness tests.

### Test

Covered by:

```bash
pytest -q tests/test_backend_readiness.py
pytest -q
```

Full suite result:

```text
63 passed, 3 warnings
```

### Audit

The structural sigma-point backend declares `compiled_status="eager_numpy"`.
The covariance Kalman reference declares `compiled_status="eager"`.  No code in
this pass claims XLA/JIT-ready HMC targets.

### Interpretation

Phase 10 is not closed as a production gate.  It is correctly blocked by
metadata.

### Next phase justified?

No HMC validation ladder is justified for structural SVD until compiled value
and gradient parity tests exist.

## Phase 11: HMC validation ladder

### Plan

Determine whether existing evidence justifies running HMC validation.

### Execute

Reviewed HMC-related gate labels and the plan's promotion rules.

### Test

No HMC validation test was run in this pass because Phases 8--10 are not closed.

### Audit

Smoke tests, finite values, and adapter readiness are not convergence evidence.
No posterior convergence claim is justified.

### Interpretation

Phase 11 is blocked.

### Next phase justified?

Only documentation/provenance updates are justified.

## Phase 12: documentation, provenance, and release gate

### Plan

Record the evidence, avoid unsupported claims, and do not commit generated PDFs
or unrelated dirty files.

### Execute

Created this reset memo and the execution audit/result note for another agent.

### Test

Ran:

```bash
git diff --check
pytest -q
```

Results:

```text
git diff --check: passed
63 passed, 3 warnings
```

### Audit

No generated PDF was staged.  Pre-existing dirty reset memo/reference/PDF work
was left untouched.

### Interpretation

Documentation/provenance for this execution pass is complete.  Release-level
structural SVD HMC readiness is not complete.

## Final status

Closed for this pass:

- Phase 1 source audit exists and supports BayesFilter-local contracts.
- Phase 2 code reuse decisions exist.
- Phase 3 structural sigma-point eager reference tests pass.
- Phase 4 exact Kalman/degenerate linear tests pass.
- Phase 5 generic structural fixtures pass.
- Phase 6 MacroFinance value adapter gates pass.
- Phase 7 DSGE metadata adapter gates pass.
- Phase 9/10 guardrail tests pass as guardrails.

Not closed:

- Phase 8 model-specific DSGE completion evidence.
- Phase 9 certified analytic gradients/Hessians for SVD sigma-point filtering.
- Phase 10 compiled static-shape production target.
- Phase 11 HMC convergence ladder.
- Phase 12 release gate for production/HMC claims.

## Completion note

This execution pass is complete.  The scoped artifacts from this pass are:

- `docs/plans/bayesfilter-structural-svd-12-phase-execution-audit-2026-05-06.md`;
- `docs/plans/bayesfilter-structural-svd-12-phase-execution-reset-memo-2026-05-06.md`.

They are committed in the Git commit that contains this completion note.  The
pre-existing dirty monograph reset memo, bibliography, Julier PDF, Chapter 18b
restructuring plan, and plan templates were not staged by this pass.

## Hypotheses for next phase

H1: Rotemberg and SGU can be promoted only if their client-owned completion maps
produce pointwise residuals on sigma-point grids, including near-boundary
parameters.

H2: A custom derivative path or non-spectral factorization policy will be needed
before SVD/eigen-based sigma-point backends are safe for HMC at larger scale.

H3: A JAX or TensorFlow backend with static shape metadata can match the eager
BayesFilter value path on toy structural fixtures, but this must be tested
before any HMC benchmark.

H4: Exact LGSSM and structural AR/lag fixtures should remain the regression
oracle ladder; DSGE models should enter only after toy and generic gates stay
green.
