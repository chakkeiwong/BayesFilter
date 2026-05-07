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


## 2026-05-06 update: final tool-gated execution pass

User asked to re-run the final structural SVD execution plan with reset-memo
updates, independent audit, phase-by-phase execution, tests, tidy-up, commit,
and a detailed summary.  The active plan is:

- `docs/plans/bayesfilter-structural-svd-final-execution-plan-2026-05-06.md`

The independent audit is:

- `docs/plans/bayesfilter-structural-svd-final-execution-audit-2026-05-06.md`

The key refinement relative to the earlier twelve-phase execution memo is that
ResearchAssistant and MathDevMCP are treated as evidence gates, not optional
helpers.

### Preflight: workspace and baseline

Phase plan:
- record the workspace status;
- protect pre-existing dirty files;
- commit only scoped structural SVD execution artifacts.

Execution:
- Ran `git status --short --branch`.
- Ran `git log -5 --oneline --decorate`.
- Observed branch `main` ahead of `origin/main` with pre-existing dirty docs:
  `docs/chapters/ch18b_structural_deterministic_dynamics.tex`,
  `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`,
  `docs/references.bib`, the Julier PDF, the Chapter 18b restructuring plan,
  an untracked remaining-gap best-path plan, and plan templates.

Test:
- No code test is required for preflight.

Audit:
- No tracked BayesFilter code file was dirty before this pass.
- The pre-existing Chapter 18b/reference/planning files were not edited by this
  structural SVD execution pass.

Interpretation:
- The pass may proceed if staging remains path-scoped.

Next phase justified:
- Yes.  Phase 1 is justified.

### Phase 1: source and derivation audit

Phase plan:
- apply the stricter ResearchAssistant/MathDevMCP evidence policy to the
  source and derivation audit;
- record `source_missing` and `human_review_required` where appropriate.

Execution:
- Added
  `docs/plans/bayesfilter-structural-svd-tool-gated-source-derivation-audit-2026-05-06.md`.
- Used ResearchAssistant:
  - workspace/privacy/tool status: read-only, offline, local workflows ready;
  - local summary searches for Julier/Uhlmann UKF support, Matrix Backprop/SVD
    derivative support, Hoffman/Gelman NUTS support, and Betancourt HMC support
    returned no matching local paper summaries.
- Used MathDevMCP:
  - `latex_label_lookup` succeeded for
    `def:bf-structural-state-partition`;
  - `latex_label_lookup` succeeded for
    `asm:bf-approximation-labeling`;
  - `latex_label_lookup` succeeded for
    `prop:bf-structural-ukf-pushforward`;
  - `latex_label_lookup` succeeded for
    `eq:bf-structural-ukf-prop-deterministic-completion`;
  - prior typed obligation routing for
    `prop:bf-structural-ukf-pushforward` required human review rather than
    backend certification.

Test:
- The audit addendum records a source-support and derivation-status table.

Audit:
- The exact Kalman value path remains source-supported through local monograph
  equations and existing tests.
- The structural UKF pushforward proposition has local proof text, but
  theorem-level certification remains `human_review_required`.
- External literature claims about UKF, Matrix Backprop/SVD, and NUTS/HMC are
  `source_missing` in the current ResearchAssistant local index and must not be
  promoted without reviewed sources.

Interpretation:
- Phase 1 passes for BayesFilter-local value validation.
- Phase 1 does not close literature, derivative, DSGE residual, or HMC
  promotion gates.

Next phase justified:
- Yes.  Phase 2 is justified as a code reuse/doc-code audit.

### Phase 2: code reuse and document-code audit

Phase plan:
- re-use existing BayesFilter code unless tests or tool gates expose a real
  gap;
- use MathDevMCP document-code tools where possible.

Execution:
- Reviewed the existing source/code audit:
  `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`.
- Reviewed core code:
  `bayesfilter/structural.py`,
  `bayesfilter/filters/sigma_points.py`,
  `bayesfilter/filters/kalman.py`,
  `bayesfilter/backends.py`.
- Attempted MathDevMCP `compare_label_code` for
  `def:bf-structural-state-partition` and
  `asm:bf-approximation-labeling`.

Test:
- MathDevMCP `compare_label_code` returned internal tool execution errors for
  both labels.
- Existing tests were used as doc-code consistency evidence in later phases.

Audit:
- The tool failure is recorded as a tool limitation, not as code proof.
- Reuse decisions from the 2026-05-04 audit remain valid:
  covariance Kalman is the exact LGSSM value reference, structural sigma points
  are approximate value reference, MacroFinance derivatives are later
  candidates, and DSGE/MacroFinance economics stay in client repos.

Interpretation:
- Phase 2 passes for current scope because code/test evidence is sufficient for
  BayesFilter value gates.
- It does not certify derivative formulas or client economics.

Next phase justified:
- Yes.  Phase 3 is justified.

### Phase 3: BayesFilter structural sigma-point core

Phase plan:
- validate the existing eager NumPy structural sigma-point backend.

Execution:
- No backend code rewrite was made.
- Reviewed structural sigma-point metadata: integration space, deterministic
  completion, approximation label, differentiability status, and compiled
  status.

Test:
- Ran:

```bash
pytest -q tests/test_structural_partition.py tests/test_structural_sigma_points.py tests/test_structural_ar_p.py tests/test_filter_metadata.py
```

- Result:

```text
14 passed
```

Audit:
- The backend remains `structural_svd_sigma_point`,
  `sigma_point_gaussian_closure`, `finite_difference_smoke_only`, and
  `eager_numpy`.

Interpretation:
- Phase 3 passes as a value-side approximate structural reference.
- It is not derivative-, compiled-, or HMC-ready evidence.

Next phase justified:
- Yes.  Phase 4 is justified.

### Phase 4: exact Kalman and degenerate linear spine

Phase plan:
- validate the exact covariance-form Kalman path separately from structural
  nonlinear approximation.

Execution:
- No Kalman code rewrite was made.

Test:
- Ran:

```bash
pytest -q tests/test_degenerate_kalman.py tests/test_filter_metadata.py
```

- Result:

```text
5 passed
```

Audit:
- Exact Kalman metadata remains `covariance_kalman`, `full_state`,
  `deterministic_completion="none"`, no approximation label, `value_only`, and
  `eager`.

Interpretation:
- Phase 4 passes as an exact LGSSM value gate.

Next phase justified:
- Yes.  Phase 5 is justified.

### Phase 5: generic structural fixtures

Phase plan:
- validate that structural fixtures are generic, not just DSGE-specific.

Execution:
- No fixture code changes were required.

Test:
- Ran:

```bash
pytest -q tests/test_structural_ar_p.py tests/test_structural_sigma_points.py
```

- Result:

```text
6 passed
```

Audit:
- The generic AR/structural fixtures remain green and support the BayesFilter
  structural contract.

Interpretation:
- Phase 5 passes.

Next phase justified:
- Yes.  Phase 6 is justified.

### Phase 6: MacroFinance adapter and analytic derivative classification

Phase plan:
- validate MacroFinance adapter/readiness gates without migrating economics or
  derivative formulas.

Execution:
- No MacroFinance code was edited.

Test:
- Ran:

```bash
pytest -q tests/test_macrofinance_adapter.py tests/test_degenerate_kalman.py
```

- Result:

```text
34 passed, 2 warnings
```

- Warnings were TensorFlow Probability deprecation warnings.

Audit:
- The adapter gate remains a value/readiness metadata gate.
- Analytic derivatives are not migrated or certified.

Interpretation:
- Phase 6 passes for adapter classification only.

Next phase justified:
- Yes.  Phase 7 is justified.

### Phase 7: DSGE adapter integration

Phase plan:
- validate BayesFilter DSGE adapter gates and client metadata contracts.

Execution:
- No BayesFilter or DSGE adapter code was edited.

Test:
- Ran:

```bash
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

- Result:

```text
5 passed
```

- Ran from `/home/chakwong/python`:

```bash
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q tests/contracts/test_structural_dsge_partition.py
```

- Result:

```text
11 passed, 3 warnings
```

- Warnings were two TensorFlow Probability deprecation warnings and one
  read-only pytest cache warning.

Audit:
- SmallNK, Rotemberg, and SGU metadata are adapter-ready.
- EZ remains fail-closed.
- Passing adapter metadata is not nonlinear filtering correctness.

Interpretation:
- Phase 7 passes as an adapter metadata gate.

Next phase justified:
- Phase 8 evaluation is justified, but only as a blocker audit unless
  model-specific residual tests exist.

### Phase 8: model-specific DSGE completion evidence

Phase plan:
- determine whether Rotemberg, SGU, and EZ can be promoted.

Execution:
- No model-specific residual tests were added or run in this pass.

Test:
- The relevant evidence remains the Phase 7 adapter gate only.

Audit:
- Rotemberg still needs second-order/pruned identity evidence for
  `dy_next = y_next - y_current`.
- SGU still needs residual evidence for `d,k,r,riskprem`.
- EZ still needs a timing/partition audit before metadata exposure.

Interpretation:
- Phase 8 is blocked.

Next phase justified:
- Phase 9 guardrail validation is justified.
- DSGE nonlinear promotion is not justified.

### Phase 9: derivative and Hessian safety gate

Phase plan:
- validate derivative/Hessian guardrails without claiming SVD/eigen derivative
  certification.

Execution:
- No backend code was changed.

Test:
- Ran:

```bash
pytest -q tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
```

- Result:

```text
5 passed
```

Audit:
- The guardrail tests block derivative/HMC promotion when JVP/VJP and spectral
  gap evidence are missing.
- ResearchAssistant local source support for Matrix Backprop/SVD derivative
  claims is currently `source_missing`.

Interpretation:
- Phase 9 guardrails pass.
- Certified SVD/eigen analytic gradients and Hessians remain blocked.

Next phase justified:
- Phase 10 metadata/JIT readiness audit is justified.

### Phase 10: JIT and static-shape production gate

Phase plan:
- determine whether current backends claim compiled target readiness.

Execution:
- Reviewed metadata through tests and code.

Test:
- Covered by:

```bash
pytest -q tests/test_backend_readiness.py
pytest -q
```

- Full suite result:

```text
63 passed, 2 warnings
```

Audit:
- Structural sigma points are `eager_numpy`.
- Covariance Kalman is `eager`.
- No compiled production HMC target is claimed.

Interpretation:
- Phase 10 is blocked as a production gate and correctly guarded by metadata.

Next phase justified:
- HMC execution is not justified.
- Phase 11 should be recorded as blocked.

### Phase 11: HMC validation ladder

Phase plan:
- determine whether an HMC ladder can run.

Execution:
- No HMC ladder was run because Phases 8--10 are not closed.

Test:
- No HMC tests were run.

Audit:
- No convergence claim is supported.
- ResearchAssistant local source support for HMC/NUTS literature claims is
  currently `source_missing` in the local summary index.

Interpretation:
- Phase 11 is blocked.

Next phase justified:
- Phase 12 documentation/provenance is justified.

### Phase 12: documentation, provenance, and release gate

Phase plan:
- update docs/provenance artifacts for the execution pass;
- keep unsupported claims blocked;
- run final validation.

Execution:
- Added:
  - `docs/plans/bayesfilter-structural-svd-final-execution-plan-2026-05-06.md`;
  - `docs/plans/bayesfilter-structural-svd-final-execution-audit-2026-05-06.md`;
  - `docs/plans/bayesfilter-structural-svd-tool-gated-source-derivation-audit-2026-05-06.md`;
  - `docs/plans/bayesfilter-structural-svd-final-execution-result-2026-05-06.md`.
- Updated `docs/source_map.yml` to register the new artifacts.
- Updated this reset memo with phase-by-phase results.

Test:
- Ran:

```bash
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
pytest -q
```

- Results:

```text
YAML parse passed
git diff --check passed
63 passed, 2 warnings
```

Audit:
- The final plan and result preserve the blockers for Phase 8 residuals, Phase
  9 derivative certification, Phase 10 JIT, and Phase 11 HMC.
- Pre-existing unrelated dirty files were not staged.

Interpretation:
- Phase 12 passes for documentation/provenance of this execution pass.
- Release-level production/HMC readiness remains blocked.

## Final tool-gated execution status

Closed:

- final execution plan and independent audit;
- Phase 1 tool-gated source/derivation addendum for BayesFilter-local value
  validation;
- Phase 2 code reuse decision for current scope;
- Phase 3 structural sigma-point value gate;
- Phase 4 exact Kalman value gate;
- Phase 5 generic structural fixtures;
- Phase 6 MacroFinance adapter classification;
- Phase 7 DSGE adapter metadata;
- Phase 9 and Phase 10 guardrail tests as blockers/metadata gates;
- Phase 12 documentation/provenance for this pass.

Still blocked:

- Phase 8 Rotemberg second-order/pruned identity evidence;
- Phase 8 SGU deterministic residual evidence;
- Phase 8 EZ timing/partition audit;
- Phase 9 SVD/eigen analytic gradient and Hessian certification;
- Phase 10 compiled static-shape production target;
- Phase 11 HMC ladder and convergence claims;
- external literature-source support in the current ResearchAssistant local
  summary index for UKF, Matrix Backprop/SVD, and HMC/NUTS claims.

## Next hypotheses

H1: Rotemberg's mixed structural metadata is adapter-ready, but nonlinear
structural promotion will fail unless the second-order/pruned implementation
preserves `dy_next = y_next - y_current` pointwise on sigma-point grids.

H2: SGU's deterministic coordinates `d,k,r,riskprem` require a model-specific
residual derivation; failure near boundary parameters should keep SGU
fail-closed for nonlinear structural sigma-point filtering.

H3: EZ can be made adapter-ready with minimal BayesFilter impact only after a
timing/partition audit identifies the stochastic and deterministic blocks
without relying solely on zero rows of `eta`.

H4: SVD/eigen derivative certification will fail near small spectral gaps
unless the backend records validity regions, uses custom derivatives, or
switches factorization policy.

H5: A compiled static-shape target can probably match the eager BayesFilter
value path on toy structural fixtures, but this must be tested before any HMC
benchmark.

H6: HMC diagnostics will be meaningful only after value, residual, derivative,
and compiled-target gates pass; finite smoke tests should remain labeled as
finite smoke only.


## 2026-05-06 update: six-blocker closure execution

User asked to execute the six-blocker closure plan with reset-memo updates,
independent audit, phase-by-phase plan/execute/test/audit/tidy cycles, commit,
and final hypotheses.

Active plan:

- `docs/plans/bayesfilter-structural-svd-six-blocker-closure-plan-2026-05-06.md`

Independent audit:

- `docs/plans/bayesfilter-structural-svd-six-blocker-closure-audit-2026-05-06.md`

Result note:

- `docs/plans/bayesfilter-structural-svd-six-blocker-closure-result-2026-05-06.md`

### Plan audit

Plan:
- audit the six-blocker plan as another developer before execution.

Execution:
- Added
  `docs/plans/bayesfilter-structural-svd-six-blocker-closure-audit-2026-05-06.md`.

Test:
- Manual audit against:
  - `docs/plans/bayesfilter-structural-svd-final-execution-result-2026-05-06.md`;
  - existing BayesFilter backend readiness tests;
  - existing DSGE strong residual gate artifacts in `/home/chakwong/python`.

Audit:
- The plan orders work correctly: residuals before derivatives, compiled
  parity, and HMC.
- The plan keeps DSGE economic semantics in `/home/chakwong/python`.
- The plan correctly treats blocker assertions as useful results, not model
  promotions.

Interpretation:
- The plan is valid for execution.

Next phase justified:
- Yes.  Blocker 0 is justified.

### Blocker 0: preflight and evidence freeze

Plan:
- record BayesFilter and DSGE client status;
- isolate the write set.

Execution:
- In BayesFilter, ran `git status --short --branch` and
  `git log -5 --oneline --decorate`.
- BayesFilter status included pre-existing dirty/untracked documentation and
  PDF/template files:
  - `docs/chapters/ch18b_structural_deterministic_dynamics.tex`;
  - `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`;
  - `docs/references.bib`;
  - Julier and van der Merwe PDFs;
  - Chapter 18b reviewer-response plan;
  - plan templates.
- In `/home/chakwong/python`, ran `git status --short --branch` and
  `git log -5 --oneline --decorate`.
- The DSGE client is ahead of `origin/main` by one commit and only `.codex/`
  and `.serena/` are untracked.

Test:
- BayesFilter guardrail baseline:

```bash
pytest -q tests/test_dsge_adapter_gate.py tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
```

- Result:

```text
10 passed
```

Audit:
- No BayesFilter code file is dirty.
- This pass owns only BayesFilter plan/audit/result/source-map/reset artifacts.
- DSGE client strong residual artifacts already exist and are treated as
  external evidence; no client files are staged by this BayesFilter pass.

Interpretation:
- The write set is scoped.

Next phase justified:
- Yes.  Blockers 1--3 can be evaluated from the DSGE client tests.

### Blocker 1: Rotemberg residual gate

Plan:
- test whether Rotemberg second-order/pruned propagation satisfies
  `dy_next = y_next - y_current`.

Execution:
- Used the DSGE client test:
  `tests/contracts/test_dsge_strong_structural_residual_gates.py`.
- The test verifies that Rotemberg second-order H-S measurement equations match
  the hand-written measurement equations.
- The test then asserts the stronger blocker:
  `blocked_pruned_second_order_dy_identity_residual`.

Test:
- Ran from `/home/chakwong/python`:

```bash
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

- Result:

```text
19 passed, 3 warnings
```

Audit:
- The passing test is a passing blocker assertion, not a Rotemberg promotion.
- Rotemberg measurement equations pass, but the second-order/pruned `dy`
  identity is materially nonzero under the current completion path.

Interpretation:
- Blocker 1 is closed as a named blocker:
  `blocked_pruned_second_order_dy_identity_residual`.

Next phase justified:
- Yes.  SGU residual evaluation is still justified.

### Blocker 2: SGU residual gate

Plan:
- test whether SGU's current completion bridge satisfies nonlinear canonical
  residuals.

Execution:
- Used the DSGE client strong residual gate.
- The test evaluates `SGUEstimable(use_second_order=True,
  estimate_obs_noise=False)` against canonical residuals after applying the
  current completion bridge.

Test:
- Covered by the same DSGE command:

```text
19 passed, 3 warnings
```

Audit:
- The result proves the existing linear completion bridge is not enough for
  nonlinear equilibrium-manifold certification.
- It does not weaken the previous `linear_completion_bridge_passed` label.

Interpretation:
- Blocker 2 is closed as a named blocker:
  `blocked_nonlinear_equilibrium_manifold_residual`.

Next phase justified:
- Yes.  EZ timing/fail-closed evaluation is justified.

### Blocker 3: EZ timing and partition gate

Plan:
- probe EZ timing/code shape without exposing metadata.

Execution:
- Used the DSGE strong residual gate.
- The test verifies EZ has two states and two shocks and that shock-impact rows
  are `(0, 1)`, while BayesFilter metadata is still absent.

Test:
- Covered by the same DSGE command:

```text
19 passed, 3 warnings
```

Audit:
- A code-level all-stochastic-looking probe is not source-backed timing
  metadata.
- EZ remains fail-closed in the BayesFilter adapter.

Interpretation:
- Blocker 3 is closed as a fail-closed blocker:
  `blocked_pending_source_backed_timing_metadata`.

Next phase justified:
- Only guardrail evaluation is justified.  Since Rotemberg, SGU, and EZ are not
  promotable, derivative/JIT/HMC implementation is not justified for those
  models.

### Blocker 4: SVD/eigen derivative and Hessian certification

Plan:
- determine whether derivative/Hessian certification can proceed.

Execution:
- Reviewed `bayesfilter/backends.py`.
- Re-ran backend guardrails.

Test:
- Ran:

```bash
pytest -q tests/test_dsge_adapter_gate.py tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
```

- Result:

```text
10 passed
```

Audit:
- Existing guardrails can block unsupported spectral derivative claims.
- No target/model pair from Blockers 1--3 is promotable.
- No new `tests/test_structural_svd_derivative_certification.py` was created,
  because model residual gates are blocked.

Interpretation:
- Blocker 4 remains blocked with label `value_only_blocked`.

Next phase justified:
- Compiled parity can be evaluated only as a blocker/metadata gate.

### Blocker 5: compiled static-shape parity

Plan:
- determine whether compiled parity is justified.

Execution:
- Reviewed current backend metadata.
- Structural sigma-point paths remain `eager_numpy`.
- Covariance Kalman remains `eager`.

Test:
- Covered by BayesFilter guardrails and full test suite:

```bash
pytest -q
```

- Result:

```text
63 passed, 2 warnings
```

Audit:
- No compiled structural SVD target exists.
- A compiled parity test would be premature for Rotemberg, SGU, and EZ because
  their model-specific residual/timing gates are blocked.

Interpretation:
- Blocker 5 remains blocked with label `compiled_parity_not_started`.

Next phase justified:
- HMC can only be recorded as not justified.

### Blocker 6: HMC diagnostics ladder

Plan:
- decide whether HMC diagnostics can run.

Execution:
- No HMC ladder was run because residual, derivative, and compiled gates are
  not closed for any target model in this blocker plan.

Test:
- No HMC tests were run.

Audit:
- Running HMC would violate the plan's stop rules.
- Finite smoke tests would not be convergence evidence.

Interpretation:
- Blocker 6 remains blocked with label `hmc_not_justified`.

Next phase justified:
- Provenance cleanup is justified.

### Provenance cleanup

Plan:
- record the blocker results;
- update source-map provenance;
- run final validation.

Execution:
- Added:
  - `docs/plans/bayesfilter-structural-svd-six-blocker-closure-audit-2026-05-06.md`;
  - `docs/plans/bayesfilter-structural-svd-six-blocker-closure-result-2026-05-06.md`.
- Updated `docs/source_map.yml`.
- Updated this reset memo.

Test:
- Ran:

```bash
pytest -q
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
rg -n "converged|production-ready|HMC-ready|certified|structurally fixed" docs bayesfilter tests
```

- Results:

```text
63 passed, 2 warnings
YAML parse passed
git diff --check passed
```

- Stale-claim search found only policy text, blocker text, tests, and allowed
  label definitions.

Audit:
- The pass did not stage PDFs/templates or client repo files.
- The client strong residual gates are evidence, but the BayesFilter commit
  records only BayesFilter-side planning/provenance artifacts.

Interpretation:
- The six-blocker execution pass is complete.

## Six-blocker closure status

Closed as explicit blockers:

- Rotemberg: `blocked_pruned_second_order_dy_identity_residual`;
- SGU: `blocked_nonlinear_equilibrium_manifold_residual`;
- EZ: `blocked_pending_source_backed_timing_metadata`.

Still blocked:

- SVD/eigen derivative and Hessian certification: `value_only_blocked`;
- compiled static-shape parity: `compiled_parity_not_started`;
- HMC diagnostics ladder: `hmc_not_justified`.

## Six-blocker next hypotheses

H1: Rotemberg needs a second-order/pruned deterministic completion design for
`dy`; the next test should drive
`dy_next - (y_next - y_current)` below tolerance on deterministic sigma-point
grids.

H2: SGU needs nonlinear equilibrium-manifold completion, not only an `h_x`
linear bridge; the next test should reduce canonical residuals below tolerance
for `d,k,r,riskprem`.

H3: EZ should remain fail-closed until a source-backed timing/provenance note
explains state ordering, shock timing, and measurement timing.

H4: Derivative/Hessian certification should wait for the first promotable model
target.  Once one exists, spectral gradients require minimum-gap telemetry plus
finite-difference and JVP/VJP checks.

H5: Compiled parity should be tested first on exact LGSSM and generic
structural fixtures, then on the first DSGE model whose residual gate passes.

H6: HMC should start only after the same model/backend pair has residual,
derivative, and compiled parity evidence.


## 2026-05-06 reboot handoff

Context:
- User is rebooting the machine and asked for a reset-memo update so work can
  continue cleanly afterward.
- BayesFilter `main` is at `70ec312 Clarify phi zero structural UKF edge case`
  and matches `origin/main`.
- The structural SVD six-blocker closure commit is already in history:
  `d6a9e4a Record structural SVD six-blocker closure`.
- BayesFilter working tree has only untracked PDF/template artifacts:
  - `docs/A general method for approximating non-linear transformations of probability distributions Julier(96).pdf`;
  - `docs/Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models Merwe(03).pdf`;
  - `docs/plans/templates/`.
- `/home/chakwong/python` is ahead of its origin and has a dirty reset memo:
  `docs/plans/dsge-structural-completion-gap-closure-reset-memo-2026-05-06.md`,
  plus untracked `.codex/` and `.serena/`.  Do not revert or overwrite those.

Current structural SVD status:
- BayesFilter value/adapter gates are validated.
- DSGE strong residual gates have produced executable blocker evidence.
- Rotemberg is blocked by
  `blocked_pruned_second_order_dy_identity_residual`.
- SGU is blocked by
  `blocked_nonlinear_equilibrium_manifold_residual`.
- EZ is blocked by
  `blocked_pending_source_backed_timing_metadata`.
- SVD/eigen derivative certification remains `value_only_blocked`.
- Compiled parity remains `compiled_parity_not_started`.
- HMC remains `hmc_not_justified`.

Validated commands from the last pass:

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_dsge_adapter_gate.py tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
pytest -q
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check

cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

Observed results:
- BayesFilter guardrails: `10 passed`.
- Full BayesFilter suite: `63 passed, 2 warnings`.
- DSGE strong residual gates: `19 passed, 3 warnings`.
- Source-map YAML parse passed.
- `git diff --check` passed.

Next recommended continuation after reboot:
1. In `/home/chakwong/python`, finish or inspect the dirty
   `dsge-structural-completion-gap-closure-reset-memo-2026-05-06.md` without
   overwriting user/agent work.
2. Continue with Rotemberg first: design a second-order/pruned deterministic
   completion map for `dy`.
3. Add a focused client test that proves
   `dy_next - (y_next - y_current)` is below tolerance on deterministic
   sigma-point grids.
4. Only if Rotemberg passes that residual gate should derivative/Hessian or
   compiled parity work begin for Rotemberg.
5. Keep BayesFilter generic; do not encode DSGE economics into BayesFilter.

Do not resume by running HMC.  HMC is downstream of model residual,
derivative/Hessian, and compiled parity gates for the same model/backend pair.


## 2026-05-06 update: gap-closure master-plan execution pass

User asked to:

1. update this reset memo;
2. audit the structural SVD remaining-gap closure master plan as another
   developer;
3. execute the phases one by one with a plan/execute/test/audit/tidy/reset-memo
   cycle;
4. continue without human intervention only while the next phase remains
   justified;
5. commit the scoped modified files after the pass;
6. update this reset memo on completion;
7. provide a detailed final summary with hypotheses and next suggestions.

Active plan:

- `docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-2026-05-06.md`

Independent audit:

- `docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-audit-2026-05-06.md`

### Master-plan audit

Plan:
- audit the active master plan as if written by another developer;
- identify missing points, ordering issues, and stop-rule issues before
  executing phases.

Execution:
- Added the independent audit artifact listed above.
- Updated the master plan with an execution note that the DSGE client now has
  commit `59c05f5 Close Rotemberg structural completion gate`.

Test:
- Manual audit against:
  - the active BayesFilter master plan;
  - this reset memo's six-blocker handoff;
  - `/home/chakwong/python` commit `59c05f5`;
  - `/home/chakwong/python/docs/plans/dsge-structural-completion-gap-closure-reset-memo-2026-05-06.md`;
  - `/home/chakwong/python/tests/contracts/test_dsge_strong_structural_residual_gates.py`.

Audit:
- The plan ordering is correct: model semantics and residuals precede
  derivative, compiled, and HMC gates.
- The main execution correction is that Rotemberg is no longer a fresh
  implementation phase in the DSGE client.  It is now a validation/provenance
  phase because commit `59c05f5` already closed the tested completion gate.
- The SGU stop rule remains necessary.  The latest DSGE evidence sharpens SGU
  into a target-definition decision rather than a missing implementation line.

Interpretation:
- The plan is safe to execute through Phase 2 as validation/provenance.
- Phase 3 and later are not automatically justified if Phase 2 reproduces the
  SGU target-definition blocker.

Next phase justified?
- Yes.  Phase 0 preflight/evidence freeze is justified.

### Phase 0: preflight and evidence freeze

Plan:
- record BayesFilter and DSGE client status;
- rerun the current BayesFilter guardrails;
- rerun the current DSGE structural residual/partition suite;
- keep the write set scoped to BayesFilter planning/provenance files.

Execution:
- BayesFilter status:

```text
## main...origin/main [ahead 1]
 M docs/chapters/ch18b_structural_deterministic_dynamics.tex
 M docs/source_map.yml
?? "docs/A general method for approximating non-linear transformations of probability distributions Julier(96).pdf"
?? "docs/Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models Merwe(03).pdf"
?? docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-2026-05-06.md
?? docs/plans/templates/
```

- BayesFilter log head:

```text
2f79f6c Add structural SVD reboot handoff
70ec312 Clarify phi zero structural UKF edge case
8371198 Rewrite Chapter 18b reviewer response
d6a9e4a Record structural SVD six-blocker closure
d8d0521 Restructure Chapter 18b around structural UKF doctrine
```

- DSGE client status:

```text
## main...origin/main [ahead 2]
 M docs/plans/dsge-structural-completion-gap-closure-reset-memo-2026-05-06.md
?? .codex
?? .serena/
```

- DSGE client log head:

```text
59c05f5 Close Rotemberg structural completion gate
bf468cc Add DSGE strong structural residual gates
e75722f Document Codex GPU sandbox escalation policy
bfa16a0 Add DSGE structural completion residual harness
a4d555f Commit NeuTra German Gate 1 data defaults
```

Test:
- Ran in BayesFilter:

```bash
pytest -q tests/test_dsge_adapter_gate.py tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
```

Result:

```text
10 passed in 0.13s
```

- Ran in `/home/chakwong/python`:

```bash
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

Result:

```text
21 passed, 3 warnings in 16.30s
```

Warnings were two TensorFlow Probability `distutils` deprecation warnings and
one read-only pytest-cache warning in `/home/chakwong/python`.

Audit:
- BayesFilter has unrelated tracked Chapter 18b work and untracked PDFs/templates
  that this pass must not stage.
- `/home/chakwong/python` has a dirty reset memo from the DSGE closure pass and
  untracked tool-state directories.  This BayesFilter pass must not overwrite
  or stage them.
- The DSGE residual suite now has 21 tests, not the older 19-test baseline,
  because Rotemberg completion evidence has been added in the client repo.

Interpretation:
- Phase 0 passed.  The write set for this BayesFilter pass is scoped to:
  - the master plan;
  - its independent audit;
  - `docs/source_map.yml`;
  - this reset memo.

Next phase justified?
- Yes.  Phase 1 is justified as validation/provenance for the committed
  Rotemberg closure evidence, not as a fresh BayesFilter implementation.

### Phase 1: Rotemberg second-order/pruned completion validation

Plan:
- validate the newer DSGE client Rotemberg completion evidence;
- inspect the implementation enough to confirm it completes only the
  second-order side-state `dy` coordinate;
- do not reimplement Rotemberg in BayesFilter.

Execution:
- Reviewed `/home/chakwong/python/src/dsge_hmc/models/rotemberg_nk.py`.
- Reviewed `/home/chakwong/python/tests/contracts/test_dsge_strong_structural_residual_gates.py`.
- The DSGE client now exposes:

```text
bayesfilter_second_order_deterministic_completion(...)
```

for Rotemberg.  The helper preserves the first-order stochastic path and
overwrites only `x_s_next[dy]` so the completed total state satisfies:

```text
dy_next = y_out_next - y_out_current
```

under the second-order policy.  It fails closed to the predicted side-state
coordinate when `1 - g_x[y_out, dy]` is near singular.

Test:
- Ran in `/home/chakwong/python`:

```bash
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py -k rotemberg
```

Result:

```text
4 passed, 2 deselected, 3 warnings in 3.88s
```

Warnings were the same TensorFlow Probability deprecation and read-only
pytest-cache warnings observed in Phase 0.

Audit:
- The raw pruned Rotemberg residual blocker remains in the test suite as a
  diagnostic for the uncompleted path.
- The positive completion label remains:

```text
rotemberg_second_order_dy_completion_residual_passed
```

- The implementation does not add artificial process noise and does not modify
  BayesFilter generic filtering code.
- This is model-specific residual evidence only.  It is not derivative,
  compiled, or HMC evidence.

Interpretation:
- Phase 1 passes as validation/provenance.  The historical raw-path blocker is
  still meaningful, but the DSGE client now has tested completion evidence for
  the Rotemberg second-order side-state `dy` identity on the committed grids.

Next phase justified?
- Yes.  Phase 2 is justified as SGU blocker validation and target-definition
  audit.

### Phase 2: SGU nonlinear equilibrium-manifold blocker validation

Plan:
- validate the current SGU blocker evidence;
- inspect the SGU deterministic-completion bridge;
- decide whether the next phase remains justified without a new SGU target
  choice.

Execution:
- Reviewed `/home/chakwong/python/src/dsge_hmc/models/sgu.py`.
- Reviewed `/home/chakwong/python/docs/plans/sgu-nonlinear-deterministic-completion-derivation-2026-05-06.md`.
- SGU currently exposes metadata:

```text
state_names = ('d', 'k', 'r', 'a', 'riskprem', 'zeta', 'mu')
stochastic_indices = (3, 5, 6)
deterministic_indices = (0, 1, 2, 4)
```

- SGU's current `bayesfilter_deterministic_completion(...)` explicitly uses a
  linear solution bridge for `(d,k,r,riskprem)` and documents that it is not
  the final nonlinear SGU equilibrium-manifold adapter.

Test:
- Ran in `/home/chakwong/python`:

```bash
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py -k sgu
```

Result:

```text
1 passed, 5 deselected, 3 warnings in 1.45s
```

The passing test is a passing blocker assertion:

```text
blocked_nonlinear_equilibrium_manifold_residual
```

Audit:
- The SGU evidence confirms that adapter metadata and the linear bridge are not
  enough for nonlinear structural promotion.
- The DSGE client derivation found that solving the natural four deterministic
  state equations `(7,8,10,11)` can close those selected residuals, but the full
  canonical residual remains around `8.6e-2`.
- Implementing only a four-state solve would overclaim nonlinear
  equilibrium-manifold completion.
- No BayesFilter generic contract change is forced by this evidence.

Interpretation:
- Phase 2 reproduces and sharpens the SGU blocker.  SGU now requires a
  structural target decision before code implementation:
  1. state-identity completion only;
  2. joint state-control nonlinear projection;
  3. perturbation-policy residual target.

Next phase justified?
- No.  Under the master-plan stop rules and the DSGE client reset memo, Phase 3
  EZ metadata promotion, BayesFilter contract patching, derivative
  certification, compiled parity, and HMC are not automatically justified.
  Execution stops here pending a user decision on the SGU target.

### Tidy, validation, and completion status

Plan:
- validate BayesFilter docs/provenance edits;
- run the BayesFilter full test suite;
- avoid staging unrelated Chapter 18b/PDF/template work;
- commit only the scoped planning/provenance files if validation passes.

Execution:
- Added:
  - `docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-audit-2026-05-06.md`.
- Updated:
  - `docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-2026-05-06.md`;
  - `docs/plans/bayesfilter-structural-svd-12-phase-execution-reset-memo-2026-05-06.md`;
  - `docs/source_map.yml`.
- Did not edit BayesFilter backend code.
- Did not edit or stage `/home/chakwong/python` files.
- Did not stage the pre-existing dirty Chapter 18b file, source PDFs, or plan
  templates.

Test:
- Ran:

```bash
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
rg -n "converged|production-ready|HMC-ready|certified|structurally fixed" \
  docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-2026-05-06.md \
  docs/plans/bayesfilter-structural-svd-gap-closure-master-plan-audit-2026-05-06.md \
  docs/plans/bayesfilter-structural-svd-12-phase-execution-reset-memo-2026-05-06.md \
  docs/source_map.yml
pytest -q
```

Results:

```text
YAML parse passed
git diff --check passed
stale-claim search found only blocker/policy text and existing negative claims
63 passed, 2 warnings in 16.07s
```

The two warnings were TensorFlow Probability `distutils` deprecation warnings.

Audit:
- The pass executed as far as justified by the plan.
- Rotemberg is now promoted only to the model-specific residual-completion
  evidence label recorded in the DSGE client, not to derivative, compiled, or
  HMC readiness.
- SGU remains blocked because the target is underspecified: state-only
  completion, joint state-control projection, and perturbation-policy residual
  target imply different tests and claims.
- EZ is still blocked because the SGU stop rule prevents automatic progression
  to later phases under the requested no-human-intervention protocol.

Interpretation:
- The BayesFilter-side gap-closure pass is complete as a validation/provenance
  pass.  It did not close SGU, EZ, derivative, compiled, or HMC gaps.

Next phase justified?
- No automatic next phase is justified.  The next decision is the SGU target
  choice:
  1. state-identity completion only;
  2. joint state-control nonlinear projection;
  3. perturbation-policy residual target.

## Gap-closure master-plan execution status

Closed or validated:

- master plan artifact exists and is source-map registered;
- independent master-plan audit exists;
- Phase 0 preflight/evidence freeze passed;
- Rotemberg second-order/pruned completion evidence is validated from DSGE
  client commit `59c05f5`;
- BayesFilter guardrails and full suite remain green.

Still open:

- SGU nonlinear structural target decision;
- EZ source-backed timing/partition metadata;
- SVD/eigen derivative and Hessian certification;
- compiled static-shape parity;
- HMC diagnostics ladder;
- release/documentation promotion claims for anything beyond current
  value/adapter/residual evidence.

Hypotheses to test next:

H1: SGU can be honestly advanced fastest if the next target is first narrowed
to `state_identity_completion_only`, with explicit non-promotion language for
the full nonlinear equilibrium manifold.

H2: A joint SGU state-control projection may close the full canonical residual,
but it will likely require solving for current controls together with
deterministic states and therefore changes the target from simple completion
to nonlinear projection.

H3: A perturbation-policy residual target may be the most faithful to the
existing second-order SGU solver, but it must define tolerances in perturbation
accuracy terms rather than exact-equilibrium residual terms.

H4: EZ should remain fail-closed until a timing/provenance note identifies
state ordering, shock timing, and measurement timing from sources rather than
from shock-impact rows alone.

H5: Derivative/Hessian certification should wait for the first promotable
model/backend pair; spectral derivatives need minimum-gap telemetry plus
finite-difference, JVP/VJP, and Hessian checks.

H6: Compiled parity and HMC should start on exact LGSSM and generic structural
fixtures before any DSGE target, and DSGE HMC should remain blocked until the
same target has residual, derivative, and compiled parity evidence.

## 2026-05-07 SGU combined-target execution addendum

### Scope

The user selected a stricter SGU target:

```text
Gate A: deterministic state identities close for (d,k,r,riskprem)
Gate B: quadratic/second-order policy residuals are better than
        linear/first-order policy residuals on the same declared grid
```

This addendum records execution of
`docs/plans/bayesfilter-sgu-combined-structural-target-closure-plan-2026-05-07.md`.

### Phase 0: preflight and audit

Plan:
- record BayesFilter and `/home/chakwong/python` status;
- protect unrelated dirty files;
- audit the plan as another developer before implementation;
- rerun targeted baseline tests.

Execute:
- Created:
  - `docs/plans/bayesfilter-sgu-combined-structural-target-closure-audit-2026-05-07.md`.
- Reviewed SGU canonical residual numbering and confirmed that plan equations
  `(7,8,10,11)` correspond to zero-based residual entries
  `H[7], H[8], H[10], H[11]`.
- The audit clarified that SGU risk premium uses current `zeta_t`, not
  next-period `zeta'`.

Test:
- Baseline targeted client suite:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-bayesfilter-sgu \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

Result:

```text
23 passed, 3 warnings in 18.32s
```

The warnings were TensorFlow Probability `distutils` deprecation warnings and
a sandbox `.pytest_cache` write warning.

Audit:
- The plan is executable only if Gate B remains a real stop rule.
- No BayesFilter backend code change is justified; SGU economics stay in the
  DSGE client.

Interpretation:
- Phase 0 passed.

Next phase justified?
- Yes.  Gate A implementation is model-local and directly required by the
  user-selected combined target.

### Phase 1: Gate A design and implementation

Plan:
- implement a model-owned SGU state-identity completion helper;
- preserve stochastic candidate coordinates `(a,zeta,mu)`;
- overwrite only `(d,k,r,riskprem)`;
- close selected state/timing equations under an explicit policy order.

Execute in `/home/chakwong/python`:
- Added `SGUEstimable.bayesfilter_state_identity_completion(...)`.
- The helper closes:

```text
H[7]:  exp(r') = r_w + riskprem'
H[8]:  riskprem' = zeta_t + psi(exp(d' - d_bar) - 1)
H[10]: ca_t = d'/GDP_t - d_t/GDP_t
H[11]: exp(k') = exp(kfu_t)
```

Test:
- Added `test_sgu_state_identity_completion_closes_selected_state_equations`.
- Narrowed test run:

```text
2 passed, 3 warnings in 2.37s
```

Audit:
- The helper is explicitly labeled as state-identity completion, not a full
  nonlinear equilibrium-manifold projection.
- The old linear bridge remains available and separately labeled.

Interpretation:
- Gate A passes on the declared local grid.

Allowed label:

```text
sgu_state_identity_completion_passed
```

Next phase justified?
- Yes.  Gate B diagnostic testing is required by the combined target.

### Phase 2: Gate B residual comparison

Plan:
- compare linear/first-order and quadratic/second-order full canonical
  residuals on the same completed-state grid;
- require both:

```text
quadratic_rms < linear_rms
quadratic_max <= linear_max
```

Execute:
- Added a same-grid SGU policy residual summary in
  `/home/chakwong/python/tests/contracts/test_dsge_strong_structural_residual_gates.py`.
- Added explicit blocker label:

```text
blocked_sgu_quadratic_policy_not_better_than_linear_residual
```

Test:
- Full targeted client suite after implementation:

```text
25 passed, 3 warnings in 20.12s
```

Default-volatility diagnostic:

```text
state_identity_max = 4.510281037540e-17
linear_rms         = 2.676197203969e-02
linear_max         = 6.448381212971e-02
quadratic_rms      = 4.389182552998e-02
quadratic_max      = 8.595595430720e-02
rms_ratio          = 1.640082
max_ratio          = 1.332985
```

Audit:
- Gate A still closes selected state identities.
- Gate B fails the user's stricter condition: quadratic residuals are worse
  than linear residuals in both RMS and max at default shock volatility.

Interpretation:
- SGU does not earn the combined target.

Blocked labels:

```text
sgu_quadratic_policy_residual_improvement_passed
sgu_combined_structural_approximation_target_passed
blocked_nonlinear_equilibrium_manifold_residual
```

Next phase justified?
- No.  BayesFilter promotion, derivative certification, compiled parity, and
  HMC are not justified after Gate B fails.  The appropriate next work is a
  new SGU diagnostic/projection plan, not automatic promotion.

### Phase 3: result note and provenance

Plan:
- write a client result note;
- register plan/audit/result provenance;
- update reset memos with phase outcomes.

Execute:
- Added in `/home/chakwong/python`:
  - `docs/plans/sgu-combined-structural-target-result-2026-05-07.md`.
- Updated in BayesFilter:
  - `docs/source_map.yml`;
  - this reset memo.

Interpretation:
- Current SGU status is `state_identity_completion_passed` plus explicit
  quadratic-improvement blocker.  It is not a combined structural
  approximation pass.

### Remaining SGU gap and hypotheses

Remaining gap:
- SGU can complete deterministic support equations pointwise, but the current
  quadratic policy does not improve the full canonical residual relative to
  the linear policy on the same default grid.

Hypotheses to test next:

H1: The default-volatility quadratic failure is dominated by second-order
volatility corrections in Euler/static/control FOC equations, not by state
identity equations.

H2: A pruned SGU comparison needs separate first-order and side-state
completion.  Feeding one completed total state into the quadratic policy with
zero side-state may be the wrong comparison object for a pruned second-order
solver.

H3: A joint state-control projection might reduce full canonical residuals,
but that is a nonlinear projection backend and should get a new label.

H4: BayesFilter may still use SGU as a state-support-complete labeled
approximation after Gate A, but documentation must not call it residual
improved, exact nonlinear, derivative-ready, compiled-ready, or HMC-ready.
