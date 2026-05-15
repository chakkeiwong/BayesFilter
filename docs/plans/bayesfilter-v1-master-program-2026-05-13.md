# BayesFilter V1 Master Program

## Date

2026-05-13

## Purpose

This is the single coherent master program for BayesFilter V1.  Earlier V1
work produced useful but dispersed plans for API freeze, external
compatibility, benchmarks, HMC gates, nonlinear model suites, SVD sigma-point
scores, and derivative-provider tests.  This document is the controlling
roadmap that keeps those artifacts aligned and prevents future drift.

The V1 objective is not to switch MacroFinance or DSGE over immediately.  The
objective is to make BayesFilter independently testable, mathematically honest,
and ready for later client integration.

## Governing Principle

BayesFilter V1 should be a TensorFlow/TensorFlow Probability filtering package
with explicit structural state-space contracts, tested local correctness, and
clear evidence boundaries.

Every claim must identify one of four statuses:

- `certified`: covered by local tests or benchmark artifacts at the stated
  scope;
- `diagnostic`: useful evidence but not a public readiness claim;
- `deferred`: intentionally left for a later V1 phase;
- `blocked`: not allowed to proceed until a named gate is passed.

## Lane Boundary

This master program owns only the BayesFilter V1 lane:

```text
bayesfilter/*
tests/test_v1_public_api.py
tests/test_linear_*
tests/test_nonlinear_*
tests/test_svd_cut*
tests/test_structural_svd_sigma_point_tf.py
docs/benchmarks/benchmark_bayesfilter_v1_*.py
docs/benchmarks/bayesfilter-v1-*
docs/chapters/ch18_svd_sigma_point.tex
docs/chapters/ch28_nonlinear_ssm_validation.tex
docs/plans/bayesfilter-v1-*.md
docs/source_map.yml entries whose keys begin with bayesfilter_v1_
pytest.ini
```

Protected unless the user explicitly opens another lane:

```text
docs/chapters/ch18b_structural_deterministic_dynamics.tex
docs/plans/bayesfilter-structural-*
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

External projects are compatibility targets, not V1 production dependencies.

## Supporting Artifacts

This master supersedes the following as control documents while preserving them
as evidence:

- `docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md`;
- `docs/plans/bayesfilter-v1-api-freeze-criteria-2026-05-10.md`;
- `docs/plans/bayesfilter-v1-ci-runtime-tier-policy-2026-05-11.md`;
- `docs/plans/bayesfilter-v1-nonlinear-filtering-master-testing-program-2026-05-11.md`;
- `docs/plans/bayesfilter-v1-nonlinear-filtering-remaining-master-program-plan-2026-05-12.md`;
- all `bayesfilter-v1-*result*.md` files as phase evidence.

Future subplans must cite this master and explain which goal/gap they close.

## V1 Goals

### G1. Stable Public API

BayesFilter V1 must have an import-stable public API for:

- linear Gaussian value filters;
- linear QR square-root value filters;
- linear QR score/Hessian filters;
- linear SVD/eigen value filters;
- structural TensorFlow state-space contracts;
- nonlinear SVD sigma-point and SVD-CUT4 value filters;
- smooth-branch SVD sigma-point scores with explicit derivative providers.

Non-goals:

- no MacroFinance production switch-over in V1 stabilization;
- no DSGE production switch-over in V1 stabilization;
- no SGU production filtering claim;
- no public SVD/eigen Hessian claim without a named consumer and gate.

### G2. Linear Filtering Correctness

Linear filtering must retain:

- dense Kalman value correctness;
- masked-observation semantics;
- QR square-root value parity;
- QR score/Hessian parity on local fixtures;
- compiled parity where already tested;
- SVD/eigen value behavior with floor and PSD-projection diagnostics.

Current status:

- mostly `certified` for local tests;
- linear QR HMC evidence remains `diagnostic`, not convergence certification;
- linear SVD/eigen derivatives remain `deferred`.

### G3. Structural State-space Contract

The structural API must keep stochastic, deterministic, innovation, and
observation blocks explicit.  Deterministic completion should be model metadata
and residual diagnostics, not hidden inside a large singular covariance when
that obscures timing or model-law claims.

Current status:

- structural block metadata and affine structural conversion are implemented;
- deterministic-residual diagnostics are used in nonlinear tests;
- DSGE-specific economics stay out of BayesFilter production.

### G4. Nonlinear Value Filters

SVD cubature, SVD-UKF, and SVD-CUT4 value filters must have:

- point-count and polynomial-degree metadata;
- support residuals;
- deterministic-completion residuals;
- floor and PSD-projection diagnostics;
- exact affine Gaussian collapse tests;
- nonlinear approximation diagnostics on Models B-C.

Current status:

- Model A exact Kalman parity is `certified`;
- Models B-C value filters are `certified` as implemented filters;
- dense one-step projection errors for Models B-C are `diagnostic`, not exact
  full nonlinear likelihood certification.

### G5. Nonlinear Analytic Scores And Derivative Validation

First-order scores for SVD cubature, SVD-UKF, and SVD-CUT4 must match centered
finite differences of the same implemented value filters on the accepted
nonlinear model suite.  Every backend/model cell must also state its Hessian
status: implemented, testing-oracle only, deferred pending a named consumer,
or unsupported on the current branch.

Current status:

- affine smooth fixture is `certified`;
- Model B score parity is `certified` on the selected smooth branch;
- Model C score parity is `certified` only on a nondegenerate phase-state
  testing variant;
- default Model C with zero phase variance is `certified` on the opt-in
  Chapter 18b structural fixed-support score branch for SVD cubature,
  SVD-UKF, and SVD-CUT4, at the local finite-difference scope recorded in
  `docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-result-2026-05-13.md`;
- the old collapsed smooth no-active-floor score path still correctly blocks
  default Model C through the active-floor gate unless the structural
  fixed-support contract is explicitly requested;
- a consolidated derivative-validation matrix over Models A-C is still
  missing.  The existing tests certify many cells, but the master program now
  requires a single auditable matrix before nonlinear HMC, GPU/XLA, or external
  integration work.

### G6. Hessian Policy

Hessian work is not a default V1 requirement.  It can proceed only when one of
the following consumers is named:

- Newton or trust-region optimization;
- Laplace approximation;
- Riemannian or curvature-aware HMC;
- observed-information diagnostics.

Current status:

- linear QR Hessian is `certified` on local fixtures;
- nonlinear SVD Hessian is `deferred`;
- testing-only raw autodiff oracles remain outside production.

### G7. GPU/XLA Policy

GPU/XLA evidence must be opt-in and escalated.  Non-escalated GPU failures are
sandbox evidence only.

Current status:

- small GPU-visible and tiny XLA-visible artifacts are `diagnostic`;
- no broad GPU speedup claim is certified;
- nonlinear CUT4 GPU/XLA scaling is `deferred` until score branch boxes are
  stable.

### G8. HMC Readiness

HMC readiness is target-specific.  Filter existence or score existence is not
enough.

Minimum gate for an HMC target:

- finite value and score on the target parameter box;
- branch-frequency diagnostics;
- compiled parity if the target will run compiled;
- sampler smoke with explicit chain diagnostics;
- convergence diagnostics only when explicitly claimed.

Current status:

- first linear QR HMC target has `diagnostic` smoke evidence;
- nonlinear HMC is `blocked` until wider nonlinear score branch diagnostics
  identify a stable target parameter box.  The default Model C structural
  score decision is now made, but it still needs branch-box evidence before it
  can be considered as an HMC target.

### G9. External Compatibility

MacroFinance and DSGE are external compatibility targets.  V1 should not make
either project a production dependency.

Current status:

- MacroFinance live checks are optional;
- DSGE inventory is read-only;
- SGU remains blocked as a production filtering target;
- actual client switch-over is `deferred` until V1 local gates pass.

## Completed Gates

C1. Chapter 18b structural sigma-point score implementation for default Model C.

Status:
- `certified` at local finite-difference scope.

Evidence:
- commit `e13be5a`, `Close structural Model C score branch`;
- `docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-2026-05-13.md`;
- `docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-audit-2026-05-13.md`;
- `docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-result-2026-05-13.md`;
- Chapter 18 label `sec:bf-svd-sp-structural-fixed-support-score`;
- full default CPU validation: `204 passed, 5 skipped, 2 warnings`.

Interpretation:
- the correct default Model C score path is the Chapter 18b structural
  sigma-point recursion, not a fake phase nugget and not a separate numerical
  fixed-null theory;
- sigma points are placed on the pre-transition structural variable
  \(A_t=(x_{t-1},\varepsilon_t)\), deterministic coordinates are completed
  pointwise by \(F_\theta\), and the score differentiates the implemented
  structural sigma-point likelihood;
- this closes the former R1 gap but does not certify nonlinear HMC, nonlinear
  Hessians, GPU/XLA scaling, or external client switch-over.

Tooling lesson:
- MathDevMCP was useful for Chapter 18b lookup and score-sign checks, but
  label/code comparison failed internally on the new local label.  Future
  code-document gates should use MathDevMCP when available but require manual
  audit plus finite-difference or reference tests as the certifying evidence.

## Current Remaining Gaps

R2. Nonlinear derivative-validation matrix.

Hypothesis:
The accepted nonlinear testing suite can support a complete derivative-status
matrix for SVD cubature, SVD-UKF, and SVD-CUT4 across Models A-C.  The matrix
should certify implemented first-order scores where tests already exist,
identify any missing score cells, and record Hessian status without forcing a
production Hessian implementation.

Test:
build and run a derivative-validation matrix covering value, score, branch,
compiled/eager, and Hessian-status evidence for Models A-C and the three SVD
sigma-point backends.  The matrix must distinguish smooth branches,
structural fixed-support branches, active-floor blockers, testing-only
autodiff oracles, and production exports.

R3. Wider nonlinear score branch diagnostics.

Hypothesis:
Model B and default Model C have practical parameter boxes where the selected
score branches remain finite across SVD cubature, SVD-UKF, and SVD-CUT4.
Smooth-phase Model C should remain as a comparison/control case, but default
Model C must be tested through `allow_fixed_null_support=True`.

Test:
run CPU branch grids over selected boxes and report ok fraction, active-floor
count, weak-gap count, structural-null diagnostics, deterministic residuals,
finite value/score status, and backend-specific failure labels.

R4. Nonlinear benchmark refresh with score-branch metadata.

Hypothesis:
The existing nonlinear benchmark suite can expose value accuracy,
score-branch stability, deterministic residuals, and support diagnostics in
one coherent artifact without claiming exact nonlinear likelihood
certification.

Test:
refresh benchmark outputs for Models B-C after R3, adding backend, branch,
point-count, polynomial-degree, deterministic residual, structural-null, and
finite-score metadata.

R5. Nonlinear HMC target selection and first smoke.

Hypothesis:
Model B is the first viable nonlinear HMC candidate because it is smooth and
score-certified.  Default Model C can become a second candidate only if R3
shows a stable structural fixed-support branch box.

Test:
write a target-specific readiness plan after R2/R3/R4; run only a tiny CPU smoke
when value, score, branch, and compiled-parity gates are satisfied.

R6. Nonlinear Hessian consumer assessment.

Hypothesis:
V1 can proceed with score-first nonlinear workflows and defer nonlinear
Hessians unless optimization, Laplace approximation, Riemannian HMC, or
observed-information diagnostics becomes a concrete consumer.

Test:
record the named consumer, required mathematical branch, expected tensor
shapes, and an implementation/test plan before any nonlinear Hessian code is
started.

R7. GPU/XLA point-axis scaling.

Hypothesis:
CUT4's larger point count becomes less costly under batched point-axis
vectorization on GPU/XLA for moderate shapes.

Test:
run escalated GPU-visible and XLA-visible benchmarks only after R3 defines
stable nonlinear score/value boxes; non-escalated GPU failures are sandbox
evidence only.

R8. Exact nonlinear reference strengthening.

Hypothesis:
For short horizons, dense quadrature or high-particle seeded SMC can provide
stronger reference evidence for Models B-C without becoming production
dependencies.

Test:
add optional reference artifacts that clearly distinguish exact, dense
projection, and Monte Carlo evidence.

R9. External client integration.

Hypothesis:
MacroFinance/DSGE switch-over should wait until V1 has stable local API,
benchmark, branch, optional GPU/HMC evidence, and optional live compatibility
evidence.

Test:
write a separate future integration plan; do not modify external source in the
V1 stabilization lane.

## Safe Execution Ladder

Each phase must follow the cycle:
plan for the phase, execute, test, audit, tidy up, update the V1 reset memo,
then decide whether the next phase is justified.  Continue automatically only
when the primary gate passes and no veto diagnostic fires.

### Phase P0: Master Reconciliation And Lane Check

Purpose:
- keep this master aligned with completed results before starting new work.

Entry condition:
- a phase result has landed or a new user decision changes priority.

Required actions:
- update the master status table and remaining gaps;
- verify `git status --short` and identify out-of-lane dirty files;
- update only V1-lane reset memo entries when a reset memo update is needed.

Primary gate:
- master names exactly one next active phase and no completed phase remains in
  "Current Remaining Gaps."

Veto diagnostics:
- out-of-lane files staged;
- shared monograph reset memo staged from this lane;
- master contradicts latest result artifacts.

Subplan status:
- this master section is the controlling P0 plan.

### Phase P1: Nonlinear Derivative-Validation Matrix

Purpose:
- close R2 by making derivative status explicit for every accepted nonlinear
  model/backend cell before branch sweeps, HMC, GPU/XLA, or external
  integration.

Entry condition:
- C1 remains green on focused and full CPU tests.

Required actions:
- execute the existing R2 derivative-validation matrix subplan:
  `docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-plan-2026-05-14.md`;
- enumerate Models A-C and backends SVD cubature, SVD-UKF, and SVD-CUT4;
- for each cell, record value status, score status, score branch, derivative
  provider, finite-difference target, branch diagnostics, compiled/eager
  parity if available, and Hessian status;
- run or cite the focused tests that certify existing score cells;
- add missing lightweight tests only when a matrix cell lacks evidence and the
  needed fixture already exists;
- keep nonlinear Hessian implementation deferred unless the matrix identifies
  an already implemented Hessian cell or a named consumer.

Primary gate:
- the matrix has no unknown score-status cells for Models A-C across the three
  backends, and Hessian status is explicit for every cell.

Veto diagnostics:
- a score cell is marked certified without finite-difference, exact affine, or
  documented oracle evidence;
- default Model C is certified without `allow_fixed_null_support=True`;
- a testing-only autodiff oracle is exposed or described as production API;
- nonlinear Hessian implementation starts without a named consumer;
- production NumPy dependency is introduced.

Artifacts:
- R2 derivative-validation matrix plan, audit, and result files under
  `docs/plans/bayesfilter-v1-*`;
- updated V1 reset memo;
- tests only if needed to close explicit matrix holes.

### Phase P2: Wider Nonlinear Score Branch Diagnostics

Purpose:
- close R3 by finding stable score branch boxes for Model B and default Model C.

Entry condition:
- P1 derivative-validation matrix passes and C1 remains green.

Required actions:
- execute the existing R3 branch-diagnostics subplan:
  `docs/plans/bayesfilter-v1-p2-branch-diagnostics-plan-2026-05-14.md`;
- design CPU grids for Model B, smooth-phase Model C, and default Model C with
  `allow_fixed_null_support=True`;
- run branch summaries for SVD cubature, SVD-UKF, and SVD-CUT4;
- record ok fractions, active floors, weak active gaps, nonfinite failures,
  structural-null counts, fixed-null derivative residuals, structural-null
  covariance residuals, support residuals, deterministic residuals, and score
  finiteness.

Primary gate:
- at least one practical Model B box passes for all selected backends, and
  default Model C has a clearly reported pass/fail structural branch box.

Veto diagnostics:
- branch summaries hide failure labels;
- default Model C is tested without the structural fixed-support option;
- active floors or weak gaps are treated as success;
- production NumPy dependency is introduced.

Artifacts:
- R3 plan, audit, and result files under `docs/plans/bayesfilter-v1-*`;
- updated V1 reset memo;
- tests or benchmark diagnostics if new code is required.

### Phase P3: Nonlinear Benchmark Refresh With Score Metadata

Purpose:
- close R4 by turning derivative and branch evidence into benchmark-style
  artifacts.

Entry condition:
- P1 and P2 pass or record a narrowed benchmark scope.

Required actions:
- refresh Model B-C benchmark outputs with score branch metadata;
- separate implemented-filter value evidence from exact nonlinear likelihood
  claims;
- include backend, point count, polynomial degree, branch label,
  deterministic residual, structural-null metadata, finite-score status, and
  failure labels.

Primary gate:
- benchmark artifacts reproduce from documented commands and make no exactness
  claim beyond their reference type.

Veto diagnostics:
- diagnostic projection errors are described as exact likelihood errors;
- missing branch metadata for any nonlinear score row;
- benchmark requires GPU or external projects by default.

Artifacts:
- benchmark command/result artifact under `docs/benchmarks` or `docs/plans`;
- updated Chapter 28 text if claims change.

### Phase P4: Nonlinear HMC Target Selection And Tiny CPU Smoke

Purpose:
- close R5 conservatively, with Model B as the default first target.

Entry condition:
- P1 passes, P2 identifies a stable target box, and P3 provides target
  metadata.

Required actions:
- create a target-specific HMC readiness subplan;
- choose Model B unless P1 gives a stronger reason to choose another target;
- verify finite value and score on the target box;
- verify branch stability and compiled parity for the intended execution mode;
- run only a tiny CPU HMC smoke when readiness gates pass.

Primary gate:
- HMC smoke produces finite chains and explicit diagnostics at the tiny-scope
  level, without claiming convergence unless convergence diagnostics were run.

Veto diagnostics:
- HMC starts before finite score and branch gates pass;
- default Model C is selected before its structural branch box is stable;
- smoke output is promoted to convergence certification;
- GPU is used without escalated sandbox permissions.

Artifacts:
- HMC readiness plan, audit, and result files;
- opt-in test marker or environment gate if code/tests are added.

### Phase P5: Nonlinear Hessian Consumer Assessment

Purpose:
- close R6 by deciding whether nonlinear Hessians are needed in V1 production,
  after the derivative-validation matrix has recorded Hessian status.

Entry condition:
- a concrete consumer is named, or P4 shows score-only workflows are
  insufficient.

Required actions:
- document the consumer and why scores are not enough;
- identify the derivative branch and covariance-factor contract;
- decide whether Hessians should be analytic, autodiff testing-only, or
  deferred;
- write an implementation plan only if the consumer justifies it.

Primary gate:
- either a named consumer justifies a Hessian implementation plan, or Hessian
  work remains explicitly deferred.

Veto diagnostics:
- Hessian code starts before a consumer is named;
- testing-only autodiff is exposed as production API;
- SVD/eigen Hessian claims ignore branch gaps or structural null contracts.

Artifacts:
- Hessian assessment plan/result, and implementation subplan only if needed.

### Phase P6: Optional GPU/XLA Scaling Diagnostics

Purpose:
- test whether point-axis vectorization improves CUT4 cost on real GPU/XLA.

Entry condition:
- P2 identifies stable nonlinear boxes and P3 gives benchmark commands.

Required actions:
- run GPU/CUDA probes and GPU/XLA benchmarks only with escalated sandbox
  permissions;
- compare CPU eager, CPU graph, GPU eager, and XLA where feasible;
- record shape, point count, backend, warmup policy, and device visibility.

Primary gate:
- diagnostic artifact states whether GPU/XLA improves the tested shapes and
  clearly limits the claim to those shapes.

Veto diagnostics:
- non-escalated GPU failure is treated as real CUDA failure;
- broad speedup claim is made from one tiny artifact;
- benchmark changes production behavior.

Artifacts:
- GPU/XLA diagnostic result under `docs/benchmarks` or `docs/plans`.

### Phase P7: Optional Exact Nonlinear Reference Strengthening

Purpose:
- close R8 by adding stronger references for short nonlinear benchmarks.

Entry condition:
- P3 identifies where current references are too weak for the claim being
  made.

Required actions:
- choose dense quadrature or seeded high-particle SMC as an optional reference;
- document approximation/error status;
- keep reference code out of production dependencies unless explicitly
  approved.

Primary gate:
- reference artifact improves claim clarity for Models B-C without turning
  Monte Carlo diagnostics into exact certification.

Veto diagnostics:
- stochastic reference lacks seed/reproducibility metadata;
- dense projection is mislabeled as exact full nonlinear likelihood;
- reference dependency leaks into production imports.

Artifacts:
- optional reference tests/results and Chapter 28 claim update if warranted.

### Phase P8: External Client Integration Plan

Purpose:
- prepare, but not execute, future MacroFinance/DSGE switch-over.

Entry condition:
- focused/full CPU tests pass, P1/P2/P3 nonlinear claims are current, and any
  desired optional GPU/HMC evidence is labeled at its true scope.

Required actions:
- write a separate integration plan;
- keep MacroFinance and DSGE as external compatibility targets;
- specify test-only adapters before any production coupling;
- define rollback and ownership boundaries.

Primary gate:
- integration plan can be reviewed without requiring changes to external
  source trees.

Veto diagnostics:
- V1 imports MacroFinance or DSGE as a production dependency;
- external source is modified from the V1 lane;
- SGU economics are promoted without the structural/DSGE lane being reopened.

Artifacts:
- future integration plan under `docs/plans`, with explicit external-owner
  gates.

## Phase/Subplan Matrix

| Phase | Gap | Status | Current subplan |
| --- | --- | --- | --- |
| P0 | Master reconciliation | active governance | this master |
| P1 | R2 derivative-validation matrix | next | `docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-plan-2026-05-14.md` |
| P2 | R3 branch diagnostics | pending after P1 | `docs/plans/bayesfilter-v1-p2-branch-diagnostics-plan-2026-05-14.md` |
| P3 | R4 benchmark refresh | pending after P1/P2 | `docs/plans/bayesfilter-v1-p3-benchmark-refresh-plan-2026-05-14.md` |
| P4 | R5 nonlinear HMC | blocked until P1/P2/P3 | `docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-plan-2026-05-14.md` |
| P5 | R6 Hessian consumer assessment | deferred | `docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-plan-2026-05-14.md` |
| P6 | R7 GPU/XLA scaling | deferred | `docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-plan-2026-05-14.md` |
| P7 | R8 exact references | optional/deferred | `docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-plan-2026-05-14.md` |
| P8 | R9 external integration | deferred | `docs/plans/bayesfilter-v1-p8-external-integration-plan-2026-05-14.md` |

## Anti-drift Rules

1. Do not start GPU/XLA, HMC, Hessian, or external-client work before its
   predecessor gate passes.
2. Do not promote a diagnostic artifact to a certified claim.
3. Do not edit MacroFinance or DSGE from this lane.
4. Do not stage the shared monograph reset memo from this lane.
5. Do not add production NumPy dependencies.
6. Do not hide structural deterministic coordinates inside numerical
   regularization when the issue is a model-law or derivative-branch question.
7. Every future subplan must name the master-program gap it closes.

## Default Validation Commands

Fast local:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Focused V1 regression:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Full default CPU:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Extended CPU branch diagnostics:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

## Master Program Decision Rule

BayesFilter V1 is ready for a client-integration planning phase when:

- the public API import gate passes;
- focused and full CPU suites pass;
- nonlinear value and score claims are documented at their true scope;
- Chapter 18b structural sigma-point score handling for deterministic
  completion is either implemented or explicitly deferred away from the first
  client target;
- GPU/XLA and HMC claims are either backed by target artifacts or labeled
  deferred;
- source-map and reset-memo provenance are current;
- no out-of-lane files are staged.
