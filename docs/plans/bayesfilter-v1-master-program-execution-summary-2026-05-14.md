# BayesFilter V1 Master Program Execution Summary

## Date

2026-05-14

## Governing Document

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

Older V1 plans were treated as evidence only.  The master program controlled
phase order, gates, and veto diagnostics.

## Executed Phases

| Phase | Result | Artifact |
| --- | --- | --- |
| P1 derivative-validation matrix | passed | `docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-result-2026-05-14.md` |
| P2 branch diagnostics | passed | `docs/plans/bayesfilter-v1-p2-branch-diagnostics-result-2026-05-14.md` |
| P3 benchmark refresh | passed | `docs/plans/bayesfilter-v1-p3-benchmark-refresh-result-2026-05-14.md` |
| P4 nonlinear HMC target | passed at tiny CPU-smoke scope | `docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-result-2026-05-14.md` |
| P5 Hessian consumer assessment | passed by explicit deferral | `docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-result-2026-05-14.md` |
| P6 GPU/XLA scaling diagnostic | passed as scoped diagnostic | `docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-result-2026-05-14.md` |
| P7 exact reference strengthening | passed by explicit deferral | `docs/plans/bayesfilter-v1-p7-exact-reference-strengthening-result-2026-05-14.md` |
| P8 external integration planning | passed as reviewable plan | `docs/plans/bayesfilter-v1-p8-external-integration-result-2026-05-14.md` |

## Main Results

Nonlinear SVD sigma-point value and score evidence is now organized around the
first-rung Models A-C:

- Model A: exact affine Gaussian Kalman reference;
- Model B: smooth nonlinear accumulation, analytic scores certified on stable
  branch boxes;
- Model C: default structural fixed-support score path with
  `allow_fixed_null_support=True`, preserving the Chapter 18b structural
  deterministic-completion contract.

The benchmark artifacts now expose value and score branch metadata, point
counts, structural-null diagnostics, failure labels, and reference scope.  They
do not claim exact full nonlinear likelihoods for Models B-C.

P4 selected Model B with SVD-CUT4 analytic score as the first nonlinear HMC
target and ran an opt-in tiny CPU smoke.  The smoke produced finite samples and
finite branch diagnostics but does not claim convergence.

P5 kept nonlinear Hessians deferred because no current V1 consumer was named.
The testing-only SVD-CUT4 autodiff oracle remains in the testing namespace.

P6 confirmed that GPU-visible TensorFlow and XLA can execute the tested Model B
SVD-CUT4 shape on this machine.  The tiny shape does not support a broad GPU
speedup claim.

P7 kept exact nonlinear references deferred because no current V1 claim needs a
stronger reference than exact Model A and dense one-step projection diagnostics
for Models B-C.

P8 prepared external integration without editing MacroFinance or DSGE.  The
first external target should be MacroFinance linear QR compatibility through a
test-only or optional-live bridge.  DSGE remains future/test-only; SGU remains
blocked in this lane.

## Final Validation

Fast public API:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Result:

```text
2 passed, 2 warnings
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

Result:

```text
73 passed, 2 skipped, 2 warnings
```

Full default CPU:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Result:

```text
204 passed, 8 skipped, 2 warnings
```

The warnings are TensorFlow Probability `distutils.version` deprecation
warnings.

## Remaining Gaps

G1. Nonlinear HMC convergence is not certified.

Hypothesis:
- Model B SVD-CUT4 can become a useful nonlinear HMC target after longer
  opt-in chains, chain diagnostics, and posterior recovery checks.

Test:
- run an opt-in CPU HMC ladder with multiple seeds, R-hat, ESS, divergence
  counts, and true-parameter coverage.

G2. Nonlinear Hessians remain deferred.

Hypothesis:
- V1 does not need production nonlinear Hessians unless a named consumer
  appears, such as Laplace approximation, Newton optimization, observed
  information, or Riemannian HMC.

Test:
- before implementation, require a consumer-specific design with tensor shape,
  memory, branch, and second-derivative provider contracts.

G3. GPU speedup is not established.

Hypothesis:
- GPU/XLA may help only when the point axis is combined with larger horizons,
  larger dimensions, batched parameter points, or batched independent filters.

Test:
- extend P6 with a shape/batch ladder and compare CPU graph, CPU XLA, GPU
  graph, and GPU XLA under escalated GPU permissions.

G4. Exact full nonlinear likelihood references for Models B-C remain blocked.

Hypothesis:
- dense multi-step quadrature or seeded high-particle SMC will improve claim
  clarity only if a future public claim needs exact/reference likelihood
  evidence.

Test:
- open a reference-only phase with seeds, particle counts or quadrature nodes,
  error labels, and strict no-production-dependency policy.

G5. External client switch-over is not started.

Hypothesis:
- MacroFinance linear QR compatibility can be integrated safely after optional
  live checks pass on a clean external checkout and client owners open a
  separate switch-over branch.

Test:
- run optional live MacroFinance compatibility with recorded external commit
  hash; keep failures external unless reproduced on BayesFilter-local fixtures.

## Recommended Next Phases

1. Prepare a V1 release-candidate checkpoint from the current passing CPU
   state.
2. Run optional live MacroFinance linear compatibility on a clean external
   checkout.
3. Expand nonlinear HMC from tiny smoke to an opt-in diagnostic ladder.
4. Add a batched/larger-shape GPU/XLA diagnostic before making any performance
   statement.
5. Keep nonlinear Hessian and exact nonlinear reference work deferred until a
   concrete consumer or claim requires them.
