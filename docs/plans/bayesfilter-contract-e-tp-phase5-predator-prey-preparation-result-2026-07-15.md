# Contract E--TP Phase 5 Predator--Prey Preparation Result

metadata_date: 2026-07-15
status: SHORT_PREFIX_VIABLE_APPROXIMATE_REFERENCE_ONLY

## Repaired Defects

Two inherited assumptions were wrong relative to the frozen target.

1. Predator--prey is not a strictly positive-state model in this repository.
   The process and observation noises are additive Gaussian, the model policy is
   `diagnose_negative_after_noise`, and the frozen path contains negative state
   coordinates.  The adapter and plan now use real-plane finite support and do
   not clip or reject negative candidates.
2. The historical LEDH runner transitions its initial cloud before processing
   `y_0`, but the dataset and `PredatorPreySSM.simulate` generate `y_0` from the
   initial state.  Contract E--TP now uses initial-law correction at `t=0` and
   transitions only for positive time indices.  A direct `T=1` identity test
   guards this convention.

## Continuation Mathematics

At `T=2`, the continuation is analytic:

\[
 p_\theta(y_1\mid x_0)
 =N\!\left(y_1;m_\theta(x_0),Q+R\right),
\]

because the RK4 transition mean is deterministic and both state and
observation noises are additive Gaussian.  The full `T=2` oracle is then a
two-dimensional initial Gauss--Hermite integral.

For longer windows, the feature uses a fixed Gaussian-quadrature closure.  It
is an approximate continuation feature, not the exact nonlinear future
likelihood.  The comparison route uses a corrected-time-order fixed-SGQF
ladder: exact Gaussian correction of the initial law by `y_0`, followed by the
existing fixed-SGQF transition/correction recursion on `y_1:`.

## Results

| Rung | Value difference | Worst score relative difference | FD | Chart | Reference status |
| --- | ---: | ---: | --- | --- | --- |
| `T=2`, analytic one-step continuation | `-3.80e-8` | `2.36e-7` | pass | pass | converged semi-analytic oracle |
| `T=5`, Gaussian-closure full-prefix continuation | `-8.09e-4` | `3.08e-4` | pass | pass | approximate corrected-time-order SGQF |

Controlling artifacts:

- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_predator_prey_t2_order5_analytic_lookahead1_result_20260715.json`;
- `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_predator_prey_t5_order5_gaussian_closure_lookahead4_stabilized_result_20260715.json`.

The first wide rectangular-grid `T=2` comparison was invalid: orders 9/11 did
not resolve standard-deviation-two Gaussian likelihoods over a 150-by-45 box
and produced a spurious value near `-43.5`.  It is historical failed-comparator
evidence, not candidate evidence.

The first `T=5` chart used a fixed initial-mean continuation reference and
reported an absolute residual `262144` due to cancellation between enormous
feature values.  Replacing it with the maximum continuation log value over the
teacher plus fixed reference point makes all exponents nonpositive.  The
common parameter-dependent scaling retains its total derivative and feature
span.  The result remained invariant at displayed precision while maximum
feature residual fell to `3.64e-12`.

## Decision

| Decision | Criterion status | Veto status | Uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| admit predator `T=2` mechanism | semi-analytic value/score agreement | all hard checks pass | center-only | retain regression | no full-horizon claim |
| retain `T=5` candidate | close to approximate SGQF; same-scalar FD passes | no engineering veto | no exact nonlinear oracle | refinement then `T=20` | no equivalence/ranking |

No Zhao--Cui production comparator, HMC readiness, full-horizon correctness, or
default status follows from these short-prefix diagnostics.

