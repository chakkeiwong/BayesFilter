# Direct-Factor SR-UKF Model Parity Report

Date: 2026-08-16
Scope: non-SSL-LSTM BayesFilter models only
Route status: direct-factor is the default for `TFFactorSRUKFModel`; principal-root is historical/reference-only

## Result

The repository-default direct-factor route has been run end-to-end on the three canonical
structural UKF fixtures. Model A is exactly equal to the principal-root
reference at float64 precision. Models B and C are finite and numerically
close, but are not mathematically required to be identical: the principal-root
reference recomputes a symmetric eigensquare-root, while the candidate carries
a lower factor. Nonlinear sigma-point transforms depend on that factor
orientation.

| model | horizon | old value | direct-factor value | abs value diff | relative value diff | max abs score diff | max relative score diff | min QR pivot | min downdate margin | classification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Model A affine | 5 | 0.6910366944078346 | 0.6910366944078346 | 0.0000000000000000 | 0.0000000000000000 | 2.22e-16 | 1.52e-16 | 0.1272250974356028 | 0.0161862254174986 | exact parity |
| Model B nonlinear accumulation | 3 | -1.5597940252733968 | -1.5596249662849013 | 1.690589884955e-4 | 1.08385457409e-4 | 8.42348382962e-4 | 7.40742591548e-4 | 0.0725473646194082 | 0.0049723419685073 | orientation-sensitive close |
| Model C nonlinear growth | 2 | -5.532858230180173 | -5.562878111183707 | 3.00198810035e-2 | 5.42574556488e-3 | 2.47199160193e-2 | 1.68559082723e-2 | 0.3518659467055995 | 0.0782984928697369 | orientation-sensitive close, larger gap |

Old scores and direct-factor scores, in parameter order, were:

```text
Model A old [ 1.4627591028507947]
        new [ 1.4627591028507945]

Model B old [-1.1376542966555383, -5.319319131948499, -1.1045691312938923]
        new [-1.1368119479340764, -5.319492587069384, -1.1041735815480669]

Model C old [-0.06162141082091739, -0.03920863611252258, -1.466499320288874]
        new [-0.05856399280274458, -0.03925711321984708, -1.4417801422724832]
```

All direct-factor values/scores were finite, with strictly positive QR pivots and
downdate margins. The route uses `float64`, direct QR stacks, and sequential
lower-factor downdates; no covariance-to-principal-root decomposition is used
in the recursive candidate path.

An independent centered finite-difference check of the direct-factor value
program at `h=1e-5` gave maximum score discrepancies of `4.1e-12` (A),
`8.9e-10` (B), and `3.5e-9` (C). The score gaps in the table therefore reflect
two factor-orientation-specific UKF programs, not a mismatch between the
direct-factor value and its direct-factor derivative.

## Commands and evidence

The focused default-route and parity checks were run with
`TF_FORCE_GPU_ALLOW_GROWTH=true`. TensorFlow reported no CUDA-capable device,
so these results are CPU diagnostics, not GPU evidence.

```text
pytest -q tests/test_factor_srukf_model_parity.py
2 passed

pytest -q tests/test_stack_qr_tf.py tests/test_lower_rank_downdate_tf.py \
  tests/test_factor_srukf_tf.py tests/test_factor_srukf_route_guard.py \
  tests/test_factor_srukf_compat.py tests/test_srukf_factor_tf.py \
  tests/test_actual_sv_srukf_tf.py
28 passed

pytest -q tests/test_nonlinear_ssm_phase2_value_paths.py \
  tests/test_nonlinear_xla_parity_tf.py tests/test_actual_sv_srukf_tf.py \
  tests/test_srukf_factor_tf.py tests/test_factor_srukf_model_parity.py
45 passed

CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true pytest -q \
  tests/highdim/test_p39_sv_mixture_cut4.py \
  tests/highdim/test_p40_sv_kalman_cut4_zhaocui.py \
  tests/highdim/test_p44_predator_prey_diagnostic.py \
  tests/highdim/test_p44_spatial_sir_diagnostic.py \
  tests/highdim/test_p47_predator_prey_filtering.py \
  tests/highdim/test_p47_spatial_sir_filtering.py \
  tests/highdim/test_p50_spatial_sir_predator_prey_ladder.py \
  tests/highdim/test_p50_sv_generalized_sv_ladder.py \
  tests/test_ksc_gaussian_sum_ukf_neutra_target.py \
  tests/test_ksc_ukf_neutra_target.py
73 passed
```

A separate broad legacy nonlinear bundle was also probed and reported three
pre-existing failures outside the direct-factor route: the default principal
root Model C fixture activates its strict placement floor, one dense-projection
test references an undefined local (`NameError`), and one historical SVD-UKF
score finite-difference assertion misses its existing tolerance. These do not
invalidate the focused default-route result, but they prevent a claim that every
legacy nonlinear test is currently green.

The corrected lower-factor sigma-point convention is in
`bayesfilter/nonlinear/factor_srukf_tf.py`: if `P=L L'`, row offsets are
formed as `offset @ L'`. The model-level parity test is
`tests/test_factor_srukf_model_parity.py`.

## Coverage ledger

### Direct-factor default route tested

- Model A affine Gaussian structural fixture: current additive Gaussian
  observation, nonzero process and observation noise, parameterized observation
  offset/score.
- Model B nonlinear accumulation fixture: nonlinear transition with a
  deterministic accumulation component and additive Gaussian observation.
- Model C nonlinear growth fixture: nonlinear growth/phase transition,
  parameterized process and observation scales, and initial variance score.

### Existing baseline suites run, but not direct-factor adapted

- Historical factor SR-UKF and actual-SV SR-UKF adapters. These passed their
  existing tests, but their APIs and contracts were not silently replaced by
  the new route.
- Existing KSC mixture, SV/CUT4, predator-prey, and spatial-SIR diagnostic and
  ladder suites: 73 tests passed. These are baseline route evidence only.
- Structural nonlinear value/score/XLA suites for the existing principal-root,
  SVD, cubature, and CUT4 paths. These establish baseline health, not
  direct-factor parity for every model.

### Not comparable to the current direct-factor API

- Austria/SIR and spatial-SIR routes: structural/author-specific transition and
  observation contracts, often with initial-observation or lagged-event
  semantics that are not represented by the current current-state additive
  Gaussian observation API.
- Predator-prey routes: author/fixed-variant and proposal-density contracts,
  not the single additive Gaussian observation-factor program required by the
  direct-factor adapter.
- KSC/transformed stochastic-volatility Gaussian-sum routes: mixture
  observation laws with multiple components, not one additive Gaussian
  observation factor.
- Generalized/actual-SV high-dimensional routes: independent-panel or
  augmented-noise/manual-score programs with different target and parameter
  contracts. The existing actual-SV tests remain historical/route-specific
  evidence.
- SSL-LSTM models: explicitly excluded by user request.

These entries are blocked from apples-to-apples comparison, not failures of the
models or evidence that the direct-factor algorithm is invalid. Adapting them
would require a separate contract-preserving adapter and a new parity plan.

## Interpretation and nonclaims

The answer to “same or very similar values and scores?” is:

- Model A: yes, exact to float64 roundoff for both value and score.
- Model B: yes for this benign fixture, with sub-`1.1e-4` relative value and
  sub-`7.5e-4` relative score differences.
- Model C: finite and directionally similar, but only moderately close here,
  with `5.4e-3` relative value and `1.7e-2` relative score differences.

The backend promotion does not certify nonlinear equivalence, promote the
route to production, or support MacroFinance integration,
NeuTra/HMC, posterior, convergence, or scientific claims.
