# Audit: MacroFinance BayesFilter Switch-over

## Date

2026-05-10

## Scope

This is the Phase B1 read-only audit for switching MacroFinance filtering calls
to BayesFilter.  No MacroFinance files were edited.  The goal is to identify the
smallest safe production pilot, the parity tests that must guard it, the import
strategy, and the rollback boundary.

## Files Inspected

Read-only MacroFinance entry points and tests:

```text
/home/chakwong/MacroFinance/inference/hmc.py
/home/chakwong/MacroFinance/filters/tf_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/filters/tf_svd_kalman.py
/home/chakwong/MacroFinance/tests/test_tf_kalman.py
/home/chakwong/MacroFinance/tests/test_production_lgssm_backend_fixture.py
/home/chakwong/MacroFinance/tests/test_tf_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/tests/test_one_country_analytic_backend_parity.py
/home/chakwong/MacroFinance/tests/test_tf_masked_kalman.py
/home/chakwong/MacroFinance/tests/test_masked_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/tests/helpers_large_scale_lgssm.py
```

Relevant BayesFilter evidence:

```text
tests/test_macrofinance_linear_compat_tf.py
bayesfilter/linear/kalman_qr_tf.py
bayesfilter/linear/kalman_qr_derivatives_tf.py
bayesfilter/linear/kalman_svd_tf.py
```

## Current MacroFinance Filtering Surface

MacroFinance currently routes TensorFlow linear Gaussian likelihoods through
`inference/hmc.py::tf_lgssm_log_likelihood_backend`.  The dispatcher supports:

- `tf_cholesky`;
- `tf_direct_qr`;
- `tf_svd`;
- dense observations and optional masked observations.

The same file also exposes the one-country analytical HMC adapter with
`analytic_backend` values:

- `tf_covariance_analytic`;
- `tf_direct_qr_analytic`.

The switch-over should not start by changing defaults.  It should add a
BayesFilter-backed option next to the existing MacroFinance backends, prove
parity, and only then decide whether a default change is warranted.

## First Switch-over Target

The first target should be:

```text
MacroFinance dense static value path:
inference/hmc.py::tf_lgssm_log_likelihood_backend(
    backend="tf_direct_qr",
    observation_mask=None,
    ...
)
```

The pilot backend should use a new explicit client-side name or feature flag,
for example:

```text
backend="bayesfilter_tf_direct_qr"
```

or a guarded wrapper that is off by default.  The wrapper should construct a
BayesFilter `TFLinearGaussianStateSpace` and call:

```text
bayesfilter.linear.kalman_qr_tf.tf_qr_linear_gaussian_log_likelihood(
    observations,
    model,
    backend="tf_qr",
    jitter=jitter,
)
```

This is the smallest pilot because it avoids masks, HMC derivative plumbing,
SVD branch policy, and client default changes while still exercising the real
MacroFinance likelihood dispatcher.

## Why Not Start With The Larger Paths

Masked QR is already a strong second target, but it adds the static dummy-row
mask convention to the pilot.  BayesFilter and MacroFinance appear aligned on
that convention, and BayesFilter tests already check it, but it is cleaner to
certify dense value dispatch first.

Analytical QR score/Hessian should be the next derivative target after dense
value dispatch.  It touches MAP/HMC behavior and therefore requires exact
parameter ordering, derivative tensor ordering, and posterior-adapter tests.

Linear SVD/eigen value should remain optional.  The current MacroFinance
production-shaped QR test documents that the selected hard surrogate does not
trigger SVD fallback thresholds.  There is no current evidence that SVD value
should be the first production switch-over.

SVD/eigen derivatives should not be part of this switch-over.  BayesFilter's
current SVD/eigen linear backend is value-only by design.

## Required Phase B2 Tests

The dense value pilot should add MacroFinance-side tests that compare the new
BayesFilter-backed option against existing MacroFinance behavior:

```text
/home/chakwong/MacroFinance/tests/test_tf_kalman.py::test_tf_lgssm_backend_selector_direct_qr_matches_default
/home/chakwong/MacroFinance/tests/test_production_lgssm_backend_fixture.py::test_production_shaped_surrogate_exercises_default_and_direct_qr_backends
```

The new assertions should verify:

- existing `tf_direct_qr` value equals the BayesFilter-backed dense QR value
  within the current tolerance;
- compiled dispatch does not retrace for the same static observation shape;
- existing `tf_cholesky` and `tf_direct_qr` defaults remain unchanged;
- no NumPy call is introduced in BayesFilter production filtering code.

After dense value passes, the next MacroFinance subsets should be:

```text
/home/chakwong/MacroFinance/tests/test_tf_masked_kalman.py
/home/chakwong/MacroFinance/tests/test_masked_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/tests/test_tf_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/tests/test_one_country_analytic_backend_parity.py
```

Those later tests guard masked value, masked score/Hessian, dense score/Hessian,
and one-country analytical posterior behavior.

## Import Strategy

The dependency direction should be:

```text
MacroFinance optional adapter imports BayesFilter.
BayesFilter production code does not import MacroFinance.
```

Recommended implementation:

- add a small MacroFinance-side bridge module or local function;
- import BayesFilter only inside the bridge or backend branch;
- use TensorFlow tensors end to end;
- keep MacroFinance economics, provider objects, and posterior adapters in
  MacroFinance;
- avoid adding MacroFinance as a BayesFilter dependency.

This avoids circular imports and preserves BayesFilter as the generic filtering
library.

## Rollback Boundary

Rollback should be trivial:

- keep MacroFinance default backend names unchanged;
- remove or disable `bayesfilter_tf_direct_qr`;
- set `RunConfig(kalman_backend="tf_direct_qr")` or the existing default;
- delete only the client-side bridge and pilot tests if needed.

No BayesFilter package rollback should be needed if the wrapper is isolated in
MacroFinance.

## Veto Diagnostics

Time-varying derivative tensors:

- not needed for the first dense value target;
- the later analytical derivative target should remain on static derivative
  fixtures unless MacroFinance supplies time-indexed derivative tensors.

Circular imports:

- no blocker if MacroFinance imports BayesFilter optionally and BayesFilter does
  not import MacroFinance.

Mask convention:

- no blocker for the first dense target;
- the later masked QR target must preserve the static dummy-row convention.

Jitter:

- no blocker if the wrapper passes the existing MacroFinance `jitter` argument
  directly into BayesFilter.

Derivative tensor ordering:

- no blocker for the first dense value target;
- derivative switch-over must keep the existing parameter-major first- and
  second-order tensor ordering.

SVD regularization policy:

- not part of the first target;
- any later SVD value option must surface implemented-law diagnostics and avoid
  derivative claims.

## Phase B1 Verdict

Phase B1 passes.

The smallest safe switch-over target is the dense static `tf_direct_qr` value
path in MacroFinance's TensorFlow LGSSM likelihood dispatcher.  Phase B2 is
mathematically and architecturally justified as a feature-flagged or explicitly
named MacroFinance-side pilot.

Phase B2 should not be executed automatically in the current BayesFilter
workspace because it requires editing `/home/chakwong/MacroFinance`, which is
outside the writable root.  The next action is a MacroFinance write/permission
decision.
