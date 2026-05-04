# Phase 8 plan: MacroFinance adapter pilot

## Status and provenance

Date: 2026-05-04

Requested action:

1. Pull the updated `/home/chakwong/MacroFinance` checkout.
2. Make a concrete plan for Phase 8 of
   `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`.

Pull result:

- `/home/chakwong/MacroFinance` is clean on `main`.
- SSH authentication to GitHub succeeded.
- Remote HEAD before fast-forward was verified by `git ls-remote`:
  `e23c31e531bdd7cb286af33c4caf154687cee634`.
- Local branch was fast-forwarded:
  `22f496e Record analytic HMC validation pilots` ->
  `e23c31e Document one-country HMC remaining test gates`.
- `HEAD`, `origin/main`, and `origin/HEAD` now point to `e23c31e`.
- `origin` is `https://github.com/chakkeiwong/MacroFinance`.
- The HTTPS pull path failed earlier because GitHub credentials were unavailable
  in the noninteractive environment, but the SSH fetch/fast-forward succeeded.

Interpretation:

- This plan is grounded in the updated MacroFinance checkout at `e23c31e`.
- The pull changed the one-country analytic HMC surface materially enough that
  the Phase 8 plan has been refreshed after inspection.

## Phase 8 objective

Build the smallest useful BayesFilter-to-MacroFinance adapter pilot without
copying MacroFinance financial-model construction or differentiated Kalman
implementations.

The pilot should prove that BayesFilter can consume a MacroFinance LGSSM fixture
through a narrow adapter boundary, evaluate the same likelihood, and preserve
clear ownership:

- MacroFinance owns AFNS, cross-currency, and production financial model
  construction.
- MacroFinance remains the reference for existing analytical score, Hessian,
  QR/square-root, SVD, TensorFlow, and masked derivative implementations.
- BayesFilter owns generic state-space contracts, reference value filters,
  structural partition metadata, approximation labels, and cross-project adapter
  contracts.

## Local MacroFinance audit summary

The current local checkout exposes the following reusable surfaces:

- `domain/types.py`
  - `LinearGaussianStateSpace` has the same core array fields as the
    BayesFilter exact LGSSM reference, but no BayesFilter `StatePartition`.
  - `LinearGaussianStateSpaceDerivatives` stores first- and second-order
    derivative tensors for every LGSSM block.
  - `RunConfig` carries `dt`, `jitter`, and TensorFlow backend selection.
  - `HMCConfig` now records chain count, target-XLA choice, full-chain-XLA
    choice, and optional latent initial scale.
- `inference/posterior_adapter.py`
  - `DifferentiableStateSpaceProvider` is a useful structural protocol:
    `parameter_names`, `build_state_space`, and
    `build_state_space_with_derivatives`.
  - `PosteriorAdapter` is the HMC-side protocol.
- `one_country_derivative_provider.py`
  - `OneCountryAFNSDerivativeProvider` is the best Phase 8 pilot target.
  - It has four free parameters:
    `theta_0`, `theta_1`, `theta_2`, `log_measurement_error_std`.
  - It builds a restricted AFNS LGSSM and analytical derivative tensors while
    keeping richer AFNS parameters fixed.
- `one_country_tf_derivative_provider.py`
  - `OneCountryTFDerivativeProvider` now supports `initial_mean_policy` values
    `zero`, `theta`, and `fixed`.
  - Its `order` argument is now meaningful for first- versus second-order
    derivative construction.
- `filters/tf_differentiated_kalman.py`
  - `tf_differentiated_kalman_loglik_grad` is a graph-native value-plus-score
    workload for MAP gradient steps and TFP HMC leapfrog.
  - `tf_differentiated_kalman_loglik_grad_hessian_graph` preserves the
    second-order workload for local-information and mass-matrix diagnostics.
- `inference/hmc.py`
  - `OneCountryAnalyticThetaNoisePosteriorAdapter` exposes
    `log_prob_and_grad`, `log_prob_grad_hessian`, custom-gradient score
    plumbing, `negative_log_prob_and_gradient`, and
    `negative_log_prob_hessian`.
  - Value/gradient paths no longer call Hessian work in the one-country
    analytic adapter.
  - The HMC target path has an analytic chain-batched custom-gradient route for
    XLA-friendly TFP target evaluation.
  - These are HMC-readiness references, not BayesFilter code to copy.
- `perf_one_country_analytic_hmc_validation.py`
  - New JSONL diagnostics cover chain policy, theta2/curvature geometry,
    finite-sample curvature, positive-curvature prior, profile grids, and HMC
    summaries.
- New MacroFinance plans and reset memos document one-country HMC remaining
  gates, chain policy, theta2 recovery, and TFP analytic filter speed closure.
- `large_scale_lgssm_derivative_provider.py`
  - `LargeScaleLGSSMDerivativeProvider` and
    `LargeScaleLGSSMPosteriorAdapter` are useful second targets after the
    one-country adapter passes.
  - It includes parameter units and masked backend dispatch; this is too broad
    for the first Phase 8 implementation slice.
- `cross_currency_structural_derivative_provider.py` and
  `production_cross_currency_derivative_provider.py`
  - These are rich structural finance providers with coverage matrices,
    finite-difference oracles, masking, and production-readiness blockers.
  - They should remain future adapter targets, not the first BayesFilter pilot.
- Existing tests already cover:
  - one-country analytic adapter value/score/Hessian parity against NumPy and
    TensorFlow paths;
  - value/gradient paths not calling Hessian work;
  - compiled one-country analytic adapter log-prob retracing behavior;
  - target-XLA one-country HMC smoke using the analytic gradient filter;
  - theta2 geometry diagnostics and HMC chain-policy scaffolding;
  - cross-currency dense backend parity and finite-difference spot checks;
  - large-scale LGSSM masked and SVD posterior adapter behavior;
  - QR/square-root derivative identities and TensorFlow backend parity.

## Non-goals

Phase 8 should not:

- port MacroFinance filters wholesale into BayesFilter;
- move AFNS, cross-currency, Riccati, production panel, or identification
  construction out of MacroFinance;
- claim HMC convergence from smoke tests;
- promote SVD/eigen gradients for HMC without a derivative policy and
  spectral-gap telemetry;
- make MacroFinance depend on BayesFilter before the adapter contract is stable.

## Execution ladder

Use the required cycle for each step:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

### 8.0 Latest-code gate

Plan:

- Obtain a pullable MacroFinance remote state.
- Record exact pre- and post-pull commits.
- Re-run the focused contract inventory after pulling.

Execute:

- Prefer SSH in this environment:
  `git fetch git@github.com:chakkeiwong/MacroFinance.git main`.
- Fast-forward with `git merge --ff-only FETCH_HEAD`.
- Refresh `origin/main` over SSH if the configured `origin` remains HTTPS.

Tests:

- `git status --short --branch`
- `git log -1 --oneline`
- Focused MacroFinance tests listed in Step 8.1 if dependencies are available.

Audit:

- If provider, derivative, or adapter signatures changed, revise this plan.
- This gate currently passes at `e23c31e`.

Memo update:

- Record pull result, commit hash, and whether the rest of Phase 8 is justified.

### 8.1 Contract and derivative audit

Plan:

- Audit the one-country provider and derivative tensors before any BayesFilter
  adapter is added.
- Treat existing MacroFinance tests as regression references, not proof by
  reputation.

Execute:

- Inspect these files:
  - `/home/chakwong/MacroFinance/domain/types.py`
  - `/home/chakwong/MacroFinance/inference/posterior_adapter.py`
  - `/home/chakwong/MacroFinance/one_country_derivative_provider.py`
  - `/home/chakwong/MacroFinance/one_country_tf_derivative_provider.py`
  - `/home/chakwong/MacroFinance/filters/differentiated_kalman.py`
  - `/home/chakwong/MacroFinance/filters/solve_differentiated_kalman.py`
  - `/home/chakwong/MacroFinance/filters/qr_sqrt_differentiated_kalman.py`
  - `/home/chakwong/MacroFinance/filters/tf_qr_sqrt_differentiated_kalman.py`
  - `/home/chakwong/MacroFinance/tests/test_one_country_analytic_hmc_adapter.py`
  - `/home/chakwong/MacroFinance/tests/test_one_country_hmc_analytic_gradient_hessian.py`
- Record derivative shape conventions, parameter ordering, jitter convention,
  initial-state convention, `order` behavior, value-plus-score behavior,
  Hessian-only diagnostics behavior, and backend parity expectations.

Tests:

- In MacroFinance, run a focused local subset when dependencies permit:
  - `pytest -q tests/test_one_country_analytic_hmc_adapter.py`
  - `pytest -q tests/test_one_country_hmc_analytic_gradient_hessian.py`
  - `pytest -q tests/test_one_country_theta2_geometry_diagnostics.py`
  - `pytest -q tests/test_filter_conventions.py`

Audit:

- Verify that the one-country provider is still the smallest safe pilot.
- Verify that score/Hessian tests are local regression references and do not
  require BayesFilter to own the derivative formulas yet.

Memo update:

- Record whether Step 8.2 remains justified.

### 8.2 Value-only BayesFilter wrapper

Plan:

- Add a BayesFilter optional adapter module that converts any MacroFinance-like
  LGSSM object into `bayesfilter.filters.kalman.LinearGaussianStateSpace`.
- Use structural typing instead of importing MacroFinance at package import
  time.
- Attach optional `StatePartition` metadata only when the provider supplies a
  declared partition; do not infer mixed structural timing from `Q`.

Likely BayesFilter files:

- `bayesfilter/adapters/__init__.py`
- `bayesfilter/adapters/macrofinance.py`
- `tests/test_macrofinance_adapter.py`

Execute:

- Implement a conversion function such as:
  `macrofinance_lgssm_to_bayesfilter(model, partition=None)`.
- Implement a value wrapper such as:
  `evaluate_macrofinance_provider_likelihood(provider, theta, observations,
  jitter)`.
- Keep imports optional so BayesFilter can test without MacroFinance installed.

Tests:

- A pure BayesFilter fake MacroFinance-shaped LGSSM object converts correctly.
- With `/home/chakwong/MacroFinance` available, the one-country provider value
  at the fixture reference point matches the MacroFinance differentiated
  Kalman log likelihood to tight tolerance.
- The adapter preserves initial mean/covariance and jitter behavior.
- Missing MacroFinance checkout produces an explicit skipped integration test,
  not an import failure.

Audit:

- Confirm no financial model construction was copied.
- Confirm no MacroFinance filter implementation was duplicated.
- Confirm the adapter result metadata labels it as exact LGSSM value-only.

Memo update:

- Record value parity result and whether Step 8.3 is justified.

### 8.3 Analytical derivative bridge, no migration

Plan:

- Add a narrow derivative bridge only after Step 8.1 passes.
- The bridge should expose MacroFinance-provided value-plus-score and
  value-plus-score-plus-Hessian as separate BayesFilter-facing workloads.
- It should not reimplement the derivative recursion inside BayesFilter yet.

Execute:

- Define a small BayesFilter protocol for derivative-capable LGSSM providers.
- Wrap MacroFinance derivative outputs into a BayesFilter result shape with:
  - log likelihood;
  - score vector;
  - optional Hessian matrix;
  - parameter names;
  - backend provenance;
  - jitter, initial-mean-policy, derivative-order, and shape metadata.

Tests:

- One-country score and Hessian from the bridge match MacroFinance's
  `differentiated_kalman_loglik` result.
- One-country value-plus-score bridge does not require Hessian work.
- Hessian is symmetric to declared tolerance.
- Parameter names and shapes match the provider protocol.

Audit:

- Confirm the derivative bridge is delegation and metadata normalization, not a
  code port.
- Do not promote the bridge to HMC-ready unless Step 8.4 passes.

Memo update:

- Record derivative parity and whether HMC smoke is justified.

### 8.4 Tiny HMC-readiness smoke gate

Plan:

- Only run this if value, score, Hessian, and eager/compiled expectations are
  already documented.
- Reuse MacroFinance's existing one-country analytic HMC fixture as the
  reference behavior.
- Treat MacroFinance's target-XLA chain policy as evidence for future
  BayesFilter HMC contracts, not a sampler implementation to move now.

Execute:

- Add no sampler implementation to BayesFilter in this phase.
- Optionally add a BayesFilter-side adapter-conformance test that confirms the
  wrapped posterior exposes the operations a future HMC driver needs.

Tests:

- `log_prob`, `grad_log_prob`, and negative Hessian are finite at the initial
  position and a nearby perturbed point.
- `log_prob_and_grad` is finite and separate from Hessian diagnostics.
- Any sampler run is labeled `smoke` only.
- Do not assert convergence diagnostics in Phase 8.

Audit:

- Confirm that the result supports only an HMC-readiness label, not a
  convergence claim.

Memo update:

- Record whether Phase 10 can later use the adapter as a candidate HMC target.

### 8.5 Large-scale and cross-currency deferral note

Plan:

- Document why large-scale and cross-currency MacroFinance providers are not
  the first adapter targets.

Execute:

- Add a short audit note covering:
  - `LargeScaleLGSSMDerivativeProvider`;
  - masked backends;
  - `CrossCurrencyStructuralDerivativeProvider`;
  - production provider readiness blockers.

Tests:

- No new integration code required.
- Optionally run focused MacroFinance tests if dependencies permit:
  - `pytest -q tests/test_large_scale_lgssm_derivative_provider.py`
  - `pytest -q tests/test_cross_currency_structural_backend_parity.py`
  - `pytest -q tests/test_provider_economics_contracts.py`

Audit:

- Confirm these are Phase 8 follow-on or Phase 10 candidates, not first-slice
  blockers.

Memo update:

- Record which provider should be second after the one-country pilot.

## Proposed first implementation slice

The first code slice should be Step 8.2 only:

1. Add `bayesfilter/adapters/macrofinance.py`.
2. Convert a MacroFinance-shaped LGSSM to BayesFilter's exact value object.
3. Add a pure BayesFilter unit test using a local fake object.
4. Add an optional integration test that imports the one-country MacroFinance
   fixture only when `/home/chakwong/MacroFinance` is importable.
5. Compare BayesFilter value likelihood against MacroFinance's current
   differentiated Kalman value at the same provider reference point.

Step 8.3 should wait until the Step 8.1 derivative audit has been recorded, but
it can now explicitly target MacroFinance's split between value-plus-score and
value-plus-score-plus-Hessian workloads.

## Go/no-go criteria

Proceed with Step 8.2 if:

- latest-code gate remains resolved at `e23c31e` or a newer audited commit;
- one-country provider signatures are stable;
- BayesFilter's local exact Kalman tests still pass;
- the adapter can be written without importing MacroFinance at BayesFilter
  package import time.

Stop and ask for direction if:

- a newer MacroFinance update changes one-country provider signatures again;
- the one-country regression tests fail before BayesFilter adapter work begins;
- the adapter requires copying MacroFinance financial model or filter code into
  BayesFilter.

## Hypotheses to test

- H8.1: MacroFinance's one-country restricted AFNS provider is a sufficient
  pilot target for the BayesFilter adapter boundary because it is small,
  deterministic, analytically differentiated, and already covered by local
  score/Hessian parity tests.
- H8.2: BayesFilter can match MacroFinance value likelihoods by converting only
  the LGSSM dataclass fields and preserving initial-state and jitter
  conventions.
- H8.3: A derivative bridge can normalize MacroFinance score/Hessian outputs
  without moving derivative recursion code into BayesFilter; the bridge should
  preserve MacroFinance's split value-plus-score and Hessian workloads.
- H8.4: Large-scale and cross-currency providers should be second-stage targets
  because their masking, production-readiness, and identification policies are
  broader than the minimal adapter boundary.
- H8.5: HMC-readiness can be tested with finite value/gradient/Hessian checks
  before any BayesFilter sampler promotion, but convergence claims must wait
  for Phase 10.
