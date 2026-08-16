# Direct-Factor SR-UKF Execution Result

Date: 2026-08-16
Plan: `docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md`
Audit gate: Fable focused re-audit `VERDICT: AGREE`

## Implemented

## Default Backend Decision

The direct-factor route is now the repository default SR-UKF backend for its
supported contract:

- backend identifier: `direct_factor_srukf`;
- model contract: `TFFactorSRUKFModel` and `TFFactorSRUKFDerivatives`;
- numerical route: direct QR residual stacks plus sequential lower-rank
  downdates; and
- public entry point: `tf_default_srukf_value_and_score`.

The former principal-square-root/eigen SR-UKF implementation is preserved for
explicit reproducibility and parity checks, but is classified as
`historical_reference` and is no longer a repository default. The policy is
fail-closed: legacy covariance models and specialized observation/transition
contracts are not silently converted or routed through the factor API. They
remain on their existing named adapters until a contract-specific migration is
reviewed. SSL-LSTM remains explicitly out of scope.

- Added the admitted TensorFlow/float64 QR stack kernel in
  `bayesfilter/linear/stack_qr_tf.py` with positive pivots and first
  derivatives.
- Added the admitted sequential lower rank-one downdate kernel in
  `bayesfilter/linear/lower_rank_downdate_tf.py` with corrected recurrence,
  derivative propagation, and hard positive-margin assertions.
- Added the batched direct-factor DZ5 SR-UKF in
  `bayesfilter/nonlinear/factor_srukf_tf.py`. It carries lower factors through
  `tf.while_loop`, uses QR for prediction/innovation factors, reuses the single
  gain solve for downdates, includes observation factors and their derivatives,
  and supports eager and XLA execution.
- Added the non-admitted one-time legacy covariance adapter in
  `bayesfilter/nonlinear/factor_srukf_compat.py`.
- Added lazy package exports, a repository-owned SR-UKF backend policy, and a
  closed, case-insensitive admitted-file route
  guard. The historical `srukf_factor_tf.py` exception remains diagnostic only.
- Added primitive, integration, derivative, compatibility, failure, batch,
  XLA/eager, and route-guard tests.

## Evidence

Commands run with `TF_FORCE_GPU_ALLOW_GROWTH=true`:

```text
pytest -q tests/test_stack_qr_tf.py tests/test_lower_rank_downdate_tf.py \
  tests/test_factor_srukf_tf.py tests/test_factor_srukf_route_guard.py \
  tests/test_factor_srukf_compat.py
13 passed

pytest -q tests/test_srukf_factor_tf.py tests/test_actual_sv_srukf_tf.py
14 passed
```

The focused tests include QR/downdate covariance and derivative reconstruction,
centered finite differences, dense gain orientation, nonfinite/indefinite
failure checks, parameter-dependent observation noise, duplicate-row
invariance, and CPU XLA/eager parity. `git diff --check` and Python bytecode
compilation pass.

## Boundary and nonclaims

This result does not execute the MacroFinance DZ5 rare-row qualification,
production batch-480 campaign, GPU execution matrix, or downstream adapter.
It does not touch MacroFinance, launch
NeuTra/HMC, or establish posterior, convergence, production, or scientific
validity. Those remain separate holdout gates under the execution plan.

## Non-SSL model parity follow-up

The model-level parity campaign is recorded at
`docs/plans/artifacts/direct-factor-srukf-model-parity-20260816/model_parity_report.md`.
It directly exercised Models A/B/C only. Model A matched the principal-root
value and score to float64 roundoff; Models B/C were finite and close but show
the expected nonlinear factor-orientation dependence. Austria/SIR,
predator-prey, KSC mixture, generalized/actual-SV, and other author-specific
high-dimensional routes were not silently compared because their observation,
initial-event, mixture, or proposal-density contracts are not represented by
the current direct-factor additive-Gaussian API. SSL-LSTM was excluded by user
request.

The parity harness also caught and corrected the direct-factor sigma-point
orientation: for a lower factor `P=L L'`, points must use row offsets
`offset @ L'`. Focused tests and the existing non-SSL baseline suites pass
after that correction. The route is now the repository default for its factor
contract; the parity table remains diagnostic evidence and does not establish
nonlinear equivalence, production readiness, or scientific validity.
