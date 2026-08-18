# GenUT Score-Gap Closure Plan

Date: 2026-08-17

Status: executed; derivative regression installed, exact-oracle gate withdrawn

## Research Intent

Determine which remaining GenUT score-validation gaps are engineering defects,
which are finite-program derivative diagnostics, and which require a separate
scientific accuracy study. Replace the legacy single-step central-difference
check with a regression-based diagnostic without inventing an exact-Kalman
accuracy threshold.

## Evidence Contract

- **Question:** does the reported score agree with the derivative of the same
  finite value program on a fixed branch, and are the remaining readiness gaps
  localized and testable?
- **Baseline:** current batch-native legacy dual-cap route and its scalar
  diagnostic route, with identical observations, innovations, design, controls,
  and parameter point.
- **Primary engineering criteria:** finite value/score, score-increment
  additivity, value-only/value-score equality, scalar/batch equality, and a
  stable finite-difference regression intercept on the same program.
- **Diagnostic only:** distance to the exact Kalman score/value, cap activity,
  seed dispersion, and approximate nonlinear comparator proximity. These do not
  become admission gates without a separately reviewed statistical contract.
- **Hard blockers:** Austria tangent-free/tangent-carrying value mismatch,
  stale or mismatched tuning scope, nonfinite output, branch changes in the FD
  stencil, scalar/batch mismatch, or failed score-increment accounting.
- **Nonclaims:** no exact nonlinear likelihood claim, no unbiasedness claim,
  no statistical ranking of dual-cap versus diagonal, and no HMC readiness.

## Regression FD Definition

For coordinate `j`, evaluate the same scalar value program at symmetric steps
`h_k` and compute

```text
d_j(h_k) = [V(theta + h_k e_j) - V(theta - h_k e_j)] / (2 h_k).
```

On a smooth fixed branch, fit `d_j(h_k) = a_j + b_j h_k^2` by ordinary
least squares using the available finite points. The intercept `a_j` is the
regression derivative diagnostic; `b_j` is a truncation-sensitivity diagnostic.
The diagnostic must also report the raw ladder, residuals, R-squared when
defined, branch validity, and the condition of the two-column regression.

This is a numerical consistency diagnostic, not a statistical confidence
interval and not an exact-filter accuracy test. A pass requires a declared
ladder with at least three finite points, no validity-branch change, and a
small scale-aware intercept residual under the existing diagnostic policy.
The pass/fail result remains diagnostic until a target-specific tolerance is
reviewed and frozen.

## Gap Closure Matrix

| Gap | Action | Role |
|---|---|---|
| Single-step FD | Add `h^2` regression helper and use it in the four-model and LGSSM diagnostics | derivative diagnostic |
| Exact-Kalman tolerance | Remove the undocumented close-oracle veto; retain oracle errors as descriptive output | governance repair |
| Austria endpoint mismatch | Add a regression test that compares value-only and value/score routes on the same frozen inputs and requires fail-closed mismatch reporting | hard blocker localization |
| Trust-region current scope | Add a current-scope construction test binding route id and controls; do not claim T=50 validation without a run | engineering coverage |
| Scope-specific tuning | Add tests rejecting stale/mismatched scope metadata and retain inherited controls as warm starts only | admission protection |
| NeuTra/HMC | Keep closed until target-specific training, heldout transport checks, and sequential HMC diagnostics exist | continuation gate |

## Skeptical Plan Audit

- **Wrong baseline:** all derivative comparisons use the same finite value
  program, not Kalman as a derivative authority.
- **Proxy promotion:** regression intercepts, residuals, and Kalman distance
  remain diagnostics; none establishes posterior correctness.
- **Hidden threshold:** no new numeric Kalman tolerance is introduced. Existing
  `<=10` gross-error thresholds remain historical diagnostics only.
- **Step selection risk:** a ladder is required; a single convenient step cannot
  pass the derivative check. Invalid branches and nonfinite points fail closed.
- **Scope drift:** current regenerated observation hashes and route controls are
  bound in tests; old artifacts cannot be promoted as current tuning evidence.
- **Stop conditions:** stop implementation if the same-program value/score
  identity or Austria endpoint test exposes a route mismatch requiring a target
  change. Do not repair by changing observations, event order, or objective.

## Execution

1. Implement the pure regression diagnostic and unit tests.
2. Replace legacy single-step benchmark diagnostics with the regression ladder.
3. Add focused route/scope/value-identity tests.
4. Run CPU-hidden tests and the bounded four-model diagnostic.
5. Update the readiness result to remove the invalid exact-oracle veto and list
   the remaining hard blockers and next evidence.

## Execution Record

- Added `docs/benchmarks/genut_fd_regression.py` with the declared
  `central_fd_h2_intercept_regression_v1` policy.
- Replaced the legacy single-step four-model diagnostic with the five-point
  `h^2` regression ladder; invalid stencil endpoints fail closed.
- Added solver-identity/scope binding coverage for the repaired trust-region
  controls.
- CPU-hidden focused tests: `14 passed` for the new regression, GenUT batch,
  and trust-region mechanics paths.
- Python compilation and `git diff --check` passed for all touched files.
- Trusted GPU/XLA LGSSM regression artifact:
  `docs/benchmarks/artifacts/genut-score-gap-closure-20260817/lgssm_oracle_regression/result.json`.
- Trusted GPU/XLA four-model regression artifact:
  `docs/benchmarks/artifacts/genut-score-gap-closure-20260817/four_model_regression/result.json`;
  wall time `583.7 s`, all 16 finite/residual-valid cells.
- Regression diagnostic passed for all KSC-SV arms. LGSSM, predator-prey,
  and Austria retain coordinate-level diagnostic failures. These are
  same-program diagnostics, not exact-oracle accuracy gates.
- The broader target-factory CPU test remains blocked by the pre-existing
  `tf-gpu` custom-op TensorFlow/Abseil ABI mismatch; the trusted `tftwogpu`
  infrastructure probe and GPU campaign are unaffected.

## Artifacts

- `docs/benchmarks/genut_fd_regression.py`
- `tests/test_genut_fd_regression.py`
- updated `docs/benchmarks/run_genut_b098_radial2_four_model.py`
- updated `docs/benchmarks/run_genut_lgssm_oracle_validation_20260816.py`
- updated readiness result memo under `docs/plans/`
