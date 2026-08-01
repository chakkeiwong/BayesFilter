# Phase 8 Owner-Decision Amendment Handoff

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `OWNER_SCIENTIFIC_DECISIONS_REQUIRED_BEFORE_NEXT_TARGET_ARM`

Decision-ready proposal:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-owner-decision-amendment-proposal-2026-07-14.md`.

The proposal completed a bounded scientific review/repair loop with final
`VERDICT: AGREE`. It does not authorize execution.

A subsequent reviewed Kalman-only decision-support result is
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-kalman-decision-support-result-2026-07-14.md`.
It evaluated no Contract E candidate and selected no margin.

## What Is Already Closed

- canonical Contract E shared-core dtype and exact tiny derivative checks;
- tiny Kalman oracle harness;
- PHILOX target preparation and complete reset/transport telemetry;
- CPU-hidden/XLA `T=1,N=4` wiring smoke under an explicitly noncandidate arm;
- gap/eigenvalue and full row/column marginal instrumentation with O(N)
  retained state and frozen scalar/score preservation.

## Why The Next Arm Is Blocked

The reviewed amendment offers a center-scoped Phase 8 decision while explicitly
deferring full-box HMC readiness. Under that scope, a fixed canonical ridge is
selected on the disjoint lower rung and is not tuned or stopped from calibration
or audit output. It need not establish full-domain validity for the narrower
center claim. A future full-box or HMC-ready claim would additionally require
the same fixed ridge to remain valid over its declared parameter domain.

For either scope, raw covariance displacement and finite-transport error must be
controlled through the reviewed downstream value/gradient stability gate; they
cannot be justified from floating-point roundoff or the completed one-step
smoke. The remaining blocker is therefore the owner decision on scope, gradient
loss and effect-size margin, staged lower-rung design/budget, and statistical
audit design.

## Decisions Needed For A Lower-Rung Numerical Arm

The owner must approve or supply. The decision-ready proposal gives one reviewed
recommended construction for each item:

1. Scope: either the reviewed center-scoped Phase 8 decision with full-box HMC
   readiness deferred, or a declared physical/transformed HMC-coordinate region
   over which one fixed prepared ridge must keep all Contract E Cholesky charts
   valid. A center result is insufficient only for the latter full-domain claim.
2. Raw covariance bias budget: the norm/scale and largest acceptable
   `||Sigma_output-Sigma_target||` for the reset. This is not the Kalman value
   `0.1%` rule and cannot be derived from it without a sensitivity argument.
3. Conditioning budget: acceptable factor/solve amplification tied to the
   downstream value/gradient use.
4. Transport budget: maximum permitted final-coupling row and column residuals,
   using the frozen targets `1` and `N*w`, plus a chunk-drift budget.
5. Candidate provenance: a pre-result ridge/epsilon/scaling/step/chunk ladder
   derived independently of the completed smoke, with deterministic
   selection/no-selection and tie rules.
6. Lower-rung execution: exact `T,N`, estimator seeds, attempt cap, and compute
   cap. The existing plan forbids jumping directly to `T=10` or GPU.

## Additional Primary-Shape Decisions

Before any `T=50,N=10000` output, the owner must also freeze:

- HMC-coordinate gradient equivalence margin;
- pilot seeds and ordered audit pool;
- simultaneous interval method and applicability/failure rule;
- deterministic audit-count/power function and compute cap; and
- any pilot tuning grid/statistic/tie/no-selection/selection-variance rule.

The reviewed proposal removes primary-shape tuning: settings freeze on the
disjoint lower rung, calibration cannot repair them, and audit cannot tune. The
owner must choose exactly one reviewed gradient loss (recommended: center-
componentwise relative error, with no floor because all five center oracle
components are nonzero) and `delta_grad`, choose one fixed audit count (`20`,
`32`, or `64`), and later approve the primary-shape GPU wall-time budget.

## Exact Kalman Decision Support

At the frozen `T=50` center, the exact HMC-coordinate Kalman gradient is

```text
(2.7236632176, -2.6749518801, 0.2653223776,
 -0.6710130947, 1.9594196636).
```

All five components are nonzero. The recommended center-scoped gradient loss is
therefore

```text
max_k abs(g_ContractE,k - g_Kalman,k) / abs(g_Kalman,k) <= delta_grad,
```

with no near-zero floor. This directly exposes sign and magnitude failure in
`q_scale`; it does not claim off-center or HMC-trajectory validity. The owner
may still choose the reviewed global-contribution metric, but the oracle result
shows it is materially more permissive for `phi3` and `q_scale`.

Illustrative absolute HMC-gradient error budgets under the componentwise
criterion are:

| `delta_grad` | phi1 | phi2 | phi3 | q_scale | r_scale |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.01` | 0.02724 | 0.02675 | 0.00265 | 0.00671 | 0.01959 |
| `0.02` | 0.05447 | 0.05350 | 0.00531 | 0.01342 | 0.03919 |
| `0.05` | 0.13618 | 0.13375 | 0.01327 | 0.03355 | 0.09797 |

These rows are interpretations only, not nominated thresholds.

The remaining minimum owner choices are:

1. approve center-scoped Phase 8 with full-box HMC readiness deferred;
2. choose the componentwise metric above and `delta_grad`, or choose the
   documented global-contribution alternative;
3. approve the reviewed `T=1,N=32` staged lower-rung design and per-node cap;
4. choose fixed audit count `20`, `32`, or `64` and approve the predeclared
   Student/Bonferroni model.

The lower-rung proposal's maximum node envelope is `4200` seconds before
harness/review overhead. It cannot fit the remaining original campaign window,
which ends at `2026-07-14T09:32:19+08:00`. Any approved lower-rung execution
therefore requires a future explicitly bounded continuation; approval must not
be interpreted as resetting this campaign clock.

## Recommended Decision Process

Do not choose numerical thresholds directly. Start from the intended HMC
failure budget: how much log-density/gradient perturbation is acceptable over a
declared region. Use a sensitivity analysis to allocate that budget among raw
reset covariance bias, finite transport marginals, chunk accumulation, and
other numerical errors. If that downstream budget is not yet known, keep the
next target arm blocked and run only independent mathematical/reference work.

## Forbidden Shortcuts

- Do not derive candidates or thresholds from the observed ridge `4` smoke.
- Do not reuse `0.05*sqrt(p)` outside same-program FD.
- Do not reuse the `0.1%` value criterion or actual-SV `6%` as a covariance,
  transport, or LGSSM gradient budget.
- Do not create a center-dependent stopped ridge.
- Do not run another target value/gradient before a reviewed amendment binds
  the decisions above.
