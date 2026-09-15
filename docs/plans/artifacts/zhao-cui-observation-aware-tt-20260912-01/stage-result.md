# Observation-aware Zhao--Cui TT analysis

Date: 2026-09-12  
Status: bounded literature and call-chain analysis; no runtime code changed and no numerical superiority claim made.

## Goal

Construct an observation-aware conditional TT proposal for Zhao--Cui Algorithm 3 that improves finite TT fitting or proposal coordinates while preserving the exact filtering target, conditional proposal law, and importance-weight identity.

## Problem found

The exact observation factor is already present in the local Algorithm 2 fit. In `bayesfilter/highdim/zhao_cui_algorithm2_preparation_tf.py`, `target_log` includes `observation_factor` for both the initial and transition fits (lines 104--117). The local `likelihood_weighted_sigma_point_chart` helper (lines 27--67) instead computes a positive weighted Gaussian projection from `g(y_t | x_t, theta)` at `2d` sigma points. It is explicitly classified as `extension_or_invention` and has no production call site; only a test imports the preparation path. Therefore the current lane does not implement an observation-responsive UKF-guided proposal.

This is a coordinate and call-chain gap, not a missing likelihood term. Replacing the exact TT target by a bare UKF Gaussian would be wrong relative to Algorithm 3.

## Correct guided construction

Let `c = (theta, x_{t-1})` and

```text
q_t(x,c) = pi_hat_{t-1}(c) f(x | c) g(y_t | x, theta).
```

Choose an observation-dependent, invertible guide

```text
x = G_t(u; c, y_t) = m_t(c,y_t) + L_t(c,y_t) u,
```

with nonsingular `L_t`. Fit the pulled-back target

```text
q_t^G(u,c) = q_t(G_t(u;c,y_t), c) |det L_t(c,y_t)|
```

or, in Zhao--Cui's bridge formulation, fit the exact target-to-bridge ratio in the declared reference measure. Build the squared-TT conditional in `u`, sample `u` by the conditional KR map, and return `x = G_t(u;c,y_t)`.

The physical proposal density required by the Algorithm 3 weight is

```text
p_hat_X(x | c,y_t)
  = p_hat_U(G_t^{-1}(x;c,y_t) | c,y_t) |det L_t(c,y_t)|^{-1}.
```

The determinant is mandatory. A guide may depend on `c` without changing this determinant because the joint map `(u,c) -> (x,c)` is block triangular. The exact `f` and `g` factors remain in the fitted target and the final correction

```text
omega_f = f(x_t | x_{t-1},theta) g(y_t | x_t,theta)
          / p_hat_X(x_t | theta,x_{t-1},y_{1:t}).
```

## Literature decision

Zhao--Cui Section 5 already provides the source-faithful route: a bridge `rho_t`, a map `T_t` that sends the bridge to a tensor-product reference, fitting the transformed target-to-bridge ratio, and composing the resulting KR map. Their Gaussian bridge estimates the full joint moments of `(x_t, theta, x_{t-1})` from weighted particles; their tempered bridge retains powers of the posterior, transition, and likelihood factors. These are the proper baselines before a new UKF extension.

Standard UKF is a different object from the local helper. Julier--Uhlmann uses `2n+1` sigma points, transforms them through the observation map, and forms predicted observation covariance and state-observation cross-covariance for a Kalman update. The helper instead softmaxes exact likelihood values at state sigma points and projects weighted state moments. It should be described as likelihood-weighted cubature/Gaussian projection, not as standard UKF.

## Recommended sequence

1. Reproduce the source-faithful Gaussian bridge and conditional proposal, including its full-joint chart and determinant bookkeeping.
2. Add a response test: changing `y_t` with the transition fixed must change the guide, transformed target, or fitted conditional proposal. A frozen chart or frozen row law fails this test.
3. Compare three coordinate providers under the same exact target and weight identity: source Gaussian bridge; standard UKF where the observation model supports its cross-covariance assumptions; and the explicit likelihood-weighted cubature extension.
4. If concentration or multimodality remains, test Zhao--Cui's tempered nonlinear bridge before claiming UKF value.
5. When a guide changes the reference measure, recompute the design measure and regression weights. Christoffel rows and inverse weights cannot be reused by assumption after a coordinate change; weighted least-squares stability depends on the sampling measure and weight relation.

For the C2 stochastic-volatility observation model, the raw signed observation has zero population state/observation cross-covariance under the usual symmetric prior/noise setup, so a standard raw-observation UKF is a weak hypothesis. A transformed observation or likelihood-weighted projection is model-specific and must remain an extension until tested.

## Decision table

| Decision | Primary criterion | Veto / uncertainty | Current status | Next action |
|---|---|---|---|---|
| Exact observation-aware target | `f*g` retained in TT target and Algorithm 3 correction | Missing factor or wrong denominator | Pass in inspected local fit and paper identity | Preserve as invariant |
| Observation-responsive guide | proposal changes with `y_t` at fixed transition | frozen chart/rows; omitted determinant | Not implemented end to end | Wire a bounded bridge/guide test |
| Standard UKF as default | valid sigma transform, innovation covariance, cross-covariance for model | weak/degenerate cross covariance; no source support that it is best | Unsupported | Keep as comparator |
| Likelihood-weighted sigma projection | finite positive SPD projection and exact downstream correction | multimodality, under-dispersion, extension mislabeling | Viable extension hypothesis | Compare against source Gaussian bridge |
| Tempered nonlinear bridge | lower TT concentration/rank at equal target/weight correctness | changed target without ratio correction | Source-supported candidate | Implement only after baseline tie-out |

## Inference status

| Evidence class | Result |
|---|---|
| Hard correctness screen | The inspected formulas require the exact likelihood factor and affine Jacobian; no new code was admitted. |
| Statistically supported ranking | None; no experiment was run. |
| Descriptive differences | None; literature and call-chain evidence only. |
| Default readiness | Not established for UKF, cubature, or any new chart. |
| Evidence needed next | Source Gaussian-bridge tie-out, observation-response test, determinant/round-trip checks, then bounded replicated proposal comparison. |

## What this does not conclude

It does not show that UKF is best, that the likelihood-weighted projection improves filtering, that the current TT fit is inaccurate, or that a nonlinear bridge will reduce rank in the target implementation. Those require a separate experiment with a declared baseline ladder and uncertainty-aware diagnostics.
