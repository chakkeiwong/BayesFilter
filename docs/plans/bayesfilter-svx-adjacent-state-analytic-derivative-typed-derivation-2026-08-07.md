# SVX adjacent-state analytic derivative typed derivation

Date: 2026-08-07
Status: `PHASE_1_TYPED_DERIVATION`

## Goal

Restate the active SVX-ZC score repair in a tool-friendly form so the local
math tooling can inspect the derivation without drifting to the nearby
transformed-SV analytic lane.

This note is intentionally typed and stepwise:
one equation, one role, one code hook family.

## Symbols

- `theta = (theta0, theta1)` are the source-probit coordinates.
- `gamma = 0.1 + 0.8*Phi(theta0)`.
- `beta = 0.1 + 0.8*Phi(theta1)`.
- `z_t = log(y_t^2)`.
- `r_t(x_t) = z_t - 2*log(beta) - x_t`.
- `ell_obs_t(x_t) = 0.5*r_t - 0.5*exp(r_t) - 0.5*log(2*pi)`.
- `ell_tr_t(x_t, x_tm1) = -0.5*log(2*pi) - 0.5*(x_t - gamma*x_tm1)^2`.

The active target is fixed; only the score backend changes.

## Step 0 — time-0 target and square root target

For the one-axis fit at `t = 0`:

- `L0_n = ell_init(x0_n) + ell_obs_0(x0_n) + c_coord`
- `c0 = max_n L0_n`
- `q0_n = exp(0.5*(L0_n - c0))`

Typed derivative obligation:

- `dot_q0_n = 0.5*q0_n*(dot_L0_n - dot_c0)`

## Step 1 — one-axis fixed-design LSQ solve

The active one-axis fit solves:

- `A0*c0 = b0`
- `A0 = B0^T*W0*B0 + Lambda0`
- `b0 = B0^T*W0*q0`

Typed derivative obligation:

- `A0*dot_c0 + dot_A0*c0 = dot_b0`
- equivalently `A0*dot_c0 = dot_b0 - dot_A0*c0`

In the active program, `B0` and `W0` are fixed at `t=0`, so `dot_A0 = 0`
and the active term is `dot_b0` through `dot_q0`.

## Step 2 — one-axis normalizer

The one-axis normalizer is:

- `Z0 = c0^T*c0`

Typed derivative obligation:

- `dot_Z0 = 2*c0^T*dot_c0`

Score increment:

- `dot_lambda0 = dot_c0 + dot_Z0/Z0`

## Step 3 — adjacent-state target for t >= 1

Let `p_hat_tm1(x_tm1)` be the retained normalized marginal from the previous
step. Then the adjacent-state target is:

- `Lt_n = log(p_hat_tm1(x_tm1_n)) + ell_tr_t(x_t_n, x_tm1_n) + ell_obs_t(x_t_n) + c_coord`
- `ct = max_n Lt_n`
- `qt_n = exp(0.5*(Lt_n - ct))`

Typed derivative obligation:

- `dot_Lt_n = dot_log_p_hat_tm1_n + dot_ell_tr_t_n + dot_ell_obs_t_n`
- `dot_qt_n = 0.5*qt_n*(dot_Lt_n - dot_ct)`

## Step 4 — two-axis weighted LSQ solve

The active two-axis fit alternates over `(0, 1, 1, 0)`.
For each core update `i`:

- `Ai*ci = bi`
- `Ai = Di^T*W*Di + Lambda_i`
- `bi = Di^T*W*qt`

Typed derivative obligation:

- `Ai*dot_ci + dot_Ai*ci = dot_bi`
- `dot_Ai = dot_Di^T*W*Di + Di^T*W*dot_Di`
- `dot_bi = dot_Di^T*W*qt + Di^T*W*dot_qt`

The active adjacent-state program depends on the current and previous cores, so
`dot_Di` is generally nonzero. This is the missing design-aware piece.

## Step 5 — two-axis normalizer

Let the fitted two-axis cores be `L` and `R`.
The normalizer is:

- `Zt = sum_{r,s} M^L_rs * M^R_rs`
- `M^L_rs = <L_r, L_s>`
- `M^R_rs = <R_r, R_s>`

Typed derivative obligation:

- `dot_Zt = sum_{r,s} dot_M^L_rs*M^R_rs + M^L_rs*dot_M^R_rs`
- `dot_M^L_rs = <dot_L_r, L_s> + <L_r, dot_L_s>`
- `dot_M^R_rs = <dot_R_r, R_s> + <R_r, dot_R_s>`

Score increment:

- `dot_lambdat = dot_ct + dot_Zt/Zt`

## Step 6 — retained marginal propagation

If `m_t(x)` is the retained numerator and `rho_t(x) = m_t(x)/Zt`, then

- `dot_rho_t = dot_m_t/Zt - m_t*dot_Zt/Zt^2`

This is the same structural role as the retained-marginal derivative helper in
`filtering.py`.

## Final score

- `grad_theta log p_hat(y_1:T | theta) = sum_t dot_lambdat`

with `dot_lambda0` from the one-axis branch and `dot_lambdat` for `t >= 1`
from the adjacent-state branch.

## Code hook map

### Value authority

- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`

### Reusable algebra

- `bayesfilter/highdim/derivatives.py` for fixed-design LSQ and normalizer
  derivatives.
- `bayesfilter/highdim/filtering.py` for retained-marginal derivative helpers
  and scalar/multistate adjacent-target derivative structure.
- `bayesfilter/highdim/zhao_cui_moment_teacher_als.py` for replaying design-
  aware ALS tangents.

### Missing implementation ingredient

- design-aware derivative propagation through the active adjacent-state ALS
  sweep.

## Non-claims

- This note does not claim the backend is implemented.
- This note does not claim the transformed-SV independent-panel analytic route
  is the same target.
- This note does not claim HMC readiness.
