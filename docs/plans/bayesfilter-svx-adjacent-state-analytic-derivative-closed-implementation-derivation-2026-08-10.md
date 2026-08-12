# SVX adjacent-state analytic derivative closed derivation

Date: 2026-08-10
Status: `CLOSED_DERIVATION_FOR_IMPLEMENTATION`

## Purpose

This note closes the derivation for the active SVX-ZC score repair at the level
needed for implementation.

It matches the **current active batched value program** in
`bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`, including:

- the time-0 one-axis fit,
- the `t >= 1` adjacent-state two-axis fit,
- the fixed sweep order `(0, 1, 1, 0)`,
- the active max-shift square-root target,
- the scaled ridge solve used by the code,
- and the retained-marginal recursion.

This is a same-program derivation. It is not the transformed-SV independent
panel route.

## Global notation

Let the unconstrained parameter row be

- `theta = (theta0, theta1)`.

The source-probit chart used by the active route is

- `gamma(theta0) = 0.1 + 0.8*Phi(theta0)`
- `beta(theta1) = 0.1 + 0.8*Phi(theta1)`.

Hence

- `d gamma / d theta0 = 0.8 * phi(theta0)`
- `d beta / d theta1 = 0.8 * phi(theta1)`.

For each batch row `r`, let the transformed observations be `y_{r,t}` and

- `z_{r,t} = log(y_{r,t}^2)`.

The active observation residual is

- `r_{r,t}(x) = z_{r,t} - 2*log(beta_r) - x`.

The active observation log density term is

- `ell_obs_{r,t}(x) = 0.5*r_{r,t}(x) - 0.5*exp(r_{r,t}(x)) - 0.5*log(2*pi)`.

The transition log density term is

- `ell_tr_{r,t}(x_t, x_{t-1}) = -0.5*log(2*pi) - 0.5*(x_t - gamma_r*x_{t-1})^2`.

The code uses fixed degree/order/rank/half-width settings and fixed quadrature
nodes/weights for the current adapter.

## Active batched value path

The active batched value path is the finite program implemented by
`batched_fixed_tt_likelihood_value_trace(...)`.

For each row `r`, it returns the total log-likelihood increment sum over time:

- `V_r(theta) = sum_t (log Z_{r,t}(theta) + c_{r,t}(theta))`

where:

- `c_{r,t}` is the max-shift selected by the code,
- `Z_{r,t}` is the squared-TT normalizer of the fitted core(s) at time `t`.

The score backend must differentiate this same program.

## Time-0 closed derivation

### Step 0.1 — exact one-axis target

For each row `r` and one-axis quadrature node `u_n`, the active time-0 target is

- `L_{r,0,n} = ell_init_r(x_n) + ell_obs_{r,0}(x_n) + c_coord`

where `x_n` is the physical point mapped from the quadrature node and
`ell_init_r` is the active stationary initial latent log density.

The code chooses the active max-shift

- `c_{r,0} = max_n L_{r,0,n}`

and forms the square-root target

- `S_{r,0,n} = exp(0.5 * (L_{r,0,n} - c_{r,0}))`.

### Step 0.2 — target derivative

Assume the max index is locally fixed on the active branch. Let `n*` be the
selected maximizer.

Then

- `dot c_{r,0} = dot L_{r,0,n*}`
- `dot S_{r,0,n} = 0.5 * S_{r,0,n} * (dot L_{r,0,n} - dot c_{r,0})`.

The derivative of `L_{r,0,n}` is the sum of the initial-state score and the
observation score:

- `dot L_{r,0,n} = dot ell_init_r(x_n) + dot ell_obs_{r,0}(x_n)`.

In the active program, the quadrature nodes, weights, and basis evaluations are
fixed with respect to `theta`, so the only dependence enters through the chart
parameters and the latent density terms.

### Step 0.3 — one-axis fit with batch column scaling

Let the one-axis design matrix for row `r` be `D_{r,0}` with columns formed by
basis evaluation at the one-axis nodes. The code’s solve is the scaled ridge
least-squares system used by `_scaled_qr_solve(...)`.

Define the column scales

- `s_{r,0,j} = max(raw_norm_{r,0,j}, scale_floor_r)`

where

- `raw_norm_{r,0,j} = sqrt(sum_n w_n * D_{r,0,nj}^2)`
- `scale_floor_r = max(sqrt(eps) * max_j raw_norm_{r,0,j}, eps)`.

Define the scaled design

- `\tilde D_{r,0,nj} = D_{r,0,nj} / s_{r,0,j}`.

Then the code solves the regularized normal system

- `N_{r,0} v_{r,0} = b_{r,0}`
- `N_{r,0} = \tilde D_{r,0}^T W \tilde D_{r,0} + diag(ridge / s_{r,0}^2)`
- `b_{r,0} = \tilde D_{r,0}^T W S_{r,0}`
- `c_{r,0} = v_{r,0} / s_{r,0}`.

This is the exact implementation-level objective.

### Step 0.4 — one-axis fit derivative

The active branch keeps the support of the scale-floor selection fixed.
Therefore the derivative is computed on the fixed branch of the column scaling.

For each row `r` and column `j`,

- `dot raw_norm_{r,0,j} = (1 / raw_norm_{r,0,j}) * sum_n w_n * D_{r,0,nj} * dot D_{r,0,nj}`
  when `raw_norm_{r,0,j}` is active, with the corresponding fixed-branch floor
  derivative when the floor branch is active.

Then

- `dot s_{r,0,j}` follows the active branch of the max between `raw_norm` and
  `scale_floor`.

The scaled design derivative is

- `dot \tilde D = dot D / s - D * dot s / s^2`.

The derivative of the normal matrix is

- `dot N = dot \tilde D^T W \tilde D + \tilde D^T W dot \tilde D + dot R`

where the ridge term contributes

- `dot R = diag(-2 * ridge * dot s / s^3)`.

The derivative of the right-hand side is

- `dot b = dot \tilde D^T W S + \tilde D^T W dot S`.

The coefficient derivative satisfies

- `dot v = N^{-1} (dot b - dot N v)`
- `dot c = dot v / s - v * dot s / s^2`.

### Step 0.5 — one-axis normalizer derivative

The one-axis fitted core has coefficients `c_{r,0}` and the normalizer is

- `Z_{r,0} = ||c_{r,0}||^2`.

Therefore

- `dot Z_{r,0} = 2 * c_{r,0}^T * dot c_{r,0}`.

The time-0 score increment is

- `dot lambda_{r,0} = dot c_{r,0} + dot Z_{r,0} / Z_{r,0}`.

## Time `t >= 1` closed derivation

### Step 1 — exact two-axis target

Let `pHat_{r,t-1}(x_{t-1})` be the retained normalized marginal from the
previous step.

The active two-axis target is

- `L_{r,t,n} = log pHat_{r,t-1}(x_{t-1,n}) + ell_tr_{r,t}(x_{t,n}, x_{t-1,n}) + ell_obs_{r,t}(x_{t,n}) + c_coord`.

The code again chooses

- `c_{r,t} = max_n L_{r,t,n}`
- `S_{r,t,n} = exp(0.5 * (L_{r,t,n} - c_{r,t}))`.

Assume the argmax branch is fixed locally.
Then

- `dot c_{r,t} = dot L_{r,t,n*}`
- `dot S_{r,t,n} = 0.5 * S_{r,t,n} * (dot L_{r,t,n} - dot c_{r,t})`.

The derivative of the target log values is

- `dot L_{r,t,n} = dot log pHat_{r,t-1}(x_{t-1,n}) + dot ell_tr_{r,t}(x_{t,n}, x_{t-1,n}) + dot ell_obs_{r,t}(x_{t,n})`.

The transition term derivative is explicit:

- `dot ell_tr_{r,t} = (x_{t,n} - gamma_r*x_{t-1,n}) * (dot gamma_r * x_{t-1,n})`.

The observation term derivative is explicit through `beta_r`:

- `dot ell_obs_{r,t} = (exp(r_{r,t}) - 1) * dot beta_r / beta_r`.

### Step 2 — exact two-axis fixed-branch ALS sweep

The active code alternates over the sweep order `(0, 1, 1, 0)`.
For each row `r`, let the current two-axis TT cores be `C^0` and `C^1`.
Each update is a weighted ridge LSQ system with design matrix `D_{r,t,k}`.

For one core update `k`,

- `N_{r,t,k} v_{r,t,k} = b_{r,t,k}`
- `N_{r,t,k} = \tilde D_{r,t,k}^T W \tilde D_{r,t,k} + diag(ridge / s_{r,t,k}^2)`
- `b_{r,t,k} = \tilde D_{r,t,k}^T W S_{r,t}`
- `c_{r,t,k} = v_{r,t,k} / s_{r,t,k}`.

Here `\tilde D_{r,t,k}` is the active scaled design matrix for that core update,
constructed from the current other core and the fixed basis evaluations.

The derivative is the same fixed-branch LSQ rule as above:

- `dot v_{r,t,k} = N_{r,t,k}^{-1}(dot b_{r,t,k} - dot N_{r,t,k} v_{r,t,k})`
- `dot c_{r,t,k} = dot v_{r,t,k} / s_{r,t,k} - v_{r,t,k} dot s_{r,t,k} / s_{r,t,k}^2`.

The key active-program fact is that `D_{r,t,k}` depends on the current and
previous sweep cores, so `dot D_{r,t,k}` is generally nonzero.
That dependence is propagated through the fixed sweep order in the same order as
the value path.

### Step 3 — two-axis normalizer derivative

Let the fitted two-axis cores be `L_{r,t}` and `R_{r,t}`.
The normalizer is

- `Z_{r,t} = sum_{a,b} M^L_{ab} M^R_{ab}`
- `M^L_{ab} = <L_a, L_b>`
- `M^R_{ab} = <R_a, R_b>`.

Therefore

- `dot Z_{r,t} = sum_{a,b} dot M^L_{ab} M^R_{ab} + M^L_{ab} dot M^R_{ab}`
- `dot M^L_{ab} = <dot L_a, L_b> + <L_a, dot L_b>`
- `dot M^R_{ab} = <dot R_a, R_b> + <R_a, dot R_b>`.

The time-`t` score increment is then

- `dot lambda_{r,t} = dot c_{r,t} + dot Z_{r,t} / Z_{r,t}`.

## Retained marginal recursion

The active route feeds the retained normalized density values forward.
Let

- `m_{r,t}(x)` be the retained numerator,
- `rho_{r,t}(x) = m_{r,t}(x) / Z_{r,t}` be the normalized retained density.

Then the derivative is the quotient rule

- `dot rho_{r,t} = dot m_{r,t} / Z_{r,t} - m_{r,t} dot Z_{r,t} / Z_{r,t}^2`.

This is exactly the role of the retained-marginal derivative helper family in
`filtering.py`.

## Final closed score formula

For each batch row `r`, the analytic score of the active finite program is

- `grad_theta log pHat(y_{r,1:T} | theta) = sum_t dot lambda_{r,t}`

with

- `dot lambda_{r,0}` from the one-axis branch,
- `dot lambda_{r,t}` for `t >= 1` from the two-axis adjacent-state branch,
- and the retained marginal derivative propagated between steps.

## Exact implementation correspondence

The active implementation now has a trace surface that matches this derivation:

- `batched_fixed_tt_likelihood_value_trace(...)` exposes per-time targets,
  shifts, square-root targets, normalizers, and sweep-local fits.
- `_OneAxisFitTrace` carries the one-axis solve state.
- `_TwoAxisFitTrace` and `_TwoAxisSweepTrace` carry the active `(0,1,1,0)`
  sweep state.
- The active score path currently remains autodiff-based; this derivation
  supports replacing that path with a same-program analytic replay.

The reusable helper algebra that this derivation depends on already exists in:

- `bayesfilter/highdim/derivatives.py`
- `bayesfilter/highdim/filtering.py`
- `bayesfilter/highdim/zhao_cui_moment_teacher_als.py`

## Assumptions

- fixed branch only;
- same active value program;
- no moving basis;
- same frozen initial and adjacent UKF core identity;
- fixed nodes, weights, order, degree, rank, and dtype;
- local active branch choices for argmax and scale-floor selection.

## Non-claims

- This note does not claim the backend is already implemented.
- This note does not claim the transformed-SV independent-panel analytic route
  is the same target.
- This note does not claim HMC readiness.
