# SVX adjacent-state analytic derivative derivation lock

Date: 2026-08-07
Status: `PHASE_1_DERIVATION_LOCK`

## Purpose

Lock the analytic target for the current active `SVX-ZC` finite program before
any more implementation work.

The target is the score backend for the active batched value program in
`bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`. The value program is
fixed; only the score backend is to be replaced.

## Active value authority

Let `V(θ)` denote the active batched likelihood value returned by
`batched_fixed_tt_likelihood_value_status(θ, ...)`.

The active route uses the source-probit chart

- `γ(θ₀) = 0.1 + 0.8 Φ(θ₀)`
- `β(θ₁) = 0.1 + 0.8 Φ(θ₁)`

with derivatives

- `dγ/dθ₀ = 0.8 φ(θ₀)`
- `dβ/dθ₁ = 0.8 φ(θ₁)`

and the transformed observation residual

- `r_t(x_t; θ) = z_t - 2 log β - x_t`
- `z_t = log(y_t^2)`

with observation log density contribution

- `ℓ_obs(x_t; θ) = 1/2 r_t - 1/2 exp(r_t) - 1/2 log(2π)`.

The transition log density is

- `ℓ_tr(x_t, x_{t-1}; θ) = -1/2 log(2π) - 1/2 (x_t - γ x_{t-1})^2`.

The time-0 latent log density is the stationary Gaussian law used by the code.

## Time-0 target

At `t = 0`, the active value program forms the one-axis target

- `L_0^{(n)}(θ) = ℓ_init(x_0^{(n)}; θ) + ℓ_obs(x_0^{(n)}; θ) + c_coord`

with shift

- `c_0(θ) = max_n L_0^{(n)}(θ)`

and square-root target

- `q_0^{(n)}(θ) = exp(1/2 (L_0^{(n)}(θ) - c_0(θ)))`.

Assuming the maximizer is locally stable,

- `dot c_0 = dot L_0^{(n*)}`
- `dot q_0^{(n)} = 1/2 q_0^{(n)} (dot L_0^{(n)} - dot c_0)`.

The one-axis fit solves a weighted ridge LSQ system

- `A_0 c_0 = b_0`
- `A_0 = B_0^T W_0 B_0 + Λ_0`
- `b_0 = B_0^T W_0 q_0`

where `B_0` is fixed at time 0.

The derivative relation is

- `A_0 dot c_0 = dot b_0 - dot A_0 c_0`.

When the basis and weights are fixed, `dot A_0 = 0`; the only active term is
`dot b_0` through `dot q_0`.

The one-axis normalizer is

- `Z_0 = c_0^T c_0`
- `dot Z_0 = 2 c_0^T dot c_0`.

So the time-0 score increment is

- `dot λ_0 = dot c_0 + dot Z_0 / Z_0`.

## Time `t >= 1` adjacent-state target

For later times, the active value program carries the previous fitted marginal
forward.

Let `p̂_{t-1}(x_{t-1}; θ)` be the retained normalized marginal from the previous
step. Then the two-axis target is

- `L_t^{(n)}(θ) = log p̂_{t-1}(x_{t-1}^{(n)}; θ) + ℓ_tr(x_t^{(n)}, x_{t-1}^{(n)}; θ) + ℓ_obs(x_t^{(n)}; θ) + c_coord`

with shift

- `c_t(θ) = max_n L_t^{(n)}(θ)`
- `q_t^{(n)}(θ) = exp(1/2 (L_t^{(n)}(θ) - c_t(θ)))`.

The derivative is

- `dot L_t^{(n)} = dot log p̂_{t-1}(x_{t-1}^{(n)}; θ) + dot ℓ_tr^{(n)} + dot ℓ_obs^{(n)}`
- `dot q_t^{(n)} = 1/2 q_t^{(n)} (dot L_t^{(n)} - dot c_t)`.

## Two-axis ALS fit

The active two-axis fit alternates over the sweep order `(0, 1, 1, 0)`.
Each core update is a weighted ridge LSQ solve of the form

- `A_i c_i = b_i`
- `A_i = D_i^T W D_i + Λ_i`
- `b_i = D_i^T W q_t`.

The derivative relation at each core update is

- `A_i dot c_i = dot b_i - dot A_i c_i`
- `dot A_i = dot D_i^T W D_i + D_i^T W dot D_i`
- `dot b_i = dot D_i^T W q_t + D_i^T W dot q_t`.

The active program depends on the current and previous cores, so `dot D_i` is
not zero in general. The derivative must be propagated through the same sweep
order as the value path.

This is the missing design-aware ingredient.

## Two-axis normalizer

Let the fitted two-axis cores be `L` and `R`. The normalizer is

- `Z_t = Σ_{r,s} M^L_{rs} M^R_{rs}`

where

- `M^L_{rs} = <L_r, L_s>`
- `M^R_{rs} = <R_r, R_s>`.

Then

- `dot Z_t = Σ_{r,s} dot M^L_{rs} M^R_{rs} + M^L_{rs} dot M^R_{rs}`
- `dot M^L_{rs} = <dot L_r, L_s> + <L_r, dot L_s>`
- `dot M^R_{rs} = <dot R_r, R_s> + <R_r, dot R_s>`.

So the time-`t` score increment is

- `dot λ_t = dot c_t + dot Z_t / Z_t`.

## Retained marginal propagation

The retained density values used at the next time step must also be differentiated.
If `m_t(x; θ)` is the retained numerator and `ρ_t(x; θ) = m_t(x; θ)/Z_t(θ)` is
the normalized retained density, then

- `dot ρ_t = dot m_t / Z_t - m_t dot Z_t / Z_t^2`.

In the active code this is the same structural role as
`_normalized_retained_log_density_derivatives_chunked(...)`.

## Final score formula

The analytic score for the active finite program is

- `∇_θ log p̂(y_{1:T} | θ) = Σ_t dot λ_t`

with

- `dot λ_0` from the one-axis branch,
- `dot λ_t` for `t >= 1` from the adjacent-state branch,
- and the retained marginal derivative propagated forward between steps.

## Exact assumptions

- fixed branch only;
- same active value program;
- no moving basis;
- same UKF-frozen initial and adjacent core identity;
- static horizon / order / degree / rank / dtype as currently configured;
- no target swap to the transformed-SV independent-panel analytic lane.

## Reused helper algebra

The following helper families are reusable:

- `fixed_design_lsq_derivative(...)`
- `squared_tt_log_normalizer_derivative(...)`
- `_normalized_retained_log_density_derivatives_chunked(...)`
- the explicit adjacent-target derivative helpers already present in
  `filtering.py`.

## New ingredient required

The active SVX route still needs:

> design-aware derivative propagation through the adjacent-state ALS sweep.

## Non-claims

- This note does not claim the backend is implemented.
- This note does not claim HMC readiness.
- This note does not claim equivalence to the transformed-SV analytic route.
