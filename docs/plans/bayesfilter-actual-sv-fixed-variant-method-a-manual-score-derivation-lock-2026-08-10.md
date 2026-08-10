# Actual-SV fixed-variant Method A manual score derivation lock

Date: 2026-08-10
Status: `DERIVATION_LOCK_FOR_IMPLEMENTATION`

## Goal

Lock the exact scalar and the exact derivative target for the fixed-variant actual-SV Method A backend before any more code changes.

The target is the **comparator-issued frozen-core batch TT finite program** implemented in:
- `bayesfilter/highdim/zhao_cui_actual_sv_batched_tt_tf.py`
- `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`

This note is not about the transformed-SV independent-panel helper route and not about any refit-at-each-theta surrogate.

## Same-scalar contract

Let the source coordinates be `theta = (theta0, theta1)` and define
- `gamma(theta0) = 0.1 + 0.8 Phi(theta0)`
- `beta(theta1) = 0.1 + 0.8 Phi(theta1)`.

The observations entering the active route are already the transformed values
- `z_t = log(y_t^2)`
for the frozen dataset produced by the comparator path.

The active route uses the residual
- `r_t(x_t; theta) = z_t - 2 log beta(theta1) - x_t`
and observation log density
- `ell_obs_t(x_t; theta) = 0.5 r_t - 0.5 exp(r_t) - 0.5 log(2*pi)`.

The transition density is
- `ell_tr_t(x_t, x_{t-1}; theta) = -0.5 log(2*pi) - 0.5 (x_t - gamma(theta0) x_{t-1})^2`.

The one-time comparator fit freezes only the **initializer cores and adjacent seed cores at theta_0**:
- `initial_core`
- `adjacent_core0`
- `adjacent_core1`

These tensors are constants with respect to later theta evaluation.

However, the runtime batch finite program still performs a deterministic two-step ALS replay at each time step starting from those frozen seed cores. Therefore the active runtime scalar is:
- a frozen-seed, fixed-design, same-program batch TT scalar,
- not a fully frozen-TT-coefficient scalar,
- not a transformed-SV independent-panel helper,
- not an SRUKF / SGQF / CUT4 surrogate.

So the implementation target is:
- **same runtime scalar as `batched_fixed_tt_likelihood_value_trace(...)`**,
- but with a manual score backend instead of TensorFlow autodiff.

## Runtime scalar decomposition

For each batch row `b` and time `t`, the value path in `batched_fixed_tt_likelihood_value_trace(...)` computes
- `increment_{b,t} = log_shift_{b,t} + log normalizer_{b,t}`
with the total value
- `V_b(theta) = sum_t increment_{b,t}`.

### Time 0

At `t = 0`:
- physical nodes are fixed from `reference_nodes` and `COORDINATE_HALF_WIDTH`
- the log target is
  - `L_{b,0,n} = ell_init_b(x_n) + ell_obs_b(x_n) + log(2*half_width)`
- the max shift is
  - `c_{b,0} = max_n L_{b,0,n}`
- the square-root target is
  - `S_{b,0,n} = exp(0.5 * (L_{b,0,n} - c_{b,0}))`
- a one-axis scaled ridge LSQ fit is run starting from `initial_core`
- the fitted coefficients define a one-axis squared-TT normalizer `Z_{b,0}`
- the increment is
  - `increment_{b,0} = c_{b,0} + log Z_{b,0}`.

### Time t >= 1

At `t >= 1`:
- the previous normalized retained density `pHat_{b,t-1}(x_{t-1})` is evaluated on the fixed previous-axis quadrature nodes,
- the adjacent-state log target is
  - `L_{b,t,n} = log pHat_{b,t-1}(x_{t-1,n}) + ell_tr_b(x_{t,n}, x_{t-1,n}) + ell_obs_b(x_{t,n}) + log(2*half_width)`
- the max shift is
  - `c_{b,t} = max_n L_{b,t,n}`
- the square-root target is
  - `S_{b,t,n} = exp(0.5 * (L_{b,t,n} - c_{b,t}))`
- a deterministic two-axis ALS sweep with order `(0, 1, 1, 0)` is run starting from:
  - `frozen_adjacent0`, `frozen_adjacent1` at `t = 1`
  - and then from the previous step’s fitted cores for `t > 1`
- the fitted two-axis cores define the squared-TT normalizer `Z_{b,t}`
- the increment is
  - `increment_{b,t} = c_{b,t} + log Z_{b,t}`.

## Derivative target

The required score is
- `grad_theta V_b(theta)`
for the exact same finite program above.

So the manual derivative must compute
- `dot increment_{b,t} = dot c_{b,t} + dot Z_{b,t} / Z_{b,t}`
with all terms taken on the same active branch.

## Required derivative layers

### 1. Local density derivatives
These are explicit and already available through the model-parameter-score helpers in `filtering.py`:
- initial log density derivative,
- transition log density derivative,
- observation log density derivative.

These provide `dot L` before the square-root target transform.

### 2. Max-shift square-root target derivative
For each fixed local branch,
- `dot c = dot L[n*]` where `n*` is the selected maximizing index,
- `dot S_n = 0.5 * S_n * (dot L_n - dot c)`.

This is exactly the `square_root_target_jvp(...)` rule.

### 3. One-axis ALS derivative
At `t = 0`, the design matrix is fixed by the one-axis basis and does not depend on theta.
So the one-axis fit derivative is the fixed-design rule:
- `N dot v = dot rhs - dot N v`
with only target dependence active in practice.

### 4. Two-axis ALS derivative
At `t >= 1`, the design matrix for each ALS subproblem depends on the current other core, so the design tangent is nonzero in general.
Therefore the correct fixed-branch derivative replay must propagate:
- target tangent,
- current core tangents,
- design tangent,
- normal-equation tangent,
through the exact sweep order `(0, 1, 1, 0)`.

This is the same algebraic obligation as the `fixed_als_value_jvp(...)` / `padded_fixed_als_value_jvp_xla(...)` helpers.

### 5. Squared-TT normalizer derivative
For one-axis and two-axis fitted cores, the normalizer derivative is the squared-TT log-normalizer derivative already formalized in
- `squared_tt_log_normalizer_derivative(...)`.

### 6. Retained marginal quotient-rule derivative
The next-step target depends on `log pHat_{t-1}`. So the derivative must propagate the retained marginal via the quotient rule:
- `dot rho = dot numerator / Z - numerator * dot Z / Z^2`
- `dot log rho = dot rho / rho` on the active positive branch.

This is exactly the role played by
- `_normalized_retained_log_density_derivatives_chunked(...)`
in the scalar fixed-design helper path.

## What earlier procedures got wrong

1. We previously mixed this runtime scalar with nearby transformed-SV helper routes.
2. We previously treated the task as a pure wiring problem, but the missing artifact is a same-program manual derivative replay.
3. We previously used finite-difference failures to discover route mismatch instead of freezing the scalar first.
4. We previously mixed “fully frozen fitted TT” semantics with the actual route, which freezes only the seed cores and then replays the deterministic fit each step.

## Consequence for implementation

The correct implementation strategy is **not**:
- swap in `sv_mixture_cut4.py`,
- reuse SRUKF/SGQF surrogates,
- differentiate a different transformed-SV helper,
- or pretend the stepwise ALS fit has no design dependence.

The correct implementation strategy is:
- preserve `batched_fixed_tt_likelihood_value_trace(...)` as the value authority,
- replace `batched_fixed_tt_likelihood_value_score_status(...)` with a manual forward derivative replay of that exact same finite program,
- likely by reusing and adapting:
  - `square_root_target_jvp(...)`,
  - `fixed_als_value_jvp(...)` and/or `padded_fixed_als_value_jvp_xla(...)`,
  - `squared_tt_log_normalizer_derivative(...)`,
  - `_normalized_retained_log_density_derivatives_chunked(...)` / `squared_tt_normalized_marginal_jvp(...)`.

## Hard non-claims

- This note does not certify the implementation is finished.
- This note does not claim the transformed-SV independent-panel route is equivalent.
- This note does not claim a fully frozen coefficient TT scalar; the active runtime route still replays the deterministic fit from frozen seed cores.
- This note does not claim HMC readiness or posterior correctness beyond the current finite program.
