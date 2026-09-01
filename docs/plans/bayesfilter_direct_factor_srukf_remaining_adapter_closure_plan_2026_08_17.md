# Direct-Factor SR-UKF Remaining-Adapter Closure Plan

Date: 2026-08-17  
Status: reviewed execution contract  
Prior evidence: `docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817/`  
New evidence root: `docs/plans/artifacts/direct-factor-srukf-remaining-adapter-closure-20260817-r4/`

The attempts without a suffix, with `-r2`, and with `-r3` are preserved but superseded.
The first emitted non-standard JSON `Infinity` tokens and imprecise source
anchors.  The second corrected the anchors but did not mechanically apply all
numerical thresholds and did not recursively normalize NumPy scalar infinity.
The `r3` harness then failed before publishing a success summary because its
route-guard return was displaced while adding numerical gates.  The numerical
tensors were unchanged.  Attempt `r4` is the final evidence root.

## 1. Objective and frozen scope

Close, or preserve with a precise blocker, the four `adapter_required` rows in
the 2026-08-17 direct-factor SR-UKF inventory:

1. `lgssm_2d_h25_rich`;
2. `range_bearing_4d_h20_rich`;
3. `predator_prey_rk4`; and
4. `LGSSM-EXACT`.

Closure means that a repository-owned TensorFlow adapter binds the model,
parameter coordinates, factor derivatives, data identity, and comparison
authority.  It does not mean that SR-UKF is applicable to every repository
model.  SSL-LSTM remains owner-excluded.  SGQF, particle/transport filters,
non-additive or non-Gaussian contracts, and singular/rank-changing analytical
scores retain their prior classifications.

This is a bounded CPU reference/diagnostic campaign.  The repository default
execution target remains TensorFlow GPU/XLA.  These results do not establish
GPU capacity, HMC readiness, posterior correctness, or exact nonlinear
Bayesian inference.

## 2. Common direct-factor mathematics

At time `t`, carry a lower factor `L_t` rather than form the covariance

```text
P_t = L_t L_t^T.
```

For augmented state/process coordinate `z=(x,w)` use

```text
L_aug = block_diag(L_t, L_Q),
z_i = z_bar + L_aug xi_i,
x_i^- = f_theta(x_i,w_i).
```

With DZ5 weights, form the weighted residual stack directly,

```text
A_x = [sqrt(W_i^c) (x_i^- - x_bar^-) ]_i,
L^- = lower_qr(A_x),
```

and never assemble `P^- = A_x A_x^T` in the temporal recursion.  For
observation deviations `d_i`, noise factor `L_R`, and state deviations `e_i`,

```text
A_y  = [sqrt(W_i^c) d_i, L_R],
A_xy = [sqrt(W_i^c) e_i, 0].
```

One block QR returns the innovation factor `L_y`, gain `K`, and conditional
factor `L_t`.  The predictive term is

```text
ell_t = -1/2 [m log(2 pi) + 2 sum_j log (L_y)_{jj} + ||L_y^{-1} r_t||^2].
```

All derivatives are derivatives of these finite QR programs on a fixed
positive-pivot/sign branch.  If `dL_y` and `z=L_y^{-1}r`, then

```text
dz = L_y^{-1}(dr - dL_y z),
d ell_t = -sum_j d(L_y)_{jj}/(L_y)_{jj} - z^T dz.
```

Fixed positive-definite covariance inputs may be converted once, before the
filter trace, by Cholesky.  For `C=L L^T`, the factor differential is

```text
E = L^{-1} (dC) L^{-T},
Phi(E) = tril(E) with diagonal divided by 2,
dL = L Phi(E).
```

This one-time adapter construction is permitted; Cholesky, eigendecomposition,
or SVD inside a filter time step is forbidden by the route guard.

## 3. Model-specific contracts

### 3.1 Common V2 LGSSM

Freeze physical coordinates

```text
theta = (a_scale, r_scale), theta_0 = (1,1),
A(theta) = a_scale A_0,
R(theta) = r_scale R_0.
```

`m0`, `P0`, `Q`, `C`, and all offsets are fixed.  Hence

```text
dA/d a_scale = A_0,       dR/d r_scale = R_0,
dA/d r_scale = 0,         dR/d a_scale = 0.
```

The observation factor derivative follows the Cholesky differential above.
Because the transition and observation are affine, DZ5 moment propagation is
exact through second order.  The direct-factor value and score must agree with
the independent linear-Gaussian SVD/Kalman authority at tight FP64 tolerance.

### 3.2 Range/bearing Common V2 model

Freeze physical positive coordinates

```text
theta = (sigma_range, sigma_bearing), theta_0=(0.12,0.04),
L_R(theta) = diag(theta).
```

The transition is `x^- = A x + L_Q w`.  The observation is

```text
h(x) = (sqrt(px^2+py^2+eps), atan2(py,px)).
```

Ordinary Euclidean averaging is invalid for bearing.  For observation sigma
points `y_i=(rho_i,phi_i)`, use

```text
rho_bar = sum_i W_i^m rho_i,
S = sum_i W_i^m sin(phi_i),
C = sum_i W_i^m cos(phi_i),
phi_bar = atan2(S,C).
```

For a parameter derivative,

```text
d phi_bar = (C dS - S dC)/(S^2+C^2),
dS = sum_i W_i^m cos(phi_i) dphi_i,
dC = -sum_i W_i^m sin(phi_i) dphi_i.
```

Both sigma deviations and the observed innovation use

```text
residual(predicted, observed)
  = (observed_range-predicted_range,
     wrap(observed_bearing-predicted_bearing)).
```

On a fixed branch its derivative is the derivative of ordinary subtraction.
The score is inadmissible if either the circular resultant `S^2+C^2` is below
a declared floor or any raw angular residual lies within a declared margin of
`+/-pi`.  The adapter must fail closed in those cases and expose the minimum
branch margin.  Tests must include observations on both sides of the branch
and an explicit near-branch rejection.

The repository's existing covariance UKF reference is comparison evidence,
not an oracle, because it performs covariance stabilization and Cholesky per
step.  The primary score authority is centered finite difference of the same
direct-factor value on an unchanged branch.

### 3.3 Common V2 predator-prey

Bind the exact Common V2 source and data:

```text
model = p30_predator_prey_fixture_model(),
theta = (r,K,a,s,u,v) at model.true_parameters(),
observations = model.simulate(theta, final_time=3, seed=4404)[1][1:].
```

The admitted score coordinate is the declared Common V2 knob `r` only.  It is
not the probit coordinate and not the longer PP-UKF fixture.  The transition
adapter uses `PredatorPreySSM.transition_mean_parameter_jacobian` and selects
row zero from its `[6,R,2]` physical-coordinate Jacobian.  Process,
observation, and initial factors are fixed.  Direct observation has identity
state Jacobian and zero direct parameter derivative.

The three-step value/score must be finite, agree with centered finite
difference in physical `r`, and agree between eager and XLA.  The existing
PP-UKF result is not reused as fixture parity evidence.

### 3.4 LGSSM-EXACT

Load the persisted `T=120`, four-state/four-observation, 18-parameter target
bundle.  Materialize its raw-coordinate tensors and first derivatives exactly
once.  Build factors from

```text
P0(theta), Q(theta), R(theta)
```

and transform their full `[18,4,4]` covariance derivatives with the Cholesky
differential.  The transition and observation derivative contractions use the
materialized `dA`, `dH`, offsets, and initial mean.  The direct-factor filter
uses `R(theta)+10^{-9}I` because that fixed jitter is part of the persisted
exact authority's finite likelihood program; its derivative remains `dR`.
The filter computes likelihood only.  Posterior comparison is then

```text
posterior value = likelihood value + identical Gaussian prior value,
posterior score = likelihood score + identical Gaussian prior score.
```

This separation prevents accidental comparison of a likelihood with a
posterior.  Full 18-dimensional value/score parity is required against the
fixture-bound exact SVD linear-Gaussian authority.  Centered finite differences
cover at least one transition-diagonal, one transition-lower, one process-log-
scale, and one observation-log-scale coordinate.  Eager/XLA parity covers the
full score.

## 4. Implementation steps

1. Extend `TFFactorSRUKFModel` with optional observation-mean and residual
   callbacks.  Extend the derivative contract with the matching mean
   derivative callback.  Defaults reproduce Euclidean subtraction bit for bit.
2. Add fixed-branch telemetry and fail-closed assertions for circular means and
   wrapped residuals.  Aggregate the minimum angular branch margin over time.
3. Add repository testing adapters for the four frozen fixture contracts.
   Reuse the one-time covariance-to-factor conversion and do not import NumPy.
4. Add unit tests for callback defaults, circular mean derivatives, wrapped
   deviations/innovation, branch rejection, adapter tensor orientation,
   Cholesky derivatives, and no decomposition in the temporal body.
5. Add integration tests for each row: value, analytical score, finite
   difference, eager/XLA, finite diagnostics, and positive pivot gates.
6. Run a versioned closure campaign and write machine-readable per-model
   results, a superseding inventory, environment/command manifest, and hashes.
7. Update the execution result and LaTeX survey; compile LaTeX twice and hash
   the PDF, source, log, and evidence tables.

## 5. Acceptance gates

The planned nominal tolerances are ceilings and may be tightened after seeing
reference-scale roundoff, but may not be relaxed silently:

| Gate | LGSSM V2 | Range/bearing | Predator-prey | LGSSM-EXACT |
|---|---:|---:|---:|---:|
| value absolute delta to authority | `1e-9` | diagnostic only | N/A | `2e-8` |
| full score absolute delta to authority | `1e-8` | N/A | N/A | `2e-7` |
| centered-FD score delta | `2e-6` | `2e-5` | `2e-5` | `5e-5` sampled |
| eager/XLA value and score delta | `1e-10` | `1e-10` | `1e-10` | `1e-9` |
| minimum QR pivot | strictly positive | strictly positive | strictly positive | strictly positive |
| factor/derivative residuals | finite | finite | finite | finite |
| circular branch margin | N/A | `>1e-8` | N/A | N/A |

Finite differences are recorded at `h=1e-5` and `h=5e-6`.  The numerical gate
uses the finer estimate after step halving demonstrates convergence; the
coarser estimate remains visible as a truncation-error diagnostic.  This is
material for range/bearing: its errors decrease from about `2.27e-5` to
`5.68e-6`, the expected factor of four for a centered difference.

A failed numerical gate remains an explicit blocker with its raw evidence.  A
row is not relabeled `eligible_score` merely because it executes.

## 6. Tests and corner cases

The focused suite must cover:

- scalar and batch shape rejection;
- non-finite factors and non-positive factor diagonals;
- default Euclidean callback backward compatibility;
- circular mean across `-pi/+pi`;
- bearing residual wrap on both sides of the cut;
- exact or near-cut score rejection;
- near-zero circular resultant rejection;
- finite differences with step halving to detect truncation/roundoff accidents;
- all four LGSSM-EXACT parameter families;
- Common V2 predator-prey data/horizon identity;
- eager and XLA execution;
- source inspection proving no Cholesky/SVD/eigendecomposition call in
  `_one_step` or its QR temporal helpers; and
- superseding inventory consistency and preservation of all non-applicable,
  blocked, historical, and owner-excluded rows.

## 7. Bounded execution and stop conditions

Budget: one focused test run, up to two localized repair reruns, one closure
campaign, and two LaTeX passes; target wall time under 20 minutes on CPU.
Preserve all evidence under the new versioned root and do not overwrite the
prior campaign.

Stop and retain a blocker if:

- a source fixture or parameter convention differs from this plan;
- a circular branch gate trips at the nominal fixture;
- a QR pivot is non-positive or non-finite;
- analytical score disagreement persists after orientation and finite-
  difference checks;
- XLA requires a semantic fallback; or
- closure would require a per-step covariance decomposition.

## 8. Nonclaims

This campaign does not claim universal model applicability, robust analytical
scores across covariance rank changes, exact nonlinear filtering, GPU
production readiness, HMC readiness, or superiority over SVD.  It certifies
four frozen full-rank/fixed-branch adapters if and only if their gates pass.
