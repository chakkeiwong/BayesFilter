# SVX adjacent-state analytic derivative derivation

Date: 2026-08-06

## Goal

Derive the analytic score backend required by the active `SVX-ZC` NeuTra/HMC
route without changing the value program. The current active value program is
left unchanged; only the HMC-facing score backend is being replaced.

This note is a math/audit artifact, not an implementation file.

## Active value program to differentiate

Let the unconstrained source coordinates be
\[
\theta = (\theta_\gamma, \theta_\beta).
\]
The physical parameters are
\[
\gamma(\theta_\gamma) = 0.1 + 0.8\Phi(\theta_\gamma),\qquad
\beta(\theta_\beta) = 0.1 + 0.8\Phi(\theta_\beta),
\]
with derivatives
\[
\frac{\partial \gamma}{\partial \theta_\gamma} = 0.8\phi(\theta_\gamma),
\qquad
\frac{\partial \beta}{\partial \theta_\beta} = 0.8\phi(\theta_\beta).
\]

The active transformed observation is
\[
z_t = \log(y_t^2).
\]
The observation residual used by the code is
\[
r_t(x_t;\theta) = z_t - 2\log \beta - x_t.
\]
The observation log density contribution is
\[
\ell_{\mathrm{obs}}(x_t;\theta)
= \frac12 r_t - \frac12 e^{r_t} - \frac12\log(2\pi).
\]
Therefore
\[
\frac{\partial \ell_{\mathrm{obs}}}{\partial \beta}
= \frac{e^{r_t}-1}{\beta},
\qquad
\frac{\partial \ell_{\mathrm{obs}}}{\partial \theta_\beta}
= \frac{\partial \beta}{\partial \theta_\beta}\frac{e^{r_t}-1}{\beta},
\qquad
\frac{\partial \ell_{\mathrm{obs}}}{\partial \theta_\gamma}=0.
\]

The stationary variance is
\[
v_\infty(\gamma)=\frac{1}{1-\gamma^2}.
\]
So the time-0 latent Gaussian log density is
\[
\ell_{\mathrm{init}}(x_0;\theta)
= -\frac12\log(2\pi) + \frac12\log(1-\gamma^2) - \frac12(1-\gamma^2)x_0^2.
\]
Hence
\[
\frac{\partial \ell_{\mathrm{init}}}{\partial \gamma}
= \gamma\left(x_0^2 - \frac{1}{1-\gamma^2}\right),
\qquad
\frac{\partial \ell_{\mathrm{init}}}{\partial \theta_\gamma}
= \frac{\partial \gamma}{\partial \theta_\gamma}\gamma\left(x_0^2 - \frac{1}{1-\gamma^2}\right).
\]

The transition log density is
\[
\ell_{\mathrm{tr}}(x_t,x_{t-1};\theta)
= -\frac12\log(2\pi) - \frac12(x_t - \gamma x_{t-1})^2.
\]
Thus
\[
\frac{\partial \ell_{\mathrm{tr}}}{\partial \gamma}
= (x_t-\gamma x_{t-1})x_{t-1},
\qquad
\frac{\partial \ell_{\mathrm{tr}}}{\partial \theta_\gamma}
= \frac{\partial \gamma}{\partial \theta_\gamma}(x_t-\gamma x_{t-1})x_{t-1},
\qquad
\frac{\partial \ell_{\mathrm{tr}}}{\partial \theta_\beta}=0.
\]

## Time-0 target

At time 0, the code forms a weighted target on the one-axis grid:
\[
L_0^{(n)}(\theta) = \ell_{\mathrm{init}}(x_0^{(n)};\theta)
+ \ell_{\mathrm{obs}}(x_0^{(n)};\theta)
+ c_{\mathrm{coord}}.
\]
The shift is
\[
c_0(\theta)=\max_n L_0^{(n)}(\theta),
\]
and the square-root target is
\[
q_0^{(n)}(\theta)=\exp\left(\frac12\bigl(L_0^{(n)}(\theta)-c_0(\theta)\bigr)\right).
\]
Assuming the argmax is locally stable,
\[
\dot c_0 = \dot L_0^{(n^*)},\qquad n^* = \arg\max_n L_0^{(n)}.
\]
Therefore
\[
\dot q_0^{(n)} = \frac12 q_0^{(n)}\bigl(\dot L_0^{(n)} - \dot c_0\bigr).
\]
This is the same pattern used in the repository’s adjacent-target derivative
helpers for the initial step.

## Time-0 one-axis LSQ fit

Let the one-axis basis matrix be \(B\), with quadrature weights \(W\).
The fit solves a ridge LSQ problem with normal equations
\[
A c = b,
\]
where
\[
A = \tilde B^\top W \tilde B + \Lambda,
\qquad
b = \tilde B^\top W q.
\]
At time 0, the design matrix is independent of \(\theta\). Therefore
\[
\dot A = 0,
\qquad
\dot b = \tilde B^\top W \dot q_0.
\]
So
\[
A\dot c = \dot b,
\qquad
\dot c = A^{-1}\dot b.
\]
The one-axis normalizer is
\[
Z_0 = c^\top c,
\qquad
\dot Z_0 = 2 c^\top \dot c.
\]
Hence the time-0 score contribution is
\[
\dot\lambda_0 = \dot c_0 + \frac{\dot Z_0}{Z_0}.
\]

## Time-\(t\ge 1\) adjacent-state target

For \(t\ge 1\), the active program carries the previous fitted marginal density
\(\hat p_{t-1}(x_{t-1};\theta)\) forward.
The target log value on the two-axis grid is
\[
L_t^{(n)}(\theta)
= \log \hat p_{t-1}(x_{t-1}^{(n)};\theta)
+ \ell_{\mathrm{tr}}(x_t^{(n)},x_{t-1}^{(n)};\theta)
+ \ell_{\mathrm{obs}}(x_t^{(n)};\theta)
+ c_{\mathrm{coord}}.
\]
Therefore
\[
\dot L_t^{(n)}
= \dot{\log \hat p}_{t-1}(x_{t-1}^{(n)})
+ \dot\ell_{\mathrm{tr}}^{(n)}
+ \dot\ell_{\mathrm{obs}}^{(n)}.
\]

Again with shift
\[
c_t(\theta)=\max_n L_t^{(n)}(\theta),
\qquad
q_t^{(n)}(\theta)=\exp\left(\frac12\bigl(L_t^{(n)}(\theta)-c_t(\theta)\bigr)\right),
\]
we get
\[
\dot q_t^{(n)} = \frac12 q_t^{(n)}\bigl(\dot L_t^{(n)} - \dot c_t\bigr).
\]

## Time-\(t\ge 1\) two-axis ALS fit

The active two-axis fit alternates left/right TT core solves. For one ALS
subproblem, the weighted ridge LSQ system has the generic form
\[
A c = b,
\qquad
A = D^\top W D + \Lambda,
\qquad
b = D^\top W q.
\]
Unlike the time-0 case, here the design matrix \(D\) depends on the current and
previous fitted cores, so it depends on \(\theta\).

Therefore
\[
\dot A = \dot D^\top W D + D^\top W \dot D,
\qquad
\dot b = \dot D^\top W q + D^\top W \dot q_t.
\]
So the coefficient derivative must satisfy
\[
A\dot c = \dot b - \dot A\,c.
\]
This is the key missing formula relative to the scalar fixed-design helper.

The adjacent-state code updates both cores over the sweep order
\((0,1,1,0)\), so the derivative must be propagated through the same ALS sweep
in the same order.

## Time-\(t\ge 1\) two-axis normalizer

Let the fitted cores be \(L\) and \(R\). The normalizer is the two-axis inner
product quantity used by the code:
\[
Z_t = \sum_{r,s} \langle L_r,L_s\rangle\langle R_r,R_s\rangle.
\]
Define Gram blocks
\[
M^{(L)}_{rs}=\langle L_r,L_s\rangle,
\qquad
M^{(R)}_{rs}=\langle R_r,R_s\rangle.
\]
Then
\[
Z_t = \sum_{r,s} M^{(L)}_{rs}M^{(R)}_{rs},
\]
so
\[
\dot Z_t
= \sum_{r,s}\dot M^{(L)}_{rs}M^{(R)}_{rs} + M^{(L)}_{rs}\dot M^{(R)}_{rs},
\]
with
\[
\dot M^{(L)}_{rs}=\langle \dot L_r,L_s\rangle + \langle L_r,\dot L_s\rangle,
\qquad
\dot M^{(R)}_{rs}=\langle \dot R_r,R_s\rangle + \langle R_r,\dot R_s\rangle.
\]
Hence
\[
\frac{d}{d\theta}\log Z_t = \frac{\dot Z_t}{Z_t}.
\]
The time-
\(t\) score contribution is therefore
\[
\dot\lambda_t = \dot c_t + \frac{\dot Z_t}{Z_t}.
\]

## Recursive propagation of the retained marginal

The active route feeds the fitted marginal forward to construct the next target.
So we also need the derivative of the retained density values used in
\(\log \hat p_t(x_t;\theta)\).

This is the same structural role played by the repository helper pattern in
`_normalized_retained_log_density_derivatives_chunked(...)`, but here it must be
applied to the active adjacent-state finite program.

So the derivative recursion must carry forward:

- fitted cores,
- retained density values,
- and their score derivatives,

from time \(t\) into the target at time \(t+1\).

## Final score formula

The analytic score for the active finite program is
\[
\nabla_\theta \log \hat p(y_{1:T}\mid\theta)
= \sum_{t=0}^{T-1}\dot\lambda_t,
\]
where

- \(\dot\lambda_0\) is the one-axis initial-step contribution;
- \(\dot\lambda_t\) for \(t\ge 1\) is the adjacent-state contribution using
  the derivative-aware ALS solve, the derivative of the retained marginal, and
  the derivative of the two-axis normalizer.

## What is already reusable in the repo

Reuse / adapt rather than re-derive from scratch:

- `fixed_design_lsq_derivative(...)`
- `squared_tt_log_normalizer_derivative(...)`
- `_normalized_retained_log_density_derivatives_chunked(...)`
- the explicit adjacent-target derivative helpers in `filtering.py`

But the active SVX case still needs the new ingredient:

> design-aware derivative propagation through the adjacent-state ALS sweep.

## Non-claims

- This note does **not** claim the analytic backend is implemented.
- This note does **not** claim the transformed-SV fixed-branch analytic route is
  the same target; diagnostics showed it is not.
- This note does **not** change the active target signature.
- This note does **not** assert HMC readiness.

## Why this derivation matters

The active SVX target is not a generic scalar transformed-SV route. Its first
nontrivial recursive dependence is the adjacent-state frozen-core branch.
That is why the convenient analytic route fails and why the missing artifact is a
true adjacent-state analytic derivative.
