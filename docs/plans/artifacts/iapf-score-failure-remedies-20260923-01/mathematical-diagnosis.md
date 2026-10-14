# Why the Gaussian score fit failed

The failures concern our independent R score-regression reconstruction on the
paper's linear Gaussian model, with 80 latent coordinates and 100 observations.
They do not establish a failure of the original iAPF implementation, KDM, or
LEDH. Here “score” means differentiation with respect to the latent state used
to fit a backward guide, not differentiation of the likelihood with respect to
model parameters. The numerical call chain is
`iapf_iterate -> recorded fit callback -> iapf_score_backward ->
iapf_score_target -> iapf_score_fit_cloud`.

## The positive floor changes the curvature

At time t, the score fitter approximates the backward target

\[
b_t(x)=g_t(x)\{s(x)+c\},\qquad
s(x)=\varphi(m;Ax,S),\qquad S=Q+V,
\]

where g is the Gaussian observation density, (m,V) are the fitted next-time
guide parameters, and c>0 is that guide's constant floor. The convolution with
the Gaussian state transition produces S=Q+V. The floor remains c because the
transition density integrates to one. This is the target actually evaluated by
`iapf_backward_log_target` and differentiated by `iapf_score_target`.

Write

\[
G=H^T R^{-1}H,\quad B=A^T S^{-1}A,\quad
u=A^T S^{-1}(m-Ax),\quad r=\frac{s(x)}{s(x)+c}.
\]

The observation contribution has precision G. The future score is r u: it
looks like the Gaussian score near the future guide, but it disappears where
the constant dominates. Since grad(log s)=u and grad(r)=r(1-r)u,

\[
\nabla\log b_t(x)=H^TR^{-1}(y_t-Hx)+r u,
\]
\[
-\nabla^2\log b_t(x)
=G+rB-r(1-r)uu^T.
\]

The first two terms are positive definite here because G=I. The last term is
negative semidefinite and can dominate along u. Thus a positive floor in a
Gaussian guide can make the backward target **non-log-concave**. The observation
Gaussian makes b integrable; it does not guarantee log-concavity everywhere.
This conclusion follows by differentiation, not from the stochastic comparison.

For a one-dimensional example, take G=B=1 and a point with u=5 and r=1/2.
The negative Hessian is 1+1/2-25/4=-4.75. Such a point is possible with A=S=1
and a floor equal to the Gaussian density at that point. A positive floor need
not be large in absolute units to create the problematic curvature.

There is a particularly simple exact test in this model. Put M=I+rB. Congruence
by M^(-1/2) transforms the negative Hessian into
I-r(1-r)M^(-1/2)u u^T M^(-1/2). It is positive definite exactly when

\[
1-r(1-r)u^T(I+rB)^{-1}u>0.
\]

The diagnostic evaluates this condition on every point of each failing cloud,
checks an explicit eigenvalue at a violating point, and verifies the score and
Hessian-vector formula by central differences at two step sizes. These are
tests of the actual target evaluated by the existing code.

## Why unconstrained score regression cannot always return a Gaussian

The code centers the particle cloud and whitens it with its empirical
covariance C=LL^T:

\[
z_i=L^{-1}(x_i-\bar x),\qquad g_i=L^T\nabla\log b_t(x_i),\qquad
N^{-1}\sum_i z_i z_i^T=I.
\]

A Gaussian score in these coordinates has the affine form a-Pz with P positive
definite. The current regression minimizes squared score error over symmetric
P without a positive-definiteness constraint. Its solution is

\[
a=\bar g,\qquad
P=-\operatorname{sym}\left\{N^{-1}\sum_i(g_i-\bar g)z_i^T\right\}.
\]

This formula follows by differentiating the least-squares objective, using the
zero mean and identity covariance of z. Symmetrization enforces a gradient
field; it does **not** enforce positive precision. With the nonlinear floor
score, the empirical cross-covariance can yield a negative eigenvalue. Then
exp(a^T z-z^T Pz/2) grows quadratically in an offending direction and cannot be
normalized as a Gaussian. Rejecting that fit is correct. Bypassing the guard
would be wrong relative to the claimed Gaussian guide.

Non-log-concavity alone does not prove every finite-cloud fit is indefinite.
Here both properties were checked separately. On the exact replayed clouds:

| Label | Backward time | Smallest fitted precision eigenvalue | Non-log-concave points / 1000 | One negative-Hessian eigenvalue |
|---|---:|---:|---:|---:|
| 291 | 78 | -0.2680 | 145 | -17.02 |
| 300 | 81 | -1.1302 | 289 | -17.04 |
| 694 | 79 | -0.1402 | 219 | -16.07 |
| 722 | 78 | -0.1652 | 162 | -17.49 |

The cloud covariance margins range from .056 to .324, versus a guard tolerance
of 1.42e-11. The precision margins range from -.015 to -.595. These are substantial
negative directions on full-rank clouds, not marginal floating-point failures.

As a controlled intervention, keep each failing cloud and the next guide's
mean/covariance fixed, and remove its floor only from the fitting target. The
score becomes exactly affine, with J=I+B positive definite. The same regression
then recovers J^(-1) and J^(-1)(y+A^T S^(-1)m) to at most 1.43e-14 absolute error.
Together with the exact baseline replay, this isolates the interaction between
the floor and the Gaussian regression; it does not show that the floor is
unnecessary for robust importance sampling.

A continuation in the floor also exposes a trap. For label 291, lowering log c
by 2,4,8 makes the precision margin more negative before a reduction by 32 gives
a valid fit. The floor changes which points are in the transition between its
constant and Gaussian components. There is no monotone rule that a modestly
smaller floor always repairs the regression. The complete curves are saved in
each `mechanism_*/floor-continuation.csv`.

## Why four otherwise valid learners hit the cap

The controller tests the sample coefficient of variation of the latest six
likelihood estimates, with the required threshold .5. It uses scaled
exponentials of log likelihoods, preserving the CV while avoiding underflow.
All four capped histories have an unusually high estimate at zero-based
iteration 6. This estimate remains in the six-value window through iteration 11.

For illustration, if one estimate is r times each of the other five, elementary
sample-variance algebra gives

\[
\operatorname{CV}=\frac{\sqrt{6}|r-1|}{r+5}.
\]

Thus one high value can keep the test above .5 after subsequent estimates have
settled. This illustration explains persistence; the saved histories, not the
equal-low-values approximation, determine the actual stopping decisions.

The old loop `0:(max_iterations-1)` with max_iterations=12 executes iterations
0 through 11. Iteration12 is the first window that excludes the high value at 6.
Rerunning with cap 18 produces exactly the same first 12 histories and fitted
guides, and **all four stop at iteration 12 with N=2000**. The fresh final APF
still runs independently, as required by Algorithm 4. The paper's stopping rule
has no 12-iteration ceiling; the ceiling was our compute bound. Raising it
repairs these four truncations without changing the stopping criterion. It
does not guarantee termination for every future seed.

## Remedies and what they establish

The evaluated diagnostic hybrid first attempts the original score backward fit.
Only on a recorded rejection does it use the existing strict diagonal QR
log-regression fitter on the same clouds for the whole backward pass. It then
resumes score fitting at subsequent iterations. All four rejected cases need
one such rescue and complete at iteration 7, N=2000. Both the local QR fit on
the exact failing target and the complete QR backward fit are valid in these
four cases. The positive floor and actual APF weighting/integrals stay in place.

This is an explicit change in the guide-fitting procedure, not a proof that
unconstrained score regression is valid. QR also has its own fail-closed
concavity/rank guards; if those fail, this hybrid fails. The code introduces no
eigenvalue clipping or implicit ridge. Healthy-output identity is checked in
the fresh paired consumer runs; likelihood accuracy is a separate question.

Other possible remedies include a positive-definite constrained score fit, a
different guide family that represents the floor transition, or a Gaussian
analytic warm start for this tractable model. None is established by this
experiment. In particular, constraining precision changes the optimization
problem and needs a declared criterion; removing the floor changes tail
behavior. Those changes must not be described as harmless numerical repairs.

The original paper uses a density least-squares fit (GJL 2017, Section 5.1,
Equation 15), followed by the positive floor in Equation16. Our score fitter
optimizes a different quantity. These findings diagnose the local extension;
the authors' precise solver and full published-result replication remain open.

Source anchors: `reference_iapf_paper.R` (`iapf_backward_log_target`,
`iapf_iterate`), `diagnostic_iapf_paper_score.R` (score, whitening and rejection),
`reference_iapf_author_choices.R` (strict QR fit), and the local GJL paper,
Algorithm 4 and Section 5.1. Per-case contexts, exact checks and finite differences
are retained alongside this note; the terminal result records the fresh probes.
