# Independent R implementation of the iterated auxiliary particle filter

This base-R implementation reconstructs the iAPF of Guarniero, Johansen and
Lee (2017). It is an independent numerical reference for BayesFilter, not
original-author code, a canonical TensorFlow implementation, or a score/HMC
method. It supports Gaussian transitions with fixed covariance and a callable
transition mean, and arbitrary observation log densities. The first replication
target is the paper's Section 5.2 linear-Gaussian likelihood experiment.

The implementation is in
[`reference_iapf_paper.R`](../benchmarks/reference_iapf_paper.R). Base R 4.1.2
is sufficient; no external R package is required. It includes backward fitting,
the complete adaptive particle filter, particle-count adaptation, stopping,
a fresh final likelihood estimate, a Kalman comparator and a fully adapted
auxiliary particle filter. Source anchors below refer to the
[locally stored paper](../../.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf).

## The proposal and its correction

Let the transition density be `N(x; a, Q)` and the fitted function be
`psi(x) = N(x; m, V) + c`, with `c > 0`, as in equation (16). Its transition
integral is

\[
 M(a)=\mathcal N(a;m,Q+V)+c.
\]

The twisted proposal `N(x; a,Q) psi(x) / M(a)` is exactly a two-component
mixture. Its original-transition component has probability `c/M(a)`. The other
component is Gaussian, with mean `a + Q(Q+V)^{-1}(m-a)` and covariance
`Q-Q(Q+V)^{-1}Q`. Thus the constant floor must affect sampling and the
normalizer as well as the weight. Adding it only to the weights computes a
different importance sampler with an incorrect proposal correction.

Writing `F_t(x) = integral f(x,z) psi_{t+1}(z) dz`, the incremental potential
is `g_t(x) F_t(x)/psi_t(x)`, with the initial normalizer included at time 1
and `F_T=1`. Multiplying the actual proposal densities by these potentials
telescopes to the original joint density; equations (5)--(6) and Proposition 1
give this construction. Algorithm 5 retains weights between ESS resampling
events and accumulates normalizing factors at those events and termination.
The implementation tests both branches through the actual sampler.

Algorithm 3 fits the backward targets `g_t(x) F_t(x)` on the particle clouds.
Algorithm 4 decides when to stop and increase particle counts. Its final
likelihood estimate comes from a fresh run with the learned twists frozen.
Reusing an adaptation estimate selected by the stopping condition would change
the returned estimator. The zero-based stopping condition is `l > k`.

## The fitting problem exposed by replication

For particle locations `x_i` and positive targets `y_i`, equation (15) specifies

\[
 L(m,V,\lambda)=\sum_{i=1}^N
       [\mathcal N(x_i;m,V)-\lambda y_i]^2.
\]

Let `p_i=N(x_i;m,V)`, `a=p'p`, `b=p'y` and `c_y=y'y`. Differentiating in
`lambda` gives `lambda*=b/c_y`. Substitution gives the profiled loss
`L*=a-b^2/c_y`. The R routine computes this same least-squares problem using
residuals, divides by N, and applies a fixed scale per fit for numerical units.
Neither factor changes the minimizers.

There is an escape direction in this objective. Fix the mean and let
`V=s^2 I`. Every density value satisfies

\[
 0<p_i\leq (2\pi)^{-d/2}s^{-d}.
\]

Because profiling cannot increase the loss relative to `lambda=0`,

\[
 0\leq L^*(m,s^2I)\leq\sum_i p_i^2
       \leq N(2\pi)^{-d}s^{-2d}\longrightarrow0.
\]

The profiled positive `lambda*` also tends to zero. This argument therefore
does not depend on allowing an exactly zero lambda. The objective can become
arbitrarily small by making the Gaussian diffuse, regardless of how well its
relative shape matches the targets. Unless some finite Gaussian achieves an
exact proportional fit, the infimum zero has no finite minimizer. This is a
mathematical property of the written fitting objective, not proof that every
local numerical fit or the iAPF importance-sampling construction fails.

The bounded R runs exhibited this direction. One fit drove squared densities
below floating-point resolution; two others reached variance
`1/.Machine$double.eps`, approximately `4.50e15`, while the optimizer reported
convergence. At the two variance boundaries, relative squared residuals were
0.594 and 0.9998. Small absolute loss was therefore not evidence of a good fit.
The reporting-underflow error was repaired without changing the objective,
gradient, optimizer or returned parameters. The variance-boundary guard remains.

The public R comparator used previously minimizes a different expression,
`||y-p/lambda*||^2 = c_y R/(1-R)`, where `R=L*/a`. That scale-independent
criterion does not share this amplitude escape. It is wrong relative to an
equation-(15) implementation claim, even if it is a useful alternative fitting
criterion. Substituting it here would require labeling a different method.

## An explicit alternative for fitting shape

The gap-repair campaign adds `fit_mode="relative_l2"`; the default remains
`paper_eq15`. The alternative divides the profiled squared error by the squared
norm of the fitted density vector:

\[
 R(m,V)=\frac{\min_\lambda\|p-\lambda y\|^2}{\|p\|^2}
       =1-\frac{(p^\top y)^2}{(p^\top p)(y^\top y)}.
\]

This measures disagreement in shape on the fitting cloud. Multiplying either
vector by a positive constant leaves R unchanged, so shrinking every density
value cannot improve it. For the diffuse Gaussian direction above, all density
ratios approach one and

\[
 R(m,s^2I)\longrightarrow
 1-\frac{(\sum_i y_i)^2}{N\sum_i y_i^2}>0
\]

when the targets are not constant. The strict inequality follows from
Cauchy--Schwarz. It removes the demonstrated amplitude escape, but does not
prove uniqueness, global convergence or good approximation away from the
fitting particles. A diagonal Gaussian also retains a restriction on the
shapes it can represent.

For `s_i=partial log p_i/partial theta`, differentiation gives

\[
 \partial_\theta R
 =\frac{2b^2}{c_y a^2}\sum_i p_i^2s_i
  -\frac{2b}{c_y a}\sum_i p_iy_is_i.
\]

The R implementation uses this analytical derivative and scales density values
inside the evaluation to avoid underflow. The cancellation implied by scale
invariance makes this the derivative of R even when the numerical scale depends
on the parameters. Finite-difference and coordinate-rescaling tests check that
calculation through the actual fitter. The old objective has separate tests.

Both fitting methods feed the same Gaussian-plus-floor proposal and exact
importance correction. Thus the alternative still constructs an iAPF with
positive twists; it does not implement equation (15). The public comparator's
normalized criterion is a monotone transformation of R where R<1, which gives
the same minimizers but does not establish identical numerical trajectories.

The relative fitter rejects nonconvergence and preserves the failed inputs.
Saved-input calibration showed that an allowance of 5000 iterations and
L-BFGS-B memory `max(5,2*d)` resolves six captured fitting failures. These are
documented local solver choices. The strict equation-(15) objective and solver
settings are unchanged; it now rejects nonconvergence and objective underflow.
Floor powers 1,2,3 and the two early-doubling interpretations are explicitly
recorded sensitivity arms. A changed early-doubling interpretation is never
silently described as matching the authors' implementation.

An explicit `relative_l2_nlminb` mode evaluates the same relative objective
with base R's nlminb solver. It resolves seven saved fits, including the first
dimension-20 failure, but fails on a fresh dimension-20 fit. It is an optional
diagnostic alternative, not an automatic fallback or a completed repair.

The first dimension-20 failure has a very concentrated fitting target. Writing
`q_i=y_i^2/sum(y^2)`, its effective number of points `1/sum(q_i^2)` is 1.272
out of 1000. This is a fitting diagnostic, not the filtering ESS. Define
`u=p/||p||`, `v=y/||y||` and residual `e=(I-vv')u`, so `R=e'e`. The residual
Jacobian in mean/log-variance coordinates has a singular-value ratio of
48,446, so the Gauss-Newton approximation `2J'J` has a condition ratio of
about 2.35e9. This is local
numerical conditioning in these coordinates, not proof of rank deficiency.
Independent finite differences agree with the analytical gradient to 2.61e-10.

The next fresh dimension-20 fit also concentrates on about 1.409 effective
points. Both 5000 and 10000 nlminb iterations fail to declare convergence,
despite relative residuals around 2--3e-10. The role of concentration can be
made explicit. Write `r_i=p_i/y_i`. Then

\[
R=1-\frac{(E_q r)^2}{E_q(r^2)}
 =\frac{\operatorname{Var}_q(r)}{E_q(r^2)}.
\]

For a single fitting point this expression is zero for any positive fitted
density value, so the training loss cannot identify the Gaussian shape. The
effective-size calculation is not a rank theorem for the observed clouds, but
independent validation now demonstrates the resulting problem in both cases.
With the next fitted twist frozen, the linear model's actual backward target
is a two-Gaussian mixture up to scale. Its continuous relative error can be
integrated exactly under the exact smoothing Gaussian. The failed fits have
errors 0.472332 and 0.261061, compared with 0.175331 and 0.250284 for their
initializers. Diagonal Gaussians formed from the actual target moments give
0.018480 and 0.017046. Thus optimization reduced training error while worsening
agreement away from the training cloud; the diagonal family alone does not
explain the failure. These moment fits are diagnostic oracles, not runtime
substitutes. The [paper/code audit](../plans/artifacts/iapf-r-paper-code-audit-20260920-01/result.md)
preserves the independent-cloud checks, exact integrals and source trace.

The [repair plan](../plans/iapf-r-reference-gap-repair-2026-09-20.md) separates
these mathematical checks from untouched repeated likelihood comparisons.

## Fitting log targets directly

An additional explicit mode, `log_quadratic`, fits the logarithm of the backward
target. Let `z_j=(x_j-center_j)/scale_j`. Ordinary least squares estimates

\[
\log y_i=b_0+\sum_j(b_jz_{ij}+c_jz_{ij}^2)+e_i.
\]

If the design has full rank and every `c_j<0`, completing the square gives

\[
m_j=\mathrm{center}_j-\frac{\mathrm{scale}_j b_j}{2c_j},
\qquad V_j=-\frac{\mathrm{scale}_j^2}{2c_j}.
\]

The intercept absorbs amplitude. QR solves the stated least-squares problem;
rank-deficient or nonconcave regressions fail explicitly. There is no clipping
or moment fallback in this mode. Every particle contributes to the squared
log error. This changes the fitting objective and can place more emphasis on
tails, so its effect must be checked through actual filtering.

At d20 and T100, four pilot repetitions and 16 repetitions on another untouched
dataset completed without a fit failure. On the second dataset the mean
likelihood ratio to Kalman was 0.989741, with a 95% bootstrap interval
[0.951537,1.029334], passing the declared conditional accuracy screen. A new
validation stage then ran 32 repetitions on each of two further d20 datasets:
the mean ratios were 0.996404 and 1.009890, with intervals [0.976202,1.016881]
and [0.985510,1.033173]. All 38,400 regressions in those batches passed the
strict QR, rank and curvature checks. Neither ordinary nor large-innovation
prefix-error comparisons raised a heuristic veto. No cross-method ranking is
claimed from these conditional fixed-dataset screens. The
[validation result](../plans/artifacts/iapf-r-log-fit-validation-20260920-01/result.md)
records all methods, source hashes, diagnostics and limits. This remains a
viable independent R alternative; it does not implement equation (15).

## What has and has not been replicated

The paper does not give the exact positive floor function or numerical solver
settings in Section 5.1. This reconstruction uses a floor at a Gaussian
chi-square contour with tail probability `N^-2`, diagonal log-quadratic
initialization, and L-BFGS-B with analytical derivatives. Those are documented
choices, not recovered author settings. The observations are newly simulated
from the published model; the original observations and random seed were not
available among the inspected sources.

There is also an early-iteration interpretation to retain: this reconstruction
first allows doubling when a full history window exists at zero-based `l=k`,
while the written stopping rule first permits stopping at `l=k+1`. The paper
does not define negative history indices. This ordering can cause an early
doubling and differs from the published average final N=1000 at dimensions
5--20. The recorded N histories make that difference visible; reproducing the
authors' numerical behavior requires resolving this detail as well.

The test suite checks Gaussian products, the floor mixture, path telescoping,
Kalman against a separately constructed joint Gaussian likelihood, both
resampling branches, fitting derivatives, T=1, stopping, doubling and the
fresh final estimate. Deliberate source mutations verify that tests reject
the wrong floor sampling, stopping index and fitting objective.

At T=100 and dimensions 5, 10, 20, 40 and 80, exact full-covariance backward
twists agree with Kalman to a maximum absolute log-likelihood error of
`1.82e-12` over three particle seeds per dimension. The fully adapted filter
also agrees with the observation-twist implementation under matched seeds.
These controls bypass estimated diagonal twists. They support the particle
filter calculations but do not reproduce the fitted-iAPF results.

The strict equation-(15) comparisons stopped on fitting failures. The subsequent
explicitly different relative-L2 implementation completed 32 repetitions each
at d5 and d10. Its mean likelihood ratios to Kalman are 1.0011 and 1.0044;
their bootstrap 95% intervals, [0.9897,1.0131] and [0.9886,1.0204], pass the
predeclared [0.9,1.1] accuracy screen. Conditional on those two observed datasets,
the variance-ratio intervals against the fully adapted filter are
[0.1212,0.4587] and [0.0575,0.1963]. These support lower likelihood variance
at the specified unequal budgets, not greater efficiency or general superiority.
Ordinary and large-innovation prefix-error checks produce no heuristic veto.

The [repair result](../plans/artifacts/iapf-r-reference-gap-repair-20260920-01/result.md)
preserves the earlier attempts, uncertainty and d20 relative-solver failures.
The current suite passes 88 reference checks, 33 alternative-fit checks and
executed-source mutation/snapshot tests. The validation stage adds 84 complete
filter repetitions and 50,400 fitted regressions. At d40, a four-repeat pilot
had mean ratio .875362, while an independent 16-repeat batch had mean ratio
1.004931 with interval [.961774,1.051352]; the pilot is descriptive and is not
pooled with the longer batch. The d80 pilot did not finish the controller's
20-iteration cap: all 2,000 fits were admissible, but the last-six likelihood
coefficient of variation was 1.401898 at N=8,000, above the .5 stopping
threshold. An exact replay reproduced the status, likelihood history, particle
counts and fits, and a full-covariance exact-twist control matched Kalman with
zero recorded log error. The original power2 d80 setting is rejected in this
scope. The subsequent [floor diagnosis and repair](../plans/artifacts/iapf-r-log-fit-validation-20260920-01/floor-result.md)
identifies the cause: the guide's constant floor overwhelms its overlap with
the transition at observation93, sending every particle through the
unguided component in two saved failures. For psi=N(m,V)+c that component has
probability c/[c+N(m;a,Q+V)]; its denominator, rather than the guide's peak,
determines whether a numerically small c is harmless.

An explicitly optional power8 floor, selected by analytic mixture-probability
calibration, completes two repaired and four fresh d80 repetitions without
extending the controller caps. All use eight adaptive passes and2000 final
particles. The fresh mean likelihood ratio is1.088298; four repetitions are
too few to establish accuracy. Healthy exact parity,1700 Gaussian-limit tail
checks,9000 fit checks and eight d20 non-harm repetitions pass. Ordinary d20
prefix MSE increases descriptively25.6-36.3%, within the declared factor2
screen, so even non-harm should not be described as statistical equivalence.
The default objective and floor remain unchanged. Larger independent
replication, all five published1000-repeat studies and the source ambiguities
remain open. This evidence does not yet establish a general BayesFilter
debugging authority. The driver executes captured source trees, preserves
failed attempts and bounds descendant processes as well as their supervisor.

The subsequent [frozen positive-floor validation](../plans/artifacts/iapf-r-positive-floor-validation-20260920-01/result.md)
completed32 repetitions on each of two untouched d80 datasets and a fresh d40
control. Mean likelihood ratios to Kalman were1.020141,.981628 and1.041181,
with conditional bootstrap95 intervals[.975004,1.073665],[.927681,1.036752]
and[1.005136,1.078378]. Each interval lies within the declared[.9,1.1]
accuracy screen. All64100 fits and9600 final-guide Gaussian-limit tail checks
pass; there is no observed conditional heuristic veto or controller-cap failure.
The d40 interval excludes1, which motivated the independent confirmation below.
These96 repetitions qualify the optional reference for
further validation in this checked model; they do not establish unbiasedness,
paper Eq.15 replication or a general method ranking. The default objective and
floor are unchanged, and author solver/floor/data reconstruction gaps remain.

The [d40 confirmation](../plans/artifacts/iapf-r-d40-confirmation-20260920-01/result.md)
then completed 32 fresh repetitions on those same observations and 32 on new
observations. The respective mean likelihood ratios were 0.967538 and 1.022861,
with bootstrap 95% intervals [0.932404, 1.002977] and [0.982292, 1.061802]. Both
passed the unchanged accuracy screen, with no fit, tail or conditional heuristic
veto. The earlier upward discrepancy did not recur. These intervals concern
finite samples on two datasets and cannot establish exact unbiasedness.

The accompanying [mathematical audit](../plans/artifacts/iapf-r-d40-confirmation-20260920-01/likelihood-audit.md)
derives likelihood preservation for the positive-floor proposal and both
adaptive-resampling branches. Given the learned guides and stopping history,
the fresh final evaluation has conditional expectation equal to the original
likelihood in ideal arithmetic. A strengthened executable test rejects returning
the stopping estimate even when a fresh evaluation was performed. The remaining
paper-replication task is to reconcile the fitting objective and missing author
settings; the successful optional log-quadratic fit does not implement Eq.15.

## Running the reference

From the repository root, the deterministic checks use:

```sh
CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  Rscript --vanilla tests/reference_iapf_paper.R "$PWD" /tmp/iapf-checks.csv
```

For a small interactive reference calculation:

```r
source("docs/benchmarks/reference_iapf_paper.R")
data <- iapf_paper_data(d=5, horizon=100, seed=58000005)
model <- iapf_linear_model(data$observations, data$A)
exact <- iapf_kalman(model)
set.seed(123)
control <- iapf_apf(model, iapf_exact_twists(model), N=1000)
control$log_likelihood - exact$log_likelihood
```

The September 21 continuation closes the missing same-setting checks at
dimensions 5, 10 and 20. Each new 32-repeat dataset passes the likelihood
accuracy screen, all fitted-regression and Gaussian-limit tail checks, and
the conditional heuristic comparisons. Their mean likelihood ratios are
0.997983, 1.000777 and 0.996912, respectively. Together with the separately
preserved d40/d80 runs, the optional log-quadratic/floor8 reference has bounded
evidence across all five dimensions. This still differs from equation 15 and
from the paper's 1000 repetitions per dimension. The reconstructed controller
also uses 2000 final particles in the new d5/d10 runs, whereas the paper
reported an average of 1000. See the
[complete result](../plans/artifacts/iapf-r-small-dimension-completion-20260921-01/result.md).

The source investigation recovered both arXiv manuscript versions but did not
recover the original numerical solver, bounds or positive-floor formula.
A deterministic counterexample now proves why those omissions matter:
unrestricted equation 15 can reduce its loss towards zero by diffusing the
Gaussian density while worsening its relative fit. On a valid correlated
Gaussian fitting cloud, no finite diagonal Gaussian can attain that zero
infimum. This identifies a problem with the written global minimization,
without claiming that the authors' unspecified local optimizer failed. The
[derivation and executed check](../plans/artifacts/iapf-r-source-reconciliation-20260920-01/result.md)
separate those statements.

Shared R and TensorFlow normalizers, mixture proposals, both profiled fitting
objectives and analytical gradients agree on a deterministic FP64 fixture,
with maximum error 6.22e-15. The
[consumer checks](../plans/artifacts/iapf-r-tf-component-parity-20260921-01/result.md)
also confirm that the current density-fit TF endpoint rejects multidimensional
input. Its model fixture and separate full-covariance log-quadratic fitter
differ from this R study. Primitive agreement therefore supplies useful
debugging evidence while a full same-method comparison remains unspecified.

Serious fitted comparisons use the bounded driver and the recorded
[experiment plan](../plans/iapf-r-paper-replication-2026-09-20.md). Runtime,
iterations and particle-count caps produce incomplete statuses; they never
return an adaptation estimate as if it were the final likelihood estimate.

For the explicitly different fitter evaluated in the gap-repair study:

```r
alternative <- iapf_iterate(model, fit_mode="relative_l2", fit_maxit=5000,
                            floor_tail_power=2,
                            doubling_mode="first_full_window")
stopifnot(alternative$status == "complete")
alternative$final$log_likelihood - exact$log_likelihood
```

This example selects the alternative explicitly. A successful single run is a
mechanics check; the repair report gives the separate repeated-run evidence.
