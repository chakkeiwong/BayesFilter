# Why the fit and prefix diagnostics can fail for different reasons

The fitting calculation and the prefix-error calculation reveal two distinct
problems. The printed fitting objective can reward a Gaussian that puts almost
no density on its training points. Separately, a guide that gives the exact final
likelihood can be a poor proposal for estimating earlier likelihoods. Neither
finding establishes that the authors' implementation failed.

## The density-fitting objective has an escape direction

The inspected primary text is the archived arXiv v2 source,
`docs/plans/artifacts/iapf-r-source-reconciliation-20260920-01/sources/arxiv-extracted/iapf_arxiv.tex`,
lines 655–684, Equation 15 and its implementation discussion. It specifies a
diagonal Gaussian density fit with a free positive scale, then adds a positive
constant to the fitted guide. It does not specify the optimizer or constant's
formula. The following argument concerns that printed objective, not an
uninspected author solver.

Let the positive backward targets on a finite cloud be b_i and write
p_i(m,V)=N(x_i;m,V). The stated objective is

\[
 L(m,V,\lambda)=\sum_i[p_i(m,V)-\lambda b_i]^2.
\]

For any fixed Gaussian, differentiation with respect to the scale gives

\[
 \partial_\lambda L=-2b^T p+2\lambda b^Tb=0,
 \qquad \lambda^*(m,V)=\frac{b^Tp}{b^Tb}>0.
\]

Substitution yields the exact profiled objective

\[
 L_*(m,V)=p^Tp-\frac{(b^Tp)^2}{b^Tb}
         =\lVert p-\lambda^*b\rVert^2.
\]

Now fix m and let V=s²I with s increasing. For every point in a finite cloud,

\[
 p_i=(2\pi)^{-d/2}s^{-d}
 \exp\{-\lVert x_i-m\rVert^2/(2s^2)\}\longrightarrow0.
\]

Consequently 0≤L_*≤p^Tp→0, irrespective of whether the fitted shape resembles
the targets. Sending the mean away from all training points provides another
escape direction. The infimum is therefore zero. If no finite diagonal Gaussian
is proportional to the target vector on this cloud, no finite global minimizer
attains it. If such a Gaussian exists, a zero-loss fit does exist, but the escape
direction can still mislead a local solver. This distinction matters at the last
observation, where the present Gaussian target is exactly representable.

This does not prove that a useful local fit is impossible. It proves that small
absolute density residual alone does not certify useful shape, and that the
printed unconstrained global minimization does not uniquely specify a reliable
numerical procedure. Multiplying the loss by a fixed positive scale or profiling
out lambda preserves this issue. A relative residual changes the objective;
restricting a neighborhood changes the feasible set. Either must be identified
as an additional numerical choice. Our earlier QR fit minimizes squared errors
of log targets and thus computes a different fit.

The new executable tests check the joint/profiled value and gradient identities,
finite differences, an exactly representable Gaussian, fixed scaling, an escape
example, and the actual controller endpoint. Fixed-cloud evidence shows why these
algebra checks are insufficient for a replication claim: a d=80 moment-start
profiled fit reduces scaled loss from 3.76e-4 to 1.53e-36 yet has
KL(exact Gaussian guide || fitted Gaussian component)=96.06 and relative
residual 0.938. That KL concerns the Gaussian component, before the positive floor.
It is a descriptive shape diagnostic, not a likelihood promotion criterion.

Across the full-filter probes, both QR-start variants reach the exact final-time
fit and then fail at backward time 99. Moment-start variants can fail at time 100
itself. The 32 attempts, including the predeclared iteration-limit retries,
contain 25 non-convergence outcomes, 5 domain-boundary outcomes and 2 density
underflows. Their saved diagnostics support local fitting failure. They do not
identify the authors' undocumented settings or prove every optimizer must fail.

## Exact final likelihood does not imply accurate earlier likelihoods

Write Z_t=p(y_1:t), let p_t(x)=p(x_t=x | y_1:t) be the filtering density, and
let B_t(x)=p(y_(t+1):T | x_t=x) be the future-data likelihood. The smoothing
density is

\[
 q_t(x)=p(x_t=x\mid y_{1:T})
       =p_t(x)B_t(x)\frac{Z_t}{Z_T}.
\]

An exact future guide samples this smoothing law. Its final importance weights
are constant, giving the exact Z_T. To estimate an earlier Z_t from the same
particles, however, the future information must be removed. For independent
exactly guided particles, the corrected prefix satisfies

\[
 \frac{\widehat Z_t}{Z_t}
   =\frac1N\sum_{i=1}^N\frac{p_t(X_t^i)}{q_t(X_t^i)},
 \qquad X_t^i\sim q_t.
\]

Its expectation is one. Its variance is

\[
 \operatorname{Var}(\widehat Z_t/Z_t)
 =\frac1N\left[\int\frac{p_t(x)^2}{q_t(x)}\,dx-1\right].
\]

At t=T, p_T=q_T, so the variance is zero. At earlier times q_t can be much
narrower or displaced because it conditions on future observations. Rare points
then receive very large correction weights. Current-time innovation size alone
does not describe that mismatch: the future observations also matter.

For Gaussian p=N(m,P) and q=N(n,Q), define delta=m−n,
M=2P^(-1)−Q^(-1), and h=Q^(-1)delta. With z=x−m, the exponent in p²/q is

\[
 -\tfrac12 z^TMz+z^Th+\tfrac12\delta^TQ^{-1}\delta.
\]

Completing the square gives, when M is positive definite,

\[
 I:=\int p^2/q
 =\frac{|Q|^{1/2}}{|P|\,|M|^{1/2}}
   \exp\{\tfrac12\delta^TQ^{-1}\delta+\tfrac12h^TM^{-1}h\}.
\]

Otherwise the integral diverges. In this linear-Gaussian model,
Q^(-1)=P^(-1)+J where J is the future-likelihood precision; the condition is
P^(-1)−J positive definite. Setting p=q recovers I=1. Equal covariances with
different means give I=exp(delta^TP^(-1)delta). Three independent one-dimensional
quadrature checks validate the finite-integral formula.

The d=80 diagnostic uses the same saved observations as the completed study and
the exact full-covariance guide with no added floor. Sixteen runs check all 1,600
prefix density-ratio identities to 1.06e-11 and final log likelihoods to 5.46e-12.
The analytic moment condition passes at every time; the smallest eigenvalue of
M is 0.946. Nevertheless, for N=1000 the exact expected squared relative prefix
error averages 2769.54 over ordinary-innovation times and 101.42 over large
innovations. These are finite but huge variances. The 16-run empirical averages
are only 2.83 and0.341, illustrating how slowly rare-weight contributions can
become visible. Those small empirical samples do not estimate the tail reliably.

These variance values belong to the exact-guide diagnostic, not to the learned
QR guide with its positive floor. Their purpose is to disprove the implication
that good final-likelihood performance guarantees good prefix performance.
They do not remove the frozen project prefix veto retrospectively. A future
plan should state separately whether it is certifying terminal likelihoods,
online filtering, or both.

The saved QR audit retains every observation and verifies all 1,000 parent-file
hashes. For the six-estimate controller, one replica contributes 83.94% of the
ordinary-period error and time 56 contributes 85.96%; for the five-estimate
extension, one replica contributes 66.09%. The previously reported means and
vetoes remain unchanged. Their paired95% difference intervals include zero,
so the observed losses to simple filters do not establish population inferiority.

Evidence: `summary.json`, `analysis-v1/fit-failures.csv`,
`analysis-v1/prefix-concentration.json`, `solver-tests-attempt01.log`,
`failure-inspection.log`, and preserved per-job inputs, RDS results and CSV files
under `runs/`. No LEDH, KDM, total-gradient or HMC claim follows from this analysis.
