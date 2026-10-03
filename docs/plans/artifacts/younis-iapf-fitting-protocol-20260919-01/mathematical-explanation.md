# What the current iAPF score experiment can and cannot prove

2026-09-19. The recent experiment tests a **locally adapted iterated auxiliary
particle filter (iAPF), followed by a Fisher-identity score estimate and linear
variance-reduction controls**. Its measured outcome is error in the gradient
of the observed-data log likelihood. It is not a test of the filtered state
mean, and this particular experiment does not execute KDM or canonical LEDH.

The most firmly supported explanation is substantial finite-particle sampling
noise. There is also a demonstrably poor proposal fit on dataset 1900. We can
prove defects in the old stopping criterion and limitations of the fitting
objective, but the existing experiments do not prove that the poor fit causes
1900's remaining observed UKF loss. Even that remaining risk ordering is
unresolved statistically.

## The quantity being estimated

The scalar model has Gaussian initial, process and observation noise, but
nonlinear conditional means:

\[
X_t=aX_{t-1}+c\sin X_{t-1}+\sigma_Q\epsilon_t,
\qquad
Y_t=HX_t+bX_t^2+\sigma_R\eta_t.
\]

The experiments use two observations, six physical parameter directions,
and either \((c,b)=(.12,.04)\) or \((.35,.12)\). Nominal values are
\(a=.62\), \(H=1.035\), \(\sigma_Q=e^{-.8}\), \(\sigma_R=e^{-.6}\),
and \(X_0\sim N(.175,1.15e^{-.6})\). The actual six-coordinate parameterization
and floating-point values are saved in the run sources and manifests.

Write \(\pi_\theta(x)=p_\theta(x_{0:T}\mid y_{1:T})\). The target is

\[
S(\theta)=\nabla_\theta\log p_\theta(y_{1:T})
=\int \pi_\theta(x)\,s_\theta(x)\,dx,
\qquad
s_\theta(x)=\nabla_\theta\log p_\theta(x,y).
\tag{1}
\]

Differentiating the marginal integral gives the integral of the joint density
times its log derivative; division by the marginal likelihood gives (1).
The Gaussian model has fixed support and the required smooth, integrable
derivatives in this scope. The code analytically accumulates \(s_\theta\)
along particle genealogies, with latent states held fixed in each density
partial. At the final time it computes

\[
\widehat S_N=\sum_{i=1}^{N}\bar w_i s_\theta(X_{0:T}^{(i)}),
\qquad
\widetilde S_N=\widehat S_N-B^\mathsf{T}C.
\tag{2}
\]

There are twelve ancestor controls and six innovation controls. Coefficients
\(B\) are fitted using independent calibration replicates and frozen before
assessment. The executed caller reads the kernel's Fisher output, `out[3]`;
the separately saved derivative `out[1]` of the fixed-branch particle program
is **not** the score used in the reported MSE. This call-chain distinction was
checked against the executed source snapshots.

## Why a likelihood-oriented iAPF need not give a precise score

The iAPF attempts to construct backward information functions. With transition
density \(f\), observation density \(g_t\), and
\(F\psi(u)=\int f(u,x)\psi(x)\,dx\), the ideal recursion is

\[
\psi_T^*(x)=g_T(x),\qquad
\psi_t^*(x)=g_t(x)F\psi_{t+1}^*(x).
\tag{3}
\]

A fitted twist steers the transition through
\(q_t^\psi(x\mid u)=f(u,x)\psi_t(x)/F\psi_t(u)\).
Its corrected incremental potential is
\(g_t(x)F\psi_{t+1}(x)/\psi_t(x)\). Substituting (3) makes that potential
constant. This cancellation motivates iAPF's likelihood-variance objective.

Guarniero, Johansen and Lee's Proposition 2 states the ideal likelihood result;
Proposition 3 and its appendix analyze likelihood variance. **Section 4,
immediately after equation (14), explicitly distinguishes that optimum from
the optimum for general smoothing expectations.** The score in (1) is such an
expectation. This qualification is in the original method, not an inference
that iAPF has been refuted by these tests. See the
[locally stored paper](../../../../.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf).

A direct calculation shows why. Even if independent exact posterior paths
were available, their sample-average score would satisfy

\[
\operatorname{Cov}(\widehat S_N)
=\frac{1}{N}\operatorname{Cov}_{\pi_\theta}(s_\theta).
\tag{4}
\]

Thus a perfect likelihood calculation does not make the score's Monte Carlo
error zero. This oracle calculation is a counterexample to that implication;
the actual resampled genealogies need not have the independence in (4).

More generally, for independent self-normalized importance sampling from a
positive proposal \(q\), set \(W=\pi_\theta/q\). Since \(E_qW=1\),

\[
\widehat S_N-S
=\frac{N^{-1}\sum_i W_i(s_i-S)}{N^{-1}\sum_iW_i}.
\]

The law of large numbers and a finite-second-moment central limit theorem give
the asymptotic squared-error coefficient

\[
V_S(q)=\int\frac{\pi_\theta(x)^2}{q(x)}
                  \|s_\theta(x)-S\|^2\,dx.
\tag{5}
\]

For likelihood estimation the analogous relative-variance coefficient is
\(\int\pi_\theta^2/q-1\), minimized by \(q=\pi_\theta\). Applying
Cauchy--Schwarz to (5) instead gives

\[
V_S(q)\ge
\left(\int\pi_\theta(x)\|s_\theta(x)-S\|\,dx\right)^2,
\qquad
q_{\rm score}^*(x)\propto\pi_\theta(x)\|s_\theta(x)-S\|.
\tag{6}
\]

Equality uses that proposal where it is positive; support constraints can
require a limiting approximation. The score-optimal proposal generally differs
from the likelihood-optimal proposal. Equations (5)--(6) explain the objective
mismatch; they are not an asserted variance formula for the full resampled
implementation or a practical algorithm requiring the unknown exact score.

There is another implementation qualification: our initial \(X_0\) law is
twisted by importance weighting and resampling a finite prior cloud. Even
with exact later twists, the code's likelihood would retain the initial
sample mean \(N^{-1}\sum_i F\psi_1^*(X_0^{(i)})\), whose variance is
\(\operatorname{Var}_{p(X_0)}[F\psi_1^*(X_0)]/N\).
The paper's ideal initial-law assumptions must not be transferred silently to
this local initialization.

## Two limitations that can be proved exactly

The original stopping configuration used two successive positive likelihood
estimates, \(a,b\), with threshold 100. Their sample coefficient of variation
is

\[
\mathrm{CV}=\frac{|a-b|/\sqrt2}{(a+b)/2}
=\sqrt2\frac{|a-b|}{a+b}\le\sqrt2<100.
\tag{7}
\]

The controller therefore always accepts its stability comparison once its
minimum iteration condition holds. This is a proof about the actual old
configuration. It cannot certify convergence. The .05/four-estimate window
can reject instability, but that still does not certify a good proposal or
a precise score. Its downstream validation produced no supported gain.

The local fitting objective is also weaker than the downstream requirement.
It fits a Gaussian shape \(\phi_\eta\) to
\(\widehat h_t=g_tF\widehat\psi_{t+1}\) on the particle cloud, with bounded
standardized center and scale. Its relative-shape loss is

\[
R_\nu=1-
\frac{\langle\phi_\eta,\widehat h_t\rangle_\nu^2}
     {\langle\phi_\eta,\phi_\eta\rangle_\nu
      \langle\widehat h_t,\widehat h_t\rangle_\nu},
\tag{8}
\]

where \(\nu\) is the empirical fitting measure. The positive floor is added
after this fit. The independent reported residual instead uses the predictive
measure, the Gaussian **plus floor**, and the grid approximation to the
**exact** backward information function \(\psi_t^*\). These differ in measure,
fitted function and, for earlier times, backward target. A small empirical loss
or projected gradient is not an error bound for the independent residual.

This failure of implication is mathematical. On any region containing no
fitting points, add a smooth nonnegative bump to a possible target function.
Its values at all fitting points, loss, gradient and stopping decision remain
identical, while its integral and independent approximation error can change
substantially. Without coverage and regularity bounds there is no deterministic
population-error guarantee from those finite evaluations alone. This argument
does not assert that the fixed target on 1900 is an arbitrary bump.

Even a small *population* shape residual does not generally control the relevant
importance variance. Consider a two-region example: the predictive measure
has \(\mu(A)=\delta\), the true twist is 1, and the fitted twist is
\(\epsilon\) on \(A\) and 1 outside. Let
\(Z=1-\delta+\delta\epsilon\). Then

\[
R_\mu=1-\frac{Z^2}{1-\delta+\delta\epsilon^2},\qquad
q(A)=\frac{\delta\epsilon}{Z},
\]
\[
\chi^2(\mu\|q)
=Z\left(1-\delta+\frac\delta\epsilon\right)-1.
\tag{9}
\]

For \(\delta=.01\), \(\epsilon=10^{-6}\), the shape residual is less than
.01 while the relative importance-weight variance is about **9,900**. The
score-like expectation of \(1_A\) has asymptotic variance coefficient about
9,703 under this proposal, versus .0099 under direct sampling from \(\mu\).
Taking \(\delta=\sqrt\epsilon\to0\) makes the residual tend to zero and
the importance variance diverge. The exact rational check is saved in
[mathematical-checks.json](mathematical-checks.json).
This proves that the metric alone is insufficient. It is not a reconstruction
of 1900 or a claim that every bounded Gaussian fit admits this example.

## The poor proposal on 1900 and the limits of its explanation

The archived first-step fit uses 16 fitting particles. Its independent shape
residual is .996807, with a predictive average pointwise floor fraction of
.997756. An earlier fit touched a bound; the final empirical loss is about
.4055 despite an almost-zero projected gradient. The projected-gradient test
only asserts approximate first-order stationarity in the constrained empirical
problem. It does not establish a global optimum or predictive accuracy.

The Gaussian-plus-floor twist also gives an exact description of how adaptation
can weaken. For a Gaussian transition \(f=N(m,Q)\), a fitted
\(\psi(x)=\phi_V(x-a)+c_0\), and
\(Z_g=\phi_{Q+V}(a-m)\), the proposal is

\[
q^\psi=\alpha q_g+(1-\alpha)f,
\qquad \alpha=\frac{Z_g}{Z_g+c_0},
\tag{10}
\]

where \(q_g\) is the normalized product of the transition and fitted Gaussian.
If \(Z_g\ll c_0\), adaptation vanishes and the proposal approaches the
untwisted transition. This follows by expanding and normalizing the two terms
of \(f\psi\); it is exactly the mixture computed by the shared kernel.
The saved predictive average of \(c_0/\psi(x)\) is **not** \(1-\alpha\)
averaged over actual ancestors, so its value .997756 does not prove that 99.8%
of executed particles chose the untwisted branch.

A single Gaussian also cannot represent this model's exact terminal twist:
\(\log g_T(x)=\text{constant}-(y-Hx-bx^2)^2/(2R)\) has a nonzero quartic
coefficient for \(b\ne0\), whereas a Gaussian log density is quadratic.
Adding a strictly positive constant floor cannot make the functions
proportional either: the floor remains positive in the tails while the
likelihood tends to zero. This proves exact family mismatch, **not** a lower
bound large enough to explain the measured error on the predictive region.
The second observation-likelihood root can lie deep in the prior tail;
bimodality by itself is not established as the operative failure mechanism.

These results identify plausible routes from a poor fit to large sampling
noise. They do not distinguish sparse-cloud coverage, optimizer failure,
family limitations, and propagation of next-step approximation error on 1900.
Nor was the newly selected fitting protocol refitted on this case in the replay.

## What the reported MSE says

For independent full-run score replicates \(H_r\) and fixed reference \(S^*\),
the exact sample identity is

\[
\frac1n\sum_r\|H_r-S^*\|^2
=\frac{n-1}{n}\operatorname{tr}(\widehat\Sigma)
+\|\bar H-S^*\|^2.
\tag{11}
\]

The variance contribution accounts for 96.6--99.8% of the selected-arm
validation MSE and 97.0% on the 1900 precision replay. Fourfold particle counts
also reduced error on every declared precision comparison. These observations
support the sampling-noise explanation. Equation (11) is an exact identity;
estimating true bias and variance from 64 replicates still has uncertainty.

For frozen \(B\), centered controls satisfy
\(E[H-B^\mathsf{T}C]=E[H]\). Thus they do not remove existing normalization
bias. For a scalar score component and nonsingular control covariance, the
minimum variance over linear coefficients is

\[
\operatorname{Var}(H)-
\operatorname{Cov}(H,C)\operatorname{Var}(C)^{-1}\operatorname{Cov}(C,H).
\tag{12}
\]

Expanding the variance of \(H-b^\mathsf{T}C\), differentiating in \(b\), and
substituting its minimizer gives (12). It is zero only if the centered score is
in the linear span of the controls almost surely. First/second innovation
moments and ancestor controls do not supply that guarantee. For example, the
observation log-noise score contains \((y-Hx-bx^2)^2/R-1\), a quartic function,
before weighting and genealogy introduce further nonlinearities. The exact
centering statements refer to the declared sampling law; floating-point
execution remains covered by numerical checks, not an exact-arithmetic claim.

On dataset 1900, the observed corrected MSE at 16,384 particles is .0150409,
versus UKF's .0138890. Their difference is .0011519. The descriptive standard
error of that sample MSE, computed as the sample SD of its 64 squared errors
divided by eight, is **.0015491**. The observed excess is only .74 such standard
errors. This is not a newly declared hypothesis test, but it makes clear why
an observed-screen failure must not be described as proof of a true risk
ordering. The earlier significant particle-count improvement compared 16,384
with 4,096 particles, not with UKF.

| Conclusion | Evidence status | Consequence |
|---|---|---|
| Old threshold cannot screen instability | Algebraic proof and actual controller | Reject its use as a convergence certificate |
| Likelihood-optimal twist need not be score-optimal | Paper Section 4; derivations (4)--(6) | Validate downstream score accuracy separately |
| Shape fitting/stationarity does not guarantee low importance variance | Exact counterexample and finite-cloud argument | Add diagnostics tied to proposal coverage and score error |
| 1900 has a poor independently evaluated twist | Saved refined-grid diagnostic | Preserve the fit as the case requiring diagnosis |
| Finite-particle noise is substantial | Sample decomposition and paired precision experiments | Remains the strongest experimentally supported mechanism |
| That bad twist causes a true MSE excess over UKF | **Not proved**; true risk ordering unresolved | Retain the declared promotion veto without claiming a causal theorem |

The next useful proof-oriented diagnosis is to reconstruct the saved fitting
cloud and its exact local target, then compare empirical stationarity with an
independent bounded optimization over the predictive measure. Coverage, bounds,
optimization and next-twist approximation must be separated. To certify a
specific cause of excess score risk, we would additionally need a validated
error bound or controlled interventions with uncertainty for the actual
estimator. A small gradient, a high ESS or a lower shape loss is not that proof.

The [audit plan](mathematical-audit-plan.md),
[executable checks](mathematical-checks.py), and
[saved output](mathematical-checks.json) record this analysis. It consumed no
new filter calls, fits, or GPU launches. Numerical artifacts and the existing
promotion vetoes remain unchanged. The argument is Codex self-reviewed;
no independent proof review is claimed.
