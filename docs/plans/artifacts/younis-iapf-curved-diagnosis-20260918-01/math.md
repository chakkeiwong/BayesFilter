# Why a correct finite-program derivative can be a wrong model score

The model likelihood is Z(theta)=p_theta(y_1:T), and the requested score is
grad_theta log Z(theta). The iAPF consumer returns the derivative of a random
finite-particle log likelihood while holding its fitted Gaussian coefficients,
realized ancestors, and mixture labels fixed. Those two derivatives need not
agree. The implementation explicitly records this distinction; source-code
agreement or a local finite-difference pass does not remove it.

To see the missing contribution, suppose a categorical choice K has
probabilities q_k(theta), followed by an output F_k(theta). Then

\[
\frac{d}{d\theta}\sum_k q_k(\theta)F_k(\theta)
=\sum_k q_k(\theta)\frac{dF_k}{d\theta}
 +\sum_k F_k(\theta)\frac{dq_k}{d\theta}.
\]

Holding the realized label fixed computes the first sum in expectation. It
does not compute the second sum. For example, if q_1=theta, q_2=1-theta,
F_1=1 and F_2=0, the output's expectation has derivative one, whereas every
fixed-label derivative is zero. Increasing the number of such derivatives
reduces sampling noise around zero without recovering the missing one.
This is a local mathematical deduction, not a claim that every particle
filter has the same bias.

There is also a normalization distinction. Even when a corrected likelihood
estimator satisfies E[Zhat]=Z, the physical score is

\[
\nabla\log Z=\frac{\nabla E[\widehat Z]}{E[\widehat Z]},
\]

which generally differs from \(\nabla E[\log\widehat Z]\). Restoring a
sampling-law term for the random log likelihood would address its expected
derivative; that alone does not establish a finite-N unbiased physical score.
Any repair must specify which expectation and normalization it estimates.

The distinction can be made more specific for a two-observation bootstrap
filter. Write X_1(theta,U_1) and X_2(theta,U_1,U_2) as the smooth latent
trajectory generated from continuous base noise. Let

\[
D_1=\frac{d}{d\theta}\log g_\theta(y_1\mid X_1),\qquad
D_2=\frac{d}{d\theta}\log g_\theta(y_2\mid X_2),
\]

where each derivative includes the trajectory's parameter dependence. Under
the usual differentiation-under-the-integral and particle consistency
conditions, the true score is the full-trajectory posterior expectation

\[
s=E[D_1+D_2\mid y_1,y_2].
\]

A filter that resamples after y_1 and then differentiates with those ancestors
fixed instead approaches

\[
s_{\rm fixed}=E[D_1\mid y_1]+E[D_2\mid y_1,y_2].
\]

The second observation changes how the first observation's derivative should
be averaged. Defining m_2(X_1)=p_theta(y_2|X_1), the discrepancy is

\[
s_{\rm fixed}-s
=E[D_1\mid y_1]-E[D_1\mid y_1,y_2]
=-\frac{\operatorname{Cov}(D_1,m_2(X_1)\mid y_1)}
         {E[m_2(X_1)\mid y_1]}.
\]

These are vector identities when theta has several coordinates. They describe
the limiting bootstrap control under the stated regularity assumptions;
finite particle ratios add Monte Carlo error. The iAPF has additional
parameter-dependent initial ancestor and Gaussian/floor mixture choices, so
this bootstrap formula must not be presented as its complete bias formula.
It provides a discriminating control: removing resampling for this short
horizon removes the categorical ancestor contribution, while preserving the
physical model and the smooth trajectory derivative.

The Gaussian fitting question is separate. For the terminal observation in
this scalar model,

\[
\log\psi_T(x)=\log g_\theta(y_T\mid x)
=C-\frac{(y_T-hx-bx^2)^2}{2R}.
\]

Its x^4 coefficient is -b^2/(2R), whereas a Gaussian log density is quadratic.
For nonzero b, an exact single-Gaussian representation on an interval is
impossible. This does not show that the approximation is poor in the region
the filter visits. The campaign therefore evaluates shape error weighted by
predictive probability, checks an independent feasible Gaussian competitor,
and measures downstream likelihood and score errors separately. A nonzero
fitting residual alone cannot identify the cause of an inaccurate score.

The backward recursion and positive-floor Gaussian family come from
Guarniero--Johansen--Lee, Proposition 4, Algorithm 3 and equations (15)--(16).
Their likelihood-preservation result concerns the corrected value estimator.
It does not establish that the derivative obtained by fixing categorical
labels is an unbiased model-score estimator. The local implementation uses
these fixed-label derivatives in `fitted_twist_tf.py::make_fitted_twist_kernel`
and `nonlinear_tf.py::make_particle_filter`; this campaign checks their
consequences without changing their mathematical targets or defaults.
