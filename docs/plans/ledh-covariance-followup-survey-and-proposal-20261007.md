# Covariance-preserving particle filtering: follow-up literature and a general proposal

Date: 2026-10-07. Checkout: sqmc-development, base 209223fdd.
Question: how should we use UKF covariance, preserve the cloud's covariance,
retain safe skewness/kurtosis repair, and obtain useful likelihoods and scores
across models?

This is a literature survey and an unimplemented algorithm proposal. It makes
no new empirical comparison and changes no runtime or default. The earlier SIR
diagnostic campaign remains closed. The proposal is a synthesis, not a claim
that any cited paper proves this complete algorithm.

## What the literature changes

The UKF should guide a proposal. Importance weighting should determine the
posterior moments represented by the particles. A second-order ensemble
transform should preserve those weighted moments when it produces an equally
weighted cloud. These are three different operations with different targets.

There are useful developments after Acevedo, de Wiljes and Reich (2017).
They address different weaknesses, rather than supplying one universally
improved replacement.

| Work and primary source | Technical contribution inspected | Consequence for this project |
|---|---|---|
| [Acevedo, de Wiljes and Reich, 2017](https://arxiv.org/abs/1608.08179), Sections 3–6 and 7.3 | A correction to an ensemble transform matches the importance-weighted mean and covariance. Their bounded-support example also exposes failure of support preservation. | Preserve weighted moments; do not call them an independent covariance oracle. |
| [Corenflos et al., 2021](https://proceedings.mlr.press/v139/corenflos21a.html), Algorithm 3, Section 4 and Proposition 4.3 | Entropy-regularized OT gives differentiable resampling. Ignoring resampling dependencies gives incorrect gradients; consistency results need substantial assumptions. The finite likelihood estimate is biased. | Differentiate the reset and retain a covariance correction. Differentiability alone does not establish likelihood or score accuracy. |
| [Popov, Subrahmanya and Sandu, 2022](https://npg.copernicus.org/articles/29/241/2022/), Sections 4–6 | Rejuvenation augments the ensemble with synthetic anomalies carrying external climatological covariance, using shrinkage. | Missing directions may require additional information or particles. This changes the represented ensemble; it is not exact recovery of an already represented covariance. |
| [Spantini, Baptista and Marzouk, 2022](https://arxiv.org/abs/1907.00389), Sections 3.3, 3.7 and Appendix A | Nonlinear triangular transport maps approximate conditional distributions beyond Gaussian updates. | A principled longer-term way to improve distributional shape. Fitting error, complexity and differentiation through fitting remain costs. |
| [Csuzdi, Törő and Bécsi, 2024](https://arxiv.org/abs/2402.16639), Sections III–V | Optimal-placement resampling supplies another differentiable construction. The inspected method is explicitly one dimensional. | Relevant to scalar diagnostics; not a ready replacement for an 18-dimensional filter. |
| [Ebeigbe et al., 2025](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.054135), Sections IV–V | Generalized unscented points encode marginal third/fourth moments. Bound constraints can force loss of exact higher-moment matching. | Keep bounded repair and report its residual. A weighted sigma-point result is not a theorem for arbitrary equal-weight particle clouds or all mixed moments. |

Corenflos and Popov are particularly direct continuations of the ensemble
transform discussion. The nonlinear-map, optimal-placement and GenUT papers
are related developments, not all direct extensions of the 2017 algorithm.
The older [unscented particle filter](https://proceedings.neurips.cc/paper/1818-the-unscented-particle-filter.pdf)
provides the established principle behind our proposed UKF role: use UKF
information in the proposal and correct using the actual importance ratio.

The strongest practical lesson is that moment preservation repairs a
resampling operation. It cannot repair an importance sample that has already
missed a posterior mode or a tail. Likewise, a UKF is a Gaussian moment
approximation; its quadrature accuracy is not a theorem that its filtering
covariance equals the nonlinear posterior covariance.

This is a focused survey through 2026-10-07, not a complete census. Separate
source, metadata, backward/forward snowball, claim and omission ledgers are in
the accompanying sources JSON. They distinguish inspected papers from leads.

## Which covariance belongs where?

Let the incoming equal-weight cloud be a_1,...,a_N. It represents the previous
filtering distribution. Define

\[
 f_j(x)=f_\theta(x\mid a_j),\qquad
 p_N^-(x)=\frac1N\sum_{j=1}^N f_j(x).
\]

The incoming cloud may depend on theta; that dependence must be carried into
the score. It is held fixed only while defining the conditional integration
problem at this observation.

| Symbol | Conditioning and meaning | Use in the proposal |
|---|---|---|
| C_j | Var(X_t given X_{t-1}=a_j); Q_j for additive Gaussian process noise | Ancestor-centred local flow |
| P^- | Covariance of the whole predictive mixture, including spread between ancestor means | Global UKF update and common proposal-map whitening |
| P_U^+ | UKF approximation to the whole filtered covariance | Common proposal-map recolouring |
| P_w | Covariance represented by the final importance-weighted children | Target of the equal-weight reset |

If m_j and C_j are available, total covariance gives

\[
 m^-=\frac1N\sum_jm_j,\qquad
 P^-=\frac1N\sum_j\{C_j+(m_j-m^-)(m_j-m^-)^\top\}.                 \tag{1}
\]

This is a general mixture identity. It is not specific to SIR. With additive
Gaussian transitions these are exact moments conditional on the incoming
cloud. If conditional transition moments need approximation, an unscented
calculation can provide proposal guides; exactness of (1) for the actual
transition must then not be claimed. Sampling and density evaluation still
use the actual transition law.

For an AR(1) with phi=0.9 and Q=1 at stationarity, the whole predictive
variance is 100/19, consisting of between-ancestor variance 81/19 plus
conditional variance 1. Using C_j=1 in an ancestor-centred flow does not erase
the 81/19. It remains in the positions and weights of the ancestors.

The proposed global guide starts each observation from the particle-based
predictive moments (1). It does not independently propagate an auxiliary
Gaussian covariance and then declare it to be the actual cloud covariance.
This is a substantive proposed change from the existing covariance lifecycle,
not a claim of conformance with the current canonical implementation.

## How the UKF variance actively enters

Apply the UKF observation update to (m^-,P^-). Writing its predicted
observation covariance as S_y and state/observation cross covariance as C_xy,

\[
 K_U=C_{xy}S_y^{-1},\quad
 m_U^+=m^-+K_U(y-\bar y),\quad
 P_U^+=P^- - K_U S_y K_U^\top.                                  \tag{2}
\]

The appropriate augmented-noise calculation is needed for a nonadditive
observation model. These moments guide sampling; the true likelihood g remains
the density used in importance weights.

Factor P^-=L_-L_-^\top and P_U^+=L_+L_+^\top and define

\[
 A_G=L_+L_-^{-1},\qquad
 T_G(x)=m_U^+ + A_G(x-m^-).                                     \tag{3}
\]

Both variances are now used explicitly. P^- scales the full predictive cloud
to standardized coordinates; P_U^+ sets the proposed filtered spread.
If X actually has moments (m^-,P^-), then

\[
 E[T_G(X)]=m_U^+,\quad
 \operatorname{Cov}(T_G(X))
 =L_+L_-^{-1}P^-L_-^{-\top}L_+^\top=P_U^+.                     \tag{4}
\]

Equation (4) needs no Gaussian assumption. It establishes only the first two
moments of this proposal branch. In the ideal linear-Gaussian case with a
Gaussian predictive distribution, the map also gives the correct posterior
distribution. A finite predictive mixture need not be Gaussian, even for a
linear model, so that special case is not a finite-N exactness claim.

A covariance factorization must be valid. Rank deficiency or a failed
positive-definiteness check is a recorded limitation, not permission to add an
undeclared ridge. Any ridge changes this guide and its derivatives and needs
an explicit numerical policy.

A UKF-only design has a concrete blind spot. For stochastic volatility,
Y=exp(X/2) epsilon with independent zero-mean epsilon gives
Cov(X,Y)=0, although Y's magnitude is informative about X. An ideal
moment-based Kalman gain is zero. This is why the proposed mixture retains a
component using the full likelihood; the UKF branch is not a universal
replacement for the existing likelihood-aware proposal.

## One mixture of proposals, one importance correction

Construct three kinds of affine maps:

1. T_0,j(x)=x, the unflowed transition branch.
2. T_G,j=T_G, the same whole-cloud UKF map for every ancestor.
3. T_L,j(x)=A_L,j x+b_L,j, the local LEDH map, using the ancestor's
   conditional prior mean m_j and covariance C_j.

Global and local maps must be constructed before drawing the particles to
which their density formula is applied. Use deterministic anchors or a
separate pilot design. If a map is fitted to its own random input, treating
its fitted A as the entire Jacobian is generally wrong. The candidate must
either avoid that dependence or evaluate the full induced density.

For b in {0,G,L}, let T_b,j(x)=A_b,j x+c_b,j. Its density is

\[
 q_{b,j}(x)=
 \frac{f_j(A_{b,j}^{-1}(x-c_{b,j}))}{|\det A_{b,j}|}.             \tag{5}
\]

With frozen nonnegative beta_b summing to one and beta_0>0,

\[
 q(x)=\frac1N\sum_j\sum_b\beta_b q_{b,j}(x),\qquad
 r(x)=g_\theta(y\mid x)\frac{p_N^-(x)}{q(x)}.                   \tag{6}
\]

Use exactly this mixture in both sampling and the denominator. Sample ancestor
indices uniformly from the equal-weight input cloud and branch labels with
probabilities beta, using fixed random streams for evaluations at different
theta. Fixed stratified branch allocations are also possible, but the
denominator weights must match the actual allocation fractions. Do not use
a density corresponding to a different sampler.

There is one change-of-variables and mixture-density implementation with
different map parameters. The identity, common and local maps do not require
three separate importance-weight algorithms.

The observation contributes

\[
 \Delta\widehat\ell=\log\left(\frac1N\sum_i r(x_i)\right),\quad
 w_i=\frac{r(x_i)}{\sum_k r(x_k)}.                              \tag{7}
\]

The UKF's Gaussian observation normalizer is not multiplied into (7).
The observation informs the proposal and appears in the actual importance
ratio; multiplying another approximate likelihood would count it again.

For any fixed incoming cloud and proposal construction,

\[
 E_q[r(X)]=\int g_\theta(y\mid x)p_N^-(x)\,dx.                  \tag{8}
\]

Thus a poor UKF guide does not redefine this integration target. It can still
make finite-sample estimates unusable. The unflowed branch provides a limited
but useful guarantee:

\[
 q(x)\geq\beta_0p_N^-(x),\qquad
 0\leq r(x)\leq g_\theta(y\mid x)/\beta_0.                     \tag{9}
\]

A bounded likelihood gives a bounded importance ratio. This does not prove
large ESS, good finite-sample coverage, bounded scores, or small cumulative
filtering error. There is no universal beta_0 in this proposal; choosing it
requires the authorized scope-specific tuning procedure.

Although (8) is a valid one-step conditional identity, deterministic moment
resets change the finite filtering approximation. It does not follow that the
product of increments is an unbiased likelihood estimator for the original
state-space model.

## Preserve the moments supported by the weighted children

Compute

\[
 \mu_w=\sum_iw_ix_i,\qquad
 P_w=\sum_iw_i(x_i-\mu_w)(x_i-\mu_w)^\top.                      \tag{10}
\]

The reset target is (mu_w,P_w), not (m_U^+,P_U^+). If the data-weighted
particles disagree with the UKF guide, forcing them back to P_U^+ would
overwrite the importance update.

With X=[x_1,...,x_N], W=diag(w), choose an ensemble transform Dhat with

\[
 \widehat D1=Nw,\quad \widehat D^\top1=1,\quad
 (\widehat D-w1^\top)(\widehat D-w1^\top)^\top
      =N(W-ww^\top).                                         \tag{11}
\]

Then Z=X Dhat has equal-weight mean mu_w and covariance P_w, using the
empirical denominator N. This follows by multiplying the last equality on
the left by X and on the right by X^\top/N. The 2017 second-order ETPF
constructs a correction to a first-order transport to meet this requirement.

For a Sinkhorn transform D and B=D-w1^\top, a symmetric correction Delta
with zero row/column sums obeys

\[
 B\Delta+\Delta B^\top+\Delta^2
       =N(W-ww^\top)-BB^\top.                                \tag{12}
\]

An actual solver must check mean, mass and covariance residuals. The
mathematical requirement (11) does not prescribe importing the paper's
iteration counts or tolerances.

Our current Contract E Cholesky reset has the same intended weighted
moment target but is a different numerical construction. This proposal
does not replace it with an untested Riccati solver. First test the full
current reset, including ridges and subsequent higher-moment operations,
against (10). Preserve the canonical route unless a reviewed candidate
comparison supports a change.

Matching covariance is not always compatible with support and fixed N.
For example, a nonnegative weighted population placing 0.99 at 0 and 0.01
at 100 has mean 1 and variance 99. Two equally weighted nonnegative particles
with mean 1 have variance at most 1. No equal-weight two-particle repair can
meet all three requirements. The 2017 paper's support warning has a genuine
mathematical obstruction behind it.

## Retain bounded skewness and kurtosis repair without losing covariance

This part is a locally derived candidate, not a result proved for our filter
in the surveyed papers.

Let E=Z-mu_w 1^\top. Change only its orientation in particle-index space:

\[
 Z_{\rm new}=\mu_w1^\top+EO,\qquad
 OO^\top=I,\quad O1=1.                                      \tag{13}
\]

The mean is unchanged because E1=0. The covariance is unchanged because
EOO^\top E^\top/N=EE^\top/N. Higher moments can change. A useful smooth
parameterization is a centred skew matrix K,

\[
 K^\top=-K,\quad K1=0,\qquad
 O=(I-K/2)^{-1}(I+K/2).                                    \tag{14}
\]

For real skew K, I-K/2 is nonsingular, O is orthogonal and O1=1.
Furthermore,

\[
 \|O-I\|_2\leq\|K\|_2,\quad
 \|E(O-I)e_i\|_2\leq\|E\|_2\|K\|_2.                         \tag{15}
\]

Consequently a trust bound on K controls displacement while preserving the
first two moments. Existing coordinate and standardized-displacement caps
should remain explicit constraints; (15) is a useful sufficient bound, not
a substitute for checking every required constraint. Low-rank centred
skew parameterizations avoid introducing a dense cubic particle-space solve.

Fit selected marginal and pairwise third/fourth moment features to the
weighted source features, minimizing their scaled residuals within those
caps. An identity rotation is feasible whenever the initial reset cloud
meets the constraints. If exact higher-moment matching is infeasible, retain
the best feasible correction under the declared solver, including identity,
and report the residual. Do not enlarge the cap to force a match.

A finite equal-weight cloud cannot reproduce every third/fourth moment
tensor. A weighted sample with poor coverage is also a poor higher-moment
target. GenUT's marginal-moment formulas neither remove these obstructions
nor establish that this rotation candidate will improve our filtering
likelihood or score. The first implementation stage should test the
proposal repair without adding this new optimizer.

## Complete candidate algorithm

Assumptions: continuous states, evaluable transition and observation
densities, differentiable reparameterized transition sampling for the
score, available observation moments/simulation for the UKF guide, and
valid invertible maps. Model support must be respected by the propagated
cloud; unrecorded clipping is not a repair. This is not a universal algorithm
for arbitrary discrete or constrained state spaces.

Controls, random streams, branch probabilities, pilot designs, transport
iterations and moment-repair solver iterations are frozen for each validated
execution scope. Analytical tangents accompany every operation.

~~~text
inputs:
    theta, observations y[1:T], equal-weight prior particles a[1:N]
    fixed ancestor/branch/noise/pilot streams
    frozen beta[0,G,L], flow/UKF/reset controls, dual caps
    optional higher-moment stage (off until separately evaluated)

ell = 0
score = 0
carry analytic da/dtheta from prior construction

for t = 1,...,T:
    # Predict a mixture; keep local and whole-cloud covariance distinct.
    for j = 1,...,N:
        m[j], C[j] = conditional_transition_moments(theta, a[j])
    m_minus = mean_j m[j]
    P_minus = mean_j(C[j] + outer(m[j]-m_minus, m[j]-m_minus))

    # UKF supplies a whole-cloud proposal guide.
    m_Uplus, P_Uplus = UKF_observation_update(m_minus, P_minus, y[t])
    L_minus = checked_cholesky(P_minus)
    L_plus  = checked_cholesky(P_Uplus)
    A_global = L_plus * inverse(L_minus)
    T_global(x) = m_Uplus + A_global * (x - m_minus)

    for j = 1,...,N:
        T[0,j] = identity
        T[G,j] = T_global
        T[L,j] = frozen_local_LEDH_affine_map(
                     mean=m[j], conditional_covariance=C[j],
                     observation=y[t], independent_pilot=pilot[t,j])

    p_predictive(x) = mean_j transition_density(x | a[j], theta)
    q(x) = mean_j sum_b beta[b] *
           transition_density(inverse(T[b,j])(x) | a[j], theta) /
           abs(det(T[b,j].linear_part))

    # Fixed streams retain the same labels/noises at nearby theta.
    for i = 1,...,N:
        j = uniform_ancestor_index(stream[t,i])
        b = branch_label(beta, stream[t,i])
        u = transition_sample(a[j], theta, fixed_noise[t,i])
        x[i] = T[b,j](u)
        log_r[i] = log_likelihood(y[t] | x[i], theta) +
                   log(p_predictive(x[i])) - log(q(x[i]))

    delta_ell = logsumexp(log_r) - log(N)
    w = softmax(log_r)
    ell += delta_ell
    score += sum_i w[i] * TOTAL_ANALYTICAL_DERIVATIVE(log_r[i])

    mu_w, P_w, M3_w, M4_w = weighted_moments(x, w)

    z = differentiable_OT_reset_with_covariance_correction(
            source=x, weights=w, target_mean=mu_w, target_cov=P_w)

    if the separately validated higher-moment stage is enabled:
        E = z - mu_w
        O = bounded_centred_orthogonal_fit(
                E, targets=(M3_w,M4_w), dual_caps, fixed_solver_iterations)
        z = mu_w + E * O

    record weight, support, moment, displacement and derivative diagnostics
    check declared validity conditions and reset residuals
    a, da/dtheta = z, TOTAL_ANALYTICAL_DERIVATIVE(z)

return ell, score, diagnostics
~~~

The notation inverse denotes a linear solve in implementation. Mixture
densities use log-sum-exp and the prescribed chunk policy. Pilot/map
construction must include the derivatives of its actual operations.

The score of the finite program is

\[
 \nabla_\theta\widehat\ell
 =\sum_t\sum_i w_{t,i}\,D_\theta\log r_{t,i}.                  \tag{16}
\]

D_theta includes the incoming particles, transition samples, UKF moments,
both map types, inverse maps, determinants, both mixture sums, reset moments,
transport and any higher-moment fit. Stopping derivatives through any of
these while calling (16) a total score is wrong.

For example, D log p_N(x)=sum_j rho_j D log f_j(x), where
rho_j=f_j(x)/sum_k f_k(x), and the analogous proposal responsibilities
cover both j and b. The derivatives include the movement of x and a_j.
Differentiating only the generating component gives the wrong derivative
of the marginal ratio.

Matching a finite-difference derivative of this program verifies its
derivative implementation. It does not establish agreement with the true
model score. Fixed iteration counts help define a reproducible program,
but active constraints, factorization margins and guard boundaries still
require a smoothness audit before HMC use.

## Skeptical review and next discriminating work

The proposal survived an algebraic review only with the following corrections:
the full predictive covariance includes between-ancestor spread; the common
map must share one global centre; the local maps must use conditional
covariances; sampling and mixture denominators must agree; UKF normalization
must not be counted twice; map fitting must not hide a random-input Jacobian;
the posterior reset must target weighted particles; and downstream higher
moment repair must retain the covariance invariant.

The remaining material risks are explicit:

- An uninformed or misleading UKF guide remains possible, especially in
  variance-only observations and multimodal filtering.
- The defensive component does not defeat high-dimensional importance
  sampling or ensure a good realization at finite N.
- Covariance correction can violate support or be infeasible at fixed N.
- Conditional moment approximations, ridges, mixture probabilities, trust
  radii and iteration budgets are hypotheses requiring scope-specific
  justification and tuning; none is promoted here.
- An orthogonal repair preserves covariance but may not improve the desired
  higher moments enough, and its constrained differentiation is new work.
- Exact all-component mixture evaluation remains quadratic in N without
  further structure; memory tiling does not remove this arithmetic cost.
- Published filtering RMSE/CRPS or training demonstrations do not establish
  our likelihood or analytical-score accuracy.

The next implementation plan should stage the changes:

1. Audit the current generating maps and marginal density call chain. Repair
   the local conditional-covariance/centering mismatch while retaining the
   current reset. Compare the existing canonical route, ordinary bootstrap,
   UKF, a UKF-guided importance filter, and the repaired local route.
2. Evaluate the common UKF and unflowed components through the same density
   evaluator. Use a fixed-design identity test, a linear-Gaussian limiting
   case, and all-component density/derivative checks before a campaign.
3. Measure the current reset's mean and covariance residuals after every
   downstream operation. Compare second-order constructions only if an
   actual defect or scientific need justifies changing the canonical reset.
4. Evaluate the orthogonal higher-moment repair separately. Its safety
   criterion is non-harm on healthy trajectories and bounded, flagged
   behaviour otherwise. Higher-moment residuals are explanatory; final
   likelihood and every score component decide scientific usefulness.
5. Conduct separate, disjoint tuning/validation/claim runs for LGSSM, KSC SV,
   range bearing, predator-prey and SIR at the required horizons. Include
   linear regimes, heteroscedastic observations, poor coverage and boundary
   regimes. Preserve oracle and independently constructed bootstrap/Zhao-Cui
   comparisons with uncertainty and full parameter-coordinate reporting.

The exact-linear reference and elementary filters are falsification
comparators, not tuning targets. A no-oracle weight-variance or predictive
diagnostic can nominate settings, but cannot certify the unknown likelihood
or score. Compare untouched reference errors and paired uncertainty before
calling any candidate better. The existing invalidated historical evidence
must not be reused as a baseline.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Use UKF covariance as proposal information | Algebra of (3)–(4) checked | Invalid covariance/map is a veto | Quality of UKF guide | Bounded implementation plan and proposal ablations | UKF is an oracle |
| Preserve importance-weighted covariance | Algebra of (10)–(12) checked | Support, residual and rank failures matter | Current full reset conformance | Audit actual endpoint and residuals | Correct weighted moments imply correct posterior |
| Keep bounded higher-moment repair | Invariant in (13)–(15) checked | Cap/support violation vetoes acceptance | Feasibility and finite-program derivatives | Separate optional candidate evaluation | Exact fourth-moment recovery or tail accuracy |
| Promote the combined filter | Not evaluated | Model/reference/score gates untested | Multi-step bias and Monte Carlo error | Cross-model, scope-specific campaign after its plan | Non-regression, HMC readiness or default readiness |

No stochastic ranking is supported by this survey. The papers' demonstrations
are evidence within their reported settings; they do not rank our unimplemented
composite. No numerical tests were run for this proposal.

## Source and code inspection limits

Local copies of every paper used substantively are under .localresources/papers.
The author FilterFlow plan/Sinkhorn sources and GenUT ensemble source were
inspected for the relevant operations. For Spantini et al., the author
StochasticMapFilter source explicitly fits a map to simulated observation/state
pairs, conditions through map inversion, and assimilates observations serially
(lines 141–188, 201–243 in the saved source). This is code corroboration, not an
execution or parity test.

Popov's article links a dated DATools release and Zenodo archive; retrieval
returned unavailable/rate-limited responses in this audit. Its code has not
been inspected. No verified original-author implementation of the 2017
Riccati correction was obtained. No local source-faithfulness or full
consumer-call-chain claim is made for any proposed implementation.

The post-review concern most likely to overturn this recommendation is that
the extra proposal components consume substantial computation without improving
held-out likelihood/score error over the simpler conditional-flow repair.
The staged plan is designed to detect that before adding the higher-moment
optimizer or changing a default.
