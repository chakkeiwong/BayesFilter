---
title: "Bayesian Inference for Nonlinear Models with a Genuine Zero Lower Bound"
subtitle: "A self-contained literature survey and project design note"
date: "18--19 August 2026"
---

**Study dates:** 18--19 August 2026 (Section 4.3 and its sources added 19 August)  
**Working repository:** BayesFilter  
**Application anchor:** `/home/ubuntu/workspace/MacroFinance`  
**Requested integration path (corrected 19 August 2026, second revision):** the originally named path `/home/ubuntu/workspace/dsge_hmc` does not exist, but the `dsge_hmc` package does, at `/home/ubuntu/workspace/python/src/dsge_hmc`. Its validated BGS restricted surface carries the linear placeholder `r = rn` (rows tagged `OBC_ZLB_NO_RUN_GUARD`) and is not a true OBC/ZLB model; see Section 14.

## 1. The question

In a linearized New Keynesian model, a zero lower bound (ZLB), or effective
lower bound (ELB), can be handled by solving two linear systems: one with the
policy constraint slack and one with the constraint binding. OccBin and its
successors iterate over a candidate sequence of regimes, apply a Kalman filter
conditional on that sequence, and verify the inequalities afterwards
(Guerrieri and Iacoviello 2015; Giovannini, Pfeiffer, and Ratto 2021). This is valuable
and fast. It is not a general solution to Bayesian inference in a nonlinear
model, because the conditional Gaussian closure has been used twice: once in
the state transition and again in the likelihood recursion.

The project question is therefore narrower and harder:

> How can we sample a posterior that retains an actual ZLB regime change in a
> nonlinear state-space model, while keeping the transition kernel exact for
> the stated target and making every approximation visible?

The word **discontinuous** needs care. The map

\[
 i_t=\max\{\ell,i_t^\star\}
\]

is continuous and has a kink at \(i_t^\star=\ell\). A model with an explicit
regime variable \(r_t\in\{n,b\}\), or a solver that selects between distinct
equilibrium branches, can instead produce a mixed discrete-continuous target or
a jump in the density. Reflection/refraction HMC, discontinuous HMC, and mixed
HMC address these three geometries differently. Treating all of them as one
kind of nonsmoothness is the first design error.

This survey has four goals. It derives the linear benchmark; derives the HMC
kernels for piecewise-smooth and mixed-support targets; derives the nonlinear
state-space likelihood and its particle approximation; and turns the comparison
into a staged BayesFilter project design. Results stated by a paper are cited to
that paper. Deductions for the proposed project are written in the notation
below and labelled as such.

## 2. A common model and notation

Let \(\vartheta\) denote structural parameters, \(x_t\in\mathbb R^{d_x}\) the
latent state, \(r_t\) the ZLB regime, \(\epsilon_t\) a structural shock, and
\(y_t\) the observation. A general model is

\[
\begin{aligned}
 r_t &= \mathcal R(x_{t-1},\epsilon_t,\vartheta),\\
 x_t &= F_{r_t}(x_{t-1},\epsilon_t,\vartheta),\\
 y_t &\sim g_{r_t}(y_t\mid x_t,\vartheta).
\end{aligned}
\tag{1}
\]

For a monetary lower bound, define a shadow or notional nominal rate
\(i_t^\star=\iota(x_{t-1},\epsilon_t,\vartheta)\) and a bound \(\ell\). The
regime rule is

\[
 r_t=n \iff i_t^\star>\ell,\qquad
 r_t=b \iff i_t^\star\leq\ell,
\tag{2}
\]

and the observed policy rate may be \(i_t=i_t^\star\) in regime \(n\) and
\(i_t=\ell\) in regime \(b\). A complementarity form introduces a multiplier
\(\lambda_t\geq0\):

\[
 i_t-\ell\geq0,\qquad \lambda_t\geq0,\qquad
 (i_t-\ell)\lambda_t=0.
\tag{3}
\]

Equation (3) says which inequality is true, but it does not by itself say how
the rest of the equilibrium is solved. That distinction matters for inference:
the likelihood is defined by the complete transition and observation law, not by
the bound in isolation.

The posterior is

\[
 \pi(\vartheta,x_{0:T},r_{1:T}\mid y_{1:T})
 \propto p(\vartheta)p(x_0\mid\vartheta)
 \prod_{t=1}^T p(r_t,x_t\mid x_{t-1},\vartheta)
 g_{r_t}(y_t\mid x_t,\vartheta).
\tag{4}
\]

A measure-theoretic caution is required before (4) is used as a density. When
the transition (1) is a deterministic map of \((x_{t-1},\epsilon_t)\), the
conditional law of \(x_t\) given \(x_{t-1}\) can be singular with respect to
Lebesgue measure (if \(\dim\epsilon<\dim x\)) or require a change-of-variables
Jacobian. The clean primitive object is the Markov transition kernel

\[
 K_\vartheta(x_{t-1},A,r)
 =\int \mathbf 1\{\mathcal R(x_{t-1},\epsilon,\vartheta)=r\}\,
       \mathbf 1\{F_r(x_{t-1},\epsilon,\vartheta)\in A\}\,
       \phi_\vartheta(d\epsilon),
\tag{4a}
\]

and \(p(r_t,x_t\mid x_{t-1},\vartheta)\) in (4) denotes a density of
\(K_\vartheta\) with respect to a *declared* dominating measure --- Lebesgue
in \(x_t\) times counting in \(r_t\) when the shock map is a smooth
bijection onto its range, and otherwise the non-centered shock chart of (56)
is the default coordinate system, in which the target is a density in
\((\vartheta,x_0,\epsilon_{1:T})\) and no transition density is needed at
all. Similarly, the initial factor \(p(x_0\mid\vartheta)\) in (4) and the
first-step factor \(p(X_1\mid\vartheta)\) used by the particle recursion
(12a) must be reconciled by declaring one of the two conventions: either
\(x_0\) is carried explicitly in every particle and (12a) begins from the
prior \(p(x_0\mid\vartheta)\) with a first transition draw, or the induced
initial law \(p(x_1\mid\vartheta)=\int K_\vartheta(x_0,\cdot)\,
p(x_0\mid\vartheta)\,dx_0\) is used and \(x_0\) is integrated out. Both are
correct; mixing them double-counts or drops the initial condition.

If regimes are integrated out, the marginal likelihood is

\[
 p(y_{1:T}\mid\vartheta)=
 \int p(x_0\mid\vartheta)
 \prod_{t=1}^T\sum_{r_t}p(r_t,x_t\mid x_{t-1},\vartheta)
 g_{r_t}(y_t\mid x_t,\vartheta)\,dx_{0:T}.
\tag{5}
\]

The summation in (5), or the corresponding nonlinear integral, is the part that
the piecewise Kalman filter avoids by conditioning on a verified regime path.

### 2.1 Four different posterior geometries

The sampling problem is determined by the measure in (4), not by the economic
word *regime*. Four cases must be kept separate.

First, a censored rule such as \(i_t=\max\{\ell,i_t^\star\}\) is continuous. If
all other model equations and densities are continuous, the posterior is also
continuous, although its gradient may jump at the kink. Ordinary Metropolized
HMC remains a valid MCMC construction when the nondifferentiable set has
Lebesgue measure zero, but an integrator that ignores the kink can be inefficient
(Afshar and Domke 2015, p. 1). There is no potential-energy jump to refract across.

Second, suppose a deterministic solution map produces different one-sided
densities. Write all continuous unknowns as \(q\), let \(R(q)\) be the implied
regime path, and let \(\pi_r(q)\) be the density obtained from branch \(r\),
where every \(\pi_r\) is expressed in one *named* branch coordinate system;
if different branches are naturally parameterized in different coordinates,
the change-of-variables Jacobian into the common chart belongs inside
\(\pi_r\). The
joint density with respect to Lebesgue measure in \(q\) and counting measure in
\(r\) is

\[
 \Pi(q,r)=\pi_r(q)\,\mathbf 1\{r=R(q)\}.
\tag{5a}
\]

At fixed \(q\), every proposal \(r'\ne R(q)\) has zero target density. A
regime-only Metropolis, mixed-HMC, or Gibbs move therefore cannot cross this
deterministic boundary. One must instead move \(q\) across the boundary with an
event-aware continuous kernel, make a joint \((q,r)\) proposal whose density and
Jacobian are included, or integrate the states with a particle method. The
one-sided potential jump for event dynamics is

\[
 \Delta U(q_b)=-\log \pi_{r^+}(q_b^+)+\log \pi_{r^-}(q_b^-),
\tag{5b}
\]

where \(q_b^-\) and \(q_b^+\) denote limits from the two regions. For a pure
continuous `max`, these limits agree and \(\Delta U=0\).

Third, if the statistical model assigns positive probabilities to two regimes,
for example \(p(r_t\mid r_{t-1},x_{t-1},\vartheta)>0\) for both values, then the
posterior really has mixed support. Mixed HMC and particle Gibbs are valid
candidates because a discrete move can have positive target probability.

Fourth, if a nonlinear equilibrium solver can return several solutions, an
algorithmic tie-break such as "first root found" is not a probability law. The
target remains undefined until the model supplies a solution-selection
distribution. HMC cannot repair an incomplete statistical model.

## 3. What OccBin and piecewise Kalman filtering actually do

### 3.1 The two-regime linear system

Guerrieri and Iacoviello (2015) write an occasionally
binding constraint as a reference regime \(n\) and an alternative regime \(b\).
After linearization around a common steady state, each regime has a linear
rational-expectations system. In reduced state-space notation, a candidate regime
path \(r_{1:T}\) gives

\[
 x_t=T_{r_t}x_{t-1}+C_{r_t}+R_{r_t}\epsilon_t,
 \qquad \epsilon_t\sim N(0,Q),
\tag{6}
\]

\[
 y_t=Z_{r_t}x_t+D_{r_t}+u_t,
 \qquad u_t\sim N(0,H_{r_t}).
\tag{7}
\]

The matrices are not chosen by splicing arbitrary policy rules. The regime
equations are solved as linear systems, and the candidate path is accepted only
when the implied inequalities agree with the regime labels.

For a finite horizon, OccBin starts from a guessed path. It solves backward from
a terminal date at which the reference regime is imposed forever, then simulates
forward, checks the binding and slack inequalities, and updates the path. Repeating
this map gives a fixed point when it converges. The method can have multiple
fixed points; convergence of one iteration is not a proof of global uniqueness
(Guerrieri and Iacoviello 2015, pp. 2--7).

### 3.2 The conditional Kalman likelihood

Conditional on a verified path, the model is time-varying but Gaussian. With
filtered mean \(a_{t-1}\) and covariance \(P_{t-1}\), prediction is

\[
 a_t^- =T_{r_t}a_{t-1}+C_{r_t},
\qquad
 P_t^- =T_{r_t}P_{t-1}T_{r_t}^{\mathsf T}+R_{r_t}QR_{r_t}^{\mathsf T}.
\tag{8}
\]

The innovation and its covariance are

\[
 v_t=y_t-Z_{r_t}a_t^- -D_{r_t},
 \qquad
 F_t=Z_{r_t}P_t^-Z_{r_t}^{\mathsf T}+H_{r_t}.
\tag{9}
\]

The update is

\[
 K_t=P_t^-Z_{r_t}^{\mathsf T}F_t^{-1},\qquad
 a_t=a_t^-+K_tv_t,\qquad
 P_t=P_t^- -K_tF_tK_t^{\mathsf T}.
\tag{10}
\]

For observed components \(\mathcal O_t\), the conditional log likelihood is

\[
 \log p(y_{1:T}\mid r_{1:T},\vartheta)
 =-\frac12\sum_{t=1}^T
 \left[|\mathcal O_t|\log(2\pi)+\log|F_t|+v_t^{\mathsf T}F_t^{-1}v_t\right].
\tag{11}
\]

Giovannini, Pfeiffer, and Ratto (2021, pp. 5--11)
make the regime path endogenous to the filter. Their piecewise Kalman filter
(PKF) performs (8)--(10), estimates shocks and smoothed states, calls an OccBin
simulation to infer a new path, replaces the transition matrices, and iterates.
Failed fixed-point convergence is treated as a bad parameter proposal. The
implementation in Dynare exposes this structure in
`+occbin/kalman_update_algo_3.m`: a standard Kalman update is followed by an
OccBin solve, a new `regime_history`, and a loop until the proposed and implied
paths agree. This is implementation evidence, not a substitute for the paper's
derivation.

### 3.3 Why this is not a nonlinear ZLB filter

The PKF is exact for its conditional time-varying linear-Gaussian model, but the
model itself is a piecewise-linear approximation. It omits precautionary effects
and higher-order nonlinearities that alter the probability of future binding
(Giovannini, Pfeiffer, and Ratto 2021, pp. 3--4, 26 ff.). OccBin also needs a reference
regime, a finite path iteration, and a rule for resolving multiple fixed points.
These are model and solver choices, not mere numerical details.

The right way to use PKF in the new project is as a **limiting benchmark**:
when the nonlinear model is replaced by its verified piecewise-linear local
representation, the new code should reproduce (11), the regime path, and the
OccBin inequality checks. It must not claim that this tie-out validates the
nonlinear sampler.

## 4. A nonlinear likelihood without regime smoothing

### 4.1 Generic particle filtering

For the nonlinear transition in (1), include the regime in the particle state
when it is stochastic; when it is deterministic, compute it from each proposed
state and shock. Let \(q_1(x_1\mid y_1,\vartheta)\) and
\(q_t(x_t\mid x_{t-1},y_t,\vartheta)\) be proposals. At \(t=1\), draw
\(X_1^j\sim q_1\) and calculate

\[
 v_1^j=\frac{p(X_1^j\mid\vartheta)g(y_1\mid X_1^j,\vartheta)}
 {q_1(X_1^j\mid y_1,\vartheta)},\qquad
 W_1^j=\frac{v_1^j}{\sum_k v_1^k}.
\tag{12a}
\]

For \(t\geq2\), resample \(A_t^j\sim\operatorname{Categorical}(W_{t-1}^{1:M})\),
draw \(X_t^j\sim q_t(\cdot\mid X_{t-1}^{A_t^j},y_t,\vartheta)\), and set

\[
 v_t^j=\frac{p(X_t^j\mid X_{t-1}^{A_t^j},\vartheta)
                  g(y_t\mid X_t^j,\vartheta)}
 {q_t(X_t^j\mid X_{t-1}^{A_t^j},y_t,\vartheta)},\qquad
 W_t^j=\frac{v_t^j}{\sum_kv_t^k}.
\tag{12b}
\]

The resample-propagate likelihood estimator is

\[
 \widehat p(y_{1:T}\mid\vartheta)=
 \prod_{t=1}^T\left(\frac1M\sum_{j=1}^M v_t^j\right).
\tag{13}
\]

To see why (13) is unbiased, condition on the particle system at \(t-1\).
Averaging over the categorical ancestor and proposal draw cancels \(q_t\) and
gives the particle approximation to the predictive integral. Repeating the
conditional expectation backward to \(t=1\) yields the Feynman--Kac normalizing
constant \(p(y_{1:T}\mid\vartheta)\). Thus (13) is nonnegative and unbiased
under the usual support and integrability conditions
(Andrieu, Doucet, and Holenstein 2010, Sec. 2.2; Aruoba et al. 2021, Sec. 6). The log
estimate is not unbiased:
\(\mathbb E[\log\widehat p]\leq\log p\) by Jensen's inequality.

### 4.2 The piecewise-linear conditionally optimal particle filter

Aruoba et al. (2021)
construct continuous piecewise-linear decision rules. In their canonical form,

\[
 s_t=
 \begin{cases}
 \Phi_0(n)+\Phi_1(n)s_{t-1}+\Phi_\eta(n)\eta_t,&
 \eta_{1,t}<\zeta(s_{t-1}),\\
 \Phi_0(b)+\Phi_1(b)s_{t-1}+\Phi_\eta(b)\eta_t,&
 \eta_{1,t}\geq\zeta(s_{t-1}),
 \end{cases}
\tag{14}
\]

with a linear measurement equation. Conditional on a previous particle, the
observation density is a mixture of two truncated Gaussian integrals. Let
\(D_t^j(n)\) and \(D_t^j(b)\) be the two regime contributions. Then

\[
 \lambda_t^j=\frac{D_t^j(n)}{D_t^j(n)+D_t^j(b)},
\tag{15}
\]

and the conditionally optimal proposal draws a regime with probability
\(\lambda_t^j\), draws the structural innovation from the corresponding
truncated Gaussian, and assigns incremental weight

\[
 \widetilde w_t^j=D_t^j(n)+D_t^j(b).
\tag{16}
\]

The derivation is simply Bayes' rule. The optimal proposal is

\[
 g_t^\star(s_t\mid y_t,s_{t-1})
 \propto p(y_t\mid s_t)p(s_t\mid s_{t-1}),
\tag{17}
\]

so the importance ratio is the predictive density
\(p(y_t\mid s_{t-1})\), which is exactly the sum in (16). With vanishing
measurement error, one, two, or no regime solutions may explain an observation;
if every particle falls in the no-solution case, the likelihood estimate is zero
(Aruoba et al. 2021, pp. 26--31).

This is the closest existing bridge to nonlinear ZLB inference: it preserves an
endogenous regime and avoids replacing the bound by a softplus. Its limitations
are equally important. The transition is still a piecewise-linear approximation,
the construction is not a parameter-gradient HMC kernel, and a regime ambiguity
must be represented as a mixture rather than silently resolved.

### 4.3 A piecewise, truncated, and mixture UKF construction

The UKF literature contains a useful intermediate family between a single
Gaussian filter and a fully particle-based filter. Gaussian-sum filtering,
which carries a weighted sum of Gaussian components and updates each component
with a Kalman-type step, goes back to Alspach and Sorenson (1972). Within the
sigma-point family, the truncated UKF conditions the Gaussian approximation on
bounded-support measurement information (García-Fernández, Morelande, and
Grajal 2012a), its mixture variant applies the truncated update componentwise
while retaining a Gaussian mixture (García-Fernández, Morelande, and Grajal
2012b), and a UKF-aided Gaussian-sum filter combines both ingredients (Gokce
and Kuzuoglu 2015). The full texts of these three IEEE/IET-archived papers and
of Alspach and Sorenson (1972) were not available in the local corpus; they are
cited from verified publication metadata as the named architectures nearest to
what the ZLB needs, and every equation in this section is derived in this
survey's notation rather than imported from them. Two constrained-UKF papers
were inspected in full: Kandepu, Imsland, and Foss (2008, Secs. II--III)
project transformed sigma points onto the feasible region, and Teixeira et al.
(2010, Secs. 4--5, Table 1) formulate interval-constrained unscented filtering,
compare eight projection-, constraint-, and truncation-based UKF variants, and
note that their motivating examples have multimodal densities. These methods
are not exact nonlinear-ZLB filters by themselves, but they give a principled
construction for preserving the binding and nonbinding pieces before a particle
correction.

#### 4.3.1 Why an ordinary UKF loses a branch

Let \(z_t\) collect the uncertain variables that determine the next regime. For
example, \(z_t=(x_{t-1}^{\mathsf T},\epsilon_t^{\mathsf T})^{\mathsf T}\), with a
predictive approximation

\[
 z_t\mid y_{1:t-1}\approx N(m_t,P_t).
\tag{18}
\]

Let \(D_b\) and \(D_n\) denote the binding and nonbinding regions, respectively,

\[
 D_b=\{z:b_t(z)\leq0\},
 \qquad
 D_n=\{z:b_t(z)>0\},
 \qquad
 b_t(z)=i_t^\star(z)-\ell .
\tag{19}
\]

The exact predictive state density is a sum of integrals over these regions:

\[
 p(x_t\mid y_{1:t-1})
 =\sum_{r\in\{b,n\}}
 \int_{D_r}p(x_t\mid z_t,r,\vartheta)
       p(z_t\mid y_{1:t-1})\,dz_t .
\tag{20}
\]

Even when \(p(z_t\mid y_{1:t-1})\) is Gaussian, (20) is generally not one
Gaussian: conditioning on \(D_r\) truncates the input, and the two maps
\(F_b\) and \(F_n\) can produce different means and covariances. A conventional
UKF propagates one sigma-point cloud through a branch function and then replaces
the transformed distribution by its first two moments. That moment matching is
valid as an approximation, but it discards branch mass, skewness, and
multimodality exactly where the ZLB is informative. The issue is not fixed by
calling the `max` function at each sigma point.

#### 4.3.2 Exact branch probabilities and truncated moments for an affine boundary

The affine case supplies a fully analytic building block. Suppose

\[
 b_t(z)=a^{\mathsf T}z-c,
 \qquad z\sim N(m,P),
 \qquad \sigma_b^2=a^{\mathsf T}Pa>0 .
\tag{21}
\]

The scalar \(u=a^{\mathsf T}z\) is \(N(\mu_b,\sigma_b^2)\), where
\(\mu_b=a^{\mathsf T}m\). Define

\[
 \gamma=\frac{c-\mu_b}{\sigma_b},
 \qquad
 \alpha_b=\Pr(b_t(z)\leq0)=\Phi(\gamma),
 \qquad
 \alpha_n=\Pr(b_t(z)>0)=1-\Phi(\gamma).
\tag{22}
\]

Here \(\Phi\) and \(\phi\) are the standard normal cdf and density. To derive the
conditional moments, standardize \(v=(u-\mu_b)/\sigma_b\). For the binding
event \(v\leq\gamma\), direct integration gives

\[
 \mathbb E[v\mid v\leq\gamma]
 =\frac{\int_{-\infty}^{\gamma}v\phi(v)\,dv}{\Phi(\gamma)}
 =-\frac{\phi(\gamma)}{\Phi(\gamma)},
\tag{23}
\]

because \(\int v\phi(v)\,dv=-\phi(v)\). Integration by parts similarly gives

\[
 \operatorname{Var}(v\mid v\leq\gamma)
 =1-\lambda_b(\lambda_b+\gamma),
 \qquad
 \lambda_b=\frac{\phi(\gamma)}{\Phi(\gamma)}.
\tag{24}
\]

For the nonbinding event \(v>\gamma\), the same calculation over
\((\gamma,\infty)\) gives

\[
 \mathbb E[v\mid v>\gamma]=\lambda_n,
 \qquad
 \operatorname{Var}(v\mid v>\gamma)
 =1-\lambda_n(\lambda_n-\gamma),
 \qquad
 \lambda_n=\frac{\phi(\gamma)}{1-\Phi(\gamma)}.
\tag{25}
\]

The Gaussian conditioning identity decomposes \(z\) into the scalar direction
and an independent residual:

\[
 z=m+\frac{Pa}{\sigma_b^2}(u-\mu_b)+\xi,
 \qquad
 \xi\perp u,
 \qquad
 \operatorname{Var}(\xi)=P-\frac{Paa^{\mathsf T}P}{\sigma_b^2}.
\tag{26}
\]

Substituting (23)--(25) into (26) yields the exact first two moments of each
truncated branch:

\[
 m_b=m-\frac{Pa}{\sigma_b}\lambda_b,
 \qquad
 P_b=P-\frac{Paa^{\mathsf T}P}{\sigma_b^2}
             \lambda_b(\lambda_b+\gamma),
\tag{27}
\]

\[
 m_n=m+\frac{Pa}{\sigma_b}\lambda_n,
 \qquad
 P_n=P-\frac{Paa^{\mathsf T}P}{\sigma_b^2}
             \lambda_n(\lambda_n-\gamma).
\tag{28}
\]

Equations (22), (27), and (28) are not a heuristic projection. They are the
moments of the original Gaussian law conditional on the branch event. They are
therefore the correct starting point for a branch-aware sigma-point filter when
the switching statistic is affine in the augmented state and shock. One
implementation requirement attaches to them: the Mills ratios
\(\lambda_b=\phi(\gamma)/\Phi(\gamma)\) and
\(\lambda_n=\phi(\gamma)/(1-\Phi(\gamma))\) underflow catastrophically in the
tails when evaluated from \(\Phi\) directly. Implementations must use
log-cdf/log-survival and log-density arithmetic (or an independently tested
inverse-Mills-ratio routine), and any variance clamp added near
\(\gamma\to\pm\infty\) must be documented as a roundoff safeguard, not as a
substitute for stable probability evaluation.

For a nonlinear switching statistic \(b_t\), no such closed form exists. One can
linearize \(b_t\) at \(m\) and reuse (21)--(28) with
\(a=\nabla b_t(m)\) and \(c=a^{\mathsf T}m-b_t(m)\), but that is an
approximation whose error grows with the curvature of the boundary inside the
predictive ellipsoid. It also inherits a sigma-point blind spot: a standard
\(2d+1\)-point set concentrated on the predictive mean can entirely miss a
low-probability binding region, so the estimated \(\alpha_b\) can be exactly
zero even when the truth is small but positive. Component splitting along the
estimated boundary normal, or the particle correction of Section 4.3.6, are the
two defensible responses; both must be labelled as the approximation or the
authority they are.

#### 4.3.3 The pure censored policy-rate likelihood

The simplest ZLB example makes the mixture structure visible without any UKF
approximation. Let the shadow rate be

\[
 i_t^\star\sim N(\mu,\sigma^2),
 \qquad i_t=\max\{\ell,i_t^\star\},
 \qquad y_t=i_t+u_t,
 \qquad u_t\sim N(0,V).
\tag{29}
\]

The bound has probability

\[
 \alpha_b=\Phi\!\left(\frac{\ell-\mu}{\sigma}\right),
 \qquad
 \alpha_n=1-\alpha_b.
\tag{30}
\]

With no measurement error, the distribution of \(i_t\) has an atom of mass
\(\alpha_b\) at \(\ell\) and a continuous upper tail:

\[
 p(i_t)=\alpha_b\,\delta_\ell(i_t)
 +\frac{1}{\sigma}\phi\!\left(\frac{i_t-\mu}{\sigma}\right)
  \mathbf 1\{i_t>\ell\}.
\tag{31}
\]

The atom is often hidden when a Gaussian observation error is present. To
derive the observed density, write the binding contribution as
\(\alpha_b N(y_t;\ell,V)\). For the nonbinding contribution, multiply two
Gaussian densities and complete the square:

\[
 N(y_t;i,V)N(i;\mu,\sigma^2)
 =N(y_t;\mu,V+\sigma^2)N(i;\widetilde\mu_t,\widetilde\sigma^2),
\tag{32}
\]

where

\[
 \widetilde\sigma^2=\frac{\sigma^2V}{\sigma^2+V},
 \qquad
 \widetilde\mu_t=\frac{V\mu+\sigma^2y_t}{\sigma^2+V}.
\tag{33}
\]

Integrating (32) over \(i>\ell\) gives the complete predictive density

\[
 p(y_t)=
 \alpha_bN(y_t;\ell,V)
 +N(y_t;\mu,V+\sigma^2)
  \left[1-\Phi\!\left(
   \frac{\ell-\widetilde\mu_t}{\widetilde\sigma}\right)\right].
\tag{34}
\]

The posterior probability of a binding observation is consequently

\[
 \Pr(b\mid y_t)=
 \frac{\alpha_bN(y_t;\ell,V)}{p(y_t)}.
\tag{35}
\]

Equation (34) is the one-period analogue of the two-regime truncated-Gaussian
mixture in COPF. It also shows why replacing the max by a softplus is not an
innocent numerical trick: it removes the atom in (31), changes (34), and hence
changes the likelihood. In a DSGE model, the other observables and the state
transition replace the scalar Gaussian factors in (34), but the same branch
decomposition remains.

#### 4.3.4 Branchwise UKF update

Suppose the filtering approximation before the next observation is a mixture

\[
 p(z_t\mid y_{1:t-1})
 \approx\sum_{j=1}^{J_{t-1}}\omega_{t-1,j}
 N(z_t;m_{t-1,j},P_{t-1,j}),
 \qquad \sum_j\omega_{t-1,j}=1.
\tag{36}
\]

For each component \(j\), calculate (22), (27), and (28), obtaining a branch
component \((j,r)\) with prior mass

\[
 \rho_{j,r}=\omega_{t-1,j}\alpha_{j,r},
 \qquad r\in\{b,n\}.
\tag{37}
\]

Use the branch-conditioned moments \((m_{j,r},P_{j,r})\) to form sigma points
\(\chi_{j,r}^{(k)}\) and mean/covariance weights \(w_m^{(k)},w_c^{(k)}\). Propagate
through the branch-specific transition and observation maps:

\[
 X_{j,r}^{(k)}=F_r(\chi_{j,r}^{(k)},\vartheta),
 \qquad
 Y_{j,r}^{(k)}=h_r(X_{j,r}^{(k)},\vartheta).
\tag{38}
\]

If process and observation noises are included in the augmented sigma point,
their covariances are already present. Otherwise add \(Q_r\) and \(V_r\) below.
The branch prediction is

\[
 \bar x_{j,r}=\sum_k w_m^{(k)}X_{j,r}^{(k)},
 \qquad
 P^x_{j,r}=\sum_k w_c^{(k)}
 (X_{j,r}^{(k)}-\bar x_{j,r})(X_{j,r}^{(k)}-\bar x_{j,r})^{\mathsf T}+Q_r,
\tag{39}
\]

\[
 \bar y_{j,r}=\sum_k w_m^{(k)}Y_{j,r}^{(k)},
\tag{40}
\]

\[
 S_{j,r}=\sum_k w_c^{(k)}
 (Y_{j,r}^{(k)}-\bar y_{j,r})(Y_{j,r}^{(k)}-\bar y_{j,r})^{\mathsf T}+V_r,
\tag{41}
\]

\[
 C_{j,r}=\sum_k w_c^{(k)}
 (X_{j,r}^{(k)}-\bar x_{j,r})(Y_{j,r}^{(k)}-\bar y_{j,r})^{\mathsf T}.
\tag{42}
\]

The Gaussian UKF predictive likelihood, branch gain, and updated component are

\[
 \widehat L_{j,r}=N(y_t;\bar y_{j,r},S_{j,r}),
 \qquad K_{j,r}=C_{j,r}S_{j,r}^{-1},
\tag{43}
\]

\[
 m^+_{j,r}=\bar x_{j,r}+K_{j,r}(y_t-\bar y_{j,r}),
 \qquad
 P^+_{j,r}=P^x_{j,r}-K_{j,r}S_{j,r}K_{j,r}^{\mathsf T}.
\tag{44}
\]

Bayes' rule combines the prior branch mass, the branch predictive likelihood,
and the evidence normalizer:

\[
 \widetilde\omega_{j,r}=\rho_{j,r}\widehat L_{j,r},
 \qquad
 Z_t=\sum_{j=1}^{J_{t-1}}\sum_{r\in\{b,n\}}\widetilde\omega_{j,r},
 \qquad
 \omega^+_{j,r}=\frac{\widetilde\omega_{j,r}}{Z_t}.
\tag{45}
\]

Thus the branch-aware UKF approximation after observing \(y_t\) is

\[
 p(x_t\mid y_{1:t})
 \approx\sum_{j,r}\omega^+_{j,r}
 N(x_t;m^+_{j,r},P^+_{j,r}).
\tag{46}
\]

If one insists on collapsing (46) to one Gaussian, the moment-preserving
collapse is

\[
 \bar m_t=\sum_{j,r}\omega^+_{j,r}m^+_{j,r},
\tag{47}
\]

\[
 \bar P_t=\sum_{j,r}\omega^+_{j,r}
 \left[P^+_{j,r}+
 (m^+_{j,r}-\bar m_t)(m^+_{j,r}-\bar m_t)^{\mathsf T}\right].
\tag{48}
\]

The second term in (48) is between-branch variance. Dropping it is not a
benign implementation detail: it makes the next threshold probability too
small or too large whenever binding and nonbinding components are separated.
Keeping all components causes up to \(2^t\) histories, the classical
Gaussian-sum growth problem, so practical filters use pruning, merging, or an
IMM-style moment collapse. Those operations are approximations and must be
recorded as such.

#### 4.3.5 Relation to IMM-UKF and to deterministic ZLB support

For a genuinely stochastic Markov regime, the preceding construction becomes a
multiple-model UKF. If \(p(r_t=r\mid r_{t-1}=s)=\Pi_{sr}\), the mode-prediction
probability is

\[
 c_{t,r}=\sum_s\Pi_{sr}\mu_{t-1,s},
 \qquad
 \mu_{t-1,s\mid r}=\frac{\Pi_{sr}\mu_{t-1,s}}{c_{t,r}}.
\tag{49}
\]

The IMM mixing mean and covariance supplied to the \(r\)-specific UKF are

\[
 m_{t-1\mid r}=\sum_s\mu_{t-1,s\mid r}m_{t-1,s},
\tag{50}
\]

\[
 P_{t-1\mid r}=\sum_s\mu_{t-1,s\mid r}
 \left[P_{t-1,s}+
 (m_{t-1,s}-m_{t-1\mid r})(m_{t-1,s}-m_{t-1\mid r})^{\mathsf T}\right].
\tag{51}
\]

After the branchwise update, the mode probability is proportional to
\(c_{t,r}\widehat L_{t,r}\). Equation (49) is Bayes' rule on the mode chain, and
(50)--(51) apply the moment-preserving collapse (47)--(48) within each
destination mode. This is the interacting-multiple-model (IMM) logic introduced
for Markovian switching systems by Blom and Bar-Shalom (1988), here with UKF
components in place of linear Kalman components. An inspected constrained
multiple-model UKF applies the same Markov-jump structure with truncated noise
supports and model-conditioned feasible regions (Zhang et al. 2020, Secs.
II--III).

The deterministic ZLB is different. There is no free \(\Pi_{sr}\) to tune: the
branch is \(r_t=R(z_t)\), and at a fixed \(z_t\) the other label has zero support
as in (5a). Equations (49)--(51) are therefore a valid approximation only if
we deliberately introduce a stochastic regime law. For the original
complementarity model, \(\alpha_{j,r}\) in (37) must be computed from the
continuous predictive law and the regime label must remain a derived quantity.

#### 4.3.6 UKF proposals inside a corrected particle filter

The branchwise UKF becomes much more defensible when it proposes particles but
does not define the likelihood by itself. Let \(q_{t,j,r}(z_t\mid y_t)\) be a
branch-conditioned UKF proposal supported on \(D_r\), and let
\(\alpha_{j,r}\) be the branch-selection probability. Draw \(r\) with probability
\(\alpha_{j,r}\), then draw \(z_t\sim q_{t,j,r}\). For a stochastic regime, the
incremental importance weight is

\[
 w_t=
 \frac{p(r_t\mid r_{t-1},x_{t-1},\vartheta)
       p(z_t\mid x_{t-1},r_t,\vartheta)
       g_{r_t}(y_t\mid x_t,\vartheta)}
      {\alpha_{j,r_t}
       q_{t,j,r_t}(z_t\mid x_{t-1},y_t,\vartheta)}.
\tag{52}
\]

For a deterministic regime, \(r_t=R(z_t)\) is not an independently sampled
variable. If the shock-augmented transition is used, the corresponding weight
is

\[
 w_t=
 \frac{p(\epsilon_t\mid\vartheta)
       g_{R(z_t)}(y_t\mid x_t,\vartheta)}
      {\alpha_{j,R(z_t)}
       q_{t,j,R(z_t)}(z_t\mid x_{t-1},y_t,\vartheta)},
 \qquad x_t=F_{R(z_t)}(x_{t-1},\epsilon_t,\vartheta).
\tag{53}
\]

The exact numerator and the proposal density, rather than the UKF's Gaussian
moment approximation, determine the weight. One coordinate obligation is
implicit in (53) and must be made explicit in any implementation: the
numerator density is written in the shock coordinate \(\epsilon_t\) while the
proposal density is written in the augmented coordinate \(z_t\). The two are
consistent only if either (a) the proposal draws the shock directly, so that
\(q\) is a density in \(\epsilon_t\) (with \(x_{t-1}\) fixed, the map
\(\epsilon_t\mapsto z_t\) is then an affine embedding and both densities live
in the same chart), or (b) the proposal density in \(z_t\) is converted to
the \(\epsilon_t\) chart with the absolute Jacobian of the transformation.
Option (a) is the recommended convention: propose shocks, derive states.
Under either convention, and under support and integrability
conditions, inserting (52) or (53) into the particle estimator (13) preserves
the usual nonnegative unbiased likelihood property. A UKF-only filter does not
have this correction; it remains an approximate Gaussian-sum likelihood.

An inspected precedent for this architecture is the iterative truncated
unscented particle filter (Wang et al. 2020, Secs. 3.1--3.2, Algorithm 1). There
an iterated-UKF update is truncated to the constraint region, the first two
moments of the truncated density define a Gaussian proposal, and each particle
is reweighted by the ratio of target factors to the proposal density. Two
differences from (52)--(53) matter for the ZLB. First, their constraint region
is exogenous, whereas the ZLB branch is selected by the model itself, which is
why the branch probability \(\alpha_{j,r}\) and the branch-specific maps
\(F_r,h_r\) appear in our weights. Second, their weight numerator uses a
Gaussian approximation of the predictive density (their eq. (36)), not the
exact transition density, so their filter is a proposal-quality benchmark
rather than an exactly weighted likelihood estimator; a claim-bearing ZLB
filter must keep the exact shock density in the numerator as in (53).

This gives a concrete hierarchy for the project. A split/truncated mixture UKF
is a useful fast diagnostic and proposal. A branch-corrected particle filter
is the nonlinear likelihood authority. The COPF construction is the exact
piecewise-linear benchmark. None of these statements turns the UKF proposal
into an exact HMC force; HMC must target either the corrected extended particle
state or a separately declared approximation. Section 13 applies this
machinery to the MacroFinance shadow-rate contract, where the switching lives
in the observation map rather than in the transition.

## 5. Ordinary HMC and how the lower bound changes its geometry

For a smooth posterior \(\pi(q)\propto e^{-U(q)}\), introduce momentum
\(p\sim e^{-K(p)}\), Hamiltonian \(H=U+K\), and equations
(Neal 2011, Sec. 5.2)

\[
 \dot q=\nabla_pK(p),\qquad \dot p=-\nabla_qU(q).
\tag{54}
\]

With Gaussian momentum \(K(p)=\frac12p^{\mathsf T}M^{-1}p\), leapfrog consists of

\[
 p\leftarrow p-\tfrac\epsilon2\nabla U(q),\quad
 q\leftarrow q+\epsilon M^{-1}p,\quad
 p\leftarrow p-\tfrac\epsilon2\nabla U(q).
\tag{55}
\]

The map is reversible and volume preserving; a final Metropolis correction
accounts for integration error. Childers et al. (2022, pp. 10--16) use this
idea on a joint posterior of parameters and latent shocks.
For a state equation \(x_t=F(x_{t-1},\epsilon_t,\vartheta)\), the joint log posterior
is, up to a constant,

\[
 \log p(\vartheta,x_0,\epsilon_{1:T}\mid y)
 =\log p(\vartheta)+\log p(x_0\mid\vartheta)
 +\sum_{t=1}^T\left[\log p(\epsilon_t\mid\vartheta)
 +\log g(y_t\mid x_t,\vartheta)\right].
\tag{56}
\]

This can remove a costly nonlinear filter, but it requires differentiating the
complete simulation map. A hard `max` is only piecewise differentiable:
automatic differentiation supplies a branch derivative but no boundary event.
A branch-dependent equilibrium solver may also jump or be multi-valued, while
discrete resampling changes index discontinuously. Replacing the bound with a
softplus creates a different posterior and must be treated as a separate
approximation, not as an exact ZLB solution.

### 5.1 Validity of Metropolized HMC on a kinked target

Because the MacroFinance target of Section 13 turns out to be continuous with
a kinked gradient rather than discontinuous, the validity question for
ordinary leapfrog HMC on such targets deserves a precise statement instead of
a passing remark. The following is a project derivation; the assumptions are
chosen to cover a continuous piecewise-quadratic-plus-smooth potential, which
is exactly what a piecewise-affine observation map inside a Gaussian
likelihood produces.

**Assumptions.** \(U\) is continuous on \(\mathbb R^d\); there is a closed
Lebesgue-null set \(N\) arising from a finite (or locally finite)
piecewise-regular partition of \(\mathbb R^d\) --- for the Section 13 target,
a finite union of hyperplanes --- such that \(U\) is continuously
differentiable on
\(\mathbb R^d\setminus N\) with \(\nabla U\) locally bounded, the branch
gradient on each piece is deterministic and single-valued, and a fixed
boundary convention assigns a value on \(N\); and
\(e^{-U}\) is integrable. These assumptions are *not* satisfied merely
because a potential is almost-everywhere differentiable: a potential
evaluated through an implicit multi-root equilibrium solver, a
state-dependent iteration count, or any branch rule that is not a fixed
partition of \(\mathbb R^d\) is outside this proposition until a separate
theorem covers it.

**Claim.** Fix a step size \(\epsilon\) and a step count \(L\). The set of
initial phase points \((q,p)\) whose leapfrog trajectory (55) evaluates
\(\nabla U\) at a point of \(N\) has Lebesgue measure zero. Off this null set,
the \(L\)-step leapfrog map is well defined, volume preserving, and reversible
in the usual momentum-flip sense, so the Metropolized chain that accepts with
probability \(1\wedge e^{-\Delta H}\) leaves \(\pi\propto e^{-U}\) invariant.

*Sketch.* Each half-kick \(p\mapsto p-\tfrac\epsilon2\nabla U(q)\) at fixed
\(q\notin N\) and each drift \(q\mapsto q+\epsilon M^{-1}p\) is a
measure-preserving shear. The set of starting points that land on \(N\) at a
given substep is the preimage of a null set under a composition of
measure-preserving maps defined off a null set, hence null; a finite union
over the \(2L+1\) substeps is null. Reversibility and volume preservation on
the remaining full-measure set follow from the standard shear argument (Neal
2011, Sec. 5.2), which nowhere uses smoothness across \(N\). The
piecewise-regular partition assumption is what makes the shear composition
well defined off a null set; the argument is *not* valid for an arbitrary
nonsmooth composition merely because individual shears preserve measure.
\(\square\)

Validity is not efficiency. Crossing the kink inside a drift substep replaces
the usual \(O(\epsilon^3)\) local energy error by

\[
 \Delta H \;=\; O\!\left(\epsilon\,
 \bigl\|\nabla U^{+}-\nabla U^{-}\bigr\|\right)
\tag{56a}
\]

at the crossing, where \(\nabla U^{\pm}\) are the one-sided gradients, so
acceptance degrades with the frequency of kink crossings and the size of the
gradient jump (this is the inefficiency, not invalidity, observed by Afshar
and Domke 2015, p. 1). Two consequences follow. First, the reflection and
refraction machinery of Section 6 degenerates on a kink: the potential jump in
(58) is \(\Delta U=0\), so (59) leaves the momentum unchanged, and event
detection buys accuracy, never correctness. Second, for a continuous
*piecewise-quadratic* potential, the Hamiltonian flow is available in closed
form on each piece; this is the construction of exact HMC for truncated
multivariate Gaussians (Pakman and Paninski 2014, metadata-verified, full text
not in the local corpus), and it is directly relevant to sampling the
cell-restricted Gaussians of Section 13.4. A boundary-exact integrator is
therefore an optimization for the kinked case and a requirement only for a
genuine density jump.

## 6. Reflection and refraction HMC for piecewise-smooth targets

### 6.1 The event calculation

Let \(U\) be smooth inside regions separated by an affine boundary with unit
normal \(\nu\). During a free-flight step, decompose momentum as

\[
 p_\perp=(p^{\mathsf T}\nu)\nu,
 \qquad p_\parallel=p-p_\perp.
\tag{57}
\]

At the boundary, define the potential jump
\(\Delta U=U(q^+)-U(q^-)\). Energy conservation requires

\[
 K(p^+)-K(p^-)=-\Delta U.
\tag{58}
\]

For unit Gaussian mass, if \(\|p_\perp\|^2>2\Delta U\), cross the boundary and set

\[
 p_\perp^+=
 \sqrt{\|p_\perp^-\|^2-2\Delta U}\,
 \frac{p_\perp^-}{\|p_\perp^-\|};
\tag{59}
\]

otherwise reflect:

\[
 p_\perp^+=-p_\perp^-.
\tag{60}
\]

The tangential momentum is unchanged. Afshar and Domke (2015, pp. 2--4) detect
the first affine-boundary intersection, apply (59) or (60),
continue the remaining flight, and repeat for later boundaries. The volume
statement must be attached to the right object: the reflection map (60) is an
orthogonal reflection with unit absolute Jacobian, but the refraction map
(59) *in isolation* rescales the normal momentum by \(s_-/s_+\) and is not
volume preserving when \(\Delta U\neq0\). What Afshar and Domke prove is that
the *assembled* flight--event--flight trajectory step is volume preserving
and reversible --- the momentum rescaling at the event is exactly cancelled
by the change in the position leg traversed at the new speed; combined
with leapfrog away from boundaries, this gives detailed balance after the
Metropolis correction (Afshar and Domke 2015, pp. 5--7).

Equations (57)--(60) are stated for unit Gaussian mass and cannot be used
verbatim in a preconditioned sampler. Project HMC uses a positive-definite
mass matrix \(M\) with \(K(p)=\tfrac12 p^{\mathsf T}M^{-1}p\), and the
correct decomposition is orthogonal in the \(M^{-1}\) inner product, not the
Euclidean one. For a boundary with (not necessarily unit) normal \(n\),
define

\[
 a=n^{\mathsf T}M^{-1}n,
 \qquad
 s_-=\frac{n^{\mathsf T}M^{-1}p_-}{\sqrt a}.
\tag{60a}
\]

For a potential jump \(\Delta U\), the trajectory crosses when
\(s_-^2>2\Delta U\), with

\[
 s_+=\operatorname{sign}(s_-)\sqrt{s_-^2-2\Delta U},
 \qquad
 p_+=p_-+n\,\frac{s_+-s_-}{\sqrt a},
\tag{60b}
\]

and reflects otherwise with

\[
 p_+=p_--2n\,\frac{n^{\mathsf T}M^{-1}p_-}{n^{\mathsf T}M^{-1}n}.
\tag{60c}
\]

Both maps change only the momentum component along \(M^{-1}n\), conserve
\(K(p)+U(q)\) across the event, and reduce to
(59)--(60) when \(M=I\) and \(\|n\|=1\). Their Jacobians differ: the
reflection (60c) is an \(M^{-1}\)-orthogonal reflection with unit absolute
Jacobian, while the refraction (60b) alone has absolute Jacobian
\(|s_-/s_+|\neq1\) for \(\Delta U\neq0\), and volume preservation holds only
for the composed flight--event--flight step, as above. A correctness test
must therefore target the composed trajectory map (or account for the event
Jacobian explicitly in the acceptance ratio), never the isolated refraction.
An implementation must additionally
fix the boundary orientation convention (the sign of \(n\) and hence of
\(\Delta U\)), the treatment of exact-equality and grazing events
(\(s_-^2=2\Delta U\) or \(s_-=0\)), and must be tested with dense and
diagonal nonidentity metrics, not only with \(M=I\).

### 6.2 What transfers to a ZLB

This method is a good match when the posterior in the sampled variables is
piecewise smooth and the switching surfaces are explicit affine or reliably
intersectable boundaries. It is not automatically valid for a nonlinear
endogenous regime solver. In that case the boundary is an implicit surface

\[
 b(q)=0,
\tag{61}
\]

and a trajectory needs (i) a first-root solver for \(b(q(t))\), (ii) the correct
one-sided values of the full log posterior, and (iii) a proof that the selected
equilibrium branch is single-valued at the event. A failed root search or a
different fixed point is not an event correction; it is a target-definition
failure.

## 7. Discontinuous HMC with Laplace momentum

### 7.1 Embedding an ordinal regime

Nishimura, Dunson, and Lu (2020, pp. 366--368) embed an
integer parameter \(N\) in a real variable \(\widetilde N\):

\[
 N=n \iff \widetilde N\in(a_n,a_{n+1}],\qquad
 \widetilde\pi(\widetilde n)=
 \sum_n\frac{\pi_N(n)}{a_{n+1}-a_n}
 \mathbf 1\{a_n<\widetilde n\leq a_{n+1}\}.
\tag{62}
\]

The interval length is essential: it is the Jacobian-like factor that makes the
embedded mass equal to \(\pi_N(n)\). An arbitrary ordering can create separated
modes, and energy-conserving dynamics may then mix poorly.

### 7.2 Why Gaussian momentum is expensive

When a Gaussian trajectory crosses many density jumps, every crossing must be
located and corrected. For a discrete coordinate with posterior uncertainty of
hundreds of units, that means hundreds of target evaluations. The standard
leapfrog error at a jump does not vanish as \(\epsilon\to0\), because the
unaccounted potential change is finite (Nishimura, Dunson, and Lu 2020, pp. 367--370).

Choose instead independent Laplace momentum,

\[
 K(p)=\sum_i |p_i|/m_i,
\qquad
 \dot q_i=m_i^{-1}\operatorname{sign}(p_i).
\tag{63}
\]

The velocity depends on the sign, not the magnitude. For coordinate \(i\),
propose the endpoint

\[
 q_i^\star=q_i+\epsilon m_i^{-1}\operatorname{sign}(p_i),
\qquad
 \Delta U=U(q^\star)-U(q).
\tag{64}
\]

If \(|p_i|/m_i>\Delta U\), accept the endpoint and reduce the momentum:

\[
 p_i\leftarrow p_i-\operatorname{sign}(p_i)m_i\Delta U.
\tag{65}
\]

Otherwise keep \(q_i\) and reflect the momentum \(p_i\leftarrow-p_i\). Sequentially
apply this coordinate map in a randomly permuted order. The random permutation
is needed for reversibility in distribution. Nishimura et al. prove that the
coordinate map is volume preserving and reversible almost everywhere, and that
the assembled integrator is valid with a final Metropolis correction
(Nishimura, Dunson, and Lu 2020, pp. 371--374). They also randomize the step size to
avoid a fixed-grid reducibility problem.

The algorithm is attractive for an explicit regime coordinate because one
endpoint likelihood evaluation can cross several ordinal intervals. It is a poor
fit for a solver that hides a complicated regime path inside one black-box
function: the endpoint energy difference is meaningful only if both endpoints
use the same declared target and all branch ambiguity is handled.

## 8. Mixed HMC for an explicit regime path

Zhou's mixed HMC (2020, pp. 2--4) retains the discrete variable rather than
pretending it is a real coordinate. Let \(z\) be discrete, \(q\) continuous, and
\(\pi(z,q)\propto e^{-U(z,q)}\). Add a clock \(s_j\) on a flat torus for each
discrete coordinate and a momentum \(p_j^D\). When a clock reaches its boundary,
propose \(\widetilde z\) using an irreducible single-site proposal
\(Q_j(\widetilde z\mid z)\). Define

\[
 \Delta E=\log\frac{\pi(z,q)Q_j(\widetilde z\mid z)}
                         {\pi(\widetilde z,q)Q_j(z\mid\widetilde z)}.
\tag{66}
\]

For a kinetic energy \(k_j^D(p_j^D)\), cross and change momentum so that

\[
 k_j^D(p_j^{D,+})=k_j^D(p_j^{D,-})-\Delta E
\tag{67}
\]

when the right-hand side is nonnegative; otherwise reflect. The continuous
coordinates are advanced by a reversible volume-preserving integrator. A final
Metropolis correction gives invariance of the joint distribution; the theorem
depends on the proposal ratio in (66), not on a conditional Gibbs draw.

For a *stochastic* ZLB path, set \(z=r_{1:T}\) or use a blocked path proposal.
Equation (66) then includes the complete regime-path probability and the
nonlinear state-space likelihood. This is conceptually clean but can be
expensive: changing one regime may require re-solving a long state path. For the
deterministic indicator in (5a), the regime-only proposal has zero support and
this construction does not apply. Either case needs an explicit probability law
for multiple equilibrium paths.

## 9. Smooth categorical relaxations and nonsmooth energies

The Concrete/Gumbel-softmax method (Torgander, Magnusson, and Wallin 2024,
pp. 2--3) maps categorical probabilities to a
continuous simplex variable. As temperature \(\tau\to0\), the variable becomes
nearly one-hot, but the transformed posterior develops steep geometry: gradients
with respect to simplex coordinates can diverge near zero and the temperature
direction becomes ill-conditioned. It is therefore a useful relaxation baseline,
not an exact categorical or ZLB sampler. Any comparison must report the
temperature and the induced posterior discrepancy.

Chaari et al.'s nonsmooth HMC (2016, pp. 1--5)
uses subgradients or a proximity operator

\[
 \operatorname{prox}_{\phi}(x)=
 \arg\min_y\left\{\phi(y)+\tfrac12\|y-x\|^2\right\}
\tag{68}
\]

for convex nondifferentiable energies such as an \(\ell_1\) penalty. A kink is
not a jump: prox-HMC cannot by itself restore a density discontinuity or enforce
an equilibrium regime selection. It is relevant only when the project
deliberately chooses a convex nonsmooth approximation.

## 10. Particle MCMC, pseudo-marginal HMC, and differentiable state-space HMC

### 10.1 Particle MCMC's exact extended target

Let \(u\) denote all random variables used by a particle filter, with density
\(m_\vartheta(u)\), and let \(\widehat L(\vartheta,u)\geq0\) be an unbiased likelihood
estimate:

\[
 \int \widehat L(\vartheta,u)m_\vartheta(u)\,du=L(\vartheta).
\tag{69}
\]

Then the extended target

\[
 \bar\pi(\vartheta,u)
 \propto p(\vartheta)\widehat L(\vartheta,u)m_\vartheta(u)
\tag{70}
\]

has marginal \(p(\vartheta)L(\vartheta)\). This is the pseudo-marginal identity
underlying particle MCMC (Andrieu, Doucet, and Holenstein 2010). It justifies a
Metropolis update that includes the random particle realization in the
acceptance ratio.

### 10.2 What pseudo-marginal HMC additionally requires

Alenlöv, Doucet, and Lindsten (2021, Secs. 2--3)
construct an extended Hamiltonian by differentiating \(\log\widehat L(\vartheta,u)\)
with respect to \(\vartheta\). Their construction requires, explicitly:
(i) a nonnegative unbiased estimator \(\widehat L(\vartheta,U)\); (ii)
auxiliary variables \(U\sim N(0,I)\); (iii) continuous differentiability of
\((\vartheta,U)\mapsto\widehat L(\vartheta,U)\); and (iv) pointwise
evaluability of \(\nabla\log\widehat L(\vartheta,U)\). Under those
conditions the algorithm is exact for the extended target,
and as the number of importance samples increases, trajectories approach
ideal HMC. They note explicitly that ordinary particle-filter likelihoods
typically *fail* condition (iii): a multinomial resampling index changes
discontinuously when a uniform random number crosses a cumulative-weight
threshold, so \(\widehat L\) is discontinuous in \((\vartheta,U)\).

The practical consequence for this survey's routes is sharp: an unbiased
bootstrap particle filter supports PMMH (Section 10.1), but it does not
support pseudo-marginal HMC by itself. A
differentiable resampling scheme or a transport-based relaxation can restore
condition (iii), but then its forward law and target must be written down
and audited: the gradient of a smoothed or transport-resampled computation is
an approximation, or the exact gradient of a *different* extended target,
until that target's full law and Metropolis correction are derived. PM-HMC is
therefore conditional on a separately constructed differentiable extended
estimator, never a free upgrade of a working PMMH route. The correlated
pseudo-marginal method (Deligiannidis, Doucet, and Pitt 2018) improves PMMH
mixing by correlating successive \(U\) draws and is the right
variance-reduction comparator; it does not repair differentiability and is
not a PM-HMC enabler.

### 10.3 Joint latent-state HMC

The alternative is to sample \((\vartheta,x_0,\epsilon_{1:T})\) directly using the
joint density (56). This avoids a particle likelihood and works well for smooth
nonlinear models (Childers et al. 2022, pp. 10--16). With a stochastic explicit
regime path, however, (56) becomes a
mixed target: the state transition and observation density are piecewise in
\(r_t\), and the path needs a supported discrete kernel. With a deterministic
regime, the density instead has the support in (5a). Joint HMC is therefore a
component of the proposed method, not the complete answer.

### 10.4 Particle Gibbs and ancestor sampling

Particle Gibbs samples a latent trajectory rather than estimating only the
marginal likelihood. Given a reference path \(x'_{1:T}\), conditional SMC keeps
one particle fixed to that path. Plain particle Gibbs tends to retain its early
ancestors when the particle genealogy collapses. PGAS repairs this by sampling
the fixed particle's ancestor at each time (Lindsten, Jordan, and Schön 2014,
Secs. 3--5).

For a Markov state-space model, one PGAS sweep is implementable as follows. At
\(t=1\), draw particles \(1{:}M-1\), set \(X_1^M=x'_1\), and weight all particles.
For each \(t=2{:}T\):

1. For \(j<M\), draw \(A_t^j\sim\operatorname{Categorical}(W_{t-1})\) and
   propagate \(X_t^j\) from the proposal in (12b).
2. Set \(X_t^M=x'_t\), but draw its ancestor with probabilities

\[
 \Pr(A_t^M=j)\propto W_{t-1}^j
 p(x'_t\mid X_{t-1}^j,\vartheta).
\tag{70a}
\]

3. Recompute the importance weights. At \(T\), draw a terminal particle from
   \(W_T\) and trace its ancestors to obtain the new path.

Equation (70a) is the Markov simplification of the general ancestor weight,
which multiplies the old particle weight by the ratio of the target density
after and before attaching the fixed future. Lindsten, Jordan, and Schön prove
that this kernel leaves the joint smoothing distribution invariant for every
particle count; their proof interprets the sweep as a partially collapsed Gibbs
update on an extended particle system (Lindsten, Jordan, and Schön 2014,
Algorithm 2 and Theorem 1). For a deterministic ZLB, the path is the continuous state or
shock path and \(r_t=R(q)\) is derived. For a stochastic regime, use
\((r_t,x_t)\) as the particle state. PGAS does not license an independent update
of a deterministic, zero-support regime label.

### 10.5 What differentiable particle filters do and do not supply

Ścibior and Wood keep the ordinary particle filter's forward values unchanged
but insert stop-gradient weight ratios so automatic differentiation returns the
Fisher-identity score estimator associated with the particle genealogy
(Ścibior and Wood 2021, Algorithm 1 and Sec. 3). Poyiadjis, Doucet, and Singh derive
the underlying path-space and forward-smoothing score estimators and show their
different cost and variance growth (Poyiadjis, Doucet, and Singh 2011, Secs. 2--3).
The estimator is consistent as the particle count grows; a finite-particle score
is not the exact deterministic force of the marginal log likelihood.

Corenflos et al. instead replace discrete resampling with a differentiable
entropy-regularized optimal-transport map. This changes the finite-particle
forward algorithm and introduces a regularization parameter; their consistency
analysis requires the particle count to grow and the regularization to shrink
(Corenflos et al. 2021, Sec. 4). Both constructions are
important comparison methods. Neither may be inserted as a noisy force into
ordinary HMC while retaining the usual exactness proof. Exact use requires an
extended target, fixed random numbers during each reversible trajectory, and a
Metropolis energy computed for that same extended density.

## 11. What the literature supports for this project

The evidence supports four connected layers, each with a different job.

**Layer A: a verified linear benchmark.** Implement OccBin/PKF path iteration
and the conditional Kalman likelihood (8)--(11). Tie it to the Dynare example
and to synthetic data with known regime paths. This establishes timing,
inequality signs, likelihood normalization, missing observations, and multiple
fixed-point diagnostics.

**Layer B: a nonlinear likelihood authority.** Implement the bootstrap particle
filter (12a)--(13) and the Aruoba et al. PLC/COPF special case. The bootstrap
filter handles a general simulable nonlinear transition; COPF is the
low-variance bridge for piecewise-linear continuous decision rules. Particle
Metropolis--Hastings then supplies a finite-particle extended target whose
parameter marginal is the exact posterior.

**Layer C: deterministic complementarity.** Here the regime is \(R(q)\), not an
independent unknown. Begin with joint-state random-walk or slice updates and
particle MCMC. Add event-aware HMC only after the complete nonlinear solution
map supplies a first-boundary oracle and both one-sided posterior values. A
fixed-branch HMC update is useful within one region, but cannot by itself cross
the support boundary in (5a).

**Layer D: stochastic or explicitly selected regimes.** When both regime values
have positive probability, alternate fixed-path HMC with PGAS or a valid
discrete Metropolis kernel. Zhou's mixed HMC becomes an acceleration candidate
after this simpler composition agrees with exact enumeration on a toy model.
Its clock changes update timing, not the target distribution.

The Concrete and softplus routes are labelled approximation arms. The local
MacroFinance file `two_currency_double_zlb_math.py` currently uses a smooth
softplus ZLB and is therefore a useful comparator, not evidence for genuine
discontinuous semantics.

The application contracts of Sections 13 and 14 assign these layers to the
two repositories. MacroFinance's coded model is a hard-bound
observation-map kink target: Layer B specializes to the bootstrap
particle-filter authority of Section 13.4, with the ideal cell decomposition
(84)--(87) as its proposal/diagnostic layer, Layer C specializes to joint
kinked-target HMC under Section 5.1, and Layers A and D do not apply to it as
coded. The future dsge_hmc true-OBC model (the existing package carries only
a no-binding placeholder; Section 14.6) is the converse: Layers A, C, and D
carry the weight, the
genuine discontinuity lives in the solution correspondence and its selection
rule (Section 14.2), and Layer D becomes available exactly when selection is
made stochastic (Section 14.3).

## 12. Exact targets and implementable transition kernels

### 12.1 Deterministic threshold target

Let \(q=(\vartheta,x_0,\epsilon_{1:T})\), recursively define
\(r_t=\mathcal R(x_{t-1},\epsilon_t,\vartheta)\) and
\(x_t=F_{r_t}(x_{t-1},\epsilon_t,\vartheta)\), and assume this solution is
unique. The posterior density on continuous variables is

\[
 \Pi_D(q)\propto p(\vartheta)p(x_0\mid\vartheta)
 \prod_{t=1}^T p(\epsilon_t\mid\vartheta)
 g_{r_t(q)}(y_t\mid x_t(q),\vartheta).
\tag{71}
\]

Inside a region with fixed \(r_{1:T}\), differentiate the same finite program
that evaluates (71). At the first event \(b_k(q(t))=0\), evaluate the two
one-sided values of the *complete* posterior, compute (5b), and apply the
general-mass event maps (60b) or (60c) --- (59)/(60) only in the unit-mass
special case. Continue for the unused part of the position step. Reverse the momentum
at the end and accept with

\[
 \alpha_D=1\wedge\exp\{H(q,p)-H(q',p')\}.
\tag{72}
\]

The full step is implementable only if the event locator returns the earliest
crossing, the branch solver is deterministic and single-valued on each side,
and the *composed* flight--event--flight trajectory map is reversible and
volume preserving (or the event Jacobian is accounted for explicitly in the
acceptance ratio); the isolated refraction map is not volume preserving and
must not be the object a correctness test targets, per Section 6.1. If any
condition is
missing, particle Metropolis--Hastings using (13), or a non-gradient joint-state
kernel, remains the exact baseline.

### 12.2 Stochastic mixed target

If regimes have a genuine transition mass, the joint target is instead

\[
 \Pi_S(\vartheta,x_{0:T},r_{1:T})\propto
 p(\vartheta)p(x_0\mid\vartheta)
 \prod_{t=1}^T p(r_t\mid r_{t-1},x_{t-1},\vartheta)
 p(x_t\mid x_{t-1},r_t,\vartheta)
 g_{r_t}(y_t\mid x_t,\vartheta).
\tag{73}
\]

A transparent first kernel composes two invariant updates:

1. Hold \(r_{1:T}\) fixed and apply ordinary Metropolized HMC to
   \((\vartheta,x_{0:T})\), using the gradient of \(\log\Pi_S\).
2. Hold the continuous variables fixed and update one regime, a regime block,
   or the whole latent path with Metropolis, forward-filter backward-sampling,
   or PGAS. For a proposal \(Q(r'\mid r,q)\), accept a regime-only move with

\[
 \alpha_R=1\wedge
 \frac{\Pi_S(q,r')Q(r\mid r',q)}
      {\Pi_S(q,r)Q(r'\mid r,q)}.
\tag{74}
\]

Each component preserves \(\Pi_S\), so their composition does as well. Mixed
HMC replaces the second component by the clock dynamics (66)--(67) and
interleaves it with the first. If a proposal jointly transforms continuous
coordinates to keep an economic constraint satisfied, its reverse proposal
density and absolute Jacobian must be added to (74).

### 12.3 Multiple equilibria

For solutions \(s_t\in\mathcal S(x_{t-1},\epsilon_t,\vartheta)\), specify a
selection mass \(p(s_t\mid x_{t-1},\epsilon_t,\vartheta)\) and include \(s_t\)
in (73), or integrate it out. Without that mass, neither (71) nor (73) is a
defined target. This is an economic-model specification requirement, not a
numerical HMC choice.

Sections 13 and 14 instantiate these abstract targets for the two application
repositories: the MacroFinance shadow-rate model that exists as code today, and
the existing `dsge_hmc` package at `/home/ubuntu/workspace/python/src/dsge_hmc`, whose
validated BGS restricted surface is a no-binding linear placeholder and whose
true-OBC model does not yet exist.

## 13. The MacroFinance shadow-rate contract

This section is the application contract for
`/home/ubuntu/workspace/MacroFinance`, derived from the code inspected on
19 August 2026 (`two_currency_double_zlb_math.py`, `_contract.py`,
`_target.py`, `_fixtures.py`, and the `dz5` campaign modules). Everything
below either restates what the code implements or derives consequences in
this survey's notation and labels them as such.

### 13.1 The model as coded

The latent state is
\(x_t=(x_t^{d\mathsf T},x_t^{f\mathsf T},x_t^{b\mathsf T})^{\mathsf T}
\in\mathbb R^8\): domestic and foreign dynamic Nelson-Siegel (DNS) factor
triples (level, slope, curvature) and two directed FX-basis factors. The
transition is a stationary Gaussian VAR(1) toward a long-run mean,

\[
 x_t=(I-\Phi)\bar\theta+\Phi x_{t-1}+\eta_t,
 \qquad \eta_t\sim N(0,Q),
\tag{75}
\]

where the inspected target variants estimate components of \(\bar\theta\) and
grouped log measurement-noise scales while freezing \(\Phi\), \(Q\), and the
initial covariance as fixture constants. The shadow instantaneous forward
curve of country \(c\in\{d,f\}\) at horizon \(s\) is affine in that country's
factors with a fixed decay \(\lambda_c\):

\[
 f_c(s;x)=a_c(s)^{\mathsf T}x^c,
 \qquad
 a_c(s)=\bigl(1,\;e^{-\lambda_c s},\;\lambda_c s\,e^{-\lambda_c s}
 \bigr)^{\mathsf T}.
\tag{76}
\]

The bound enters through one of two scalar maps applied to the forward curve,
the hard censoring map and its softplus smoothing,

\[
 m_\ell(u)=\max\{\ell,u\},
 \qquad
 s_{\ell,\alpha}(u)=\ell+\alpha\log\!\left(1+e^{(u-\ell)/\alpha}\right),
 \qquad \alpha>0.
\tag{77}
\]

The coded model uses \(s_{\ell,\alpha}\) exclusively, with per-country
constants \((\lambda_d,\ell_d,\alpha_d)=(0.65,\,0,\,1.5\times10^{-3})\) and
\((\lambda_f,\ell_f,\alpha_f)=(0.45,\,-0.005,\,1.0\times10^{-3})\); the
foreign bound is a negative effective lower bound, and \(\alpha\) is a fixed
model constant, not an estimated parameter. Observed zero-coupon yields are
maturity averages of the bounded forward curve, evaluated by an order-\(K\)
Gauss--Legendre rule on \([0,1]\) (the code fixes \(K=40\)):

\[
 y_c(\tau_i;x)=\frac1{\tau_i}\int_0^{\tau_i}
 s_{\ell_c,\alpha_c}\bigl(f_c(u;x)\bigr)\,du
 \;\approx\;\sum_{k=1}^{K}w_k\,
 s_{\ell_c,\alpha_c}\bigl(f_c(\tau_i v_k;x)\bigr),
\tag{78}
\]

with six maturities per country. The sole FX observation is the raw log
forward with observed log spot subtracted, tied to the two yield curves and
the basis factors by the covered-parity identity

\[
 \log F_t(\tau)-\log S_t=\tau\bigl(
 y_d(\tau;x_t)-y_f(\tau;x_t)+b(\tau;x_t^b)\bigr),
\tag{79}
\]

where \(b(\tau;\cdot)\) is affine in the basis factors. All measurement noise
is diagonal Gaussian with grouped scales. The active inference route evaluates
the marginal likelihood of (75)--(79) with BayesFilter's direct-factor
square-root UKF and runs HMC and NeuTra-HMC over a nine-parameter chart of
\(\bar\theta\) and noise scales against that filter's value and score. Two
declared-target facts follow immediately. First, the coded posterior is a
smooth-model posterior: \(s_{\ell,\alpha}\) is \(C^\infty\), so no
discontinuity of any kind is present in the coded target. Second, the UKF
likelihood is itself a deterministic, well-defined function of the parameters,
so HMC on it is exact *for that declared target*; the open question is its
distance from the exact-model likelihood, which Sections 13.4 and 13.7
address.

### 13.2 Where this sits in the shadow-rate literature

The design is a reduced-form member of the shadow-rate term structure family,
and the survey's earlier machinery maps onto that family precisely. Black
(1995) introduced the observation that currency makes the nominal short rate
an option, \(r_t=\max\{0,r_t^\star\}\) (metadata-verified attribution; full
text not in the local corpus). Krippner (2012, Secs. 1--2) made the
multi-factor Gaussian case tractable by bounding the *forward* curve --- the
ZLB forward is the shadow forward plus a closed-form currency-option effect
--- and obtaining yields by elementary numerical integration of the bounded
forward curve; MacroFinance follows exactly this bounded-forward-then-integrate
architecture with the softplus in place of Krippner's Gaussian-based map.
Priebsch (2013) developed the cumulant-based arbitrage-free approximation for
the same class. Wu and Xia (2016, Sec. 2.3, eqs. (6)--(8)) write the
one-period forward observation as
\(\underline r+\sigma^Q\,g\!\left((a_n+b_n^{\mathsf T}X_t-\underline r)/
\sigma^Q\right)\) with the closed-form \(g(z)=z\Phi(z)+\phi(z)\) and estimate
by extended Kalman filtering, noting that monotonicity of \(g\) keeps the
likelihood surface well behaved. Christensen and Rudebusch (2015, Appendix B)
estimate arbitrage-free shadow-rate Nelson-Siegel models with the extended
Kalman filter built on the discretized Ornstein--Uhlenbeck transition ---
literally the \((I-e^{-K\Delta})\theta+e^{-K\Delta}X_{t-1}\) form of (75).
Kim and Priebsch (2013, Sec. 4, eqs. (8)--(9)) replace the extended filter
with the *unscented* Kalman filter for quasi-maximum-likelihood estimation,
documenting that first-order linearization can be numerically unstable; every
earlier zero-bound study they name --- Ichiue--Ueno, Kim--Singleton, and the
2013 Christensen--Rudebusch draft --- had used the extended filter. Lemke and Vladu
(2017) estimate a euro-area shadow-rate model with a time-varying, estimated
lower bound using the extended filter. Finally, Opschoor and van der Wel
(2024, Secs. 2.1--2.3, eqs. (2)--(7)) build the reduced-form smooth
shadow-rate DNS model closest to the MacroFinance design: AR(1) DNS factors,
a lower bound applied through a smooth approximation of the max --- their
menu is the Gaussian-based map \(f_G(z)=z\Phi(z)+\phi(z)\), the softplus
\(f_S\), and an inverse-exponential map, each with a sharpness scaling ---
and two-step least-squares estimation, with nonlinear state-space estimation
via the extended Kalman filter noted as the alternative.

Three points of contrast matter for the contract. First, Opschoor and van der
Wel bound the *yield* directly, \(y^Z(\tau)=r_{LB}+f(y(\tau)-r_{LB})\),
whereas Krippner-type models --- and MacroFinance --- bound the *forward*
curve and integrate. The two choices differ: by pointwise convexity,

\[
 \sum_{k}w_k\max\{\ell,\,a(s_k)^{\mathsf T}x\}
 \;\geq\;
 \max\Bigl\{\ell,\;\sum_{k}w_k\,a(s_k)^{\mathsf T}x\Bigr\},
\tag{80}
\]

with equality only when the forward curve does not cross the bound inside the
maturity, so forward-level censoring produces weakly higher yields near the
bound and the two models are not reparameterizations of each other. Second,
Opschoor and van der Wel defend smoothing as an *economic* specification ---
yields empirically leave the bound gradually --- which licenses reading
\(\alpha\) as a structural smooth-transition constant rather than as a
numerical tolerance; Section 13.5 keeps both readings apart. Third, the
domain-standard estimators are all single-Gaussian moment-closure filters
(extended or unscented), that is, exactly the object whose branch-mass
deficiency Section 4.3.1 quantifies; none of the inspected shadow-rate papers
uses a mixture, particle, or exactly weighted likelihood, and none samples the
posterior with gradient MCMC. That statement must, however, be scoped to the
inspected closure-filter line: outside it, Pericoli and Taboga (2018, Secs.
2, 4--5) estimate a Black-type max shadow-rate model
(\(r_t=\max\{\bar r_t,\,a+bX_t\}\) with a time-varying bound) by fully
Bayesian data-augmentation MCMC --- blockwise random-walk Metropolis over
parameters *and* the entire latent factor path, with no Gaussian
moment-closure filter of any kind --- using a neural-network bond-pricing
surrogate trained to sub-basis-point accuracy; their "nearly exact" refers to
that pricing approximation, not to the sampler. The exact-likelihood
(closure-free) route in this literature is therefore already occupied. The
MacroFinance UKF-plus-HMC route can claim a gradient-sampler frontier within
the closure-filter family, but not a likelihood-fidelity frontier; the exact
constructions below are measured against Pericoli--Taboga's closure-free
standard, not merely against the filtering papers.

### 13.3 The target ladder and the geometry of the hard-bound variants

The word "exact" is reserved below for inference relative to a named target
and numerical program. The contract distinguishes three targets, which are
*different posteriors*, and every fixture, likelihood, sampler, and result
must carry one of these identifiers:

| `target_id` | Definition | Allowed claim |
|---|---|---|
| `mf_s1_k40_softplus` | the current coded finite program: \(K=40\) quadrature of the softplus map (77)--(78) | smooth-model inference (Sec. 13.5) |
| `mf_c1_k40_hardmax` | the same nodes and weights with \(s_{\ell,\alpha}\) replaced by \(m_\ell\) | exact relative to the finite hard-quadrature model |
| `mf_c0_root_integral_hardmax` | the continuous-maturity hard integral \(\tau_i^{-1}\int_0^{\tau_i} m_{\ell_c}(f_c(u;x))\,du\), evaluated by root-bracketed integration to a declared tolerance | exact relative to the declared integral/tolerance contract |

C1 is the natural first hard-bound counterfactual: it changes exactly one
operation in the current source. C0 and C1 are *not* the same posterior, and
the difference is structural, not merely numerical. Under a simple transverse
root of the bound gap, the moving-boundary terms of the C0 integral cancel
because the two integrand branches agree at the crossing, so
\(y_c^{C0}(\tau_i;x)\) is continuously differentiable in \(x\) there, whereas
the finite pointwise-max sum \(y_c^{C1}\) has kinks whenever any node crosses
the bound. Tangencies, endpoint roots, and an identically bound curve are
separate cases requiring their own treatment in the C0 integration contract.
Root-aware C0 evaluation and a C0-vs-C1 discretization sensitivity experiment
are required before any event/kink-aware HMC design is promoted as the final
hard-ZLB architecture; that experiment is Phase-2 implementation work and is
specified here, not run here.

Replace \(s_{\ell,\alpha}\) by \(m_\ell\) in (78) to obtain C1. Each yield becomes a
nonnegatively weighted sum of maxima of affine functions of the state,

\[
 y_c^{\max}(\tau_i;x)=\sum_{k=1}^{K}w_k
 \max\bigl\{\ell_c,\;a_c(s_{ik})^{\mathsf T}x^c\bigr\},
 \qquad s_{ik}=\tau_i v_k,
\tag{81}
\]

which is continuous, convex, and piecewise affine in \(x^c\). The FX row (79)
is then a difference of such functions plus an affine basis term: continuous
and piecewise affine, though no longer convex. The complete observation map
\(h\) is continuous and piecewise affine with kinks on the hyperplane
arrangement \(\bigcup_{c,i,k}\{x:a_c(s_{ik})^{\mathsf T}x^c=\ell_c\}\), a
Lebesgue-null set. Because the transition (75) is a fixed linear-Gaussian map
with no branch, and the observation noise is full-support Gaussian, the joint
posterior over \((\vartheta,x_{0:T})\) --- equivalently, over
\((\vartheta,x_0,\eta_{1:T})\) in the non-centered chart of (56) --- is
continuous, is smooth off a finite hyperplane arrangement, and has no atoms
and no density jumps. In the classification of Section 2.1 this is geometry
one: a kink target, not a jump target. Consequently ordinary Metropolized
leapfrog HMC is valid on the C1 target by Section 5.1, the
refraction machinery of Section 6 degenerates (\(\Delta U=0\)), and
discontinuous or mixed HMC address structure that this model simply does not
have. (The C0 target is smoother still at transverse roots, per the ladder
discussion above; the geometry-one classification covers both hard-bound
targets.)

The classification is a statement about the model *as coded*, and it is
fragile in three specific directions, each of which the owners may adopt
deliberately. If the factor dynamics switch at the bound --- a
binding-dependent \(\Phi\) or \(Q\) --- the transition acquires a genuine
branch and the target moves to geometry two, with everything that Sections
2.1, 6, and 12.1 say about deterministic jumps. If a policy rate is added as
an observation *without* measurement error, \(i_t=\max\{\ell,f_c(0;x_t)\}\)
observed exactly, the observation law acquires an atom at \(\ell\) and the
likelihood becomes the censored mixture of Section 4.3.3 with \(V\to0\). And
if the lower bound \(\ell\) is estimated, as in Lemke and Vladu (2017), the
kink location becomes parameter-dependent, which leaves continuity intact but
makes the kink arrangement move with \(\vartheta\). The contract must declare
which of these extensions, if any, is in scope; the geometry claim above holds
for none of them automatically.

One structural lemma sharpens everything downstream. Fix \(x^c\) and write
the bound gap along the horizon as

\[
 g(s)=f_c(s;x)-\ell_c=(L-\ell_c)+e^{-\lambda_c s}(S+C\lambda_c s),
 \qquad
 g'(s)=\lambda_c e^{-\lambda_c s}\bigl(C-S-C\lambda_c s\bigr),
\tag{82}
\]

with \((L,S,C)=x^c\). If \(L-\ell_c=0\), \(S=0\), and \(C=0\), the gap is
identically zero and the curve sits exactly on the bound at every horizon;
this degenerate case must be handled by convention (it is a Lebesgue-null
event under the Gaussian predictive but is reachable by fixtures). Away from
it, the bracket in \(g'\) is affine in \(s\), so \(g'\)
changes sign at most once; \(g\) is monotone or single-peaked/troughed on
\((0,\infty)\) and therefore has **at most two zeros**. The binding set
\(\{s:f_c(s;x)\leq\ell_c\}\) along any Nelson-Siegel forward curve is an
interval, the complement of an interval, empty, or everything. Two
consequences follow. The reachable binding patterns over the pooled quadrature
horizons \(\{s_{ik}\}\) are *interval patterns*, \(O(\text{grid}^2)\) many
rather than \(2^{\text{grid}}\); and the *per-country* kink cells in factor
space are polyhedra cut by constraints whose normals \(a_c(s)\) span at most
that country's three factor dimensions, so per-country cell probabilities
reduce to at most three-dimensional Gaussian integrals. The *joint*
two-country cell probability required by the evidence decomposition below is
generically six-dimensional: the FX row and the measurement update correlate
the two factor blocks, so the joint mass factorizes into two
three-dimensional integrals only if a factorization of the relevant
covariance is proved for the case at hand.

### 13.4 The bootstrap authority and the ideal fully adapted proposal identity

The claim-bearing likelihood authority for `mf_c1_k40_hardmax` is the
**bootstrap particle filter**: per particle, draw the state from the exact
Gaussian transition and weight by the exact observation density,

\[
 x_t^j\sim N\bigl((I-\Phi)\bar\theta+\Phi x_{t-1}^{A_t^j},\,Q\bigr),
 \qquad
 v_t^j=N\bigl(y_t;\,h_{\max}(x_t^j),\,R\bigr),
\tag{82a}
\]

where \(h_{\max}\) stacks the C1 yield rows (81) and the FX row (79). Both
operations are available in closed form, the estimator (13) is nonnegative
and unbiased under the standard support and integrability conditions of
Section 4.1, and PMMH then carries the ordinary particle-MCMC extended-target
guarantee of Section 10.1. This route is less elegant than the cell
decomposition below but is implementable today, unbiased for the named finite
model, and honest about its Monte Carlo variance.

The cell decomposition is nonetheless the structure the model has, and it is
recorded here as an **ideal fully adapted proposal identity** --- a valid
mathematical decomposition whose literal implementation requires numerical
components that are not closed-form operations. Conditional on a binding pattern \(B\) --- an assignment of bind/slack to
every quadrature horizon --- the observation map is affine,
\(h_B(x)=H_Bx+d_B\), on the polyhedral cell

\[
 C_B=\bigl\{x:\;a_c(s_{ik})^{\mathsf T}x^c\leq\ell_c\ \text{for}\
 (c,i,k)\in B,\ \ >\ell_c\ \text{otherwise}\bigr\}.
\tag{83}
\]

Let the one-step predictive be Gaussian, \(x_t\mid\mathcal F\sim N(m,P)\) ---
exactly true per particle, since (75) gives \(x_t\mid x_{t-1}\sim
N((I-\Phi)\bar\theta+\Phi x_{t-1},Q)\). The exact one-step posterior and
evidence then decompose over cells with no approximation:

\[
 p(x_t\mid y_t,\mathcal F)\;\propto\;
 \sum_{B}\mathbf 1\{x_t\in C_B\}\,
 N\!\bigl(y_t;H_Bm+d_B,\,S_B\bigr)\,
 N\!\bigl(x_t;m_B^+,P_B^+\bigr),
\tag{84}
\]

where each cell carries the standard conjugate update

\[
 S_B=H_BPH_B^{\mathsf T}+R,\quad
 K_B=PH_B^{\mathsf T}S_B^{-1},\quad
 m_B^+=m+K_B(y_t-H_Bm-d_B),\quad
 P_B^+=P-K_BS_BK_B^{\mathsf T},
\tag{85}
\]

in which \(S_B\) is invertible for every cell because the measurement
covariance \(R\) is positive definite and \(H_BPH_B^{\mathsf T}\) is positive
semidefinite, and the exact evidence is the polyhedrally truncated mixture
mass

\[
 p(y_t\mid\mathcal F)=\sum_B Z_B,
 \qquad
 Z_B=N\!\bigl(y_t;H_Bm+d_B,\,S_B\bigr)\,
 \Pr_{x\sim N(m_B^+,P_B^+)}\!\bigl(x\in C_B\bigr).
\tag{86}
\]

Equations (84)--(86) are the hard-bound observation-map counterpart of the
branch-split update (36)--(45): the branch structure has moved from the
transition maps \(F_r\) into the observation cells \(C_B\), the truncated
moments of (21)--(28) reappear whenever a single constraint dominates, and by
the crossing lemma the sum over \(B\) runs over interval patterns. Three
caveats separate this identity from an implementable exact algorithm. First,
the cell masses \(\Pr_{x\sim N(m_B^+,P_B^+)}(x\in C_B)\) are multivariate
Gaussian polytope probabilities --- generically six-dimensional across the
two correlated factor blocks per the crossing-lemma discussion --- and these
are numerical computations in the sense of Genz and Bretz (2009), not
closed-form operations. Second, restricting the sum to patterns with
"non-negligible" mass is a *pruning approximation*: dropping any cell changes
\(\sum_B Z_B\), the cell-selection probabilities, and the weight, so the
result is exact only if the omitted masses are certified to be exactly zero.
Third, what (84)--(86) do *not*
give is a closed-form multi-step filter: propagating a cell-restricted
Gaussian through (75) leaves the Gaussian family, so an exact density filter
must carry a growing truncated-mixture and any pruning or moment collapse
(47)--(48) is a declared approximation, exactly as in Section 4.3.4.

Inside a particle filter the identity yields an *ideal* fully adapted
construction. Per particle \(j\) with ancestor
\(x_{t-1}^j\), the predictive is exactly Gaussian, so (84)--(86) hold exactly
with \((m,P)=((I-\Phi)\bar\theta+\Phi x_{t-1}^j,\,Q)\), and the conditionally
optimal proposal is, in the ideal,

\[
 q^\star(x_t\mid x_{t-1}^j,y_t)=p(x_t\mid x_{t-1}^j,y_t),
 \qquad
 v_t^j=p(y_t\mid x_{t-1}^j)=\sum_B Z_B\bigl(x_{t-1}^j\bigr),
\tag{87}
\]

sampled by drawing a cell with probability \(\propto Z_B\), then drawing from
the cell-restricted Gaussian \(N(m_B^+,P_B^+)\mathbf 1_{C_B}\). The
point-independence of the incremental weight (87) is the fully adapted
property of (12b)--(13), and *if* every \(Z_B\) is computed exactly and the
cell-restricted draw is exact, the resulting likelihood estimator is
nonnegative and unbiased for the C1 model by the argument of Section 4.1.
Neither ingredient is free. Exact cell-restricted draws by rejection sampling
are exact in principle but can be unusably inefficient in the tails; the
exact piecewise-quadratic HMC of Pakman and Paninski (2014) supplies an
*invariant kernel* for the truncated Gaussian, not an independent exact draw
after a finite trajectory, so substituting one trajectory for an exact draw
changes the proposal law and forfeits the constant weight (87); and Botev's
(2017) minimax-tilting method is the modern benchmark for both the
probabilities and the draws but must be inspected and its error included in
the proposal contract before any exactness language attaches to a cell
implementation. Until those numerical components are controlled, this
construction is a *proposal and diagnostic layer*: an implementation that
prunes cells or approximates draws must use its actual proposal density and
the corresponding importance ratio in (12b), not the ideal constant weight
(87). The bootstrap route (82a) remains the claim-bearing authority. This
positioning is the Aruoba et al. conditionally-optimal construction
transplanted from a piecewise-linear transition to a hard-bound observation
map --- simpler here because the transition never branches, but gated on the
same numerical honesty.

The Tobit line of the censored-filtering literature must be positioned
carefully against this. In the Tobit-type-1 measurement model surveyed by
Geng et al. (2021), censoring is applied *after* the noise,
\(y=\max\{c,\,Hx+u\}\), which produces an atom at the censoring limit and is
the setting of the Tobit Kalman filter (Allik et al. 2016,
metadata-verified). MacroFinance's yields add noise *after* the max, so they
have no atom, and the mixture structure lives in the state, not in the
observation support. The two orderings coincide only in the exactly observed
policy-rate extension of Section 13.3, which is where the Tobit machinery
would become the right reference.

### 13.5 The softplus route: bias bounds and two readings of alpha

The pointwise gap between the coded map and the hard bound is explicit:

\[
 0\;<\;s_{\ell,\alpha}(u)-m_\ell(u)
 =\alpha\log\!\left(1+e^{-|u-\ell|/\alpha}\right)
 \;\leq\;\alpha\log 2,
\tag{88}
\]

with the maximum attained exactly at the kink and exponential decay away from
it: at \(|u-\ell|\geq5\alpha\) the gap is below \(6.8\times10^{-3}\,\alpha\).
Averaging (88) through the quadrature (78) bounds the yield gap uniformly,

\[
 0\;<\;y_c^{\alpha}(\tau_i;x)-y_c^{\max}(\tau_i;x)\;\leq\;\alpha_c\log 2,
\tag{89}
\]

and the bound is attained only when the whole forward path sits at the kink;
in general the gap is \(\alpha\log2\) times the quadrature mass within
\(O(\alpha)\) of the bound plus an exponentially small remainder. With the
coded constants the worst-case yield gaps are \(1.04\times10^{-3}\) (domestic)
and \(0.69\times10^{-3}\) (foreign) in annualized rate units --- roughly ten
and seven basis points --- and, crucially, the gap is *one-signed*: softplus
yields always exceed hard-censoring yields, so fitted factors and long-run
means absorb a systematic downward shift near binding rather than symmetric
noise. The per-observation log-likelihood perturbation obeys

\[
 \bigl|\log N(y;h_\alpha,R)-\log N(y;h_{\max},R)\bigr|
 \;\leq\;
 \bigl\|R^{-1/2}(y-h_{\max})\bigr\|\,
 \bigl\|R^{-1/2}\Delta\bigr\|
 +\tfrac12\bigl\|R^{-1/2}\Delta\bigr\|^2,
\tag{90}
\]

with \(0\le\Delta_i\le\alpha\log2\) componentwise, which is small per
observation. The signed difference itself is, for
\(\Delta=h_\alpha-h_{\max}\),

\[
 \log N(y;h_\alpha,R)-\log N(y;h_{\max},R)
 =(y-h_{\max})^{\mathsf T}R^{-1}\Delta
 -\tfrac12\,\Delta^{\mathsf T}R^{-1}\Delta,
\tag{90a}
\]

and this can take *either sign* even though the map shift \(\Delta\) is
one-sided on the yield rows: the sign depends on the residual
\(y-h_{\max}\), and the FX row's \(\Delta\) component is a
domestic-minus-foreign combination that can itself be signed either way. The
honest statement is therefore that the *mean-map* discrepancy is one-sided
and may induce a systematic fitted-factor adjustment near binding, while the
likelihood and posterior perturbations must be reported as computed signed
quantities under (90)--(90a), not asserted to accumulate with one sign. With
measurement-error standard deviations of a few
basis points, a ten-basis-point systematic map gap is not negligible relative
to the noise scale, and (90)--(90a) are the honest currency in which to
report it.

Sharpening \(\alpha\) is not free for gradient samplers. The scalar curvature
is

\[
 s_{\ell,\alpha}''(u)=\frac{\sigma(z)\bigl(1-\sigma(z)\bigr)}{\alpha}
 \leq\frac1{4\alpha},
 \qquad z=\frac{u-\ell}{\alpha},
\tag{91}
\]

so the observation Hessian acquires ridges of height \(O(1/\alpha)\)
concentrated in an \(O(\alpha)\)-neighborhood of the kink, and leapfrog
stability suggests a local step size \(\epsilon=O(\sqrt\alpha)\) there. That
scaling is a local stability heuristic under a fixed metric and comparable
posterior curvature, not a universal necessity theorem: the binding step size
depends on the chart, the mass matrix, the trajectory, and how much time the
sampler spends near the bound. The qualitative conclusion survives the
qualification: the
\(\alpha\to0\) limit of the smooth route is *harder* for HMC than
the C1 hard-bound model itself, whose one-sided curvature is bounded and
whose kink cost is the \(O(\epsilon)\) energy error of (56a). Sharpening the
softplus to approximate the hard bound is the wrong direction; the right
choices are either a genuinely smooth model or the exact machinery of
Section 13.4.

That leaves two defensible readings of \(\alpha\), which must not be mixed.
Read as a *numerical approximation* to the hard-ZLB model, \(\alpha\) needs an
error protocol: report (89)--(90) alongside binding fractions, rerun matched
fixtures at \(\alpha/2\) and \(2\alpha\) (the existing
`dz5_alpha_counterfactual` fixtures implement exactly this variation), and
require declared posterior functionals to move by less than their Monte Carlo
uncertainty before any hard-ZLB language is used. Read as a *structural
smooth-transition model* in the sense argued empirically by Opschoor and van
der Wel (2024) --- yields leave the bound gradually, so the smooth map may be
the better description --- \(\alpha\) is a model constant or even an
estimable parameter, no bias language applies, and every claim must name the
smooth model as the target. What remains forbidden under both readings is the
current silent hybrid: sampling the smooth model while describing the result
as inference under the ZLB.

### 13.6 Identification at the bound

Differentiating (78) gives the yield sensitivity to the factors,

\[
 \frac{\partial y_c(\tau_i;x)}{\partial x^c}
 =\sum_{k=1}^{K}w_k\,
 \sigma\!\left(\frac{f_c(s_{ik};x)-\ell_c}{\alpha_c}\right)
 a_c(s_{ik}),
\tag{92}
\]

with the logistic factors replaced by indicators \(\mathbf 1\{f>\ell\}\) in
the C1 hard-bound model. When the forward curve sits far below the bound
over the horizons that matter for maturity \(\tau_i\), the corresponding rows
of the observation Jacobian \(J\) vanish, and the per-period observed Fisher
information \(J^{\mathsf T}R^{-1}J\) about the shadow factors degenerates in
exactly the binding directions. During a long binding spell the shadow-factor
posterior is driven by the prior and the transition (75) alone, and the
long-run means \(\bar\theta\) of the binding country are informed only through
occasional excursions of the forward curve above the bound. This is the
formal counterpart of two inspected empirical findings: Bauer and Rudebusch
(2016) document that shadow-rate estimates are highly sensitive to the assumed
lower-bound value and to model specification, and Christensen and Rudebusch
(2015) report the same sensitivity for shadow-rate paths. The consequences
for the MacroFinance campaign are direct: priors on \(\bar\theta\) do real
work during binding spells and must be reported as such; the HMC mass matrix
will be ill-conditioned along binding-era directions, which is a tuning-scope
fact under the repository's per-scope tuning rule; and identification
diagnostics of the existing `dz5_identification` lineage are a required part
of any claim, not an optional extra.

### 13.7 Four routes and what each can claim

The contract admits five inference routes, which differ in target and in
authority, not merely in cost.

| Route | Target | Status and allowed claims |
|---|---|---|
| (i) SR-UKF marginal likelihood + HMC/NeuTra (active route) | The UKF likelihood as a declared approximate functional of `mf_s1_k40_softplus` | Exact MCMC for its own declared target; domain precedent is Kim and Priebsch (2013). Its single-Gaussian closure discards branch mass exactly as in Sec. 4.3.1; its distance from route (iii) on matched binding fixtures is a required diagnostic, not an assumption. |
| (ii) Softplus model with any exact filter/sampler | `mf_s1_k40_softplus` | Legitimate as a structural model (Opschoor--van der Wel reading) or as an approximation under the (89)--(90a) protocol; never a hard-ZLB claim by itself. |
| (iii) Bootstrap hard-bound observation-map particle filter (82a) + PMMH | `mf_c1_k40_hardmax` | The likelihood authority: exact Gaussian transition draws, exact observation-density weights, nonnegative unbiased estimator, exact extended-target MCMC by Sec. 10.1. Claim-bearing for the finite hard-quadrature posterior. Supports PMMH, not pseudo-marginal HMC, absent the Sec. 10.2 differentiability conditions; correlated pseudo-marginal (Deligiannidis, Doucet, and Pitt 2018) is the variance/mixing comparator, not a differentiability fix. |
| (iv) Joint \((\vartheta,x_0,\eta_{1:T})\) HMC on `mf_c1_k40_hardmax` | The C1 model, sampled in the non-centered chart of (56) | Valid by Sec. 5.1 because the target is a continuous piecewise-quadratic-plus-smooth kink target; needs no filter at all. Gradient cost is one simulation sweep; the linear-Gaussian prior structure supplies the natural preconditioner, and MacroFinance's existing large-scale LGSSM HMC lineage is the engineering base. Kink crossings cost acceptance per (56a), not correctness. |
| (v) Cell-adapted fully adapted proposal (84)--(87) inside a particle filter | `mf_c1_k40_hardmax` | Conditional: admissible as a claim-bearing route only after the Genz--Bretz polytope-probability and Botev truncated-draw components are inspected, benchmarked, and carried with declared numerical error in the proposal contract; with pruning or approximate draws it must use its actual proposal density and importance ratio, and it is a proposal/diagnostic layer for (iii) until then. |

The branch-split mixture UKF of Section 4.3 remains what it was: a fast
diagnostic and a proposal ingredient, never a claim-bearing
likelihood. Routes (iii) and (iv) target the same C1 model and must agree
within Monte Carlo error on fixtures; that cross-check is the cheapest
end-to-end correctness test the campaign can buy, and it is executable at the
existing recovery-fixture scale before any production run. The
`mf_c0_root_integral_hardmax` target enters through the Sec. 13.3
discretization sensitivity experiment before any event/kink-aware design is
promoted.

## 14. A worked contract for dsge_hmc

**Correction (19 August 2026, second revision).** Earlier drafts of this
section asserted that `dsge_hmc` did not exist. That is false as a workspace
statement: the originally named path `/home/ubuntu/workspace/dsge_hmc` is absent, but
the package exists at `/home/ubuntu/workspace/python/src/dsge_hmc` with a full model,
solver, filter, and estimation tree. What does *not* exist is a true OBC/ZLB
model inside it: the currently validated BGS restricted surface implements the
notional policy rule for `rn`
(`models/bgs_restricted_surface_generated.py:225`) and then closes the
constrained rate with the linear placeholder `r = rn` (line 228), with both
rows tagged `OBC_ZLB_NO_RUN_GUARD` in
`models/bgs_restricted_surface_tf_coefficients.py:17`. The package's own
master program states that no OBC/ZLB logic is active in the witness
likelihoods and its evidence contract explicitly excludes OBC/ZLB estimation
(`/home/ubuntu/workspace/python/docs/plans/actual-bgs-restricted-surface-port-master-program-v2-2026-07-09.md`).
This section is therefore two things, kept separate below: Sections
14.1--14.4 are a **pedagogical layer** --- a generic three-equation NK/LCP
analysis of where genuine discontinuity lives in an estimated ZLB DSGE model
--- and Section 14.5 combined with Section 14.6 is the **BGS layer**: the
contract that a true BGS OBC/ZLB target must satisfy before any sampler code.
The NK toy's solution geometry must not be transferred to the 46-parameter
BGS surface without deriving the active equations.

The pedagogical punchline is precise: with
a unique verified solution, the piecewise-linear ZLB model is a *kink* target
in states and shocks, and the genuine discontinuities practitioners meet come
from the solution *correspondence* --- nonexistence and multiplicity of
verified regime paths --- and from the selection rule imposed on it.

### 14.1 Solution operators for the benchmark class (pedagogical)

The benchmark model class is the three-equation New Keynesian model with a
Taylor rule truncated at the bound, written in the notation of (1)--(3), and
solved by a piecewise-linear method. Three inspected solution operators
exist. OccBin (Guerrieri and Iacoviello 2015, Sec. 2) guesses a regime path,
solves the two linear systems backward from a terminal reference-regime date,
simulates forward, and iterates until the implied inequalities verify the
guess; the survey's (6)--(11) is its conditional filtering counterpart.
Holden (2016) reformulates the bound as anticipated news shocks added to the
unconstrained linear solution and implements the search in DynareOBC; the
anticipated-shock route originates with Holden and Paetz (2012,
metadata-cited). Boehl
(2022) derives a closed-form expression for the whole trajectory given the
guessed spell durations, which reduces the iteration to a fast search over two
integers. All three produce, when they converge to a verified path, the same
object: a solution that is linear on each regime cell of the
state-shock space.

Holden (2023, Secs. 2--3) supplies the algebra that makes the solution
correspondence analyzable. Stack the bound violations of the unconstrained
solution into \(q\in\mathbb R^T\) and let \(M\in\mathbb R^{T\times T}\) collect
the responses of the bounded variable to unit news shocks; imposing the bound
over a horizon of \(T\) periods is the linear complementarity problem

\[
 y\geq0,\qquad q+My\geq0,\qquad y^{\mathsf T}(q+My)=0,
\tag{93}
\]

whose solution \(y\) gives the news-shock loadings that hold the variable at
the bound during the binding spell. Existence and uniqueness of the verified
path are exactly existence and uniqueness for (93); Holden's central result is
that the solution is unique for *every* \(q\) if and only if \(M\) is a
P-matrix (all principal minors positive), and that standard New Keynesian
models fail this condition, having multiple perfect-foresight paths that
escape the bound.

The P-matrix property is a uniqueness gate, and it must not be silently
promoted to a continuity claim. Pointwise uniqueness of the LCP solution does
not by itself establish that the solution map
\(\vartheta\mapsto y(q(\vartheta),M(\vartheta))\) is continuous. The
regularity statement usable by the contract is parametric: for a *fixed
finite horizon* \(T\) and terminal condition, with \(q(\vartheta)\) and
\(M(\vartheta)\) continuous on a declared parameter domain over which
\(M(\vartheta)\) remains a P-matrix, the unique LCP solution is continuous in
\(\vartheta\) by standard parametric-LCP regularity. Any of a changing
horizon, a changed terminal condition, or loss of the P-matrix property on
the domain voids the conclusion and can create genuine discontinuity even
under pointwise uniqueness; the contract must state which of these conditions
has been verified, on which domain.

### 14.2 A fully derived multiplicity example and the discontinuity mechanism (pedagogical)

The smallest model that shows the mechanism is Holden's (2023, Sec. 2.2)
lagged-response economy, reproduced here in full because the contract leans
on it. The Fisher equation with a constant real rate \(r>0\) and a truncated
Taylor rule with lagged response give

\[
 r+\pi_{t+1}=i_t=\max\{0,\;r+\phi\pi_t-\psi\pi_{t-1}\},
 \qquad \phi-\psi>1,\ \psi\in(0,1),
\tag{94}
\]

with \(\pi_0\) given. Away from the bound the stable solution is
\(\pi_t=A\pi_{t-1}\) with \(A^2-\phi A+\psi=0\); taking \(\phi=2\) for
transparency, \(A=1-\sqrt{1-\psi}\in(0,1)\) and the useful identity
\(A(\phi-A)=\psi\) holds. Holden shows the economy can be at the bound for at
most the first period on any path that escapes, so \(\pi_2=A\pi_1\), and
substituting into (94) at \(t=1\) leaves a scalar instance of (93):

\[
 0=\max\bigl\{-r-A\pi_1,\;(\phi-A)\pi_1-\psi\pi_0\bigr\}.
\tag{95}
\]

Both branches can be solved explicitly. The slack branch gives the
fundamental solution \(\pi_1^{\mathrm f}=\psi\pi_0/(\phi-A)\), admissible when
\(r+A\pi_1^{\mathrm f}\geq0\), that is when \(\pi_0\geq-r/A^2\) by the
identity above. The binding branch gives \(\pi_1^{\mathrm b}=-r/A\),
admissible when \((\phi-A)\pi_1^{\mathrm b}\leq\psi\pi_0\), which is the
*same* condition \(\pi_0\geq-r/A^2\). Therefore

\[
\begin{aligned}
 \pi_0>-\frac{r}{A^2}&:\ \text{two solutions},\\
 \pi_0=-\frac{r}{A^2}&:\ \text{one},\\
 \pi_0<-\frac{r}{A^2}&:\ \text{none returning to the standard steady state}.
\end{aligned}
\tag{96}
\]

Multiplicity is not a knife-edge event here: it covers an open half-line of
initial states, and Holden proves it becomes pervasive in richer New
Keynesian models. Each branch map is *continuous* in \((\pi_0,r,\psi)\) ---
\(\pi_1^{\mathrm f}\) is linear in \(\pi_0\), \(\pi_1^{\mathrm b}\) is
constant --- so the discontinuity is created entirely by *selection*: any
deterministic rule \(\sigma\) that picks a branch defines a transition kernel

\[
 \pi_1=T_{\sigma(\pi_0,\vartheta)}(\pi_0,\vartheta),
 \qquad
 T_{\mathrm f}\neq T_{\mathrm b}\ \text{on the whole multiplicity region},
\tag{97}
\]

which jumps by an \(O(1)\) amount wherever \(\sigma\) switches branch. A
likelihood built on (97) inherits that jump in the state, and --- because the
multiplicity boundary \(-r/A(\psi)^2\) and any solver-dependent selection move
with \(\vartheta\) --- also in the parameters. This is the formal content of
the folklore that piecewise-linear ZLB likelihoods "spike": the spike is not
caused by the max, whose solution branches are individually continuous, but
by an undeclared, possibly solver-order-dependent selection rule on a
positive-measure multiplicity region, plus outright nonexistence below the
threshold. In the taxonomy of Section 2.1 the object is geometry four until a
selection law is declared; after a deterministic declaration it is geometry
two along the selection switch set; and only the stochastic completion below
makes it geometry three, where the mixed-support kernels of Sections 8 and
10.4 legitimately apply.

### 14.3 The stochastic completion and the sunspot literature (pedagogical)

The statistically coherent repair is to make selection part of the model.
Attach to every multiplicity region a probability law over its branches ---
in (95), a probability \(p_t\) of the bound branch, possibly Markov in a
latent sunspot state \(s_t\) --- and the target becomes the mixed-support
posterior (73) with \(p(s_t\mid s_{t-1},x_{t-1},\vartheta)>0\) on both
branches. The construction machinery for linear models under indeterminacy is
inspected and local: Lubik and Schorfheide (2003) show that under
indeterminacy the endogenous forecast errors are not uniquely determined by
fundamentals, and sunspot shocks enter as additional structural disturbances
with estimable loadings. The published instance for the ZLB itself is Aruoba,
Cuba-Borda, and Schorfheide (2018, metadata-verified; full text not in the
local corpus), who build a New Keynesian model whose targeted-inflation and
deflation regimes are selected by a Markov sunspot and estimate it with a
particle filter; Holden (2023) explicitly contrasts his within-steady-state
multiplicity with that steady-state-switching literature. For the dsge_hmc
contract the consequence is structural: the sunspot completion is the *only*
reading under which regime-path Gibbs, PGAS, or mixed HMC are valid, because
only there does the regime have positive conditional probability on both
values; under a deterministic selection they collapse to the zero-support
situation of (5a).

### 14.4 Estimation baselines to beat (pedagogical)

Three inspected filter families define the current practice for this class,
and all are approximations relative to the particle authority of Section 4.
The piecewise Kalman filter of Section 3 conditions on the verified regime
path. Boehl and Strobel (2023) estimate medium-scale ELB models with an
ensemble Kalman filter --- ensemble members updated by linear shifting rather
than reweighting --- on top of the Boehl (2022) solver, and compare against
the inversion filter; like every Gaussian-closure method, the EnKF inherits
the branch-mass deficiency of Section 4.3.1 at the bound, now with sampling
noise on top. Holden (2017) proposes an extended skew-t cubature Kalman
filter with dynamic state-space reduction, tracking third and fourth moments
precisely because the Gaussian closure loses the censoring asymmetry.
The dsge_hmc benchmark should therefore report PKF, an EnKF arm, an
inversion-filter arm (Cuba-Borda, Guerrieri, Iacoviello, and Zhong 2019,
metadata-cited; primary-source inspection is an expansion requirement before
the empirical baseline freezes), and the
bootstrap particle filter on the same fixtures, with the particle
route as authority, mirroring the Section 13.7 table.

### 14.5 The contract checklist

No dsge_hmc OBC/ZLB sampler code is admissible before the following are
frozen in writing, in the notation of (1)--(3): the bound and the truncated
rule; the expectation formation and terminal condition; the solution operator
with its verification loop; the LCP matrix \(M\) of (93) for the chosen
horizon with a computed P-matrix verdict on the estimation-relevant parameter
region, so that the uniqueness domain is known rather than hoped; the
selection law on every multiplicity region --- deterministic and declared, or
stochastic with its sunspot process --- and the treatment of nonexistence
regions in the likelihood; the measurement equation and error structure; and a
smoke-scale enumeration toy on which the PKF, the EnKF arm, and the particle
authority can be compared against exact enumeration. Phases 0--3 of
Section 16 then apply verbatim, with the geometry classification of Section
2.1 executed on the *frozen* contract rather than on the informal model
description.

### 14.6 The actual BGS layer: package status and architecture gap

The contract of Section 14.5 must be discharged against the actual BGS
source, not against the pedagogical NK toy. The starting facts, verified by
inspection on 19 August 2026, are:

- the package exists at `/home/ubuntu/workspace/python/src/dsge_hmc` with `models/`,
  `solvers/`, `filters/`, `estimation/`, `validation/`, and Stan-reference
  trees;
- the validated BGS restricted surface implements the notional rule for
  `rn` and the linear no-binding placeholder `r = rn`
  (`models/bgs_restricted_surface_generated.py:225` and `:228`), with the two
  rows tagged `OBC_ZLB_NO_RUN_GUARD`
  (`models/bgs_restricted_surface_tf_coefficients.py:17`);
- the package's master program records that no OBC/ZLB logic is active in the
  witness likelihoods and that its evidence contract excludes OBC/ZLB
  estimation, and its Phase-10 subplan states that true OBC/ZLB estimation
  requires a new architecture.

Before proposing a sampler for a true BGS ZLB target, the following must be
documented from source: (i) the actual BGS source and version, lower-bound
level, policy equation, expectations convention, terminal condition, and
measurement system; (ii) every active-constraint alteration, including QE
feedback and lagged-state carriers; (iii) a source-anchored solution operator
and its verification loop; (iv) the uniqueness/nonexistence domain and a
selection law on multiplicity regions; (v) the likelihood value assigned when
no verified path exists; and (vi) source/Dynare path parity before any
particle or HMC promotion. The first BGS deliverable is therefore a
transition/solution kernel, not a particle filter. Once that kernel is
defined, inversion filtering, the piecewise Kalman filter, EnKF, the
bootstrap particle filter, and any gradient method can be compared on the
same declared target. The existing restricted-surface parity program remains
a valuable no-binding baseline and must not be silently relabelled as ZLB
evidence.

## 15. Solver-induced and code-induced discontinuities

The preceding sections classify discontinuity in the *model*. A second,
independent family lives in the *evaluation* of the model, and it matters for
gradient samplers even when the declared target is smooth. The distinction:
a model-level jump changes the target and belongs to Sections 2.1 and 12;
an evaluation-level branch leaves the intended target smooth but makes the
*computed* log density or its gradient discontinuous.

Three inspected mechanisms occur in the application repositories. First,
matrix-function kernels with integer branch decisions: MacroFinance's
continuous-time lineage evaluates matrix exponentials by Padé approximation
with scaling-and-squaring, and the integer scaling count jumps at norm
thresholds; the repository's own diagnostic
(`ccma_g_v7_pade_frechet_diagnostic.py`) freezes the scaling integer when
differentiating and explicitly declares that no smooth-derivative claim holds
across the branch. At such a branch the *value* is continuous up to the
approximation tolerance while the automatic derivative can jump by a finite
amount. Second, factorization conventions: QR and SVD sign and ordering
choices, and square-root filter downdates near rank changes, are
branch-discontinuous functions of their inputs even though the objects they
represent vary smoothly. Third, almost-everywhere gradients: reverse-mode
differentiation of `max`, `where`, `abs`, and sorting picks one subgradient
at ties, which is harmless for validity by Section 5.1 *provided the value is
continuous*.

The HMC consequences separate cleanly along that proviso. If the computed
log density is continuous and only its computed gradient jumps at evaluation
branches, then the situation is exactly the kink case: the Metropolis
correction keeps the declared target invariant, and the cost is the
\(O(\epsilon\|\Delta\nabla U\|)\) acceptance penalty of (56a) at branch
crossings. If the computed *value* itself jumps --- a solver switching
branches beyond its tolerance, an iteration truncated by a state-dependent
stopping rule, a fixed-point loop finding a different verified path --- then
the sampler is targeting a discontinuous computed density it does not know
about: rejections concentrate on branch boundaries, apparent reversibility
holds only up to the jump size, and no diagnostic inside the sampler
separates this from a genuine model jump. The requirement for serious runs is
therefore stated at the evaluation level:

\[
 \sup_{\text{branch boundaries}}
 \bigl|\log\widehat p_{\,\text{branch}\,1}-\log\widehat p_{\,\text{branch}\,2}\bigr|
 \;\ll\;1
\tag{98}
\]

in Metropolis energy units, verified by test, together with branch-stable
derivative implementations (the repository's fixed-scaling matrix-exponential
work is the template), deterministic tie-breaking, and value-continuity tests
across every known integer branch: scaling orders, factorization signs,
regime-path verification loops, and truncation horizons. These are BayesFilter
test obligations, not sampler tuning knobs, and Section 16's gates assume
they pass.

## 16. Project roadmap and evidence gates

### Phase 0: target and source contract

Write the model in the notation of (1)--(3), identify whether the object is a
kink, a density jump, or an explicit discrete path, and specify how multiple
equilibrium solutions are treated. No sampler code is admitted before this
contract is complete. Section 13 discharges this phase for the MacroFinance
shadow-rate model; Section 14.5 is the checklist the future dsge_hmc model
must discharge, including a computed LCP uniqueness verdict and a declared
selection law.

### Phase 1: linear path and likelihood tie-out

Reproduce the Dynare/OccBin regime path and PKF likelihood. Compare every
innovation, covariance, regime inequality, and log-likelihood contribution with
the source implementation. The promotion criterion is same-object agreement on
synthetic data; fixed-point nonuniqueness is a veto against silent promotion.

### Phase 1a: the standard test-model ladder (added 2026-08-21)

Every phase below verifies code against a reference problem whose answer is
known independently. The manuscript's Table 3 assembles the standard ladder;
this is its record, ordered by the source of the independent answer:

| Tier | Model | Independent answer / what it isolates |
|---|---|---|
| 0 | linear-Gaussian state-space model | closed-form Kalman likelihood; every PF/sigma-point/joint-HMC route must reproduce it before any bound is added |
| 0 | conjugate parameter posterior with known states | analytic posterior; isolates the parameter sampler |
| 1 | scalar censored-observation model, eqs. (29)--(35) | closed-form mixture likelihood; atom handling, branch masses, softplus bias (90)--(90a) against truth |
| 1 | affine-boundary truncated Gaussian, eqs. (21)--(28) | closed-form moments; unit tests for truncation steps |
| 1 | truncated MVN sampling | Pakman--Paninski exact HMC and Botev tilting reference draws; cell-restricted proposal correctness |
| 2 | two-state switching linear-Gaussian model, horizon T<=12 | exact likelihood/smoothing by 2^T path enumeration; PF unbiasedness, PGAS invariance, mixed-HMC path moves, deterministic-regime zero-support rejection |
| 2 | univariate nonlinear growth model (Gordon--Salmond--Smith 1993) | dense-grid filter reference; canonical PF stress case |
| 3 | one-factor shadow-rate model (Gorovoi--Linetsky 2004) | closed-form bond prices under a hard bound; analytic anchor for the Sec. 13 case study |
| 3 | one-node quadrature reduction of `mf_c1_k40_hardmax` (K=1) | collapses to the Tier-1 censored-scalar likelihood; exact posterior by grid integration; routes (iii)/(iv) must agree with the grid before K=40 |
| 3 | lagged-rule economy, eqs. (94)--(96) | analytic multiplicity boundary -r/A^2; nonexistence/selection-rule detection |
| 3 | published OccBin examples (Guerrieri--Iacoviello 2015) | Dynare reference output; regime-path and PKF bookkeeping |
| 3 | small-scale NK ZLB model (Aruoba et al. 2021) | published COPF comparisons; conditionally optimal proposal |
| 4 | joint-distribution tests (Geweke 2004) | simulator-vs-sampler comparison of the same joint law; no analytic posterior needed |
| 4 | simulation-based calibration (Talts et al. 2018) | rank-uniformity across prior-simulated data sets; exercises binding/nonbinding in prior proportion |

Gordon--Salmond--Smith, Gorovoi--Linetsky, Geweke, and Talts et al. are
metadata-cited; the models and harnesses are standard and their use here
imports no equation-level content. A promoted route passes every applicable
rung, and rungs remain as regression tests thereafter.

### Phase 2: nonlinear filtering authority

Implement the bootstrap particle filter, retained particle ancestry, likelihood
replicates, PGAS, and the COPF special case. Check the likelihood estimator
against exact enumeration in a tiny model, then measure its variance across
particle counts. For MacroFinance, the authority is the bootstrap
hard-bound observation-map filter (82a) for the `mf_c1_k40_hardmax` target;
the ideal cell decomposition (84)--(87) and the branch-split mixture UKF of
Section 4.3 are its diagnostic and proposal layer, with unit tests against the
affine closed forms (21)--(28), and the cell route is promotable only under
the Section 13.7(v) conditions. Root-aware `mf_c0_root_integral_hardmax`
integration and the C0-vs-C1 sensitivity comparison of Section 13.3 belong to
this phase, before any event/kink-aware design is selected. A
differentiable particle-filter experiment is
diagnostic only until its extended target and Hamiltonian correction are
derived.

### Phase 3: target-specific exact kernels

For a stochastic regime model, implement fixed-path HMC plus Metropolis or PGAS
path moves and validate its invariant distribution against exact enumeration.
For a deterministic threshold model, implement a non-gradient joint-state or
particle-MCMC baseline and verify that regime-only moves are rejected as the
support calculation predicts. For the MacroFinance kink target, the two exact
routes are (iii) and (iv) of Section 13.7, and their agreement on matched
fixtures is the promotion gate; solver-branch value-continuity tests per
Section 15 are a precondition for the gradient route. Do not reuse evidence
between these targets.

### Phase 4: event-aware continuous HMC

Only if Phase 0 identifies an explicit continuous switching surface, implement
RHMC/DHMC event handling for the deterministic target. Test first-crossing
accuracy, one-sided full-posterior evaluation, reflection/refraction energy
conservation, reversibility, and boundary grazing. A boundary locator failure
is a hard veto, not an invitation to smooth the surface.

### Phase 5: scale and application

Apply the selected route to MacroFinance under the Section 13 contract and its
route table. On the DSGE side, the `dsge_hmc` package exists at
`/home/ubuntu/workspace/python/src/dsge_hmc` but carries only the no-binding
restricted-surface placeholder; integration claims require the Section 14.5
checklist discharged against the actual BGS source per Section 14.6, and the
first deliverable there is a source-anchored transition/solution kernel.
GPU/XLA and memory-growth requirements apply to serious BayesFilter runs, but no
GPU run is part of this literature task.

## 17. Conclusions and limits

The literature gives useful exact building blocks, not a finished nonlinear-ZLB
HMC package. OccBin and PKF solve the conditional linear problem. Aruoba et al.
show how a piecewise-linear regime mixture can support an efficient particle
likelihood. Afshar--Domke and Nishimura--Dunson--Lu show how Hamiltonian motion
can cross true energy discontinuities; Zhou shows how to preserve a mixed
discrete-continuous target without an artificial ordinal embedding. Childers et
al. show the efficiency of joint latent-state HMC for smooth differentiable
state-space models, while Alenlöv et al. state the additional conditions needed
for pseudo-marginal HMC.

The defensible project direction is therefore conditional on the model's
support. A deterministic ZLB calls for a linear PKF benchmark, a nonlinear
particle authority, and eventually an event-aware continuous kernel. A
stochastic regime calls for a mixed target, fixed-path HMC, and PGAS or mixed
HMC. Softplus and Concrete relaxations are useful comparison arms but change the
target.

The application contracts added on 19 August 2026 sharpen this conditional
into two concrete programs. For MacroFinance, the inspected code defines a
hard-bound observation-map shadow-rate model with a declared target ladder
(`mf_s1_k40_softplus`, `mf_c1_k40_hardmax`, `mf_c0_root_integral_hardmax`)
whose hard-bound variants are continuous kink targets: the claim-bearing
routes are the bootstrap hard-bound observation-map particle filter with PMMH
and joint kinked-target HMC, which must
agree on fixtures; the ideal fully adapted cell decomposition is a proposal
and diagnostic layer conditional on controlled polytope-probability and
truncated-draw numerics; the softplus and UKF routes remain declared
approximations with stated bias bounds and a sensitivity protocol; and the
closure-filter shadow-rate line from Black through Krippner, Wu--Xia,
Kim--Priebsch, Christensen--Rudebusch, and Opschoor--van der Wel is the
domain benchmark family, with Pericoli and Taboga's closure-free
data-augmentation MCMC the standard against which any likelihood-fidelity
claim must be measured. For dsge_hmc --- whose package exists with a
no-binding restricted surface, per Section 14.6 --- the worked
linear-complementarity analysis shows the
genuine discontinuity lives in the solution correspondence --- nonexistence
and pervasive multiplicity of verified paths --- so the first modeling act
must be a declared selection law, deterministic or sunspot-stochastic, and the
mixed-support kernels apply only under the stochastic completion. No result
in this survey establishes posterior correctness,
convergence, HMC readiness, production readiness, or statistical superiority
for a BayesFilter implementation.

## References

Afshar, Hadi Mohasel, and Justin Domke. 2015. "Reflection, Refraction, and
Hamiltonian Monte Carlo." *Advances in Neural Information Processing Systems*
28: 3007--3015. <http://hdl.handle.net/1885/103828>.

Alenlöv, Johan, Arnaud Doucet, and Fredrik Lindsten. 2021. "Pseudo-Marginal
Hamiltonian Monte Carlo." *Journal of Machine Learning Research* 22(141):
1--45. <https://jmlr.org/papers/v22/19-486.html>.

Allik, Bethany, Cory Miller, Michael J. Piovoso, and Ryan Zurakowski. 2016.
"The Tobit Kalman Filter: An Estimator for Censored Measurements." *IEEE
Transactions on Control Systems Technology* 24(1): 365--371.
<https://doi.org/10.1109/TCST.2015.2432155>.

Alspach, Daniel L., and Harold W. Sorenson. 1972. "Nonlinear Bayesian Estimation
Using Gaussian Sum Approximations." *IEEE Transactions on Automatic Control*
17(4): 439--448. <https://doi.org/10.1109/TAC.1972.1100034>.

Andrieu, Christophe, Arnaud Doucet, and Roman Holenstein. 2010. "Particle Markov
Chain Monte Carlo Methods." *Journal of the Royal Statistical Society: Series
B* 72(3): 269--342. <https://doi.org/10.1111/j.1467-9868.2009.00736.x>.

Aruoba, S. Borağan, Pablo Cuba-Borda, Kenji Higa-Flores, Frank Schorfheide, and
Sergio Villalvazo. 2021. "Piecewise-Linear Approximations and Filtering for DSGE
Models with Occasionally Binding Constraints." Federal Reserve Bank of
Philadelphia Working Paper 20-13.
<https://doi.org/10.21799/frbp.wp.2020.13>.

Aruoba, S. Borağan, Pablo Cuba-Borda, and Frank Schorfheide. 2018.
"Macroeconomic Dynamics Near the ZLB: A Tale of Two Countries." *Review of
Economic Studies* 85(1): 87--118. <https://doi.org/10.1093/restud/rdx027>.

Bauer, Michael D., and Glenn D. Rudebusch. 2016. "Monetary Policy Expectations
at the Zero Lower Bound." *Journal of Money, Credit and Banking* 48(7):
1439--1465. <https://doi.org/10.1111/jmcb.12338>.

Black, Fischer. 1995. "Interest Rates as Options." *Journal of Finance* 50(5):
1371--1376. <https://doi.org/10.1111/j.1540-6261.1995.tb05182.x>.

Blom, Henk A. P., and Yaakov Bar-Shalom. 1988. "The Interacting Multiple Model
Algorithm for Systems with Markovian Switching Coefficients." *IEEE
Transactions on Automatic Control* 33(8): 780--783.
<https://doi.org/10.1109/9.1299>.

Boehl, Gregor. 2022. "Efficient Solution and Computation of Models with
Occasionally Binding Constraints." *Journal of Economic Dynamics and Control*
143: 104523. <https://doi.org/10.1016/j.jedc.2022.104523>.

Boehl, Gregor, and Felix Strobel. 2023. "Estimation of DSGE Models with the
Effective Lower Bound." *Journal of Economic Dynamics and Control* 158:
104784. <https://doi.org/10.1016/j.jedc.2023.104784>.

Botev, Zdravko I. 2017. "The Normal Law Under Linear Restrictions: Simulation
and Estimation via Minimax Tilting." *Journal of the Royal Statistical
Society: Series B* 79(1): 125--148. <https://doi.org/10.1111/rssb.12162>.
Metadata-cited; inspection required before any cell-draw exactness claim
(Sec. 13.7(v)).

Chaâri, Lotfi, Jean-Yves Tourneret, Caroline Chaux, and Hadj Batatia. 2016. "A
Hamiltonian Monte Carlo Method for Non-Smooth Energy Sampling." *IEEE
Transactions on Signal Processing* 64(21): 5585--5594.
<https://doi.org/10.1109/TSP.2016.2585120>.

Childers, David, Jesús Fernández-Villaverde, Jesse Perla, Christopher
Rackauckas, and Peifan Wu. 2022. "Differentiable State-Space Models and
Hamiltonian Monte Carlo Estimation." NBER Working Paper 30573.
<https://doi.org/10.3386/w30573>.

Christensen, Jens H. E., and Glenn D. Rudebusch. 2015. "Estimating Shadow-Rate
Term Structure Models with Near-Zero Yields." *Journal of Financial
Econometrics* 13(2): 226--259. <https://doi.org/10.1093/jjfinec/nbu010>.

Cuba-Borda, Pablo, Luca Guerrieri, Matteo Iacoviello, and Molin Zhong. 2019.
"Likelihood Evaluation of Models with Occasionally Binding Constraints."
*Journal of Applied Econometrics* 34(7): 1073--1085.
<https://doi.org/10.1002/jae.2733>. Metadata-cited; primary-source inspection
is an expansion requirement before the OBC empirical baseline freezes
(Sec. 14.4).

Corenflos, Adrien, James Thornton, George Deligiannidis, and Arnaud Doucet.
2021. "Differentiable Particle Filtering via Entropy-Regularized Optimal
Transport." *Proceedings of the 38th International Conference on Machine
Learning*, PMLR 139: 2100--2111.
<https://proceedings.mlr.press/v139/corenflos21a.html>.

Deligiannidis, George, Arnaud Doucet, and Michael K. Pitt. 2018. "The
Correlated Pseudomarginal Method." *Journal of the Royal Statistical Society:
Series B* 80(5): 839--870. <https://doi.org/10.1111/rssb.12280>.
Metadata-cited; PMMH variance/mixing comparator, not a PM-HMC
differentiability fix (Sec. 10.2).

García-Fernández, Ángel F., Mark R. Morelande, and Jesús Grajal. 2012a.
"Truncated Unscented Kalman Filtering." *IEEE Transactions on Signal
Processing* 60(7): 3372--3386. <https://doi.org/10.1109/TSP.2012.2193393>.

García-Fernández, Ángel F., Mark R. Morelande, and Jesús Grajal. 2012b.
"Mixture Truncated Unscented Kalman Filtering." In *Proceedings of the 15th
International Conference on Information Fusion (FUSION)*, 479--486. Singapore.

Geng, Hang, Hongjian Liu, Lifeng Ma, and Xiaojian Yi. 2021. "Multi-Sensor
Filtering Fusion Meets Censored Measurements Under a Constrained Network
Environment: Advances, Challenges and Prospects." *International Journal of
Systems Science* 52(16): 3410--3436.
<https://doi.org/10.1080/00207721.2021.2005178>.

Genz, Alan, and Frank Bretz. 2009. *Computation of Multivariate Normal and t
Probabilities*. Lecture Notes in Statistics 195. Berlin: Springer.
<https://doi.org/10.1007/978-3-642-01689-9>. Metadata-cited; inspection
required before any polytope-probability exactness claim (Sec. 13.7(v)).

Giovannini, Massimo, Philipp Pfeiffer, and Marco Ratto. 2021. "Efficient and
Robust Inference of Models with Occasionally Binding Constraints." JRC Working
Papers in Economics and Finance 2021/3.
<https://hdl.handle.net/10419/249365>.

Gokce, Murat, and Mustafa Kuzuoglu. 2015. "Unscented Kalman Filter-Aided
Gaussian Sum Filter." *IET Radar, Sonar & Navigation* 9(5): 589--599.
<https://doi.org/10.1049/iet-rsn.2014.0088>.

Guerrieri, Luca, and Matteo Iacoviello. 2015. "OccBin: A Toolkit for Solving
Dynamic Models with Occasionally Binding Constraints Easily." *Journal of
Monetary Economics* 70: 22--38.
<https://doi.org/10.1016/j.jmoneco.2014.08.005>.

Holden, Tom D. 2016. "Computation of Solutions to Dynamic Models with
Occasionally Binding Constraints." Unpublished working paper, University of
Surrey. <https://github.com/tholden/dynareOBC>.

Holden, Tom D., and Michael Paetz. 2012. "Efficient Simulation of DSGE Models
with Inequality Constraints." Quantitative Macroeconomics Working Papers
21207b, Hamburg University. Metadata-cited; original anticipated-shock route
(Sec. 14.1).

Holden, Tom D. 2017. "Tractable Estimation and Smoothing of Highly Nonlinear
Dynamic State-Space Models." Unpublished working paper, University of Surrey.

Holden, Tom D. 2023. "Existence and Uniqueness of Solutions to Dynamic Models
with Occasionally Binding Constraints." *Review of Economics and Statistics*
105(6): 1481--1499. <https://doi.org/10.1162/rest_a_01122>.

Kandepu, Rambabu, Lars Imsland, and Bjarne A. Foss. 2008. "Constrained State
Estimation Using the Unscented Kalman Filter." In *Proceedings of the 16th
Mediterranean Conference on Control and Automation*, 1453--1458. Ajaccio:
IEEE. <https://doi.org/10.1109/MED.2008.4602001>.

Kim, Don H., and Marcel Priebsch. 2013. "Estimation of Multi-Factor Shadow-Rate
Term Structure Models." Preliminary draft, October 9. Washington: Board of
Governors of the Federal Reserve System.

Krippner, Leo. 2012. "Modifying Gaussian Term Structure Models When Interest
Rates Are Near the Zero Lower Bound." Reserve Bank of New Zealand Discussion
Paper DP2012/02.

Lemke, Wolfgang, and Andreea Liliana Vladu. 2017. "Below the Zero Lower Bound:
A Shadow-Rate Term Structure Model for the Euro Area." ECB Working Paper
No. 1991. Frankfurt: European Central Bank.

Lindsten, Fredrik, Michael I. Jordan, and Thomas B. Schön. 2014. "Particle Gibbs
with Ancestor Sampling." *Journal of Machine Learning Research* 15: 2145--2184.
<https://jmlr.org/papers/v15/lindsten14a.html>.

Lubik, Thomas A., and Frank Schorfheide. 2003. "Computing Sunspot Equilibria in
Linear Rational Expectations Models." *Journal of Economic Dynamics and
Control* 28(2): 273--285. <https://doi.org/10.1016/S0165-1889(02)00153-7>.

Neal, Radford M. 2011. "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov
Chain Monte Carlo*, edited by Steve Brooks, Andrew Gelman, Galin L. Jones, and
Xiao-Li Meng, 113--162. Chapman and Hall/CRC.
<https://doi.org/10.1201/b10905-6>.

Nishimura, Akihiko, David B. Dunson, and Jianfeng Lu. 2020. "Discontinuous
Hamiltonian Monte Carlo for Discrete Parameters and Discontinuous Likelihoods."
*Biometrika* 107(2): 365--380.
<https://doi.org/10.1093/biomet/asz083>.

Opschoor, Daan, and Michel van der Wel. 2024. "A Smooth Shadow-Rate Dynamic
Nelson-Siegel Model for Yields at the Zero Lower Bound." *Journal of Business
& Economic Statistics* 43(2): 298--311.
<https://doi.org/10.1080/07350015.2024.2365779>.

Pakman, Ari, and Liam Paninski. 2014. "Exact Hamiltonian Monte Carlo for
Truncated Multivariate Gaussians." *Journal of Computational and Graphical
Statistics* 23(2): 518--542. <https://doi.org/10.1080/10618600.2013.788448>.

Pericoli, Marcello, and Marco Taboga. 2018. "Nearly Exact Bayesian Estimation
of Non-Linear No-Arbitrage Term-Structure Models." Banca d'Italia Temi di
discussione (Working Papers) 1189, September 2018. Inspected as local full
text (the recovered PDF is the 2018 working paper; a later journal version is
reported to exist but was not verifiable from the local corpus and is not
cited here).

Poyiadjis, George, Arnaud Doucet, and Sumeetpal S. Singh. 2011. "Particle
Approximations of the Score and Observed Information Matrix in State Space
Models with Application to Parameter Estimation." *Biometrika* 98(1): 65--80.
<https://doi.org/10.1093/biomet/asq062>.

Priebsch, Marcel A. 2013. "Computing Arbitrage-Free Yields in Multi-Factor
Gaussian Shadow-Rate Term Structure Models." Finance and Economics Discussion
Series 2013-63. Washington: Board of Governors of the Federal Reserve System.
<https://doi.org/10.17016/feds.2013.63>.

Ścibior, Adam, and Frank Wood. 2021. "Differentiable Particle Filtering without
Modifying the Forward Pass." arXiv:2106.10314v2.
<https://arxiv.org/abs/2106.10314>.

Teixeira, Bruno O. S., Leonardo A. B. Tôrres, Luis A. Aguirre, and Dennis S.
Bernstein. 2010. "On Unscented Kalman Filtering with State Interval
Constraints." *Journal of Process Control* 20(1): 45--57.
<https://doi.org/10.1016/j.jprocont.2009.10.007>.

Torgander, Jakob, Måns Magnusson, and Jonas Wallin. 2024. "Hamiltonian Monte
Carlo with Categorical Parameters Using the Concrete Distribution." Workshop at
the 6th Symposium on Advances in Approximate Bayesian Inference, non-archival,
1--12.

van der Merwe, Rudolph, Arnaud Doucet, Nando de Freitas, and Eric A. Wan.
2000. "The Unscented Particle Filter." In *Advances in Neural Information
Processing Systems* 13: 584--590.

Wang, Yanbo, Fasheng Wang, Jianjun He, and Fuming Sun. 2020. "Iterative
Truncated Unscented Particle Filter." *Information* 11(4): 214.
<https://doi.org/10.3390/info11040214>.

Wu, Jing Cynthia, and Fan Dora Xia. 2016. "Measuring the Macroeconomic Impact
of Monetary Policy at the Zero Lower Bound." *Journal of Money, Credit and
Banking* 48(2--3): 253--291. <https://doi.org/10.1111/jmcb.12300>.

Zhang, Hongwei, Xiaohu Zhang, Weixin Xie, and Xia Yang. 2020. "Constrained
Multiple Model UK Filter." In *Proceedings of the 15th IEEE International
Conference on Signal Processing (ICSP)*, 48--51. Beijing: IEEE.
<https://doi.org/10.1109/ICSP48669.2020.9320991>.

Zhou, Guangyao. 2020. "Mixed Hamiltonian Monte Carlo for Mixed Discrete and
Continuous Variables." *Advances in Neural Information Processing Systems* 33.
<https://arxiv.org/abs/1909.04852>.

The machine-readable version is `references.bib`. Inspected technical anchors,
version and retraction checks, metadata caveats, and claim mappings are recorded
in the six audit ledgers next to this file.
