# C2 Coherent TT, Reference, and Exact-Proposal Testing Plan

Date: 2026-08-31

Status: Stage 0 and Stage 1 executed on 2026-08-31; Stage 2 executed and hit
its predeclared precision continuation veto. Stages 3--5 remain conditional;
no candidate is promoted by this document.

## 1. Purpose

This plan combines four related ideas without treating them as one method:

1. use the retained TT, UKF, and LEDH constructions to build proposals;
2. evaluate the exact transition/observation factors against the finite carried
   target with complete-mixture importance weights;
3. use moments of the retained predicted density to construct the next affine
   fitting coordinates; and
4. test whether a richer basis or a different reference density improves the
   TT representation.

The separation is essential. Exact importance correction can remove the fitted
TT normalizer from the finite-target likelihood estimator, but it cannot make a
poor proposal efficient or remove error already present in the carried density.
A new basis can reduce projection error, but it cannot establish likelihood
correctness. A recursive coordinate map can improve conditioning, but it can
also feed a bad tail approximation back into the next step. Each claim is
tested against the diagnostic that can answer it.

## 2. Research question and evidence contract

### 2.1 Main question

For the C2 stochastic-volatility filtering problem, which mechanism is
responsible for the current n=4 failure, and which bounded repair gives the
best evidence for a usable analytical-gradient route?

### 2.2 Separate questions

| Question | Candidate mechanism | Primary evidence |
| --- | --- | --- |
| Does exact-factor importance correction remove the gross TT-normalizer error for the finite carried-density program? | UKF/TT/LEDH/Student proposals with complete DMIS weights | finite-difference score, finite-target identity, independent finite-target integral, PF compatibility as a separate diagnostic, ESS |
| Does a filter-derived coordinate map improve the TT fit? | propagate retained density, compute predictive moments, rebuild the affine map | held-out density error and per-time recursive error versus the frozen-hint map |
| Does basis capacity or shape cause the failure? | Hermite degree ladder, fixed RBF, Hermite/RBF hybrid | held-out and shell errors, direct normalizer gap, Gram health |
| Does the Gaussian reference cause the failure? | Student or Gaussian/Student reference, implemented consistently | same diagnostics after complete measure and Gram rewrite |

### 2.3 Exact model factors and the finite carried target

Let \(x_t\) be the filtering state, \(f_t(x_t\mid x_{t-1})\) the exact
transition density, and \(g_t(y_t\mid x_t)\) the exact observation density.
For a normalized previous density \(\pi_{t-1}\), the exact model recursion is

\[
  \gamma_t^\star(x_t,x_{t-1})
  = f_t(x_t\mid x_{t-1})
    g_t(y_t\mid x_t)
    \pi_{t-1}(x_{t-1}).
\]

At the initial time the corresponding object is
\(\gamma_0^\star(x_0)=p_0(x_0)g_0(y_0\mid x_0)\); in the formulas below,
\(\gamma_t\) for \(t\geq1\) is the joint current/previous object and the
initial step is tested separately.

In the candidate algorithms \(\pi_{t-1}\) is replaced by the normalized
carried approximation \(\widehat p_{t-1}\). The finite target actually
evaluated is therefore

\[
  \gamma_t^{\mathrm{fin}}(x_t,x_{t-1})
  = f_t(x_t\mid x_{t-1})
    g_t(y_t\mid x_t)
    \widehat p_{t-1}(x_{t-1}).
\]

The transition and observation factors in
\(\gamma_t^{\mathrm{fin}}\) are exact. Its normalizer is the exact integral
conditional on the carried approximation,

\[
  Z_t^{\mathrm{fin}}
  = \iint \gamma_t^{\mathrm{fin}}(x_t,x_{t-1})\,dx_t\,dx_{t-1},
\]

and the normalized finite-program filtering approximation is the current-state
marginal

\[
  \widehat p_t(x_t)
  = (Z_t^{\mathrm{fin}})^{-1}
    \int \gamma_t^{\mathrm{fin}}(x_t,x_{t-1})\,dx_{t-1}.
\]

The density in \(\gamma_t^{\mathrm{fin}}\) is a physical Lebesgue density. If
the retained quadratic form is stored as a density ratio
\(\widehat p_{t-1}^{\mathrm{ref}}\) with respect to
\(\mu_{t-1}(du)=r_{t-1}(u)\,du\), and
\(x=R_{t-1}(u)=m_{t-1}+L_{t-1}u\), the factor inserted into
\(\gamma_t^{\mathrm{fin}}\) is

\[
  \widehat p_{t-1}^{\mathrm{phys}}(x)
  =\widehat p_{t-1}^{\mathrm{ref}}\!\left(R_{t-1}^{-1}x\right)
    r_{t-1}\!\left(R_{t-1}^{-1}x\right)
    |\det L_{t-1}|^{-1}.
\]

Using the reference ratio directly as a physical density drops both the
reference weight and the Jacobian and changes the target. The same conversion
must be applied to every TT, Gaussian, Student, and RBF proposal component.

If \(\widehat p_{t-1}=\pi_{t-1}\) at every step, then
\(Z_t^{\mathrm{fin}}\) equals the true predictive likelihood. Otherwise
\(\sum_t\log Z_t^{\mathrm{fin}}\) is the declared finite carried-density
scalar, not the exact model likelihood.

The cumulative finite-program target is

\[
  \ell_T^{\mathrm{fin}}
  = \sum_{t=0}^{T-1}\log Z_t^{\mathrm{fin}}.
\]

Every proposal arm evaluates \(\gamma_t^{\mathrm{fin}}\) using the exact
transition and observation densities. It does not substitute a fitted TT
normalizer or a Gaussian observation closure. The distinction between exact
model factors and the approximate carried prior must remain explicit.

### 2.4 Contract and nonclaims

The evidence contract is:

- baseline: current retained-TT proposal, current Gaussian-hint proposal,
  transformed-observation Student proposal, bootstrap conditional proposal,
  and stationary Gaussian proposal;
- primary proposal criterion: exact-factor finite-program estimates agree with
  an independent \(Z_t^{\mathrm{fin}}\) reference and remain finite, while a
  nominated candidate has an uncertainty-supported increase in per-time
  minimum normalized ESS relative to retained TT;
- primary representation criterion: held-out target-only density error and
  direct normalizer error decrease relative to the current Hermite route;
- uncertainty rule: use paired bootstrap intervals (and a paired sign test) on
  predeclared salient times. A positive point estimate without an interval
  excluding zero nominates a candidate but does not establish an improvement;
  if a non-inferiority margin is needed for a tail or normalizer diagnostic,
  determine it from the Stage 0 fixture before looking at claim data;
- hard vetoes: wrong target or measure, incomplete mixture denominator,
  nonnormalized proposal, nonfinite value or score, failed same-scalar finite
  difference, invalid Gram/factorization, or failed uniform-route regression;
- explanatory diagnostics: training RMS, shell residual, condition number,
  rank, maximum weight, and component contributions unless explicitly promoted
  to a criterion above;
- not concluded: exact inference, posterior correctness, HMC readiness,
  universal superiority, source-faithful Zhao--Cui reproduction, or a new
  production default.

### 2.5 Defaults, assumptions, and heuristic adversaries

The following choices are hypotheses or baselines, not established defaults.
Each one has an early diagnostic that can reject the choice without rejecting
the whole research direction.

| Choice | Provenance and role | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- |
| Gaussian reference with normalized Hermite factors | current implementation; frozen baseline | polynomial capacity and Gaussian tails misrepresent the target | held-out central/tail residuals, Gram spectrum, and direct normalizer gap |
| retained-TT proposal | current implementation; baseline comparator | a good-looking fit has a poor proposal tail and weight collapse | complete-mixture ESS and maximum weight by time |
| UKF Gaussian/Student proposals | prior C2 construction; heuristic adversaries | moment matching can miss skewness or multimodality | shell residuals and proposal-vs-target log-density checks |
| half defensive mixture | prior pilot; candidate, not a default | fixed alpha can spend too much mass on a weak component | pilot-only alpha grid on disjoint data, then freeze alpha |
| degrees \(d=6,8,10\) and fixed RBF widths | user-proposed finite ladder | higher capacity can amplify ill-conditioning or tail oscillation | condition number, positive-definiteness margin, held-out shells |
| one-step lagged moment map | recursive construction proposed here | feedback can accumulate moment or Cholesky error | per-time map shift, covariance margin, and first-error-time trace |
| frozen rows, proposals, and branch decisions | required analytical-gradient contract | freezing changes the adaptive algorithm being differentiated | same-scalar finite-difference test and explicit route identity |
| twelve paired branches | minimum uncertainty pilot for stochastic diagnostics | too few replications leave ESS and tail comparisons descriptive | paired bootstrap interval and sign test; expand only under budget |

The heuristic adversary set is constructed from what a filtering practitioner
would use without the TT representation:

| Salient situation | Cheap adversary | Why it is a meaningful check |
| --- | --- | --- |
| ordinary innovation and early transient | UKF/Kalman Gaussian proposal | tests whether a complicated proposal adds value over moment matching |
| large innovation or tail state | transformed-observation Student proposal | tests whether heavier proposal tails are the real repair |
| every time, including multimodal-risk times | bootstrap conditional proposal | uses the model transition directly and has no fitted normalizer |
| weakly informative observation | stationary Gaussian proposal | tests whether adaptation is helping or merely adding variance |
| minimum-ESS time and full horizon | retained-TT proposal | identifies whether the representation itself is the bottleneck |

Evaluate these adversaries conditionally at the predeclared times \(t=3,t=4\),
the largest-innovation time, the minimum-ESS time, and the full horizon. Losing
to a cheap adversary in one salient situation is a promotion veto, even when an
unconditional average looks favorable.

## 3. Mathematical construction

### 3.1 Current Gaussian-reference TT fit

Let \(u=T_t(x)\) be the per-step affine coordinate map

\[
  u = L_t^{-1}(x-m_t),
  \qquad x=m_t+L_tu,
\]

and let \(r_t(u)=\eta(u)\) be the standard-normal reference density. The
pulled-back density relative to the reference measure is

\[
  F_t(u,v)
  = \frac{
      \gamma_t^{\mathrm{fin}}\bigl(\Psi_t(u,v)\bigr)
      \left|\det D\Psi_t(u,v)\right|
    }{r_t(u,v)},
\]

where \(u\) and \(v\) denote current and previous reference coordinates.
The square-root target is

\[
  s_t(u,v)=\exp\left(\frac{1}{2}[\log F_t(u,v)-c_t]\right),
  \qquad
  c_t=\max_{\text{fit rows}}\log F_t.
\]

The current finite fitter computes

\[
  h_t = \Pi_{t,\mathcal V_t}(s_t),
\]

where \(\mathcal V_t\) is a fixed-rank tensor-train space with normalized
Hermite factors. The represented nonnegative target is

\[
  \widetilde F_t(u,v)
  = h_t(u,v)^2 + \tau_t q_{0,t}(u,v),
\]

with normalizer

\[
  \widetilde Z_t
  = \iint \widetilde F_t(u,v)\,d\mu_t(u,v).
\]

The scale \(c_t\) is bookkeeping, not a second target. The displayed
\(\widetilde F_t\) is in the scaled coordinates of the finite least-squares
problem, so its physical normalizer is \(e^{c_t}\widetilde Z_t\). Before
adding the defensive floor,

\[
  Z_t^{\mathrm{fin}}
  =e^{c_t}\|s_t\|_{L^2(\mu_t)}^2,\qquad
  Z_{H,t}=e^{c_t}\|h_t\|_{L^2(\mu_t)}^2.
\]

Consequently \(\log Z_{H,t}-\log Z_t^{\mathrm{fin}}\) measures the fit's
normalizer error for the same finite target. With a scaled floor, the
corresponding complete physical normalizer is
\[
  Z_{\mathrm{complete},t}
  =e^{c_t}\bigl(\|h_t\|_{L^2(\mu_t)}^2+\tau_t Z_{0,t}\bigr).
\]
Report this deliberately modified target separately from the no-floor
comparison.

After integrating the previous axes, the retained density is generally a
quadratic form, not a scalar squared TT:

\[
  \widehat p_t(z)
  = \frac{H_{L,t}(z)E_tH_{L,t}(z)^\top
          +\tau_tq_{0,t}(z)}{Z_{c,t}}.
\]

The carried state is therefore

\[
  S_t=(H_{L,t},E_t,Z_{c,t},T_t,\tau_t),
\]

not just the coefficient vector of \(h_t\).

At \(t=0\) there is no previous coordinate. Use the separate target

\[
  \gamma_0^{\mathrm{fin}}(x_0)=p_0(x_0)g_0(y_0\mid x_0),
\]

and, for \(x_0=m_0+L_0u_0\),

\[
  F_0(u_0)=
    \frac{\gamma_0^{\mathrm{fin}}(m_0+L_0u_0)|\det L_0|}
         {r_0(u_0)},
  \qquad
  Z_0^{\mathrm{fin}}=\int F_0(u_0)\,d\mu_0(u_0).
\]

The same square-root fit, floor bookkeeping, proposal normalization, and
finite-difference score checks apply, but no transition or ancestor term is
present. This initial case is a required Stage 0 fixture rather than an
implicit \(t\geq1\) specialisation.

### 3.2 Exact DMIS proposal route

Let \(q_{t,1},\ldots,q_{t,J}\) be normalized proposal components, for example
the retained TT component, a UKF Gaussian or Student component, and a local
LEDH component. Let \(\alpha_j\geq0\) with \(\sum_j\alpha_j=1\). The complete
proposal density is

\[
  q_t(x)=\sum_{j=1}^J\alpha_jq_{t,j}(x).
\]

Samples may be drawn from any component, but every sample is weighted using
the same complete denominator:

\[
  w_{t,i}=\frac{\gamma_t^{\mathrm{fin}}(X_{t,i})}{q_t(X_{t,i})}.
\]

With normalized base masses \(b_{t,i}\),

\[
  \widehat Z_t
  =\sum_i b_{t,i}w_{t,i},
  \qquad
  \widehat p_t^{\mathrm{IS}}(\varphi)
  =\frac{\sum_i b_{t,i}w_{t,i}\varphi(X_{t,i})}
          {\sum_i b_{t,i}w_{t,i}}.
\]

The base masses encode how the bank was generated. For iid draws from the
mixture, \(b_{t,i}=1/N\); for a deterministic bank with \(N_j\) draws from
component \(j\), use \(b_{t,i}=\alpha_j/N_j\). In either case
\(\sum_i b_{t,i}=1\). Using \(1/N\) for a stratified bank with unequal
\(N_j\) changes the finite estimator even though the pointwise ratio looks
correct.

The selected component density is never substituted for \(q_t\). Omitting
the other mixture terms produces the wrong importance ratio.

For normalized \(q_t\) with \(q_t>0\) wherever
\(\gamma_t^{\mathrm{fin}}>0\), and with
\(\int\lvert\gamma_t^{\mathrm{fin}}\rvert<\infty\), the identity behind DMIS is

\[
  \mathbb E_{q_t}[w_t]
  =\int q_t(\xi)\frac{\gamma_t^{\mathrm{fin}}(\xi)}{q_t(\xi)}\,d\xi
  =Z_t^{\mathrm{fin}}.
\]

Its finite-sample difficulty is quantified by

\[
  \operatorname{Var}_{q_t}(w_t)
  =\int \frac{(\gamma_t^{\mathrm{fin}}(\xi))^2}{q_t(\xi)}\,d\xi
    -(Z_t^{\mathrm{fin}})^2.
\]

For a defensive mixture
\(q_t=(1-\alpha)q_{t,\mathrm{loc}}+\alpha q_{t,\mathrm{def}}\),
\(q_t\geq\alpha q_{t,\mathrm{def}}\). Therefore, whenever the right-hand
side is finite,

\[
  \int\frac{(\gamma_t^{\mathrm{fin}})^2}{q_t}
  \leq \alpha^{-1}
       \int\frac{(\gamma_t^{\mathrm{fin}})^2}{q_{t,\mathrm{def}}}.
\]

For iid samples, the corresponding large-\(N\) normalized ESS fraction is
\[
  \frac{\mathrm{ESS}_\infty}{N}
  =\frac{(Z_t^{\mathrm{fin}})^2}
         {\int(\gamma_t^{\mathrm{fin}})^2/q_t}
  \geq
  \alpha\,
  \frac{(Z_t^{\mathrm{fin}})^2}
       {\int(\gamma_t^{\mathrm{fin}})^2/q_{t,\mathrm{def}}}.
\]

The defensive component guarantees support and gives a variance bound; it does
not guarantee a useful ESS when \(\alpha\) is too small or the defensive tail
is badly scaled. In particular, "Student has heavier tails" is a hypothesis,
not a theorem about the C2 target: the finite-second-moment condition
\(\int(\gamma_t^{\mathrm{fin}})^2/q_{t,\mathrm{def}}<\infty\) must be checked
on analytic fixtures and monitored numerically on the model bank.

For a frozen proposal, samples, ancestry, and base masses, the analytical
score of the finite estimate is

\[
  \nabla_\theta\log\widehat Z_t
  =\sum_i\bar w_{t,i}\nabla_\theta\log\gamma_{t,\theta}^{\mathrm{fin}}(X_{t,i}),
  \qquad
  \bar w_{t,i}=\frac{b_{t,i}w_{t,i}}{\widehat Z_t}.
\]

If \(q_t\) depends on \(\theta\), the complete derivative instead contains

\[
  \nabla_\theta\log\gamma_{t,\theta}^{\mathrm{fin}}(X_{t,i})
  -\nabla_\theta\log q_{t,\theta}(X_{t,i}).
\]

This route separates two questions. Exact-factor importance correction removes
the use of \(Z_H\) as if it were \(Z_T\) for the declared finite target, while
ESS measures whether the proposal is useful at finite particle count. It does
not remove error already present in \(\widehat p_{t-1}\), and it does not turn
\(Z_t^{\mathrm{fin}}\) into the true predictive likelihood unless the carried
density equals the true filter.

For an APF implementation with resampling, the displayed one-step identity is
implemented with the selected ancestor weight and auxiliary-law correction.
Those terms are part of the finite proposal density and must be included in
the complete denominator; the simplified equation above is the normalized
joint-target form.

For a (d)-dimensional local Gaussian and defensive Student component, use the
normalized physical densities

\[
  q_{\mathrm G}(x;m,S)
  =(2\pi)^{-d/2}|S|^{-1/2}
    \exp\!\left[-\tfrac12(x-m)^\top S^{-1}(x-m)\right],
\]

\[
  q_{\mathrm{St}}(x;m,S,\nu)
  =\frac{\Gamma((\nu+d)/2)}
          {\Gamma(\nu/2)(\nu\pi)^{d/2}|S|^{1/2}}
    \left[1+\frac{(x-m)^\top S^{-1}(x-m)}{\nu}\right]^{-(\nu+d)/2}.
\]

For \(\nu>2\), the Student covariance is \(\nu S/(\nu-2)\). Thus a
covariance-matched defensive component can use
\(S=(\nu-2)P^-/\nu\), optionally multiplied by a predeclared inflation
factor. The inflation, degrees of freedom, and mixture mass \(\alpha\) are
selected on calibration data and then frozen. Under the affine coordinates
\(x=m+Lu\), evaluate either component as

\[
  q_x(x)=q_u(L^{-1}(x-m))|\det L|^{-1},
\]

before taking the log-sum-exp mixture. This keeps the proposal's tail choice
separate from the TT reference measure: a Student proposal can be tested
without changing any Hermite Gram contraction.

### 3.3 Recursive density-derived coordinate map

Suppose the transition is linear-Gaussian,

\[
  x_t=A_tx_{t-1}+a_t+\epsilon_t,
  \qquad \epsilon_t\sim\mathcal N(0,Q_t).
\]

If the retained density has mean \(m_{t-1}\) and covariance \(P_{t-1}\),
then the predicted moments are exactly

\[
  m_t^- = A_tm_{t-1}+a_t,
  \qquad
  P_t^- = A_tP_{t-1}A_t^\top+Q_t,
  \qquad
  P_{t,t-1}^- = A_tP_{t-1}.
\]

For a nonlinear transition, these equations become an approximation supplied
by a tested TT operator, UKF, quadrature, or particle bank; that approximation
must be named rather than treated as exact.

More generally, if \(x_t=a_t(x_{t-1})+\varepsilon_t\), with
\(\mathbb E[\varepsilon_t\mid x_{t-1}]=0\) and
\(\operatorname{Cov}(\varepsilon_t\mid x_{t-1})=Q_t(x_{t-1})\), then

\[
  m_t^-=\mathbb E_{\widehat p_{t-1}}[a_t(X)],
  \qquad
  P_t^-=\operatorname{Cov}_{\widehat p_{t-1}}(a_t(X))
        +\mathbb E_{\widehat p_{t-1}}[Q_t(X)].
\]

These expectations are the quantities a UKF or a TT moment operator is
approximating. If the process noise has a nonzero conditional mean or is
correlated with \(X\), the displayed covariance decomposition is incomplete
and the corresponding cross terms must be added.

Use the predicted moments to define

\[
  T_t(x)=L_t^{-1}(x-m_t^-),
  \qquad L_tL_t^\top=P_t^-.
\]

When the TT fit is on the joint pair, use the joint predicted covariance,
\[
  P_t^{\mathrm{joint}}
  =\begin{bmatrix}
      P_t^- & P_{t,t-1}^-\\
      (P_{t,t-1}^-)^\top & P_{t-1}
    \end{bmatrix},
\qquad
  \begin{bmatrix}x_t\\x_{t-1}\end{bmatrix}
  =\begin{bmatrix}m_t^-\\m_{t-1}\end{bmatrix}
   +L_t^{\mathrm{joint}}
    \begin{bmatrix}u_t\\u_{t-1}\end{bmatrix},
\]
where \(L_t^{\mathrm{joint}}(L_t^{\mathrm{joint}})^\top
=P_t^{\mathrm{joint}}\). The current implementation uses the equivalent
block-lower-triangular form
\(x_{t-1}=m_{t-1}+L_{pc}u_t+L_{pp}u_{t-1}\). A product of separate current and
previous maps is a different candidate and must not be silently compared as
the same \(T_t\).

The fitting target in the new coordinates remains the exact pulled-back target
from Section 3.1. The proposal is therefore a lagged recursive map:

\[
  S_{t-1}
  \longrightarrow
  \widehat p_t^-
  \longrightarrow
  (m_t^-,P_t^-)
  \longrightarrow
  T_t
  \longrightarrow
  h_t
  \longrightarrow
  S_t.
\]

This is preferable to making the map depend on the current fit in an unlimited
inner loop. A bounded optional self-consistency variant may repeat the map-fit
cycle a fixed number of times, with a declared smooth relaxation, but it is a
separate candidate because it changes the finite program.

The reason to measure the recursive error separately is visible from the
filtering operator. Define

\[
  K_t[p](x)
  =g_t(y_t\mid x)\int f_t(x\mid z)p(z)\,dz,
  \qquad
  Z_t[p]=\int K_t[p](x)\,dx,
  \qquad
  \mathcal F_t[p]=K_t[p]/Z_t[p].
\]

With
\(\ell_t(z)=\int f_t(x\mid z)g_t(y_t\mid x)\,dx\), the normalizer error caused
solely by the previous-density error
\(e_{t-1}=\widehat p_{t-1}-\pi_{t-1}\) is exactly

\[
  Z_t[\widehat p_{t-1}]-Z_t[\pi_{t-1}]
  =\int \ell_t(z)e_{t-1}(z)\,dz.
\]

For example,
\(\lvert Z_t[\widehat p]-Z_t[\pi]\rvert
\leq\|\ell_t\|_\infty\|e_{t-1}\|_1\). If both normalizers stay above a
declared \(z_{\min}>0\), the normalized filtering operator is locally
Lipschitz, so a representation error \(\delta_t\) satisfies the schematic
recursion

\[
  \|e_t\|_1
  \leq C_t\|e_{t-1}\|_1+\delta_t.
\]

Under the simple bounded-likelihood condition
\(0\leq g_t(y_t\mid x)\leq G_t\), one may take the explicit local bound
\[
  \|\mathcal F_t[p]-\mathcal F_t[\pi]\|_1
  \leq \frac{2G_t}{z_{\min}}\|p-\pi\|_1
\]
whenever \(Z_t[p]\) and \(Z_t[\pi]\) are at least \(z_{\min}\). This bound is
not a claim that the C2 nonlinear observation satisfies one global constant;
it is the reason to record \(G_t\), \(Z_t\), and the first time the recursion
becomes unstable.

The coefficient \(C_t\) can be larger than one in an informative or tail
observation. This is why a map that lowers the one-step held-out residual can
still make the final horizon worse: it may reduce \(\delta_t\) centrally while
amplifying the carried tail error in the next step. The first time at which
\(C_t\|e_{t-1}\|_1\) dominates \(\delta_t\) is a root-cause diagnostic, not an
argument to conflate proposal and representation errors.

### 3.4 Basis candidates

The first basis ladder keeps the reference \(r_t=\eta\) and changes only the
fixed representation space.

#### Hermite ladder

Test maximum one-dimensional degrees \(d\in\{6,8,10\}\):

\[
  \mathcal V_t^{\mathrm H}(d)
  =\operatorname{TT}_R
   \left\{\prod_{k=1}^D\overline{\mathrm{He}}_{b_k}(u_k):
          0\leq b_k\leq d\right\}.
\]

The standard-normal mass matrix remains the identity. This isolates capacity
from measure changes, but higher degree can worsen row and Gram conditioning.

#### Fixed RBF basis

For fixed centers \(c_{j,t}\) and coordinate widths \(s_{j,t,k}>0\), use

\[
  \psi_{j,t}(u)
  =\prod_{k=1}^D
    \exp\left[-\frac{(u_k-c_{j,t,k})^2}{2s_{j,t,k}^2}\right].
\]

Centers may be fixed offsets around the UKF predicted mean in the current
coordinate frame. If the fit coordinates are \(u=L_t^{-1}(x-m_t^-)\), a
physical UKF center \(\mu_{j,t}^{\mathrm{UKF}}\) becomes
\(c_{j,t}=L_t^{-1}(\mu_{j,t}^{\mathrm{UKF}}-m_t^-)\). Widths are selected from a
finite predeclared grid. The displayed factors are separable and therefore
compatible with tensor-product TT contractions; a fully elliptical
multivariate RBF is a distinct, nonseparable candidate. Do not add a second
literal constant because the Hermite channel
\(\overline{\mathrm{He}}_0=1\) already supplies it. Use broad but nonconstant
RBF factors, or explicitly orthogonalize them against the Hermite constant.
RBF Gram entries under a Gaussian reference can be computed analytically one
coordinate at a time or by an independent high-accuracy reference integral;
the same entries, including Hermite--RBF cross terms, must be used in fitting,
marginalization, normalization, and the score.

#### Hermite-plus-RBF hybrid

Use a fixed direct sum or fixed TT channel concatenation

\[
  h_t(u)=h_t^{\mathrm H}(u)+h_t^{\mathrm R}(u),
\]

with fixed channel ranks and a broad, nonconstant tail channel. The hybrid has
exactly one constant channel. Adding another literal constant makes two
pointwise-identical columns and therefore a singular Gram matrix. The hybrid
is not allowed to select channels adaptively during the claim run. Its
cross-Gram terms must be included; treating the two blocks as orthogonal when
they are not is a measure error.

### 3.5 Reference-density candidates

Only after the Hermite-reference proposal and basis tests are complete, test
the reference itself. To preserve product-measure TT contractions, the first
Student candidate is a product of univariate laws:

\[
  r_t^{\mathrm G}(u)=\eta_D(u),
  \qquad
  r_t^{\mathrm{St},\nu}(u)=\prod_{k=1}^D t_\nu(u_k),
\]

and a fixed mixture

\[
  r_t^{\mathrm{mix}}(u)
  =(1-\alpha)r_t^{\mathrm G}(u)+\alpha r_t^{\mathrm{St},\nu}(u).
\]

An elliptical multivariate Student law is not the displayed product law. It
introduces a non-product mass measure and must be treated as a separate route,
not as a parameter setting of the product-measure TT.

Changing \(r_t\) requires all of the following to change consistently:

1. basis orthogonalization or the full mass matrix
   \(M_{bb'}=\int\phi_b\phi_{b'}r_t\);
2. Gram and marginal contractions;
3. the Christoffel or other row-sampling law;
4. the pulled-back target and defensive density;
5. normalizer and score formulas; and
6. normalization and tail-domination tests.

The Student reference must not be combined with polynomial Gram contractions
that require nonexistent moments. For a degree-\(d\) polynomial basis, the
mass matrix contains moments up to order \(2d\). A product Student reference
guarantees all required entries are finite when \(\nu>2d\); when
\(\nu\leq2d\), the highest-order entries can diverge. A Student component is
safe as a proposal before it is adopted as the TT integration measure.

### 3.6 Moment contractions and the proposal geometry contract

The retained quadratic form supplies moments through the same measure used for
its normalization. For a one-state marginal

\[
  \widehat p(z)
  =\frac{H(z)E H(z)^\top+\tau q_0(z)}{Z_c},
\]

define the basis moment matrices

\[
  M^{(0)}_{ab}=\int \phi_a(z)\phi_b(z)\,d\mu(z),\qquad
  M^{(1)}_{ab}=\int z\,\phi_a(z)\phi_b(z)\,d\mu(z),\qquad
  M^{(2)}_{ab}=\int zz^\top\,\phi_a(z)\phi_b(z)\,d\mu(z).
\]

Then

\[
\begin{aligned}
  Z_c
    &=\operatorname{tr}\!\left(E\,\int H(z)^\top H(z)\,d\mu(z)\right)
      +\tau\int q_0\,d\mu,\\
  m
    &=Z_c^{-1}\left[
       \int z\,H(z)E H(z)^\top\,d\mu
       +\tau\int z\,q_0(z)\,d\mu\right],\\
  P
    &=Z_c^{-1}\left[
       \int zz^\top H(z)E H(z)^\top\,d\mu
       +\tau\int zz^\top q_0(z)\,d\mu\right]-mm^\top.
\end{aligned}
\]

For the normalized probabilists' Hermite factors used by the current route,

\[
  u\,\overline{\mathrm{He}}_k(u)
  =\sqrt{k+1}\,\overline{\mathrm{He}}_{k+1}(u)
   +\sqrt{k}\,\overline{\mathrm{He}}_{k-1}(u),
\]

so \(M^{(1)}\) and \(M^{(2)}\) are finite banded contractions of the ordinary
identity mass matrix. This is the inexpensive exact moment path for a
Gaussian-reference retained TT. An RBF or Student-reference route must provide
its own \(M^{(0)},M^{(1)},M^{(2)}\); reusing the Hermite identity is a measure
error.

The proposal and target must live on the same random object. For an auxiliary
particle filter let \(I\) be the discrete ancestor, \(J\) a proposal-component
label, and \(x\) the new state. With previous normalized weights
\(\bar W_{t-1}^{(i)}\), auxiliary probabilities \(\rho_{t,i}\), and conditional
components \(q_{t,j}(x\mid x_{t-1}^{(i)})\),

\[
\begin{aligned}
  \Gamma_t(i,x)
    &=\bar W_{t-1}^{(i)}
      f_t(x\mid x_{t-1}^{(i)})g_t(y_t\mid x),\\
  Q_t(i,x)
    &=\rho_{t,i}\sum_{j=1}^J\alpha_j
      q_{t,j}(x\mid x_{t-1}^{(i)}),\\
  w_t(i,x)
    &=\frac{\Gamma_t(i,x)}{Q_t(i,x)}.
\end{aligned}
\]

If \(J\) is sampled explicitly, it is still the marginalized denominator
\(\sum_j\alpha_jq_{t,j}\) that belongs in the balance/MIS weight. A current-state
marginal proposal cannot be divided into this joint target without either
including the ancestor/conditional law or analytically integrating the
previous state. The Stage 0 geometry tests must therefore check both the
joint density and every coordinate/Jacobian conversion.

For an affine physical map \(x=m+Lu\) with nonsingular \(L\),
\(dx=|\det L|\,du\). Thus a reference-coordinate Gaussian or Student component
has physical density

\[
  q_x(x)
  =q_u\!\left(L^{-1}(x-m)\right)|\det L|^{-1}.
\]

The corresponding log-density term is
\(\log q_u(u)-\log|\det L|\). Every proposal arm must use the same orientation
of \(L\) (the current code uses a lower-triangular Cholesky factor) in sampling,
inverse evaluation, and the Jacobian term. A sign or transpose mismatch can
leave samples finite while biasing every importance weight.

UKF sigma-point weights are quadrature weights and can be negative; they are
not proposal-mixture weights by themselves. A UKF-guided proposal must turn the
moments or sigma points into a bona fide normalized Gaussian/Student density
or a nonnegative mixture, and must check that density independently. Likewise,
an LEDH transport proposal must include the determinant of its full map, not
only the local Gaussian factor.

### 3.7 Implementation trace and known scope boundaries

The current call chain provides useful anchors but does not yet implement all
of the proposed candidates:

- squared_tt_engine_gaussian_tf.py::_hermite_product_basis (lines 79--83)
  and _christoffel_rows (130--181) define one Gaussian-reference Hermite basis
  and its frozen row law. A Student or RBF measure needs a new mass/row-law
  route.
- run_value_filter_branch_axis_gaussian (319--505) constructs a basis once and
  consumes frozen moment hints. Its time loop changes the affine map supplied
  by those hints, but it does not recursively derive the hints from the
  retained quadratic form.
- The same loop creates fresh random cores0 at t>=1 (462--476); this is not a
  continuation or warm-start algorithm. Any continuation experiment must be
  named and measured separately.
- retained_quadratic_form_tf.py (201--337) carries the prefix cores, suffix
  Gram, normalizer, map, and floor. Moment-derived maps must contract this
  full state rather than treating the prefix coefficient vector as a
  normalized density.
- zhao_cui_frozen_proposal_apf_tf.py::_evaluate_core (787--935) contains the
  initial and transition log-weight sums. A true DMIS arm must bind
  initial_log_proposal_density and transition_log_proposal_density to
  log-sum-exp complete mixtures. The block-combination helper uses log-density
  addition for a product of block conditionals; that operation must not be
  mistaken for a mixture over proposal components.
- c2_sv_frozen_proposal_apf_tf.py::compile_c2_dmis_proposal_branch
  (552--774) already constructs the two-bank TT/Student denominator with
  log-sum-exp and component base masses. Stage 0 must recompute that denominator
  independently, including the discrete ancestor law, before reusing the
  branch as the correctness comparator.
- The existing defensive_nu option changes a Student defensive floor inside
  an otherwise Gaussian-reference TT calculation. It is not evidence that the
  TT integration measure or Hermite mass matrix has become Student.

These are implementation boundaries to test, not assumptions to silently
paper over. A passing structural function test does not establish that every
claim-bearing endpoint reaches the intended general route.

## 4. Candidate matrix and staged execution

The stages are deliberately ordered from lowest mathematical risk to highest.

### Stage 0: invariant and target checks

Run on one- and two-dimensional analytic fixtures:

- exact normalization of every Gaussian, Student, and mixture proposal;
- complete-mixture denominator versus direct recomposition;
- permutation invariance of component labels and samples;
- Gaussian transition and observation density against independent formulas;
- retained-quadratic-form moment and Cholesky identities;
- the closed-form Gaussian-reference RBF Gram entry against quadrature;
- failure of a duplicated Hermite/RBF constant channel and nonsingularity
  after removing or orthogonalizing it;
- the product-Student moment boundary through order \(2d\);
- the C2 nonzero-observation likelihood envelope on the frozen fixture;
- finite-difference score of the exact frozen-proposal scalar;
- current uniform-route regression;
- direct \(Z_t^{\mathrm{fin}}\) integration versus the proposal estimator on a
  tractable fixture.

No ESS or likelihood ranking is interpreted until these pass.

### Stage 1: exact-factor finite-target proposal ladder

Keep the current Hermite TT representation unchanged. Compare:

1. retained TT only;
2. current Gaussian-hint marginal proposal;
3. transformed-observation UKF/Kalman Student proposal;
4. bootstrap conditional proposal;
5. LEDH-local Gaussian proposal, when its density correction is available;
6. fixed half-mixture of TT and Student;
7. pilot-nominated interior-alpha DMIS mixture.

Use the same target data, particle count, ancestry contract, and seeds within
paired replicates. The primary quantities are per-time minimum normalized ESS,
maximum normalized weight, and the finite-program log normalizer versus an
independent \(Z_t^{\mathrm{fin}}\) reference. The complete DMIS result is the
correctness comparator for later TT changes; PF compatibility is an
explanatory/reference diagnostic, not an exactness certificate.

The direct TT route and the APF route are not silently assigned the same
finite target. The direct route carries a normalized TT density, while the APF
route is conditional on its realized empirical ancestor cloud. Stage 1 records
both target identities and compares their outputs only as a compatibility
diagnostic unless the carried measures are independently shown equal.

The existing C2 audit shows why this stage comes first: exact-factor DMIS was
near the PF total while retained TT was far away, but simple Gaussian and
Student proposals still outperformed the tested half-mixture at salient times.
That is proposal evidence, not a superiority claim. Compare DMIS first with an
independent integral or high-particle estimate of \(Z_t^{\mathrm{fin}}\);
agreement with a PF run is reported separately because the PF carries a
different previous-density approximation.

For each time step, the executable order is fixed:

1. Read the frozen carried state \(S_{t-1}\) (or the initial prior at
   \(t=0\)) and the frozen proposal bank.
2. Evaluate the exact target log density, the complete marginalized proposal
   log density, and the declared base mass for every bank row.
3. Form log weights with one log-sum-exp denominator, then compute the finite
   normalizer, normalized weights, ESS, and score.
4. Independently evaluate or quadrature-check \(Z_t^{\mathrm{fin}}\) on the
   fixture; only then record the proposal comparison.
5. If the route is a filtering candidate, construct the retained quadratic
   state from the same finite target and pass that state to the next step.

The bank, ancestry, component labels, map, and fitting decisions are frozen
for the score check. Any adaptive version is a separate finite program and is
not compared under the frozen-program score contract.

### Stage 2: recursive moment-derived map

Keep the Hermite basis and proposal route fixed. Replace only the external
Gaussian-hint map with the lagged map derived from the retained predicted
density:

\[
  \widehat p_{t-1}
  \to \widehat p_t^-
  \to (m_t^-,P_t^-)
  \to T_t.
\]

Compare against the current frozen-hint map using held-out target rows. First
use one map update per time step. Do not begin with an inner fixed-point loop.

Promotion of this stage requires lower held-out error without worse direct
normalizer error or recursive PF discrepancy. A lower training RMS alone is
not sufficient.

### Stage 3: fitter and basis ladder

For the winning map choice, test:

- Hermite degrees 6, 8, and 10;
- fixed separable RBF centers at declared UKF offsets and a finite width grid;
- Hermite-plus-RBF hybrid with the existing Hermite constant exactly once and
  broad nonconstant tail channels.

Begin with one frozen target at the earliest divergent time. Freeze the map,
stabilization constant, fit rows, held-out rows and weights, TT rank, ALS sweep
order, and stopping schedule across arms. Only a candidate that survives this
one-step comparison enters the recursive horizon. Record actual rank, ALS
sweeps, solve condition, and canonicalization policy. If rank enrichment is
tested, select the schedule on calibration data and freeze it before the claim
run.

### Stage 4: reference-density rewrite

Only if a Student or mixture proposal is consistently useful relative to the
Gaussian arms in Stages 1--3 should the Student-based TT reference be
implemented. The first route is the product-Student measure; an elliptical
Student measure is a separate non-product candidate. This stage must be a new
route identity, not a silent replacement of the Gaussian route. Implement and
test the complete mass matrix, row law, retention, normalizer, and analytical
derivative together.

### Stage 5: integrated candidate

Combine only the mechanisms that independently pass their prior stages:

\[
  \text{recursive map}
  +\text{selected basis}
  +\text{exact DMIS proposal correction}.
\]

This integrated run remains an optional diagnostic candidate until it clears
the heuristic set and multi-replicate uncertainty analysis.

## 5. Diagnostics and interpretation

### 5.1 Representation diagnostics

For independent held-out rows, report:

\[
  E_{L^2}^{\mathrm{hold}}
  =\frac{\sum_j w_j[h_t(u_j)-s_t(u_j)]^2}
         {\sum_j w_js_t(u_j)^2},
\]

shell-wise versions of the same quantity, and the direct normalizer gap

\[
  \Delta_Z=\log Z_H-\log Z_T.
\]

Here \(Z_H\) is the Gram contraction of the fitted representation and \(Z_T\)
is an independent integral (or high-particle estimate) of the same
finite-program target \(Z_t^{\mathrm{fin}}\). A PF estimate using the true
particle cloud is a different, reference-compatibility quantity. The two
normalizers must never be conflated.

If \(s_t\) is the scaled square-root target and \(h_t\) its fitted
representation, then

\[
  \bigl|\|h_t\|_{L^2(\mu_t)}^2-\|s_t\|_{L^2(\mu_t)}^2\bigr|
  \leq
  \|h_t-s_t\|_{L^2(\mu_t)}
  \bigl(\|h_t\|_{L^2(\mu_t)}+\|s_t\|_{L^2(\mu_t)}\bigr).
\]

Thus a small global \(L^2\) residual controls the normalizer gap only when the
tail contribution is included in the integration measure. Central-row RMS
alone gives no such guarantee.

For a fair comparison of basis arms at a fixed map, use the same stabilization
constant \(c_t\) and the same held-out integration rows and weights, or also
report the residual after undoing the scale. A candidate that changes \(c_t\)
through its row set changes the finite least-squares objective as well as the
basis, so that result is not a basis-only comparison.

Also record Gram eigenvalue margins, condition numbers, TT ranks, ALS residuals,
and the fraction of held-out error in central versus tail shells.

### 5.2 Recursive diagnostics

Record at every time:

- direct per-step normalizer error against PF or an independent reference;
- cumulative error and its increment;
- retained mean/covariance versus the reference;
- map shift and covariance condition;
- held-out central and tail residual;
- TT and proposal ESS;
- maximum normalized weight; and
- whether the first large discrepancy is representation or proposal driven.

The first time at which the error grows is more informative than the final
cumulative error.

### 5.3 Statistical interpretation

Use at least twelve paired diagnostic branches after the mechanics smoke, or a
predeclared smaller number for a bounded pilot. Report paired bootstrap
intervals and sign tests for ESS and likelihood differences. One seed or a
single short run is descriptive only. Extreme minima and maxima are not ranked
without uncertainty support.

Apply the heuristic-dominance gate conditionally at the predeclared salient
times \(t=3,t=4\), the minimum-ESS time, the largest innovation time, and the
full horizon. A complex method losing to the simple Gaussian, Student, or
bootstrap proposal remains a promotion veto even if its average is favorable.

## 6. Analytical-gradient contract

For every claim-bearing arm, freeze before evaluation:

- proposal component identities and mixture weights;
- random samples, ancestry, and base masses;
- basis family, centers, widths, rank, and channel layout;
- map-update count and ALS sweep order;
- any enrichment or stopping schedule; and
- all numerical branches that affect the returned scalar.

For a fixed ALS update

\[
  A_uc_u=b_u,
\]

the exact tangent is

\[
  \dot c_u=A_u^{-1}(\dot b_u-\dot A_uc_u).
\]

If the recursive map is derived from TT moments, differentiate the moment
contractions, Cholesky factor, affine map, Jacobian, and all downstream target
terms. If the map or proposal is deliberately frozen, the resulting score is
the score of that frozen finite program, not the total derivative of an
adaptively refitted algorithm.

More explicitly, if the carried density is parameter dependent, then

\[
  D_\theta\log\gamma_{t,\theta}^{\mathrm{fin}}
  =D_\theta\log f_{t,\theta}
   +D_\theta\log g_{t,\theta}
   +D_\theta\log\widehat p_{t-1,\theta}.
\]

The last term is included only when the recursive carried state is part of the
differentiated finite program. If the state is a frozen snapshot, omitting it
is intentional and the result must be labelled a frozen-program (partial)
score. If samples are generated as \(X_i(\theta)\) from a parameter-dependent
map, \(D_\theta\) also includes the pathwise derivative through
\(X_i(\theta)\); freezing physical samples removes that term by construction.

Hard rank selection, adaptive pivoting, data-dependent stopping, and online
component selection are not ordinary analytical operations. They must be
performed offline and frozen, or replaced by fixed-size smooth parameterizations.

## 7. Preconditions and stop conditions

Before a long run, verify:

- the target and proposal are evaluated in the same physical coordinate
  convention;
- every proposal has full support where the target is positive;
- every finite target is measurable, nonnegative, and integrable;
- every defensive mixture has \(\alpha\in(0,1]\), a normalized
  \(q_{t,\mathrm{def}}\) that is positive on the target support, and a checked
  finite second-moment integral \(\int(\gamma_t^{\mathrm{fin}})^2/q_{t,\mathrm{def}}\);
- every mixture density includes all components;
- all base masses sum to one;
- every Gram/mass matrix is positive definite where required;
- maps and scales are finite and invertible;
- the XLA graph has a stable signature and no unauthorized pfor path; and
- the run manifest records commit, environment, dtype, device, seeds, command,
  plan, and artifact paths.

Stop interpretation for a failed identity, nonfinite value, wrong measure,
failed finite difference, or corrupted artifact. A low ESS or a candidate
loss is a repair trigger, not by itself a reason to reject the research
direction.

### 7.1 Skeptical pre-execution audit and bounded budget

The design passed the 2026-08-31 skeptical audit after the documentation pass
made four hidden assumptions explicit: the direct TT and empirical-APF finite
targets differ; RBF channels must be separable for the planned TT contractions;
the Hermite basis already contains the only literal constant; and the first
Student TT reference must be a product law rather than an unannounced
elliptical measure. The executable stages below test those assumptions. The
baseline is the current route plus the cheap Gaussian, Student, bootstrap, and
stationary adversaries; no weak baseline is used as the sole comparator. ESS,
shell error, rank, and conditioning remain explanatory unless the evidence
contract explicitly promotes them. The PF comparison is labelled a
reference-compatibility diagnostic because it uses the true particle filter,
whereas DMIS targets the finite carried-density program. Historical attempt05
artifacts are not reused as current measurements; only their documented
failure mechanism informs the hypotheses. A change of basis or reference
changes the represented finite program and is compared as a new candidate, not
silently treated as the same estimator.

The campaign budget is bounded as follows:

- Stage 0: one CPU reference fixture and at most three focused repair attempts;
- Stages 1--3: one calibration partition, one untouched claim partition, and
  at most twelve paired branches per retained candidate;
- at most two pilot mixture-weight choices and three total infrastructure
  retries per stage, all in fresh versioned output directories;
- no Student-measure TT rewrite, HMC run, package installation, or expanded
  particle/seed campaign under this plan; and
- a localized harness or serialization repair is allowed only when the target,
  method, hardware class, criteria, and total budget remain unchanged.

Every serious run writes a manifest containing the commit, exact command,
environment, TensorFlow/XLA and device settings, seeds, data partition,
resolved proposal/basis/map settings, wall time, and artifact hashes.

The pre-mortem is:

| How the run could mislead us | Discriminating check |
| --- | --- |
| DMIS appears correct because the PF and DMIS share a target or Jacobian bug | independent analytic/quadrature \(Z_t^{\mathrm{fin}}\) fixture and a second implementation of each density |
| a Student arm wins only because it received a different bank or seed | deterministic paired banks, identical base masses, and component-label permutation tests |
| a recursive map lowers central RMS while worsening tails and later times | shell-wise residuals, retained moment error, and the first-time error recursion |
| a higher-degree or hybrid fit lowers training RMS but has an invalid Gram | untouched rows, eigenvalue margins, and direct \(Z_H\) versus \(Z_T\) |
| a local finite-difference score passes while map/proposal dependence is omitted | perturb the map and proposal parameters separately and compare the declared partial and total programs |
| an eager/reference path differs from the claim-bearing XLA path | same-input parity at the callable boundary, stable signatures, and device/precision manifest fields |

## 8. Expected outcomes and decision logic

The most likely outcomes are:

1. DMIS passes, TT remains poor: use TT/UKF/LEDH only as proposals or
   control variates; retain exact target correction as the likelihood route.
2. Recursive map improves held-out error: continue with the map as a
   candidate, but verify that recursive error does not grow at later times.
3. Higher Hermite degree helps central error but not tails: add localized
   RBF channels or a mixture proposal; do not infer that degree alone solves
   the problem.
4. RBF or hybrid helps representation: implement its full Gram and tangent
   path before making a gradient claim.
5. Student proposal helps but Student TT reference does not: keep Student
   as a defensive proposal and do not rewrite the TT measure.
6. Student reference passes all representation tests: open a new reviewed
   implementation plan for the complete Student-measure TT route.

The recommended default research direction is therefore:

\[
  \boxed{
  \text{exact-factor finite-target DMIS first, recursive moment map second,
  basis changes third, Student TT reference last}
  }
\]

This ordering gives a useful result even if the global TT approximation remains
inadequate, while preserving a clean analytical-gradient contract.

## 9. Planned artifacts

Use a fresh versioned artifact directory for each attempt. Preserve:

- this plan and its review note;
- the MathDevMCP scope audit in
  docs/plans/bayesfilter-c2-coherent-tt-proposal-testing-plan-20260831-mathdevmcp-audit.md;
- invariant and finite-difference test logs;
- proposal and basis manifests with hashes;
- held-out and shell diagnostic tables;
- per-time recursive summaries;
- paired uncertainty calculations;
- run manifest with environment and device provenance; and
- a result note that separates implementation validity, numerical validity,
  proposal quality, representation quality, and scientific interpretation.

## 10. Documentation audit record

The proposition--proof revision was compiled with two successful
`pdflatex -interaction=nonstopmode -halt-on-error` passes (27-page PDF; no
fatal errors, undefined references, or duplicate labels). The final manuscript
SHA-256 is
`11d8622befa67e4d00d51b0f425442e09969a6ada143f2b977dc67a1d21ada34`.
The Hermite antiderivative proof was corrected to handle $r=1$ separately,
avoiding an undefined $\operatorname{He}_{-1}$ notation; this is a
proof-presentation repair, not a change to the identity.

MathDevMCP was rerun after that repair with `audit-math-document-rigor`,
SymPy validation, and the actionable report profile. It found no algebraic
refutation. Its coverage is
explicitly `partial_coverage` (30 selected equations, six distinct exposition
issues, five open diagnostic obligations); the durable report is
`docs/plans/bayesfilter-c2-coherent-tt-proposal-testing-plan-20260831-latex-mathdevmcp-audit.md`
with the machine-readable companion `.json`. The remaining flags concern
formalization routing and local dimension prose, not a demonstrated false
identity. They remain non-certification boundaries and are covered by the
Stage 0 fixture and score checks.

## 11. Execution record

The bounded execution result is
`docs/plans/bayesfilter-c2-coherent-tt-proposal-testing-execution-result-2026-08-31.md`.
It records the exact commands, manifests, artifact digests, decision table,
and inference status. In brief, all 48 Stage 0 checks passed; the 72-branch
Stage 1 ladder passed engineering and exact-factor score screens, but retained
TT had minimum normalized ESS below `7.5e-4` across branches and
the heuristic-dominance veto fired. The independent Stage 2 integration
completed with the predeclared `christoffel_qmc_uncertainty` continuation veto.
Because target-integration precision is not yet sufficient at all diagnostic
times, no recursive-map or basis candidate is promoted. The next run must
repair that precision boundary before interpreting those stages.
