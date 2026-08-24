# LEDH-PFPF-GenUT dual-cap and trust-region methods for the q=20 NeuTra problem

Date: 2026-08-24  
Status: `CONDITIONAL_ROUTE_POSSIBLE_CURRENT_ROUTE_NOT_ADMISSIBLE`

## Executive verdict

Yes, the LEDH-PFPF-GenUT dual-cap/trust-region construction can help, but only
at one layer of the problem. It is useful as a numerically controlled local
proposal and finite-cloud shape correction. It can reduce ill-conditioned
higher-moment updates, preserve selected empirical first/second moments after
the reset, and make a local LEDH proposal closer to the observation geometry.

It does **not**, by itself, do any of the following:

- turn GenUT sigma points into IID draws from a density;
- prove that the capped cloud has the target density;
- provide a full-support normalizing flow;
- remove finite-particle self-normalization bias from replay;
- create mass in a mode absent from the proposal support; or
- prove Gaussian whitening of a learned NeuTra pullback.

There is a conditional solution if the dual-cap route is embedded in a larger
measure-correct construction: use a genuinely invertible, density-evaluable
LEDH map for proposal generation; retain a defensive full-support proposal;
use unnormalized mass-carrying blocks or deterministic-mixture (AMIS) weights
for replay; and use a tempered SMC sequence with invariant mutation kernels for
mode traversal. The current artifacts and code do not establish all of these
assumptions. Therefore the dual-cap/trust-region route should remain a
proposal-candidate, not a NeuTra/HMC admission route.

The strongest negative conclusion is also precise:

> If the required claim is "dual-cap GenUT alone turns the existing finite
> replay cloud into a density-faithful, globally mode-covering IID Gaussian
> whitening map," there is no such guarantee, and the claim is mathematically
> false in general.

The remainder gives the derivations, the useful conditional route, and the
tests required before implementation or promotion.

## 1. Target and notation

Let the current q=20 parameter target be represented by an unnormalized density

\[
  \widetilde\pi(\theta) > 0, \qquad
  Z = \int \widetilde\pi(\theta)\,d\theta \in (0,\infty), \qquad
  \pi(\theta)=Z^{-1}\widetilde\pi(\theta).
\]

The target here is the repository's declared UKF-defined q=20 target. It is not
automatically the exact nonlinear state-space posterior. Let \(\rho(z)\) be the
standard Gaussian density on \(\mathbb R^d\), and let \(T_\phi\) denote a
candidate NeuTra map. A density-level claim requires \(T_\phi\) to be a
\(C^1\) diffeomorphism (or a carefully declared alternative chart):

\[
 q_\phi(\theta)
 =\rho(T_\phi^{-1}(\theta))
   \left|\det D T_\phi^{-1}(\theta)\right|.
\]

For a weighted source cloud write

\[
 X=\{(x_n,w_n)\}_{n=1}^N,\qquad w_n\geq 0,\qquad \sum_n w_n=1,
\]

with

\[
 \mu_X=\sum_n w_nx_n,\qquad
 \Sigma_X=\sum_n w_n(x_n-\mu_X)(x_n-\mu_X)^\top.
\]

The Contract-E/OT reset and the dual-cap correction produce an equal-weight
cloud \(Y=\{y_n\}_{n=1}^N\). This is a finite empirical representation. It is
not a density unless an additional sampling law and density evaluation rule are
specified.

## 2. Literature survey and exact relevance

The survey was restricted to primary technical sources. Local copies of the
material papers are stored under
`.localresources/papers/ledh_replay_solution_20260824/`; existing local copies
of Neal AIS, Del Moral--Doucet SMC samplers, Parno--Marzouk transport maps, and
the Ebeigbe GenUT paper is retained at
`.localresources/papers/ebeigbe-et-al-genut-2104.01958.txt`.

| Source | What the source actually establishes | Use for this problem | Boundary |
|---|---|---|---|
| Ebeigbe et al., *A Generalized Unscented Transformation for Probability Distributions*, arXiv:2104.01958, Secs. III--V | A (2d+1) sigma-point rule can match mean/covariance and selected diagonal skewness/kurtosis under feasibility conditions; constraints can sacrifice exact higher moments | Justifies selected-moment local shape control and the feasibility diagnostics | A sigma-point quadrature rule is not an IID density representation; constrained moments are not all retained |
| Li and Coates, *Particle Filtering with Invertible Particle Flow*, arXiv:1607.08799, Sec. IV and Algorithm 1 | A discretized LEDH map can be made invertible under step-size/regularity conditions; the proposal density then uses the pre-flow density and the map Jacobian, and the PF-PF weight includes post-flow transition and observation factors | Supplies the correct density-correction contract for an LEDH proposal | It does not make a Gaussian/local-linear flow exact for a nonlinear target; a later non-invertible cap is outside the identity |
| Cornuet et al., *Adaptive Multiple Importance Sampling*, arXiv:0907.1254, Secs. 2--5 | Recycle all past samples by recomputing weights with the deterministic mixture of all proposal densities; adaptive convergence needs additional conditions | Direct answer to experience-replay weighting | Keeping old normalized weights, or silently dropping proposal components, is not AMIS |
| Hesterberg, *Weighted Average Importance Sampling and Defensive Mixture Distributions*, Technometrics 37 (1995), Sec. 6 | A positive defensive mixture can bound ratios when the target is a mixture component; more generally it supplies coverage relative to a safe component and yields a second-moment bound conditional on integrability | Provides the full-support safety component | Support alone does not imply finite variance, and the unknown target cannot simply be inserted as a sampled component |
| Neal, *Annealed Importance Sampling* (2001), and Del Moral--Doucet SMC samplers | A sequence of intermediate measures can connect an easy law to a difficult target through incremental weights and mutation | Supplies a bridge for separated modes | A finite schedule and finite mutation budget do not prove mode mixing |
| Fearnhead and Taylor, *An Adaptive Sequential Monte Carlo Sampler*, arXiv:1005.1193 | Adaptive, target-invariant MCMC kernels can be tuned within an SMC sequence under stated assumptions | Supports a mutation/tuning lane after proposal generation | It does not repair a wrong proposal density or a biased replay denominator |
| Gerber and Chopin, *Sequential Quasi-Monte Carlo*, arXiv:1402.4039 | Randomized low-discrepancy particle constructions can improve integration error under regularity and bounded-weight assumptions | Useful variance reduction for the base-noise lane | RQMC cannot create support or guarantee discovery of a missed mode |
| Reich, *A non-parametric ensemble transform method for Bayesian inference*, arXiv:1210.0375 | A finite weighted-to-equal ensemble transform is consistent in a large-ensemble limit under its assumptions | Explains why OT/Contract-E can be a useful representation reset | The finite transform is not a proposal-density change of variables and is not exact at finite (N) |

The source distinction matters: GenUT and ensemble transforms address finite
quadrature/representation; Li--Coates addresses proposal correction; AMIS and
defensive mixtures address sampling measure and weight stability; tempering and
mutation address global exploration. No single source supplies all four.

For reproducibility, the technical statements above were checked against these
local text anchors (line numbers refer to the stored `pdftotext` files):

- GenUT: `.localresources/papers/ebeigbe-et-al-genut-2104.01958.txt:84-95,114-164`
  (constraint count, selected diagonal moments, and constrained second-order
  accuracy).
- PF-PF: `.localresources/papers/ledh_replay_solution_20260824/li-coates-2017-particle-filtering-invertible-flow.txt:236-270`
  (proposal density, Jacobian, weight, and the non-invertible-flow warning).
- AMIS: `.localresources/papers/ledh_replay_solution_20260824/cornuet-et-al-2009-amis.txt:68-110,208-222,428-465`
  (deterministic mixture, recycling, and adaptive-convergence caveat).
- Defensive mixture: `.localresources/papers/ledh_replay_solution_20260824/hesterberg-1995-defensive-mixture.txt:295-340`
  (bounded weight and support/coverage discussion).
- Adaptive SMC: `.localresources/papers/ledh_replay_solution_20260824/fearnhead-taylor-2013-adaptive-smc.txt:269-300,620-640`
  (invariant mutation and assumptions for adaptation).
- SQMC: `.localresources/papers/ledh_replay_solution_20260824/gerber-chopin-2015-sqmc.txt:230-245,638-650`
  (regularity-dependent rates and explicit transition-map requirement).
- Ensemble transform: `.localresources/papers/ledh_replay_solution_20260824/reich-2013-ensemble-transform.txt:100-170`
  (finite weighted-to-equal transform and its large-ensemble framing).

## 3. What the local implementation actually computes

The following source anchors were inspected.

1. `bayesfilter/highdim/genut_shape_lm_tf.py:121-167` implements the smooth
   row-RMS cap
   \[
     u\mapsto u\left(1+\frac{\|u\|^2/d}{\rho^2}\right)^{-1/2}
   \]
   and its total JVP. It also reports Pearson and finite-particle feasibility
   margins; those are necessary diagnostics, not realizability theorems.
2. `bayesfilter/highdim/dual_cap_genut_primal_tf.py:193-245` computes weighted
   source moments, standardizes an equal-weight reset cloud, performs diagonal
   and pairwise corrections, applies the coordinate cap, restandardizes, and
   restores the source mean and Cholesky covariance.
3. In that same file, lines 235--242 apply
   \[
     g_{b,p}(u)=u\left(1+(u/b)^p\right)^{-1/p}
   \]
   coordinatewise. The implementation reports cap activity and the post-cap
   covariance residual; it does not evaluate a joint proposal density for this
   cloud operation.
4. `bayesfilter/highdim/genut_guided_proposal_tf.py:900-1045` applies the
   Contract-E reset and then the dual-cap/trust-region shape operation. The
   PF-PF importance logits are formed earlier in
   `ledh_pfpf_genut_initial_rqmc_tf.py:500-527` and subsequent transition blocks
   before the reset output is used for the next equal-weight representation.
   Thus the current log determinant belongs to the LEDH proposal map; it is not
   the determinant of the later OT/GenUT cloud transformation.
5. The local monograph
   `docs/fable-rewrite/monograph/chapters/ch19c_dpf_implementation_literature.tex:156-183,230-342`
   records the PF-PF ratio, the particle-specific LEDH determinant product, the
   UKF/EKF covariance lifecycle, and the warning that all density terms must be
   evaluated on the same pre/post-flow path. Its section
   `:477-512` explicitly separates proposal correction from exact nonlinear
   filtering.
6. The q=20 replay note
   `docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md:298-388`
   already classifies the stored normalized SMC populations as finite empirical
   blocks, not as an unnormalized unbiased target-measure estimator.

These facts prevent a category error: a stable equal-weight cloud after a
reset is not automatically the density from which its rows were sampled.

## 4. Moment retention is not density retention

### Proposition 1 (selected moments do not identify a density)

Suppose two probability measures have the same first four scalar moments. They
need not be the same measure, and therefore matching mean, variance, skewness,
and kurtosis cannot establish density faithfulness or IID sampling.

#### Proof

Let (P) be the standard normal law. Let (Q) put mass (2/3) at (0) and
mass (1/6) at each of (+\sqrt 3) and (-\sqrt 3). Both laws have

\[
 E[X]=0,\quad E[X^2]=1,\quad E[X^3]=0,\quad E[X^4]=3.
\]

They are nevertheless different: (P) is absolutely continuous and (Q) is
discrete. Hence equality of those four moments does not imply equality of
measures. In several dimensions, product versions give the same counterexample
for marginal moments, and additional cross-moments are still insufficient in
general. A finite GenUT cloud supplies only finitely many constraints, whereas
the target density is an infinite-dimensional object. `QED`

The GenUT source makes the same dimensional point in a constructive way: with
(2d+1) points it targets selected diagonal components rather than the full
third- and fourth-order tensors. The local dual-cap route adds pairwise moment
directions, but remains a finite correction of a finite cloud.

### Proposition 2 (what affine restoration does prove)

Let `Y` be an equal-weight cloud with finite mean and positive-definite sample
covariance `C_Y`. Let `\mu_X,\Sigma_X=L L^\top` be the weighted source mean
and covariance. Define

\[
  \widehat Y_n=\mu_X+L C_Y^{-1/2}(y_n-\bar y).
\]

Then the equal-weight mean and covariance of `\widehat Y` are exactly
\(\mu_X\) and \(\Sigma_X\), up to the chosen matrix square-root convention.

#### Proof

The centered equal-weight mean is zero by construction. Its covariance is

\[
 L C_Y^{-1/2} C_Y C_Y^{-1/2}L^\top
 = L L^\top=\Sigma_X.
\]

This is a finite empirical identity. It says nothing about the distribution of
the rows beyond those two moments, and it does not restore moments altered by a
nonlinear cap unless those moments are recomputed and corrected afterward.
`QED`

This is the valid mathematical interpretation of the local "variance retaining"
claim: first and second empirical moments can be restored. It is not a claim of
variance-preserving density transport.

## 5. Trust-region caps are stable finite maps, not global normalizing flows

### Proposition 3 (support of the implemented smooth caps)

For `d >= 1`, define the row-RMS cap used by the local code,

\[
 C_\rho(u)=u\left(1+\frac{\|u\|^2}{d\rho^2}\right)^{-1/2},
 \qquad \rho>0.
\]

Then `C_\rho` is `C^1`, injective along each ray, and

\[
 \|C_\rho(u)\|<\sqrt d\,\rho.
\]

Likewise, for even `p >= 2`,

\[
 g_{b,p}(u)=u\left(1+(u/b)^p\right)^{-1/p},
 \qquad b>0,
\]

is `C^1`, strictly increasing, and satisfies `|g_{b,p}(u)| < b`.
Neither map is a surjection from \(\mathbb R^d\) to \(\mathbb R^d\).

#### Proof

For the radial map, write (r=\|u\|). Its radial magnitude is

\[
 h(r)=\frac{r}{\sqrt{1+r^2/(d\rho^2)}}.
\]

Differentiation gives

\[
 h'(r)=\left(1+\frac{r^2}{d\rho^2}\right)^{-3/2}>0,
 \qquad \lim_{r\to\infty}h(r)=\sqrt d\,\rho.
\]

Thus it is smooth and monotone on each ray but has a bounded image. For the
coordinate map, set (s=(u/b)^p\geq0). Direct differentiation gives

\[
 g'_{b,p}(u)=(1+s)^{-1/p-1}>0,
 \qquad \lim_{u\to\pm\infty}g_{b,p}(u)=\pm b.
\]

The image bounds prove non-surjectivity. A normalizing flow on an unconstrained
parameter space requires a bijection between the declared supports (or an
explicit bounded-support base and inverse chart). A bounded cap alone is not
such a flow. `QED`

### Consequence for the current route

The cap may be differentiated, and its derivative may be included in a
**finite-cloud JVP**, as the local implementation does. But if one claims a
proposal density after the cap, one must specify the law before the cap, the
support-restricted pushforward density, and the joint dependence introduced by
restandardizing the whole cloud. The current PF-PF determinant does not include
that joint cloud operation. A cap-active fraction or a small covariance residual
therefore cannot certify density faithfulness.

The cap can be used safely in either of two ways:

1. as an internal bounded update of a proposal **parameter** while the actual
   sampled proposal remains a separately density-evaluable full-support law; or
2. as a component of a declared bounded-support proposal whose density and
   support are evaluated explicitly, with a defensive full-support mixture
   component retained for the target.

It must not be silently appended to the Li--Coates map while retaining the old
log determinant.

## 6. PF-PF correction: what LEDH can make exact

### Proposition 4 (change of variables for an invertible LEDH proposal)

Let `q_0` be a density on `\mathbb R^d`, positive on the relevant preimage
support, and let `\Phi` be a `C^1` diffeomorphism with nonzero determinant. If
`X ~ q_0` and `Y = \Phi(X)`, then

\[
 q_1(y)=q_0(\Phi^{-1}(y))
       \left|\det D\Phi^{-1}(y)\right|.
\]

For any integrable `f`,

\[
 E_{q_0}\left[
   \frac{\widetilde\pi(\Phi(X))\,|\det D\Phi(X)|}{q_0(X)}
   f(\Phi(X))
 \right]
 =\int \widetilde\pi(y)f(y)\,dy.
\]

#### Proof

The first identity is the change-of-variables theorem. For the second, set
`y=\Phi(x)`:

\[
 \int q_0(x)\frac{\widetilde\pi(\Phi(x))|\det D\Phi(x)|}{q_0(x)}
 f(\Phi(x))\,dx
 =\int \widetilde\pi(y)f(y)\,dy.
\]

`QED`

For the discrete LEDH map in Li--Coates, each pseudo-time step is an affine
map

\[
 x_{j+1}=(I+\varepsilon_jA_j)x_j+\varepsilon_jb_j,
\]

so, when every factor is nonsingular,

\[
 |\det D\Phi|=
 \prod_j|\det(I+\varepsilon_jA_j)|.
\]

The corresponding PF-PF weight must contain, on the same path,

\[
 \widetilde w_i\propto
 \frac{p(x_i^{\rm post}\mid x_{i}^{\rm prev})
       p(y\mid x_i^{\rm post})
       |\det D\Phi_i(x_i^{\rm pre})|}
      {q_0(x_i^{\rm pre}\mid x_i^{\rm prev},y)}
      w_i^{\rm prev}.
\]

This is exactly the source-faithful part of LEDH-PFPF. It restores the
proposal-to-target ratio **for the declared invertible proposal**. It does not
remove local-linearization, numerical integration, covariance-closure, or
finite-particle error.

### Corollary 4.1 (why the current post-flow reset is outside the identity)

Suppose the sampled proposal is `Y = \Phi(X)`, but the stored equal-weight
cloud is then replaced by `R(Y_{1:N})`, where `R` is the Contract-E/OT plus
GenUT dual-cap operation depending on all `N` rows. The determinant in
Proposition 4 is the determinant of `\Phi`, not a density correction for
`R`. Treating `R(Y_{1:N})` as if it were IID from `q_1` is therefore
wrong relative to the density claim unless a separate joint empirical-map
theorem and density evaluation are supplied.

#### Proof

Proposition 4 applies to a single-point map `\Phi:\mathbb R^d\to\mathbb R^d`.
The reset `R` takes an entire cloud in `(\mathbb R^d)^N`, includes
weighted OT and global restandardization, and generally changes all rows when
one row changes. Its Jacobian would be a joint (Nd\times Nd) object, and its
output law is an empirical transformation rather than the single-point
pushforward (q_1). The current code records no such joint determinant and
forms PF-PF weights before this reset. Thus the old determinant cannot certify
the post-reset density. `QED`

## 7. Replay: the measure must be fixed before rows are reused

### Definition 1 (known-density deterministic-mixture block)

At block (b), freeze proposal densities (r_{b1},\ldots,r_{bH}) before
drawing. Draw (n_{bh}) rows from (r_{bh}), let

\[
 N_b=\sum_h n_{bh},\qquad
 \alpha_{bh}=n_{bh}/N_b,\qquad
 m_b(\theta)=\sum_h\alpha_{bh}r_{bh}(\theta).
\]

For a fixed integrable function (f), define

\[
 \widehat\gamma_b(f)=
 \frac1{N_b}\sum_h\sum_{i=1}^{n_{bh}}
 \frac{\widetilde\pi(X_{bhi})}{m_b(X_{bhi})}f(X_{bhi}).
\]

The proposal definitions, allocation fractions, map determinants, and evaluated
\(\log m_b\) must remain attached to the block.

The support condition `m_b(\theta)>0` wherever
`\widetilde\pi(\theta)|f(\theta)|>0` is part of the block contract.

### Proposition 5 (conditional unbiasedness of a mass-carrying block)

If the proposals are selected from past history but frozen before the block is
drawn, and the support condition above holds, then

\[
 E[\widehat\gamma_b(f)\mid\text{past, frozen proposals}]
 =\int\widetilde\pi(\theta)f(\theta)\,d\theta.
\]

#### Proof

By independence within each stratum and linearity,

\[
\begin{aligned}
 E[\widehat\gamma_b(f)\mid\cdots]
 &=\sum_h\alpha_{bh}
   \int r_{bh}(\theta)\frac{\widetilde\pi(\theta)}{m_b(\theta)}f(\theta)d\theta\\
 &=\int\left[\sum_h\alpha_{bh}r_{bh}(\theta)\right]
   \frac{\widetilde\pi(\theta)}{m_b(\theta)}f(\theta)d\theta\\
 &=\int\widetilde\pi(\theta)f(\theta)d\theta.
\end{aligned}
\]

`QED`

For the forward NeuTra objective `F(\phi)=\mathrm{KL}(\pi\Vert q_\phi)`,
with `s_\phi(\theta)=\nabla_\phi\log q_\phi(\theta)`,

\[
 \nabla_\phi F(\phi)=-E_\pi[s_\phi(\theta)],
 \qquad
 -\widehat\gamma_b(s_\phi)
 \text{ targets } Z\nabla_\phi F(\phi).
\]

The unknown `Z` changes the scale but not the stationary set of the pure
forward term. It does matter when that term is combined with a reverse term, so
the relative scale must then be estimated or tuned explicitly.

### Proposition 6 (why normalized-only replay is not an unbiased repair)

Let a block produce unnormalized weights `U_i` and normalized weights
`W_i=U_i/\sum_jU_j`. The estimator `\sum_iW_if(X_i)` is a ratio estimator and
is not generally unbiased for `E_\pi[f]` at finite `N`. Replacing one finite
normalized block by another, or averaging infinitely many independent fixed-size
normalized blocks, does not remove the finite-(N) ratio bias in general.

#### Proof

At `N=1`, `W_1=1` regardless of `U_1`, so the estimator has expectation
(E_r[f(X)]), which differs from (E_\pi[f]) whenever (r\ne\pi) on (f). For
fixed (N), averaging independent blocks converges by the law of large numbers
to the expectation of this ratio estimator, not generally to the target
expectation. `QED`

This is the precise status of the existing six 100-particle normalized SMC
populations: they can support a finite empirical training objective and a
large-(N) consistency hypothesis, but not a finite-sample unbiasedness claim.

### Proposition 7 (AMIS-compatible experience replay)

Suppose samples are retained from proposal components (q_0,\ldots,q_t), with
sample counts (N_0,\ldots,N_t). Define the deterministic-mixture denominator

\[
 \bar q_t(\theta)=
 \frac{\sum_{\ell=0}^tN_\ell q_\ell(\theta)}
      {\sum_{\ell=0}^tN_\ell}.
\]

Reweight **every retained sample** (X_{\ell i}) by

\[
 U_{\ell i}^{(t)}=
 \frac{\widetilde\pi(X_{\ell i})}{\bar q_t(X_{\ell i})},
\]

not by its old (\widetilde\pi/q_\ell) ratio. For a proposal schedule frozen
before all draws, the resulting deterministic-mixture estimator is the
ordinary importance estimator for the stratified mixture and has the same
unnormalized target expectation as Proposition 5.

#### Proof

The pooled sample is stratified from the components with fractions
(N_\ell/\sum_jN_j). Applying Proposition 5 with those fractions gives the
identity. The denominator must include every component because the pooled
sampling law is the mixture, not the component that happened to generate a
particular row. `QED`

If proposals are adapted from previous rows, the proposal schedule is random
and dependent on the sample history. AMIS provides a route for this dependence,
but its convergence result requires tail and adaptation conditions; the simple
conditional-unbiasedness proof above no longer applies automatically. A bounded
replay buffer can therefore be valid only if it either:

- retains enough proposal metadata to recompute the full historical mixture;
- uses a separately proved finite-window target and labels that target; or
- emits an unnormalized block with a proof like Proposition 5.

Arbitrary row eviction with frozen old normalized weights is not one of these
routes.

## 8. Defensive support and tempering for modes

### Proposition 8 (defensive mixture bound)

Let (q_t) and (r_{\rm safe}) be densities, with (r_{\rm safe}) satisfying

\[
 r_{\rm safe}(\theta)>0\quad\text{whenever}\quad
 \widetilde\pi(\theta)>0,
\]

and define

\[
 m_t(\theta)=(1-\epsilon_t)q_t(\theta)
                 +\epsilon_t r_{\rm safe}(\theta),
 \qquad 0<\epsilon_{\min}\leq\epsilon_t\leq1.
\]

Then `m_t` covers the target support and, for any `f`,

\[
 \int\frac{\widetilde\pi(\theta)^2f(\theta)^2}{m_t(\theta)}d\theta
 \leq
 \frac1{\epsilon_{\min}}
 \int\frac{\widetilde\pi(\theta)^2f(\theta)^2}
                 {r_{\rm safe}(\theta)}d\theta.
\]

#### Proof

The mixture inequality `m_t\geq\epsilon_{\min}r_{\rm safe}` gives the support
statement and, after taking reciprocals, the displayed second-moment bound.
Finiteness of the right-hand integral is an additional assumption; support
coverage alone does not establish it. `QED`

This is the mathematically useful version of defensive replay. A broad
Student-t or transformed heavy-tailed component can protect sign/mode regions,
but its tail scale and mixture weight must be audited against the actual target
and score class. Hesterberg's stronger bounded-weight statement applies when
the target itself is a mixture component; that is not available merely because
the target density can be evaluated.

### Proposition 9 (tempered bridge and mutation)

Let `r_0` be a full-support reference density and define

\[
 \gamma_\beta(\theta)=
 r_0(\theta)^{1-\beta}\widetilde\pi(\theta)^\beta,
 \qquad 0=\beta_0<\cdots<\beta_L=1.
\]

At stage `\ell`, the incremental potential is

\[
 G_\ell(\theta)=
 \frac{\gamma_{\beta_{\ell+1}}(\theta)}
      {\gamma_{\beta_\ell}(\theta)}
 =\exp\left[(\beta_{\ell+1}-\beta_\ell)
        (\log\widetilde\pi(\theta)-\log r_0(\theta))\right].
\]

If resampling is applied to these potentials and the subsequent kernel
`K_\ell` leaves `\pi_{\beta_{\ell+1}}\propto\gamma_{\beta_{\ell+1}}`
invariant, the ideal Feynman--Kac recursion targets the stated bridge sequence.

#### Proof

Multiplication by `G_\ell` changes the unnormalized measure
`\gamma_{\beta_\ell}` into `\gamma_{\beta_{\ell+1}}`. An invariant mutation
kernel preserves the normalized measure after that update. Induction over
`\ell` gives the sequence. Finite-particle resampling and imperfect mixing
introduce the usual SMC approximation error, but do not change the formal target
sequence. `QED`

The bridge can make mode traversal more plausible because the reference and
target are connected through intermediate energy landscapes. It does not prove
that a finite run crosses a sign boundary. If a mode region (A) has reference
mass (p_A), even independent reference draws only give the elementary bound

\[
 \Pr(\text{at least one hit in }n\text{ draws})=1-(1-p_A)^n;
\]

when (p_A) is unknown or exponentially small, no finite deterministic mode
coverage guarantee follows. Mutation kernels and bridge diagnostics must report
actual cross-mode transitions and stage-wise mass, not merely ESS.

## 9. RQMC and OT: useful variance controls with explicit limits

Randomized QMC can reduce integration error when the inverse-CDF/transition map
and weight functions satisfy the regularity assumptions in Gerber--Chopin. It
does not change the target measure or supply an absent support component. Thus
RQMC should be applied to the safe base-noise and mutation lanes only after the
proposal density is correct.

The Contract-E/OT reset is likewise useful for constructing a well-conditioned
equal-weight representation. Reich's ensemble-transform result supports a
large-ensemble consistency interpretation, not a finite-(N) density identity.
The reset should therefore be recorded as a representation layer after the
proposal-corrected weighted cloud, not as a replacement for proposal-density
correction.

## 10. A conditional solution architecture

### Proposition 10 (conditional validity of the combined route)

Assume the following.

1. **Target and support.** The target is measurable with (0<Z<\infty), and a
   full-support (r_{\rm safe}) satisfies the second-moment condition in
   Proposition 8 for the forward score class.
2. **LEDH proposal.** Every claim-bearing LEDH map is a (C^1) diffeomorphism
   on the declared proposal support; every pseudo-time factor is nonsingular;
   the actual pre-flow density, post-flow transition, observation, covariance
   lifecycle, and matching determinant are evaluated together.
3. **Cap boundary.** The dual-cap/trust-region operation is either (a) used only
   to choose parameters of the actual proposal, with the actual density still
   evaluated, or (b) given an explicit bounded-support pushforward density and
   mixed with (r_{\rm safe}). It is not silently treated as a full-support
   normalizing flow.
4. **Replay.** Retained rows use the full deterministic-mixture denominator,
   or each block is an SMC-U/known-density mass-carrying block. Proposal choices
   are frozen before the rows to which the unbiasedness statement is applied.
5. **Modes.** The bridge in Proposition 9 uses valid incremental weights and
   mutation kernels invariant for each bridge target. The claimed mode result is
   limited to consistency/asymptotic statements unless mixing bounds are proved.
6. **NeuTra.** (T_\phi) is a genuine diffeomorphism in the admitted family,
   differentiation/integration interchange is justified, and optimization uses
   fresh base draws for the reverse term and valid mass-carrying blocks for the
   forward term.

Then the proposal/replay blocks target the stated unnormalized integrals, and
the ideal bridge recursion targets its declared sequence; a finite-particle
implementation has the usual SMC approximation error. Its forward NeuTra
gradient is (Z\nabla F(\phi)), and the fresh base reverse term is the gradient
of the stated pullback objective. Under the
usual additional stochastic-approximation and expressivity assumptions, a
population stationary point can correspond to a Gaussian pullback. No claim of
finite-run mode discovery or optimizer convergence follows without further
bounds.

#### Proof

Assumptions 1--3 make each proposal block a well-defined measure with a finite
second moment and invoke Proposition 4 for its LEDH part. Assumption 4 invokes
Propositions 5--7, so replay estimates the same unnormalized target integral
rather than a stale normalized empirical measure. Assumption 5 invokes
Proposition 9 for the ideal bridge recursion. Assumption 6 gives the standard
change-of-variables identities for (T_\phi) and permits the forward and
reverse derivative interchanges. Linearity combines the block estimates and
fresh-base estimates; the unknown (Z) is a positive scale on the pure forward
gradient. The remaining convergence statement is conditional on the explicitly
listed stochastic-approximation, mixing, and expressivity hypotheses. `QED`

### Proposed implementation sequence

The proposition gives a concrete, staged route rather than a default change.

1. **Proposal contract.** Keep the canonical LEDH-PFPF route and its analytical
   recursive score under the repository's current Contract-E policy. Add a
   per-proposal artifact containing the pre-flow density, post-flow transition,
   observation density, covariance state, pseudo-time matrices, determinant
   product, and support declaration.
2. **Dual-cap role.** Use the LM/trust-region and dual-cap operations to control
   local proposal construction and finite reset conditioning. Record cap-active
   fractions, derivative minima, feasibility margins, and post-reset moments.
   Do not call the reset output IID or use its cap derivative as a missing
   single-particle density determinant.
3. **Full-support safety.** Add a repository-owned broad component in the
   parameter chart (for example a heavy-tailed transformed mixture) with an
   explicit density. Retain it with \(\epsilon\geq\epsilon_{\min}\) and audit
   the score-class second moment. Include mode-seeded components only as
   additional mixture components, never as the sole support.
4. **Replay.** Replace frozen normalized row replay by either:
   - deterministic-mixture/AMIS reweighting over all retained proposal
     components, with proposal metadata and recomputable log densities; or
   - a fresh known-density/SMC-U block whose total mass is carried without
     per-block normalization.
   A fixed-capacity buffer may evict rows only if the resulting estimator and
   target are redefined/proved; arbitrary experience-replay eviction is not
   valid evidence.
5. **Bridge and mutation.** Run a separate tempered SMC lane from the safe
   reference to the target. Use LEDH as a stage proposal, then an invariant
   mutation kernel. Tune the bridge increments by a declared stage ESS rule,
   while treating ESS as a tuning diagnostic rather than a posterior proof.
6. **NeuTra training.** Keep the reverse objective on fresh Gaussian base draws.
   Feed the forward objective only from the valid mass-carrying lane. Calibrate
   the relative forward/reverse scale; do not infer it from normalized replay
   losses. Enforce the repository's batch-native GPU training policy.
7. **Admission tests.** Before HMC, require affine-map density identities,
   finite-support/mixture checks, unnormalized integral checks on a tractable
   target, independent two-mode coverage tests, held-out pullback moments, and
   the canonical sequential HMC gates. A short whitening screen is explanatory,
   not an admission criterion by itself.

## 11. What would falsify the route

The conditional route should be rejected or repaired if any of the following is
observed:

- a claim-bearing LEDH step has a singular or incorrectly signed determinant;
- the covariance used to build the local flow is not the covariance carried by
  the source-faithful UKF/EKF lifecycle;
- the post-cap proposal density is used without a support declaration and a
  matching density calculation;
- replay rows have no recoverable proposal identity or mixture denominator;
- the safe component fails the target/score second-moment check;
- bridge mutation kernels are not invariant for their declared stage target;
- independent audit particles show missing mode mass despite the claimed
  global-density result; or
- the learned transport is not a diffeomorphism, so its reported pullback is
  not the claimed density.

These are correctness vetoes, not merely poor-performance observations. Poor
whitening with all identities intact would instead be evidence that the current
proposal family, capacity, schedule, or optimizer is inadequate and would trigger
a scoped repair study.

## 12. Final answer to the question

**Can dual-cap/trust-region LEDH-PFPF-GenUT help?** Yes. It is a credible local
proposal-conditioning and variance-control component, and PF-PF's determinant
correction can make that proposal density-correctable when the map is genuinely
invertible.

**Does it solve the present density-unfaithfulness and replay problem alone?**
No. The implemented caps are bounded-support finite-cloud operations, GenUT
matches only selected moments, and the current normalized replay remains a
finite ratio estimator. None of those facts implies a target density or global
mode coverage.

**Is there a solution in principle?** Yes, conditionally: exact proposal-density
bookkeeping plus a full-support defensive component, deterministic-mixture or
unnormalized replay, and tempered invariant mutation. This is a new combined
algorithmic route whose assumptions must be tested; it is not established by the
existing artifacts.

**What should happen next?** Preserve dual-cap/trust-region as an opt-in
proposal candidate, implement the proposal/replay/bridge contracts in separate
reviewed phases, and block NeuTra/HMC promotion until the density and independent
mode-coverage gates pass. If the requirement is a guarantee from the dual-cap
reset alone, the mathematically correct answer is that no such guarantee exists.

## References and local provenance

1. D. Ebeigbe et al., *A Generalized Unscented Transformation for Probability
   Distributions*, arXiv:2104.01958, https://arxiv.org/abs/2104.01958, and
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8043458/. Local copy:
   `.localresources/papers/ebeigbe-et-al-genut-2104.01958.txt`.
2. Y. Li and M. Coates, *Particle Filtering with Invertible Particle Flow*,
   arXiv:1607.08799, https://arxiv.org/abs/1607.08799. Local copy:
   `.localresources/papers/ledh_replay_solution_20260824/li-coates-2017-particle-filtering-invertible-flow.txt`.
3. J.-M. Cornuet et al., *Adaptive Multiple Importance Sampling*,
   arXiv:0907.1254, https://arxiv.org/abs/0907.1254. Local copy:
   `.localresources/papers/ledh_replay_solution_20260824/cornuet-et-al-2009-amis.txt`.
4. T. Hesterberg, *Weighted Average Importance Sampling and Defensive Mixture
   Distributions*, Technometrics 37 (1995),
   https://doi.org/10.1080/00401706.1995.10484303. Local copy:
   `.localresources/papers/ledh_replay_solution_20260824/hesterberg-1995-defensive-mixture.txt`.
5. P. Fearnhead and B. M. Taylor, *An Adaptive Sequential Monte Carlo Sampler*,
   arXiv:1005.1193, https://arxiv.org/abs/1005.1193. Local copy:
   `.localresources/papers/ledh_replay_solution_20260824/fearnhead-taylor-2013-adaptive-smc.txt`.
6. M. Gerber and N. Chopin, *Sequential Quasi-Monte Carlo*, arXiv:1402.4039,
   https://arxiv.org/abs/1402.4039. Local copy:
   `.localresources/papers/ledh_replay_solution_20260824/gerber-chopin-2015-sqmc.txt`.
7. S. Reich, *A non-parametric ensemble transform method for Bayesian
   inference*, arXiv:1210.0375, https://arxiv.org/abs/1210.0375. Local copy:
   `.localresources/papers/ledh_replay_solution_20260824/reich-2013-ensemble-transform.txt`.
8. A. S. S. Neal, *Annealed Importance Sampling* (2001), and Del
   Moral--Doucet SMC-sampler sources are retained in
   `.localresources/papers/multimodal_hmc/` and are used for the bridge boundary,
   not as evidence for a finite-run mode guarantee.
