# Mitigating the joint TT regression failure

The first repair should change where the TT has to represent strong dependence: place each current coordinate beside its corresponding past coordinate, preferably inside a two-variable core. Retain the adjacent-state joint target. Replace the existing prefix-only marginal and suffix-only conditional implementations with contractions appropriate to this order. Combine this structural repair with rank adaptation, an adequately converged fitter and correctly weighted regression rows.

This is a proposed extension of the current regression implementation. The identities below establish how its marginals and conditionals can be computed. They do not establish filtering accuracy or an implemented runtime capability. The supporting experiment remains the frozen first-transition diagnosis in [the result note](observation-tt-first-transition-root-cause-20260914-result.md); no new numerical campaign was launched for this design.

## Why pair blocks address the observed restriction

Write u and v for reference coordinates of the current and previous states. The physical target remains

\[
\gamma_t(x,z)=\widehat\pi_{t-1}(z)f_\theta(x\mid z)g_\theta(y_t\mid x).
\]

Separate invertible charts supply x=G_x(u) and z=G_z(v). With product standard Gaussian reference density rho(u)rho(v), the target amplitude includes both Jacobians and the division by the reference density. These charts do not assert independence of the states.

Consider the limiting case where state components, their transitions, their incoming laws and observation likelihoods factor across i, and the charts preserve that factorization. Then the target amplitude is

\[
H(u,v)=\prod_{i=1}^d H_i(u_i,v_i).
\]

If pair i has separation rank s_i between u_i and v_i, the product has separation rank equal to the product of the s_i across the all-current/all-past partition, when the component decompositions have nonzero independent singular factors. This follows by taking tensor products of their singular decompositions. The rank can therefore grow exponentially with d even in this simple independent-components example.

A TT whose sites are the pairs (u_i,v_i) has rank one between sites for the same product. Local polynomial approximation still limits the accuracy inside each pair. The construction removes the unnecessary demand that all temporal dependence pass through one TT bond; it does not remove genuine dependence across state components.

The existing experiment supports this mechanism for the recorded d=4 transition: grouped rank-three fitting gave relative amplitude error .4608; an algebraically constructed grouped rank-three TT gave .2830; a scalar interleaved rank-three TT gave .0949. Merging adjacent scalar pairs can represent that interleaved TT exactly, so pair blocks contain that particular reference approximation. The error of a newly trained pair-block TT is not known.

A pair block is more flexible than scalar interleaving with rank three at every bond: it places no separate rank-three constraint inside a current/past pair. For degree p per scalar coordinate, a block has (p+1)^2 local basis functions. Its coefficient storage is approximately d(p+1)^2 r^2 at uniform inter-block rank r, versus 2d(p+1)r^2 for scalar cores. At p=3 this is about twice the coefficient storage at the same bond rank, before endpoint corrections. It is not a full 2d-dimensional tensor grid.

Physical-coordinate pairing is a hypothesis motivated by the nearly diagonal transition in this case. For dense interactions, pair according to model structure or consider within-state rotations before pairing. For a Gaussian joint guide, singular vectors of its whitened cross covariance identify correlated mode pairs. Such separate rotations preserve the all-current/all-past separation spectrum, but make a different ordering useful. Non-Gaussian departures and changed polynomial resolution still require measurement.

## Exact marginalization of a squared pair-block TT

Let phi_a be normalized Hermite polynomials under the scalar standard Gaussian density phi. Use pair cores

\[
G_i(u_i,v_i)=\sum_{a,b=0}^{p}C_i^{ab}\phi_a(u_i)\phi_b(v_i),
\qquad h(u,v)=G_1(u_1,v_1)\cdots G_d(u_d,v_d),
\]

where each C_i^{ab} is an r_{i-1} by r_i matrix and r_0=r_d=1. Preserve the explicitly defined defensive joint density

\[
q(u,v)=\frac{h(u,v)^2+\tau}{Z+\tau}\rho(u)\rho(v),
\quad Z=\int h^2\rho(u)\rho(v)\,du\,dv,\quad \tau>0.
\]

For fixed u_i, put B_{ib}(u_i)=sum_a C_i^{ab} phi_a(u_i), and define the linear matrix map

\[
\mathcal E_i^{u_i}(R)=\sum_b B_{ib}(u_i)R B_{ib}(u_i)^\top.
\]

**Proposition.** The exact marginal of this fitted joint is

\[
q_u(u)=\frac{\rho(u)}{Z+\tau}
\left[\mathcal E_1^{u_1}\circ\cdots\circ\mathcal E_d^{u_d}(1)+\tau\right].
\]

**Proof.** Integrating the last pair's v coordinate in G_d G_d^T eliminates cross terms with b unequal to b', because integral phi_b phi_b' phi equals the Kronecker delta. It produces E_d^{u_d}(1). Repeat with G_{d-1} and the resulting right matrix, working backwards. Fubini is applicable since the squared polynomial times the Gaussian density is integrable. The constant defensive term integrates to tau. Each map sends positive semidefinite matrices to positive semidefinite matrices, proving nonnegativity as well as the formula. Dividing by Z+tau normalizes it.

This retained density should be stored as this matrix contraction. It need not be approximated again by a square-root TT. Its ordinary density-TT bond dimensions are at most r_i^2, while direct evaluation can keep r_i by r_i matrix environments. Computing the B matrices and applying the maps costs approximately O(d[(p+1)^2 r^2+(p+1)r^3]) per density evaluation with dense ordinary matrix products. Constants, batching and actual ranks matter in practice.

The next regression step needs the retained physical log density. Applying G_x^{-1} and subtracting log|det DG_x| supplies it. That step fits a new joint TT, so the retained representation need only store the latest pair cores; this construction does not require stacking an ever-growing history of doubled ranks.

The identity is for the fitted q, including its declared defensive term. Exact integration of q does not prove that q equals the model posterior.

## Conditioning preserves a scalar TT over the current variables

Fix v. Define particle-dependent scalar cores

\[
D_i^a(v_i)=\sum_b C_i^{ab}\phi_b(v_i),\qquad
h_v(u)=\prod_i\left[\sum_a D_i^a(v_i)\phi_a(u_i)\right].
\]

Then h_v is a scalar-coordinate TT in u, with the same inter-block ranks, and

\[
q(u\mid v)=\frac{h_v(u)^2+\tau}{Z(v)+\tau}\rho(u),
\qquad Z(v)=\int h_v(u)^2\rho(u)\,du.
\]

This follows directly by cancellation of rho(v)/(Z+tau) between the joint and its v marginal. Complete and incomplete Hermite Gram contractions give its one-dimensional conditional CDFs, as in the present squared-polynomial sampler. The Gaussian mixture weight is tau/(Z(v)+tau), which must be evaluated for the actual fixed v.

The required engineering change is material: each particle has a different set of D cores, rather than merely a different final suffix matrix. The existing shared sampler accepts unbatched cores. It must be extended to batch-native cores, with complete/incomplete Gram contractions and both sampling directions covered by tests. Passing batched cores to the present API is not supported, and row-mapping a reduced sampler is not the proposed solution.

A small nonseparable example makes both operations explicit. Take

\[
h(u,v)=(1+u_1v_1)(1+u_2v_2)+c.
\]

It has pair cores [1+u_1v_1,c] and [1+u_2v_2,1]^T. Direct Gaussian integration gives

\[
\int h(u,v)^2\rho(v)\,dv
=(1+u_1^2)(1+u_2^2)+2c+c^2.
\]

For the first current coordinate conditional on v, integrating over u_2 gives the polynomial A u_1^2+B u_1+C, where

\[
A=v_1^2(1+v_2^2),\quad B=2v_1(1+c+v_2^2),\quad C=(1+c)^2+v_2^2.
\]

With the defensive term, its normalized CDF is

\[
F(a\mid v)=\Phi(a)-\frac{(Aa+B)\phi(a)}{A+C+\tau}.
\]

Differentiation using Phi'=phi and phi'=-a phi gives (Aa^2+Ba+C+tau)phi(a)/(A+C+tau). The polynomial arose as an integral of a square, so tau>0 makes the density strictly positive. The CDF limits are zero and one. This verifies that pairing is compatible with normalized, invertible scalar conditional CDFs.

MathDevMCP verified two bounded algebraic identities used in this example. Its initial direct special-function derivative call was inconclusive because its expression encoder failed; after supplying the derivative rules explicitly, it verified the remaining algebra. The record is `artifacts/observation-tt-mitigation-20260914/mathdev-scoped-algebra.json`. This is not an audit pass for the general sampler or the full algorithm.

## Rank and fitting accuracy require separate repairs

Pairing should be combined with a rank profile and bounded enrichment, rather than another universal fixed rank. The existing order-11 projection already gives these optimistic errors for the grouped middle cut:

| Middle block rank | Best degree-three block-rank error |
|---|---:|
| 3 | .2804 |
| 6 | .1506 |
| 12 | .1037 |

These are best block-matrix approximation errors on the frozen finite quadrature, not errors attained by an entire trained TT with all its other rank constraints. They explain why increasing grouped rank can help, while also showing why repairing the ordering is attractive. All these observations are diagnostic and must not become tuning data for a promoted candidate.

Zhao and Cui explicitly describe residual-based rank adaptation in Section 2.2, rather than a universal rank-three restriction. Their author implementation uses local tolerances, a maximum rank and residual enrichment in `@TTFun/build_basis_amen.m:34`, `:118` and `:127`; `@TTFun/cross.m:33` initializes residual data. This supports rank adaptation as an established ingredient. It does not make the proposed L1 regression method identical to their cross-interpolation method. [Zhao and Cui, JMLR 2024](https://jmlr.org/papers/v25/23-0743.html).

The saved .4608 error also exceeds the feasible .2830 grouped approximation by a large margin. Increasing capacity without fixing this gap can simply produce a larger poorly fitted TT. Record per-core conditioning, nonzero effective ranks, relative objective decrease and a proximal/KKT residual. Increase solver work under a declared budget and compare initializations. Objective nonincrease after 128 iterations is not convergence.

Gauge normalization and L1 require a defined objective. Transformations G_i -> G_i M and G_{i+1} -> M^{-1}G_{i+1} leave h unchanged, while generally changing coefficient L1 penalties. Thus QR normalization, enrichment and block merging cannot silently retain the interpretation of an old L1 value. Define the coefficient gauge and penalty convention, retune L1 on fresh calibration/validation partitions, and preserve the unpenalized integrated residual as a separate diagnostic. A zero-L1 arm is a comparator; it does not replace the repository's L1-tuning policy.

## Use the joint guide for rows without changing the integration measure

Let rho(w) be the product Gaussian reference density, H(w) the properly pulled-back target amplitude, and s(w) an evaluable sampling density. For frozen h,

\[
\int (h-H)^2\rho\,dw
=\mathbb E_{w\sim s}\left[\frac{\rho(w)}{s(w)}(h(w)-H(w))^2\right].
\]

The equality follows by cancelling s in the integral. It preserves the population data-fit objective; it does not imply that fitting the sampled rows reaches a global optimum. Multiplying design rows and responses by sqrt(rho/s) gives the weighted least-squares core problem. This is the measure relation w dmu=d rho in Cohen and Migliorati, Section 2. Their near-optimal sampling theory concerns a specified linear approximation space and does not by itself prove nonlinear adaptive TT convergence. [Cohen and Migliorati, arXiv:1608.00512v1](https://arxiv.org/abs/1608.00512v1).

A useful candidate is s=epsilon rho+(1-epsilon)s_joint, with 0<epsilon<1, where s_joint is an evaluable joint SGQF-based Gaussian guide. Then rho/s <= 1/epsilon. This bounds the sampling weights; it does not bound the residuals or guarantee adequate effective sample size. Epsilon is a calibration choice, not an inherited constant.

The joint guide need not require a new 2d-dimensional quadrature. Given Gaussian previous moments (m,P), linear transition A with covariance Q, S=APA^T+Q and K=PA^T S^{-1}, use the SGQF-updated current moments (m_x^+,P_x^+) to construct

\[
m_z^+=m+K(m_x^+-Am),\quad
P_z^+=P-KSK^T+KP_x^+K^T,\quad
P_{xz}^+=P_x^+K^T.
\]

These formulas follow from the Gaussian predictive conditional z|x and total covariance. They are exact for that Gaussian predictive approximation if its current posterior moments are exact, and approximate when supplied by SGQF. The previous target density remains the retained TT. The joint Gaussian is only a row proposal; it need not become a mixed physical coordinate map.

The implementation must weight the target scale too. If logtarget=log(H^2), use a common frozen scale across comparison arms or the training-only estimate

\[
\mathrm{logscale}=\log\sum_j\exp\{\mathrm{logtarget}_j+\log\rho_j-\log s_j\}-\log N.
\]

For the L1 objective, divide the weighted residual sum by N. Dividing by the random sum of weights while leaving lambda unchanged alters the relative regularization strength. Validation must estimate the same reference-measure objective. Freezing the proposal before each independent sample batch preserves the conditional importance identity; recycling adaptively selected rows needs its own sampling argument.

## Implementation sequence and acceptance

| Change | Concrete consumer or implementation | Required check |
|---|---|---|
| Pair blocks and per-bond ranks | `observation_guided_tt_tf.py::build_tt_path` and fitter shapes | Target density, both Jacobians and physical coordinate meaning agree on common points |
| Retained positive density | Replace the prefix-only logic in `TTStep.retained` with pair-wise integrated matrix maps | Density and normalized moments agree with independent tiny quadrature; serialized reload preserves the actual callable/settings |
| Conditional draws | Extend shared Hermite Gram/KR kernels to batched conditional cores; connect `sample_tt_step` | CDF derivative, normalization, inverse residual, physical log density and sampling distribution checks; existing unbatched lane parity |
| Regression row sampling | Weighted Gram/rhs, target scale and validation in `fit_amplitude` | Weighted/unweighted population identities, independent Gram conditioning and objective checks |
| Rank/solver adaptation | Budgeted enrichment, convergence diagnostics and L1 selection | Fresh validation nominates; untouched audit can veto; increasing rank is not itself success |
| Downstream filter | Existing SMC weight calculation consumes the actual repaired proposal density | Reference likelihood/state agreement, weight diagnostics and conditional heuristic comparisons under the existing evidence contract |

The current local call chain was inspected at `observation_guided_tt_tf.py:176`, `:219`, `:273`, `:286`, `:333` and `:361`; the shared kernel restriction is visible at `c2_gaussian_hermite_proposal_tf.py:129`, `:143` and `:231`. The author source's contiguous marginal and conditional operations were inspected at `@TTSIRT/marginalise.m:25` and `@TTSIRT/eval_cirt_reference.m:103`. The proposed new call chain is **not implemented or executably checked**. It must generalize shared kernels rather than create a reduced sampling fork.

The next discriminating experiment should use fresh calibration observations to compare grouped rank adaptation, scalar pairing and pair blocks at declared evaluation/compute budgets. Separate row-design and solver comparisons from the representation comparison. Include the original grouped arm, the same method after each repair, and the existing filtering heuristics in low/high volatility and unusually large-observation situations. A failed TT candidate triggers the next planned repair unless numerical validity, target identity, artifact validity or the total campaign budget is lost. No posterior/default promotion can be inferred from a small one-step residual alone.

The dominant alternative explanation is that cross-component or non-Gaussian dependence may defeat a cheap paired representation outside the diagnosed case. Measured per-bond spectra, independent approximation error and downstream reference agreement distinguish that outcome from insufficient solver work. If pairing and justified rank growth remain too costly, a more expressive transport or another proposal family becomes a separate research choice; an arbitrary joint whitening map is not the first required fix.

No new filtering/training run was launched for this design. The prior numerical total remains approximately 124.225 of 2400 seconds, with one of three full filtering launches used. The primary weighted-least-squares preprint, text and version/checksum record are stored under `.localresources/papers/cohen-migliorati-2016-optimal-weighted-least-squares-arxiv-v1.*`.
