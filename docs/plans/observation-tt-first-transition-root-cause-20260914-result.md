# The joint target is present; its TT ordering and fitted accuracy are inadequate

The code fits the adjacent states jointly. The earlier suggestion that separate coordinate charts meant the filter was failing to treat the states jointly was misleading. The joint transition density is present, with the correct likelihood, incoming retained density, reference-density factors and Jacobians. The central structural restriction is a rank-three amplitude across the cut between all four current-state coordinates and all four previous-state coordinates. The saved fit also falls far short of a feasible approximation with that same ordering, rank and degree.

The frozen first-transition diagnostic distinguishes this from inherited error. On the finest tested quadrature, replacing the incoming TT with the directly integrated initial posterior changes the normalized joint amplitude by 0.0321, while the saved fit has relative amplitude error 0.4608 against its actual target and 0.4625 against the replacement target. Accumulated historical error therefore does not explain the large first-transition failure. This is a local finding at t=1, not a bound on error accumulation later.

## What is joint in the mathematical filter and in the code?

Write z=x_{t-1}, x=x_t and Y_t=y_{0:t}. Given the previous filtering density, the unnormalized adjacent-state target is

\[
 \gamma_t(x,z)=\pi_{t-1}(z)f_\theta(x\mid z)g_\theta(y_t\mid x).
\]

Normalizing this density gives p(x_t,x_{t-1}|Y_t). Integrating over z gives the current filtering density. Conditioning on a particular previous state gives

\[
 p(x\mid z,Y_t)=\frac{f_\theta(x\mid z)g_\theta(y_t\mid x)}
 {\int f_\theta(s\mid z)g_\theta(y_t\mid s)\,ds}.
\]

The same joint density also updates the past-state marginal: it is proportional to pi_{t-1}(z) times the denominator above. Thus y_t informs both states in the joint even if the chart for z was constructed using observations only through t-1.

`build_tt_path` (`bayesfilter/highdim/observation_guided_tt_tf.py:286`) implements this joint with the previous retained TT approximation replacing pi_{t-1}. Its closure is evaluated immediately by `fit_amplitude` before the retained density advances; there is no stale or future-density capture in this call chain. It uses the prior and first observation at t=0, and transition plus observation at t=1 onward. The conditioning chart is the previous SGQF posterior chart, but the previous *density factor* is the retained TT; these roles are distinct.

The charts are

\[
 x=m_x+L_xu,\qquad z=m_z+L_zv.
\]

For standard Gaussian reference densities r(u), r(v), the amplitude fitted in `fit_amplitude` is, up to its frozen scalar normalization,

\[
 H(u,v)=\sqrt{\frac{\widehat\pi_{t-1}(m_z+L_zv)
 f_\theta(m_x+L_xu\mid m_z+L_zv)g_\theta(y_t\mid m_x+L_xu)
 |\det L_x|\,|\det L_z|}{r(u)r(v)}}.
\]

Independent Gaussian regression rows do not replace this joint density by a product density. They specify the integration/design measure for approximating H. The transition still couples u and v in every target evaluation.

`TTStep.retained` (line 273) integrates the past-state suffix using exact Hermite Gram contractions. `compiled_conditional_sampler` (line 333) evaluates that suffix at the actual previous particle and samples the current prefix. `sample_tt_step` (line 361) applies the physical coordinate transform and determinant. `filter_kernel` in the master (line 77) uses original-model importance factors. These are the two distinct consumers of the same fitted joint. Algebraic conditional/marginalization structure was checked against Zhao–Cui Section 3.1 and Algorithms 2–3, and the author's `@TTSIRT/eval_cirt_reference.m:103` and `marginalise.m:25`; the fixed Hermite/SGQF/L1 implementation remains a local extension.

## Why the grouped rank-three representation is restrictive

The axis order in `build_tt_path` is

\[
 (u_1,u_2,u_3,u_4,v_1,v_2,v_3,v_4).
\]

`compiled_fitter` (line 176) sets every interior TT rank to three. Contracting the first four and last four cores separately therefore gives

\[
 h(u,v)=\sum_{a=1}^{3}\ell_a(u)b_a(v).
\]

This is a rank-three restriction on the *amplitude* across the entire current/past partition. It allows dependence and is not an independence approximation. Squaring the amplitude adds cross-products, but does not remove the restriction on the amplitude-regression objective actually optimized.

For the executed linear Gaussian transition with noise covariance sigma^2 I, expansion of the log transition gives

\[
 -\frac{\|m_x-Am_z+L_xu-AL_zv\|^2}{2\sigma^2}.
\]

All terms depending on only one block can be collected into functions a(u), b(v). Taking the square root leaves the exact form

\[
 H(u,v)=a(u)b(v)\exp\left\{\frac{u^T L_x^T A L_zv}{2\sigma^2}\right\}.
\]

The transition matrix is nearly diagonal with diagonal entries approximately 0.57–0.62. It creates four substantial pairwise dependence directions. Expanding the exponential generates multiple products of these directions. Placing every current coordinate before every previous coordinate forces all of them across the same rank-three boundary. At t=0, that two-time boundary does not exist. If A=0, the cross term vanishes and the joint amplitude separates across the time partition; this limiting case illustrates the mechanism.

The directly integrated joint has four nonzero canonical correlations, approximately 0.557, 0.411, 0.373 and 0.318. These moments describe the dependence but do not themselves prove a rank bound. The singular-value diagnostic below supplies that bound for the finite quadrature problem.

## Frozen-target decomposition and executable evidence

Target A uses the saved t0 retained mixture. Target B replaces only that density by

\[
 \pi_0^{\rm ref}(z)=p_0(z)g(y_0\mid z)/Z_0,
\]

where Z_0 is independently integrated. At t=1 this is a reference to the actual previous posterior, up to the recorded quadrature error. Both targets use the same physical model, observations, charts and amplitude scale. No refitting or particle resampling changes those quantities.

At product Gaussian nodes u_i,v_j with positive weights w_i,w_j, form

\[
 M_{ij}=\sqrt{w_iw_j}\,H(u_i,v_j).
\]

The weighted degree-three block basis matrices U,V have orthonormal columns; the program checks their Gram matrices. C=U^T M V is the unrestricted tensor-product polynomial projection. For any degree-three approximation represented by coefficient matrix D, orthogonality gives

\[
 \|M-UDV^T\|_F^2
 =\underbrace{\|M\|_F^2-\|C\|_F^2}_{\text{polynomial projection error}}
 +\|C-D\|_F^2.
\]

Let s_j be the singular values of C in descending order. A rank-r D can capture at most the first r singular components, and their truncated SVD attains that bound. Consequently,

\[
 E_{\mathrm{poly},r}^2=
 \frac{\|M\|_F^2-\sum_{j=1}^{r}s_j^2}{\|M\|_F^2}.
\]

This is a lower bound for a grouped degree-three TT with the same middle rank, because additional internal TT ranks may constrain it further. Even an unrestricted-degree block-rank-r approximation has squared error at least

\[
 \frac{\|C\|_F^2-\sum_{j=1}^{r}s_j^2}{\|M\|_F^2},
\]

since projecting such an approximation into these polynomial subspaces cannot increase its rank or increase the norm of its error. The code also checks the saved fitted mass against the analytic Hermite contraction and verifies the error decomposition directly. These statements are exact for each specified finite weighted matrix. Transfer to the continuous objective is limited by quadrature accuracy.

On the order-11 rule, the results for Target A are:

| Approximation or diagnostic | Relative amplitude RMS |
|---|---:|
| Saved grouped rank-three fit | 0.4608 |
| Best degree-three approximation with block rank three | 0.2804 |
| Feasible grouped rank-three TT from algebraic TT-SVD | 0.2830 |
| Feasible paired-order rank-three TT from the same projection | 0.0949 |
| Unrestricted degree-three tensor-product projection | 0.0802 |
| Projected lower bound for any grouped block-rank-three approximation | 0.2687 |

The paired order is (u1,v1,u2,v2,u3,v3,u4,v4). Grouped and paired TT-SVD use the same projected target, degree, rank cap and algebraic construction. Their difference directly diagnoses an axis-order restriction on this target. The paired construction is an expensive reference calculation, not a fitted filtering implementation or a demonstration of general superiority.

The saved squared error decomposes as 0.21236 = 0.00643 (polynomial truncation) + 0.07221 (necessary block-rank-three error inside that polynomial space) + 0.13372 (remaining error). The feasible grouped TT-SVD nearly attains the block-rank bound, so the remaining error is largely avoidable within the same rank/degree/order space under the unpenalized integrated objective. Its allocation among finite regression rows, optimization, initialization and L1 regularization is **not isolated** by this experiment. Calling that entire gap an optimizer bug would be unsupported.

The source explains why this remaining gap is possible: it uses 1024 rows, four sweeps and 128 proximal iterations per core, checks objective nonincrease, and does not certify stationarity or convergence. The selected L1 penalty at t=1 is 0.001. Training error is 0.4093; the original Gaussian audit-row error is 0.4745. The much larger quadrature audit yields 0.4608, so the failure persists beyond the original finite audit sample. More rows or more solver work can address part of the gap, but cannot remove the grouped middle-rank lower bound while that representation is fixed.

The scaled saved polynomial normalizer is log Z_H=-4.99029 versus the frozen target's log integral -4.81037, about 16.5% less mass. With the directly integrated incoming posterior, log Z_1=-4.80819. The inherited-law effect on this normalizer is about 0.00217 nats, whereas the local fitted-mass discrepancy is about -0.180 nats. These compare pre-defensive fitted polynomial mass with the stated local target. They do not reinterpret the added Gaussian mixture mass as model evidence, and are not cumulative particle-likelihood estimates.

Orders 5,7,9,11 give saved-fit errors 0.4605–0.4620, best grouped degree-three rank-three errors 0.2799–0.2809, and paired TT-SVD errors 0.0930–0.0956. All point to the same large structural and fitting gaps. Some smaller quantities, including polynomial residual and inherited distance, still move by more than 0.001 at the last refinement. The diagnostic therefore does not claim that every reported digit has converged or that the continuous-space bounds are certified.

## Why separate whitening is not the missing mathematical ingredient

For an invertible transform G of one block, the map

\[
 (\mathcal U_G a)(u)=a(G(u))\sqrt{|\det DG(u)|/r(u)}
\]

preserves the L2 norm by change of variables. The corresponding operation on both blocks is a product of two unitary transformations. It preserves the separation singular values of the normalized square-root joint density across the physical current/past partition. Thus separate invertible charts can improve polynomial resolution and sampling coverage, but cannot remove that intrinsic block-rank restriction. The claim concerns the properly transformed density with its Jacobians and reference measures, not an unweighted arbitrary function fit.

A map mixing the two state blocks can change the rank needed in its reference coordinates. However, consider the orientation that preserves convenient conditioning:

\[
 z=m_z+L_zv,\qquad x=m_x+Bv+L_xu.
\]

For fixed z, v is known and sampling u|v remains convenient. The current physical marginal becomes

\[
 p_x(x)=\frac1{|\det L_x|}\int q_{u,v}
 \left(L_x^{-1}(x-m_x-Bv),v\right)\,dv.
\]

This is not the integral over v at fixed u that `TTStep.retained` computes. Using the current retained implementation unchanged would return the wrong physical marginal.

Conversely, for

\[
 x=m_x+L_xu,\qquad z=m_z+Bu+L_zv,
\]

the current marginal remains a prefix marginal, but conditioning on physical z makes v=L_z^{-1}(z-m_z-Bu) vary with u. Holding the suffix v fixed in the current conditional sampler would be wrong. An arbitrary joint Cholesky factor is therefore not a safe replacement for the two current charts without deriving and implementing both consumers again.

A joint SGQF moment guide is nevertheless mathematically available. With Gaussian previous approximation N(m,P), predicted current covariance S=APA^T+Q and K=PA^T S^{-1}, let the likelihood-weighted SGQF current moments be m_x^+,P_x^+. Gaussian conditioning and total covariance give

\[
 m_z^+=m+K(m_x^+-Am),\qquad
 P_z^+=P-KSK^T+KP_x^+K^T,\qquad
 \operatorname{Cov}(x,z\mid y_t)=P_x^+K^T.
\]

The derivation uses z|x ~ N(m+K(x-Am),P-KSK^T); observing y_t changes the distribution of x but not z|x. These identities are exact for the Gaussian predictive approximation when the x moments are exact, and quadrature-approximate with SGQF moments. They need no additional 2d-dimensional quadrature for this linear Gaussian transition. They describe a possible guide, not a newly implemented coupled map or a repair already proved necessary.

## Decision, limitations and next work

| Decision | Primary finding | Veto/validity status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| The joint target is implemented correctly in the inspected call chain | Transition, previous density, likelihood and change of measure all present | Direct density, mass and projection checks passed | No full proof-system certification | Preserve the target and importance correction | Accurate fitted TT |
| Grouped rank three is an inadequate representation for a small-error amplitude fit on this transition | About 28% degree-three rank floor; paired rank-three reference about 9.5% | Finite-matrix bound checked; continuous precision incomplete | Other models, dimensions and observations | Compare larger time-boundary rank or a correctly supported paired ordering on fresh calibration | General superiority of pairing |
| Current fitting leaves substantial attainable accuracy unused | Saved 46.1% versus feasible grouped 28.3% | Existing particle screens do not remove this regression failure | Optimization versus L1/row estimation | Same-objective convergence and regression-row diagnostics | All excess error is an optimizer bug |
| Coupled whitening is optional and requires both consumers to change | Conditioning and marginal identities constrain map orientation | Reusing either incompatible consumer would be a mathematical error | Cost and approximation quality of a revised marginal representation | Prefer the smallest rank/ordering repair first; derive both consumers before a map change | Joint filtering is absent |

| Inference status | Assessment |
|---|---|
| Hard validity screen | Density ratio, polynomial Gram/mass, projection identity and finite-value checks passed |
| Statistically supported ranking | None; this is a deterministic frozen-target diagnostic |
| Descriptive differences | Feasible approximation errors on the same quadrature target; quadrature sensitivity reported |
| Default readiness | False; no runtime candidate was changed or promoted |
| Next evidence needed | Fresh calibration, verified ordering/rank consumers, regression convergence and independent downstream evaluation |

Post-run red team: a quadrature artifact is the strongest numerical alternative explanation, but the large differences persist over four orders and both incoming laws. Remaining quadrature variation prevents a sharp population certificate. The paired TT-SVD accesses a large tensor-product integration and cannot be presented as a scalable fitting algorithm. The diagnosis could be overturned for later steps or other models; it is strongest for the recorded first transition.

## Reproduction

Program: `docs/benchmarks/diagnose_observation_tt_first_transition.py`. Plan: `docs/plans/observation-tt-first-transition-root-cause-20260914.md`. Artifacts: `docs/benchmarks/artifacts/observation_tt_first_transition_20260914/run-02/`, including per-order JSON, result and manifest. Run-01 is preserved and used 13.104 seconds; a localized refinement-condition repair enabled the declared order-11 check in run-02, which used 11.547 seconds. Total diagnostic time was 24.651 seconds, within its 600-second budget and the existing campaign budget. GPU: RTX5080, verified memory growth, float64, TensorFlow 2.20.0-dev0+selfbuilt, XLA target/projection kernels; diagnostic decompositions are explicit non-XLA reference operations. Run-02 allocator peak: 180,209,408 bytes. No package, production code, model default or HMC consumer changed.

The exact launched command and source/data hashes are in each `run_manifest.json`. The source campaign used one full launch of 99.574 seconds; this diagnosis brings the recorded numerical total to approximately 124.225 of 2400 seconds. Unused budget does not authorize selecting a new candidate on these reported holdout observations.
