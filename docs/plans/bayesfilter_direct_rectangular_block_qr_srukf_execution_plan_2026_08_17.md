# Direct Rectangular and Block-QR SR-UKF Execution Plan

Date: 2026-08-17  
Status: `EXECUTION_AUTHORIZED_LOCAL_RESEARCH`  
Owner: BayesFilter  
Scope: direct-factor sigma-point filtering, including fixed-rank and singular-support diagnostics

## 1. Objective

Extend the repository-default direct-factor SR-UKF so that it has a mathematically explicit and testable robustness contract for:

1. full-rank but ill-conditioned covariance and innovation stacks;
2. structurally positive-semidefinite state/process covariances represented by rectangular factors;
3. singular innovation covariance with a support-aware likelihood; and
4. analytical derivatives on a fixed rank, pivot, sign, and support branch.

The implementation must not form `P = A A'` and then call an eigendecomposition, SVD, or Cholesky factorization on `P` during the admitted recursion. The existing principal-root/eigen routes remain historical comparison routes. Direct-stack SVD is allowed only in a separately labeled rank/conditioning diagnostic or value-only support route.

This is a serious local implementation campaign with bounded scope. It changes only BayesFilter source, focused tests, and the two plan-owned documentation files. Unrelated dirty worktree changes are preserved. No HMC, NeuTra training, package installation, or external publication is in scope.

## 2. Literature and local baseline

The design follows the established square-root pattern in Kaminski--Bryson--Schmidt, Bierman, van der Merwe--Wan, and Arasaratnam--Haykin: factor residual/loadings directly with orthogonal triangularization, use triangular solves, and avoid covariance square-rooting and subtraction. The local survey is `docs/plans/bayesfilter_square_root_sigma_point_filter_literature_survey_2026_08_17.tex`.

The current baseline is:

- `bayesfilter/nonlinear/factor_srukf_tf.py`: square lower factors, direct prediction/innovation QR, posterior sequential lower Cholesky downdates;
- `bayesfilter/linear/stack_qr_tf.py`: direct residual-stack QR and first derivatives;
- `bayesfilter/linear/lower_rank_downdate_tf.py`: signed covariance downdate primitive;
- `tests/test_stack_qr_tf.py`, `tests/test_lower_rank_downdate_tf.py`, and `tests/test_factor_srukf_tf.py`: focused baseline tests.

The baseline is valid only on a full-rank positive-pivot branch. In particular, a positive diagonal assertion is not a singular-covariance policy, and a downdate margin is not a support-aware likelihood. The current temporal model contract still requires square factors; rectangular support handling is therefore exposed as an independent value-only primitive until a separate rectangular model contract is promoted.

## 3. Mathematical contract

### 3.1 State and point representation

Carry a mean `m` and a rectangular loading factor

\[
  G_x\in\mathbb R^{n_x\times r_x},\qquad P_x=G_xG_x^T,
\]
where `r_x` may equal or be smaller than `n_x`. For fixed latent offsets `xi_j` in `R^{r_x}`,

\[
  x_j=m+G_x\xi_j.
\]

For additive process noise with `G_q in R^{n_x x r_q}`, form the augmented loading

\[
  G_a=\begin{bmatrix}G_x&0\\0&G_q\end{bmatrix},
  \qquad \xi_j\in\mathbb R^{r_x+r_q},
\]

without forming `G_a G_a'`. The DZ5 point rule uses `2(r_x+r_q)+1` points with

\[
 \xi_0=0,\quad \xi_i=\sqrt d e_i,\quad \xi_{d+i}=-\sqrt d e_i,
\]
\[
 w_0^{(m)}=0,\quad w_0^{(c)}=2,\quad
 w_j^{(m)}=w_j^{(c)}=1/(2d),\quad j>0.
\]

All covariance weights are nonnegative. A general negative-weight UKF is a separate signed-update contract and is not silently covered by this route.

### 3.2 Direct prediction stack

After propagating points `x_j^- = f(x_j,q_j)`, define

\[
 \bar x^- = \sum_j w_j^{(m)}x_j^-,\qquad
 A_x=[\sqrt{w_j^{(c)}}(x_j^- -\bar x^-)]_j.
\]

If process noise was not augmented into the point cloud, append `G_q` as loading columns exactly once. The represented prediction covariance is `A_x A_x'`, but this Gram matrix is a test-boundary identity, not a runtime object.

For full rank, factor the transpose directly:

\[
 A_x^T=Q_xR_x,\quad Q_x^TQ_x=I,\quad R_{x,ii}>0,
 \qquad G_x^-=R_x^T.
\]

Then `G_x^- G_x^{-T}=A_x A_x^T` without covariance formation.

### 3.3 Innovation stack

For observation points `y_j=h(x_j^-)`, define

\[
 \bar y=\sum_jw_j^{(m)}y_j,\qquad
 A_y=[\sqrt{w_j^{(c)}}(y_j-\bar y)]_j.
\]

With observation-noise loading `G_r`, use `[A_y, G_r]`. Parameter derivatives of `G_r` are columns of the stack derivative. Do not differentiate a reconstructed covariance and refactor it.

### 3.4 Full-rank block conditional QR

For the measurement update, use one joint stack with observation rows first:

\[
 A=\left[
 \begin{array}{c|c}
 A_y&G_r\\
 A_x&0
 \end{array}\right]\in\mathbb R^{(n_y+n_x)\times k},
 \qquad A^T=QR,
\]

where columns are ordered `[observation point columns, observation-noise columns, state point columns]` as required by the physical stack; the final matrix has observation rows and state rows. Partition

\[
 R=\begin{bmatrix}R_{yy}&R_{yx}\\0&R_{xx}\end{bmatrix},
 \qquad G=R^T=\begin{bmatrix}L_y&0\\L_{xy}&L_f\end{bmatrix}.
\]

Because `A A^T=R^T R`,

\[
 S=R_{yy}^TR_{yy}=L_yL_y^T,
\]
\[
 P_{xy}=R_{yx}^TR_{yy}=L_{xy}L_y^T,
\]
\[
 P^- =R_{yx}^TR_{yx}+R_{xx}^TR_{xx}.
\]

The gain and posterior factor are therefore

\[
 K=L_{xy}L_y^{-1},
 \qquad G_f=L_f,
\]

and

\[
 G_fG_f^T=P^- -P_{xy}S^{-1}P_{yx}.
\]

The conditional factor is obtained by orthogonal elimination. The runtime never forms the Schur-complement covariance and never applies a sequence of covariance downdates for this nonnegative-weight branch.

### 3.5 Score derivative on a fixed QR branch

For `B(theta)=Q(theta)R(theta)` with full column rank and positive diagonal, define

\[
 E=Q^T\dot B R^{-1},
 \qquad \Omega=\operatorname{sl}(E)-\operatorname{sl}(E)^T,
\]
\[
 \dot R=(E-\Omega)R,
 \qquad \dot Q=Q\Omega+(I-QQ^T)\dot B R^{-1}.
\]

Extract `dot L_y`, `dot L_xy`, and `dot L_f` from `dot G=dot R^T`. The gain derivative is the solve-form identity

\[
 \dot K L_y +K\dot L_y=\dot L_{xy},
 \qquad
 \dot K=(\dot L_{xy}-K\dot L_y)L_y^{-1}.
\]

The innovation solve is `z=L_y^{-1}e`, `e=y_obs-bar y`, and

\[
 \ell_t=-\frac12\left[n_y\log(2\pi)+2\sum_i\log(L_{y,ii})+z^Tz\right],
\]
\[
 \dot z=L_y^{-1}(\dot e-\dot L_yz),
 \qquad
 \dot\ell_t=-\frac12\left[2\sum_i\frac{\dot L_{y,ii}}{L_{y,ii}}+2z^T\dot z\right].
\]

The derivative is admitted only while dimensions, rank, column ordering, pivot/sign convention, and positive pivot margins remain fixed.

### 3.6 Fixed-pivot rectangular QR chart

For a rank-`r` stack `A` with `B=A^T in R^{k x n}`, a rank-revealing preflight supplies a fixed permutation `perm` of state coordinates. Let

\[
 B\Pi=[B_1\ B_2],\quad B_1\in\mathbb R^{k\times r},\quad \rank(B_1)=r,
\]
and compute ordinary full-column-rank QR

\[
 B_1=Q_1R_{11},\qquad R_{12}=Q_1^TB_2,
 \qquad \widehat R=[R_{11}\ R_{12}].
\]

The rectangular factor is

\[
 G=\Pi\widehat R^T\in\mathbb R^{n\times r}.
\]

The chart residual is

\[
 E_2=(I-Q_1Q_1^T)B_2.
\]

The fixed chart is valid only if `E_2` is below the declared scale-aware tolerance and the selected `R11` pivots remain positive. Its derivative is obtained by differentiating the full-rank QR of `B1` and

\[
 \dot R_{12}=\dot Q_1^TB_2+Q_1^T\dot B_2,
 \qquad \dot G=\Pi[\dot R_{11}\ \dot R_{12}]^T.
\]

If the chart residual or pivot margin fails, value mode may repivot or use direct-stack SVD; score mode fails closed.

### 3.7 Rank-revealing SVD and support likelihood

For value-only rank discovery, factor the stack itself:

\[
 A=U\Sigma V^T,\qquad G_r=U_{:,1:r}\Sigma_{1:r,1:r}.
\]

The rank is selected by a declared relative cutoff. This SVD is on `A`, never on `A A^T`. Repeated singular values and rank-threshold crossings are value branches, not smooth score branches.

For innovation support factor `G_y=U_y\Sigma_y` with rank `r_y<n_y`, set `e=y-\bar y` and

\[
 e_\perp=(I-U_yU_y^T)e.
\]

If `||e_perp||` exceeds the support tolerance, return `off_support` and log likelihood `-infinity` (or the repository’s explicit failure status). Otherwise,

\[
 z=\Sigma_y^{-1}U_y^Te,
\qquad
 \ell_t=-\frac12\left[r_y\log(2\pi)+2\sum_{i=1}^{r_y}\log\Sigma_{y,ii}+z^Tz\right].
\]

This is a density on the affine support, not an `n_y`-dimensional Lebesgue density with a zero determinant. A hard rank cutoff or support basis change invalidates the analytical score unless a new fixed branch is opened.

## 4. Implementation work packages

### WP1: Direct block-QR kernel

Add `bayesfilter/linear/block_qr_conditional_tf.py` with TensorFlow float64 kernels that:

- build and factor a batched joint stack directly;
- return `innovation_factor`, `conditional_factor`, `gain`, and derivative counterparts;
- enforce static dimensions, finite inputs, positive diagonal sign normalization, and column-count requirements;
- report minimum innovation/conditional pivots, relative pivots, full-stack reconstruction residual, and derivative reconstruction residual; and
- provide a scale-aware optional pivot threshold for robust fail-closed behavior.

### WP2: Route the full-rank SR-UKF through block QR

Modify `_one_step` in `factor_srukf_tf.py` to use the joint stack and conditional factor for the posterior. Preserve the current full-rank API and compatibility aliases for old downdate diagnostics only where tests require them; mark the aliases deprecated in metadata. The admitted factorization metadata becomes `direct_qr_block_conditional`.

### WP3: Rectangular factor primitives

Add `bayesfilter/linear/rectangular_factor_tf.py` with:

- fixed-pivot rectangular QR chart and first derivative;
- direct-stack rank-revealing SVD diagnostic/value factor;
- support decomposition and support residual; and
- support-aware Gaussian log likelihood with pseudodeterminant.

These primitives must be independent of the admitted full-rank SR-UKF route. They may use `tf.linalg.svd` only in the explicitly diagnostic/value-only module and must expose rank cutoff and branch status.

### WP4: Rectangular SR-UKF adapter

Add a separate rectangular model/result contract with state/process/observation loading factors of shapes `[B,n,r]`, `[B,q,rq]`, and `[B,m,rr]`. Use fixed latent-coordinate sigma points and the rectangular prediction/innovation primitives. Initially admit value-only fixed-support operation; admit analytical score only after the chart residual, pivot gap, and support identity are all checked.

### WP5: Tests and evidence

Add focused unit tests for each primitive and an integration test for the rectangular route. Record commands and outcomes in `docs/plans/bayesfilter_direct_rectangular_block_qr_srukf_execution_result_2026_08_17.md`.

## 5. Test matrix and gates

### Algebra and derivative gates

1. Full-rank random stacks: `G G^T=A A^T`, positive diagonal, solve residual.
2. Batched derivative reconstruction: `dG G^T+G dG^T=dA A^T+A dA^T`.
3. Center-point DZ5 residual inclusion when `w0^(m)=0` but `w0^(c)>0`.
4. Block QR: innovation, cross covariance, predicted covariance, gain, and conditional-factor Schur identity.
5. Block-QR derivative finite differences for `L_y`, `K`, `L_f`, likelihood, and filtered mean.

### Ill-conditioning gates

1. Direct stacks with singular scales from `1` through `1e-14` and `1e-15`.
2. Compare direct QR against an independent float64 SVD-of-stack reconstruction at the test boundary.
3. Verify no normal-equation covariance factorization is called by static route guards.
4. Check finite values, solve residuals, relative pivot telemetry, and deterministic failure when the declared pivot threshold is crossed.
5. Repeat in batch and eager/XLA modes where supported.

### Exact singular/rank-deficient gates

1. Rank-one and rank-zero residual stacks with rectangular factors.
2. Fixed-pivot chart with exact `B2 in range(B1)` and derivative reconstruction.
3. Chart residual failure when `B2` leaves the selected support.
4. Direct-stack SVD rank selection for repeated singular values and cutoff changes; assert value status, never score admission.
5. Singular innovation with on-support observation: finite support likelihood and correct pseudodeterminant.
6. Singular innovation with off-support observation: explicit `off_support` status and no finite ordinary Gaussian score.
7. Near-singular families on both sides of the cutoff: branch transition is visible and score route fails closed.

### Corner and failure gates

1. NaN/Inf observations, factors, stacks, and derivatives.
2. Wrong ranks, too few stack columns, duplicate pivot indices, negative/zero pivots, and inconsistent static dimensions.
3. Zero observation dimension is rejected explicitly; zero retained rank has a defined support policy.
4. Parameter-dependent process and observation loading derivatives.
5. Long finite nonlinear horizon with no negative covariance weights: no factor drift and no hidden covariance reconstruction.
6. Negative-weight UKF configuration is rejected by the block-QR route and directed to the separately audited signed-update route.
7. Route guard rejects SVD/eigen/cholesky/covariance-to-factor tokens in admitted source files while allowing them in diagnostic rectangular modules.

### Integration/parity gates

1. Affine linear model: direct block-QR means, factors, gains, likelihood, and score agree with covariance-form Kalman within declared float64 tolerance.
2. Existing non-SSL model suite: values and scores remain comparable to the pre-block-QR direct-factor artifacts; nonlinear differences are recorded rather than forced to zero.
3. Historical principal-root route remains callable only by explicit selector and is never a fallback.
4. Eager and XLA results agree on benign full-rank fixtures.

## 6. Stop conditions and nonclaims

Stop a score-bearing row when a pivot, chart residual, rank, support basis, sign convention, or likelihood measure changes. Do not add an unreported nugget, silently repivot, or silently fall back to the historical eigen route.

This plan does not claim exact nonlinear Bayesian inference, HMC readiness, broad model-suite promotion, or statistical superiority. It establishes a numerically auditable square-root representation and a bounded value/score contract.

## 7. Execution order

1. Add and test WP1 block-QR kernel.
2. Route the existing full-rank SR-UKF through WP1 and update metadata/tests.
3. Add WP3 rectangular primitives and support likelihood.
4. Add WP4 rectangular value-only adapter and fixed-chart score checks.
5. Run the full focused test matrix and integration parity tests.
6. Write the execution result with commands, environment, statuses, residual limitations, and artifact paths.
