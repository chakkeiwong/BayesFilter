# Plan Review: Direct Rectangular and Block-QR SR-UKF

Reviewed: 2026-08-17  
Reviewer: Codex implementation review  
Plan: `docs/plans/bayesfilter_direct_rectangular_block_qr_srukf_execution_plan_2026_08_17.md`

## Verdict

`REVISE_AND_EXECUTE`

The plan is mathematically suitable for execution after the following corrections and boundaries are made explicit.

## Algebra review

1. The current augmented DZ5 implementation already includes process noise in the propagated point cloud. The block measurement stack must therefore be

   \[
   A=\begin{bmatrix}A_y&G_r\\A_x&0\end{bmatrix},
   \]

   where `A_x` is the complete predicted-state residual stack. Appending `G_q` again would double-count process noise. A non-augmented process-noise implementation may append `G_q`, but it must choose one formulation, never both.

2. For `A^T=QR`, partitioning `R=[[R_yy,R_yx],[0,R_xx]]` gives

   \[
   S=R_{yy}^TR_{yy},\quad P_{xy}=R_{yx}^TR_{yy},\quad
   G_f=R_{xx}^T,
   \]

   and `K=R_yx^T R_yy^{-T}`. In lower-factor blocks this is `K=L_xy L_y^{-1}`, which is the identity implemented by the block kernel; all dimensions are checked in tests.

3. The block QR requires the total joint-stack column count to be at least `n_y+n_x`. If it is not, the full-rank branch must fail closed rather than padding with arbitrary random columns. Rectangular/value mode uses direct-stack SVD and support projection instead.

4. The rectangular fixed-pivot chart is valid only when the non-pivot columns lie in the selected column span. The residual `||(I-Q_1Q_1^T)B_2||` is a hard score gate. A tolerance is scale-aware and recorded.

## Numerical review

1. QR is the score-bearing factorization only on a fixed rank/pivot/sign branch.
2. Direct-stack SVD is allowed only in `rectangular_factor_tf.py`, which is explicitly value/diagnostic. SVD of a formed covariance is forbidden.
3. Singular innovation likelihood uses support dimension and pseudodeterminant, with an off-support rejection. It must not use `log(diag(L))` for zero pivots.
4. Ill-conditioned fixtures include scales around the dtype resolution boundary and a declared pivot threshold. A finite result alone is not a pass; reconstruction and solve residuals are required.
5. The existing lower-rank downdate remains available for the historical/general signed-weight branch but is not called by the nonnegative DZ5 block route.

## Differentiation review

1. QR derivatives differentiate the same fixed positive-diagonal factor gauge used by the value path.
2. Rank, pivot pattern, support basis, and likelihood measure are branch identity fields. Any change fails score admission.
3. SVD singular-vector derivatives are not used. At repeated singular values or rank cutoffs the route emits a value-only branch status.
4. Finite differences perturb the same finite value program and are compared only within a fixed branch.

## Test review

Coverage is required at four levels: primitive reconstruction, derivative reconstruction, filter integration/parity, and explicit failure semantics. The minimum required cases are full-rank benign, ill-conditioned, exact rank-one/rank-zero, repeated singular values, near-cutoff, on-support singular innovation, off-support singular innovation, NaN/Inf, dimension/column errors, batch, eager/XLA, and negative-weight rejection.

## Execution boundary

The first implementation increment changes the current full-rank default from sequential posterior downdates to block conditional QR and adds independent rectangular/value-support primitives. A complete rectangular temporal score adapter is not promoted until its fixed-chart derivative and support identity tests pass; its initial value route is explicitly labeled value-only.

## Residual risks

- A nonlinear sigma-point approximation depends on factor gauge; orthogonally equivalent rectangular factors need not yield identical nonlinear values.
- A singular support that changes with parameters has no single smooth score without an additional chart/measure construction.
- Full repository model parity remains a separate integration campaign; this plan does not claim scientific or HMC readiness.
