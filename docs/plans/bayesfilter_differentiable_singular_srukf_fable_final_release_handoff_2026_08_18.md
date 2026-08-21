# Fable Final-Release Handoff: Differentiable Singular SR-UKF

Date: 2026-08-18  
Requester: Codex  
Review type: read-only final mathematical, numerical, documentation, and
release audit.  
Required final response: `VERDICT: AGREE` or `VERDICT: REVISE`.

## Review boundary

The implementation target is the direct-factor square-root UKF with a
full-rank direct block-QR route and a fixed-rank rectangular QR route for
singular supports. SSL-LSTM is outside scope. The intended claim is local:
values and first derivatives are valid on a preflighted fixed rank, fixed row
pivot permutation, fixed QR sign convention, fixed support/chart, and fixed
observation branch. Rank discovery and direct-stack SVD remain value-only.

Please use bounded one-path reviews first, in accordance with repository
governance. Do not review the whole repository as the first action. The exact
paths below are ordered so each question can be answered from the smallest
relevant source, then expanded only if a cited dependency is needed.

## Required audit paths and questions

### 1. Plan and execution contract

Read:

- `docs/plans/bayesfilter_differentiable_singular_srukf_gap_closure_plan_2026_08_18.md`
- `docs/plans/bayesfilter_differentiable_singular_srukf_gap_closure_plan_review_2026_08_18.md`
- `docs/plans/bayesfilter_differentiable_singular_srukf_gap_closure_execution_result_2026_08_18.md`

Audit whether the implementation stays inside the reviewed bounded scope,
whether every claimed gate has an artifact or test, and whether the stated
nonclaims are sufficient. Identify any claim that must be removed before
release.

### 2. Canonical documentation

Read the canonical paths first:

- `docs/main.tex`
- `docs/chapters/ch12_factor_derivatives.tex`
- `docs/chapters/ch14_derivative_validation.tex`
- `docs/chapters/ch17_square_root_sigma_point.tex`
- `docs/chapters/ch18_svd_sigma_point.tex`
- `docs/chapters/ch23_boundary_gradients.tex`

Then compare the corresponding five files under
`docs/fable-rewrite/monograph/chapters/` byte-for-byte. Check that the
chapters fully explain the problem, the direct block-QR algorithm, the
rectangular fixed-chart extension, the affine-support measure, the
renormalized epsilon limit, derivative boundaries, SVD's value-only role, and
the testing/nonclaim contract. Flag undefined symbols, dimension ambiguity,
incorrect transpose orientation, or a statement that implies a score across a
branch boundary.

### 3. Full-rank direct block QR

Read:

- `bayesfilter/linear/block_qr_conditional_tf.py`
- `bayesfilter/linear/stack_qr_tf.py`
- `bayesfilter/nonlinear/factor_srukf_tf.py`
- `tests/test_block_qr_conditional_tf.py`

Verify the identity for a direct residual/loading stack and the block
partition:

\[
A^T=Q\begin{bmatrix}R_{yy}&R_{yx}\\0&R_{xx}\end{bmatrix},\qquad
S=R_{yy}^T R_{yy},\quad K=R_{yx}^T R_{yy}^{-T},\quad
P_f=R_{xx}^T R_{xx}
\]

with the repository's lower-factor convention. Check that runtime code never
materializes `S`, `P_f`, or an equivalent covariance merely to factor it, and
that positive pivot/sign and signed-weight boundaries are explicit.

### 4. Rectangular QR mathematics and derivatives

Read:

- `bayesfilter/linear/rectangular_factor_tf.py`
- `bayesfilter/linear/rectangular_factor_tf.py::batched_fixed_support_qr_likelihood`
- `bayesfilter/linear/rectangular_factor_tf.py::batched_fixed_support_qr_update`
- `tests/test_rectangular_factor_tf.py`

Audit all dimensions and orientations for `B=A^T`, the retained base chart,
`R11`, `R12`, residual chart, permutation restoration, triangular solves,
projectors, and derivative contractions. In particular, independently verify
the rank-two non-diagonal authority rather than accepting rank-one evidence.
Confirm that the derivative includes support-coordinate, innovation, gain,
mean, and posterior-factor terms.

The fixed-support likelihood must be interpreted as an `r`-dimensional density:

\[
\ell_{supp}=-\frac12\left[r\log(2\pi)+2\sum_j\log R_{jj}
 +\|R^{-1}U^Te\|^2\right].
\]

Audit that this is not presented as the raw ambient Gaussian limit. For
`P_e=GG^T+eI`, verify that the on-support renormalization is
\(\ell_e+(n-r)\log(2\pi e)/2\), while off-support the ambient value tends to
`-infinity`.

### 5. Temporal fixed-rank route

Read:

- `bayesfilter/nonlinear/rectangular_srukf_tf.py`
- `tests/test_rectangular_srukf_tf.py`
- `tests/test_factor_srukf_route_guard.py`

Verify that sigma points use the retained latent rank, that the temporal
`tf.while_loop` has static shapes, and that the score route calls only fixed
QR primitives. Confirm there is no hidden SVD, eigendecomposition,
covariance-to-factor operation, or dynamic rank selection in the score call
graph. Check generated branch identity, permutation-bijection validation,
minimum pivot/chart/support telemetry, XLA behavior, and fail-closed score
invalidity.

### 6. Numerical stability and branch semantics

Audit the scale-aware pivot policy (`1e-12` default), chart residual and
support tolerances, triangular solve conditioning, QR sign convention, and
failure behavior for NaN/Inf, zero rank, rank changes, repeated singular
values, anisotropic near-rank changes, duplicate permutations, and off-support
observations. Confirm no silent nugget/jitter is introduced and that the
rank-discovery SVD is not promoted to a differentiable route.

### 7. Test coverage and independent authorities

Read:

- `tests/test_rectangular_factor_tf.py`
- `tests/test_rectangular_srukf_tf.py`
- `tests/test_block_qr_conditional_tf.py`
- `tests/test_factor_srukf_route_guard.py`
- `docs/plans/bayesfilter_differentiable_singular_srukf_gap_closure_execution_result_2026_08_18.md`

Check that tests cover values and scores, eager/XLA parity, dense covariance
authorities, centered finite differences, epsilon-limit renormalization,
on/off support, rank-zero/value-only behavior, malformed inputs, and route
closure. Check the reported `25` focused and `143` terminal passes against the
actual commands and ensure warnings are not being treated as passes. Identify
any missing test needed for release, especially a GPU/XLA execution artifact.

### 8. GPU and documentation release gates

Read:

- `scripts/run_fixed_rank_srukf_gpu_gate_20260818.py`
- `docs/plans/bayesfilter_differentiable_singular_srukf_gap_closure_execution_result_2026_08_18.md`

Verify GPU preference `3,2,1,0`, memory-growth-before-initialization, exact
one-visible-GPU validation, CPU parity, XLA, allocator telemetry, versioned
output, and checksums. The current gate was not executed because escalated
authorization returned HTTP 502; ensure this is recorded as an unresolved
release evidence gate rather than silently inferred from CPU tests. Review the
pre-existing TeX error `Font tcrm1200 at 600 not found` and confirm the PDF
non-emission is reported accurately.

## Mathematical acceptance criteria

1. Every matrix product has consistent dimensions under the repository's
   factor orientation.
2. The rank-two non-diagonal conditional update agrees with the dense authority
   and finite differences.
3. The support likelihood uses the correct Hausdorff/affine-support measure and
   the epsilon-limit normalization is stated with the correct sign.
4. Derivatives are total derivatives of the same fixed finite value program,
   including moving support coordinates and innovation terms.
5. No derivative or smoothness claim crosses rank, pivot, support, chart, sign,
   or signed-weight boundaries.

## Numerical and software acceptance criteria

1. No covariance-to-factor decomposition occurs in the admitted score path.
2. SVD is confined to direct-stack rank discovery/value-only diagnostics.
3. Static-shape/XLA execution is valid on the tested route.
4. Invalid branches return a finite/value-only result where defined and a
   clearly invalid score, never a silently repaired score.
5. Public exports, route guards, docs, and tests agree on the same canonical
   route identity.

## Required final report

Please end the audit with:

```text
VERDICT: AGREE
```

only if all mathematical, numerical, software, documentation, and evidence
criteria pass. Otherwise end with:

```text
VERDICT: REVISE
```

and list findings ordered by severity with exact file and line anchors,
including whether each finding blocks the fixed-rank score claim, the
full-rank direct-QR default, documentation completeness, or only the GPU/PDF
release evidence gate.
