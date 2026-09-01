# BayesFilter Direct-Factor SR-UKF Execution Plan

Date: 2026-08-15
Status: `EXECUTED_DIRECT_FACTOR_DEFAULT_2026_08_16`
Owner: BayesFilter
Downstream consumer: MacroFinance DZ5 B3 NeuTra target (future handoff only)

## 1. Purpose and boundary

This plan turns the handoff in
`/home/ubuntu/workspace/MacroFinance/docs/plans/two_currency_double_zlb_dz5_bayesfilter_direct_factor_srukf_handoff_2026_08_15.md`
into an executable BayesFilter repair program. The objective is a mathematically
correct, batch-native square-root unscented Kalman filter that carries a lower
state factor through time and does not perform a covariance-to-principal-root
eigendecomposition at every update.

The implementation is a new backend. It must not silently change the semantics
of `tf_principal_sqrt_ukf`, change MacroFinance's
`PRINCIPAL_SQRT_BACKEND`, resume NeuTra training, launch HMC/NUTS, or make a
posterior, convergence, production, or scientific claim. Following execution,
the direct-factor route is the repository default for its factor contract;
the principal-root route remains available only as an explicit historical
reference route.

Revision note, 2026-08-16: this version incorporates Fable findings FS-1 through
FS-9 from
`docs/plans/bayesfilter_direct_factor_srukf_fable_audit_reply_to_codex_2026_08_15.md`.
The focused re-audit handoff received `VERDICT: AGREE` before implementation.

The active repository checkout is `/home/ubuntu/workspace/BayesFilter` at
commit `3030d86df9cb00346df82c7c19f015c09c7c6e1f`. Existing unrelated dirty
files and `build-tf221/` are preserved. Only BayesFilter source, focused tests,
and plan-owned artifacts may be changed by this work.

## 2. Scientific and numerical contract

### 2.1 State-space law

At update (t), the latent state and observation law are

\[
  x_t = f_t(x_{t-1}, q_t;\theta), \qquad q_t\sim N(0,Q_t),
\]
\[
  y_t = g_t(x_t;\theta) + r_t, \qquad r_t\sim N(0,R_t(\theta)).
\]

The implementation may use the existing structural transition and observation
hooks, including the explicit lagged-observation contract. The route must state
which observation contract is active and must differentiate the same contract
used for the value.

### 2.2 Factor convention

All factors are lower triangular with a strictly positive diagonal:

\[
  P_x = S_x S_x',\quad Q=S_qS_q',\quad R=S_rS_r',\quad
  P_y=S_yS_y',\quad P_f=S_fS_f'.
\]

Sigma points use the repository convention

\[
  \chi_i = m + o_i S',
\]

where (o_i) is a row offset. This convention is part of the backend
identity and must be tested directly; transposing it changes the nonlinear
sigma-point orientation.

### 2.3 Exact DZ5 unscented rule

For augmented dimension (d=n_x+n_q), use `alpha=1`, `beta=2`, `kappa=0`:

\[
  \lambda=0,\quad w_0^{(m)}=0,\quad w_0^{(c)}=2,
  \quad w_i^{(m)}=w_i^{(c)}=\frac{1}{2d},
  \quad i=1,\ldots,2d.
\]

All covariance weights are nonnegative. Therefore prediction and innovation
covariances can be represented by QR residual stacks. The center point must
still be included: nonlinear propagation can make
\(f(\chi_0)\ne\bar x\), so the center residual has nonzero covariance weight
even though its mean weight is zero.

### 2.4 Batch contract

The leading batch dimension indexes independent parameter proposals. A single
compiled graph must evaluate all rows with the same static shape. No Python loop
over batch rows or parameter rows is allowed. Time recursion may use
`tf.while_loop`; loops over the fixed state/observation dimension in a primitive
must also be graph-native where they affect the batch or derivative axes.

## 3. Mathematical derivation

### 3.1 Direct augmented placement

At the start of a prediction step, the carried state is `(m,S_x)`. Construct the
augmented mean and factor without constructing an augmented covariance:

\[
  m_a = \begin{bmatrix}m\\0\end{bmatrix}, \qquad
  S_a = \operatorname{blockdiag}(S_x,S_q).
\]

Then

\[
  S_aS_a'=\operatorname{blockdiag}(P_x,Q),
\]

so the sigma points have exactly the required joint state/innovation law.
Parameter derivatives are

\[
  d_p m_a = \begin{bmatrix}d_p m\\0\end{bmatrix},\qquad
  d_pS_a=\operatorname{blockdiag}(d_pS_x,d_pS_q).
\]

For the current DZ5 model (Q) and the initial state covariance are fixed, so
their derivative blocks are zero. The API nevertheless carries them explicitly
so a future parameterized process noise model cannot be silently omitted.

### 3.2 Sigma-point propagation and prediction QR

Let (x_i=f(\chi_i)),

\[
  \bar x=\sum_iw_i^{(m)}x_i,\qquad
  \delta x_i=x_i-\bar x.
\]

The covariance is

\[
  P^- = \sum_iw_i^{(c)}\delta x_i\delta x_i'.
\]

Build the horizontal residual stack

\[
  A_x = [\sqrt{w_i^{(c)}}\,\delta x_i]_{i=0}^{2d}.
\]

Because all weights are nonnegative, no signed-rank update is required for this
rule. Factor (A_x') by thin QR:

\[
  A_x'=Q_xR_x,\qquad S^- = R_x'.
\]

With a positive diagonal normalization of (R_x),

\[
  S^-(S^-)'=R_x'R_x=A_xA_x'=P^-.
\]

The center column is included in (A_x). Process noise is included exactly
once through (S_q) in (S_a); do not append (S_q) again to (A_x).

### 3.3 QR derivative identity

For (A=QR), (Q'Q=I), and a fixed positive-diagonal sign branch, define

\[
  E=Q'(dA)R^{-1},
\]
\[
  \Omega=\operatorname{strictLower}(E)-
          \operatorname{strictLower}(E)',
\]
\[
  dR=(E-\Omega)R,\qquad
  dQ=Q\Omega+(I-QQ')dAR^{-1}.
\]

For the lower covariance factor (S=R'), (dS=dR'). The implementation must
verify both

\[
  dA=dQR+QdR,
  \qquad dS\,S'+S\,dS'=d(AA').
\]

The sign branch is not differentiable when an (R_{jj}) pivot crosses zero.
Such a row fails closed; the implementation must not replace the sign decision
with a smooth but different factor convention.

### 3.4 Observation factor and parameter-dependent noise

Propagate observation points (y_i=g_i(x_i)) and define

\[
  \bar y=\sum_iw_i^{(m)}y_i,\qquad
  \delta y_i=y_i-\bar y.
\]

The exact innovation covariance is

\[
  P_y=\sum_iw_i^{(c)}\delta y_i\delta y_i'+R(\theta).
\]

Construct

\[
  A_y=[\sqrt{w_i^{(c)}}\,\delta y_i]_{i=0}^{2d}\;\Vert\;S_r,
\]

where `||` means horizontal concatenation. QR of (A_y'), followed by positive
diagonal normalization, gives (S_y) with

\[
  S_yS_y'=A_yA_y'=P_y.
\]

If (R) depends on parameters, its derivative must enter the stack derivative:

\[
  d_pA_y=[\sqrt{w_i^{(c)}}\,d_p\delta y_i]_i\;\Vert\;d_pS_r.
\]

The route must not compute a covariance derivative and refactorize it as a
replacement for (dS_y).

### 3.5 Gain and filtered mean

The cross covariance is

\[
  P_{xy}=\sum_iw_i^{(c)}\delta x_i\delta y_i'.
\]

The gain is evaluated without an inverse:

\[
  K=P_{xy}(S_yS_y')^{-1}.
\]

Implementation form:

\[
  U:=K S_y,\qquad U S_y'=P_{xy},
\]
\[
  U'=S_y^{-1}P_{xy}',\qquad K=US_y^{-1},
  \qquad\text{equivalently}\qquad K'=S_y^{-T}U'.
\]

The first equation is a lower-triangular solve with coefficient \(S_y\) after
transposing \(U S_y'=P_{xy}\); the second is a right solve with \(S_y\), or an
upper-triangular solve for \(K'\). Thus

\[
  K=P_{xy}S_y^{-T}S_y^{-1}=P_{xy}(S_yS_y')^{-1}
\]

without forming an inverse. The matrix \(U\) is intentionally the same matrix
later called \(V\) in Section 3.7, so the solve is performed once and reused
for both the filtered mean and the downdate.

With innovation (e=y_t-\bar y),

\[
  m_f=\bar x+Ke.
\]

For derivatives, first solve

\[
  dU'=S_y^{-1}(dP_{xy}'-dS_yU'),
\]
\[
  dK=dU\,S_y^{-1}-K\,dS_y\,S_y^{-1},
\]
\[
  d m_f=d\bar x+dK\,e+K\,d e,
  \qquad d e=-d\bar y.
\]

The derivative solve follows by differentiating
\(S_yU'=P_{xy}'\):
\[
  S_y\,dU'+dS_y\,U'=dP_{xy}'.
\]
It uses the same lower-triangular orientation as the value solve.

This keeps the derivative path factor-native while retaining the exact gain
mathematics.

### 3.6 Likelihood and score

Solve the innovation in factor coordinates:

\[
  z=S_y^{-1}e.
\]

The log likelihood increment is

\[
  \ell_t=-\frac12\left[n_y\log(2\pi)+
  2\sum_j\log(S_{y,jj})+z'z\right].
\]

For parameter (p),

\[
  d_pe=d_p(y_t-\bar y)=-d_p\bar y,
\]
\[
  d_pz=S_y^{-1}(d_pe-d_pS_yz),
\]
\[
  d_p\log\det(P_y)=2\sum_j\frac{(d_pS_y)_{jj}}{(S_y)_{jj}},
\]
\[
  d_p(z'z)=2z'd_pz,
\]
\[
  d_p\ell_t=-\frac12\left[
    2\sum_j\frac{(d_pS_y)_{jj}}{(S_y)_{jj}}+2z'd_pz
  \right].
\]

This must agree with the covariance-form identity on benign fixtures:

\[
  d_p\ell_t=-\frac12\left[
    \operatorname{tr}(P_y^{-1}d_pP_y)+
    2(d_pe)'P_y^{-1}e-
    e'P_y^{-1}(d_pP_y)P_y^{-1}e
  \right].
\]

The covariance expression is an independent comparator only; it is not used to
produce the repaired route's score.

### 3.7 Filtered factor by downdates

Let

\[
  V=U=KS_y,\qquad dV=dU.
\]

Then

\[
  P_f=P^- - K(S_yS_y')K'=S^-(S^-)' - VV'.
\]

If (v_j) denotes column (j) of (V), apply sequential rank-one lower
Cholesky downdates:

\[
  L^{(0)}=S^-,\qquad
  L^{(j+1)}L^{(j+1)'}=L^{(j)}L^{(j)'}-v_jv_j',
\]

and return (S_f=L^{(n_y)}). No (P_f) tensor is materialized and no covariance
refactorization via an eigendecomposition, SVD, or library factorization is
allowed.

For a lower factor (L) and vector (x), the scalar-pivot downdate is:

\[
  r=\sqrt{L_{kk}^2-x_k^2},\quad c=r/L_{kk},\quad s=x_k/L_{kk},
\]
\[
  L_{kk}\leftarrow r,
\]
\[
  a_{old}\leftarrow L_{k+1:n,k},\quad
  u_{old}\leftarrow x_{k+1:n},
\]
\[
  a_{new}\leftarrow(a_{old}-s u_{old})/c,\qquad
  u_{new}\leftarrow c u_{old}-s a_{new}.
\]

The evaluation order is sequential: compute \(a_{new}\) from old \(a,u\), then
compute \(u_{new}\) from old \(u\) and the new \(a\). The equivalent all-old
form \(u_{new}=(u_{old}-s a_{old})/c\) may be used, but the implementation and
derivative tests must document which form is used. Every pivot requires finite
values and \(L_{kk}^2-x_k^2>0\); a zero or nonfinite margin is a hard failure,
not a reason to add a nugget.

For first derivatives, with (dL_{kk},dx_k):

All right-hand-side pivot quantities in the following equations are the old,
pre-update values.

\[
  dr=(L_{kk}dL_{kk}-x_kdx_k)/r,
\]
\[
  dc=(drL_{kk}-rdL_{kk})/L_{kk}^2,
  \qquad ds=(dx_kL_{kk}-x_kdL_{kk})/L_{kk}^2.
\]

For old (a,u,da,du), differentiate the two vector assignments. Compute
\(da_{new}\) first, then use it when computing \(du_{new}\):

\[
  da_{new}=\frac{(da-ds\,u-s\,du)c-(a-su)dc}{c^2},
\]
\[
  du_{new}=dc\,u+c\,du-ds\,a_{new}-s\,da_{new}.
\]

The primitive must carry the batch and parameter axes through these equations.
It may use a `tf.while_loop` over the fixed pivot index and downdate-column
index, but never a Python loop over (B) or (P). Tests must verify

\[
  dS_fS_f'+S_fdS_f'=dP_f,
\]

where the independent comparator assembles

\[
  dP_f=dS^-(S^-)' + S^-d(S^-)' - dV\,V' - V\,dV'.
\]

The runtime itself never assembles \(P_f\) or \(dP_f\) to obtain \(S_f\) or
\(dS_f\).

Feasibility and failure attribution: after downdating the first \(k\) columns,
the exact partial covariance satisfies

\[
  P^{(k)}=(L^{(k)})(L^{(k)})'
        =P_f+\sum_{j>k}v_jv_j'\succeq P_f.
\]

Consequently, if the exact target \(P_f\) is SPD, every intermediate target is
SPD and every exact scalar-pivot margin is positive for every column order. A
failed runtime margin is therefore classified only diagnostically, after the
original failure is retained. For a row whose failure code is
`downdate_margin_nonpositive`, an independently assembled comparator with
\[
  \lambda_{\min}(P_f)\le 0
\]
is `downdate_target_indefinite`; a comparator with strictly positive minimum
eigenvalue is `downdate_roundoff_or_implementation_suspected`. This offline
classification may use an eigenvalue routine, never the admitted runtime and
never as a fallback. The comparator covariance, minimum eigenvalue, first
failed time/column/pivot, margin, and conditioning diagnostics are all retained
in the failure artifact.

As an independent block comparator, define the zero-extended prediction stack
\(A_x^+=[A_x,0_{n_x\times n_y}]\) and form

\[
  \widetilde A_f=A_x^+-K A_y.
\]

Since \(A_x^+(A_x^+)'=P^-\), \(A_x^+A_y'=P_{xy}\),
\(A_yA_y'=P_y\), and \(K=P_{xy}P_y^{-1}\),

\[
  \widetilde A_f\widetilde A_f'
  =P^- -K P_y K'=P_f.
\]

QR of \(\widetilde A_f'\), with its independently propagated derivative, is a
diagnostic comparator for \(S_f,dS_f\) and sequential-rounding error. It is
never a runtime fallback and must not be used to repair a failed downdate.
Its stack derivative is

\[
  d\widetilde A_f=dA_x^+-dK\,A_y-K\,dA_y,
\]

which must be propagated through the independent QR derivative before comparing
with the sequential-downdate \(dS_f\).

## 4. Repository implementation design

### 4.1 New files and bounded modifications

Preferred new files:

```text
bayesfilter/linear/stack_qr_tf.py
bayesfilter/linear/lower_rank_downdate_tf.py
bayesfilter/nonlinear/factor_srukf_tf.py
bayesfilter/nonlinear/factor_srukf_compat.py
tests/test_stack_qr_tf.py
tests/test_lower_rank_downdate_tf.py
tests/test_factor_srukf_tf.py
tests/test_factor_srukf_route_guard.py
```

Expected small integration edits:

```text
bayesfilter/linear/__init__.py
bayesfilter/nonlinear/__init__.py
bayesfilter/nonlinear/srukf_route_guard.py
```

Do not modify MacroFinance files in this phase. Do not modify the principal-root
implementation except for a compatibility-preserving export or test fixture if
strictly necessary.

### 4.2 Factor adapter

Define an explicit factor adapter with static dimensions and fields equivalent to:

```text
initial_mean          [B, nx]
initial_factor        [B, nx, nx]
process_factor        [B, nq, nq]
observation_factor    [B, ny, ny]
d_initial_mean       [B, P, nx]
d_initial_factor     [B, P, nx, nx]
d_process_factor     [B, P, nq, nq]
d_observation_factor [B, P, ny, ny]
```

The existing `TFBatchedStructuralStateSpace` and
`TFBatchedStructuralFirstDerivatives` remain accepted through a reviewed
compatibility constructor. A covariance-to-factor conversion, if needed for a
legacy caller, is implemented in the separate non-admitted module
`bayesfilter/nonlinear/factor_srukf_compat.py`. It is performed once before
the recursive function is traced and is recorded as
`compatibility_factorization_boundary`; it is never imported by the admitted
recursive runtime file and never performed inside the update loop. The active
DZ5 adapter should provide `S_r,dS_r` directly.

### 4.3 Static validation and fail-closed policy

At API entry validate static `B,P,nx,nq,ny,T` and exact rule dimensions. In the
recursive body assert:

- all means, factors, derivatives, and residual stacks are finite;
- all factor diagonals are strictly positive;
- QR has at least as many columns as rows;
- downdate margins are strictly positive;
- derivative reconstruction residuals are finite;
- likelihood and score increments are finite.

Return per-row diagnostics for minimum QR pivot, minimum downdate margin,
maximum factor residual, maximum derivative residual, relative pivot and
margin conditioning, first failed time/column/pivot, and failure counts. The
failure code is one of `qr_pivot_nonpositive`,
`downdate_margin_nonpositive`, `nonfinite_derivative`,
`invalid_observation_factor`, or `batch_contract_violation` (with
`none` for a successful row). Reject NaN/Inf inputs and intermediate values
explicitly. Do not replace failed rows with a floor, principal-root fallback,
SVD fallback, or silent NaN-to-large-negative likelihood.

Relative conditioning diagnostics are advisory only: record each QR pivot
relative to the corresponding stack column norm and each downdate margin
relative to \(L_{kk}^2\). Strict positive pivots and margins remain hard gates;
conditioning thresholds must not silently convert a valid row into a different
algorithm or excuse a failed row.

### 4.4 XLA and execution mode

The public default-route API defaults to `tf.function(jit_compile=True)` or the
repository equivalent. Eager and non-XLA execution are reference/debug modes,
not production evidence. GPU tests must set `TF_FORCE_GPU_ALLOW_GROWTH=true`
before TensorFlow import and use the repository GPU memory-policy helper. Every
serious artifact records dtype, XLA status, device name/UUID, memory-growth
verification, and wall time.

## 5. Ordered execution phases

Each phase has a stop condition. A failed phase produces a retained diagnostic
artifact and blocks downstream promotion; it does not authorize a fallback to
the principal-root route.

### Phase -1: focused Fable re-audit gate

Before editing implementation files, send
`docs/plans/bayesfilter_direct_factor_srukf_fable_focused_reaudit_handoff_2026_08_16.md`
to Fable. Fable must inspect the revised Section 3.5 and 3.7 equations, the
explicit admitted-file boundary, the feasibility lemma, the block comparator,
and the FS-5--FS-9 test hardening using the bounded MathDevMCP protocol in that
handoff. Retain the report under its unique artifact root with source hashes,
commands, numerical counterexamples, tool abstentions, and the terminal
verdict. The only gate that permits Phase 0 implementation preparation is
`VERDICT: AGREE`; `VERDICT: REVISE`, a missing report, or an unresolved
counterexample stops the plan with `REPAIR_INCOMPLETE`. This is a plan-review
gate only, not implementation, integration, or production authority.

### Phase 0: source and contract freeze

Read and hash:

- the MacroFinance handoff;
- `MacroFinance/docs/latex-papers/CIP_monograph/chapters/ch17_nonlinear_filtering.tex`;
- `bayesfilter/nonlinear/srukf_factor_tf.py`;
- `bayesfilter/linear/qr_factor_tf.py`;
- `bayesfilter/linear/kalman_qr_derivatives_tf.py`;
- `bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py`;
- current focused SR-UKF tests and route guard.

Write a source manifest containing paths, SHA-256 digests, commit, Python,
TensorFlow/TFP versions, and the exact dirty-worktree status. Confirm the
scientific target, nonclaims, dimensions, float64 requirement, and no-switch
boundary before editing code.

### Phase 1: QR factor and derivative primitive

Implement positive-diagonal batched QR factorization and derivative wrappers in
`bayesfilter/linear/stack_qr_tf.py`.
Add unit tests for reconstruction, orthogonality, positive pivots, finite
differences, batch permutation invariance, failure on rank-deficient stacks,
and explicit NaN/Inf rejection for stacks and stack derivatives.

Required commands:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true pytest -q tests/test_stack_qr_tf.py
```

### Phase 2: rank-one downdate primitive

Implement value and derivative downdates in
`bayesfilter/linear/lower_rank_downdate_tf.py`. Include batched and scalar
reference paths only as independently tested comparators; the admitted path is
the batched TensorFlow implementation.

Tests must cover:

- one downdate with known (P-vv');
- multiple sequential columns;
- reconstruction against an independently assembled covariance;
- positive diagonal and pivot-margin diagnostics;
- rejection of indefinite downdates;
- derivative reconstruction;
- centered finite differences for both factor and derivative;
- explicit NaN/Inf rejection for factors, vectors, and derivatives;
- no NumPy import in the runtime primitive;
- no covariance-factorization/eigh/SVD fallback.

### Phase 3: one-step direct-factor SR-UKF

Replace the current prototype's observation-noise omission and filtered-factor
refactorization with the factor contract above. Add a one-step result carrying
value, score, filtered mean, filtered factor, optional comparator covariance,
and diagnostics.

Use these fixtures:

1. affine linear-Gaussian state/observation with nonzero (Q,R);
2. nonlinear transition with nonzero propagated center residual;
3. parameterized observation noise with nonzero (dS_r);
4. a deliberately failed downdate;
5. a mathematically indefinite target; and
6. a near-margin finite-precision target.

The one-step route must match closed-form Gaussian likelihood and score in the
affine fixture. It must differ from a center-omitting implementation in the
nonlinear fixture, proving the center column is active.
The affine fixture must also assert filtered mean and filtered factor against
closed-form Kalman values, and the gain primitive must be compared with a dense
\(P_{xy}P_y^{-1}\) oracle so a wrong triangular orientation cannot pass by
checking likelihood alone. Fixtures 5 and 6 retain the original failure and
carry the diagnostic-only `downdate_target_indefinite` versus
`downdate_roundoff_or_implementation_suspected` classification.

### Phase 4: batched multi-step recursion

Implement `tf_batched_factor_srukf_value_and_score` with a graph-native time
loop. Carry `mean,S_x,d_mean,dS_x`; never carry covariance as the authority.
Record optional factor histories for diagnostics and filtered means/factors for
tests. Add static shape validation and route-guard tests before integrating any
DZ5 adapter.

### Phase 5: benign parity and derivative cross-checks

Use stable affine and mildly nonlinear synthetic models. Compare:

- factor covariance reconstruction against an independent covariance assembly;
- sequential filtered factor against the all-additive block-stack QR comparator
  and its derivative, as a diagnostic only;
- value against the existing covariance/principal-root route where sigma-point
  orientation is intentionally equivalent;
- score against centered finite differences;
- score against an independent reverse/output-cotangent calculation;
- factor-route score against the covariance-form score on benign fixtures.

Do not claim general nonlinear equivalence to the principal-root route because
factor orientation is a legitimate nonlinear algorithmic choice.

### Phase 6: exact DZ5 rare-row qualification

Use the MacroFinance frozen rows `111` and `251`, prefixes `1,16,31,44,96`,
contexts batch `1`, duplicate batch `2`, mixed batch `2`, batch `36`, and exact
production batch `480`. Include duplicates, mixtures, row permutations, and
the exact production ordering.

For every row, prefix, and context retain:

```text
value, score, filtered mean/factor, innovation factor,
minimum QR pivot, minimum downdate margin,
factor reconstruction residual, derivative residual,
finite/failure status, execution mode, device provenance
```

The hard row-independence gate requires identical physical rows to have
invariant value and score within predeclared float64 tolerances across all
contexts and permutations. A failure is `REPAIR_INCOMPLETE`.

### Phase 7: execution-mode matrix

Requalify the default route on CPU eager, CPU graph, CPU XLA, trusted-GPU eager,
trusted-GPU graph, and trusted-GPU XLA where supported. Record failures rather
than silently dropping modes. GPU artifacts must include the managed-session
trust basis only when all managed-session conditions are met.

### Phase 8: result memo and downstream integration note

Write a result memo under a unique versioned artifact root containing source
hashes, commands, environment, seeds, devices, timings, tolerances, raw JSON,
aggregated JSON/Markdown, test logs, and all pass/fail decisions. State clearly
that MacroFinance adapter changes, target switching, NeuTra, HMC, and scientific
promotion remain blocked pending separate review.

## 6. Test matrix and acceptance thresholds

Thresholds and the finite-difference protocol are declared before the rare-row
run and remain fixed during that run. Suggested initial float64 thresholds, to
be confirmed on Phase 0/1 fixtures, are:

```text
QR reconstruction relative residual       <= 1e-12
QR derivative relative residual           <= 1e-10
downdate reconstruction relative residual <= 1e-12
downdate derivative relative residual     <= 1e-10
finite-difference score agreement         <= 1e-7 relative, 1e-9 absolute
row value/score batch invariance          <= 1e-10 relative, 1e-11 absolute
minimum factor diagonal                   > 0
minimum downdate margin                   > 0
```

Finite differences use centered differences
\((F(p+h)-F(p-h))/(2h)\), with a per-parameter step
\(h_i=\eta\max(1,|p_i|)\). Record \(\eta\), the actual \(h_i\), and repeat every
check at \(h_i/2\); require agreement with the analytic derivative at both
steps and report the change between the two estimates. The step sizes are
chosen from benign-fixture conditioning before any rare-row result is
inspected.

The final thresholds must be justified by benign-fixture conditioning and
reported with condition diagnostics. A tolerance may not be widened after a
rare-row failure without a new reviewed plan and fresh holdout evidence.

Unit-test coverage must include value, derivative, shape, dtype, failure, and
route-guard behavior. Integration coverage must include complete time recursion,
parameter batching, observation-noise derivatives, lagged observation if active,
XLA/eager parity, duplicate/mixed/permuted rows, exact DZ5 prefixes, and
artifact schema validation.

## 7. Static route guard

The admitted direct-factor source set is an explicit, closed file list:

```text
bayesfilter/linear/stack_qr_tf.py
bayesfilter/linear/lower_rank_downdate_tf.py
bayesfilter/nonlinear/factor_srukf_tf.py
```

The compatibility converter `bayesfilter/nonlinear/factor_srukf_compat.py`
and all principal-root/SVD implementations are non-admitted and cannot be
imported by the recursive runtime. The QR kernel is standalone and must not
import the legacy mixed-purpose `qr_factor_tf.py`, which contains compatibility
factorization helpers. The guard scans every admitted file before tests and in
CI and applies a strict substring ban for:

```text
tf.linalg.eigh
tf.linalg.svd
cholesky
tf_principal_sqrt_ukf
principal_sqrt
tf_svd_sigma_point_filter
tf_svd_cubature
tf_svd_ukf
experimental_batched_svd_sigma_point_tf
strict_spd_principal_sqrt
principal_sqrt_frechet_derivative
covariance_to_factor
covariance_to_root
refactorize_covariance
```

These exact strings are added to `FORBIDDEN_SRUKF_ROUTE_PATTERNS` while
retaining its existing prohibitions. Matching is case-insensitive so spelling a
forbidden route in a comment or alias with different capitalization cannot
evade the guard. The guard test also asserts the closed
admitted file list, so a new numerical kernel cannot enter the backend without
an explicit guard update.

The guard is lexical source evidence, not a mathematical proof. Runtime
diagnostics and tests must independently establish that factors are actually
carried, that the filtered factor comes from sequential downdates, and that
the compatibility boundary is not reachable from the admitted import graph.

## 8. Deliverables and stop rules

Deliverables:

1. New factor QR/downdate and factor-SR-UKF source files.
2. Minimal exports and route guard updates.
3. Focused primitive, one-step, multi-step, derivative, batch-invariance, and
   execution-mode tests.
4. Source-level forbidden-route report.
5. Raw and aggregated qualification artifacts.
6. Result memo and MacroFinance integration note.
7. Fable/MathDevMCP audit packet and terminal audit result.

Stop with `REPAIR_INCOMPLETE` if any mathematical identity, finite-difference
check, factor/downdate margin, row-independence gate, execution mode, or route
guard fails. No production switch, NeuTra training, HMC, or posterior claim is
allowed after a partial implementation.

## 9. Nonclaims

Passing this plan would establish only that the new BayesFilter backend is a
factor-native, numerically tested SR-UKF default route for the declared factor
contract and fixtures/execution modes. It does not establish statistical
superiority, nonlinear equivalence to the principal-root orientation, posterior correctness,
identification, HMC readiness, NeuTra training readiness, production readiness,
or broad scientific validity.
