# Normative Mathematics: Canonical Contract E--Chol LEDH Reset

Date: 2026-07-13

Status: `CHECKED_NORMATIVE_PHASE1_CLOSED`

Canonical reset ID: `contract_e_chol_v1`

Canonical derivative composition ID:
`contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`

## Scope And Authority

This document defines the finite computation that later schema and production
code must implement. Existing benchmark helpers and LaTeX are comparison sources,
not authority. Where they disagree with this specification, they are wrong
relative to the canonical target and must be corrected or retained as historical.

All equations apply per batch and per active reset time. Batch indices are
suppressed. Particles are stored as row vectors, so a cloud is
(X\in\mathbb R^{N\times d}).

## 1. Filtering Time Order

At time (t), with previous equal-weight reset cloud (X_{t-1}^\star) and
uniform log weights, the canonical finite program is:

1. apply the declared transition with fixed prepared noise;
2. apply the finite LEDH flow;
3. form corrected log weights
   \[
   a_{ti}=\ell_{t-1,i}+\log p(x_{ti}\mid x_{t-1,i})
   +\log p(y_t\mid x_{ti})-\log q_{0,t}(x_{ti})+\log|J_t|;
   \]
4. compute the observed-data likelihood increment
   \[
   \Delta_t=\operatorname{logsumexp}_i(a_{ti}),\qquad
   w_{ti}=\exp(a_{ti}-\Delta_t);
   \]
5. add (\Delta_t) to the scalar before resetting;
6. compute finite positive transport and the row quotient;
7. apply Contract E--Chol using (X_t,w_t,Y_t^+), fixed residual design, and
   prepared ridge;
8. carry (X_t^\star) and uniform log weights to time (t+1).

The reset does not alter the already-computed current increment. It affects later
increments through the next-time cloud. Moving the reset before the current
increment computes a different scalar and is wrong relative to this target.

## 2. Weight Coordinates And Normalization Pullback

The reset consumes normalized probabilities (w_i>0), \(\sum_iw_i=1\). The
transport consumes exact normalized log weights \(\log w_i=a_i-\Delta\). A
probability floor is not part of the canonical route. A zero or nonfinite
probability is a failed numerical chart; it is not silently clipped.

For upstream (g^{\log w}\) from transport and scalar upstream (g^\Delta),
the total cotangent of corrected log weights is
\[
g^a_i=g^\Delta w_i+g^{\log w}_i-w_i\sum_jg^{\log w}_j.
\]

If Contract E produces an upstream (g^w\) in probability coordinates, convert
it to normalized-log-weight coordinates before the formula above:
\[
g^{\log w,\mathrm{mom}}_i=w_i g^w_i.
\]
Equivalently, its direct contribution to (a) is
\[
g^{a,\mathrm{mom}}_i=w_i\left(g^w_i-\sum_jw_jg^w_j\right).
\]

The second expression is the simplex-tangent pullback. Consequently, adding a
constant to every component of \(g^w\) cannot change the logit cotangent, and
the resulting cotangent sums to zero. A probability-coordinate VJP must not be
added directly to a log-weight or logit VJP.

## 3. Moment Convention

All covariances are population empirical covariances. There is no (1/(N-1))
sample covariance in the target moments.

For source cloud (X\) and normalized weights (w\),
\[
\mu_w=\sum_iw_iX_i,\qquad
\Sigma_w=\sum_iw_i(X_i-\mu_w)(X_i-\mu_w)^\top.
\]
For an equal-weight cloud (Y\),
\[
\bar Y=N^{-1}\sum_iY_i,\qquad
\Sigma(Y)=N^{-1}\sum_i(Y_i-\bar Y)(Y_i-\bar Y)^\top.
\]

The factor \(\sqrt{N/(N-1)}\) used below standardizes a centered Gaussian
residual design in expectation. It does not change the covariance denominator.

## 4. Positive Transport And Row Quotient

Let the finite fixed-iteration streaming Sinkhorn computation emit nonnegative
coupling blocks. Define, in the executed row orientation,
\[
M_i=\sum_jP_{ij},\qquad Q_i=\sum_jP_{ij}X_j,\qquad
Y_i^+=Q_i/M_i.
\]

This quotient is mandatory even when the finite solver intends (M_i=1). The
canonical target has no denominator floor. Every (M_i) must be finite and
strictly positive; otherwise the reset is invalid. The JVP is
\[
\dot Y_i^+=\frac{\dot Q_i-Y_i^+\dot M_i}{M_i}.
\]
For upstream (G_i^Y\), the quotient VJP is
\[
G_i^Q=G_i^Y/M_i,\qquad
G_i^M=-\langle G_i^Y,Y_i^+\rangle/M_i.
\]
Both terms must enter the streaming transport pullback. Treating (Q) as the
cloud or omitting (G^M) is wrong relative to the canonical target.

The row-quotient cloud, not a dense matrix, is the production interface. A dense
matrix may exist only in small reference tests.

## 5. Fixed Residual Design

The canonical route fixes \(\rho=1\). The prepared residual input is
(Z\in\mathbb R^{N\times d}\), independent of model parameters and hashed into
the prepared-input identity. Define
\[
H=I_N-N^{-1}\mathbf1\mathbf1^\top,\qquad
\Xi=\sqrt{N/(N-1)}\,HZ.
\]
Thus \(\mathbf1^\top\Xi=0\). A stateless Gaussian seed schedule is admissible,
as is a fixed deterministic design; the realized tensor is part of the finite
target and must be identical across value, gradient, FD center, and endpoints.

The design need not be orthogonal to (Y^+\), because the realized covariance
and cross terms are recomputed in \(\widetilde\Sigma\) and handled by the final
affine map. Its required gates are exact identity/hash, zero column mean to
roundoff, finite values, and a valid Cholesky chart after injection. Orthogonality
or empirical whitening may be reported but is not assumed.

## 6. Contract E--Chol Forward Program

Let
\[
\Sigma_+=\Sigma(Y^+),\qquad G=\operatorname{sym}(\Sigma_w-\Sigma_+).
\]
The prepared ridge \(\lambda>0\) is a batch scalar and model-parameter-
independent input. It is selected outside the differentiated callable and is
hashed into route identity. An adaptive selector evaluated at each candidate and
then stopped is forbidden for a total-gradient claim.

Define lower-triangular Cholesky factors
\[
L_G=\operatorname{chol}(G+\lambda I),\qquad
B=L_G,
\]
and inject
\[
\widetilde Y=Y^+ + \Xi B^\top.
\]
This is the row-vector form of the existing helper's batch-linear operation. Let
\[
\bar{\widetilde Y}=N^{-1}\sum_i\widetilde Y_i,
\qquad \widetilde\Sigma=\Sigma(\widetilde Y),
\]
\[
L_w=\operatorname{chol}(\Sigma_w+\lambda I),\qquad
L_{\widetilde{}}=\operatorname{chol}(\widetilde\Sigma+\lambda I).
\]
Define the column-vector affine operator
\[
A=L_wL_{\widetilde{}}^{-1}.
\]
For row-vector storage, the output is
\[
X_i^\star=\mu_w+(\widetilde Y_i-\bar{\widetilde Y})A^\top.
\]

No explicit inverse is formed; (A\) is evaluated by a triangular solve. Every
Cholesky factor must be finite with positive diagonal. A failed factorization is
a hard invalid-chart veto, not a trigger for an unrecorded adaptive ridge.

## 7. What Is Restored Exactly

The equal-weight mean is exactly \(\mu_w\) up to floating-point error. The exact
matrix identity of the ridged chart is
\[
A(\widetilde\Sigma+\lambda I)A^\top=\Sigma_w+\lambda I.
\]
The raw output covariance is therefore
\[
\Sigma(X^\star)-\Sigma_w=\lambda(I-AA^\top).
\]

For \(\lambda>0\), calling this exact raw covariance restoration is **wrong
relative to the stated target**. The canonical numerical chart exactly restores
the ridged identity and separately vetoes excessive raw-covariance residual. A
ridge large enough to make the raw residual material changes the finite reset and
must fail the Phase 3 numerical gate even if all Cholesky factors exist. That
gate remains explicitly blocked pending a pre-result scientific ridge-bias
requirement; Phase 1 does not invent one.

## 8. Complete Pullback

Let an arbitrary downstream scalar provide (G^{X^\star}\). Reverse the centered
affine map, triangular solve, three Cholesky factorizations, realized uniform
moments, residual injection, source weighted moments, and row quotient.

The cloud-level reset returns five primary adjoints:
\[
G_X^{\mathrm{mom}},\quad G_w^{\mathrm{mom}},\quad G_{Y^+},\quad G_\Xi,
\quad g^\lambda.
\]
The transport pullback takes (G_{Y^+}\), including both quotient numerator and
row-mass terms, and returns
\[
G_X^{\mathrm{transport}},\quad G_{\log w}^{\mathrm{transport}},
\]
plus any adjoints of declared finite Sinkhorn inputs.

The canonical totals are
\[
G_X=G_X^{\mathrm{mom}}+G_X^{\mathrm{transport}},
\]
\[
G_w=G_w^{\mathrm{mom}}+G_w^{\mathrm{transport}}
\]
only when both terms have first been represented in probability coordinates.
The executed transport naturally returns a normalized-log-weight cotangent, so
the implementation normally does **not** form the second displayed sum.
Instead, it first forms
\[
g^{\log w}=G_{\log w}^{\mathrm{transport}}+w\odot G_w^{\mathrm{mom}},
\]
then applies the Section 2 log-normalization pullback exactly once to this total.
Equivalently, the moment contribution after that pullback is the
simplex-tangent expression in Section 2. Mixing coordinate systems or applying
normalization twice is wrong relative to the target.

The fixed residual design and prepared ridge have no parameter derivative:
\[
\dot\Xi=0,\qquad \dot\lambda=0.
\]
Their adjoints are still returned by the cloud primitive for independent parity
tests, but are not propagated to model parameters. If a future route makes
either parameter-dependent, it is a different route ID and must differentiate
that dependence fully.

For completeness, the normative VJP uses the Frobenius inner product and these
primitive adjoints. Define
\[
\mathcal H(V)=V-\mathbf1\,N^{-1}\mathbf1^\top V,
\qquad
\Phi(M)=\operatorname{tril}(M)-\tfrac12\operatorname{diag}(\operatorname{diag}M).
\]
For \(L=\operatorname{chol}(S)\), with symmetric \(S\),
\[
\mathcal C_L(\bar L)
=\operatorname{sym}\!\left[L^{-\top}\Phi(L^\top\bar L)L^{-1}\right]
\]
is the cotangent of \(S\). For a uniform covariance
\(\Sigma(V)=N^{-1}\mathcal H(V)^\top\mathcal H(V)\), the covariance-only
cotangent is
\[
\bar V=\frac{2}{N}\mathcal H(V)\operatorname{sym}(\bar\Sigma).
\]
All triangular systems are solved; no inverse is formed in code.

Write \(C=\mathcal H(\widetilde Y)\) and let the downstream cotangent be
\(U=G^{X^\star}\). Reversing
\(X^\star=\mathbf1\mu_w^\top+CA^\top\) gives
\[
g^\mu=\mathbf1^\top U,\qquad
\bar C=UA,\qquad
\bar A=U^\top C,\qquad
\bar{\widetilde Y}=\mathcal H(\bar C).
\]
For \(A=L_wL_{\widetilde{}}^{-1}\),
\[
\bar L_w=\bar A L_{\widetilde{}}^{-\top},\qquad
\bar L_{\widetilde{}}
=-A^\top\bar A L_{\widetilde{}}^{-\top}.
\]
Therefore
\[
\bar\Sigma_w^{(A)}=\mathcal C_{L_w}(\bar L_w),\qquad
\bar{\widetilde\Sigma}=\mathcal C_{L_{\widetilde{}}}(\bar L_{\widetilde{}}),
\]
and the realized-covariance path adds
\[
\bar{\widetilde Y}\mathrel{+}=
\frac{2}{N}C\operatorname{sym}(\bar{\widetilde\Sigma}).
\]

Reversing \(\widetilde Y=Y^++\Xi L_G^\top\) gives
\[
\bar Y^+\mathrel{+}=\bar{\widetilde Y},\qquad
\bar\Xi=\bar{\widetilde Y}L_G,\qquad
\bar L_G=\bar{\widetilde Y}^{\top}\Xi.
\]
With \(\bar G=\mathcal C_{L_G}(\bar L_G)\), the covariance-gap paths are
\[
\bar\Sigma_w=\bar\Sigma_w^{(A)}+\operatorname{sym}(\bar G),\qquad
\bar\Sigma_+=-\operatorname{sym}(\bar G),
\]
\[
\bar Y^+\mathrel{+}=
\frac{2}{N}\mathcal H(Y^+)\operatorname{sym}(\bar\Sigma_+).
\]
The prepared scalar ridge adjoint, returned only for parity in this route, is
\[
g^\lambda
=\operatorname{tr}(\bar G)
+\operatorname{tr}(\bar\Sigma_w^{(A)})
+\operatorname{tr}(\bar{\widetilde\Sigma}).
\]
This term is zero in the model-parameter total derivative only because the route
declares \(\dot\lambda=0\), not because the reset is insensitive to the ridge.

The weighted-moment VJP for upstream (g^\mu,G^\Sigma=\operatorname{sym}(G^\Sigma))
is
\[
G_{X_i}^{\mathrm{mom}}=w_i\left[g^\mu+2G^\Sigma(X_i-\mu_w)\right],
\]
\[
G_{w_i}^{\mathrm{mom}}=\langle X_i,g^\mu\rangle
+(X_i-\mu_w)^\top G^\Sigma(X_i-\mu_w).
\]
These direct terms are not contained in (G_{Y^+}\) and must not be dropped.

Applying this weighted-moment VJP to
\((g^\mu,\bar\Sigma_w)\) completes the direct source-cloud and weight paths. The
displayed VJP is evaluated on the normalized-weight
simplex. Its probability-coordinate weight adjoint is defined only up to an
additive constant after composition with the softmax/log-normalization map;
Section 2 gives the unique logit cotangent used by the filter.

## 9. Numerical Gate Classification

Two different numerical questions must not be hidden behind one tolerance:

1. **Algebraic identity checks** compare two evaluations of an exact identity,
   such as the row-quotient JVP/VJP dual pairing or the ridged covariance
   equation. Their reports use a componentwise backward-error scale assembled
   from the absolute values of the terms that were actually summed or
   multiplied. A forward-error threshold based only on \(N\), \(d\), and a
   condition-number proxy is not claimed rigorous here because the executed
   TensorFlow/XLA reduction tree and Cholesky kernels are implementation
   dependent.
2. **Algorithm-adequacy checks** decide whether finite Sinkhorn marginal error,
   raw ridge bias, conditioning, or chunk drift is small enough for the
   scientific computation. Roundoff bounds cannot answer that question. These
   gates remain explicit blockers in the Phase 1 design freeze until a reviewed
   engineering requirement or downstream error budget is supplied before the
   phase that first uses each result for promotion.

Diagnostic multipliers such as \(8\), \(32\), \(64\), or \(256\) are not
mathematical derivations. Phase 1 therefore does not promote them to hard
vetoes. Later calibration may report such multiples descriptively, but it may
not choose a pass boundary after seeing the candidate output.

## 10. Active Sets And Nonsmooth Boundaries

The canonical route records these branch identities at center and every FD
endpoint:

- reset mask at every time;
- finite Sinkhorn iteration count/schedule and fixed chunk sizes;
- finite/nonfinite status of every transport block;
- minimum row mass and row-quotient validity;
- prepared residual-design hash;
- prepared ridge hash/value;
- Cholesky success and minimum diagonal for (G+\lambda I\),
  \(\Sigma_w+\lambda I\), and \(\widetilde\Sigma+\lambda I\);
- any model support-transform branch.

There is no probability floor, adaptive ridge escalation, eigenvalue clipping,
temperature clipping, or parameter clipping in the canonical reset. If another
component has a declared floor or clip, its branch mask must be identical at FD
center/endpoints or the derivative check is inconclusive.

## 11. Same-Scalar Identity

One repository-owned callable must return the finite scalar and gradient. The
score path and every FD call must return the callable's primal. At the center,
serialized float values must be bitwise identical after the same dtype/cast
path; an algebraic reconstruction or separately compiled value is not accepted.

## 12. Source Reconciliation

| Source | Verdict relative to this specification |
| --- | --- |
| `docs/benchmarks/contract_e_reset_tf.py` fixed-ridge forward/VJP | Broadly matches row-vector Cholesky orientation, population moments, direct weighted-moment adjoints, residual adjoint, and ridged identity. It is reference-only until independent checks pass. |
| Same helper's adaptive ridge selector used inside a candidate-dependent graph | Wrong relative to the canonical total-gradient target if the selected value is stopped; only a prepared parameter-independent ridge is admissible. |
| `docs/chapters/ch32c_entropic_ot_sinkhorn.tex` Contract E--Chol proposition | Correct about the ridged identity and raw residual. Its recommendation of candidate-dependent bounded ridge escalation is not canonical gradient semantics. |
| Same LaTeX stochastic residual discussion | Correct that residual covariance restoration before the affine stage is only in expectation and that the final realized affine stage handles sample mismatch. |
| Current raw streaming output | Wrong relative to canonical row-quotient semantics when row mass differs from one, because it returns the numerator without division. |
| Current custom-gradient chapter | Incomplete for Contract E: it documents raw barycentric VJPs but not the direct moment/weight plus row-quotient composition required here. |

## 13. Nonclaims

This specification does not prove the helper, future code, streaming
implementation, LGSSM agreement, nonlinear validity, production feasibility,
HMC readiness, posterior correctness, leaderboard completeness, or scientific
superiority. Those require the later phase gates.
