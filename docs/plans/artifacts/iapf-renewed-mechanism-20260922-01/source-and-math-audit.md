# What the reference and GPU failures mean

The filtering identity, the learned guide, and the compiled implementation
are three different possible sources of error. The renewed campaign tests
them separately. Its reference is the first linear-Gaussian study in
Guarniero, Johansen and Lee (GJL): state dimensions 5, 10, 20, 40 and 80,
100 observations, identity initial/process/observation covariances and
transition entries A_ij = 0.42^(|i-j|+1).

## Source and consumer audit

The local primary source is
`.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf`
and its tractable text. The inspected technical material includes
Propositions 1–3, Algorithms 3–5, Sections 3.3–3.4 and 5.1–5.2,
Equations 15–16, and the variance proof in the appendix (text lines
1970–2030). This is the locally stored arXiv working-paper version. No new
original-author implementation, unpublished numerical choices, or published
correction was recovered in this campaign. The local executable references
therefore remain independent implementations, not recovered author code.

| Question | Checked source and consumer | Conclusion |
|---|---|---|
| Twisted-filter identity | GJL Propositions 1–2, Algorithm 3; `reference_iapf_paper.R::iapf_apf` | Exact full Gaussian guides agree pathwise with Kalman in the tested R model. |
| Backward learning | GJL Algorithm 4; `iapf_iterate` → fitter → `iapf_backward_log_target` | Both tested R learners use the newly fitted next-time guide. |
| Paper fitting objective | GJL Equation 15; `reference_iapf_author_choices.R` and `reference_iapf_constrained_diagnostic.R` | Log-quadratic QR and weighted log regression compute different objectives from Equation 15. |
| Positive floor | GJL Equation 16; local `tail8` choice | Positivity is explicit; the numerical floor rule is a local hypothesis. |
| Full TF filter | `iapf_adapter.py::execute_iapf` → `make_fitted_twist_kernel` | Lower-level d=1,2,5 fixed-guide comparisons execute; the adaptive consumer still rejects d=o>1. |
| TF guide update | `twisted_transition` in `fitted_twist_tf.py` | A shared GPU/XLA/TF32 expression has a reproducible time-varying-center failure; see source snapshots and GPU probes. |
| Score target | `make_fitted_twist_kernel` analytic tangent | Fixed-guide, fixed-label derivative of its finite program. This is not automatically the marginal likelihood's model score. |

R starts from X_1 and uses adaptive ESS resampling with retained weights.
The current TensorFlow consumer starts from X_0, transitions before the first
observation, and resamples every time. The bridge deliberately implements
that TensorFlow finite program independently in R, including its supplied
random draws. Passing it cannot establish equivalence of the two complete
adaptive learners. The bridge checks finite-program derivatives only where
the independent finite differences preserve every discrete choice.

The paper's asymptotic variance statement concerns its specified particle
construction and limiting regime. The current KL diagnostic is a local
derivation, not that variance formula, and is not substituted for downstream
likelihood accuracy or a finite-N adaptive-resampling theorem.

## Why a correct optimizer can fit a poor guide

Write a dense Gaussian log guide in standardized coordinates z as

\[
\log h(z)=a+b^Tz-\tfrac12z^TJz.
\]

The fitted diagonal family uses only the columns D=(1,z,z²). All cross
terms z_j z_k, j≠k, are omitted. If r denotes those omitted terms and any
other mismatch, unconstrained weighted regression gives

\[
\widehat\beta=\beta_{\rm diag}+(D^TWD)^{-1}D^TWr,
\]

when the inverse exists. The second term need not preserve negative fitted
quadratic coefficients. A valid full Gaussian can therefore produce an
invalid diagonal log-regression fit on a particular weighted cloud.

The weighted arm takes w_i proportional to exp(y_i-max y), where y is the
backward log target. Its weight ESS is 1/sum_i w_i². Near ESS=1, one particle
dominates an objective containing 161 coefficients at d=80. ESS is not
algebraic rank, but the observed small singular values and activated ridges
confirm the numerical concentration problem. A ridge and curvature bounds
can make the optimization well defined; a small KKT residual then certifies
that constrained objective. Neither operation recovers missing cloud
information or omitted cross terms.

The exponent-zero intervention keeps the same solver, guard, floor and
filter but makes W uniform. On the initial 40 fresh paired cells, it agrees
with QR terminal likelihoods within 3.64e-12, with no ridge or active bound
in 25,200 fits. The weighted arm needs 1,685 positive ridges in 25,300 fits.
This isolates the weighting as the source of the observed need for numerical
repair in these cases. It does not prove weighting is always worse: the
weighted arm has lower descriptive likelihood error at d=20 in that batch.

Equation 15 is a separate difficulty. If p is the vector of candidate
Gaussian densities and b the positive backward targets, profiling its
positive scale gives

\[
\lambda_*={p^Tb\over b^Tb},\qquad
L_* = \|p\|^2-{(p^Tb)^2\over\|b\|^2}.
\]

As a Gaussian spreads out and its density at every finite training point
tends to zero, L_* tends to zero as well. Thus an unrestricted global
minimization can favor disappearing densities rather than a useful guide.
This is a property of the stated objective. It does not prove that the
authors' initialized local numerical procedure follows that limiting route;
that procedure remains incompletely recovered.

## Separating family error from learning error

Let the exact prediction be p=N(a,P), and the exact smoothed marginal be
s=N(b,S). A diagonal Gaussian guide with precision diag(j), j≥0, produces
q proportional to p times the guide. After optimizing its linear term,
q has mean b and covariance C=(P⁻¹+diag(j))⁻¹. The Gaussian KL formula is

\[
\operatorname{KL}(s\Vert q)=\tfrac12\{
\operatorname{tr}(C^{-1}S)+(b-m_q)^TC^{-1}(b-m_q)-d
+\log\det C-\log\det S\}.
\]

Minimizing over j therefore amounts to minimizing
F(j)=½[tr(JS)-log det J], J=P⁻¹+diag(j). Its gradient is
½[diag(S)-diag(C)] and Hessian ½(C∘C), which is positive semidefinite.
The resulting convex optimum is the best attainable Gaussian-component
discrepancy in this diagonal-precision family, including its closure.
Independent RTS moments, finite-difference gradients, KKT residuals and
feasible-comparator inequalities check the implementation.

Across the saved d=80 guides, mean KL is 0.198683 for QR and 0.497642
for weighted regression; the attainable diagonal optimum is 0.164935.
Thus diagonal-family error explains much of QR's discrepancy, while the
weighted learner has substantial additional fitting error. These are guide
distances, not likelihood-performance rankings.

The actual positive-floor guide yields a mixture of q and p. The convex
Gaussian result is not a lower bound for this larger mixture family. The
separate mixture calculation finds floor probabilities below 3.5e-23 on
these saved cases; this rules out a material floor effect in the inspected
regime, not in unseen extreme tails. Local backward-fit and inherited-guide
errors are measured separately on common independent smoothing draws.

## Why precision can change a path, and why this case also has a bug

Categorical resampling picks the first index k with cumulative probability
C_k greater than a supplied uniform u. A perturbation δC_k changes that
choice when it crosses u. A small arithmetic error can therefore cause a
large later particle displacement. This is a discontinuity, not by itself
proof of biased likelihood estimates.

The trace checks bind CDFs and indices to the actual consumer, and verify
that exposing them leaves its original outputs bit-for-bit unchanged.
Some TF32 path changes are ordinary boundary crossings. However, the d=2
fitted case has a large common particle displacement before any changed
ancestor or mixture choice. The transition mean should be

\[
m+K(c_t-m),\qquad K=Q(Q+V_t)^{-1}.
\]

In the isolated original GPU/XLA/TF32 loop, the error instead agrees with
K(c_0-c_t). The same routine is correct in FP64, ordinary FP32, non-XLA,
and statically expanded TF32 evaluations up to their respective precision
errors. This is a wrong result at the compiled-loop boundary on the tested
stack, rather than ordinary TF32 rounding.

The shared repair computes m-Km+Kc_t and expands its analytic tangent
consistently, keeping the time-varying center out of the broadcast GEMM
operand. The precision-form independent Gaussian product and full consumer
checks evaluate the repair. Ordinary TF32 rounding and boundary crossings
remain distinct from this removed wrong-center behavior. No general claim
about every GPU, compiler version, or canonical LEDH route follows.

The standalone probe in `gpu-attempt008` removes all filtering operations.
For K=[[.47,-.01],[-.01,.37]] and centers (0,0), (1,-1), (-2,2),
(3,-3), (-4,4), the original broadcast product's successive maximum errors
are approximately 0.00017, 0.48008, 0.96007, 1.44008 and 1.92007.
Those displacements agree with K(c_0-c_t). Splitting the expression keeps
the error below 0.000167. Static unrolling keeps it below 0.00124.
The optimized HLO preserves the time index but incorporates the dynamic
slice, broadcast and subtraction into a Triton GEMM fusion.

In `gpu-attempt009`, disabling that fusion for this diagnostic process
changes the original expression to a cuBLAS GEMM and reduces every error
below 1.69e-7. Together, the isolated expression, time dependence and
compiler control localize the defect to the fused route on TensorFlow
2.20.0-dev0+selfbuilt, CUDA 12.8.1, RTX 4080 SUPER. The exact compiler
source-level error and its presence in other builds remain untested. The
repository retains its numerical defaults and uses the algebraic repair.

Fifty-four focused and consumer tests pass after the repair. The final
GPU bridge passes all 18 FP64 and all 18 FP32-without-TF32 comparisons;
11 of 18 strict TF32 comparisons still fail. The trace binds these remaining
discrepancies to ordinary rounding and, in some cases, changed discrete
choices. That explanation does not establish likelihood accuracy under
TF32. See [the terminal result](result.md) for the statistical evidence and
remaining comparison boundaries.
