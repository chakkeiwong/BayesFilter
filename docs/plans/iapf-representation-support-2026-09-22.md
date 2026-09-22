# Gaussian representation and sampled-density geometry

Status: COMPLETE; 63 terminal checks pass. [Result](artifacts/iapf-representation-support-20260922-01/result.md).
Reviewed continuation of the authorized campaign. Prior:
[optimizer isolation](artifacts/iapf-optimizer-isolation-20260922-01/result.md).

Question: why can small empirical shape residual coexist with poor global
Gaussian geometry even after an independent solver? Distinguish the omitted
cross terms in diagonal log-quadratic initialization from concentration of the
sampled density criterion. All 16 oracle diagonal guides were inside the box;
another blind box or iteration expansion does not answer this question.

Evidence contract: use the same 16 clouds, both actual and exact targets, with
no new data, randomness or optimization. Compare original diagonal QR against
full quadratic QR on log targets in base R. For d variables, full design is
[1,z,z²,{z_j z_k:j<k}] with 1+2d+d(d-1)/2 columns. Require N at least this
size and a numerical full-rank margin above eps*max(N,p). Recover a full
Gaussian only when the implied precision is positive definite. Reject a
nonconcave actual-target approximation; never ridge or silently clip it.

An exact Gaussian log target must be reconstructed to 1e-9 maximum residual,
and its mean/covariance must match the independently saved R message to 1e-8.
Diagonal QR initialization must agree with the actual TF initializer to 1e-8.
These are representation checks, not proof of a deployable full-covariance
learner. At d80 the full design needs 3321 coefficients, exceeding N=1000;
this small diagnostic cannot be transferred into a paper-scale default.

Let D=[1,z,z²], C contain cross terms, and a full log target be D beta+C gamma.
The reduced least-squares coefficients satisfy
beta_reduced=beta+(D'D)^-1 D'C gamma. Compute this projection with QR, not normal
equations. Verify it to 1e-8 and report its size. Recovering full coefficients
and then taking the Gaussian's diagonal covariance is a distinct oracle
diagnostic; it is not the paper's Equation-15 minimizer. Check original bounds
again before interpreting any difference.

For sampled target b, define pi_i=b_i²/sum b_j². Under exponential tilts
p_i(theta)=b_i exp(phi_i'theta), phi=[z,z²], the normalized shape residual
S=1-(p'b)²/((p'p)(b'b)) has Hessian 2 Cov_pi(phi) at theta=0.
Derivation: the derivative of u=p/||p|| is u_i(phi_i-E_pi phi), and at u=b/||b||
the Hessian is twice its Gram matrix. This is target-centered local geometry,
not the exact Hessian of a misspecified fitted diagonal Gaussian. Whiten phi
using its unweighted empirical covariance, then report eigenvalues, pi ESS,
largest weight and smallest/largest directional curvatures. All are explanatory
diagnostics, never promotion thresholds. Verify the curvature identity with
symmetric perturbations at h=1e-3,1e-4,1e-5; require scaled discrepancy <=1e-5
at h=1e-4. Save every step rather than selecting one after viewing results.

Constructed comparisons: original diagonal QR; full quadratic recovery followed
by KL-optimal diagonal projection; full recovered Gaussian; saved exact
Gaussian and previous density/shape minima. Compare conditionally by dimension,
time, initialization and actual/exact target. Downstream filtering heuristic
vetoes remain unresolved. A representation pass or ill-conditioned local
metric can explain or nominate; neither promotes a filter or ranks methods.

Default audit: full QR is an independent known-Gaussian representation probe,
not a new runtime algorithm. Standardized coordinates and observations are
frozen from the actual captured clouds. Rank/positivity guards use machine
precision and dimension scaling; no regularization. pi follows algebra from
the existing shape criterion, not a chosen reweighting. Whitening removes
arbitrary feature units; failure of its empirical covariance is a validity
veto. The perturbation ladder checks truncation and rounding without tuning a
scientific candidate. CPU/base R is an independent-reference exception.

Pre-mortem: a full quadratic can interpolate noise or overfit a non-Gaussian
actual target. Exact Gaussian recovery and reported actual-target residuals
separate these cases; unknown actual curvature is never silently made positive.
Tiny eigenvalues need rounding qualification, and pi ESS alone does not prove
rank deficiency. Omitted-term identity is linear algebra, not by itself proof
of likelihood improvement. No original-author implementation, paper replication,
default, TF32, model-score, canonical LEDH or HMC conclusions.

Continuation vetoes: wrong identity/source, insufficient rank, wrong exact
reconstruction, failed curvature check, nonfinite accepted Gaussian or budget
exhaustion. Nonconcave actual targets and poor empirical conditioning are
diagnostic findings, not reasons to stop the investigation.

Skeptical review PASS: this tests the smallest remaining representation and
criterion hypotheses on preserved data, separates exact/approximate targets,
and cannot silently justify a quadratic-size production extension. No repeated
optimizer search or changed acceptance tolerance.

Execution root `artifacts/iapf-representation-support-20260922-01/`; transfer
45.850 CPU / 47.803 GPU hours from the exact preceding ledger. Phase cap one
CPU hour, no GPU work, four <=300s launches including repairs. Command:
`python3 docs/benchmarks/diagnose_iapf_representation.py --attempt <unique>`.
Base R, GPU hidden, one BLAS/OMP thread, preserved sources/input hashes, exact
commands/version, wall time and remaining budget. Terminal report includes
conditional tables, decision/inference status and skeptical review.
