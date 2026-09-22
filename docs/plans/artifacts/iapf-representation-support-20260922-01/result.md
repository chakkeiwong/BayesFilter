# Cross terms and concentration explain additional guide-fitting gaps

All 32 preserved fitting targets passed the independent base-R representation
and curvature checks. The full quadratic recovered every exact Gaussian message:
maximum log residual 3.20e-14, center error 3.22e-15, covariance error 1.06e-15.
Original diagonal QR parameters match TensorFlow to 8.44e-15. The omitted-term
coefficient identity holds to 7.21e-15; symmetric curvature checks pass with
maximum scaled error 1.15e-7 at the predeclared step 1e-4.

Two specific mechanisms are supported. First, diagonal log regression omits
cross-coordinate terms present even in an exactly Gaussian target. Those terms
project into the retained coefficients on the actual sampled cloud. In the
d10 time-0 exact target, diagonal QR gives KL 8.575 while full reconstruction
followed by diagonal projection gives the family minimum 0.4381. Both lie
inside the original box. This gap is a representation effect, not a failed
gradient calculation or inadequate box size.

Second, the normalized sampled density criterion concentrates its information
on very few points. At d10 time 1, the target-squared weights give one particle
97.1% of their mass and an effective count of 1.061 out of 128. After removing
unweighted feature scaling, the largest and smallest local curvatures differ
by a factor 6.98 million. The d5 time-1 exact-target fit is a useful converged
counterexample: R passes the unchanged optimizer gate with sampled shape error
1.49e-5, yet KL is 1.946 versus a diagonal-family minimum 0.1822. Its effective
count is 1.834 of 256. A small sampled residual does not certify global guide
geometry. These are deterministic diagnostics on selected existing cases;
neither a population failure rate nor a stochastic ranking is estimated.

## Derivation and interpretation

Write the full design as F=[D,C], with D=[1,z,z²] and C containing all z_j z_k,
j<k. Let the full least-squares coefficients be (beta,gamma). Its residual is
orthogonal to D, so diagonal regression satisfies
beta_reduced=beta+QRsolve(D,C gamma). This also holds when the actual target is
not exactly quadratic. The projection term is a real change to the fitted
linear/quadratic coefficients. For a Gaussian target, recovering all quadratic
terms yields precision Lambda_jj=-2 beta_quadratic,j and Lambda_jk=-gamma_jk,
then V=Lambda^-1 and m=V beta_linear. All 32 recovered precisions were positive
definite; every full recovery's diagonal projection was inside the old box.

For an exact Gaussian N(m*,V*), the comparator N(m*,diag(V*)) minimizes
KL[exact || diagonal Gaussian]. It does not minimize the paper's Equation 15.
The actual recursive targets can include a Gaussian-plus-floor future message;
their full log-quadratic residual reaches 0.583, so recovering a positive
Gaussian approximation to them does not recover the exact future likelihood.

For p_i(theta)=b_i exp(phi_i' theta), normalize u=p/||p|| and v=b/||b||.
At theta=0, u=v and du_i=u_i(phi_i-E_pi phi), with pi_i=b_i²/sum_j b_j².
Since S=1-(u'v)², its Hessian is twice the Gram matrix of du:
H=2 Cov_pi(phi). Whitening phi=[z,z²] by its unweighted empirical covariance
removes arbitrary feature units. The reported Hessian is target-centered local
geometry; it is not the Hessian at a misspecified fitted diagonal Gaussian.
All three symmetric perturbation sizes and both extreme eigenvectors are saved.
The concentration is explanatory evidence. It does not by itself prove how
many particles would repair the fit, nor distinguish finite-cloud error from
the population density criterion's difference from KL.

## Conditional exact-target results

Time is zero-indexed. Effective count uses target-squared weights, not the
particle filter's importance-weight ESS. R and TF prior minima, actual-target
rows, complete coefficients and curvature directions are retained in CSV/RDS.

| Case | Time | N | Diagonal QR KL | Full recovery then diagonal KL | Target-squared effective count | Largest weight | Curvature ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| d2 moments | 0 | 128 | 0.23677 | 0.034332 | 4.3061 | 0.35421 | 873.81 |
| d2 moments | 1 | 128 | 0.079598 | 0.034356 | 42.006 | 0.041981 | 29.62 |
| d2 moments | 2 | 128 | 0.035209 | 0.034556 | 80.959 | 0.020103 | 3.4759 |
| d2 moments | 3 | 128 | 0.21042 | 0.03403 | 20.335 | 0.089778 | 22.305 |
| d2 QR | 0 | 128 | 0.07762 | 0.034332 | 36.12 | 0.047061 | 283.76 |
| d2 QR | 1 | 128 | 0.066523 | 0.034356 | 63.276 | 0.026426 | 38.513 |
| d2 QR | 2 | 128 | 0.039215 | 0.034556 | 86.869 | 0.017473 | 3.6727 |
| d2 QR | 3 | 128 | 0.076506 | 0.03403 | 70.701 | 0.024565 | 21.709 |
| d5 QR | 0 | 256 | 3.3646 | 0.18202 | 1.7696 | 0.74362 | 11563 |
| d5 QR | 1 | 256 | 1.3471 | 0.18222 | 1.834 | 0.65996 | 1.9943e+05 |
| d5 QR | 2 | 256 | 1.0025 | 0.18456 | 3.9249 | 0.4376 | 1842.8 |
| d5 QR | 3 | 256 | 0.54484 | 0.19794 | 9.3633 | 0.17343 | 375.98 |
| d10 QR | 0 | 128 | 8.5748 | 0.43808 | 1.9724 | 0.67125 | 9.7303e+06 |
| d10 QR | 1 | 128 | 4.6102 | 0.43839 | 1.061 | 0.97066 | 6.9827e+06 |
| d10 QR | 2 | 128 | 3.7954 | 0.44236 | 1.0777 | 0.963 | 86641 |
| d10 QR | 3 | 128 | 1.9229 | 0.4776 | 5.5761 | 0.32382 | 1.2147e+05 |

## Decision and inference status

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close representation diagnosis | Exact recovery, QR parity and omitted-term identity pass | No validity veto | Small selected clouds | Test population geometry on a controlled Gaussian design | Full quadratic paper-scale learner |
| Retain concentration as a supported mechanism | Hessian identity and directional checks pass | No validity veto | Local curvature is not the fitted Hessian | Exact integration and independent finite-cloud comparison | Particle count alone fixes quality |
| Keep fitted filter unpromoted | Prior conditional heuristic and guide-quality failures persist | Prior filtering vetoes remain | Fresh downstream accuracy untested | Evaluate a concrete repair after the objective/design distinction | Rejection of iAPF or KDM as methods |

| Inference status | Finding |
|---|---|
| Hard veto screen | All input, representation and curvature checks pass; prior candidate failures remain |
| Statistically supported ranking | None |
| Descriptive-only differences | KL, effective count, curvature and residuals on selected fixed cases |
| Default readiness | No runtime or default change |
| Next evidence needed | Population-versus-finite-cloud diagnosis, then fresh downstream tests of a nominated repair |

Terminal skeptical review PASS for this diagnosis. The strongest alternative
is that a diagonal family's population density optimum also differs from its
KL optimum; this phase has not equated them. Next tests must separate that
difference from finite sampled coverage. A full quadratic needs 3321 parameters
at d80, exceeding the paper's N=1000, so this diagnostic cannot be transferred
to that setting as an ordinary unconstrained regression. An invalid oracle,
input mismatch, or failed identity would overturn the result; all were checked.
The original-author choices and paper timing/controller gaps remain unresolved.

One CPU/base-R 4.1.2 launch used the unchanged 16 clouds with two targets each.
GPU devices were intentionally hidden, BLAS/OMP threads were one, and there
was no optimizer, new randomness, dependency change or production edit.
The manifest preserves commands, source/input hashes, versions, wall time,
budget and all conditional outputs. The accompanying verification is terminal.
