# Lorenz centered backward-recursion repair

Attempt29 passes every downstream comparison against independent SVD: maximum
path difference2.22e-12, weight difference9.34e-10, prefix logZ difference5.39e-10,
identical ancestry, valid fresh forward runs and one trace. Original-coordinate
linear coefficients still fail the unchanged tolerance by a maximum factor2.19.
The failure remains recorded; solver promotion has not occurred.

The remaining numerical hypothesis is cancellation in the backward recursion.
Both solvers form log M(psi_new)-log M(psi_old) from large completed-square
terms, then fit concentrated clouds. Small target errors become larger physical
coordinate coefficient errors. Centering the design alone does not fix this.

Use the exact identity, for delta=psi_new/psi_old:

    log M(psi_new)-log M(psi_old) = log M^psi_old(delta).

The left side is the logarithm of a ratio of Gaussian integrals. The right
side integrates only the new quadratic correction under the already twisted
Gaussian. If z=(x-mu)/s is the fitted coordinate system, the Gaussian has
mean (m_old-mu)/s and covariance diag(1/s) K_old diag(1/s). Preserve the fitted
quadratic coefficients in z for this integration. This avoids reconstructing
and subtracting two large log integrals. The final physical coefficients are
still returned to the same forward consumer. The least-squares target and
controlled filter are mathematically unchanged; there is no ridge or clipping.

Center the regression response as well, restoring its mean through the constant
basis coefficient. With an intercept this is the same least-squares problem.
The independent CPU SVD reference uses covariance transformation; the TF
implementation uses precision congruence. This gives different numerical
implementations of the same coordinate change. Both retain full rank.

Before accepting the change, require the previous coefficient and complete
downstream tolerances unchanged (1e-7 absolute +1e-9 relative; exact ancestry).
Also require a well-conditioned analytic identity check of the Gaussian ratio,
and rejection of invalid precision. Preserve healthy exact-polynomial and
source-forward evidence; raw source normal equations remain unchanged and are
not a fallback. Extra batch dimensions must pass the same Gaussian identity.
The actual fit-to-forward endpoints are the claim-bearing check, not only the
presence of a new helper. Fresh forward streams and single-trace checks remain.

Question, comparator, promotion scope, vetoes and nonclaims otherwise follow
controlled-smc-lorenz-solver-validation-20260929.md. This is a representation
repair, not a new fitting objective or relaxed promotion criterion. No likelihood
ranking, empirical replication, general default or HMC claim follows.

Skeptical audit PASS: the ratio identity is exact, target and thresholds stay
fixed, and the independent numerical implementations plus downstream test can
falsify the repair. Failure is a repair trigger until the bounded budget is
exhausted; source raw-coordinate coefficient parity remains a separate unmet
source-numerics claim.

Add c1_lorenz_centered_reference (60 CPU seconds per attempt,120 aggregate,2
attempts) and c2_lorenz_centered_validation (120 GPU per attempt,240 aggregate,2
attempts), using versioned source snapshots of the reference and validation
scripts. Raise the total attempt ceiling from30 to34 only for this localized
repair; total allocation remains CPU300/GPU18000 and inherited balances remain
binding. Previous failed attempts remain charged. Update the master and result
afterward with the observed outcome, even if the repair fails.

Attempt31 failed before numerical checks because LogMatrixDeterminant (slogdet)
is unsupported by the installed XLA GPU backend. Replace only that SPD
log-determinant with twice the sum of log Cholesky diagonals; identical target,
no fallback or threshold change. The existing identity and precision-rejection
checks verify this localized compatibility repair in attempt32. The failed
5.976186633 GPU seconds remain charged.
