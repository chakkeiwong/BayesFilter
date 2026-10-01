# Symmetric final factor precision repair

Exact-input diagnosis04606--04610 identifies a final Cholesky-inverse
representation error: GPU precision columns differ from their transposes by
about5e-16 of matrix norm. Small entries fail the unchanged elementwise symmetry
guard. Both pre-anchor and repaired fitters fail; accepted CPU records match.
The actual covariances are finite/SPD with condition numbers near1.2e4, and
their inverse backward errors are below2e-17. This is distinct from the earlier
ill-conditioned precision cases, which must remain rejected with diagnostics.

For a symmetric positive-definite covariance C, the mathematical precision
P=C^-1 is symmetric. The projection S=(P+P^T)/2 is the nearest symmetric
matrix in Frobenius norm: writing P=S+A with A skew-symmetric gives orthogonal
symmetric/skew components. Applying this identity after the final Cholesky
solve removes its roundoff antisymmetry without changing the covariance or
the mathematical inverse. It introduces no ridge, clipped spectrum, threshold
relaxation or alternative estimator.

First run a diagnostic trial on exact saved before/after GPU covariances and
CPU covariances from04606--04609. Record raw inverse, symmetric inverse,
relative/elementwise skew, independent inverse error, residual and conditioning.
Include analytically constructed SPD, scaling and repeated-eigenvalue cases,
plus a clearly invalid/indefinite input. Compare graph/XLA on CPU/GPU and check
derivatives of the symmetric inverse against d(C^-1)=-C^-1(dC)C^-1. NumPy is
independent reference/inspection only. The runtime remains untouched until
the projection preserves the tested inverse accuracy and detects invalid inputs.

If the trial passes, apply the projection **only after the final fitted
Cholesky solve** in the shared factor program. L-BFGS loss/gradient and its
path remain unchanged; final reported prediction errors use the same returned
symmetric precision. Preserve the raw final solve in diagnostic test artifacts;
do not replace a rejected ill-conditioned case with a symmetrized acceptance.
The covariance, raw optimizer state, anchors, rank/condition caps and loading
domain checks stay unchanged. Existing nonfinite/condition gates and downstream
elementwise symmetry checks remain active. This plan does not authorize
symmetrizing arbitrary user-supplied matrices before validation.

Qualification must compare exact saved operands before/after on CPU/GPU, inspect
every fit's matrix and unchanged optimizer fields, retain old full-record
differences, and pass existing factor, stability and derivative regressions.
When the older GPU fit was rejected solely for roundoff asymmetry, report its
change in status explicitly; it is repaired behavior, not old-record equivalence.
Diagnostic trial success cannot establish full initializer or source admission.

Reserve at most12 workers and2400 combined CPU/GPU seconds from the existing
56CPU/52GPU-hour caps. The extra24CPU hours were already counted; remaining
before this unit is27.981186CPU/26.143528GPU hours. Each worker has a300-second
deadline, one numerical worker at a time, unique directories, trusted eligible
GPU with memory growth and explicit CPU references. At most two harness repairs
fit inside the allowance; no unchanged numerical retry. No packages, environment
mutation, HMC, training or live MacroFinance edit. The prior exact-input unit
closes through04610; no prior failed result is overwritten.

Skeptical review: symmetrization alone can hide an inaccurate solve. Preserve
raw matrices and enforce independent inverse/residual/conditioning tests; an
unexpected non-roundoff change or newly accepted ill-conditioned case vetoes
the candidate. The final projection can change downstream metrics near a gate,
so complete records and exact decisions must be inspected. A healthy inverse
does not imply the fitted statistical model is adequate or the unconverged
optimizer is converged. Cost observations are descriptive until repeated,
matched, uncontended measurements qualify them. No change to the old equivalence
tolerances or condition-number policy is authorized.

Trial04611/04612 passes CPU/GPU, including derivative and invalid-input checks.
Proceed with the single final-precision projection in the shared factor program.
The exact before records are04607GPU and04609CPU (clipped anchors already
repaired); use these preserved raw outputs as the immediate before authority.
Run `factor_precision_symmetry_fit_cpu/gpu` on the exact same saved04591 bytes,
then `factor_precision_symmetry_regression_cpu/gpu` and readback. These are all
inside the12-worker/2400-second allocation; no extra before optimizer replay
is necessary. Record raw-before versus returned-after correction, covariance
and optimizer-field equality, status changes and remaining complete-record
differences. Runtime changes must not affect the loss or inner optimizer.

The unit closes through04617 with seven workers and no failures. CPU/GPU
regressions and complete saved-fit readback pass; see the result note. This
qualification is specific to the final precision projection. Renew source
evidence before running the complete actual consumer; all strict full-record
differences and terminal master gates retain their existing requirements.
