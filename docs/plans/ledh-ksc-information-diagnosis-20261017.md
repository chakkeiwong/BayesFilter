# KSC saved-fit information diagnosis

17 October 2026; branch ledh-refactor-with-policy-fix, worktree
ledh-iapf-execution-20260922. This follows the closed fitted-guide campaign,
not a third launch within that campaign. The owner's continued-execution
instruction covers this bounded local mechanism diagnosis.

## Research intent and skeptical pre-execution audit

Question: does the shared consumer's responsibility-averaged component
precision omit substantial mixture uncertainty in the saved KSC fits?
Mechanism: J=E_rho[S_k^-1], whereas -Hess(log psi)=J-Cov_rho(s_k).
For equal component variances, the consumer Gaussian-product covariance omits
the between-component posterior-mean variance of the exact Gaussian product.
Expected failure: an accurate kernel density can yield an overly concentrated
Gaussian proposal because its bandwidth, not the full density width, determines J.

Audit PASS for this diagnostic: use the existing guide and build_maps entry
points, not a replacement LEDH implementation. Compare with independently
derived mixture derivatives and exact Gaussian-product moments. The comparator
product uses the same current-observation Gaussian approximation as the global
map; it is NOT the exact filtering or smoothing distribution. Existing final
fits/cloud seeds may explain the completed experiment but cannot select a repair
or provide untouched confirmation. Inspect both global and local call chains.
Do not infer score improvement from a covariance difference or use an indefinite
Hessian as a silent replacement. No production source or scientific default changes.

## Evidence contract and diagnostic roles

- Engineering pass criterion: actual guide J/r equals independent mixture
  formulas, its directional score derivative equals the analytical log-mixture
  Hessian, global map covariance equals (S^-1+J)^-1, and the shared consumer
  calls the same guide at both global and local anchors. Relative tolerance
  1e-9, absolute scale floor 1, is a generous FP64 mechanics threshold, not
  a fitted scientific criterion. A failed identity is a continuation veto.
- Mathematical check: for homoscedastic mixtures, exact Gaussian-product
  variance equals consumer covariance plus between-component posterior-mean
  variance. A single component has zero gap. These are identities, not
  stochastic promotion claims.
- Explanatory only: omitted score variance, negative Hessian counts, ratios of
  exact-product variance to consumer covariance and differences from the exact
  seven-component future-message control. No score rankings or new confidence
  intervals are computed. These diagnostics cannot establish causation for the
  full finite-filter score error.
- Repair trigger: a material covariance discrepancy motivates an optional
  consumer design preserving mixture uncertainty; it does not justify another
  fit-capacity ladder. No numerics-altering repair is adopted here.
- Heuristic set: single Gaussian (zero-gap identity), moment-matched Gaussian
  message (cheap representation comparator), no guide (current Gaussian), and
  exact Gaussian product of the same mixture (local conditional oracle).
  Evaluate separately for ordinary, negative-future-tail and positive-future-tail
  conditions. This diagnostic has no promotion decision; previous conditional
  score-heuristic vetoes remain in force.
- Nonconclusions: canonical LEDH conformance, native-SV correctness, downstream
  score improvement, KDM-training benefit, optimal bandwidth, or default readiness.

## Frozen choices and budget

Use all 24 saved final kernel fits (three conditions x eight fitting clusters)
and each cluster's first already-recorded cloud. Pair the exact seven-component
future message on the same cloud. All 32 local anchors and one global anchor
are inspected; no subsampling by result. First-cloud selection is a convenience
for bounded diagnosis, not representativeness evidence. Its limitation is
failure to cover later clouds and filtering times; all-cluster coverage and the
algebraic identity are the early checks against an isolated-case explanation.

Reuse theta, observations, N=32 and controls from the frozen final manifest.
These are inherited comparison settings, not promoted defaults. Reuse the
manifest's seeds; no new stochastic fitting, tuning or score campaign occurs.
CPU-only FP64, CUDA_VISIBLE_DEVICES=-1 before TensorFlow import, explicit stable
tf.function signatures, no XLA for this debugging/reference exception. This
avoids claiming GPU/XLA equivalence. No pfor, autodiff score path or new NumPy
dependency. Test finite differences only as an independent derivative check.

Maximum two diagnostic attempts, 180 process-wall seconds total, timeout each
attempt at 90 seconds. Separate focused tests have a 90-second limit. Output:
docs/plans/artifacts/ledh-ksc-information-20261017/attempt-01 (fresh directory
per attempt). Stop on invalid saved artifacts, source drift, missing cells,
nonfinite accepted calculations, identity mismatch, or exhausted budget.
Preserve source/input hashes, actual command, environment, Git identity, CPU
choice, wall time, rows, conditional summaries and decision table. No new
inference may use the old final scores as an untouched holdout.

## Execution checklist

1. Add a diagnostic-only TensorFlow calculation and independent identity tests.
2. Run focused CPU mechanics and global/local call-chain tests.
3. Execute the saved-fit diagnosis once; repair only localized harness errors
   within the stated budget, preserving prior attempts.
4. Audit identities, refresh the manuscript/result/master/checkpoint, and build
   and inspect the changed manuscript pages. MathDev review is advisory and
   must not be represented as a proof of an unsupported scientific claim.
