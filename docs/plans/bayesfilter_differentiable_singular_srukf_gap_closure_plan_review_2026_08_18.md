# Review of Differentiable Singular SR-UKF Gap-Closure Plan

Date: 2026-08-18  
Reviewed plan: `bayesfilter_differentiable_singular_srukf_gap_closure_plan_2026_08_18.md`

## Verdict

`PROCEED_WITH_BOUNDED_SCOPE`

The plan is mathematically coherent for fixed-rank, fixed-chart branches and
correctly refuses to promise a classical score across rank/support changes.
The block-QR identities and the renormalized `epsilon -> 0` support likelihood
are dimensionally consistent. The plan preserves the repository's distinction
between direct-stack rank diagnostics and admitted differentiable runtime.

## Required constraints confirmed

1. No covariance-to-factor decomposition is introduced in the temporal runtime.
2. SVD remains rank discovery/value-only unless a separate fixed-support proof
   is added.
3. The ambient Gaussian density is not conflated with the finite affine-support
   limit.
4. A positive pivot/chart policy is bound before tracing and is not selected
   dynamically inside XLA.
5. Score claims stop at rank, support, chart, pivot, sign, or angular branch
   events.
6. Signed sigma-point weights remain an explicit separate boundary.
7. Existing historical artifacts remain readable and are not silently upgraded.

## Risks to monitor during execution

- The existing DZ5 route has nonnegative covariance weights; the rectangular
  score route must not accidentally broaden that claim to signed UKF rules.
- A rectangular QR factor is a chart object. Its metadata must include the
  pivot permutation and retained rank so a caller cannot self-attest identity.
- The support likelihood derivative must include support-coordinate changes and
  the innovation derivative, not only the triangular log-determinant term.
- Epsilon-limit tests must compare the renormalized ambient value, not the raw
  divergent ambient log density.
- GPU evidence must record the physical selection (`3`, then `2`, `1`, `0`),
  memory growth, XLA, dtype, seed, and artifact path.

## Acceptance gate

Proceed only if the implementation and tests preserve all constraints above,
the canonical chapters compile without unresolved references, and every new
artifact is versioned with checksums. Failure to implement a mathematically
valid fixed-support score is a scientific blocker to record, not a reason to
relax the branch policy.

## Execution-stage review addendum

The first rank-one implementation passed its finite-difference tests but was
not sufficient evidence for the matrix orientations. A non-diagonal rank-two
authority was therefore added. It exposed and repaired three defects before
release: the conditional right solve used `R^{-T}` instead of `R^{-1}`, the
column-space projector contracted different retained-rank indices, and two
support-coordinate derivatives used independent observation indices instead
of dot products. The corrected rank-two gain, conditional covariance, and mean
increment agree with the dense covariance authority to machine precision; its
likelihood, mean-increment, and posterior-factor derivatives pass centered
finite differences. This test is now mandatory release evidence.

The execution remains within the reviewed boundary. The fixed-score source
closure has no SVD/eigendecomposition or covariance-to-factor call. Direct
stack SVD remains an explicitly value-only discovery route. No score is
claimed across rank, chart, pivot, sign, support, or signed-weight boundaries.
