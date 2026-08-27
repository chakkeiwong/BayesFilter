# Phase 35R Repair and Refresh Note

| Attempt | Failure class | Repair | Result |
|---|---|---|---|
| pre-run audit | measure mismatch in initial Phase 35 | bind affine factor to optimizer's train-split weights | repaired before execution |
| identity arm, pre-version attempt | static-shape/dict diagnostic harness failures | preserve roots; repair eager moment diagnostic | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` in `identity-attempt3/` |
| affine arm, pre-version attempt | none | none | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` in `affine/` |
| final versioned identity arm | none | none | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` (`identity-final/`) |
| final versioned affine arm | none | none | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` (`affine-final/`) |

The initial Phase 35 full-bank factor is preserved under
`phase35-affine-neutra-repair/` but is not a valid whitening comparator for
the 40-row optimizer measure. The repaired receipt must include the exact
training-measure oracle, train/validation/audit separation, target/status
gates, GPU/XLA policy, source hashes, and unique roots. The final versioned
roots must bind `plan_version=v2.1-training-measure-bound` and the v2 runner
schema before Phase 36 can adjudicate.

Both final manifests bind the current continuation hash and runner hash. The
affine training-measure oracle residuals are `1.25e-16` (mean) and
`8.88e-16` (covariance). The affine arm's held-out residuals remain large at
step 200, so Phase 36 must preserve candidate non-promotion and decide whether
the next discriminating artifact is a particle-size/support ladder or a
target-specific objective/tuning repair.
