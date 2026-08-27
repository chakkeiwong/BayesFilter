# Corrected Parameter-Authority Phase 35 Result

Date: 2026-08-25  
Status: `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED_AFFINE_REPAIR_REJECTED_FOR_WHITENING`

## Receipts

- GPU/XLA affine screen: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35-affine-neutra-repair/attempt1/`
- independent weighted affine oracle: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35-affine-neutra-repair/affine-oracle/`
- MathDevMCP raw audit: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35-affine-neutra-repair/mathdevmcp-affine.txt`

The target remains `theta in R^4`. The 60-dimensional UKF state and the
20-dimensional innovation remain internal target-evaluation quantities; no
internal state was passed to the transport.

## Hard-gate evidence

| Check | Result |
|---|---:|
| weighted affine covariance positive definite | `true` |
| affine covariance condition estimate | `49.72885` |
| affine eigenvalues | `0.28113, 1.44119, 6.46531, 13.98017` |
| theta/chart round-trip residual | `8.88e-16` |
| compact transport round-trip residual | `6.66e-16` |
| wide transport round-trip residual | `4.44e-16` |
| determinant round-trip | `4.44e-16` compact; `2.22e-16` wide |
| target/status audit | `12/12` for both arms |
| batch size / XLA | `40 / true` |
| GPU memory growth | verified on both RTX 4080 SUPER devices |
| HMC launched | `false` |

Both learned arms returned `PASS_NEUTRA_BOUNDARY_CANDIDATE` and all hard
engineering gates passed. The affine density composition recorded by the
runner is

`log q_theta(theta) = log q_chart(L^-1(theta-m)) - log|det L|`.

The MathDevMCP expression audit returned `inconclusive` because its generic
parser could not encode the matrix/domain assumptions. It is retained as a
limitation, not as a proof certificate. The executable round-trip and the
independent oracle are the bounded numerical checks for this phase.

## Independent affine oracle

The CPU-hidden oracle applied the same weighted Cholesky map directly to all
64 M0 rows. It passed finite input, positive-definite covariance, exact
weighted mean, and exact weighted covariance gates. The maximum weighted mean
residual was `1.25e-16`; the maximum covariance residual was `7.77e-16`.
This validates the finite affine construction only. It does not identify the
target density or establish an IID law.

## Learned-trace evidence

| Arm | Step | Loss | Validation latent mean max | Validation covariance off-diagonal max |
|---|---:|---:|---:|---:|
| compact | 1 | `5.40454` | `0.55534` | `0.39748` |
| compact | 200 | `3.73135` | `1.23934` | `0.76628` |
| wide_low_lr | 1 | `5.40454` | `0.55379` | `0.39669` |
| wide_low_lr | 200 | `4.21329` | `1.22865` | `0.50048` |

For comparison, the identity 200-step trace ended at mean residuals
`0.37773` (compact) and `0.41820` (wide), with covariance off-diagonal
residuals `0.49573` and `0.47097`. Thus the affine map itself is exact, but
the learned updates moved away from the finite-bank whitening point during
this one-seed run. This is a candidate/training limitation, not evidence that
the target or all NeuTra transports are invalid.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain affine chart as a bounded conditioning diagnostic | finite map, density composition, target/status, batch/XLA, and GPU gates pass | no engineering veto | one bank, one seed, fixed normalized empirical weights; validation drift | adjudicate the corrected continuation and require a fresh tuned/held-out plan for any claim-bearing training | no IID whitening, posterior correctness, mode discovery, authority, HMC convergence, ranking, or default |
| Do not promote the learned affine arm for whitening | validation moment residuals worsen by step 200 | whitening promotion veto only | no multi-seed tuned comparison | keep identity/affine evidence role-limited | no rejection of all affine or NeuTra methods |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for both learned arms and the affine oracle |
| Statistically supported ranking | none; one seed and no predeclared uncertainty analysis |
| Descriptive-only differences | loss, latent moments, covariance residuals, condition number, and runtime |
| Default-readiness | not ready |
| Next evidence needed | disjoint calibration/validation/claim data, scope-specific tuning, fresh seeds, and downstream posterior checks |

## Red-team note

The strongest alternative explanation is optimization overfit to a finite,
normalized M0 empirical bank: the exact affine chart starts with the desired
first two moments, while the learned flow is not selected by an untouched
whitening or target-density criterion. The result would be overturned for the
current role decision only by a predeclared held-out, multi-seed run showing
finite target-density composition and stable downstream posterior evidence.
The weakest evidence remains the short one-bank training comparison.

## Budget and provenance

The GPU attempt consumed `36.8839 s`; the oracle consumed `0.3123 s`.
The result manifests record the command, Python/TF versions, git commit and
dirty state, seeds, device policy, XLA setting, source hashes, and artifact
paths. All output roots are unique and no prior artifact was overwritten.

