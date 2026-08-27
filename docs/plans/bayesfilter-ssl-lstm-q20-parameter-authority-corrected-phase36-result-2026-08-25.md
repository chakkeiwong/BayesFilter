# Corrected Parameter-Authority Phase 36 Adjudication

Date: 2026-08-25  
Status: `PASS_ADJUDICATION_CONTINUE_PHASE37_SUPPORT_LADDER`

## Scope and corrected version

This adjudication uses continuation version `v2.1-training-measure-bound` and
the final versioned Phase 35R receipts:

- identity: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35r-affine-training-measure-repair/identity-final/`
- affine: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35r-affine-training-measure-repair/affine-final/`
- MathDevMCP: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35r-affine-training-measure-repair/mathdevmcp-affine-v21.txt`

Both manifests bind the same current plan hash and runner hash. The declared
target is `theta in R^4`; the 60D UKF state and 20D innovation are internal
target-evaluation quantities only.

## Skeptical audit

The initial Phase 35 full-bank affine factor was correctly identified as a
measure mismatch and superseded. Phase 35R computes the factor from the exact
40-row normalized, floored training weights used by the optimizer and gates the
finite oracle. The identity and affine arms use the same frozen split, target,
seed, 200-step budget, batch-native GPU/XLA path, and untouched target audit.
No stale artifact is used for the current comparison.

The affine map is therefore valid as a finite conditioning map. It does not
identify a target density, an IID law, or a posterior sampler. MathDevMCP
returned `inconclusive` because its generic symbolic route cannot encode the
matrix/domain identity; the raw output is preserved and not treated as a
certificate.

## Evidence ledgers

### Engineering correctness

| Check | Identity | Affine |
|---|---:|---:|
| GPU memory growth on both visible GPUs | pass | pass |
| XLA and batch size 40 | pass | pass |
| finite gradients/traces | pass | pass |
| target/status audit | 12/12 | 12/12 |
| transport round-trip | `6.66e-16` | `8.88e-16` |
| logdet round-trip | `2.22e-16` | `4.44e-16` |
| train-measure affine oracle | identity N/A | mean `1.25e-16`, covariance `8.88e-16` |
| HMC launched | no | no |

Engineering status: both role-limited boundary candidates pass.

### Numerical validity

The affine covariance is positive definite with condition estimate `44.2356`.
The recorded composition is
`log q_theta=log q_chart-log_abs_det_chol`. The exact finite train-measure
oracle passes. These are finite numerical identities, not general proofs of a
learned density.

### Scientific interpretation

At step 200, the identity arm's validation mean/off-diagonal maxima were
`0.3746/0.5076` (compact) and `0.4882/0.4679` (wide). The affine arm's were
`1.1418/0.7848` and `1.2218/0.6622`. The affine arm had lower training losses
(`4.2329`, `4.3680` versus `6.9267`, `7.3816`), but this one-seed loss
difference is descriptive only and does not establish a ranking. Both arms
remain far from an IID-Gaussian held-out diagnostic.

The exact affine map initially whitens the training measure, while the learned
updates move away from that point. This weakens “mere linear conditioning” as a
complete explanation for the persistent held-out residuals, but it does not
reject affine maps or NeuTra generally.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue the corrected parameter-space investigation | target/common theta measure, artifacts, and hard gates valid | no continuation veto | finite bank may lack support; one seed and fixed empirical objective | run Phase 37 particle-size/support ladder with predeclared protocol | no posterior, IID whitening, exhaustive modes, HMC, canonical LEDH, or default |
| Retain identity and affine as role-limited candidates | both final arms pass engineering boundary | whitening promotion veto remains open | held-out residuals and objective mismatch | compare support/particle-size before changing objective | no superiority or universal affine benefit |
| Keep unmodified GenUT unpromoted | Phase 30 negative central weights | candidate veto only | alternative quadrature designs untested | revisit only under a separate reviewed design | no theorem that all GenUT fails |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | passed for current final receipts |
| Statistically supported ranking | none; one paired seed, no uncertainty analysis |
| Descriptive-only differences | losses, train/validation moments, condition, ESS, mode fractions |
| Default-readiness | not ready |
| Next evidence needed | particle-size/support ladder, disjoint tuning/claim data, and downstream posterior checks |

## Strongest alternative and overturning evidence

Strongest alternative explanation: the 64-row M0 empirical bank and its fixed
normalized weights do not represent enough target support for a learned flow to
generalize from train to held-out/audit rows. A competing explanation is that
the weighted forward-KL objective and current capacity/step schedule are not
aligned with the desired transport criterion.

The support explanation would be weakened by stable held-out residuals and
target-density diagnostics across larger fresh theta banks under the same
protocol. The objective explanation would be weakened by a tuned, disjoint
objective/architecture run on a bank whose support diagnostics pass. Neither
has been tested yet.

## Stop classification

The result does not invalidate the harness, target, or corrected measure. It
found and repaired one implementation/plan mismatch, then produced valid
role-limited evidence. Poor whitening is a promotion veto and repair trigger,
not a continuation veto. Continue to Phase 37 unless the support ladder finds
that a common-support theta proposal cannot be constructed or a required
artifact becomes invalid.

