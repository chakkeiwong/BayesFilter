# Phase 7 Repair and Refresh Note

Status: `REPAIR_TRIGGERED_MODE_AXIS_ALIGNMENT`

## Attempt 1 harness repair

The first launch stopped before TensorFlow import because the supplied Phase 6
parent directory did not itself contain `pilot.json`; the receipt is one level
below in `seed-1701/`. No training metric was produced and no scientific gate
was evaluated. The output directory was preserved as a failed launch. The
rerun uses the exact `seed-1701` receipt directory and a new output root, with
all target, split, seed, and promotion criteria unchanged.

## Bank 1701 receipt

The corrected rerun completed with `PASS_CANDIDATE_ROLE_LIMITED`. The frozen
split was train/validation/audit `180/60/60`, and both architecture arms used
batch-native 20-update training. GPU memory growth was verified on both visible
RTX 4080 SUPER devices before logical-device use; XLA and TF32 were enabled.
Forward/inverse parity was finite with round-trip residuals at most
`8.88e-16` and logdet residual `1.39e-17`; all 60 untouched audit rows had
finite target value/score and valid status. The compact arm was selected by
validation loss (`18.1832` versus `18.8586`), with descriptive latent
`max|mean|=0.5446` and max off-diagonal covariance `1.1234`. No ranking or
whitening claim is made.

## Bank 1801 receipt

The second corrected bank completed with `PASS_CANDIDATE_ROLE_LIMITED` and the
same `180/60/60` frozen split. GPU memory growth, XLA, TF32, batch-native
updates, parity, and all 60 untouched audit target/status checks passed. The
compact arm was selected descriptively (`loss=16.8566` versus `17.4573`);
its validation latent `max|mean|=1.4673` and max off-diagonal covariance
`1.0693` remain explanatory and are not IID-Gaussian evidence.

This note is completed after each Phase 7 bank. Classify failures in this
order: launcher/GPU policy, input receipt/hash, split or weight alignment,
target/status, transport parity, batch-native training, tuning, and only then
candidate quality. A poor latent whitening diagnostic or validation loss is an
explanatory repair trigger, not a continuation veto.

## Required receipt fields

Record the bank root, output root, command, git commit/dirty state, Python and
TensorFlow versions, GPU devices, memory-growth verification, XLA/TF32 state,
seed, steps, split counts, exact tensor hashes, parity residuals, target/status
checks, wall time, and whether HMC was launched (must be false).

## Repair rules

- A launcher or memory-policy failure: preserve the failure artifact, repair
  initialization order or trusted launch, rerun the same bank, and do not
  interpret training metrics.
- A row/weight/hash failure: stop downstream interpretation, repair the
  partition or receipt alignment, and rerun the same bank from its immutable
  Phase 6 input.
- A target/status or parity failure: classify the affected component and run a
  focused target/transport diagnostic before changing architecture settings.
- A batch-native or audit-leakage failure: reject the attempt and repair the
  harness; no result can support a NeuTra claim until it passes.
- A finite hard-gate pass with poor whitening: retain the candidate result,
  label the metric descriptive, and refresh a target-specific tuning or
  representation diagnostic. Do not call it IID Gaussian or HMC-ready.

## Refresh rule

Once the three bank attempts are complete, write the Phase 7 result and update
the master program. The next subplan must state whether the remaining issue is
an input-authority gap, a training/tuning gap, or a true target/transport
contradiction. HMC remains deferred until a separate plan satisfies the
canonical sequential HMC policy and an admitted target/transport measure.

## Bank 1901 and phase disposition

Bank 1901 completed with the same hard receipts as the first two banks. The
compact arm was descriptively selected (`loss=12.7963` versus `13.1670`), with
latent `max|mean|=0.9307` and max off-diagonal covariance `1.2009`. The Phase 7
result records all three banks and explicitly leaves whitening, authority,
posterior, and HMC claims open. Because the upstream Phase 6 audit found an
acceptance-receipt denominator bug after these banks were generated, this phase
is retained as role-limited evidence and is not used to promote or rank the
mutation branch. The next refresh is the corrected Phase 6 receipt audit.

## Post-phase mode-label audit

The pilot declares the signed mode coordinate as `theta[:, 2]`, while the
original NeuTra screen partitioned audit rows using the last coordinate. A
guarded CPU diagnostic on the N=300 bank found `47%` label disagreement
(`0.43` negative by the pilot axis versus `0.48` by the last axis). The prior
NeuTra runs remain valid transport/status smoke artifacts, but their
mode-stratified split is not valid for a mode-balanced training claim. The
runner is repaired to use `MODE_AXIS = 2`; after the upstream Phase 6 receipt
repair completes, Phase 7 must be rerun on fresh output roots before any
whitening or partition interpretation. The unguarded TensorFlow probe that
preceded the guarded diagnostic is launch-invalid under the GPU policy and is
excluded from all evidence.
