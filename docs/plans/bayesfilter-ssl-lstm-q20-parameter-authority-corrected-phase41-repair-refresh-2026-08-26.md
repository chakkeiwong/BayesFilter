# Phase 41 Repair and Refresh Note

Date: 2026-08-26  
Continuation version: `v2.3-independent-audit-bank`

## Attempt ledger

| Attempt | Failure class | Evidence | Repair | Status |
|---|---|---|---|---|
| fresh CPU-hidden N=256 pilot | none | passing `theta_R4` C0/M0 receipt; target signature and M0/C0 protocol hashes match the frozen bank; all particle tensor hashes differ | continue to GPU audit | `PASS_THETA_MEASURE_PILOT` |
| GPU frozen-training wrapper, initial launch | harness contract defect | failed before optimizer construction because the wrapper required the M0 protocol hash for C0; C0 has a distinct arm-bound protocol hash by design | require the exact M0 hash and exact C0 hash separately, and compare old/fresh hashes within each arm | repaired |
| GPU frozen-training wrapper, attempt 2 | harness/static-shape defect | GPU memory growth and XLA initialized, but the shared compiled validation function relaxed its row dimension after the 12-row v2.2 holdout and rejected the fresh 256-row audit | keep validation traces shape-specific (`reduce_retracing=False`) and assert fresh audit shapes explicitly | repaired |

The focused regression `test_validation_accepts_distinct_static_partition_sizes`
passed together with the existing weighted-loss and invalid-shape checks
(`3 passed, 28 deselected`). This supports the harness repair only; it is not
scientific evidence about whitening or posterior quality.

The initial GPU launch did not consume fresh rows or perform training. No
scientific result was produced; its output root was not created and therefore
could not overwrite evidence. The failure is recorded here as a harness
diagnostic, not as evidence against the target or transport.

## Refresh decision

The corrected runner remains bound to the same target signature,
`theta_R4` measure, M0 protocol hash, C0 protocol hash, root-group split, and
200-step arm settings. The next launch uses a new output root
`phase41-independent-audit-bank/frozen-training-audit-attempt3/`. Fresh rows
remain post-training-only. No checkpoint selection, objective retuning, or
promotion criterion changes.

## Remaining gates

Require finite target/status, GPU memory growth before TensorFlow
initialization, XLA/batch-native training, exact affine training-measure
oracle, transport round trips, and explicit false fresh-use flags. A valid
but poor fresh audit remains a repair trigger; it is not an IID, posterior,
mode-discovery, HMC, or LEDH claim.
