# q=20 Direct GPU/XLA r2 Budget Preflight Reset Memo

Date: 2026-07-31
Status: `HOST_CALLBACK_ROOT_CAUSE_FOUND_GPU_NATIVE_LOCALIZATION_BLOCKED`

## Current State

The r2 hybrid-backend timing is complete. Do not resume either timing process
and do not launch tuning, final training, or HMC. The previous 20-day budget
must not be requested as a GPU-native campaign budget.

- repaired CPU/GPU target identity parity: passed;
- target signature: `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`;
- adapter signature: `c990a3a97d62a2557f0466a7ca1f0e009d5e35708156aeca1ce801257db48c73`;
- direct GPU/XLA timing: complete for `(32,32)` and `(64,64)`;
- exact static 256-row audit: complete and hard-valid;
- hybrid host-staged prior-protocol projection: `1,755,960.4640257251 s`;
- GPU-native campaign budget: `not checked`;
- current r2 authorization unused: `2,152.4224067549985 s`;
- tuning selection, final training, and HMC: not run.

## Root Cause

The trainer graph was XLA-compiled, but the active
`compiled_custom_op` backend was not GPU-native. Its CUDA callbacks:

1. call `cudaDeviceSynchronize()`;
2. copy covariance/factor/RHS tensors from device to host;
3. loop serially over batch rows with Eigen eigensolvers;
4. copy outputs from host to device; and
5. synchronize the stream.

The q=20 forward-sensitivity score repeats this inside a sequential 30-step
filter recursion. The measured `~568 s` warm update and 20-day projection are
valid only for that exact hybrid implementation.

An existing `tensorflow_eigh_strict` implementation is the smallest repair
hypothesis. It keeps strict eig/Sylvester tensor algebra in TensorFlow/XLA.
It has not yet passed q=20 trusted-GPU parity or timing.

## Authority Boundary

The completed hybrid projection is historical implementation-cost evidence
only. Do not authorize a serious campaign until a GPU-native backend passes
parity and a new timing preflight establishes its budget. The unused current
allowance cannot be treated as a first installment on an arm.

## Canonical Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Identity comparison | `5f2e01964659aba51f4a7cabf81b443c1ee533808a23d0696f07b5ef8810a688` |
| Preserved `(32,32)` progress | `7815f618e6d7ac96dd23b73b1c9d46cb19d5b2b2822d5950b4c73c53306644b2` |
| Completed `(64,64)` result | `d5fbd064ccba228458c23c4a46c2682e32521d3b56b18ef9fe69a7834f3ec7d7` |
| Projection | `ec90eabae6df92abe10c56246d8254c8e2421f3092e93c062bd17ec6e269f0ec` |

Result note:
`docs/plans/bayesfilter-ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-result-2026-07-31.md`.

Original plan:
`docs/plans/bayesfilter-ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-plan-2026-07-30.md`.

Recovery plan:
`docs/plans/bayesfilter-ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-recovery-plan-2026-07-30.md`.

## Failure And Repair Ledger

1. The first r2 timing process completed all `(32,32)` update, validation,
   status, and support receipts, then failed when the shared validation wrapper
   generalized the leading batch dimension before a 256-row target call.
2. The failure invalidated that audit operation, not the target, completed
   receipts, or candidate. No continuation veto against NeuTra fired.
3. A fresh recovery root reissued exact CPU/GPU parity after the source change.
4. A dedicated XLA function with exact `[256,4]` input signature completed the
   audit in `2,926.5343564180075 s` with finite/status-valid rows and zero floors.
5. The recovery process completed with verified memory growth and no GPU
   preallocation violation.

## Next Justified Action

When trusted GPU launch is available:

1. execute the bounded localization plan
   `docs/plans/bayesfilter-ssl-lstm-q20-gpu-native-eigh-localization-plan-2026-07-31.md`;
2. compare custom and `tensorflow_eigh_strict` value/score/status on identical
   q=20 rows before timing any update;
3. if parity passes, measure first and warm batch-100 XLA updates and extract
   HLO;
4. if parity fails, preserve the artifact and implement/audit a real CUDA
   solver rather than relaxing the target; and
5. derive a fresh campaign budget only after the repair timing is valid.

Two escalated localization launches were rejected because automatic permission
review timed out before process creation. One non-escalated diagnostic failed
closed because the sandbox exposed no physical GPU. These are launch-boundary
results only, not backend failures.

## Nonclaims

Both architectures are mechanics-viable only. No statistically supported
ranking, tuning selection, training-quality result, posterior result, HMC
readiness, production/default readiness, or scientific conclusion exists.
