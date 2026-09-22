# 72-core process-parallel canary attempt-06

Date: 2026-09-03  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`  
Machine-readable receipt: `docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/canary/attempt-06/canary_summary.json`

## Verdict

`PASS_72CORE_PROCESS_CANARY`

The repaired controller passed the mechanics canary in `585.254545083968`
seconds.  Screen, replicated selection, and scope-finalization barriers all
reached readiness, completed their declared records, and exited successfully.
The fixed-seed serial/process comparison passed for samples, target log
probabilities, log acceptance ratios, and target scores at the declared
tolerances.

The topology was exactly:

| Barrier | Workers | Cores/worker | Worker cores |
|---|---:|---:|---:|
| Screen | 8 | 4 | 32 |
| Selection | 2 | 8 | 16 |
| Scope finalization | 6 | 4 | 24 |

The preparation used GPU0 with memory growth.  Child workers reported
`CUDA_VISIBLE_DEVICES=-1`, no visible TensorFlow GPUs, XLA enabled, and
disjoint affinity sets.  The machine exposed 256 logical CPUs.  The run used
Python 3.13.13, TensorFlow 2.20.0, target signature
`9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, and
bridge backend `tensorflow_eigh_strict`.

## Repair coverage

This canary follows the localized cap-closeout repair.  The companion focused
suite (`9 passed`) exercises the typed deadline exception, partial task
coverage, non-finite JSON encoding, topology arithmetic, and bridge identity.
The canary itself remained under cap and therefore did not need to terminate a
live worker; it validates the repaired controller and artifact boundary in the
normal path.

## Evidence boundary

This is an engineering/mechanics fixture only.  It is not fresh tuning or
posterior evidence and does not establish whitening, mode discovery,
convergence, HMC readiness, sampler ranking, CPU-default status, or GPU
speedup.
