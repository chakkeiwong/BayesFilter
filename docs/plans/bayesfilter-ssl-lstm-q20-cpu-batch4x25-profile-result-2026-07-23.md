# q=20 CPU Batch-4-Per-Core Profile Result

Date: 2026-07-23
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-cpu-batch4x25-profile-plan-2026-07-23.md`
Artifact: `docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-profile-2026-07-23/r4/result.json`

## Result

The r4 profile completed with TensorFlow 2.20.0, CPU-only `float64`, non-XLA
execution, 25 persistent workers, one configured compute thread per worker,
and four rows per worker. The first call took 19.719 s. The five steady-state
calls took 13.009, 12.914, 13.820, 14.275, and 14.065 s, for a mean of
13.616 s. The four-process/batch-25 baseline mean was 27.256 s, giving an
observed ratio of 0.500.

All repeated value and score deltas from the first call were zero. Every
batch-100 call used all 25 persistent worker PIDs, and every worker's native
threads were bound to its assigned logical CPU. Maximum combined parent and
worker RSS was 18.364 GB, below the 64 GiB cap.

## Decision Table

| Decision item | Result |
| --- | --- |
| Primary topology screen | Passed: finite/parity-valid and descriptively faster than the four-process baseline |
| Hard vetoes | None fired |
| Main uncertainty | One host and five steady-state repetitions; timing differences have no inferential ranking support |
| Next justified action | Use the explicit `--cpu-processes 25 --batch-per-process 4` option for bounded CPU diagnostic training |
| Not concluded | No NeuTra quality, HMC, posterior, CPU-default, GPU-comparison, or scientific-validity claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for process identity, CPU affinity, finite outputs, exact repeat parity, and memory cap |
| Statistically supported ranking | None |
| Descriptive-only difference | 13.616 s candidate mean versus 27.256 s baseline mean |
| Default readiness | Optional CPU diagnostic topology only |
| Next evidence needed | A bounded training run if training behavior, rather than target-evaluation throughput, is the question |

The result is topology evidence only. It does not change the repository policy
that claim-bearing NeuTra training is a GPU workload.
