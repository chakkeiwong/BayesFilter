# q=20 CPU Batch-4-Per-Core Profile Plan

Date: 2026-07-23
Status: `EXECUTED_R4`

## Evidence Contract

| Role | Contract |
| --- | --- |
| Question | Does a separate CPU topology of 25 persistent processes, each assigned one CPU core and a batch of 4, materially reduce q=20 target value/score time for total batch 100? |
| Baseline | Existing four-process topology with batch 25 per worker: repeated mean 27.256 seconds. |
| Candidate | `batch4x25`: 25 persistent TensorFlow workers, one configured compute thread and one assigned logical CPU each, four proposal rows per worker. |
| Primary criterion | Finite/parity-valid repeated batch-100 evaluations with descriptive steady-state wall time materially below the 27.256-second baseline. |
| Hard vetoes | Visible GPU; missing/duplicate worker processes; worker not assigned to its declared CPU; configured compute cores above 50; RSS above 64 GiB; nonfinite/wrong-shaped values or scores; backend not batch-native; 1,800-second cap. |
| Explanatory | Native TensorFlow housekeeping thread count, startup time, worker skew, serialization overhead, memory, and speed ratio. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-profile-2026-07-23/r4/`. |
| Nonclaims | No NeuTra quality, HMC, convergence, posterior, default, CPU/GPU, or architecture claim. |

## Resource Semantics

`batch4x25` assigns each process, including all its TensorFlow native threads,
to one logical CPU. The configured compute-core count is therefore 25. The
native OS thread count will exceed 50 because TensorFlow creates housekeeping
threads in every process; it is recorded but is not represented as 200 compute
cores. This topology is separate from the literal-50-native-thread strict mode.

The shared pool exposes this topology through `batch_per_worker=4` and
`worker_cpu_ids`. CPU affinity is assigned once in each persistent worker
initializer before TensorFlow import, and each worker records its realized PID,
CPU, and thread-affinity snapshot. The q=20 launcher accepts
`--cpu-processes` and `--batch-per-process`; the explicit 100-row training
configuration is `--cpu-processes 25 --batch-per-process 4`. Omitting these
options preserves the historical four-worker path.

## Execution

- CPU-only; q=20; `float64`; non-XLA.
- Parent affinity restricted to CPUs `0-49`.
- Worker `i` and all its native threads bound to CPU `i`, for `i=0..24`.
- Total batch 100 split exactly into 25 shards of 4.
- One first call and five repeat calls on identical deterministic rows.
- Persist progress after every completed call.

## Skeptical Audit

- Same total batch, target, dtype, and proposal rows as the baseline question.
- No scalar or row-mapped target fallback is allowed.
- Timing is not a transport promotion criterion.
- Worker affinity is verified from `/proc`, not inferred from configuration.
- A faster result nominates this CPU topology only; it does not establish
  statistical or scientific superiority.

Audit decision: `PASS_FOR_BOUNDED_TOPOLOGY_PROFILE`.

## R4 Implementation And Checks

The persistent-affinity repair and launcher options were checked with six
focused process-parallel/strict-launcher tests, Python compilation, and
`git diff --check`. A fresh q=20 profile is recorded under
`docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-profile-2026-07-23/r4/`.
This remains timing/topology evidence only; it does not promote CPU NeuTra
training, HMC, posterior correctness, or a scientific default.

## Command

```bash
timeout 1800 taskset -c 0-49 python \
  docs/benchmarks/profile_ssl_lstm_q20_cpu_batch4x25_2026_07_23.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-profile-2026-07-23/r4 \
  --cap-seconds 1800
```
