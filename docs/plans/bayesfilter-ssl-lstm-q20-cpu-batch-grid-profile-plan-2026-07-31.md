# q=20 CPU Worker/Shard Grid Profile Plan

Date: 2026-07-31
Status: `READY_FOR_BOUNDED_DIAGNOSTICS`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Question | How does warmed q=20 CPU target throughput change for `10 workers x 10 rows`, `15 workers x 15 rows`, `32 workers x 8 rows`, and the validation-scale `32 workers x 2 rows`? |
| Mechanism | Persistent batch-native TensorFlow CPU workers using the existing strict `tensorflow_eigh` route, one configured compute core per worker, pinned to distinct CPUs. |
| Exact comparator | Existing `25 workers x 4 rows` profile: 100 rows, warmed wall mean `14.364853534396389 s`. |
| Primary diagnostic | Warm wall time and rows/second for each topology, with worker skew, RSS, thread count, and finite/parity checks. |
| Hard vetoes | Visible GPU; worker/shard mismatch; CPU pinning failure; nonfinite/wrong-shaped values or scores; RSS above `64 GiB`; wall-cap exhaustion. |
| Explanatory only | Startup, first-call time, worker skew, serialization, RSS, native thread count, and speed ratio. |
| Must not be concluded | No CPU-vs-GPU superiority claim, no production/default change, no NeuTra quality, HMC, posterior, convergence, or scientific claim. The CPU route is not backend-identical to the current hybrid GPU route. |
| Artifacts | `docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/{10x10,15x15,32x8,32x2}/`. |

## Configurations

- q=`20`, `float64`, non-XLA CPU, CUDA hidden before TensorFlow import.
- `10x10`: 10 workers, 10 rows per worker, 100 total rows.
- `15x15`: 15 workers, 15 rows per worker, 225 total rows.
- `32x8`: 32 workers, 8 rows per worker, 256 total rows.
- `32x2`: 32 workers, 2 rows per worker, 64 total rows; compare directly with
  `32x8` to isolate shard-size behavior at the same worker count.
- Four calls per topology: one first call plus three warmed repeats on deterministic rows.
- Worker CPU IDs are `0..workers-1`; strict wall cap is `1,800 s` per process and combined RSS cap is `64 GiB`.

## Skeptical Pre-Execution Audit

- Wrong baseline: no. Both runs use the same target route, dtype, worker implementation, pinned-core policy, and four-row reference family; only worker and shard counts change.
- Proxy promotion: no. Runtime is diagnostic only.
- Hidden assumptions: neither linear scaling nor fixed per-row cost is assumed; worker skew and resource telemetry are recorded.
- Stop conditions: exact topology, finite values/scores, affinity, RSS, and wall cap are explicit.
- Artifact adequacy: each call is written to `progress.json`, and final results include command, source hashes, topology, and per-worker receipts.

Audit decision: `PASS_FOR_BOUNDED_DIAGNOSTICS`.

## Commands

```bash
timeout 1800 python docs/benchmarks/profile_ssl_lstm_q20_cpu_batch4x75_2026_07_31.py \
  --workers 10 --rows-per-worker 10 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/10x10 \
  --cap-seconds 1800

timeout 1800 python docs/benchmarks/profile_ssl_lstm_q20_cpu_batch4x75_2026_07_31.py \
  --workers 15 --rows-per-worker 15 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/15x15 \
  --cap-seconds 1800

timeout 1800 python docs/benchmarks/profile_ssl_lstm_q20_cpu_batch4x75_2026_07_31.py \
  --workers 32 --rows-per-worker 8 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/32x8 \
  --cap-seconds 1800

timeout 1800 python docs/benchmarks/profile_ssl_lstm_q20_cpu_batch4x75_2026_07_31.py \
  --workers 32 --rows-per-worker 2 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-batch-grid-profile-2026-07-31/32x2-standalone-r1 \
  --cap-seconds 1800
```
