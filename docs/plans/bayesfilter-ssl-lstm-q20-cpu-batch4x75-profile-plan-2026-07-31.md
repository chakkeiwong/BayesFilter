# q=20 CPU Batch-4-Per-Worker 75-Worker Profile Plan

Date: 2026-07-31
Status: `READY_FOR_BOUNDED_DIAGNOSTIC`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Question | Does a warmed CPU topology of 75 pinned workers, each evaluating four q=20 rows, complete a 300-row batch near the 25-worker reference time? |
| Mechanism | Persistent batch-native TensorFlow CPU workers using `tensorflow_eigh`, one configured compute core and four rows per worker. |
| Exact comparator | Existing 25-worker x 4-row profile: 100 rows, repeated wall mean `14.364853534396389 s`. |
| Primary diagnostic | Warm 300-row wall time, rows/second, worker-runtime maximum, and worker skew over three repeats. |
| Hard vetoes | Visible GPU; worker-count or shard mismatch; CPU pinning failure; nonfinite/wrong-shaped values or scores; RSS above `64 GiB`; cap exhaustion. |
| Explanatory only | Startup time, first-call time, worker skew, serialization overhead, RSS, native thread count, NUMA placement, and speed ratio. |
| Must not be concluded | No CPU-vs-GPU superiority claim, no production/default change, no NeuTra quality, HMC, posterior, convergence, or scientific claim. The CPU route and current hybrid GPU route are not backend-identical. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x75-profile-2026-07-31/r1/`. |

## Configuration

- q=`20`, `float64`, non-XLA CPU, CUDA hidden before TensorFlow import.
- `75` persistent workers, `1` configured compute core per worker, `4` rows per worker.
- Total batch=`300`; worker CPU IDs=`0..74`.
- Three warmed repeats after one first call on deterministic rows.
- Strict wall cap=`1,800 s`; combined parent/worker RSS cap=`64 GiB`.

## Skeptical Pre-Execution Audit

- Wrong baseline: no. The comparator is the same strict CPU batch-native route and same four-row shard size; only worker count and total rows change.
- Proxy promotion: no. Runtime is diagnostic only and cannot establish method quality or backend superiority.
- Hidden assumptions: perfect scaling from 25 to 75 workers is a hypothesis, not a result. The run records worker skew, RSS, native threads, CPU/socket placement, and first/warm separation.
- Missing stop conditions: initialization, pinning, finite values/scores, exact 75 x 4 partition, RSS, and wall cap are explicit vetoes.
- Artifact adequacy: every completed call is written to `progress.json`; the final result records command, environment, source hashes, topology, and all per-worker receipts.

Audit decision: `PASS_FOR_BOUNDED_DIAGNOSTIC`.

## Command

```bash
timeout 1800 python docs/benchmarks/profile_ssl_lstm_q20_cpu_batch4x75_2026_07_31.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x75-profile-2026-07-31/r1 \
  --cap-seconds 1800
```
