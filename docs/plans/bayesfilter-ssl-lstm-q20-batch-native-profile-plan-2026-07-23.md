# q=20 Batch-Native CPU Profile Plan

Date: 2026-07-23  
Status: `COMPLETED_PARTIAL_PROFILE_BOTTLENECK_CLASSIFIED`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Question | Is the q=20 batch-native CPU bottleneck worker startup, per-shard graph tracing, first execution, or steady-state target evaluation? |
| Comparator | The exact `TFBatchValueScorePool` and `batch_native_complexity_posterior_target` route used by the terminal CPU attempt, with four workers and one CPU core per worker. |
| Primary measurements | Wall time for pool startup; each worker's first and repeated calls for shard sizes 2, 3, 16, and 25; parent serialization/aggregation time; worker runtime telemetry. |
| Hard vetoes | Visible GPU; process-tree native threads above 50; combined parent/worker RSS above 64 GiB; nonfinite or wrong-shaped values/scores; worker backend not `batch_native_value_score`; profile cap `1,800 s`. |
| Explanatory only | Absolute runtime, compile/steady-state ratios, worker skew, and graph cache behavior. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-batch-native-profile-2026-07-23/r1/`. |
| Nonclaims | No NeuTra quality, HMC readiness, posterior correctness, convergence, architecture ranking, or GPU/CPU ranking claim. |

## Execution Contract

- q=20; dtype `float64`; non-XLA CPU; CUDA hidden before TensorFlow import.
- Four persistent workers, one intra-op thread each, one inter-op thread each.
- Parent TensorFlow uses four intra-op and one inter-op thread.
- Affinity is restricted to CPUs `0-49`; realized `/proc` process-tree threads are audited after every call.
- The same deterministic proposal rows are reused for repeat calls, so target values and scores must match exactly within recorded floating-point tolerance.
- Profile-only route; no optimizer update, transport mutation, HMC, or package/environment change.

## Phase Measurements

1. Import and target-construction time in the parent.
2. Pool startup/readiness time and startup RSS/thread snapshot.
3. For shard sizes `2`, `3`, `16`, and `25`: first call, second call, and third call. Record parent serialization, worker runtime, result aggregation, and total wall time.
4. Repeat the size-25 call three additional times to estimate steady-state spread.
5. Run scalar-authority parity on two rows only as a diagnostic comparator; it cannot become a training route.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Four workers | Inherited from the revised CPU route | Worker startup dominates | Startup timing is isolated before calls |
| Shard sizes 2/3/16/25 | Derived from four-way partition of validation/support/training batches | Missing static graph cache entry | Each size is profiled explicitly |
| Three warm calls | Convenience diagnostic, not a quality threshold | Too few repetitions for noisy timing | Additional size-25 repeats are recorded |
| 1,800-second cap | Bounded diagnostic budget derived from the prior 1,360-second manual observation | Profile itself may be slow | Stop and preserve partial timing artifact |

## Skeptical Audit

- The comparator is the exact route implicated by `r4`; no scalar fallback is used.
- Runtime is explanatory only and cannot promote a transport.
- The profile distinguishes startup, first trace, and repeated calls rather than attributing all delay to the optimizer.
- All resource and backend gates are checked before interpreting timings.
- A failed profile diagnoses infrastructure/performance only; it does not reject NeuTra.

Audit decision: `PASS_FOR_BOUNDED_PROFILE`.

## Exact Command

```bash
timeout 1800 taskset -c 0-49 python \
  docs/benchmarks/profile_ssl_lstm_q20_batch_native_cpu_2026_07_23.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-batch-native-profile-2026-07-23/r1 \
  --cap-seconds 1800
```
