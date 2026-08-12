# q=20 CPU Full Optimizer Update `25x4` Benchmark Plan

Date: 2026-08-01
Status: `READY_FOR_BOUNDED_XLA_COMPARISON`

## Research Intent

Measure the complete CPU q=20 SSL-LSTM NeuTra optimizer update for a 100-row
training batch using `25` persistent workers with `4` target rows per worker.
The result will be compared descriptively with the existing CPU and hybrid GPU
full-update receipts.

After the non-XLA baseline completed, the user requested an otherwise matched
CPU XLA run. The XLA comparator keeps the baseline artifact immutable and
changes only `jit_compile` from false to true in both the 25 worker target
graphs and the parent transport/optimizer graph.

## Evidence Contract

| Item | Contract |
| --- | --- |
| Target | q=20 status-bearing batch-native target, `principal_sqrt_backend="tensorflow_eigh"`, CPU-only, `float64`; baseline non-XLA and explicit XLA comparator |
| Training batch | 100 rows, partitioned exactly as `25x4` |
| Complete update | transport forward/log-determinant, pooled target value/score, external-value/score gradient calculation, and optimizer parameter update |
| Primary timing | first and three warm complete-update wall times, with component timings |
| Hard vetoes | visible GPU, wrong worker/shard partition, pinning failure, nonfinite values/scores/loss/gradients, RSS over 64 GiB, incomplete worker execution |
| Explanatory diagnostics | startup, worker skew, serialization/parent overhead, RSS, native threads, component timings |
| Nonclaims | no CPU default, GPU superiority, NeuTra quality, convergence, HMC, posterior, or scientific claim |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/r1/` |
| XLA comparator artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/xla-r1/` |
| XLA `50x2` artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/xla-50x2-r1/` |

## Skeptical Pre-Execution Audit

- Baseline: the comparator is the existing CPU batch-native target/trainer
  route; the requested change is only the explicit `25x4` topology.
- Proxy risk: target-only timings are insufficient, so this benchmark includes
  the transport and optimizer update in the measured total.
- Hidden assumption: the `25x4` target result is not assumed to scale linearly
  from the standalone grid profile; all components are measured directly.
- Stop conditions: finite outputs, exact partition, pinning, RSS, and a
  bounded four-call run are enforced.
- Artifact adequacy: the manifest records source hashes, command, environment,
  seeds, hardware visibility, timings, and output paths.

Audit decision: `PASS_FOR_BOUNDED_DIAGNOSTIC`.

### XLA Comparator Audit

- The non-XLA baseline is the exact comparator; target, batch, topology,
  parameters, rows, seeds, dtype, and backend remain fixed.
- First-call compilation is not treated as warm throughput. The artifact
  preserves first and three warm calls separately.
- Failure to compile is an implementation/backend result, not evidence against
  CPU training or NeuTra.
- The same finite, partition, pinning, RSS, and wall-cap vetoes apply.

Audit decision: `PASS_FOR_BOUNDED_XLA_COMPARISON`.

### XLA `50x2` Comparator Audit

- The 100-row batch and all model/training settings remain fixed; only the
  worker/shard topology changes from `25x4` to `50x2`.
- The configured CPU count is exactly 50 and the benchmark fails closed if
  requested CPU IDs are unavailable.
- XLA first-call compilation remains separate from the three warm updates.
- The same finite, partition, pinning, RSS, and worker-completeness vetoes
  apply.

Audit decision: `PASS_FOR_BOUNDED_XLA_50X2_COMPARISON`.

## Planned Command

```bash
timeout 1800 python \
  docs/benchmarks/profile_ssl_lstm_q20_cpu_full_update_25x4_2026_08_01.py \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/r1 \
  --cap-seconds 1800

timeout 1800 python \
  docs/benchmarks/profile_ssl_lstm_q20_cpu_full_update_25x4_2026_08_01.py \
  --jit-compile \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/xla-r1 \
  --cap-seconds 1800

timeout 1800 python \
  docs/benchmarks/profile_ssl_lstm_q20_cpu_full_update_25x4_2026_08_01.py \
  --jit-compile --workers 50 --rows-per-worker 2 \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-full-update-25x4-2026-08-01/xla-50x2-r1 \
  --cap-seconds 1800
```
