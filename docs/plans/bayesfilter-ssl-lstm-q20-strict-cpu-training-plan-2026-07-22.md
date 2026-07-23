# q=20 SSL-LSTM Strict CPU NeuTra Training Plan

Date: 2026-07-22  
Tier: bounded serious diagnostic CPU campaign  
Status: `TERMINAL_PERFORMANCE_BLOCKER_RECORDED`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Main question | Can both independent q=20 `(32,32)` NeuTra training streams complete the existing adaptive batch-100 procedure under a literal process-tree limit of 50 native OS threads? |
| Exact inputs | Existing q=20 synthetic target, fixed hyperparameters `learning_rate=0.0004`, `initialization_scale=0.01`, `gradient_clip_norm=10`, and existing independent seed-a/seed-b definitions. |
| Candidate mechanism | Existing adaptive NeuTra procedure, executed by a standalone batch-native CPU runner and TensorFlow-only target boundary. |
| Primary completion criterion | Each stream reaches its existing plateau stop or 2,000-step maximum, writes checkpoints and final audit, and remains finite under the strict thread cap. |
| Hard vetoes | Visible GPU; process-tree native threads above 50; nonfinite target/score/loss/gradient/support/audit; host memory above 64 GiB; corrupted or incomplete artifact; `13,500 s` campaign cap. |
| Explanatory only | Loss trajectory, saturation telemetry, learning-rate reductions, runtime, and differences between two seeds. |
| Artifact | Attempts preserved under `docs/plans/artifacts/ssl-lstm-q20-strict-cpu-training-2026-07-22/`; terminal attempt `r4`. |
| Nonclaims | CPU results cannot establish transport promotion, HMC readiness, posterior correctness, scientific validity, architecture ranking, or a change to the GPU NeuTra default. |

## Execution Contract

- q=20, `(32,32)`, three-stage dense IAF, batch size 100.
- Existing stream seeds and target/data definitions remain unchanged.
- Validation/support checks every 250 steps.
- Existing loss-based plateau controller and learning-rate repair remain unchanged; saturation is telemetry only.
- Maximum 2,000 optimizer steps per stream.
- Four persistent CPU workers, each evaluating a TensorFlow batch shard of 25 rows for training.
- Validation/support/audit shard sizes are fixed and declared as `(2, 3, 16, 25, 64)`; no scalar shard is admitted.
- Parent TensorFlow: 4 intra-op plus 1 inter-op thread.
- Worker TensorFlow: one intra-op plus one inter-op thread each.
- Target route: `batch_native_tensorflow_no_row_mapping_v1` using the batched principal-square-root UKF kernel; no `tf.map_fn`, Python row loop, NumPy target path, or scalar fallback.
- Target values and scores cross worker boundaries as serialized TensorFlow tensors.
- Parent optimizer uses the explicit non-XLA CPU diagnostic exception because
  CPU XLA was measured to create more than 50 native threads by itself.
- Full process tree is bound to logical CPUs `0-49`.
- Every target evaluation audits `/proc` thread counts and fails closed above
50. The realized `/proc` count, rather than a thread-count formula, is authoritative.

## Budget

The campaign cap remains `13,500 s`, inherited from the authorized CPU campaign
and treated as a cumulative wall cap, not an expected-use claim. The prior
`10,574.85 s` estimate was measured for the rejected scalar-worker route and is
not a runtime prediction for this revised route. Sequential stopping returns
unused time. It does not authorize HMC or another training candidate.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| `(32,32)` | User-selected architecture and existing two-seed lane | Insufficient capacity | Final audit and support checks are descriptive only; no HMC promotion |
| Batch 100 | Prior user-selected batch and strict CPU timing | Different stochastic loss behavior than batch 480 | Preserve exact seed/batch contract and report separately |
| Four batch-native workers | Target-specific execution design | Worker compile or shard mismatch | Spawn-safe pool smoke plus per-evaluation backend and shape checks |
| Non-XLA parent | Required strict-thread diagnostic exception | Runtime differs from repository XLA default | Record backend and forbid cross-backend quality/ranking claims |
| 2,000 max / 250 cadence | Existing reviewed adaptive training procedure | Plateau between checks or cap truncation | Preserve existing controller decisions and stop reason |
| Two seeds | Existing independent stream protocol | Underpowered general robustness evidence | Report each stream separately; no population ranking |

## Skeptical Pre-Execution Audit

- Wrong baseline: checked; the same q=20 target equations, architecture, batch,
  seeds, optimizer settings, plateau controller, support probe, and final audit
  are used. The batch-native implementation was checked against the scalar
  authority with value error below `5e-12` and score error below `4e-11` on
  q=1, q=2, and q=20 smoke points.
- Proxy promotion: prevented; training loss and CPU screen status cannot admit
  a transport for HMC.
- Missing stop: repaired; 2,000 steps per stream and 13,500 cumulative seconds.
- Hidden thread assumption: repaired; realized process-tree threads are checked
  during every target evaluation, not inferred from environment variables.
- Batch-training policy: repaired; the prior `r1` route reported
  `scalar_eager_value_score` and is invalid. The revised route fails closed on
  any worker backend other than `batch_native_value_score`.
- Environment mismatch: explicit and unavoidable; non-XLA CPU output is a
  diagnostic exception and cannot be compared as an exact GPU/XLA replication.
- Artifact adequacy: each checkpoint, best state, frozen diagnostic payload,
  final audit, thread telemetry, summary, and source hashes are preserved.
- Misleading pass: a passing CPU loss screen does not establish good HMC
  geometry or posterior validity.
- Misleading failure: a failed CPU candidate does not reject NeuTra; classify
  thread/resource/infrastructure failures separately from training-screen vetoes.

Audit decision: `PASS_FOR_BOUNDED_CPU_DIAGNOSTIC_EXECUTION_REVISED_BATCH_NATIVE_ROUTE`.

## Exact Command

```bash
timeout 13500 taskset -c 0-49 python \
  docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-strict-cpu-training-2026-07-22/r4 \
  --cap-seconds 13500
```

## Rejected Attempt Record

`r1` was interrupted after seed-a step 250. Its worker metadata reports
`scalar_eager_value_score`; it is preserved as a harness-invalid artifact and
must not be interpreted as CPU training evidence. It is not resumed.
