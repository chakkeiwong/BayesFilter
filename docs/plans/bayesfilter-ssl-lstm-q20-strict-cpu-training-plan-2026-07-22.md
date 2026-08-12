# q=20 SSL-LSTM Strict CPU NeuTra Training Plan

Date: 2026-07-30
Tier: bounded serious diagnostic CPU campaign  
Status: `SEED_A_CPU_DIAGNOSTIC_COMPLETED_SCREEN_PASSED`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Main question | Can one q=20 `(32,32)` NeuTra stream complete the existing adaptive batch-100 procedure using the validated 25-process CPU topology? |
| Exact inputs | Existing q=20 synthetic target, fixed hyperparameters `learning_rate=0.0004`, `initialization_scale=0.01`, `gradient_clip_norm=10`, and existing independent seed-a/seed-b definitions. |
| Candidate mechanism | Existing adaptive NeuTra procedure, executed by a standalone batch-native CPU runner and TensorFlow-only target boundary. |
| Primary completion criterion | Seed A reaches its existing plateau stop or 2,000-step maximum, writes checkpoints and final audit, and remains finite under the 25-process topology. |
| Hard vetoes | Visible GPU; configured compute cores above 50; nonfinite target/score/loss/gradient/support/audit; host memory above 64 GiB; corrupted or incomplete artifact; `51,500 s` cumulative cap. |
| Explanatory only | Loss trajectory, saturation telemetry, learning-rate reductions, and runtime. |
| Artifact | Fresh seed-A attempt under `docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30/`. |
| Nonclaims | CPU results cannot establish transport promotion, HMC readiness, posterior correctness, scientific validity, architecture ranking, or a change to the GPU NeuTra default. |

## Execution Contract

- q=20, `(32,32)`, three-stage dense IAF, batch size 100.
- Existing stream seeds and target/data definitions remain unchanged.
- Validation/support checks every 250 steps.
- Existing loss-based plateau controller and learning-rate repair remain unchanged; saturation is telemetry only.
- Maximum 2,000 program updates for seed A.
- Twenty-five persistent CPU processes, each pinned to one logical CPU and evaluating a TensorFlow batch shard of 4 rows for training.
- All validation/support/audit batches are partitioned into batch-native shards
  of at most four rows; a final 1--3 row remainder is allowed. There is no
  scalar-target fallback.
- Parent TensorFlow: 4 intra-op plus 1 inter-op thread.
- Worker TensorFlow: one intra-op plus one inter-op thread each.
- Target route: `batch_native_tensorflow_no_row_mapping_v1` using the batched principal-square-root UKF kernel; no `tf.map_fn`, Python row loop, NumPy target path, or scalar fallback.
- Target values and scores cross worker boundaries as serialized TensorFlow tensors.
- Parent optimizer uses the explicit non-XLA CPU diagnostic exception because
  CPU XLA was measured to create more than 50 native threads by itself.
- Full parent process is bound to logical CPUs `0-49`; each pinned worker owns one
logical CPU. Native TensorFlow housekeeping threads are recorded, not counted as
configured compute cores.

## Budget

The original seed-A cap was `31,500 s` (`8.75` CPU-hours), derived from the
measured `13.616 s` steady-state q=20 batch-100 target evaluation time, 2,000
maximum updates, validation/audit overhead, and a 15% margin. On 2026-07-30,
the owner authorized exactly `20,000 s` more headroom after attempt 001 stopped
at step 1750. The active cumulative cap is therefore `51,500 s`. This is a cap,
not an expected-use claim; sequential stopping returns unused time. It still
authorizes only one CPU diagnostic seed and no HMC or candidate search.

Each update admits only when at least 180 s remain. Final support and audit
admit only when at least 120 s remain. These reserves are derived from the r4
profile's worst observed batch-100 call and the four-wave upper structure of
the final support/audit workload with more than a twofold timing margin. A
terminal stopped-controller checkpoint remains resumable for finalization if
an infrastructure interruption occurs after the stop decision.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| `(32,32)` | User-selected architecture and existing two-seed lane | Insufficient capacity | Final audit and support checks are descriptive only; no HMC promotion |
| Batch 100 | Prior user-selected batch and strict CPU timing | Different stochastic loss behavior than batch 480 | Preserve exact seed/batch contract and report separately |
| 25 batch-native workers, batch 4 each | Measured q=20 profile; reviewed target-specific execution option | Worker compile, affinity, or shard mismatch | Spawn-safe pool smoke plus per-evaluation PID, affinity, and shape checks |
| Non-XLA parent | Required strict-thread diagnostic exception | Runtime differs from repository XLA default | Record backend and forbid cross-backend quality/ranking claims |
| 2,000 max / 250 cadence | Existing reviewed adaptive training procedure | Plateau between checks or cap truncation | Preserve existing controller decisions and stop reason |
| Seed A only | Smallest end-to-end mechanism test | Cannot establish seed robustness | Require separate seed B before any robustness statement |

## Skeptical Pre-Execution Audit

- Wrong baseline: checked; the same q=20 target equations, architecture, batch,
  seeds, optimizer settings, plateau controller, support probe, and final audit
  are used. The batch-native implementation was checked against the scalar
  authority with value error below `5e-12` and score error below `4e-11` on
  q=1, q=2, and q=20 smoke points.
- Proxy promotion: prevented; training loss and CPU screen status cannot admit
  a transport for HMC.
- Missing stop: repaired; 2,000 seed-A program updates and 51,500 cumulative seconds.
- Hidden topology assumption: repaired; pinned worker identity, CPU affinity,
  shard sizes, and configured compute cores are checked on every target evaluation.
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

## Attempt-002 Budget Extension Addendum

Attempt 001 stopped cleanly at the original campaign cap after writing a valid
step-1750 checkpoint. The owner then authorized exactly `20,000 s` additional
wall time. Attempt 002 conservatively charges prior cumulative time as
`31,350 s`, one second above the attempt-001 manifest value
`31,349.25759465 s`, and uses a `20,000 s` outer timeout. The launcher cap is
`51,500 s`, leaving 150 seconds of conservative accounting slack beyond the
outer timeout.

The research question, target, architecture, batch, stream seeds, optimizer,
adaptive controller, topology, promotion criterion, vetoes, nonclaims, and
hardware class are unchanged. The latest checkpoint records program step 1750,
controller status `running`, learning rate `0.0002`, and restored best trainer
step 1500. This is the declared adaptive repair state, not a state mismatch.

```bash
timeout 20000 taskset -c 0-49 python \
  docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py \
  --stream seed-a --cpu-processes 25 --batch-per-process 4 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30 \
  --cap-seconds 51500 \
  --resume-checkpoint docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30/seed-a/checkpoint-1750.json \
  --prior-wall-seconds 31350
```

The additional budget does not authorize seed B, HMC, a new architecture,
retuning, threshold relaxation, or promotion from CPU-only evidence.

## Attempt-002 Terminal Result

Attempt 002 completed the declared 2,000 program updates and final support and
heldout audit work. The terminal summary status is
`CPU_DIAGNOSTIC_COMPLETED`; the seed-A result status is
`CPU_DIAGNOSTIC_SCREEN_PASSED` with no vetoes. The controller selected best
step 1500, reduced the learning rate once at step 1750, and stopped at step
2000 with `maximum_steps_reached`.

This closes the bounded CPU seed-A campaign. It does not authorize more use of
the remaining headroom: no further optimizer updates, seed B, HMC, retuning,
new architecture, or GPU campaign is implied. The result remains a one-seed,
non-XLA CPU diagnostic exception and is ineligible for transport promotion,
posterior claims, HMC readiness, or a repository default change.

## Seed-A Execution Addendum

The launcher now requires `--stream seed-a` or `--stream seed-b`, supports a
validated `--resume-checkpoint` with charged `--prior-wall-seconds`, and writes
an immediate launch-attempt manifest. A debug-only optimizer-update smoke can be
run with `--debug-stop-after-steps 1..249`; it cannot resume or support a
training-quality claim. The full seed-A command is:

```bash
timeout 31500 taskset -c 0-49 python \
  docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py \
  --stream seed-a --cpu-processes 25 --batch-per-process 4 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30 \
  --cap-seconds 31500
```

Seed B requires a separate fresh output root and is not implied by this run.

## Preflight Result

The exact q=20 seed-A path completed a two-update smoke under the 25-process by
4-row topology in 68.15 s. The terminal update had finite loss `69.4076` and
finite gradient norm `74.4038`; the support probe was finite with round-trip
maximum absolute error `1.78e-15`. The artifact is
`docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-smoke-2026-07-30/r1/`.
This is mechanics evidence only and does not satisfy the training-quality gate.

## Historical Attempt Record

`r1` was interrupted after seed-a step 250. Its worker metadata reports
`scalar_eager_value_score`; it is preserved as a harness-invalid artifact and
must not be interpreted as CPU training evidence. It is not resumed.

The four-worker batch-native timing attempt and earlier `13,500 s` command are
historical performance evidence only. They are not active launch instructions
for this campaign.
