# q=20 CPU Batch4x25 Seed-A Reboot Reset Memo

Date: 2026-07-30
Status: `REBOOT_PENDING_ACTIVE_PRECHECKPOINT_ATTEMPT`

## Objective

Complete one CPU-only diagnostic NeuTra training stream for the q=20
SSL-LSTM using the reviewed `(32,32)` transport, batch 100, adaptive
learning-rate repair, and the 25-process by 4-row target-evaluation topology.
This campaign does not authorize HMC, seed B, another architecture, posterior
claims, transport promotion, or a change to the repository GPU NeuTra default.

## Trusted Source State

- Git commit: `eae7955bbe8a8970328a162a1504e8b04b1ad57c`
  (`Add pinned CPU batch-4 NeuTra training mode`).
- Worktree was clean at memo inspection on 2026-07-30 02:55:27 Asia/Shanghai.
- Active plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-strict-cpu-training-plan-2026-07-22.md`.
- Launcher:
  `docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py`.
- Shared pool:
  `bayesfilter/inference/tf_batch_value_score_pool.py`.
- Focused prelaunch tests: `8 passed`.

The launch manifest binds the plan, launcher, pool, and target SHA-256 values.
Do not edit those files and then represent a resumed or restarted run as the
same source state. If the commit differs after reboot, reconcile source drift
before execution.

## Completed Preflight

The exact q=20 seed-A path completed a two-update optimizer smoke under the
25-process by 4-row topology:

- artifact:
  `docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-smoke-2026-07-30/r1/`;
- status: `CPU_DEBUG_SMOKE_COMPLETED`;
- wall time: `68.1474 s`;
- terminal loss: `69.4076091913525`;
- terminal gradient norm: `74.40382930958019`;
- support finite: true;
- support round-trip maximum absolute error:
  `1.7763568394002505e-15`.

This is mechanics evidence only. It is not evidence of plateau convergence,
heldout improvement, transport quality, HMC readiness, or posterior validity.

## Attempt 000 At Reboot Request

Fresh seed-A attempt 000 was launched at
`2026-07-29T18:12:26.314607+00:00`
(`2026-07-30 02:12:26.314607` Asia/Shanghai) with:

```bash
timeout 31500 taskset -c 0-49 python \
  docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py \
  --stream seed-a \
  --cpu-processes 25 \
  --batch-per-process 4 \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30 \
  --cap-seconds 31500
```

At 2026-07-30 02:55:27 Asia/Shanghai:

- timeout PID: `557247`;
- parent Python PID: `557248`;
- all 25 persistent spawned workers were alive and consuming CPU;
- process elapsed time was approximately `2,584 s`;
- no stderr or early hard-veto output had appeared;
- only `launch-attempt-000.json` existed in the output root;
- no `checkpoint-0250.json`, `progress.json`, result, or summary existed.

Therefore, at memo time attempt 000 was healthy but pre-checkpoint. A reboot at
that state interrupts valid optimizer work but leaves no replayable trainer or
controller state. The launch manifest is useful infrastructure provenance only;
it is not training-quality or scientific evidence.

## Campaign Budget Accounting

The authorized cumulative seed-A cap is `31,500 s`. Reboot does not reset that
cap. Charge attempt 000 from its launch timestamp through the actual process
termination or reboot timestamp:

```text
attempt_000_spent_seconds = termination_time - 2026-07-29T18:12:26.314607Z
remaining_seconds = 31500 - attempt_000_spent_seconds
```

The measured lower bound at memo inspection was approximately `2,584 s`, so
the remaining budget was no more than approximately `28,916 s` at that time.
Do not use `2,584 s` as the final charge if the process continued running after
memo creation. After reboot, obtain the prior boot's shutdown/end timestamp
from the system journal or another trusted host record and round the charged
duration upward to a whole second. If no trustworthy termination timestamp is
available, use the reboot boundary's conservative upper bound, not a smaller
estimate.

## Post-Reboot Recovery

First verify source and artifact state:

```bash
cd /home/ubuntu/python/BayesFilter
git rev-parse HEAD
git status --short
pgrep -af 'run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py|spawn_main'
find \
  docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30 \
  -maxdepth 3 -type f -print | sort
```

Expected commit is `eae7955bbe8a8970328a162a1504e8b04b1ad57c`. No old
training process should survive a reboot. Preserve attempt 000; do not delete,
overwrite, or relabel it as completed.

### Branch A: No Checkpoint

This was the observed state at memo time and is the expected reboot outcome.
Do not use `--resume-checkpoint`. Start fresh in a new versioned output root
with `remaining_seconds` as the cap:

```bash
timeout "${remaining_seconds}" taskset -c 0-49 python \
  docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py \
  --stream seed-a \
  --cpu-processes 25 \
  --batch-per-process 4 \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30/restart-r1 \
  --cap-seconds "${remaining_seconds}"
```

The variable must be a positive finite number no greater than `31,500`. Record
the derived value and termination-time source in a restart note before launch.
Because the restarted optimizer begins from step zero, the prior attempt's
scientific work is discarded but its wall time remains charged.

### Branch B: Checkpoint 250 Appeared Before Reboot

Use this branch only if all of these files exist and the launcher's built-in
joint-checkpoint validation passes:

- `seed-a/checkpoint-0250.json` or a later checkpoint;
- sibling `seed-a/progress.json`;
- matching latest checkpoint path, SHA-256, checkpoint hash, stream identity,
  trainer config, controller config, best-state hash, and program step.

Resume in the original output root with the full cumulative cap and actual
charged prior wall time:

```bash
timeout "${remaining_seconds}" taskset -c 0-49 python \
  docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py \
  --stream seed-a \
  --cpu-processes 25 \
  --batch-per-process 4 \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30 \
  --cap-seconds 31500 \
  --resume-checkpoint <latest-checkpoint-path> \
  --prior-wall-seconds "${attempt_000_spent_seconds}"
```

Here the outer `timeout` limits only the remaining host time, while the launcher
charges prior wall time against the cumulative `31,500 s` campaign cap. Never
resume from a checkpoint other than the latest receipt in `progress.json`.

## Evidence And Stop Contract

- Promotion criterion: seed A reaches the declared plateau stop or 2,000
  program updates, then completes finite final support and heldout audit checks.
- Promotion veto: failed heldout paired-loss screen or failed support screen.
- Continuation veto: nonfinite target/score/loss/gradient/support/audit,
  corrupted checkpoint/artifact, source/config mismatch, configured compute
  cores above 50, RSS above 64 GiB, or exhausted cumulative budget.
- Explanatory only: raw loss values, saturation telemetry, gradient norms,
  runtime, and intermediate checkpoint differences.
- One failed CPU seed does not reject NeuTra. It distinguishes implementation,
  tuning/training, resource, and candidate evidence as specified by the plan.
- A passing CPU seed still cannot authorize HMC. Seed B and claim-bearing
  GPU/XLA training remain separate future evidence requirements.

## Next Handoff

After reboot, the next agent must:

1. verify commit and worktree state;
2. determine the actual attempt-000 termination time and charged seconds;
3. inspect whether a valid checkpoint appeared after this memo;
4. select exactly one recovery branch above;
5. write a short restart record containing the arithmetic and exact command;
6. run focused launcher tests if source or environment changed; and
7. continue only within the remaining seed-A campaign budget.

Do not launch HMC, seed B, a new architecture, or a new hyperparameter search
from this memo.
