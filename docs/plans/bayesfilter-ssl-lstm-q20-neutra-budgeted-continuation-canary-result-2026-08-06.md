# SSL-LSTM q=20 NeuTra continuation canary result (2026-08-06)

## Outcome

The final seed-A canary passed the mechanics and resource gate. It restored optimizer step 1500, completed two GPU/XLA updates through optimizer step 1502, evaluated a fixed 500-row descriptive monitor, and passed the support/round-trip probe. This authorizes the unchanged bounded two-seed continuation campaign; it does not establish training convergence or HMC validity.

## Attempt ledger

| Attempt | Status | Finding and repair |
|---|---|---|
| `canary-seed-a-r1` | Launch invalid; no artifact | NeuTra imports initialized the GPU before the memory-growth helper. Moved the helper before every algorithmic import. |
| `canary-seed-a-r2` | Infrastructure failure before update | Spawned CPU workers re-imported the GPU-parent assertion after the pool hid GPUs. Added explicit main-process versus CPU-worker bootstrap roles. |
| `canary-seed-a-r3` | Two updates completed; monitor resource diagnostic failed | The 500-row monitor exposed task-summed worker `ru_maxrss` accounting. |
| `canary-seed-a-r4` | Measurement diagnostic | Confirmed the task-summed value was `126,450,745,344` bytes because 125 task results repeated 25 worker high-water marks. |
| `canary-seed-a-r5` | `GPU_CONTINUATION_CANARY_COMPLETED` | Deduplicated process RSS by worker PID using `/proc/<pid>/status` `VmHWM`; all mechanics and support checks passed. |

All repairs were localized harness/resource-accounting changes. Scientific target, resume state, training/evaluation seeds, optimizer schedule, batch topology, GPU identity, evidence roles, and campaign budget were unchanged.

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` with unrelated dirty worktree |
| Command | `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 taskset -c 0-24 python docs/benchmarks/run_ssl_lstm_q20_neutra_budgeted_continuation_2026_08_06.py --stream seed-a --cpu-processes 25 --batch-per-process 4 --resume-checkpoint docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/seed-a/seed-a/checkpoint-1500.json --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-budgeted-continuation-2026-08-06/canary-seed-a-r5 --canary-updates 2 --cap-seconds 3600` |
| Environment | `tfgpu`; TensorFlow version is recorded in the run manifest |
| CPU/GPU | 25 pinned CPU/XLA workers on CPUs 0--24; physical GPU 1 only; memory growth verified; logical GPU `/GPU:0` |
| Batch/XLA/dtype | 100 = 25x4; CPU target and GPU transport paths XLA compiled; FP64 |
| Random seeds | Seed-A 2026-08-06 training and monitor streams in the run manifest |
| Wall time | `119.8982255729934 s` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-neutra-budgeted-continuation-plan-2026-08-06.md` |
| Result | `docs/plans/artifacts/ssl-lstm-q20-neutra-budgeted-continuation-2026-08-06/canary-seed-a-r5/result.json`, SHA-256 `e2ab129fa72a3c6b4c24896b12c2cc719c15d833bf49e58b36531deb2e2c7607` |

## Diagnostics

| Diagnostic | Result | Role |
|---|---:|---|
| Optimizer steps | `1500 -> 1502` | Mechanics pass criterion |
| Update losses | `41.5980281061`, `41.4122463407` | Explanatory only |
| Gradient norms | `13.0094270043`, `15.7707001325`; both clipped | Explanatory only |
| Monitor | 500 rows, mean `41.2811667814`, descriptive SE `0.0697240072` | Explanatory only; cannot control training |
| Workers | 25 unique PIDs; 125 four-row monitor tasks; admitted status-bearing batch-native route | Hard topology gate |
| Unique-process RSS | `17.5897 GiB` combined parent and workers | Hard resource gate passed |
| Raw task-summed RSS | `117.3830 GiB` | Invalid as process-memory total; retained as diagnostic |
| GPU placement | All trainer variables and representative outputs on logical `/GPU:0`, mapped from physical GPU 1 | Hard device gate passed |
| Memory growth | Verified `true` before logical-device initialization | Hard launch gate passed |
| Support finite | `true` | Hard numerical gate passed |
| Round trip | `7.9936057773e-15` maximum absolute error | Hard support gate passed |

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Launch serious two-seed continuation | Passed mechanics, device, worker, finite, monitor, support, and resource checks | No canary veto remains | Two concurrent parents may add GPU scheduling overhead | Launch the frozen 43,200-second supervisor and inspect initial progress | NeuTra convergence, HMC readiness, posterior correctness, superiority |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for canary mechanics and resources |
| Statistically supported ranking | None |
| Descriptive-only differences | Loss, gradient, clipping, monitor mean and runtime |
| Default-readiness | Not evaluated |
| Next evidence needed | Complete both continuation budgets, select checkpoints, then perform fresh per-seed fixed-HMC tuning |

## Post-run red team

The strongest alternative explanation is that two concurrent GPU parents interfere even though the single-parent canary passed. The supervisor's per-child device and timing artifacts will expose that. The weakest evidence is training quality: two updates and one monitor cannot establish it. A concurrent resource or device failure would overturn launch readiness, not the mathematical target or NeuTra direction. A finite but unhelpful full training result would reject the current schedule/candidate and trigger target-specific optimizer review.
