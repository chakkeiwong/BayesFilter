# KDM replication repair: isolate the pretraining boundary

13 October 2026. Active under the owner's request to review and continue the
replication repair. This is a diagnostic within the existing allocation;
the conditional extra 48 hours have not been added.

## Question and source

Does changing only the pretrained measurement modules explain the poor saved
unit01 MDPF result? Check the actual stage transition before interpreting
independently trained final checkpoints as a translation failure.

Baseline: pinned author commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18;
the existing unit01/02 source runs and repaired TensorFlow calibration runs.
Paper: local Younis--Sudderth 2023, Appendix D.2--D.3, text lines 1598--1633:
measurement pretraining precedes full filter training; Adam and T-BPTT are
specified separately. Author code: weight_model_trainer.py:107--167,
kde_particle_filter.py:280--355, trainer.py:575--611 and 613--682.
The active consumer is diagnose_kdm_fresh_training_cost_tf.py ->
diagnose_kdm_source_epoch_tf.run_epochs -> SourceFullTraining.update ->
BearingsNeuralReplay._apply_update. Existing local parity is reused.

## Execution and evidence contract

A. Reconstruct each author MDPF starting state with the actual load_pretrained
method and the saved fresh constructor. Compare every parameter to the saved
TF starting state. Only the three measurement modules may differ; constants,
optimizer state, rates and frozen/non-transferred parameters must agree.
Evaluate author and TF pretrained measurement models on eight identical
author-generated validation cloud draws per unit (1000 sequences, 5000 labeled
states, P=50). The clouds are parameter-independent, so equal RNG inputs and
outputs establish a real common-draw comparison here. Save per-batch results,
input hashes, source anchors and the reconstructed starting states.

B. First run two MDPF epochs from the old TF starting state using exactly its
original seed schedule (1100000000), data and epoch driver. Require matching
row order, valid/committed updates, controller events and losses within 1e-5
absolute / 1e-6 relative to the saved trajectory. A failed control is a harness
repair trigger and continuation veto until explained. Then run MDPF from the
reconstructed author starting state, with that same seed schedule and unchanged
source stopping rule/cap (800 epochs). Preserve the original global stage index
(1 for MDPF). This is a controlled within-TF intervention in starting weights;
it is not a new default and does not redo pretraining or A-MDPF.

C. Evaluate the new best checkpoint, original TF best and source best through
the unchanged author full validation forward method on eight fresh native
draws each. Use the original 1000 rows and batch sizes. Preserve per-batch
values and verify no parameters change. Same initial RNG is not asserted to
give common realized draws for full filtering. Report checkpoint-conditional
means/dispersion and individual draws; do not turn draws into training units.

Primary diagnostic: which change in final native validation NLL follows from
the isolated starting-state intervention, compared with the original TF run.
This is explanatory evidence, not a replication completion or quality ranking.
An exact wrong parameter transfer is a repair trigger; source hash drift,
unexplained control mismatch, nonfinite state, invalid sampler, missing rows,
budget exhaustion or corrupt checkpoint is a continuation veto. An unfavorable
valid new trajectory is a candidate result and motivates the next localization;
it does not invalidate the paper or authorize altered floors/losses/schedules.
No population noninferiority, general superiority, heldout RMSE, A-MDPF repair,
or paper-wide reproduction is concluded. The old 0.01 nat/1% population claim
remains unestablished and is not the gate for executing this diagnostic.

## Assumptions and skeptical review

| Choice | Provenance / status | Failure mode and early check |
|---|---|---|
| Unit01 intervention; unit02 stage audit | Existing unfavorable saved units; diagnostic hypothesis | Selection limits generality; retain both stage checks and prohibit population claims |
| Original TF seed schedule | Frozen counterfactual control | Wrong stage index or seed offsets; two-epoch replay control |
| Author pretrained modules | Released stage loader, baseline | Incomplete transfer; exact full parameter and optimizer comparison |
| Eight validation draws | Bounded explanatory convenience | Weak tail precision; preserve draws and do not rank populations |
| Original FP32/Adam/no additional damping | Replication target, prior local arithmetic controls | Sensitive angle/CDF branches; preserve flags, no silent protection changes |
| Native CPU author evaluator | Independent reference exception | Different method/metric; extracted original forward method and complete batch coverage |

Audit PASS for diagnostic execution: the prior master conflated a population
precision problem with source replication and left no executable next action.
This plan uses the exact author comparator, a changed-start counterfactual and
a reproducibility control. It avoids validation-loss promotion, seed shopping,
relaxed acceptance margins and an unpowered new population ladder. Initial
quality differences can be small while optimization basins differ; phase A
cannot by itself veto phase B. A single trajectory cannot distinguish all
optimization variability from undiscovered translation errors. Cheap heuristic
adversaries are not promotion baselines here: the target is the author's code,
not an improved filter; no learned-method superiority claim is made.

## Budget, commands and artifacts

Total phase caps: 16200 CPU wall seconds and 14400 active GPU seconds, charged
to the existing balance (216933.270848659 CPU / 88339.909896140 GPU seconds).
At most three versioned attempts; retries only for localized harness/resource
failures, never to select a favorable seed. CPU reference phases cap at 1800s;
GPU full training cap at 14400s and two-epoch control at 300s, together bounded
by the total cap. Measured prior MDPF runtime was 4112s for319 epochs; expect
about one to three hours depending on the unchanged source stopping rule.

Supervisor command:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_stage_isolation.py`

Run with trusted GPU permissions. GPU worker uses TensorFlow/XLA FP32, TF32
off, stable signatures and verified memory growth. Author worker uses the
tf-gpu2 environment with CUDA_VISIBLE_DEVICES=-1. No package changes.

Output root:
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/stage-isolation-20261013/`.
Each stage preserves exact command, source/runtime hashes, data hashes, seed,
environment, hardware, wall time, logs, states and result. Supervisor records
progress, enforces timeouts, charges once and writes a terminal decision.
Result note: controlled-smc-kdm-stage-isolation-result-20261013.md.


## Bounded harness repair after attempts 01–02

Attempt01 exposed JSON string keys in the new CPU cloud diagnostic. The pinned
native YAML now supplies the keys; a CPU-only finite-cloud check passed. The
author audit subsequently passed 484 checks, including actual module transfer,
constants and realized common clouds. Eight draws give TF-minus-author
pretraining NLL differences of -0.00016066 and +0.00314893 nat. These are
checkpoint-conditional descriptions, not a population quality decision.

Attempt02's GPU control stopped before any update because the referenced
verification file predates diagnostic-output additions in two runtime modules.
Inspection found optional captured sampler fields (off for training) and extra
pretraining diagnostics; no changed training formula. Nevertheless, rerun the
existing source-stage verification on current files and preserve its fresh
manifest and checksum before replaying. Its 900-second cap is inside the same
16200 CPU / 14400 GPU phase budget; it does not enlarge that budget. The final
attempt is attempt03 under the existing three-attempt cap.

Also pin CUDA_VISIBLE_DEVICES to GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a (RTX
5080), exactly as the saved TF comparator. Attempt02 exposed both GPUs; the
verification guard prevented training on the unintended device. Record the
explicit environment for every stage. The source-full and source-cost inputs
have identical configuration, dataset and constructor hashes; this was checked
before retry. Skeptical re-audit passes after these harness repairs. A failed
fresh verification or replay control still stops the intervention.


## Evaluation-only repair after completed training, 13 October 2026

Attempt03 completed 477 MDPF epochs and 37,683 optimizer updates, passing
6,680 checks; termination was SOURCE_EARLY_STOP. The best checkpoint is epoch
431 (zero-based), SHA256 b39244cc5c5aada6a24f50be1cd9cb615cb48d5352108e0f66f93af590db5468.
Evaluation failed before its first loss: saved JSON architecture has string
dimension keys, whereas author create_initial_dpf_state indexes integers
(kde_particle_filter.py:592). The earlier cloud-only repair did not cover the
model constructor. This is a reference-harness defect, not a training failure.

Repair model construction and loss configuration from pinned native YAML,
checking JSON-normalized equality with the saved configuration and integer
keys. Preserve every trained checkpoint, seed, draw count, row, metric,
numerical setting and author source file. Do not rerun training or select
another seed. Compare the same three checkpoints on eight draws each and all
1000 validation sequences through the unchanged native author methods.

Skeptical audit PASS: exact author baseline, unchanged target and criteria,
checkpoint/data hashes, explicit CPU-only reference exception. No population
ranking or KDM completion follows from this one diagnostic. Configuration
equality, complete rows, finite losses, immutable parameters, pinned sources
and exact native aggregation remain vetoes. The first native forward call
is the focused regression; remaining draws follow only if it passes.

The three full-campaign attempts are exhausted. Add at most two evaluation-only
harness retries within the SAME 16200 CPU / 14400 GPU second phase caps.
This localized repair follows existing authorization; it permits no further
training attempt or compute expansion. Prior phase usage is 11170.930965392 CPU
and 11090.917908120 GPU seconds; remaining 5029.069034608 CPU and
3309.082091880 GPU seconds. Each evaluation retry caps at 1800 CPU wall seconds
and zero GPU, also bounded by remaining campaign allocation.
Versioned evaluation-repair01/02 roots preserve attempt03 unchanged. Stop on a
scientific/provenance veto, budget exhaustion or the bounded retry limit;
record each charge once. Conditional extra48h has not been added.

Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/resume_kdm_stage_isolation_evaluation.py
The supervisor uses the existing tf-gpu2 author evaluator with
CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=1 and its matching library path.
No package changes. Preserve exact commands, sources, checkpoint hashes,
environment, wall time, terminal decision and native statistics.

## Terminal outcome, 13 October 2026

Completed: attempt03 trained for 477 epochs and evaluation-repair01 passed the
native author comparison (260 checks). The initial-weight intervention reduced
the saved unit01 NLL gap from 0.736745 to 0.034902 nat, or 95.26%. This answers
the planned stage-isolation question for that trajectory. Independent
pretraining divergence and whole KDM completion remain unresolved.
See the [terminal result](controlled-smc-kdm-stage-isolation-result-20261013.md)
and evaluation-repair01/terminal-review.json for interpretation and provenance.
Phase usage: 11250.946277 CPU / 11090.917908 GPU seconds; no run remains active.
Next is focused pretraining source/port localization, not a repeat of this run.
