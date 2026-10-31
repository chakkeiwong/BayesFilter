# Complete KDM source/TF development training — 7 October 2026

Continue the authorized replication campaign. The fresh-cost phase completed all
three actual author epochs and 36 GPU/XLA checks. This phase measures complete
training from fresh constructors under the released 400/800/800 epoch caps and
source early stopping. It is one development pair, not untouched qualification.

## Research intent and evidence contract

Question: can both implementations complete the released weight-pretraining,
MDPF and A-MDPF schedule, starting from identical constructor parameters and data
and transferring each implementation's own validation-selected checkpoints?
Comparator: author commit b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, actual
setup_training -> Trainer.train path. The TF side uses the source-checked
initialization, kernels, optimizer, scheduler and early-stop controller.

Pass requires finite accepted updates, complete 5000/1000 row coverage per epoch,
unchanged frozen parameters, source learning-rate/early-stop semantics, stable
traces, correct best-checkpoint transfer, and terminal source-cap or source-stop
reason. This is an implementation completion screen, not numerical acceptance.
Missing provenance, broken data/checkpoints, nonfinite updates, extra source/port
failure, budget exhaustion or a 30-minute heartbeat stall stop the affected
attempt. Repair localized harness failures within the phase allocation.
Existing float32 compatibility failures veto replication promotion, but do not
veto this training/cost collection. cSMC fitting remains a separate open task.

Validation loss, epoch counts, time, memory and scheduler events are explanatory.
Do not rank source/port using these development losses or describe a trained model
as a qualified replication. Independent trained-model replication, source-only
precision calibration and untouched NLL/RMSE confidence intervals remain required.
Heuristic adversaries and conditional heldout evaluation belong to that later
quality assessment; this phase makes no claim that either learned method is good.

## Defaults, mathematical scope and provenance

Use the source configurations and initializers preserved in the fresh-cost result.
No optimizer, clipping, bandwidth floor, objective, architecture, or protection
changes. Author training/config anchors and source API compatibility patches are
recorded in the fresh-cost plan and reference worker. Model seed 202610072..74,
data seed 202610071, TF shuffle/native bases 202610081/82 remain development
choices. Reuse the fresh-cost data and constructor exports to make contamination
checks exact. Never load a source-trained checkpoint into a running TF stage.
Independent Torch and TF native streams have the same checked law, not equal draws.
Every epoch uses a distinct TF key; the shuffle remains CPU orchestration.

CPU source: tf-gpu2 Python, GPU hidden, one Torch thread and one loader worker.
TF: tftwogpu Python, RTX5080 UUID GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a,
float32, TF32 off, GPU/XLA stable signatures, verified memory growth. The source
Torch/NumPy lane is an independent reference exception. Runtime code is unchanged.

Early diagnostic: run exactly two epochs of every stage before the full pair.
Require 158 updates per TF stage, distinct epoch keys, complete row coverage,
scheduler state advancing twice, and transfer from the selected best checkpoint.
The cap of two is a harness smoke only. It cannot substitute for full training.
Existing controller unit parity already covers learning-rate reductions and
early stopping; do not fabricate source events to make this smoke exercise them.

## Budget, execution and artifacts

Reserve at most 18 CPU hours and 16 GPU hours from the existing active allocation,
including smoke, failures and retries; at most six attempts. One fresh full pair
is the target. The one-epoch measured steady work extrapolates to roughly 13 CPU
and 12 GPU hours at all caps, excluding unpredictable overhead/early stopping.
These estimates are descriptive and do not establish a sufficient number of
trained units for qualification. Stop within the reservation; do not extend it
silently. Release unused reservation after terminal accounting.

Launcher: scripts/run_kdm_complete_training.py --mode source|tf --epochs smoke|full
--reference PATH --output PATH. Reference is the successful fresh-cost source
result. Output prefix: docs/plans/artifacts/controlled-smc-kdm-replication-20260928/
kdm-complete-training-20261007-XX. Run source/TF independently in parallel, each
with unique paths, logs, manifests, dataset/initialization hashes, progress,
per-epoch results, best/latest checkpoints and terminal reason. Record runtime
and helper hashes. The launcher enforces remaining phase time and heartbeat
stalls and kills the entire child process group on timeout.

After the smoke, inspect all three stages, key separation and source/TF checkpoint
semantics. Launch the full pair only if the smoke and code audit pass. Monitor
initial full epochs, preserve any failure and repair only what can be repaired
without changing the scientific target. Refresh the master and concise checkpoint
with running PIDs, outputs, live reservation and next action.

## Skeptical pre-execution audit

The fresh-data run addresses initialization; it does not repair earlier float32
bounds. One epoch cannot test repeated controller state or epoch RNG separation:
the required two-epoch smoke addresses these earliest harness risks. All caps
must remain source values and early stopping must end only under the source
controller. Select the best checkpoint independently in each implementation;
final-weight transfer is wrong. CPU cost is now measured directly, replacing the
earlier GPU-derived estimate. Development data cannot later become untouched
qualification data. The phase passes review for full-training feasibility only;
no scientific acceptance or ranking follows from successful execution.

Post-run alternative explanation: completed training may still differ in quality
because of float32 arithmetic or RNG-stream effects. Complete trained-model
replications and heldout uncertainty, after numerical compatibility is resolved,
are needed to distinguish these explanations.
