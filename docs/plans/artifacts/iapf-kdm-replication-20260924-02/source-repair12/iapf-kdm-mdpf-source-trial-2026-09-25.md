# Complete MDPF NLL source trial within the authorized 48-hour campaign

Date: 2026-09-25 HKT. Parent: iapf-kdm-replication-48h-continuation-2026-09-25.md.
Authorization: owner supplied 48 additional GPU hours and asked to continue.
This implements one complete source training trial; the eleven-trial comparison,
Appendix B synthetic replication, iAPF and TensorFlow admission remain separate.

## Intent, source anchors and evidence contract

Question: does the pinned author's full MDPF NLL training protocol complete with
finite parameters, its actual IWSG consumer, source-selected checkpoints and a
reproducible independent evaluation? The comparator for implementation validity
is the original author trainer/configuration at
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18, not another learned method.
The original source is under .localresources/code/younis-mdpf-neurips-2023.
The resolved configuration preserved in campaign -01/attempt09 binds its source
files. trainers/trainer.py:610-701 runs epochs, scheduler updates, validation
selection and early stopping. The resolved individual-training configuration
specifies 400 maximum epochs; full MDPF NLL specifies 800. The source consumer
training_runner.setup_training must execute both stages. Preserve source Adam,
learning rates, clipping, scheduler and early-stopping settings.

The paper's Section 6 distinguishes likelihood training from optional MSE
refinement. The source MSE-named evaluation loads the NLL checkpoint. This trial
is explicitly NLL-trained MDPF; later NLL and RMSE measurements must both name
that checkpoint, and cannot be called an MSE-refined result.

Primary pass criterion: each real training stage returns after all source
epochs or its own early-stopping condition; train/validation epoch counts agree;
finite losses, gradients and parameters; actual intended resampler and optimizer;
unchanged frozen parameters; saved and reloaded best checkpoints with hashes;
same generated train/validation data in both stages; complete source/environment
and timing records. A missing or invalid record, changed source semantics,
nonfinite quantity or timeout vetoes protocol completion. A source numerical
failure is candidate failure and a repair trigger, not research-direction
rejection. Wrong data/source or a shared environment failure stops dependent
execution. Short integration smoke is engineering evidence only.

Epoch losses, wall time, optimizer rates, skipped steps and memory are explanatory.
No learning-quality, gradient-unbiasedness, ranking, default-readiness, TensorFlow
GPU/XLA admission, or completed paper replication follows from a training pass.
Untouched evaluation must precede a quality claim. One trial is descriptive.

## Defaults and source-preserving differences

Source settings: 5000 training / 1000 validation trajectories, T=17, full-filter
N=25, training/validation batches 64/256, sparse supervision and truncated BPTT
modulo four, the published source architecture and pretraining checkpoint map.
Check source epoch/data caps before execution; reject mismatches rather than
silently inventing a replacement. Pretraining particle count stays in source.

Seeds 925201 (data), 925202 (pretraining), 925203 (full training) are predeclared
reproducibility choices; original experiment seeds were not recovered. They are
fresh relative to the cost probes. One persistent loader worker is an explicit
resource choice, not a scientific default. Record it and actual GPU placement.
Use the isolated author PyTorch/NumPy reference environment, 4080 SUPER, eager
execution; this does not change the BayesFilter TensorFlow/TFP/XLA backend.
Retain only previously tested API compatibility patches in an isolated source
copy. The source tree remains untouched. Guard against nonfinite gradients before
Adam updates; this guard rejects invalid inputs and does not alter accepted ones.

Premortem: short early epochs can underpredict later cost; source early stopping
may never fire; loss decrease does not prove estimator correctness; one seed can
look misleadingly good; a misleading experiment label can select the wrong
checkpoint. The 12-hour timeout, exact-source protocol checks, checkpoint hashes,
explicit NLL naming and no-ranking restriction address these risks. Do not
shorten source epoch caps after seeing outcomes.

## Heuristic and conditional evaluation obligation

Before promoting learned quality, construct and evaluate on identical untouched
trajectories: (1) static initial-position persistence, (2) constant-velocity
extrapolation from initial state, (3) geometric least-squares triangulation of
the two bearing sensors, (4) bootstrap PF with the known simulator dynamics and
bearing likelihood. The first two test whether tracking learns beyond motion
persistence; triangulation tests whether observations alone suffice; BPF tests
the learned filter against a classical filtering baseline. Handle angle wrapping,
coordinate units and likelihood normalization explicitly. Report per-time error
for early/middle/late trajectory windows and low/high bearing-intersection
conditioning strata fixed without learned outputs. Nonfinite results or learned
underperformance in a salient stratum veto quality promotion. These are sanity
checks, not tuning targets. Implement/source-check these consumers before using
the trial as evidence for learned quality. This obligation does not prevent
collecting source training evidence while quality promotion remains blocked.

## Budget, artifacts and exact execution

Use the existing -02 cumulative ledger: GPU cap 173700 seconds, CPU cap1800,
attempt cap60. Raising the per-child ceiling from300 to43200 seconds is an
allocation inside the existing owner budget, not added compute. Ordinary jobs
retain300 seconds; only the explicitly selected mdpf_source_trial job gets43200.
Long jobs are excluded from implicit/default selection. Reserve a300-second
trial_smoke first and900 seconds for subsequent untouched evaluation. At most
two infrastructure-repair smokes before reassessment; no unconditional long retry.
Up to43200 seconds training, at most900 evaluation, within parent phase3 and
its six-hour evaluation/repair reserve. Preserve all charged failed attempts.

Master job mdpf_trial_smoke runs the exact instrumentation and setup with one
pretraining/two full epochs, marked short and non-complete. After it passes,
mdpf_source_trial uses original epoch caps and fresh, identically predeclared
trial seeds/data. Smoke does not select hyperparameters or rank outputs.
Invocation uses the already approved prefix:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B scripts/run_replication_master.py
with run --stage stage0 --resume --retry-incomplete, the existing output/evidence
root, this --continuation-plan, a recorded --repair-note, and explicit --jobs.
Each run creates a fresh attempt directory and immutable code/plan snapshot.
Save actual command, git/source hashes, device/library probe, seeds, wall time,
protocol configuration, progress.json, append-only events.jsonl, phase results,
checkpoint hashes, terminal result and campaign budget. Evaluation must use the
completed trial's checkpoint and separate data; it is not automatically included
in training completion.

## Skeptical pre-execution audit

PASS for the limited source-training question: the actual original consumer,
stage order, objective, schedule and checkpoint mapping were inspected; short
MDPF, TG, IRG and DIS source consumers run successfully. TG cost74.946067 seconds,
IRG111.836848 seconds; IRG invoked injected gradients. Existing tests18 pass.
Remaining Appendix B and iAPF scientific issues do not invalidate independent
source training, but remain blockers to overall replication. This is independent
collection of the bearings source trial while those prerequisites are repaired,
not a declaration that Appendix B is complete. Require focused tests and the
new guarded smoke before the long launch. A successful process alone cannot
satisfy the explicit stage/checkpoint checks or establish quality.

## Automatic evaluation follow-up

After attempt16 completes successfully, the master's followup-evaluation action
waits read-only for a terminal ledger entry and complete-source-protocol result.
It must not evaluate a failed, shortened, incomplete or timed-out trial. It then
runs mdpf_trial_evaluation with that exact training directory, freshly snapshotted
code and this plan, a900-second GPU limit and the existing budget. A failed
training trial records BLOCKED and requires scientific/infrastructure diagnosis;
the follow-up does not retry or switch checkpoints.

Evaluate5000 untouched trajectories of T150 with source N25 and batch1024,
NLL/RMSE computed by original source consumers; save all raw sequence outputs.
Seeds925231 for data and925232 for initialization are fresh relative to training.
The source resets evaluation randomness from its actual default argument0 after
dataset creation; retain and disclose that behavior. Require five sequence calls,
745 resampler calls,10 metric calls, one data generation, finite metrics, saved
raw output and matching best-checkpoint hash. Metrics are descriptive only;
heuristic comparison and stochastic uncertainty remain quality-promotion blockers.
No MSE refinement is implied. The watcher consumes no GPU while waiting.
