# Repaired KDM paired training calibration

11 October 2026. Active bounded phase of the replication master. This continues
the authorized repair campaign and uses its existing CPU/GPU allocation.

## Question, comparator and decision

Does variability between independently trained, repaired TensorFlow models and
the released author models permit an affordable no-worse qualification? Pair
each model with the same author constructor weights and the same fixed training
and validation data. Preserve each backend's native training randomness and its
own preceding-stage best-checkpoint transfers. MDPF and A-MDPF are both required.
This is development calibration, not qualification or model selection.

Reuse all three audited author units already generated without selecting among
their outcomes: the original unit and source-variability units 01 and 02. The
original unit already has a completed repaired TF partner. Train the other two
TF partners once, after fresh two-epoch full-data smokes. Never replace a seed
because its result is unfavorable. Do not change architecture, learning rate,
clipping, bandwidth, controller, stopping rule, precision or Adam behavior.

The primary phase completion criterion is three correctly matched, completed
and audited pairs with common-author evaluation, or an explicit bounded
incomplete result with its cause. The final replication criteria remain upper
one-sided 95% bounds of 0.01 nat NLL difference and 1% RMSE increase. This small,
already inspected development set cannot issue that final verdict. Report
per-unit seed-averaged NLL differences, log RMSE ratios, their sample SDs and
source/TF covariance. Repeated filter seeds do not increase the training-unit
sample size. Any normal-approximation precision calculation is illustrative;
three pairs cannot validate its distributional assumptions or tail behavior.

Stop launching further training on a source/data/hash mismatch, failed smoke,
non-finite training, wrong backend/device, failed terminal audit, code drift,
30-minute heartbeat stall, or exhausted phase/campaign budget. Preserve the
failed attempt. These invalidate the current comparison or trigger a bounded
harness repair; they do not prove failure of the scientific method. A finite
unfavorable metric is evidence to report, not grounds to discard or retrain.

NLL, RMSE and angle-seam slices explain repaired behavior; this phase cannot
promote defaults, claim superiority, reproduce all paper figures, complete the
original iAPF, or start E0/Task 3. The author method is the required replication
baseline; a new heuristic competition would answer a different question.

## Source and call-chain audit

The pinned author repeat-run generator uses 11 trials and changes run/output
and checkpoint paths, not the training dataset paths:
`.localresources/code/younis-mdpf-neurips-2023/src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/scripts/generate_multiple_runs_config_files.py`.
The adjacent `configs/dataset_params_training.yaml` fixes shared 5000/1000-row
saved datasets. The paper's Figure 5 and Section 6 describe variability over
training runs and the 5000/1000 training/validation sizes (local published text
`younis-sudderth-2023-long-range-tracking.txt:609,653`). Independent data per unit
would change this conditional-on-data experiment; the previous roadmap
suggestion to regenerate it was a planning error and is withdrawn.

The executable chain is `diagnose_kdm_fresh_training_cost_tf.py` ->
`diagnose_kdm_source_epoch_tf.run_epochs` -> source stage wrappers ->
`BearingsNeuralReplay` and the repaired Adam. The wrapper formerly hardcoded
one seed. It now accepts and records an explicit seed argument, retaining the
historical default. Use disjoint seed ranges 1100000000 and 1200000000 for the two new
units, leaving room for the epoch offsets. Each smoke and full run starts from
the corresponding source constructors; the smoke checkpoint is not a warm
start. Hash equality of datasets/fixtures and executable smoke seed records
must verify this wiring. The independent terminal audit replays the pinned
author scheduler and early stopping on saved losses. The common-author
evaluator loads each unit's own source and TF checkpoints, audits their
provenance and runs the source NLL/RMSE classes. Native TF filter/metric parity
was already checked in native-evaluation-01; no runtime algorithm changes are
made in this phase.

## Assumptions, budget and execution

| Choice | Provenance and status | Risk and earliest check |
|---|---|---|
| Fixed training/validation data | Author repeat-run baseline | Different bytes would invalidate pairing; compare all dataset hashes |
| Same initial weights within pairs | Existing source constructor exports; variance-reduction hypothesis | Training streams differ, so covariance may be small; report paired SD without claiming exact RNG parity |
| Two additional TF units | All available independent author calibration units | n=3 is too small for final population inference; retain development label |
| FP32, TF32 off, GPU/XLA | Checked repaired training route | Scalar/cast or graph drift; source proof, fresh smokes, memory-growth and stable-trace checks |
| One CPU/inter/intra-op thread | Completed 8846-second pilot; execution choice | Different stopping paths change cost; bounded per-unit timeout and heartbeat |
| Two evaluation seeds, 256 trajectories, T=150, N=25 | Existing common-author development evaluator | Limited evaluation resolution; no paper-wide or independent-training-count claim |
| 0.01 nat and 1% RMSE | Unchanged KDM criteria | May require infeasible precision; estimate feasibility before untouched qualification |

Total cap: 30000 GPU wall seconds and 1800 CPU wall seconds, debited from the
remaining 107533.433034 GPU-tranche and 220739.425683 CPU seconds. Each smoke
has a 600-second cap and each full training a 14400-second cap, subject to the
remaining total. Two units, one smoke and one full training each; at most two
fresh campaign attempts for localized harness repair, sharing the same total
budget. No new author training, new hyperparameter search or extra grant.

The supervisor `docs/benchmarks/run_kdm_paired_calibration.py` owns the versioned
root `artifacts/controlled-smc-kdm-replication-20260928/paired-calibration-20261011/`.
It records exact commands, Git/source/input hashes, seeds, environment,
GPU-growth/TF32/XLA provenance, child PIDs, heartbeat, all logs, wall charges and
terminal decision. GPU commands use escalated access. CPU reference workers
hide the GPU before import. Start with its `--preflight`, then `--execute`.

Skeptical pre-execution review: the old TF checkpoint is excluded; the actual
author models and pinned metric consumers are the comparator. Source units
were not selected on favorable results. Fixed data follows author semantics.
Matching initialization does not guarantee low variance. No proxy or small-n
result can qualify the port. Caps and audit stops prevent unbounded spending.
The likely misleading success is three favorable development pairs being
called a population pass; the supervisor must always emit replication
INCOMPLETE. The likely operational failure is a different convergence path
exceeding the cap; record it as incomplete, not as a numerical failure.

Final pre-execution audit: all three units share the same saved data hashes.
The evaluator now binds each TF model to its actual source constructor/data
reference. Before any new training, replay evaluation of the completed first
pair and require exact equality of its four comparison records. Source unit 01
has an audited artifact-based completion recovery with an unobserved process
exit code; verify its archived manifest, final result and terminal progress
hashes without inventing a zero exit code. Unit 02 has normal recorded
completion. The supervisor checks these conditions in preflight and records
live usage and its current step in the master status command.

Afterward, report a decision table and inference-status table in the result
note, then freeze a separate untouched qualification design only if precision
and remaining compute make it defensible. Otherwise report that limitation
without spending the remaining allocation on an unpowered ladder.
