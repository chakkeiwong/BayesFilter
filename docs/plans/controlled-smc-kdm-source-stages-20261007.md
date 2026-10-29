# KDM source-stage mechanics and complete-epoch pilot — 7 October 2026

Authority: continue the owner's replication-only program within the existing
allocation. The preceding bridge passed controller/native/checkpoint screens.
This phase closes remaining stage mechanics and measures one real data epoch;
it is not a full trained-method qualification.

## Research intent / evidence contract
Question: does TF implement the author's initial-cloud and weight-pretraining
operations and execute a complete 5000-row training / 1000-row validation
epoch with the correct final partial batches?
Comparator: pinned Younis source b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18,
actual WeightModelTrainer.do_forward_pass/get_true_masked_data, cloud generator,
create_initial_dpf_state, KDE_NLL and Adam. Source checkpoint is a development
fixture. A-MDPF from a converted MDPF checkpoint is a warm start only.
Primary implementation criteria: exact data/shape/cadence and parameter
coverage; source replay values/gradients/Adam within existing dtype bounds;
initialization noise replay matches source; native draws satisfy the previously
checked law and feed the same call chain; all epoch rows used exactly once,
drop_last=False, weighted epoch means, validation changes no optimizer state,
one stable trace per required batch shape and shared variables across shapes.
Stop/veto invalid source/data/provenance, GPU/memory configuration, missing
gradient, unexplained mismatch, nonfinite source-valid update or exhausted
budget. A failing candidate triggers focused localization, never relaxed bounds.
Runtime/learning loss over one epoch are explanatory only; no learning ranking.
No full replication, default, HMC, eleven-trial paper or no-worse claim.
Heuristic-performance comparisons are inapplicable to an implementation/cost
pilot; the accepted final NLL/RMSE criteria remain required later.

## Source/default audit
Pinned src/ anchors:
- models/kde_particle_filter/kde_particle_filter.py:573-671:
  true state plus independent N(0,.5) x/y noise and VM(0,1/.03) angle noise,
  then uniform weights and .001 initial posterior bandwidth; A-MDPF computes
  its separate initial resampling bandwidth. The initial angle is added to
  the true angle without wrapping the sum.
- trainers/specific_trainers/weight_model_trainer.py:23-83,108-167,170-231:
  select times0,4,8,12,16 in time-major order, draw one cloud center with
  scales2,2,.1, then50 points with3,3,.1; internal sin/cos coordinates,
  normalized positive weights and manual KDE bandwidth1,1,.1.
- experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/
  individual_training/config.yaml: Adam lr1e-4, coupled weight_decay.001,
  no gradient clip specified; three trained modules. Confirm trainer's missing
  clip default is None. Full-stage Adam remains clip100/decay0.
- trainers/trainer.py:144-145,407-564,579-610: shuffled train rows, ordered
  validation rows, retain last partial batches, weight epoch loss by actual
  effective batch size. Data config: train5000,val1000,T17,sparse modulo4.
- Stage dependencies: pretrain components -> MDPF; MDPF full checkpoint ->
  A-MDPF; NLL evaluation consumes NLL-trained checkpoints. MSE training is
  not silently substituted for the active NLL checkpoint comparison.

Float32 source constants remain float32 when explicitly promoted for f64
reference; source source-data saved tensors are fixed/hashes verified.
TF stateless RNG is an independent implementation of the same law, not Torch
random-bit replay. Source export is CPU-only diagnostic; TF algorithm uses no
NumPy. No stabilizing clip/ridge/damping added. Existing proposal singularities
and finite checks are retained; native sampler cap128 rejects on exhaustion.
Frozen source settings are comparison choices, not optimization recommendations.
An untrained or fitted starting checkpoint is a diagnostic fixture; a single
epoch is never counted as completed training.

## Execution/budget/artifacts
At most six fresh attempts, total CPU <=1800s and GPU <=3600s within the existing
ledger; no new grant. Preserve all commands, full logs and manifests under:
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/
kdm-source-stages-20261007-XX/.
1. Export source pretraining/initialization replay and all train/validation
   data using the actual source dataset accessor. Verify source against Git pin.
2. Implement shared-variable stable batch specializations, initial clouds and
   pretraining using the existing TF neural/KDE/sampler kernels. Check source
   replay and native-to-replay wiring before epoch execution.
3. Execute one source-sized TF epoch (pretraining plus MDPF/A-MDPF full-stage
   cost probes) only after those checks pass. Record initialization/checkpoint
   identity, every row index, seed, batch size, loss, validity and trace counts.
   Frozen development seeds; no untouched final claim data.
4. Update the master, stage result, once-only resource ledger and checkpoint.
   Full source/TF paired training still requires a separately bounded reviewed
   cost/power design under the accepted criteria.

## Skeptical audit before execution
Pass for this bounded implementation/cost scope. Main misleading-success risk:
reusing a trained model can hide learning failures. Therefore pilot checkpoint
roles are explicit and no training-quality conclusion is available. Wrong
batch aggregation is tested on partial batches; identical initial variables
must be shared by all shape-specialized graphs. Pretraining weight decay must
enter Adam's gradient, not AdamW decay; replay gradients and both moments
discriminate this error. Full epoch costs may be confounded by GPU sharing and
compilation, so emit separate first-call and subsequent timings and do not use
a single epoch as a statistical performance ranking.

## Continuation audit and exact commands

The 7 October continuation checked the source initialization, cloud generator,
masked-data ordering, loss and optimizer consumers. The source has 5000 training
and 1000 validation rows. Training uses batch 64 (last batch 8); validation uses
256 (last batch 232). Weight pretraining includes five labelled times in
time-major order, so its network batches are five times those sizes. It has
coupled Adam weight decay .001 and no gradient clipping. Full training retains
its source clipping at 100 and zero weight decay. Initialization uses
concentration 1/.03 exactly; pretraining cloud generation uses 1/(bandwidth+1e-8).
Do not substitute the latter concentration in initialization. Preserve source
angle addition without a final wrap for initialization.

Audit passes for implementation and cost only, conditional on matched replay
checks before epoch execution. GPU sharing is recorded and does not define a
performance ranking. Use the less occupied RTX 5080 by its GPU UUID; verify
memory growth, TF32 off and XLA on. The source export deliberately hides GPUs.

Exact command family (existing approved interpreter; no shell environment prefix):
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_kdm_source_stages.py --mode MODE --output OUTPUT`
where MODE is `export`, `verify` or `epoch`, OUTPUT is a fresh numbered directory
under the artifact root above, and GPU modes additionally consume `--reference`
from the completed export. The launcher fixes the worker, environment, logging
and timeout; it cannot execute arbitrary commands. Up to six attempts total and
the original CPU 1800s / GPU 3600s phase caps still apply.

Attempt02 stopped after five passing checks on a diagnostic adapter error: the
source represents unused MDPF resampling bandwidth as None. The adapter now
uses the posterior bandwidth for that unused field. Review also found that
the pretraining fixture batch_size=320 is the effective five-time batch; the
consumer must bind the sequence batch from its 64 input rows. Both harness
repairs preserve source values and tolerances. No epoch has run.

Attempt03 reached 784 passing checks, then a diagnostic seed-signature error:
TF fold_in returns int32, while the standalone sampler accepts int64. The
full-sequence replay now consumes its captured initial particle cloud directly,
so it tests the actual executed call chain without changing any random draw.
Initialization and pretraining replay, both dtypes, had passed. Stage transfer
checks now export the author's actual partial-module and full-checkpoint loads.

## Epoch launch audit — attempt06

Attempt05 passed all 2204 source-stage checks with zero failures on RTX5080
GPU/XLA, including both precisions and actual author checkpoint transfers.
Allocator peak282652928 bytes, memory growth verified. Earlier float32 failures
remain open. The epoch consumer binds the completed verification and runtime
hashes, checks every train/validation row and partial batch, shares variables,
freezes validation state and saves three stage checkpoints. Source-sized batch
and weighted-mean mechanics answer the cost question. Fitted initialization
cannot establish learning quality. Audit passes for this bounded pilot.
Attempt06 is the last launch under the unchanged six-attempt CPU1800s/GPU3600s
phase caps; use source04 and verification05. No full-training claim.

## Local infrastructure repair allocation after attempt06

Attempt06 stopped before the first numerical batch: StatelessShuffle has only a
CPU kernel, but the epoch runner inherited GPU placement with soft placement
disabled. This invalidates the runner's row-order placement, not the previously
verified numerical kernels or method. Explicit CPU row ordering preserves the
seeded shuffle law; training and validation kernels remain GPU/XLA. Record and
check the row-order device. Runtime code and verification05 remain unchanged.

The initial six-launch allocation is closed (CPU14.311843927949667s,
GPU258.54160681995563s). Authorize two additional infrastructure-repair launches
under the same original total CPU1800s/GPU3600s phase resource caps, including
all initial attempts. This is a recorded local retry allocation within the
owner-approved campaign, with no change in scientific criteria or hardware.
Audit: unchanged source/data/RNG/optimizer/baseline; placement exception applies
only to host row ordering, with exact coverage checks. No numerical gate relaxed.
Proceed with fresh attempt07, same epoch command/reference04/verification05.
