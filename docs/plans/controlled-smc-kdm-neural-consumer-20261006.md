# KDM source neural consumer continuation — 6 October 2026

Authority: the owner accepted the numerical criteria and directed continuation.
The master and accepted criteria remain authoritative. This is a bounded
implementation phase, not a final stochastic qualification.

## Question and evidence contract

Does the TensorFlow bearings implementation compute the same released MDPF and
A-MDPF neural filtering/loss/Adam operations on replayed source inputs?
Comparator: pinned Younis source b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18,
actual KDEParticleFilter.forward and source full-sequence loss/truncation where
used. Preserve the actual NLL checkpoint consumer. A saved source MDPF trial
(attempt16 in the September24 reference campaign) is a checkpoint/conversion
fixture only after its hash and architecture are verified. It cannot stand in
for eleven independent trials or an A-MDPF-trained checkpoint.

Primary engineering criteria: source parameter/key/shape conversion is complete;
matched values, trainable gradients, global clipping, and first/subsequent Adam
state/updates are compared; no omitted consumer branch or trainable parameter.
Report float64 and float32 separately. Use the existing absolute-plus-relative
diagnostic bounds, 1e-9*(1+abs(reference)) for float64 and
128*2^-23*(1+abs(reference)) for float32. Report normalized update errors and
zero/missing gradients independently. These fixture screens cannot themselves
establish the accepted no-worse stochastic margins or complete training.
Full training remains gated on complete consumer implementation and cost.

Vetoes: wrong source/checkpoint/configuration, missing/unexplained parameters,
nonfinite source-valid outputs/gradients, omitted source normalization/detach,
unrecorded device mode, failed memory-growth configuration, or budget exhaustion.
A valid numerical mismatch rejects that candidate and triggers localization,
not rejection of KDM. Runtime, trace count, compilation, memory and historical
training time are explanatory/cost diagnostics. Circular random sampling is
replayed from the author for this phase; it is not yet a TF sampling-law test.
No default, superiority, HMC or full replication claim is available from it.

## Source audit and defaults

Call chain anchors under the pinned author's src:
- models/kde_particle_filter/kde_particle_filter.py:280-342 checkpoint conversion;
  675-755 IWSG; 812-848 proposal; 850-970 weight; 972-1128 neural forward.
- models/particle_transformer.py:112-176 circular representation and normalization.
- models/internal_models/particle_encoder_models.py:20-158; proposal_models.py:
  20-211; weight_models.py:20-317; bandwidth_models.py:29-177.
- trainers/specific_trainers/full_sequence_trainer.py:124-247 loss masking,
  temporal truncation and averaging; trainers/trainer.py:327 Adam and 509 clipping.
- bearings_only_experiments_methodical_25_particles_multiple_runs configs:
  25 particles, latent width8, three linear layers, ReLU, no batch/weight norm,
  source learning rates, true-state initialization, and source stage dependencies.
- Paper Section5, equation18, AppendixD.1 equation23; prior circular/source
  specification audits retain equation and source anchors.

Every replay fixture is development data. Small batches are diagnostic cost
choices; no training-quality inference. Initial/fitted MDPF weights and
source-defined A-MDPF conversion test distinct mechanisms; converted A-MDPF is
a warm start only. Preserve source FloatTensor constants when casting a fixture.
Adam means the installed author's torch.optim.Adam defaults, not a convenient
Keras default: verify epsilon placement and bias correction against the actual
source optimizer. No ridge, clipping or damping is added beyond source policy.
Sparse label masks and truncated-BPTT cadence must be recorded, not guessed.
Record any fixture that deliberately changes them as diagnostic only.

## Execution, budget and stop conditions

1. Verify pinned source, saved checkpoint/configuration and historical timing.
2. Export actual neural source fixtures and optimizer states; preserve draw replay.
3. Implement missing TensorFlow consumer and validate through stable GPU/XLA
   signatures, source values/gradients and optimizer updates.
4. Inspect discrepancies, execute a bounded localized repair where justified,
   and record cost/remaining training work. Then select the next independent
   phase; do not launch unspecified full training or rerun identical failures.

Phase cap: six numerical attempts, at most 7200 CPU wall seconds and 7200 GPU
wall seconds total, charged once to the existing campaign allocation. Subprocess
timeouts enforce the cap; preserve each attempt in a fresh
artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-20261006-NN directory.
Routine source inspection/compile checks are recorded separately. No grant is
added. Initial source extraction is CPU-only with CUDA_VISIBLE_DEVICES=-1.
GPU commands use trusted permissions, TF_FORCE_GPU_ALLOW_GROWTH=true, verified
memory growth, explicit dtype/TF32 and physical device. Exact interpreter,
commands, source hashes, Git identity, seeds, wall time and outputs go in manifests.

Intended entry points: docs/benchmarks/reference_kdm_neural.py (author reference)
and docs/benchmarks/diagnose_kdm_neural_tf.py (GPU/XLA comparison), controlled by
a bounded repository launcher. Tests must answer wiring/numerical/state questions,
not merely repeat the implementation.

## Skeptical audit before implementation

PASS for this bounded phase: the comparator is actual source consumers, not a
constructed neural output; checkpoint label ambiguity is already resolved.
The saved trial is not counted as the active eleven-trial scope. Fixed draw
replay, short horizons and single-checkpoint results remain implementation
evidence only. The source CPU reference vs TF GPU comparison is explicit.
Float32 failures are retained, not relabelled by the accepted scientific
margins. Full-suite cost and stochastic precision remain unresolved.
A six-attempt/dual-resource cap and fresh outputs prevent open-ended retries.
Cheap first checks are source hash, architecture/key equality, scalar loss and
gradient connectivity; these catch an invalid comparison before a costly run.

## Local execution repairs

Attempts 01–03 stopped before numerical comparison: source namespace collision,
strict GPU placement of an integer optimizer counter, then XLA gradient rewriting
of a conditional around the nested I0 polynomial loop (FakeParam, variant).
The source import is now explicit, the counter uses the computation dtype, and
the first non-resampling transition is peeled before the recurrent TF loop.
This preserves the source operations and truncation boundaries; no eager or
non-XLA fallback is promoted. Attempt 04 reuses the verified source tape and
retains every numerical bound. These are infrastructure repairs under the
existing resource allocation, not changes to acceptance.

Attempt 04 executes GPU/XLA successfully: 880/880 float64 checks and 878/880
float32 checks pass. Only the first-update particle coordinates exceed the
unchanged screen in both variants. Extend the bounded attempt count to six,
within the unchanged 7200-second per-resource phase cap, to retain and inspect
pre-normalization proposal vectors and the exact failing coordinates. The
atan2 condition number is an explanatory diagnostic: its differential is
(cos_coordinate*d_sin - sin_coordinate*d_cos)/(sin_coordinate^2+cos_coordinate^2).
A small vector magnitude can amplify otherwise small neural arithmetic error.
Do not enlarge acceptance bounds from this fixture. Added captures must leave
the source loss/gradient/optimizer fixture unchanged.

Attempt 05 added diagnostics but its host report requested GPU UnravelIndex,
which has only a CPU kernel under strict placement. This is a reporting failure;
replace only index decoding with Python integer divmod at the artifact boundary.
All 108 source no-change comparisons (inputs, losses, gradients, parameters and
Adam states) pass after adding observability. Attempt 06 reuses that source tape.
