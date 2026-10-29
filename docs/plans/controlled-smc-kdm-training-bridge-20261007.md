# KDM controller and native-consumer continuation — 7 October 2026

Authority: continuation under the accepted 6 October criteria and existing
48-hour CPU/GPU extension. Both replications remain incomplete. This phase
does not change the author algorithm or the numerical acceptance margins.

## Research intent and evidence contract
Question: does the TF consumer execute source sampling inside the actual
neural training call chain, and does its host controller select the same
learning rates, stopping epoch and checkpoints as the pinned source?
Comparator: Younis source b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
Primary engineering criteria: exact controller event/state equality on
predeclared loss streams through the actual source Trainer.train body;
native draws replay through the existing neural consumer with equal losses,
gradients, clipping and Adam updates; complete checkpoint restoration,
including a resumed update; stable GPU/XLA traces and sampler-valid status.
These are implementation checks, not the accepted stochastic NLL/RMSE claim.

Promotion vetoes: extra invalid outputs on valid fixtures; lost parameters,
optimizer states or controller state; wrong source provenance; native/replay
mismatch outside the existing diagnostic bounds (f64 1e-9*(1+abs(reference)),
f32 128*2^-23*(1+abs(reference))); missing source branch.
Continuation vetoes: corrupt comparator, wrong data, unavailable trusted GPU,
memory-policy failure, or exhausted phase/campaign budget.
A candidate failure triggers the smallest localized repair and a fresh output.
Timing, memory, compiler/trace counts and rejection iterations are explanatory.
No superiority, default, HMC, full-training or full-replication conclusion.
The heuristic-performance gate is inapplicable to this exact implementation
comparison; no performance-quality claim is sought. Source training and final
NLL/RMSE qualification remain separate required phases.

## Source anchors and assumption audit
Paths below are relative to the pinned source src/.
- trainers/trainer.py:185-204,298-346,613-699: one optimizer/scheduler per
  module, minimum relative plateau rule, strict best validation checkpoint,
  scheduler before early stop, break before saving a stopped epoch.
- trainers/trainer_utils.py:8-58,260-311: equality does not advance early
  stopping; plateau reduction EVENT is true even at the rate floor; only the
  first three such events reset its counter. Preserve these source behaviors.
- experiments/bearings_only_experiments_methodical_25_particles_multiple_runs/
  configs/common_parameters_training.yaml: full 800 epochs, N25, train64,
  validation256, truncate4, clip100, Adam decay0, scheduler .001/.1/20/4/0,
  early stopping patience25 and start offset200.
- The same configs/our_experiments_exp3_init.yaml:16-44 loads each trained
  MDPF checkpoint for A-MDPF and trains its decoupled components. A converted
  MDPF checkpoint alone is not a trained A-MDPF result.
- Native sampling and neural math retain the inspected anchors in
  controlled-smc-kdm-native-sampling-20261006.md and
  controlled-smc-kdm-neural-consumer-20261006.md.

Inherited scheduler threshold mode rel and epsilon1e-8 are pinned Torch
defaults, checked directly against its implementation. Negative losses,
ties, cooldown, delayed start, floor events and stopping boundaries receive
explicit cases. Host-only controller arithmetic is intentional, never a TF
numerical-kernel fallback. No clipping, ridge or damping is added.
Source clip100 and no added numerical stabilization are frozen comparison
choices; nonfinite gradients and undefined proposal angles are recorded and
reject the attempt, never silently repaired into accepted source behavior.
Sampler cap128 is a reject-only bound, not a distribution modification;
exhaustion must propagate to the consumer and reject its update.
The initial state remains the captured source initialization in this phase.
Native initial-state generation, pretraining, complete data epochs and
untouched qualification remain open. Seeds are development-only.

## Execution and budget
At most six versioned attempts, combined at most 1800 CPU seconds and 3600 GPU
seconds, charged once against the active allocation. This is not a new grant.
Output root:
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/
kdm-training-bridge-20261007-XX/
A repository launcher must preserve commands, source/file checksums, environment,
git state, full logs, return codes and wall time. CPU oracle uses tf-gpu2 with
CUDA_VISIBLE_DEVICES=-1 before Torch import. GPU code uses tftwogpu,
trusted permissions, memory growth, float64/float32, TF32 off, XLA on.
No persistent background campaign is launched by this diagnostic.

1. Compare actual source epoch-control execution with the port on fixed streams:
   positive plateaus; negative losses; exact ties; improvements; cooldown;
   rate-floor events; delayed scheduler start; early-stop boundaries.
   Test JSON controller resume against uninterrupted events.
2. Integrate the already checked sampler in the existing recurrent neural
   consumer, capture its consumed draws/noise, and replay those through the
   same source-matched numerical path. Confirm invalid-sampler propagation.
   Permit host learning-rate assignment and strict model/Adam checkpoint
   restoration without retracing.
3. Run bounded B64 native updates and B256 evaluation if compatible.
   Report compile and repeated-call timings separately; do not extrapolate a
   one-batch time into a validated full-training cost.
4. Record result, resource charges and next action in master and checkpoint.
   Then implement the remaining initial-state/pretraining/data-epoch route
   before source-only cost/power calibration and untouched qualification.

## Skeptical pre-execution audit
Pass for this bounded engineering phase. Wrong-baseline risk is controlled by
running the actual pinned Trainer.train method and preserving the author
checkpoint/conversion identity. A model fixture cannot certify learning;
the plan explicitly withholds that claim. No heldout claim rows are consumed.
Native draws use the source distribution rather than shared RNG sequence;
same-draw replay is therefore a wiring test, not Torch/TF random-bit equality.
The key pre-mortem is a disconnected but passing sampler: captured draws must
be consumed by the claim-bearing update and replayed with all optimizer states.
Full stage/data coverage and training quality remain visible gaps.

Controller attempt01 completed its event comparisons but JSON rejected the source scheduler's initial positive-infinity sentinel on delayed-start cases. Encode the uninitialized best as null in both diagnostic records; this changes serialization only. Preserve the failed log and charge its 2.620265 CPU seconds. Current Torch requires the documented inherited-helper alias and omission of the removed logging-only verbose keyword; source class and Trainer.train bodies are unchanged.

After attempt03 passed 1648 checks, the review identified a missing discriminating guard check: invalid bandwidth also invalidates posterior loss, so it alone cannot prove sampler-status propagation. Attempt04 forces cap1 on healthy inputs before tracing, requires finite loss/gradients and unchanged state on rejection, and verifies restore into two fresh consumers plus atomic rejection of an incomplete checkpoint. Runtime cap128 and the scientific criteria remain unchanged.

Terminal: attempts02–04 pass; source controller423, native1648 and guard9 checks. Attempt01 serialization failure retained. Charged CPU 34.739317156s and GPU 261.898277840s once; details in [result](controlled-smc-kdm-training-bridge-result-20261007.md). Next: Implement source true-state initialization, weight pretraining, shared-parameter batch specializations and complete data epochs, with source stage/checkpoint transfer. Then measure an actual native epoch before source-only cost/power calibration and untouched paired qualification. Controller/native wiring passed; preserve prior float32 failures and the cSMC conditional-solver gap; no E0.
