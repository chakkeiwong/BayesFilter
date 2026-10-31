# KDM root-cause trace: pretraining and ineffective perturbation

12 October 2026. Authorized by the owner's request to trace the code. This is
a fresh, bounded diagnostic following the terminal coupled-trajectory phase;
it neither reopens its exhausted attempt cap nor changes replication criteria.

## Question and evidence contract

Why can the full training comparisons disagree despite close coupled updates,
and where did the previous representable 1e-8 noise perturbation disappear?
Compare the saved actual source and TensorFlow MDPF starting checkpoints, then
replay the pinned author's proposal at the saved unit01 starting/best states.
Use the exact previous initial cloud and GPU/XLA Philox proposal noise. Verify
the unperturbed raw proposal is bitwise identical to the saved author tensor
before interpreting any perturbation. Probe the previously selected coordinate
607 at fixed positive displacements 1e-8, 1e-7, 1e-6, and 1e-5, retaining zero
effects. Capture every Linear/ReLU output and the sigmoid, scale, and addition
operations. This is a localization experiment, not a tolerance calibration.

For starting checkpoints, verify identical fresh constructors, the three
transferred modules against each engine's own selected pretraining checkpoint,
and untouched parameters against the common constructor. Report parameter
distances descriptively. They cannot rank functions or identify which training
distribution is better. No optimization updates, checkpoint selection, full
training, population qualification, or algorithm repair will be performed.

Promotion criterion: none for replication. Diagnostic validity requires exact
saved-case replay, matched inputs/weights, finite tensors, and genuine changes
to the noise. Failed replay or provenance is a continuation veto; a zero
downstream effect is an explanatory result. Layer differences localize an
operation, not the origin of final trained quality. Cheap baselines are the
unperturbed source replay and repeated identical replay; neither is an added
performance requirement for the author's algorithm.

## Defaults, audit, and execution

The source pin, unit01 checkpoints, batch64/particles25, root seed202610122,
update0/time1, coordinate607, FP32 and CPU author execution reproduce the
completed diagnostic. GPU/XLA is used only to regenerate its exact base noise,
with TF32 disabled and verified memory growth. The fixed scale ladder is a
local sensitivity hypothesis; it must not be selected to excuse a port error.
An exact replay check detects any reconstruction, device, constant, or random
key mismatch before interpretation. CPU/GPU effects remain confounded in
existing engine comparisons; this experiment makes no causal precision claim.

Skeptical audit: passed for this narrow question. Baselines are actual saved
source computations and actual starting checkpoints, not final independent
training outcomes. Proxy parameter distances are explicitly nonqualifying.
The major remaining alternative—engine differences versus learned-start
differences—still requires a crossed-start trajectory experiment. This probe
will not be presented as closing that gap.

At most two launches, 600 wall seconds each; charge actual elapsed wall time to
both resources conservatively. Existing remaining budget: CPU219836.468876s,
GPU90889.071432s. Fresh output root:
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/root-cause-trace-20261012/attempt01.
The diagnostic supervisor will preserve source hashes, command, git revision,
environment, hardware, random key, logs, result and wall time there. Stop on
invalid replay, nonfinite values, source drift, timeout, or exhausted cap.

Exact launch (GPU sandbox escalation required):

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/diagnose_kdm_root_cause_trace.py --output docs/plans/artifacts/controlled-smc-kdm-replication-20260928/root-cause-trace-20261012/attempt01
```

The source worker uses /home/chakwong/anaconda3/envs/tf-gpu2/bin/python with
CUDA_VISIBLE_DEVICES=-1 and its environment lib path. Result note:
controlled-smc-kdm-root-cause-trace-result-20261012.md.

Attempt01 stopped after 10.277816 seconds because this diagnostic reader
incorrectly expected weights/models/full_dpf_model_best.pt. The author saves
three separate pretrained module checkpoints. No source or TensorFlow
algorithm failed. Exact GPU noise regeneration passed. The reader now loads
and prefixes those three actual files; attempt02 uses the identical contract
and a fresh output directory. Attempt01 remains preserved and chargeable.
