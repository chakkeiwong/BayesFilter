# Diagnose the failed simulated-moment training

20 October 2026. Branch `ledh-refactor-with-policy-fix`, base `4d6915cc66`.
Owner request: trace the executed code and mathematics to explain the failure.
This is a bounded diagnosis, not another training or promotion campaign.

## Question and evidence contract

Why did optimizing simulated terminal first/second-moment loss fail to produce
a supported gain over the untrained guides? Trace the actual consumer from the
adapter through proposal sampling, importance weights, correction/reset and
loss; distinguish bugs, restrictions, optimization evidence and uncertainty.
The squared-error identity remains the target justification. Its population
statement is not a convergence guarantee for the implemented optimizer.

Comparators are the saved zero and selected adapters, the same-lane unguided
half-mixture, and the conditional-only proposal. Preserve the exact executed
SIR d18/T5/N64, FP64 GPU/XLA, TF32-off controls, including OT (1,5), balance32,
UKF covariance, dual-cap correction, Contract E and chunk64. Historical
pre-reset LEDH results are not comparison evidence. The completed 32-dataset
holdout stays sealed against tuning; its published negative result is context.

Pass criterion for a causal implementation finding: an inspected call-chain
and algebraic identity, together with an executable wiring/parity or focused
counterfactual check where feasible. Trace summaries, local sensitivity, loss
variation and ESS are explanatory diagnostics only. They cannot promote a
candidate or establish a population ranking. Invalid/nonfinite replay, changed
inputs, missing required traces or a source mismatch veto the affected check.
Invalidity prompts localization; it is not evidence that the training identity
is false. Stop for exhausted budget or an invalid assumption with no local fix.

## Execution and skeptical review

1. Inspect adapter, fitter, optimizer, shared executor and loss. Summarize saved
   training/validation records: accepted updates, parameter movement, probe
   contrasts, selected checkpoints, kernel activity and guide-fit diagnostics.
2. Derive what exact importance correction preserves, what proposal changes can
   affect, and how a future guide differs from controlled-SMC twisting. Check
   the restrictions imposed on fitted quadratic/kernel messages and final time.
3. If saved evidence cannot distinguish mechanisms, replay saved **training or
   validation** observations with fixed independent particle streams. Retain
   per-time component allocation, normalized weight masses, ESS and first/second
   moments before and after reset. Check adapter sensitivity and any claimed
   cancellation using the existing shared consumer, never a simplified filter.
4. Write a result with confirmed findings, untested explanations, repair targets
   and source/evidence anchors. Refresh the concise checkpoint. Do not retune,
   alter scientific defaults, modify the runtime algorithm, or open a new claim.

Skeptical review PASS for this diagnostic scope. Wrong-baseline risk is handled
by replaying the same consumer and streams. Low-data validation selection and
the single-latent-label noise are hypotheses to measure, not assumed causes.
No descriptive comparison or proxy becomes a promotion criterion. The old
theta-derivative parity checks do not verify optimization with respect to eta.
The fixed beta=.5 and restricted adapter may limit the family; neither can be
called the cause until their actual effects are checked. The recursive-Gaussian
heuristic failure remains a promotion veto from the completed campaign; this
diagnosis does not select on that heuristic or claim to clear it.

## Defaults, assumptions and budget

All numerical controls and fitted messages are replay inputs from train-01,
not newly endorsed defaults. Reusing them isolates the failure; their limitations
must stay visible. Small fixed datasets and finite differences are diagnostic
choices, with local-only interpretation and step-size agreement checks when
derivatives are used. Independence of latent labels and particle/fitter streams
conditional on observations is checked in the call chain. A complete trace is
preferred to a newly implemented numerical surrogate.

Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`. CPU inspection uses
stdlib; any CPU TensorFlow check sets `CUDA_VISIBLE_DEVICES=-1`. GPU replay uses
escalated access, `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0`, verified
memory growth and the existing stable-signature XLA kernels. At most two GPU
launches and 1,200 GPU-process seconds, charged to the parent remaining
24,245.397855901 seconds. Reserve no extra training budget. No network or paid
compute, package mutation, commits or publication.

Versioned output root:
`docs/plans/artifacts/ledh-moment-training-failure-diagnosis-20261020/`.
Save exact commands, source hashes, seeds, environment, device policy, wall time
and validity in the run manifest; retain full logs on disk. Exact replay commands
will be recorded before launch after the minimal diagnostic is specified.
Result: `result.md` in this root, including decision/inference-status tables.

Pre-mortem: a valid filter can still implement too small a learning family;
correct importance weights can undo a guide's intended population tilt; noisy
sample-label losses can obscure a useful moment change; post-reset moment
preservation can make a proposed decomposition uninformative. Distinguish these
with inspected algebra and matched executions. A failed local probe weakens
only its stated mechanism, not all future-guidance research.

## Frozen diagnostic replay

Replay training seeds263021001–004 and validation seeds263022001–004,
two independent particle streams each (277000000–015), with the saved fitted
messages. Seven arms: plain, conditional-only, half-mixture, and each guide
before/after its frozen selected adapter. This is 112 shared-kernel traces;
seven direct consumer checks verify the trace-derived loss. On two training and
two validation datasets, compute coordinate central differences at eta=0,
radius0.1, for the selected pooled/time-specific capacities. Radius0.05/0.025
checks on the first coordinate in one dataset of each partition distinguish
local nonlinearity from numerical resolution. No parameter is selected here.
Record the exact sampled training batches as an additional coverage diagnostic.

Reviewed command (first launch, 1,000-second process cap; remaining200 seconds
reserved for a localized harness retry):

```sh
PYTHONPATH=. TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/plans/artifacts/ledh-moment-training-failure-diagnosis-20261020/replay_diagnostic.py --output docs/plans/artifacts/ledh-moment-training-failure-diagnosis-20261020/replay-01 --budget-seconds 1000
```

The diagnostic reuses `PhysicalCampaign` construction and the repository-owned
kernels, verifies all eight original runtime hashes, and inspects returned
traces. Eager TensorFlow moment reductions are a post-run diagnostic exception.
The main numerical filter retains GPU/XLA and analytical theta derivatives.

Launch01 stopped before any filter call: the diagnostic imported the supervision
module (and its model dependencies) before the campaign configured GPU memory
growth. This is a new diagnostic import-order error, not the original training
failure. Move that import after campaign initialization, as the original runner
does. Preserve replay-01.log. Conservatively charge60 seconds for this startup
failure because process wall time was not emitted. Retry once in replay-02 with
the same 1,000-second cap, within the unchanged 1,200-second total budget.
The retry's manifest and successful initialization are the focused regression.

## Completion

Replay-02 completed in359.199960844 seconds,255 shared-kernel/endpoint calls,
112 traces and7 exact consumer-parity checks. All validity checks pass and all
eight recorded runtime hashes match the completed training. No holdout was
reopened, no adapter selected, and no runtime code or default changed.
Total stage charge419.199960844 seconds over two launches, including the
conservative60-second startup-failure charge. Parent remaining23826.197895057
seconds. No further diagnostic run is needed or queued.

The [terminal result](artifacts/ledh-moment-training-failure-diagnosis-20261020/result.md)
derives the objective and weight identities, reports allocation and reset
checks, distinguishes the executed kernel approximation from replicated KDM,
and records the limits of the optimization evidence. Review found no basis
to treat the valid population objective or roundoff-level reset discrepancies
as the cause. Candidate rejection remains distinct from research-direction
rejection; no population ranking is inferred from the local replay.
