# KDM unresolved-gradient follow-up

Question: do the remaining first gradient flags persist with identical source
pre-update state, and do source/TF backward operations explain them? This is
replication only. The resolved crossed-start CDF boundary is not rerun.

Comparator: pinned released Younis code b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18,
with the exact saved constructors, checkpoints, optimizer moments, data and
common realized base variates. Initial evidence is in the October 12 causal
localization result and coupled-trajectory result.

## Evidence contract and intent

The primary localization criterion is agreement of source and TF outputs,
raw gradients, optimizer gradients, moments and updates at identical inputs and
pre-state. Existing arithmetic thresholds (2e-5 FP32, 2e-9 FP64, each multiplied
by 1+abs(source)) are explanatory flags, not learned-quality criteria.
A source deviation triggers the smallest source-grounded repair. A discrepancy
that clears upon state restoration is accumulated-trajectory sensitivity on
that case; it does not explain final quality.

Promotion remains blocked by unresolved trained-quality evidence. Continuation
vetoes are mismatched inputs, failed exact-repeat/control checks, nonfinite
artifacts, missing decisive measurements or resource exhaustion. A candidate
discrepancy is a repair trigger, not an automatic continuation veto. No
scientific numerical default will change. No population ranking, final-error
envelope, native RNG equivalence or complete replication follows.

The exact author implementation and identical-state author repeat are the
relevant baselines. Heuristic filtering competitors cannot answer a translation
question and are excluded from this diagnostic. No performance promotion is
being sought.

## Ordered execution

1. Reconstruct unit01 FP32 fresh pretraining through update index 67 using the
   previously frozen data, order and clouds. Check every saved input hash and
   scalar loss against attempt05. Restore both engines to the source state at
   index 67 and compare features/logits, loss, raw and decay-adjusted gradients,
   moments and parameters. Retain complete replay arrays. Index 59's existing
   replay is a control, not evidence for index 67.
2. Restore the earlier trained checkpoint, replay updates 0 and 1, and localize
   its first gradient flag before any ancestor difference. Compare an evolving
   trajectory with the same-source-state replay. Preserve actual in-kernel
   sampling decisions. Only if a same-state gradient discrepancy persists,
   capture source/TF adjoints at the first affected angle operation and test
   the predeclared signed perturbations 0, +/-1e-8, +/-1e-7, +/-1e-6, +/-1e-5
   with both ordinary- and small-radius controls. Record representable changes
   and preserve IWSG stop-gradient semantics; ordinary finite differences of
   exp(logq-stopgrad(logq)) are not an admissible adjoint check.
3. Record the result and identify the next discriminating native-size validation
   experiment. Do not start full training inside this diagnostic allocation.

## Assumptions, bounds and review

The FP32 arithmetic threshold is inherited only as a localization trigger.
The exact-repeat controls and input hashes expose a misleading replay early.
Using common random numbers changes the native RNG mapping; this tests
conditional numerical behavior, not native seed-by-seed identity.
TF is GPU/XLA with TF32 off for reference comparisons, stable signatures and
verified memory growth. Author execution is explicitly CPU-only. CPU Python
orchestration and diagnostic serialization do not become numerical runtimes.
Capturing diagnostics must preserve the numerical result; otherwise repair
instrumentation before interpretation.

At most two pretraining launches and two trained-case launches, at most
20 minutes each, and at most one hour charged to EACH CPU/GPU resource.
Failed launches count. These are a new bounded follow-up inside the existing
campaign authorization, not a reopening of the previous exhausted attempt caps
or a fresh compute grant. Reconcile the ledger before every launch.
Starting balance: CPU 217986.948231 seconds; active GPU tranche 89039.550787
seconds. Output root:
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/gradient-followup-20261013.
Commands use the approved absolute interpreter:
 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_gradient_followup.py --stage pretraining
 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_gradient_followup.py --stage trained
Result: docs/plans/controlled-smc-kdm-gradient-followup-result-20261013.md.

Skeptical pre-run audit: the earlier index-59 replay was an inadequate proxy for
the later gradient flag; reconstructing index 67 repairs that coverage gap.
Saved data hashes, exact scalar trajectory reproduction and same-state
intervention directly answer the question. Existing common-evaluator
checkpoint quality gaps remain visible and cannot be erased by these tests.
Budget, source pin, environment exception and stopping conditions are explicit.
No material unexamined default is being promoted. Pretraining execution may
proceed after syntax and bounded-scope checks. The trained-case implementation
must receive its own focused code audit before its launch.

Trained-case implementation audit: the existing two-update source/TF trajectory
must reproduce prior comparisons and losses exactly before interpretation.
Full author parameters, moments and step are captured before each update;
the restored TF update consumes that exact state. Numerical capture-on/off
equality is required. The observed sampler is the already tested shared
factory; ancestry is captured inside the update. Source early stopping,
learning rate and clipping remain unchanged. Source/TF ordinary-start and
trained-start cases are retained; all prior adapter controls still run.
The existing precision-native FP64 random stream is retained solely for exact
historical reproduction; it is NOT a matched-FP32-to-FP64 sensitivity test.
Further matched-precision/adjoint probes run only if a same-state flag persists.
Syntax, bounded loop count, restoration and call-chain checks passed.

First trained launch failed in diagnostic location reporting: argmin returned
int64 while unravel_index dimensions were int32. Numerical capture equality
and the original two-update comparisons had passed before this reporting error.
Repair: explicitly request int32 from argmin. The focused CPU-only no-GPU
regression below checks the index calculation. This is a harness repair with
no change to the target, data, gradients or acceptance criteria. The failed
150.343406 seconds are charged; one trained retry remains.

## Conditional angle-adjoint stage, 13 October

The completed trained02 replay triggers the conditional stage: at trained
update 1 the same-source-state FP32 raw-gradient difference remains
1.580119133e-4 (scaled 4.9993), with zero ancestor disagreements. The ordinary
starting-state control passes. The minimum raw angle radius in the trained
source replay is 0.00351619, versus 0.850254 in the starting control.

First capture actual source adjoints through normalization and atan2 on the
saved case, preserving the original loss, detach points and gradients exactly.
Use the saved realized samples only for an explicitly conditional backward
reference; this cannot establish a native sampling distribution. Evaluate the
same local operations and upstream adjoints in GPU/XLA TensorFlow, at both
source and TF raw inputs. Compare with the derived Jacobians
  d atan2(u,v) / d(u,v) = (v,-u)/(u^2+v^2),
  d(q/||q||)/dq = (I - nn^T)/||q||.
These derivatives apply for positive radius and fixed supplied adjoints.
Record which discrepancy arises before the angle operation, in its local
backward arithmetic, or in the upstream loss derivative. Preserve complete
per-time arrays, including the first differing angle and the smallest radius.

Run signed additive perturbations 0, +/-1e-8, +/-1e-7, +/-1e-6, +/-1e-5 on
actual saved raw angle coordinates for the small-radius case and an ordinary
control. Record representable changes. Such internal interventions test
conditioning of the operation, not perturbations of the RNG distribution.
A later end-to-end intervention is warranted only if this smallest diagnostic
identifies a specific causal candidate; a local sensitivity match alone cannot
close final trained quality or attribute the whole gradient discrepancy.

Two angle-stage launches are allowed, each at most 20 minutes, within the
UNCHANGED one-hour cumulative charge to each resource for this entire follow-up.
The preceding stages consumed 544.989408 seconds each. The two trained replay
slots are exhausted and are not reopened. Source is CPU-only; TF local kernels
use GPU/XLA, stable signatures, memory growth and TF32 off. No numerical policy
changes or damping are introduced.

Skeptical audit: a generic finite-difference loss check would differentiate the
wrong function because of IWSG stop-gradient. Actual reverse-mode adjoints plus
analytic local Jacobians answer the operation question without that mistake.
The recorded original source replay must agree exactly before interpretation;
local forward/adjoint comparisons are explanatory and cannot substitute for
trained quality. Cross-precision comparisons must promote the exact FP32
arrays, not draw unrelated FP64 random numbers. This conditional diagnostic
may proceed after syntax, instrumentation and source-call-chain review.

Angle implementation review passed. Source anchors are the pinned
src/models/particle_transformer.py:112-156 and
src/models/kde_particle_filter/kde_particle_filter.py:1018-1023, 738-749;
TF anchors are bayesfilter/replication/kdm_neural_replay_tf.py:153-190.
The diagnostic obtains actual native-loop TF adjoints using transparent
identity taps. Original value derivatives remain identity; dummy variables
receive duplicate adjoints solely for observation and are not scientific
parameter derivatives. AST instrumentation is restricted to exactly three
value taps and one transition-call argument. A tiny CPU-only while-loop
regression preserves value 9 and original derivative 6 and observes the
expected adjoints [9,3]. Native loss, parameter gradients, raw proposals and
samples must remain bit-for-bit identical to the saved uninstrumented replay.
The source run uses unchanged author forward/backward with retained tensor
gradients and saved detached draw inputs; its loss, all raw gradients, raw
proposals and outputs must exactly reproduce the saved author case.

The signed panel perturbs each raw angle coordinate separately at the selected
minimum-radius location in each of the ordinary and trained controls. These
are conditional fixed-draw full backward replays, not native RNG perturbation
experiments. Both local FP32 and exact-promoted FP64 kernels are diagnostic.
No pfor, per-particle Python numerical loop, numerical protection or parameter
update is introduced. Complete derivative arrays and source checksums are saved.
Command:
 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_gradient_followup.py --stage angle

## Final conditional intervention, 13 October

Angle01 completes the actual-adjoint and signed-source panel. At the trained
minimum radius (time 11, row 36, particle 2), the actual TF-minus-source raw
angle difference is (-7.152557373e-7, 0). Source/TF local pullbacks agree
within 1.490116120e-8; analytic FP64 pullbacks agree within 5.551115123e-17.
The author itself changes its full parameter gradient by about 2.58e-4 for
a signed 1e-6 perturbation of the selected first angle coordinate.

Use the final angle launch for a causal diagnostic, with three TF arms on
each of the ordinary and trained saved states: unchanged; replace just the
selected angle pair by the saved source values; replace all raw angle pairs
by the saved source values. Replacements are diagnostic forward interventions
with the original derivative through the TF proposal retained explicitly.
They are artificial hybrids, not algorithms or proposed repairs. Both source
and TF remain at the saved original pre-state; no optimizer update is made.
The unchanged arm must reproduce the saved native loss, parameter gradients,
raw proposals, samples and actual ancestors exactly. Expose all actual
ancestor choices in each intervention and retain complete gradients.

Also replay the source with the actual selected TF-minus-source coordinate
difference, using the already validated detached source draw tape. The
selected second coordinate is identical, so perturbing the first reproduces
the entire observed pair discrepancy at this location. Require a zero-change
source control. This tests the effect in the reverse direction without
silently differentiating through an IWSG-detached draw.

The localization criterion is whether the targeted replacement removes the
existing arithmetic gradient flag, with no changed ancestor choices. Report
both the maximum discrepancy and the residual gradient-vector norm; partial
reduction supports only a partial explanation. All-pair replacement diagnoses
other continuous-input differences if the single location is insufficient.
The reverse source experiment is supporting conditional evidence, not a
requirement that its downstream trajectory equal native TF. None of these
comparisons qualifies trained quality or licenses altering source numerics.

Skeptical audit: radius alone was correlational; this intervention tests a
specific observed input difference. Full-angle forcing could hide unrelated
errors, so it cannot establish whole-program correctness and follows the
single-location test. Exact unchanged controls protect against altered XLA
fusion. Any changed ancestry is reported and prevents attributing that arm
solely to continuous angle conditioning. Fixed-draw source replay is explicitly
conditional. No new numerical default, margin, resource grant or launch slot
is introduced. Angle01 consumed 73.789083 seconds each; total follow-up charge
is 618.778491 seconds each. One angle launch remains under the existing caps.

Implementation audit passed before the final launch. The actual intervention
function passes a CPU-only extracted-function check: unchanged squared output
and derivative (36,24), forced-value output and original-path derivative
(25,20). Fixed masks and saved inputs are diagnostic-only, without optimizer
updates. The source zero-perturbation repeat is asserted. Parameter gradient
vectors align by parameter name. Source snapshots, input hashes, exact native
controls, actual ancestors, complete gradient arrays and trace counts are
preserved; the stage supervisor retains all existing resource and attempt caps.
Command:
 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_gradient_followup.py --stage angle --angle-intervention
