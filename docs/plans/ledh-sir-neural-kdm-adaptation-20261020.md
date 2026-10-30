# SIR neural KDM proposal: implementation and diagnostic training

20 October 2026. User-selected continuation of Phases B/C in
`ledh-guidance-repair-plan-20261020.md`. Worktree and branch remain the LEDH
checkout. This is an explicitly new adaptation, not a replication claim.

Terminal status,21 October: implementation and the bounded diagnostic comparison
are complete. All16 initial neural fits and both128-update continuations finish.
The first optional512 attempt timed out; a reviewed recovery continuation now
completes512 for the nominated initialization, preserving exact state and all234
overlapping update records. Its validation loss is.707064; no later checkpoint
has an independent evaluation. The selected
32-update neural fit and fresh restricted cSMC/RBF adapters complete2496 valid
paired evaluation rows with unchanged package source. The neural primary
criterion fails; every learned candidate loses to the recursive Gaussian
heuristic in each required regime. No candidate is promoted. Independent
model-score accuracy and broader canonical tuning remain unevaluated.
Cumulative charge after recovery is21785seconds, leaving1981.197895057 of the
original allowance. The other initialization's optional512 rung is
under-budgeted. See the
[continuation result](artifacts/ledh-sir-neural-kdm-20261020/attempt-20-recovery/result.md).
See [terminal result](artifacts/ledh-sir-neural-kdm-20261020/attempt-01/result.md)
and [terminal review](artifacts/ledh-sir-neural-kdm-20261020/attempt-01/terminal-review.md).
The protocol below records the predeclared design; its later resource repairs
are linked in the result. No tuning may reuse this exposed holdout.

## Question and mathematical design

Can an observation-conditioned neural Gaussian mixture, trained through the
actual corrected LEDH terminal moments with simulated labels, improve those
moments relative to the restricted frozen-message adapters?

For parent a, let m(a,y), C(a,y)=L L' be the existing current-observation
conditional Gaussian. A neural network of the fixed observation sequence Y
returns displacement vectors delta[t,k] and log scales s[t,k]. Define

    q_eta(x|a,Y) = (1/K) sum_k Normal(x; m+L delta_k,
                                    L diag(exp(2 s_k)) L').

The physical proposal is (1-beta) q_LED H + beta q_eta. Sample one component,
then evaluate the FULL mixture density at the selected point. Retain the
physical model numerator f_theta(x|a) g_theta(y_t|x). The learned mixture is
an evaluable proposal, not an asserted future-likelihood message. No cSMC
twisted-target normalizer is needed for these ordinary filtering weights.
The existing cSMC-derived proposals remain comparators.

The first adaptation fixes categorical probabilities at 1/K and beta per
candidate. This is an explicit family restriction: only Gaussian locations and
bandwidths are learned. It permits exact conditional reparameterization for
training; no omitted derivative of a learned categorical probability exists.
Do not describe this as the author's complete IWSG learner. The network's
every parameter is trainable; no three-scalar adapter or frozen neural layer.
All observations are fixed during theta differentiation. The network has no
theta input in this theta0-scoped experiment. Analytical theta derivatives
include all dependence of m, L, the sampled point and both density components.
Training autodiff is with respect to network parameters only, not a replacement
for analytical LEDH model scores.

The terminal loss uses fixed calibrated first/raw-second features of the
actual final shared Contract-E cloud. Labels are simulated X_T; independent
particle randomness U ensures the conditional squared-error decomposition.
No model-score label, posterior moment oracle or heuristic score enters training.
Intermediate future-informed estimates are not called filtering moments.

Source boundary: inspected Younis--Sudderth Section 4 equations (14)--(15),
Section 5 and Section 6 equation (18), and local pinned author
`models/kde_particle_filter/kde_particle_filter.py:675` (sampling), `:738`
(IWSG injection), `:1092` (bandwidth prediction). The native experiment uses
three-state position/angle, learned posterior mixtures and NLL. The local
`BearingsNeuralReplay` preserves that native scope. Our eighteen-dimensional
conditional proposal, fixed probabilities, full physical density correction
and terminal moment loss are extensions. The paper is stored at the root
checkout's `.localresources/papers/younis-sudderth-2023-long-range-tracking.pdf`.

## Evidence contract and phases

1. Implement the neural mixture in the shared physical-proposal consumer.
   Independent TFP mixture-density comparison; zero displacement/scale reduces
   exactly to the existing conditional proposal; analytical theta tangent
   agrees with finite differences; actual terminal-loss gradients agree with
   directional differences and reach every neural tensor. Invalid scale/inputs
   reject. CPU tiny fixtures first, then GPU/XLA target-scope gradient smoke.
2. Calibrate fixed features on 256 new simulated paths. Train on fresh paths
   generated independently per update. Validation uses 32 fixed new paths and
   two paired particle draws; it selects moment loss only. Two capacities
   (K=2,width=16 and K=4,width=32), two Adam rates (1e-4,1e-3), two independent
   initialization seeds. Start with 32 updates, then 128, then 512 for viable
   nominated configurations as the measured budget permits. Each update
   averages four independent full-consumer losses; a native TensorFlow loop
   bounds memory across those four rows. This is not a NeuTra training lane.
   No silent scalar-loss substitution or loss on the uncorrected proposal.
3. Freeze network/checkpoint and beta before final evaluation. Diagnostic beta
   candidates .5 and .9 retain a positive LEDH component; beta=1 is a
   conditional-only comparator. If the initial gradient path is too expensive,
   record measured compile/runtime and stop as under-budgeted; do not replace
   it with a different loss or ineffective random-direction neural training.
4. Evaluate on 64 untouched paths, four paired particle draws. Comparators:
   plain LEDH, conditional-only, unguided physical mixtures, untrained neural
   mixture, current cSMC/RBF messages and their restricted adapters refitted on
   the new training/validation partitions. Do not reuse exposed old holdouts or
   select a comparator on the final data. The restricted adapter is a baseline,
   not a tuned classical optimum. If budget prevents its refit, report that
   missing comparison and forbid claims against the previously trained method.

Primary diagnostic success: paired terminal moment-loss reduction of at least
10% versus plain LEDH and the same untrained proposal, with a dataset-cluster
bootstrap 95% interval entirely below zero for the paired difference. Keep all
particle replications within each dataset cluster. Require separately reported
first and second moment losses. This can nominate an optional candidate only;
it cannot establish score improvement or default readiness.

Constructed heuristic adversaries: prior predictive Gaussian moments (ignore
observations); observation-only final Gaussian update (exploit direct SIR
observations); recursive Gaussian filter (exploit temporal information without
particles); conditional-only physical proposal (exploit current observations
without paying for LEDH draws). Evaluate them on low/high final observation
regimes and boundary/interior latent regimes. Thresholds come only from the
calibration partition. Any observed conditional heuristic underperformance is
a promotion veto, reported prominently, never a tuning target. Small regime
counts are uncertainty limitations, not evidence of equivalence.

Interpretation clarification, before the final partition is opened: pooled and
observation-defined label-risk differences have the original posterior-moment
MSE interpretation. Latent-compartment strata condition on the evaluation label;
their cross term need not vanish for the original filtering mean. Those strata
retain their predeclared predictive-loss heuristic veto, but cannot establish a
conditional posterior-moment MSE ranking. Proposition sg-subgroup-risk in the
manuscript derives the distinction. No selection rule or experiment changes.

Separate downstream model-score evaluation requires an independently validated
reference on those same observations; finite differences of the finite LEDH
value check derivative correctness only. Do not rename them a model-score oracle.
If no adequate reference can be constructed within the remaining budget, report
score accuracy as unevaluated and leave the score-improvement objective open.

## Defaults, resources and failure handling

Execution detail fixed before validation: screen all sixteen combinations of
two capacities, two beta values, two rates and two initialization seeds at32
updates. Within each initialization seed, nominate the lowest valid validation
loss for128 updates and then512 if measured costs fit the remaining budget.
Resume Adam state and the fresh-data update index exactly. Save every rung,
including the untrained checkpoint. Freeze the lowest validation-loss trained
checkpoint among completed rungs before opening the final partition. This is
model selection, not a statistically supported ranking of architectures.

Use the median final-observation average for low/high regimes. Define the
nearer-boundary regime by the lowest decile of minimum terminal compartment
counts in calibration; this describes a relative low-compartment tail, not
states arbitrarily close to the physical zero boundary. Report the numerical
threshold, regime counts and this limitation. Evaluate each observation regime
and each compartment regime separately, retaining dataset clusters. Do not tune
against the heuristic comparisons. Missing cSMC/RBF refits must be stated as an
incomplete comparison, never inferred from the new neural result.

Runtime repairs and exact charges after attempt07 are recorded in
`artifacts/ledh-sir-neural-kdm-20261020/attempt-01/compiler-cache-repair.md`.
These repairs preserve the objective, proposal and numerical estimator.

Retain diagnostic SIR theta0,T=5,N=64,FP64,TF32 off, shared UKF/GenUT/Contract-E,
OT(1,5), balance32,Kchunk=N and all physical pilot controls. They are a fixed
diagnostic baseline, not scope-tuned default evidence. Their inherited risk is
tested by endpoint validity, moment preservation and healthy baseline parity.
No numerical safeguard/default change is authorized by this plan.

Neural capacities and Adam rates are target-specific search hypotheses, not
transfers of the bearings architecture. Standardized observation inputs and
small random output-layer initialization keep the initial proposal close to
the conditional Gaussian; check actual scale extrema and loss at initialization.
Unbounded log scales preserve the stated family; non-finite or inadmissible
outputs reject, and a failed run consumes the budget. A stabilizing cap would
require a separately declared non-harm check, not silent clipping. Fixed mixture
weights may limit adaptation; K=2/4 and learned bandwidths test a bounded family,
not general neural KDM optimality. Adam defaults beta1=.9,beta2=.999,epsilon=1e-8
are optimizer hypotheses; record gradient norms, parameter coverage and the
two-rate comparison. No gradient clipping in this first diagnostic.

Seeds: calibration 263040001; training namespace 263041000; validation
263042001--263042032; evaluation 263043001--263043064; initialization
263044001/263044002. Particle streams use disjoint 283000000+ namespaces and
separate normal/component-uniform keys. Preserve all actual seeds in artifacts.

Budget available: 23766.197895057 GPU-process seconds. Reserve at most 1800 for
engineering/compilation, 12000 for training/search, 7000 for evaluation and 2000
for localized repair/reference work; total cap remains the available allowance.
Measure each launch including setup and deduct it once. Before a larger rung,
project cost from the smaller rung; do not launch a rung that cannot finish.
At most three localized repair attempts for one failure cause. Never overwrite
prior artifacts. GPU memory growth and escalated execution are mandatory.

Artifacts under `docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-NN`:
plan, exact command, source snapshot/diff, environment/device, seeds, timing,
gradient coverage, checks, trained parameter checkpoints, complete paired rows,
decision/inference tables. Entry point and exact CLI are recorded before launch.

First GPU/XLA smoke (CPU regressions: 23 pass including actual loss gradients):
`CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=2
TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2 PYTHONPATH=.
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -m
bayesfilter.score_study.neural_kdm_diagnostic_campaign --mode smoke --updates 2
--components 2 --width 16 --beta .5 --learning-rate .001 --seed 263044001
--budget-seconds 880 --output
docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-02`.
External `timeout 900` bounds compilation; stdout/stderr go to attempt-02.log.
This run establishes gradient wiring and cost only, not efficacy. An initial
CPU invalid-scale check exposed a downstream solve before rejection; rejecting
invalid scales now substitutes safe intermediates only after setting validity
false. All accepted finite-scale calculations are unchanged.

Attempt-02 failed during XLA gradient compilation in 25.54 seconds: existing
flow and higher-moment loops omitted static maximum iteration bounds, so XLA
could not size reverse-mode TensorLists. Add bounds equal to their existing
termination counts; this changes neither accepted values nor the number of
iterations. Charge 35 seconds including startup. Attempt-03 repeats the exact
smoke command with only the output suffix changed. Remaining campaign allowance
23731.197895057 GPU-process seconds before that retry. No training result was
produced by attempt-02 and no efficacy data were opened.

Attempt-03 also failed before an optimizer update, in 25.33 seconds. The earlier
diagnosis identified some missing bounds but was incomplete: the SIR transition
and its analytical tangent each have a four-step integration loop without a
declared maximum. Those loops are absent from the two-dimensional gradient
fixture. Add `maximum_iterations=4` to both existing SIR loops; preserve their
body and termination condition. The first repair therefore did not demonstrate
the root cause was fully resolved. Charge another 35 seconds including startup;
23696.197895057 seconds remain. Attempt-04 repeats the same GPU smoke with this
localized graph-bound repair and a fresh output directory. No scientific target,
training objective, or numerical policy changes.

Attempt-04 resolves the unbounded TensorList error, then exposes a different XLA
limitation: the reverse derivative of batched `GatherV2` cannot recover a constant
axis through the nested time-step function. Use equivalent explicit particle /
component indices with `gather_nd` for the four neural-component selections.
The density and selected points are unchanged. The SIR analytical-score CPU
regression passes (14.55 seconds). Attempt-04 takes 35.87 seconds; charge 45 with
setup, leaving 23651.197895057 seconds. Attempt-05 repeats the same smoke after
the equivalent indexing repair. This is the first repair for this separate cause.

Attempt-05 passes the gather operation but fails at a concat-gradient constant
axis in the same nested time-step derivative. This identifies a function-boundary
problem, not a bad component-selection formula. Revert the gather rewrite;
inline the SAME shared `time_loop_body` in its enclosing traced while loop.
Retain the separate concrete function only for constructing diagnostic-buffer
shapes. There is one numerical implementation, with unchanged recurrence,
diagnostics and analytical score. This allows XLA to see constant shape/axis
arguments during reverse differentiation. Attempt-05 takes34.25 seconds; charge
45, leaving23606.197895057 seconds. Seven neural CPU regressions pass after the
gather rewrite; repeat the consumer gradient and physical tests after inlining.
Attempt-06 is the second repair of the nested-gradient constant issue, using the
unchanged smoke command and a fresh directory.

Attempt-06 passes the actual SIR GPU/XLA endpoint: both optimizer updates are
valid, all six parameter tensors change, all gradients are finite, and the
source snapshot is unchanged. Times179.02/157.88 seconds, total343.21; peak
TensorFlow allocation1.131GB. Charge355 including startup, leaving
23251.197895057 seconds. CPU numerical regressions pass; the call-chain spy now
correctly expects one shape-template trace plus one execution-body trace. Its
focused rerun passes. This changes tracing count, not time-step execution count.

The observed repeat-update cost projects at least11.2hours for even the
eight-configuration32-update rung, before validation; that cannot fit the
remaining6.46hours. Before declaring the protocol under-budgeted, attempt-07
profiles the second of three otherwise identical updates and records function
trace counts plus XLA compilation-cache diagnostics. This distinguishes repeated
compilation from irreducible numerical runtime. It is an engineering diagnostic,
not a smaller efficacy campaign. Use the explicit RTX4080SUPER UUID to avoid
confusing CUDA ordinal0 with nvidia-smi index0. External cap900seconds, internal
880. Keep all numerical settings, data and seeds unchanged. Source instrumentation
adds synchronization before timing and host-side profiler/trace metadata only.

Attempt-07 completes three valid updates in497.72seconds. Both TensorFlow trace
counts remain1, but the XLA device cache misses on every update and spends about
2.6minutes compiling each executable. Data generation hits its cache. This is
an engineering repair trigger, not evidence that the estimator intrinsically
costs158seconds per update. CUPTI GPU tracing is unavailable (error999); host
compiler logs provide the positive evidence. Charge510seconds, leaving
22741.197895057 seconds. Attempt-08 inspects unoptimized HLO and input signatures
with `attempt-01/inspect_sir_compiler.py --output .../attempt-08`, cap180seconds,
same GPU UUID/environment. No optimizer updates or scientific claims. Identify
which data inputs are being treated as compile-time constants before changing
function boundaries. The full diagnostic source is saved beside this plan's
artifacts; no TensorFlow installation or compiler flag is modified.

Continuation vetoes: wrong conditioning/importance density/derivative; label
leakage; missing gradients; invalid baseline; corrupted evidence; or exhausted
budget. A candidate loss regression triggers the next capacity/rate/budget rung,
not rejection of the scientific idea. NaNs trigger localization before retraining.
Validation loss is selection evidence; SPD margins, ESS, mixture occupancy and
runtime explain behavior. Heuristics veto promotion only.

Skeptical audit: the design avoids substituting a filtering density for a future
message, trains the actual corrected estimator, and retains the full mixture
denominator. Fixed categorical weights make its reparameterized derivative
well defined. The main risks are reverse-mode graph cost, weak finite-particle
training signal, insufficient capacity and the lack of an independently accurate
SIR score reference. Earliest checks are exact consumer gradients and measured
GPU compilation before any campaign. Passing this plan does not assume benefit;
missing training rungs or references must remain explicit incomplete evidence.
