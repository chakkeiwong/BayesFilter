# Modern importance sampling and FAB training of the canonical IAF

## Intent and scope

Owner request: write a self-contained modern importance-sampling chapter in
`docs/main.tex`, derive FAB training of IAF for NeuTra, audit with MathDevMCP,
compile, and implement and execute a reviewed training plan. The subsequent
question asks whether the author's `lollcat/fab-jax` code can be reused.

FAB changes the training objective, not the canonical IAF architecture or the
posterior sampled by the eventual frozen-map NeuTra HMC. The application target
is the four-parameter q20/T30 UKF approximate posterior. It is not an exact
state-space likelihood or a request for a broad sampler comparison.

## Source and backend decision

The author repository has been downloaded, without installing or executing it,
to `.localresources/fab-jax-c9f9913`, pinned to
`c9f991366ca94b2678a7ed620bc9e12655cfef1d`. Its MIT license permits reuse and
adaptation with the copyright and license notice retained. Any derived source
must preserve that notice and cite Midgley et al. (2023).

Its sampling, replay and training modules are reusable independently of its
bundled coupling flow. They require JAX-traceable functions. Our canonical IAF
and UKF evaluator are TensorFlow functions. Direct unchanged import is therefore
not a drop-in integration. The default compatible route is a source-anchored
TensorFlow translation of the FAB components, retaining the canonical transport.
The owner explicitly selected a TensorFlow port preserving the existing IAF
and target. No JAX backend migration is authorized or needed.

## Work sequence

1. Inspect the FAB paper and the pinned author's sampling, loss, replay and
   transition source, recording exact ordering and departures from the paper.
2. Complete the literature chapter: importance-weight variance and dimension,
   mixture adaptation, defensive sampling, annealing/SMC, AFT/CRAFT, FAB,
   gradient-based mixtures, Stein methods, controlled diffusions, and empirical
   evaluation. Include all papers in the prior literature review; explicitly
   distinguish target-specific demonstrations from dimension-uniform guarantees.
3. Derive FAB's alpha-2 objective and its gradient, AIS weights, detached loss,
   replay correction, and its interface to the existing IAF and frozen NeuTra.
4. Run the installed MathDevMCP on bounded derivations, inspect limitations,
   repair material findings, compile the complete monograph and inspect the new
   rendered chapter.
5. After resolving the backend choice, implement the documented FAB variant,
   with independent known-law and derivative/weight/replay checks. Preserve the
   canonical map and batch-native target. Prepare the exact target-specific
   run configuration and compute budget before GPU research execution.
6. Execute the bounded calibration/training and required post-training checks;
   preserve all attempts and separate coverage, whitening, and downstream HMC
   evidence. Record result limitations rather than promoting a short test.

## Evidence contract

Engineering question: does the implementation compute the declared FAB
alpha-2 annealing and replay updates, while training the existing IAF?
Pass criteria: algebraic identities, known-law weighting, detached-gradient
checks, replay corrections, canonical map identity, and CPU/GPU compilation
parity appropriate to the chosen backend. Non-finite values, incorrect weights,
changed target, unreported clipping, scalar-row training or map identity drift
veto an implementation/run.

Scientific question: can FAB address missing posterior regions in the existing
reverse-KL-trained IAF? The comparator is the same canonical architecture and
declared initialization. Region occupancy, independent weighted target checks,
and the standard 1,000-point whitening diagnostic are separate measurements.
Training loss and acceptance alone cannot promote posterior readiness. A
candidate's missing coverage triggers the planned exploration/calibration
repair; it is not evidence against all FAB training. Broken math, corrupted
artifacts, target drift or exhausted authorized compute stop continuation.

The exact training ladder and numeric choices remain uncalibrated until source
and backend review is complete. No serious training is authorized by a number
invented in this document. The last preserved campaign handoff records 6.40
worker-hours and a Sep25 18:00 Asia/Shanghai deadline; reconcile the live ledger
before allocation, and charge failed attempts. The newer request authorizes a
bounded implementation/run, not unlimited compute.

## Skeptical pre-execution audit

Reviewed before implementation or research runs. Material stale-context defect:
shared `main` is at `de80aaff5`, has extensive unrelated dirty changes and is
behind the remote. The preserved canonical worktree is at `6ccfebc02`. Do not
reset the shared checkout or build on its old committed numerical source.
Use an isolated canonical baseline for execution and synchronize only owned
documentation/source changes.

The paper's Equation 7 uses a weighted sum; the author's no-buffer JAX function
uses the mean of normalized-weight times log-density, adding a factor 1/N.
The replay implementation caps correction weights at 10 by default, samples
without replacement and generates fresh AIS data with the pre-update flow.
These are explicit source choices, not proofs of unbiasedness or sensible q20
defaults. The run must distinguish the implemented variant and test gradient
scaling, clipping frequency and coverage before any training-quality claim.

The document/source-review phases pass this audit because they neither choose
a new backend nor spend research compute. The implementation and research
phases require the resolved backend and a completed numeric/budget annex.

## Approved backend and bounded execution annex

The owner selected the TensorFlow port. The implementation uses the pinned
author AIS/replay ordering and documented paper loss scaling. CPU-hidden
known-law, inverse-gradient, replay-correction, resume and XLA checks precede
GPU use. The execution checkout is `/tmp/BayesFilter-neutra-fab-20260925`,
based on canonical `6ccfebc02`; shared dirty work remains untouched.

This first execution is calibration and an exploratory training ladder, not
production/default promotion. Total cap is 14,400 GPU worker-seconds (4 hours),
inside the preserved 23,037-second balance, including failed attempts and
post-training diagnostics. At most 1,800 seconds may be used for initial
engineering/pricing/calibration; then at most three 3,600-second seeded workers;
the remaining 1,800 seconds cover localized repairs/diagnostics. CPU-hidden
mechanics are bounded to 600 wall-seconds. Deadline remains Sep25 18:00 Asia/
Shanghai. These are conservative convenience ceilings, not predicted runtimes;
reprice after the first pass. Fresh directories preserve every attempt.

| Choice | Provenance and status | Failure mode / early check |
|---|---|---|
| alpha=2; AIS without resampling | FAB paper Section 3 and source baseline | Requires finite integral p²/q; inspect tails before updates and weight concentration per pass |
| IAF: three stages, two width-16 ELU layers, cap 2, author masks/init | Existing owner canonical q20 profile | Capacity remains unproved; no architecture replacement or superiority claim |
| FP32 map/Adam, TF32; FP64 target | Existing target-specific policy | Compare fixed-input FP32/FP64 values/gradients; final represented map in FP64 |
| Initial proposal | Fresh canonical IAF with prior center/scale from the actual bridge, rather than reusing a collapsed map | Near-identity initialization is a tail/coverage hypothesis; invalid initial samples veto attempt, radial log(p²/q) checks can trigger repair |
| AIS batch 32 | Prior target pricing warm start and explicit native batching | Weight collapse; calibrate larger batch only when measured cost permits |
| 10 interior temperatures; linear beta schedule | Author example configuration, uncalibrated q20 warm start | ESS collapse/exploration failure; inspect all temperatures, then increase to 32 within remaining budget if warranted |
| HMC five leapfrog steps, one transition/interior beta, identity mass | Author example; no mass adaptation prerequisite | Poor movement/acceptance; initial epsilon .01 from author example is a pilot only |
| Step-size adaptation target .65, multiplier 1.02 | Author `sampling/mcmc/hmc.py` | Slow adaptation; preserve temperature-specific traces. Adaptive training weights are not unbiased evidence |
| Adam LR .001, (beta1,beta2)=(.9,.999), epsilon=1e-8 | Conservative optimizer warm start; not transferred as a tuned optimum | Check objective scale, gradient finiteness and actual update magnitudes; comparison to neighboring rates is conditional on pricing |
| Replay cap 12,800; minimum 1,280; four updates/pass | Author 400/40 batches and four updates; batch32 gives these counts | Filling may exceed available compute; price before replay phase. A no-replay paper baseline is a complete named FAB variant, not production proof |
| Correction cap10, no global gradient clip | Author replay cap, explicit diagnostic hypothesis | Record cap activation; cap dominates => repair, not improved training claim |
| Seeds0/1/2 with SHA256 role separation | Reproducibility convenience | Three exploratory fits do not establish a ranking |
| Standard 1,000-point final probe | Existing owner post-training requirement | Finite does not imply whitened, full coverage or converged HMC |

Before replay filling, run a fresh no-replay pricing/calibration pass. If the
author replay warm-up and useful follow-up do not fit, report under-budgeted
for replay and execute the named fresh-AIS baseline within the allowance. Do
not shrink replay until it masquerades as a production protocol. Hold out
diagnostic seeds; never use the final 1,000-point bank to tune.

Initial scientific criteria: numerically valid implementation and completed
diagnostics permit an exploratory candidate to be retained. Positive/negative
observation-weight occupancy, weights, residuals and losses are descriptive.
They cannot establish a statistically supported ranking or a posterior estimate.
A later posterior claim requires the established frozen-map tuner and full
posterior checks, outside this training-only evidence. Continuing training after
poor coverage remains permitted if the planned annealing/calibration repair
fits the same budget. Nonintegrable auxiliary target, changed posterior,
non-finite initial law, incorrect weights, or exhausted compute veto continuation
of that run; preserve the checkpoint and diagnose the specific cause.

Skeptical audit of this annex: prior-scaled initialization avoids silently
assuming a collapsed reverse-KL map covers all modes, but does not prove finite
alpha-2 divergence. Author numbers are labeled warm starts, with measured
cost/coverage needed before a longer phase. The plan does not mistake smoke
success or training loss for map/potential/HMC validity. Available GPUs may be
shared under memory growth; no existing process will be terminated. Pass for
bounded calibration, conditional on the focused engineering checks.

## Measured-cost repair: author Metropolis option

The first three-pass HMC calibration cost 410.65 seconds, with about 123
seconds per steady AIS pass (10 interior temperatures, five leapfrog steps).
Its weights often had ESS approximately one of 32 despite near-unit acceptance.
The .1 step-size pilot also retained severe weight concentration. The planned
32-temperature check at .07 follows this failure; it does not promote acceptance
or a single ESS value into coverage evidence.

The author also supplies `sampling/mcmc/metropolis.py::build_metropolis`.
Port this optional isotropic Gaussian random-walk transition with exactly the
same symmetric MH ratio. Its step size is constant across all mutations at one
temperature, then adjusted once using mean acceptance, as in that source.
The target and IAF stay unchanged. This reduces target evaluations from
five per HMC transition to one per random-walk transition; the resulting
exploration must be measured rather than presumed equivalent. Keep the
existing target's validity callback, even where it still computes scores.

A 10-temperature, one-mutation, epsilon .3 pilot is a coarse operational
hypothesis between the previously ineffective .01 displacement and the author's
unit Gaussian proposal. It is not a tuned setting. Price two passes and examine
movement, validity, weight concentration and regions. This repair stays within
the original 14,400-second total cap; transfer unused seeded-worker allowance
to calibration if necessary, recording the transfer. If replay filling now fits,
use the author's 40-batch minimum and 400-batch capacity, with four distinct
minibatch updates per pass. The three-seed exploratory allowance remains at
most 3,600 seconds each. The standard 1,000-point diagnostic remains mandatory
before map handoff. No production/default or statistical ranking follows from
these bounded fits.

Skeptical review: switching the AIS mutation is already an author-supported FAB
configuration. It changes efficiency and finite-sample exploration, so HMC
calibration does not validate the Metropolis settings. A matching-normal weight
check, compiled smoke, and target-specific pilot precede the training ladder.
The replay minimum is not reduced to disguise insufficient compute.

Calibration 03 completed in 434.407 worker-seconds (409.577 for its single
32-temperature HMC pass). ESS was 2.066 of 32 and maximum normalized weight
0.623, with no invalid proposal. These descriptive values do not establish
coverage or superiority over the shorter schedule. All three calibrations
used zero optimizer updates. The source-supported Metropolis repair remains
the next discriminating step. Its pilot command uses two passes, 10 interior
temperatures, epsilon .3 and a 350-second worker bound with a 400-second
external timeout. The bound is an operational ceiling; measured cost will
determine the subsequent replay allocation. The implementation also corrects
the frozen-map binding to the existing beta-1 adapter signature before any
training, leaving all historical calibration artifacts intact.

## Priced replay execution

The Metropolis pilot passed validity checks in 92.589 seconds: 41.597 seconds
for its first pass and 26.520 for its second. Initial AIS weights remain highly
concentrated (ESS 1.008 and 1.000 of 32); this is a repair trigger for the
planned adaptive replay fit, not evidence of map quality or grounds to discard
the entire research direction. All calibration attempts total 1,349.695 worker
seconds. Focused CPU-hidden tests: 13 passed in 48.13 seconds, including
compiled Metropolis and its once-per-temperature adaptation rule.

Launch one exploratory worker on each GPU, seeds 0/1/2. Each has a 3,600-second
internal ceiling and 3,650-second external timeout, at most 120 fresh passes,
10 interior temperatures, Metropolis epsilon .3, replay minimum 1,280 and
capacity 12,800, four updates per pass, LR .001. At the measured 26.520 seconds
per pass, initialization costs about 1,061 seconds; 120 passes cost about
3,182 seconds before compilation, replay updates and diagnostics. Therefore
the time bound can stop before 120 passes. A completed 120-pass run would
contain 320 optimizer updates after forty initialization passes; this is an
exploratory fit, not a calibrated stopping point or production qualification.
Three worst-case external timeouts plus existing calibration total
12,299.695 seconds, leaving 2,100.305 inside the 14,400-second cap for localized
repair or required diagnostics. Actual time, including failure and shared-GPU
slowdowns, will be charged.

Preserve an initial 1,000-point probe and the mandatory final probe with the
same held-out seed for a descriptive before/after check. Neither bank is
consumed by optimization, adaptation or candidate selection. Invalid initial
target/probe values veto that attempt. The budget loop reserves a measured
pass plus 180 seconds for final checks; external timeouts bound synchronous
calls. Final map export is loaded through the existing artifact consumer with
the beta-1 adapter signature, checking the actual binding. No HMC launch or
posterior promotion is included. The initial IAF and prior are comparators
for exploratory geometry only; no heuristic-superiority or default claim is
made, and a full decision-grade comparison remains outside this fit.

Skeptical audit: replay initialization now fits without reducing the author's
buffer minimum. Concentrated weights may still produce ineffective replay or
missing modes; correction caps, regions, update norms, and final geometry
will expose some of those failures without turning them into a ranking.
Source adaptation heuristics and integrability remain explicit limits. Pass
for these bounded exploratory fits and their full post-training checks.

## Source-initialization boundary repair

A second call-chain inspection of `fabjax/train/fab_with_buffer.py::init`
found `n_forward_pass = floor(minimum/batch_size) + 1`. The previous annex
described the configured 40-batch minimum but missed this extra batch in the
author initializer. Correct the port to require 41 initialization passes,
with the first optimizer update on pass 42. Thus 120 total passes can produce
316 updates, not the previously stated 320. The earlier numbers above are
preserved as the superseded estimate, not as active source claims.

Interrupted only the three owned workers before any optimizer updates.
Seed 0/1/2 preserved 3/3/5 complete initialization passes; partial in-flight
passes are discarded and their cost remains charged. Actual interruption
costs are 305.072, 258.948 and 318.202 seconds. Total spent including all
calibrations is 2,231.917 seconds. The original harness did not catch
KeyboardInterrupt, so its result status stayed `started`; separate
`interruption-note.json` files record the exact termination and costs without
rewriting those original results. The repaired harness records interruption
and resume provenance explicitly.

Skeptical audit: the preserved prefixes have zero optimizer updates and exactly
the same AIS law, map, target, random counter and step adaptation as the repaired
initialization. JSON checkpoint round-trip and next-step equivalence are covered
by the focused regression. They can therefore be resumed without warm-starting
from a wrongly trained map. Relaunch into fresh `replay-seed-{0,1,2}-r2`
directories with 117/117/115 additional-pass ceilings, retaining the one-hour
internal and 3,650-second external bound. At most 120 lifetime passes are
permitted. Even three full external timeouts keep cumulative allocation at
13,181.917 seconds, below 14,400; the remaining 1,218.083 seconds are reserved
for required diagnostic repair. This is a localized source-correspondence
repair under the unchanged scientific contract and total budget.

Additional independent CPU-hidden reference checks use fixed, nonadaptive AIS
kernels on a shifted normal with an analytically known alpha-2 normalizer and
first moment. A 4,096-row batch with a fixed seed and a five-estimated-standard-
error tolerance is a mechanics check, not evidence about q20 exploration. Its
purpose is to detect weight/ordering defects that matching-target constant
weights cannot expose. The broad tolerance avoids treating Monte Carlo noise
as a mathematical discrepancy.

## Independent posterior-weight verification

After fitting, evaluate each initial/final pair on a separate 1,000-point
normal-base bank, whose SHA256 role-separated seed is
`weighted-target-verification`. This bank never enters training or adaptation.
Use the same latent bank within each before/after pair, and concatenate two
20-row map outputs into a native 40-row target batch. The exact importance
log weight is `log_gamma(T(z)) + log_det(T,z) - log_phi(z)`. These weights
concern the declared posterior, not the auxiliary FAB `gamma²/q` target.
Preserve raw rows, finite/status checks, weight ESS and maximum weight,
unweighted and posterior-weighted sign fractions for coordinate 2, and
descriptive weighted means. This fulfills the independent weighted-target
check separately from the geometry probe. Unknown true region masses and
weight concentration forbid a coverage or posterior-estimation certificate.

Question: does the final proposal produce usable posterior importance weights
on a held-out sample, and does it represent both sign regions? Comparator:
its exact initial map. Finite target/proposal values and complete rows are
engineering pass criteria; missing rows, non-finite values or a changed target
veto that diagnostic. ESS, weight maxima and region fractions are explanatory
only and cannot rank the three stochastic fits. No true mode mass is assumed
to be one half. The sign partition is inherited from the original coverage
problem; it does not enumerate all possible modes.

Reserve at most 350 worker-seconds per pair (external timeout), using
`docs/benchmarks/diagnose_q20_fab_2026_09_25.py`. The worst-case 1,050 seconds
fit inside the 1,218.083 seconds left by the preceding conservative allocation.
Reconcile actual training time before launching. No training continuation is
smuggled into this verification. Skeptical review: an apparently balanced
unweighted cloud can still have a one-sided or concentrated weighted measure;
recording both exposes that distinction. Finite empirical weights do not prove
finite population variance or exhaustive coverage. Pass for the bounded
diagnostic with these limits.

## GPU replay compilation repair

The first replay call failed on GPU because this TensorFlow build has no
`StatelessShuffle` XLA_GPU kernel. This is an implementation-compatibility
failure, not evidence against FAB or the posterior. The earlier XLA tests
compiled AIS and the update independently but omitted compiled replay
selection. Add that case, and run a complete tiny GPU replay cycle before
resuming the expensive target. Replace the shuffle by sorting iid FP64
random keys, preserving the uniform-permutation law for distinct keys;
machine-precision ties are negligible at 128 selected rows. The preceding
Gumbel top-k weighting and without-replacement selection stay unchanged.

Seed 1/2 failed after 41 complete initialization passes, with zero optimizer
updates. Seed 0 was deliberately interrupted at 40 complete passes before the
same failure. Resume the last completed checkpoints, not the failure snapshot:
the latter can contain newly adapted steps from the unsuccessful partial pass.
All elapsed cost is charged. Total spent through those attempts is 5,982.020
worker-seconds. Preserve every result, including the compatibility tracebacks.

Bound the full GPU replay smoke to 150 seconds. Then resume each seed with
a 2,300-second internal bound, 2,350-second external timeout and at most
80/79/79 additional passes into `replay-seed-{0,1,2}-r3`. This keeps the
120-lifetime-pass ceiling. Three external timeouts, the smoke and the three
350-second weighted diagnostics would total 14,232.020 seconds including all
prior attempts, below 14,400. These are ceilings, not promised training counts.
The one-hour worker allocation is redistributed within the unchanged total
cap, with compilation failures explicitly charged.

The harness may reuse an existing initial 1,000-point report only after exact
equality of the map parameters, transport configuration, target signature and
probe seed. It records the source file and hash. This avoids repeating the same
unchanged-map diagnostic after infrastructure retries. Final-map diagnostics
remain mandatory. No data or promotion criterion changes.

Skeptical review: the fix changes a backend primitive while retaining the
sampling operation. The actual complete replay consumer must compile on GPU;
a CPU check or an AIS-only GPU pass cannot close this failure. Pass for the
focused repair and checkpoint continuation after that GPU smoke succeeds.

## Deadline reconciliation

The complete GPU replay smoke passed in 15.971 seconds, including actual
optimizer updates, distinct replay selection and a JSON checkpoint round trip.
The subsequent wall-clock check returned Sep25 18:52 Asia/Shanghai, after the
owner's 18:00 campaign deadline. No new q20 training worker was launched.
Routine implementation repair and final documentation continue; campaign
continuation awaits the requested deadline extension.

Actual charged GPU worker time is 5,997.991 seconds (1.666 hours), leaving
8,402.009 seconds (2.334 hours) within the 14,400-second FAB allocation. The
broader preserved campaign balance would be 17,039.168 seconds after these
charges, but it does not override the current FAB allocation or wall deadline.
Seed 0 has 40 complete passes and 1,280 replay rows; seeds 1 and 2 each have
41 passes and 1,312 rows. All three have zero q20 optimizer updates. Their
initial 1,000-point probes are complete and finite; no final trained-map probe
or posterior estimate exists.

`artifacts/neutra-fab-2026-09-25/prepared-continuation.json` contains exact
conditional commands for the three resumes and preserves checkpoint hashes.
They require the pending owner deadline extension before launch. The harness
now checks an expired deadline before GPU initialization and records any
explicitly supplied replacement deadline. Disabling the calendar deadline is
appropriate only for the requested owner extension to bounded completion;
the per-worker and aggregate compute ceilings remain in force.
