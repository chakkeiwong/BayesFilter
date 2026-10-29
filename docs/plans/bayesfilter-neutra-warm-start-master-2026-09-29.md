# Executable simpler-model NeuTra warm-start master

Owner request: build the proposed pipeline, review it thoroughly, and execute
on simpler models before q20. This plan authorizes the local execution under
that request; no q20 run or external publication is included.

## Question, decision and bounded scope

Can mode discovery and approximate forward initialization produce a useful
canonical IAF, and does classic RKL preserve its coverage? Separate optimizer,
sampler, training and posterior failures. The baseline is ordinary RKL;
exact-example forward fitting is a diagnostic control where available.
Gabrié, ordinary annealed SMC, waste-free SMC, AFT and CRAFT are distinct
initialization arms. Combined Gabrié/SMC is a later optional composition,
not needed to assess each mechanism. SMC² remains a separate state-space
extension: it cannot be tested by the static-density suite here.

Execution has a convenience-chosen ceiling of 7,200 GPU process-seconds on
the idle host GPU 1 and 7,200 CPU core-seconds, charged cumulatively across
attempts. This is a bounded first campaign within the user's execute request,
not a claim that two hours suffices for a scientific conclusion. Preserve
unique attempt directories, checkpoints, budgets and an executable next-phase
status. Stop at the ceiling and report under-budgeted questions honestly.
Other active jobs are not stopped. Preparation/reference processes hide GPUs;
GPU workers require memory growth before initialization and explicit XLA.

Promotion requires independently replicated, frozen-map posterior accuracy
and precision under the existing sequential NeuTra controller. A finite map,
low loss, source parity or passing 1000-point probe does not establish this.
Numerically invalid target/derivatives/weights, corrupted evidence and absent
required reference are continuation vetoes for the affected run. A missing
basin, poor fit or RKL collapse rejects that candidate and triggers a repair,
not rejection of all warm-start methods. All such repairs consume the budget.

## Execution phases

1. Build native batched analytic target fixtures: anisotropic Gaussian,
   unequal Gaussian mixture, invertibly warped mixture, author's wiggle and
   the ten-dimensional NeuTra-paper funnel. Preserve exact sampler/reference
   roles and independent validation/confirmation streams.
2. Implement JAMS-style multistart search with the TF/TFP optimizer; validate
   actual score/Hessian, merge numerical duplicates, preserve failures and
   uncertain curvature. Assess known-mode initialization and blinded search
   independently. Do not assign posterior weights from optimizer frequencies.
3. Port the selected flonaco global-MH/local-MALA numerical loop. Compare
   controlled random inputs against the pinned author's actual functions in
   the existing CPU PyTorch environment. Reuse the configured canonical IAF
   and existing weighted/RKL trainers; no historical map architecture.
4. Implement ordinary and waste-free annealed SMC with explicit stage target,
   weights, genealogy and correction. Implement AFT/CRAFT source-mapped
   transport and gradient/update ordering, with tests before enabling them.
5. Price compile/steady-state cost. Calibrate on each target using disjoint
   pilot/validation data: architecture width, learning rate, clipping and
   local-kernel step. Freeze nominated settings before independent runs.
6. Run each eligible warm-start arm then classic RKL, preserving before/after
   maps, optimizer states, standard 1000-point probes, independent physical
   region/reference diagnostics and charged costs. Continue the declared
   training ladder when still improving and budget permits. Diagnose failures
   at the stage that produced them; repeat only repaired runs in fresh paths.
7. For viable frozen maps, use the supported fixed-transport HMC tuner and
   sequential retained controller; no new mass adaptation. Archive warm-up,
   report reference error and precision, and assess independent replications.
   Insufficient budget is not permission to substitute a short chain for this.
8. Terminal result review and master refresh: decision/inference tables,
   candidate failures versus remaining viable mechanisms, next executable
   phase and remaining budget. No q20 or default-readiness claim.

## Numerical hypotheses and how to check them

| Choice | Provenance and justification | Earliest failure check / disposition |
|---|---|---|
| FP64 transport | Actual checked-out shared authority; preserving it avoids an unrelated core migration | Record actual dtype, TF32 flag and GPU timings; no FP32 performance claim |
| 3-stage author-masked ELU IAF, cap 2 | Canonical configured family, not new architecture | Inverse/Jacobian/gradient tests; width search stays in this authority |
| Widths 8/16 for 2D, 20/40 for 10D | Diagnostic capacity hypotheses, each a dimension multiple | Disjoint exact-example/reference validation; inability to use good data is a capacity/optimization repair trigger |
| Adam LR 0.001/0.003; moments 0.9/0.999; epsilon 1e-8 | Conventional initial hypotheses, not target-validated defaults | Objective/actual parameter-step and clipping/noise checks, then validation nomination |
| Batch 256 initially; 512 repair | Bounded powers-of-two hypotheses preserving native batch | Repeated batch-gradient noise and update stability; not an optimization claim |
| Forward/RKL rungs 256/1024/2048 | Geometric work ladder, not a convergence theorem | Common validation and continued-improvement flag; extend or report under-budgeted |
| Clipping threshold | Derive a candidate from pilot raw-gradient distribution, compare its actual Adam update to unclipped control | Clip fraction, update cosine/norm, finite loss; never inherit near-100% clipping silently |
| MALA step grid | Geometric grid priced/calibrated separately per target/bridge | Finite states, actual displacement, acceptance and region retention; acceptance is explanatory, not convergence |
| 1,024 SMC particles; 4,096 repair | Finite-cloud cost hypotheses | Independent region/reference errors and weight concentration; genealogy for waste-free retention |
| Temperature/stage and mutation budgets | Explicit finite configuration, refined if weight/movement checks fail | Endpoint must reach beta 1; failure to reach is not posterior teacher success |
| 3 independent confirmation seeds | Small first replicated screen; seeds are stream identifiers | Wide/insufficient uncertainty prevents rankings and scientific promotion |
| 1,000 post-training probes | Existing owner requirement | Numerical invalidity veto; score residuals are explanatory only |
| Gaussian covariance eigenvalues 1/25 | Moderate conditioning diagnostic hypothesis | Exact moments/normalizer/affine oracle |
| Mixture means ±5, unit covariance, weights 1:2 | Gabrié Appendix G.1, with declared centering/orientation | Exact responsibilities and half-space probabilities |
| Warp b=0.1, centering 26 | b is a diagnostic curvature hypothesis; 26 is the derived second moment | Exact inverse, unit determinant and transformed independent draws |
| Wiggle parameters | Exact pinned author's example | Refined integration with domain/tail checks; oracle arm is unavailable until its reference is verified |
| Funnel v~N(0,1), x=exp(v)z | NeuTra paper/source convention | Exact samples, neck/tail regions; do not initialize every walker at its atypical density mode |

Mode search uses independent starts from declared broad laws, with low-D
space-filling starts as a separate diagnostic. Endpoint/merge tolerances are
derived from reference precision and repeated convergence; failed searches
are retained. Full mode enumeration is not asserted on unknown targets.

## Skeptical pre-execution review

Reviewed wrong baselines, proxy promotion, stop conditions, cost fairness,
hidden defaults, stale context, environment mismatch and artifact adequacy.
Material findings and repairs built into the plan:

- Old q20 mode-discovery code uses known-region distances and a peak-height
  filter; it is not used as the new general discovery implementation.
- flonaco's standalone reverse branch detaches samples; use the existing
  checked RKL trainer instead. Forward samples remain deliberately detached.
- Source examples use ULA; the selected source route is corrected MALA.
- Ordinary/WF ordering differs from AFT/CRAFT; no generic engine may silently
  impose the same ordering or drop a Jacobian. Each has explicit source tests.
- Disjoint exact/reference banks calibrate the common benchmark capacity and
  optimizer protocol. Practical warm-start updates consume only their own
  teacher data or current walkers. This is an oracle-calibrated fixture
  protocol; its hyperparameter-selection procedure cannot be transferred to
  an unknown posterior without target-specific calibration evidence.
- The current checkout is FP64 despite newer text elsewhere. No precision
  change is assumed. Existing CPU Torch is a reference exception only.
- Particle ESS and Gaussian-looking samples can hide missing modes. Use
  reference region checks before and after RKL, with independent seeds.
- Every arm includes setup and warm-start cost. One expensive arm cannot
  exhaust the entire budget before ordinary controls are measured.
- Calibration and increasing training cost may exhaust this first allowance.
  The master must preserve an executable continuation, never manufacture a
  completed scientific comparison or skip the required downstream criteria.

This review supports bounded implementation and diagnostic execution; it
does not preapprove any method or numerical setting as successful.

## Artifacts and commands

Master entry point: `scripts/run_neutra_warm_start_master.py`; explicit
configuration and attempt outputs live under
`docs/plans/artifacts/neutra-warm-start-master-2026-09-29/`.
The script's plan/prepare/check/run/status modes preserve state across phases.
Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` for TensorFlow and the
existing Torch environment only for CPU author-reference fixtures.
Every run records the actual argv, source hashes/Git commit, environment,
seeds, device/memory/XLA status, wall/core time, plan and result paths.
Exact launch commands and measured pricing are appended before GPU training.

## Execution review and first GPU launch

CPU preparation completed for all five targets. The blinded search recovered
one Gaussian mode and two modes for each mixture. It recovered three modes
for the actual author-code wiggle density. That source fixture is treated as
multimodal; the paper's informal description does not override the checked
density. The funnel mode is at v=-9; its initialization uses declared scale
regions rather than putting all walkers at that atypical density maximum.
Wiggle domain/resolution refinement agrees in log normalizer and second moments
at the predeclared tolerance. Its reference remains numerical quadrature.

The expanded CPU suite passes 19 checks, including the actual pinned Gabrié
global/MALA function, target derivatives, valid zero weights, AFT's detached
free-energy derivative and pre-update selection, CRAFT's between-pass update
ordering, and real shared optimizer updates for all three training paths.
AFT's per-stage reset agrees with author aft.py lines 268-279 and 340-355.
These establish the tested mechanics, not full author-training equivalence.
The local controller replaces the author's flow with our configured IAF,
uses corrected MALA and explicit finite budgets, and omits target-energy terms
constant with respect to forward-KL parameters. The standalone source RKL
branch is not used. AFT/CRAFT kernels are source-mapped TF adaptations; their
whole implementations have not been proved equivalent to every JAX feature.

Review repaired zero-weight rejection and a clipping diagnostic omission.
Each training objective now records pilot gradients, gradient-noise ratio,
and the actual clipped/unclipped Adam-step norm ratio and cosine with persistent
scratch optimizer moments. An RKL phase recalibrates at its own initial map.
Assessment graphs are cached per map. No threshold is interpreted as proof of
training convergence. GPU calibration first performs deterministic graph/XLA
value-score equivalence and device checks before fitting. GPU process time
and CPU core time are both charged, including compilation and failed attempts.

First command (trusted GPU access, host GPU 1, memory growth before import):

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_warm_start_master.py run \
  --output docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1 \
  --through calibrate --targets gaussian
```

This launch prices the target-specific four-setting pilot. Subsequent launches
use the same master with target/arm filters or resume the entire matrix. The
source hashes in each manifest distinguish revisions; preparation is reusable
because its target definitions and independent reference streams are unchanged.

## Completed supervisor and downstream phases

The master now executes focused CPU checks, preparation, GPU calibration,
training, bounded repair and frozen-map qualification. Expected numerical
candidate failures are recorded separately from harness failures and do not
stop other candidates. A candidate rejected by the finite/coverage screen
receives a fresh run with twice the batch, teacher-cloud and training allowance,
up to five repairs per seed. These are convenience-chosen extension hypotheses,
not an optimization claim. All retries count against the original two budgets.
The job caps are 120 seconds for checks/preparation and 240 for calibration,
training or qualification. Gaussian measurements (58 seconds calibration,
23/40/47 seconds for the first RKL/oracle/Gabrié runs) motivate this first cap;
transfer to harder targets remains a cost hypothesis. A cap records incomplete
work, not method failure. Three attempts per job prevent endless infrastructure
retry. A local process lock and active-PID recovery prevent duplicate workers;
no approval-token or launch-authority machinery is involved.

Qualification uses the registry-supported fixed-transport tuner, its identity
latent mass, batched four-chain execution, and the shared sequential controller.
Initial epsilon .5 and L=(3,9,18) are search hypotheses, with pilots, measured
same-L repairs, fresh verification, evidence rungs (1,2), twelve candidates and
48 scheduler work units. Epsilon is bounded above by 2, the unit-Gaussian
leapfrog stability boundary used only as a search ceiling. A learned map need
not be Gaussian, so every actual pair must pass numerical checks. The 220-second
cooperative qualification cap leaves supervisor closeout time within 240 seconds.
Tuning uses 64 decisions per chain, the existing acceptance evaluator's minimum;
these decisions never become posterior draws. A verified member is selected
by declared order, with all other verified members retained in the tuning result.
There is no descriptive efficiency ranking.

Warm-up retains the repository's 2,000 minimum, recent 1,000 window and 1.05
R-hat screen. Retained sampling grows in 1,000-transition chunks to at most
10,000, with R-hat <=1.01, bulk/tail ESS >=400 and mean MCSE <=.03 posterior SD
for coordinates, coordinate squares and declared region quantities. The ESS
and precision numbers are first diagnostic information requirements; .03 is
approximately the independent-sample standard error at 1,112 effective draws.
Reference agreement requires errors within four combined estimated standard
errors plus .03 reference SD. This is a declared coarse operational screen,
not simultaneous or sequential confidence coverage. Both warm-up and retained
chunks are archived, and warm-up is excluded from all posterior estimates.
All additional quantities and checks must pass; the canonical controller owns
these decisions. A single successful replication cannot establish method ranking.

The skeptical review accepts these as bounded engineering and research screens.
It explicitly rejects interpreting a time cap, an incomplete temperature
ladder, pilot loss, or a Gaussian score probe as successful posterior estimation.
Successful requested subsets are labeled by their actual targets/arms/seeds;
the unexecuted matrix is still visible in the campaign configuration.

Wiggle coverage uses three fixed Voronoi regions around the maxima measured
in CPU preparation, with reference probabilities evaluated on the disjoint
sample banks. The old two-half-space screen was inadequate for three modes
and was replaced before any wiggle calibration/training run. Initial supplied
representatives likewise cover those three measured maxima. No exhaustive
mode-enumeration theorem is inferred. The Gaussian Gabrié seed-11 pilot used
reference examples in its clipping pilot; later Gabrié clipping pilots use
current walkers only. Preserve that first result as a fixture diagnostic,
not a demonstration of a wholly reference-free training protocol.

Whole-repository route-policy audit: the new qualification route is registered
and delegates to the shared retained-member controller. The broader check
also finds the pre-existing unregistered
`docs/benchmarks/run_q20_configured_hmc_2026_09_24.py`; this independent q20
migration debt is recorded and does not authorize modifying or executing it.

The first Gaussian HMC search exhausted twelve candidates without a verified
pair: the high-acceptance pilot steps were too small, epsilon 2 was too large,
and the geometric interior remained inconclusive under the first evidence cap.
This is a kernel-search repair trigger. The master now gives a failed
qualification one fresh search with up to 36 candidates, 144 work units,
three refinement rounds and evidence multipliers (1,2,4), with distinct
search/warm-up/retained streams and the same 220-second wall cap. It preserves
all first-search evidence. The numerical domain, target, frozen map, acceptance
and posterior requirements remain unchanged. These larger finite work limits
are a bounded search hypothesis; they do not guarantee a verified pair.

Source licenses are retained under `docs/reference/third-party/`: flonaco MIT,
annealed-flow-transport Apache 2.0, and particles' license. The code remains
a TensorFlow adaptation of selected source operations with our canonical IAF,
not a claim to reproduce every option in the original training programs.

First terminal outcomes illustrate why the promotion criterion is downstream:
plain RKL's separated-mixture maps can have small base-score residuals while
missing the minority basin. Such maps fail the physical coverage screen and
do not reach posterior promotion. A Gaussian RKL map passed the shared
posterior checks after the widened search, with 2,000 archived warm-up and
1,000 retained transitions per chain. This is one frozen-map replication.
The wiggle still lacks a verified kernel in the initial search/repair scope;
failed qualification also triggers the bounded training-repair queue. Teacher
particle counts, complete temperature ladders and map fitting remain separate
from those eventual inference checks.

Additional source checks execute the author's actual Croissants class in the
CPU reference environment and compare both wiggle values and scores. The
funnel normalization is checked independently through its exact noncentered
change of variables. All now-tested training blocks preserve native batches;
SMC temperature weighting/resampling has explicit compiled signatures. The
weighted-cloud clipping pilot now samples according to the same weights as
its forward-loss objective, tested using an extreme zero-mass particle.

The unequal-mixture exact-example arm passed coverage before RKL and lost it
after RKL. Repeating the entire warm start with more of the same refinement
would not isolate that failure. The repair therefore restores the preserved
before-RKL map, resets only the new RKL optimizer, doubles its batch and uses
the lower learning rate already in the calibration grid (.001 here). Its
256/1,024/2,048-update ladder is unchanged. This is a targeted schedule/noise
hypothesis under the same classic RKL objective; no forward penalty is added.
Other deficient warm starts retain the fresh doubled-budget repair. The
reused warm map is explicitly marked as prior evidence, not a new replicate.

The first warped-mixture oracle qualification reached 10,000 retained draws
per chain and passed R-hat, reference agreement, bulk ESS and mean precision.
Its sole failure was undefined tail ESS for soft responsibilities numerically
saturated at zero/one. Applying continuous 5%/95% quantile ESS to those
quantities was a diagnostic-design error. Diagnostic revision 2 retains
coordinate and squared-coordinate bulk/tail checks and assesses region mass
using the exact half-space indicator with the shared controller's event ESS.
Responsibility means remain in training/reference diagnostics. This changes
neither the target nor the HMC kernel nor the 400-ESS/.03-MCSE requirements.
The revision does not retroactively stamp the original qualification as passed;
a separately recorded reassessment/qualification is required.

Measured budget refresh: after twelve training jobs, the observed mean was
about 40 wall/GPU process seconds and 81 CPU core-seconds per training job.
The remaining 99 training jobs in the full three-seed matrix alone were
estimated at about 3,983 GPU seconds and 8,058 CPU core-seconds, excluding
qualification and repairs. That exceeds the first campaign's remaining CPU
allowance. These means are descriptive and later AFT/CRAFT may be costlier.
The active queue therefore completes all seed-11 training arms before further
qualification and prioritizes isolated RKL-collapse repairs before repeating
the already failed plain baseline. Other seeds remain explicitly unfunded
within this first allowance until actual remaining cost supports them.

The supervisor was refreshed at a worker boundary; a worker that had started
just before the refresh was recovered by PID and allowed to finish. No learned
map, chain or artifact was discarded. The local lock rejected a racing second
supervisor as intended. This is an orchestration change under the same targets,
methods, scientific checks, output root and total budgets.

Execution-target exception for teacher generation: the independent exact and
quadrature reference banks are CPU preparation work. Gabrié transitions and
SMC/AFT/CRAFT teacher generation run on the assigned GPU in this diagnostic
campaign because they exercise the same native-batch TF/XLA target/flow kernels
resident there, including differentiable stage maps, rather than transferring
each small mutation to CPU. This is a reviewed campaign-specific exception to
the default CPU sample-generation lane, not a new default or a claim of GPU
speed superiority for toy targets. Compilation and sampling time are charged.

AFT cost repair: its mixture and warped-mixture teachers reached only the
third of eight stages before the 240-second job cap. The completed timed-out
mixture attempt used 818 CPU seconds. GPU utilization was zero during one
slow interval while the Python process consumed several CPU cores, so the
next diagnostic isolates gradient, validation, snapshot, optimizer and mutation
calls over two partial stages, capped at 120 wall seconds. It is engineering
pricing only and cannot support a trained-map or method-quality claim.

The old orphan-recovery path lost the warped attempt's final CPU counter.
Its charge is explicitly estimated as 960 CPU seconds (240 seconds times the
maximum observed campaign CPU/wall ratio rounded up to four), with the launch
RLIMIT giving a hard upper bound of 1,874 seconds. The final result must retain
this accounting uncertainty. Future recovery reads the worker's /proc CPU
counters before termination; it must not silently substitute two TF threads
for the actual compiler's CPU usage. The completed scientific artifacts remain
independent of this cost-accounting defect.

## Terminal audit and bounded closeout

The pricing attempt completed in 26.03 wall/GPU seconds and 88.46 CPU seconds.
Its first gradient/validation/advance calls took 2.07/7.82/4.00 wall seconds;
the second took 1.51/.0015/1.88 seconds. This isolates startup costs but does
not reproduce the full 32-update AFT stage loop. The earlier mixture attempt
completed stage 4, and the warped-mixture attempt stage 3, before their caps.
Neither timeout establishes failure of AFT's mathematics. Repeated tracing,
compilation and host dispatch remain hypotheses, not diagnosed causes.

The current charge is 6,465.24 CPU seconds with one estimated attempt; charging
that attempt at its launch CPU limit gives 7,379.24 seconds. Thus the first
7,200-second CPU allowance cannot be shown to have any remaining capacity.
The conservative remaining value is -179.24 seconds, although the point
estimate is +734.76. No more scientific or GPU pricing jobs may launch from
this allocation. This is a resource continuation veto, not candidate or
research-direction rejection. GPU process usage is 2,969.54 seconds.

Skeptical closeout review found a material supervisor defect: launch decisions
and summaries used the estimated CPU charge without carrying its uncertainty.
Repair the supervisor to reserve the upper charge for launch decisions, expose
both values, preserve budget-stop status through phase wrappers, and enumerate
the unexecuted matrix explicitly. Missing orphan counters must reserve the
recorded launch allowance; observed process counters alone are only a lower
bound if the worker has not completed. Unit checks for those engineering
changes are routine CPU-only repair verification, separately metered and not
additional scientific campaign work. No new statistical interpretation is
authorized by those tests.

Complete a result review and resumable continuation specification before
requesting any additional campaign allocation. Prioritize the isolated
ordinary-mixture RKL-collapse repair and existing waste-free map qualification;
then diagnose the full AFT inner loop before retrying expensive AFT/CRAFT work.
The remaining seeds and discovery-initialized training are pending. An updated
program and partial scientific evidence are not a completed replicated study.

The next pricing revision must exercise all 32 updates and the three disjoint
AFT populations for the first two stages of the existing eight-stage schedule,
recording every gradient, validation, snapshot, update and advance. Synchronize
each timed call and record graph trace counts. This deliberately adds diagnostic
overhead and is not a throughput benchmark. Keep the existing 120-second cap;
an incomplete record still identifies the active call. Give this revision a new
job identifier so the completed one-update pricing attempt is never mistaken
for the stronger diagnostic. Implementation of this instrumentation is routine
engineering work; its GPU execution remains deferred at the campaign budget
boundary.
