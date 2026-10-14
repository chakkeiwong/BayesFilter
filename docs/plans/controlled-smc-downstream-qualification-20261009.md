# Controlled-SMC downstream qualification and conditional repair

Active master: `controlled-smc-kdm-replication-master-program-20260928.md`.
Owner authorization: refresh, review and execute, reaffirmed 9 October 2026.
This phase uses the existing allocation, not an additional compute grant.

## Question and decision

Does the existing complete TensorFlow controlled-SMC implementation reproduce
the author's numerical results under the accepted 6 October criteria? Test this
before attempting another regression-solver repair. Exact regression
coefficients, matching LAPACK branches and identical resampling ancestors after
legitimate numerical divergence are explanatory diagnostics, not acceptance
requirements. Author failures are accepted baseline outcomes.

The initial candidate is the existing FP64 `LorenzRefinement.run`, centered QR,
unprotected precision, one repetition per graph call, GPU/XLA with TF32 off.
The reference is the frozen original `cSMC_Resample.m`, executed by Octave 6.4
and its pinned statistics/LAPACK dependencies. This does not establish original
MATLAB equivalence. Both independently fit their own clouds, start at the
author's APF policy, and use the author's adaptive stopping rule.

Source anchors: Heng et al. published Section 4.2 (quadratic least squares) and
Section 6.2/Figure 6 (`.localresources/papers/heng2020-controlled-smc-published.txt:1188`);
`StateSpaceModels/Lorenz/cSMC_VaryF.m:1` specifies data, covariances, N=512,
d=8, six observations and T=100 (101 observation times);
`cSMC_Resample.m:187` specifies backward fitting and
`cSMC_Resample.m:237` specifies the adaptive controller. The source's strict
ESS >90% comparison governs this replication. QR solves the same full
quadratic least-squares objective; its finite-precision trajectory is a
candidate for qualification, not assumed equivalent to normal equations.

## Execution sequence and bounded resources

1. **Q0, executable controls.** Freeze source and TF dependency hashes. Check
   instrumented native RNG/source execution against a pristine execution on
   the first seed of each pilot stratum, and compare initial APF log likelihood
   with the TF consumer. Check covariance/finite validity, device placement,
   graph signatures, trace counts and the adaptive stopping predicate. Keep
   independent seeds; perturbation directions are not replication units.
2. **Q1a, calibration.** Run 32 independent pairs at each of F=2.5, 4.8801,
   and 8.5 on the exact saved author data. Seeds start at 261009001 with
   disjoint blocks per stratum. This is development/calibration, never
   qualification. Measure complete-run costs, paired differences, source
   log-likelihood/likelihood moments, failure denominators and stopping counts.
   Q0/Q1a have caps of 3,600 CPU seconds and 3,600 GPU seconds including
   initialization. At most three localized infrastructure retries, each in a
   new output directory, are allowed within those caps.
3. **Q1b, precision and scope.** If the harness is valid, extend source-only
   calibration to the original 40-point forcing grid before freezing a full
   Figure-6 qualification. Preserve the central generating-parameter check
   separately. Compute a costed fixed sample size from source calibration and
   the available budget; paired pilot covariance is descriptive sensitivity,
   not permission to change margins. If the planned precision is unaffordable,
   report that explicitly and retain a bounded, clearly labelled partial-scope
   comparison rather than declaring full replication complete.
4. **Q2, untouched qualification.** Freeze candidate hashes, all grid cells,
   independent seed blocks, sample sizes and the interval implementation in
   `qualification-spec.json` before launching any qualification seed. Run the
   full native adaptive algorithm in both implementations. Analyze once at
   the frozen sample size. No repeated conventional-CI peeking until a pass.
5. **Q3, conditional repair.** Additional TF invalidity triggers localization
   of the first bad fit/forward operation using its saved seed and tapes.
   A demonstrated mean/variance discrepancy triggers matched-input versus
   autonomous comparisons to separate solver arithmetic, controller and
   forward-kernel errors. Implement a solver change only if implicated by
   this evidence. Previously used qualification seeds become development;
   a changed candidate needs a fresh qualification spec. A variance/mean
   interval that is simply too wide triggers a precision/budget assessment,
   not a claim that the algorithm or paper is wrong.

Total new phase ceiling: 36,000 CPU and 108,000 GPU seconds, all charged to
the remaining 242,911.285 CPU and 159,949.756 GPU seconds. Q1a is the first
bounded launch. Before Q1b/Q2, record the actual cost and remaining ceiling.
No full DGELSD port, protection/ridge, new model, KDM training or E0 work is
authorized by this phase. Such scientific changes require a separate decision.

## Evidence contract and assumptions

Promotion criteria are exactly the accepted mean-likelihood equivalence
interval within ±log(1.01), upper 95% variance(log Z) ratio <=1.05, and no
additional failures on author-valid qualification pairs. Use stable shifted
exponentials for likelihood means, not a percentage of mean log likelihood.
Use independent run pairs within each fixed grid cell. Report continuous
metrics on their explicit valid-pair denominators; source failures remain
recorded and cannot be silently imputed. If failures make the unconditional
mean undefined, qualify only the explicitly conditional valid-case estimand
and report the incomplete unconditional claim. Zero source variance leaves the
ratio unresolved until an independent absolute bound is established.

Q1 records influence-function standard errors and source-only independent-arm
sample-size approximations as planning diagnostics. Q2 must freeze an
appropriate paired interval procedure after inspecting source moments. Require
tail/moment adequacy; a wide or unreliable interval is inconclusive. No
population ranking is inferred from Q1. Failures of an otherwise valid candidate
are promotion vetoes and repair triggers, not continuation vetoes. Corrupt
data/provenance, wrong baseline, invalid RNG instrumentation, time/resource
limits and broken controls are continuation vetoes until locally repaired.

Runtime, memory, trace counts, ESS, fit residuals, mean(log Z), source relative
variance and the existing 1e-8 sensitivity experiment explain behavior; they
cannot replace the acceptance criteria. No unrelated heuristic superiority
requirement is added to author replication. Existing Gaussian exact controls
and fixed-policy source comparisons remain correctness evidence. This phase
does not close neuroscience/other paper experiments, MATLAB compatibility,
KDM, HMC, default readiness or superiority claims.

| Material choice | Provenance and status | Failure mode and earliest check |
|---|---|---|
| Model/data/N/stopping | Source baseline above | Wrong horizon/constants; Q0 APF parity and source assertions |
| Centered QR, no projection | Existing candidate, unchanged least-squares objective | Different numerical solution/failure; full-run Q1 and Q2, no blanket coefficient gate |
| FP64, TF32 off, GPU/XLA | Existing reference-precision candidate | Compile/device discrepancy; placement, controls, stable signatures |
| One run per graph call | Convenience for independent failure accounting | Slow but valid; measure compile and steady-state costs before grid |
| 32 pilot pairs per stratum | Bounded calibration choice | Imprecise tails; descriptive only, never qualification |
| Three pilot strata | Low/truth/high source range | Miss interior failures; full original grid required for full Figure-6 scope |
| Shared native RNG tapes | Variance-reduction coupling, native marginal laws | Tape order changes source behavior; pristine-source exact control |
| Accepted margins | Owner-approved 6 October, unchanged | Underpowered or unstable moments; source calibration and frozen final inference |

Pre-mortem: a passing three-stratum pilot could conceal an interior-grid
failure; successful old fixed-policy checks could conceal refitting errors;
mean log likelihood could mask mean-likelihood bias; dropping source failures
could change the estimand. The sequence and denominator reporting above prevent
those results from being called full replication.

Skeptical audit, 9 October: **PASS for Q0/Q1a execution only**. Baseline and
consumer are explicit, the three-stratum pilot is not promoted to full scope,
the source horizon/normalization are preserved, and failure/compute stops are
defined. Q2 is not executable until its data-independent fixed spec and interval
controls have been recorded. No extra human approval is needed within scope.

Artifacts: unique directories under
`docs/plans/artifacts/controlled-smc-kdm-replication-20260928/csmc-qualification-20261009-*`.
Each launch records commit, source/dependency hashes, exact commands, CPU/GPU
choice, seeds, timings, logs and all trial outcomes. The result note carries
decision and inference-status tables, alternative explanations and next action.

Execution review addendum: estimator controls pass exact identity, a known
multiplicative likelihood shift, known variance scaling, common-normalization
invariance and zero-variance rejection. Dispatcher discovery/accounting and
syntax checks pass. The new master command is `start-qualification`; it
uses a detached supervisor, explicit worker timeouts, versioned output,
incremental running charges and terminal charges without double counting.
The first serialization-only failure is preserved and charged; the checked
MAT reference decoder repairs it without modifying the numerical candidate.
The pilot GPU is the idle RTX 4080 SUPER (UUID in the spec); FP64/TF32-off
settings and memory growth are recorded. No free-memory reservation is claimed.

## Q1b executable refinement after the pilot

The pilot completed on 9 October (result note above). Continue the already
planned source-only calibration at the author's exact 40-point `linspace`
grid plus F=4.8801, with 32 fresh seeds per cell (1,312 runs). Octave itself
emits the grid to preserve source floating-point constants. Seeds begin at
261020001 with disjoint 10,000-wide cell blocks, excluding pilot seeds.
Use four CPU workers, one short-lived Octave process per 32-run cell to bound
the previously observed retained-closure memory growth. Each cell has a
150-second worker cap plus at most five seconds cleanup; 41 such cells cost
at most 6,355 worker seconds, within a 7,200-CPU-second Q1b cap. All charges
are included in the original 36,000-CPU-second phase ceiling. No GPU time
is used in this stage. Source mathematical code and criteria are unchanged.

Record every failure and pristine-source/RNG control. Random tapes need not
be stored for source-only precision calibration: exact seeds and source
dependencies reproduce them, and these runs never become qualification.
Commands and summaries remain versioned. The master command is
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py start-source-grid`.

Skeptical review: **PASS for Q1b**. The original parameter grid is retained,
not inferred from the three-stratum pilot. No candidate outcome chooses the
source sample size, no acceptance margin changes, all source failures remain
in denominators, and the previously observed long-process memory failure is
addressed by bounded worker lifetime. This stage estimates cost/moments only.

## Q2 fixed design after completed Q1b

Q1b completed all 1,312 source runs; 1,309 were valid. Failures: two at
F=2.5, one at F=7.423076923076923. All 41 pristine source/RNG controls passed.
CPU charge: 2,074.901118 seconds. Independent-arm variance sample-size
approximations range from 9,685 to 198,508 per cell. Equal-variance 90% power
is unaffordable in this phase; those unstable small-sample approximations do
not predict the benefit of pairing or substantially lower candidate variance.

Execute **320 fresh pairs per cell**, 13,120 total: the 40 exact author grid
points plus F=4.8801. This is a budgeted precision compromise, ten bounded
32-run chunks per cell and approximately a 1% upper extra-failure probability
if none is observed. It does not promise power for the 5% variance margin.
Expected cost: 20,750 CPU / 59,200 GPU seconds. Q2 hard caps: 32,000 CPU /
100,000 GPU seconds; all Q1 charges plus 300 CPU / 60 GPU seconds reserved
for engineering checks fit the original 36,000 / 108,000 phase ceiling.

Freeze qualification-spec.json before any run: hashes, grid, sample sizes,
margins, diagnostics, inference and seeds 271100001 + 10000*cell + repetition
(0..319). These exclude calibration seeds. Each complete source/port pair is
one independent unit. At most four source workers run concurrently, one
32-run chunk per process. One TF process handles one 320-run cell.
Each source chunk starts with a pristine-output/RNG control. Keep all tapes
and failures. Estimated tape storage: 218 GB. Require 280 GB free at launch;
stop new work below 50 GB free. No token/approval ceremony is added.

Compute inference once after all frozen cells finish: 20,000 paired
nonparametric bootstrap draws per cell, fixed analysis seeds [261109, cell].
Likelihood statistic: log(mean(Z_TF)/mean(Z_source)), using a common
exponential shift. Variance statistic: log(var(logZ_TF)/var(logZ_source)).
Use the envelope of bootstrap-percentile and paired influence-function normal
intervals: minimum lower, maximum upper. Likelihood uses two-sided 90%;
variance uses a one-sided 95% upper limit, exponentiated. Retain lower
endpoints for diagnosing adverse differences. Bootstrap indices select the
same run in both arms. All qualification statistics use TF FP64 through a
stable CPU/XLA graph, an explicit reference-statistics exception. No NumPy
enters qualification computation; Python combines Boolean decisions only.

Influence values are Z_TF/mean(Z_TF)-Z_source/mean(Z_source) for likelihood
and centered(logZ_TF)^2/var(logZ_TF)-centered(logZ_source)^2/var(logZ_source)
for variance. Divide their sample standard deviations by sqrt(n) for standard
errors. Influence denominators use population sample moments; variance ratios
use sample variances, whose common n/(n-1) factor cancels. These expressions
follow by differentiating log means and log central second moments. Neither
interval assumes normal logZ, but both are asymptotic and require moments.

Frozen empirical adequacy vetoes: at least 300 valid pairs, positive source
and TF variances, largest likelihood contribution <=5% in each arm,
squared-residual effective count (sum r^2)^2/sum r^4 >=30 in each arm, and
largest centered influence contribution <=10% of the empirical influence
variance for the corresponding statistic. These conservative diagnostics
prevent a few extremes from manufacturing acceptance; they do not prove finite
population moments. Failed adequacy makes the corresponding interval
inconclusive. Zero variance has no artificial floor. Bootstrap degeneracy
is inconclusive. Retain inadequate intervals labelled diagnostic.

Global acceptance requires every fixed cell to pass: an intersection-union
claim, with no separately advertised per-cell acceptance or optional stopping.
Tables contain pointwise/descriptive evidence. Report conditional valid-pair
continuous estimates where source failures occur; unconditional likelihood
claims remain unavailable there. No extra TF failure on an author-valid pair
is allowed. Report exact upper failure bounds within each stratum, never
pooling different forcings. Paper relative variance and stopping counts remain
explanatory diagnostics. Do not claim superiority, full paper completion,
MATLAB equivalence, or GPU-default readiness.

Source-output/RNG/APF/controller control failure invalidates the harness and
triggers repair. Additional TF invalidity rejects the candidate and stops new
qualification work for Q3 localization using its saved first failing state;
it does not reject the research direction. Infrastructure errors stop new
work with evidence retained; local repair/retry uses fresh versioned output
within the same total caps. No CI is issued for a prematurely stopped run.
Statistical rejection/inconclusiveness is assessed only at fixed final n.
Budget/storage exhaustion records a partial run and forbids acceptance.

Skeptical audit: PASS subject to executable controls before launch. Exact
baseline/grid/native controller and source failures are preserved; candidate
outcomes did not choose margins or eliminate regimes. Precision limitations
are explicit. Required checks: identity, common-normalization invariance,
known likelihood/variance shifts, paired-index preservation, zero-variance
rejection, tail-veto behavior, dispatcher accounting, bounded child cleanup.
No source-solver, ridge, projection, protection or E0 change is introduced.

Executable pre-launch review completed: PASS. CPU/XLA estimator controls,
exact pristine source/RNG plus known-valid/known-failed outcome regression,
GPU/XLA parity and complete failure-diagnostic serialization, one trace per
forcing, verified memory growth, AST checks and live process-group cancellation
all passed. Focused source runtime was 7.747 s; GPU regression 19.650 s; cleanup
1.003 s; estimator checks 2.599 s. A conservative 300 CPU / 60 GPU second
engineering allowance is now charged, including prior short checks. Fresh
seeds and every frozen source/candidate hash are checked again before launch.
The active allocation, rather than the larger historical balance, is now
reported by the master command.
