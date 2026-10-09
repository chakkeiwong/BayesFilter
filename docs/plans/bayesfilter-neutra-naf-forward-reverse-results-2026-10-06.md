# NAF approximate-forward-KL to reverse-KL execution

## Completed continuation: revised intermediate criterion, unchanged final screen

The owner authorized replacing the strict intermediate accuracy requirement
with a warm-start suitability rule and continuing execution. The revised
criterion and review are in
[the repair plan](bayesfilter-neutra-forward-warm-start-criterion-repair-2026-10-06.md).
Forward feature z is now explanatory. The forward map must still have finite
calculations, a complete finite probe, a reloaded checkpoint, adequate component
mass and the inherited gross-fit guard. Every component must retain at least
half its reference mass; this is an explicit undercoverage heuristic. The final
RKL distribution screen is unchanged.

All six calibration and six fixed cases pass the revised pair criterion, with
the selected recipe frozen. Fixed forward maps retain at least .9801 of each
reference component mass. Their earlier failures remain recorded as failures
of the strict forward-only diagnostic. This reassessment is retrospective;
original worker results and manifests have not been modified. The controller
archived its v1 state/result/selection and wrote
[the criterion revision](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/forward-reverse-criterion-revision-v2.json)
with result hashes, old/new decisions and calibration rechecks.

**All 12 reserved random cases passed.** Their native teachers, forward warm
starts and final RKL maps each passed their respective screens. The queue is
complete and idle. These random results were obtained prospectively under v2;
the six earlier fixed-case reassessments remain retrospective. All 12 random
forward maps also happen to pass the old full accuracy diagnostic, which remains
recorded. No final threshold or training recipe changed, and no holdout was
retuned or rerun.

| Random geometry | Fitting seed | Forward maximum feature z | Final maximum feature z | Final estimated forward KL (nats) | Warm start / final |
|---|---:|---:|---:|---:|---|
| Two modes, 1103 | 51 | 4.221 | 2.282 | .002527 | Pass / Pass |
| Two modes, 1103 | 52 | 2.616 | 3.000 | .004778 | Pass / Pass |
| Two modes, 1103 | 53 | 2.336 | 2.706 | .004508 | Pass / Pass |
| Two modes, 1104 | 51 | 1.780 | 2.491 | .003351 | Pass / Pass |
| Two modes, 1104 | 52 | 2.403 | 2.847 | .002830 | Pass / Pass |
| Two modes, 1104 | 53 | 4.191 | 1.883 | .004061 | Pass / Pass |
| Three modes, 2103 | 51 | 3.289 | 2.276 | .007068 | Pass / Pass |
| Three modes, 2103 | 52 | 2.307 | 1.927 | .005692 | Pass / Pass |
| Three modes, 2103 | 53 | 2.615 | 2.770 | .006707 | Pass / Pass |
| Three modes, 2104 | 51 | 2.453 | 2.689 | .008358 | Pass / Pass |
| Three modes, 2104 | 52 | 2.791 | 1.831 | .017656 | Pass / Pass |
| Three modes, 2104 | 53 | 4.105 | 3.979 | .011442 | Pass / Pass |

The smallest forward component-mass ratio was .9182, well above the declared
.5 warm-start bound. Final absolute component-mass discrepancies ranged from
.00465 to .03028. These are observed finite-sample summaries. Estimated forward
KL decreased after RKL in 11 of 12 cases; for 1103/seed 52 it increased from
.003428 to .004778. No predeclared uncertainty analysis establishes a ranking
or systematic improvement from these differences. The primary result is that
the frozen procedure retained all 12 cases under the declared screens.

The 1,000-point probes also show a remaining geometry limitation. Across the
12 final maps, median score-residual norms ranged from .0967 to .3885, but
their 99th percentiles ranged from 10.38 to 67.65 and maxima from 38.15 to
324.90. These probes use standard-normal base draws. The results do not show
uniform whitening; localized large score residuals remain. Finite residuals
are explanatory under the declared contract, so they do not retroactively
reject a passing density/coverage candidate. Their effect on HMC stability and
mixing still needs downstream testing. This is a bounded two-dimensional
training result, not q20 transfer or HMC readiness.

Verification passed 41 criterion/controller checks and five actual CPU numerical
call-chain checks; a fixture tuple/list comparison was the only initial test
failure. The master resumed under trusted GPU permission using
`bash scripts/run_neutra_scientific_campaign.sh forward-reverse` and source
`source-fa00123930f45395` (Git base
`70a6d7e9611a9f859a2519f37f236b61cac35c1e`, dirty source preserved by snapshot).
NAF training ran on physical GPU 2 with verified memory growth, batches of 64,
FP64 under the existing diagnostic exception and XLA. Each CPU teacher used two
workers with GPUs hidden. The frozen recipe remained `smc/native_0`, 8,192
forward updates at .001 plus 8,192 at .0003, followed by 256 RKL updates at
.0001. Every training update was finite and none were clipped.

The fresh [terminal audit](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/forward-reverse-terminal-audit-r2.json)
checks all 48 scientific workers from calibration through random tests: 936
artifact hashes, 1,439 source hashes, 78 complete finite 1,000-point probes,
memory-growth/device/XLA provenance and unchanged final thresholds. The 24
original result hashes recorded at criterion revision still match. A replay of
the frozen controller using a copy of the completed state and the actual saved
evidence returned zero and launched no workers; the live state was unchanged.
The [saved audit program](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/forward-reverse-terminal-audit-r2.py)
records the exact check. The prior r1 audit remains preserved.

The random stage consumed 5,905.96 GPU-process seconds (1.64 hours) and
6,994.74 CPU-core seconds (1.94 hours), over approximately 102.6 wall minutes.
The criterion tests separately consumed 94.88 CPU-core seconds, including the
initial fixture failure. The [accounting check](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/forward-reverse-terminal-accounting-r2.json)
verified one matching local and shared charge for each of the 48 workers, no
unsettled charges, and the cumulative remaining budget: 111,947.28 GPU-process
seconds (31.10 hours) and 104,304.11 CPU-core seconds (28.97 hours). Previous
attempts remain charged. No fixed fit or calibration worker was repeated.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain the six fixed pairs under v2 | Warm-start/final criteria pass 6/6 | No revised pair veto | Reassessment followed inspection of their results | Preserve original evidence and retrospective label | Fresh fixed-case confirmation under v2 |
| Retain all 12 randomized pairs | Prospective warm-start/final criteria pass 12/12 | No teacher, training, mass, feature or finite-probe veto | Four geometries, three fitting seeds each; exploratory finite-sample screens | Use preserved maps for separately planned downstream validation | Universal transfer, exhaustive discovery or posterior correctness |
| Keep geometry quality distinct from density fit | Density/coverage screen passes | Large finite score residuals are explanatory, not an added veto | Rare high-curvature regions may limit HMC | Assess frozen-map HMC with target-specific tuning and convergence checks | Uniform whitening or HMC readiness |
| Close this executable phase | 48 workers complete; audit passes | No infrastructure, evidence or budget veto | Subsequent scientific direction is outside this phase | Preserve idle terminal state and complete notes | Further workers needed to finish this queue |

| Inference status | Completed continuation evidence |
|---|---|
| Hard veto screen | All 12 randomized teachers, warm starts and final maps pass; all updates finite, zero clipping, complete probes. |
| Statistically supported ranking | None. The frozen recipe remains viable under this bounded screen. |
| Descriptive-only differences | Per-seed KL, feature z, mass errors, score-residual quantiles and runtime. |
| Default readiness | NAF remains the owner-selected architecture for this study; TF32, q20 and downstream posterior readiness are not established. |
| Next evidence needed | Downstream sampling validation with frozen maps; broader independent geometries and uncertainty analysis for comparative claims. |

Post-run review: the strongest alternative explanation is that four favorable
two-dimensional geometries and finite diagnostic samples leave rare regions
and wider transfer failures undetected. The large tail score residuals make
that limitation concrete. A failed independently tuned downstream sampler,
unseen-geometry failure, or evidence that the teacher misses relevant regions
would overturn a stronger general-readiness interpretation. The actual supported
conclusion is narrower: approximate native-teacher fitting followed by the
frozen short RKL phase passed the declared coverage and density screens here.
The original two fixed forward rejections were failures of an intermediate
accuracy requirement; they were not final-map failures or infrastructure errors.

## Historical completed round: strict intermediate prerequisite

The master completed six calibration pairs and six fixed-target pairs (24
workers). No worker is active. All native teachers passed. Calibration selected
the first common passing recipe: `smc/native_0`, 16,384 forward updates, then
256 reverse updates at .0001. All three .0001 reverse rungs passed all six
development fits; selection of the smallest rung was predeclared, not a claim
that it is statistically best.

**All six fixed-target final maps passed after reverse KL.** Four of six
forward-only checkpoints passed. The two remaining forward checkpoints failed
the feature screen, so the pair-level rule `forward_passed AND reverse_passed`
rejected those pairs and blocked random-geometry testing. The final maps did
not fail; it would be wrong to describe the result as RKL coverage collapse.

| Fixed target | Seed | Forward feature z | Forward screen | Final RKL feature z | Final screen | Pair rule |
|---|---:|---:|---|---:|---|---|
| Unwarped | 51 | 5.555 | Reject | 2.058 | Pass | Reject |
| Unwarped | 52 | 3.965 | Pass | 3.073 | Pass | Pass |
| Unwarped | 53 | 2.203 | Pass | 2.813 | Pass | Pass |
| Warped | 51 | 4.038 | Pass | 2.939 | Pass | Pass |
| Warped | 52 | 3.223 | Pass | 2.747 | Pass | Pass |
| Warped | 53 | 7.786 | Reject | 3.097 | Pass | Reject |

The two forward rejections are responsibility-weighted standardized first-moment
discrepancies, not missing mixture mass. For unwarped seed 51, the first
component's first-coordinate feature was .07830 versus reference .007684
(combined SE .012712). For warped seed 53, the second component's second
coordinate in unwarped coordinates was .13963 versus -.001887 (SE .018175).
Their maximum mass discrepancies were .00398/.00401. After RKL the corresponding
whole-map feature maxima pass. The six final forward-KL estimates range from
.00361 to .00803 nats, descriptively; all six are lower than their own forward
checkpoint estimates, but no paired uncertainty analysis was declared to rank
or promote a general improvement claim.

All training updates were finite and none were clipped. The terminal check
verified 528 artifact hashes and all 54 complete finite 1,000-point probes:
[terminal audit](artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/forward-reverse-terminal-audit-r1.json).
Cost for the 24 workers: 6,577.48 GPU-process seconds (1.83 hours),
7,841.23 CPU-core seconds (2.18 hours). Remaining: 117,853.24 GPU-process seconds
(32.74 hours), 111,393.73 CPU-core seconds (30.94 hours).

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain all six final maps as screen-passing candidates | Final endpoint screens pass 6/6 | No final finite/feature/mass/probe veto | Few fitting seeds and exploratory finite-sample checks | Review downstream adequacy separately | Posterior convergence or HMC readiness |
| Stop before randomized holdouts under the declared plan | Both-endpoint rule passes 4/6 | Two forward-only feature vetoes | Whether a strict intermediate fit criterion answers the approximate-warm-start question | Review the scientific role of that intermediate gate before changing the contract or repairing training | Failure of all final maps or NAF direction |

| Inference status | Completed-round evidence |
|---|---|
| Hard veto screen | Two forward checkpoints fail z<=5; all six final RKL endpoints pass. No worker crash or incomplete probe. |
| Statistically supported ranking | None established for the full procedure; calibration chose the first eligible setting. |
| Descriptive-only differences | Per-seed KL reductions, feature discrepancies, mass errors and costs. |
| Default readiness | Scoped NAF architecture remains the owner default; randomized transfer and downstream inference are untested. |
| Next evidence needed | An explicit decision about intermediate versus final criteria, then fresh evidence under any revised protocol and untouched random geometry tests. |

Post-run review: the completed data expose a tension in the plan. The overall
question permits an approximate forward warm start that reverse training may
repair, but the current rule demands a fully screen-passing intermediate map.
This is a documented protocol prerequisite, not an infrastructure fault. Do not
silently relax it after seeing holdout outcomes, relabel these six cases as
untouched under a revised rule, or repeat the same command expecting progression.
Preserve the failures while reviewing the criterion against the research intent.
The strongest alternative explanation to a broadly effective procedure remains
favorable finite seeds and insufficiently sensitive exploratory screens. Random
geometries and downstream inference remain untested.

## Why the two forward checkpoints failed

The checked failure is a within-component first-moment bias. With unwarped
coordinates U, component responsibility r_k(x), center mu_k and scale sigma_k,
the evaluator uses f_kj(x)=r_k(x)(U_j-mu_kj)/sigma_k. Under the exact mixture,
E_p[f_kj]=0: multiplying the mixture density by r_k leaves the k-th weighted
Gaussian, whose centered first moment integrates to zero. The warp has unit
Jacobian and is inverted before forming this feature. This is therefore a
directly interpretable location diagnostic, not an unexplained proxy.

The reported screen compares the sample feature mean from 2,048 iid map draws
to the mean from 32,768 independent exact draws, divided by the square root of
the sum of estimated variances of these means. `mean_variance` computes
sum((f-mean(f))^2)/(n(n-1)), which is the usual unbiased iid variance-of-mean
estimate. Neither the teacher particle count nor its ESS is substituted for
the map/reference sample sizes. The cutoff is the predeclared exploratory
z<=5 screen, not a proof of accuracy for passing maps.

| Failed feature | Teacher train mean | Teacher validation mean | Forward map mean | Exact-reference mean | Combined SE | z |
|---|---:|---:|---:|---:|---:|---:|
| Unwarped s51: component 1, coordinate 1 | -.001283 | -.002031 | .078302 | .007684 | .012712 | 5.555 |
| Warped s53: component 2, coordinate 2 after unwarping | .004326 | -.003922 | .139628 | -.001887 | .018175 | 7.786 |

Dividing each feature mean by its corresponding component responsibility mass
gives standardized within-component means of .2324 and .2078 for the forward
maps. The teacher training banks give -.00383 and .00646. After 256 RKL updates
the same map diagnostics give .01393 and -.00096. The teacher banks thus do
not exhibit the large observed shift; both component masses remain close to
their target values. The evidence localizes the discrepancy to the fitted map
rather than reproducing a large bias already present in these teacher moments.
It does not certify every aspect of the finite teacher distribution.

The forward code optimizes the finite empirical objective
Lhat_F(theta)=-sum_i wbar_i log q_theta(x_i). It samples weighted indices in
batches of 64, takes 8,192 Adam steps at .001 and 8,192 at .0003, then saves the
last iterate. It does not perform a final learning-rate decay to zero, average
parameters, select among late checkpoints using independent teacher validation,
or test a full-bank gradient for stationarity. Validation cross entropy is
computed after the endpoint is selected. The last four block-average forward
losses fluctuate (3.4860/3.4791/3.4875/3.4807 for the unwarped case;
3.4776/3.4690/3.4711/3.4729 for the warped case). These are stochastic
block averages, not evidence of convergence of the final parameters.

The supported diagnosis is residual forward fitting error under this finite
training recipe. A noisy constant-step last iterate and inadequate final
refinement are plausible explanations. Distinguishing them from capacity or
finite-bank effects requires further diagnostics; it has not been proved by
the saved loss history. The successful RKL correction using the same architecture
weakens an unavoidable representational-obstruction explanation for these
particular shifts. Clipping and nonfinite training are ruled out for these
runs, and the checked uncertainty formula does not justify calling the failed
screen a calculation bug or dismissing it as ordinary sampling noise.

The distinction matters for the next decision: a real residual bias in an
intermediate approximate fit is compatible with a useful warm start and a
passing final map. The intermediate failure is real relative to its declared
screen; whether it should block the whole FKL-to-RKL procedure is a separate
protocol question. No screen or training rule has been changed in this analysis.

## Historical launch and first-pair checkpoint

Status: campaign running. The owner added 24 GPU-process and 24 CPU-core hours
and authorized program review and execution on October 6. Review and the exact
evidence contract are in
[the active plan](bayesfilter-neutra-naf-forward-reverse-master-2026-10-06.md).
The existing .25-nat cross-entropy difference relative to the identity Gaussian
was added to its numerical table. The initial note mislabeled this as KL;
inspection of `evaluate_student` corrected the description before interpretation.
Actual forward KL is a separate reported estimate. No criterion was relaxed. Previous costs remain charged.

Command: `bash scripts/run_neutra_scientific_campaign.sh forward-reverse`, with
trusted GPU permission. The source snapshot is `source-d44adeb70522aa32` under
`artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1/`. The wrapper launches
CPU-native teacher populations with GPUs hidden, followed by NAF training on
physical GPU 2 with verified memory growth, FP64 and XLA. Each worker manifest
records its source, target, seeds, command, hardware, time and artifact hashes.

The first SMC teacher on the two-mode development geometry with seed 601 passed
both independent-population bank screens. Mode search found two stationary
positive-curvature representatives. Training/validation maximum standardized
feature discrepancies were 2.440/1.162, minimum population weight-ESS fractions
were .951/.954, and maximum mass discrepancies were .00285/.000054. The raw
proposal importance-sampling baseline also passed. These measurements retain
the teacher as viable under the exploratory screen; they establish neither
equilibrium nor a method ranking. Cost: 19.60 wall and 51.06 CPU-core seconds.
The first complete forward/reverse fit finished in 730.81 wall and 814.85
CPU-core seconds. Its forward endpoint passed with estimated forward KL
.006246 nats, maximum feature z=1.7195 and mass discrepancy .01130. All training
was finite with zero clipped updates. Five of six reverse endpoints passed:

| Reverse rate | Updates | Endpoint screen | Estimated forward KL | Maximum feature z |
|---|---:|---|---:|---:|
| .0001 | 256 | Pass | .004343 | 1.549 |
| .0001 | 1,024 | Pass | .002556 | 1.978 |
| .0001 | 4,096 | Pass | .008070 | 4.548 |
| .0003 | 256 | Pass | .005752 | 2.572 |
| .0003 | 1,024 | Reject | .015959 | 7.003 |
| .0003 | 4,096 | Pass | .005573 | 3.803 |

The .0003 / 1,024 endpoint is rejected despite finite training. These are
one-seed descriptive estimates; they do not justify a ranking among passing
endpoints. Calibration continues with seed 607 and must find one common
passing setting across all six development fits. The first pair supplies a
1,097-second GPU wall ceiling and 1,223 CPU-core-second ceiling for subsequent
fits, from measured cost times the predeclared 1.5 margin. The remaining
calibration reservation passed.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue first calibration pair | Both native teacher banks pass | No numerical/provenance veto observed | Approximate-bank NAF fit and RKL coverage | Finish forward map and frozen reverse branches | Complete recipe validated |
| Preserve final geometries | Development recipe not yet frozen | No final target exposed | Calibration reproducibility and full-stage costs | Finish declared calibration before fixed tests | Random geometry generalization |

| Inference status | Current evidence |
|---|---|
| Hard veto screen | No veto in the first native teacher; fitted maps pending. |
| Statistically supported ranking | None for this campaign. |
| Descriptive-only differences | Initial teacher feature, mass, ESS and timing summaries. |
| Default readiness | NAF is the scoped owner default; approximate FKL/RKL remains under test. |
| Next evidence needed | Common passing recipe across development seeds, then fixed cases before randomized holdouts. |

At launch 124,430.72 GPU-process seconds and 119,234.95 CPU-core seconds remained.
Subsequent worker costs update the shared/local ledgers automatically. The first
complete fit will supply measured stage pricing; the initial ceilings are not
a runtime forecast. The strongest failure alternatives remain finite-bank
error, incomplete coverage and collapse under RKL, each evaluated separately.
