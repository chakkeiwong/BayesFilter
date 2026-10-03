# HMC repair M8 execution

Status: complete, September 19. The active contract and resource authorization
are in the [master program](bayesfilter-hmc-repair-master-program-2026-09-16.md).
Results live in [m8-r1](artifacts/hmc-repair-master-2026-09-16/m8-r1/).
All planned numerical workers have terminated. No
sampling, tuning or diagnostic default has been promoted.

## Question and implementation

The remaining questions concern exact-pair tuning qualification, difficult
coordinates, sensitivity to subtle transition errors, and uncertainty at the
posterior stopping time. Every verified tuning candidate remains in the native
result. Expensive posterior experiments preselect the first verified identity;
unassessed siblings remain explicit. R-hat, ESS and MCSE never affect tuning
membership, ranking or repair.

The validation changes add an exact noncentered version of the existing funnel,
a Gaussian-energy invariance test, and an actual public-tuner acceptance
experiment. The centered and noncentered funnels have identical model laws by
the checked Jacobian identity. Fresh ordinary fits share the initial model
point; automatic preparation constructs its own later metric and start bank.
Only the saved-pair diagnostic holds the entire historical geometry fixed.

Acceptance calibration has three distinct routes. Synthetic `controller`
experiments exercise the screen with known-mean marks. Numerical `frozen`
experiments measure stationary TFP-HMC acceptance. Numerical `prepared`
experiments execute `tune_hmc_kernel`, with fixed starts, fresh measurement and
verification, and no repair children. Separate exact Gaussian anchors and
independently computed endpoint energies supply the stationary reference.
Stationary acceptance and finite-window fixed-start acceptance are different
quantities; the reference never decides qualification.

The initial frozen CPU suites mislabeled their start regime as dispersed;
their actual starts were independent exact Gaussian draws. Their frozen
manifests are preserved, and they supply no public-qualification evidence.
New frozen designs require `start=reference`.

## Completed reference results

All rows below are CPU/non-XLA reference work, with GPUs deliberately hidden.
They do not replace the GPU/XLA campaign.

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Fresh centered funnel, three fits | One bootstrap health veto; two complete searches with zero verified members | Failure remains in the planned denominator |
| Fresh noncentered funnel, three fits | 24, 28, 23 verified members | The coordinate repair found qualified kernels; no statistical ranking from three fits |
| Selected noncentered member, fit 0 | Warmup cap at 10,000, no retained draws | Posterior unavailable despite successful tuning |
| Selected noncentered member, fit 1 | 2,000 warmup, 10,000 retained; requested precision not met | Reference assessment was available, but runtime checks did not pass |
| Selected noncentered member, fit 2 | 2,000 warmup, 7,500 retained; declared checks passed | Limited, target-specific evidence |
| Normal-conjugate stopped means / medians | 31/32 and 32/32 covered exact references | One fixed observed dataset; conditional repeated fits, not SBC |
| Normal-conjugate fixed means / medians | 31/32 and 31/32 covered | Same verified members, independent streams, 10,000 retained per chain |
| Beta-binomial stopped means / medians | 27/32 and 28/32 covered | Mean coverage interval [.6721, .9472]; investigate finite-sample uncertainty |
| Beta-binomial fixed means / medians | 28/32 and 30/32 covered | The result does not isolate optional stopping as the cause |

All 64 stopping fits had available intervals and passed their declared runtime
checks. [The aggregation](artifacts/hmc-repair-master-2026-09-16/m8-r1/stopping-cpu-summary.json)
preserves exact input hashes, paired errors, availability and pointwise
intervals. The four paired mean-absolute-error intervals favor the longer
fixed arm, but they are exploratory pointwise approximations, not a
multiplicity-adjusted sampler ranking or a change to stopping defaults.

An independent NumPy implementation reproduced all 128 saved mean MCSEs using
the stated lugsail complete-batch formula, with maximum absolute difference
`1.5612511283791264e-17`. See
[the arithmetic check](artifacts/hmc-repair-master-2026-09-16/m8-r1/lugsail-arithmetic.json).
This checks the computed quantity, including pooling and truncated batches.
It does not prove stationarity, a CLT, or interval coverage.

The saved CPU interval-error diagnostic (`stopping-cpu-errors.json`) gives
empirical error SD divided by RMS reported MCSE of 1.281 for beta stopped means,
1.212 for beta stopped medians, .895 for beta fixed means and 1.013 for beta fixed
medians. The normal-arm ratios range from .925 to .998. The beta stopped-mean
coverage interval [.6721,.9472] excludes .95 in this pointwise comparison;
multiple target/quantity comparisons and only one fixed dataset limit the
interpretation. Variable variance-estimator error, tail behavior, initialization
and chance remain possible explanations. These exploratory ratios are not
calibration factors, and no estimator or threshold was changed.

## GPU progress

The trusted GPU1 runs record verified memory growth, TensorFlow 2.20.0,
TFP 0.25.0, FP64 target calculations, TF32 policy and XLA execution. Each suite
uses a frozen package snapshot; source differences remain visible.

| Fixed-look defect experiment | Baseline / no-op rejections | Reversed-MH detections | Decision |
| --- | --- | --- | --- |
| 8,192 anchors, epsilon .3, 32 fresh experiments | 1/32, 0/32 | 30/32; 95% exact interval [.79193,.99234] | Missed the predeclared .8 lower-bound requirement |
| 8,192 anchors, epsilon .6, 32 fresh experiments | 2/32, 2/32 | 32/32; interval [.89112,1] | Passed this sensitivity screen; null size remains imprecise |

The .3 follow-on doubled independent anchors to 16,384 while preserving K=32,
L=5, observables, significance and controls. In 32 fresh experiments it detected
the defect 32/32 times, with exact 95% interval [.89112,1]. Baseline and no-op
controls each rejected 2/32, interval [.00766,.20807]. This meets the declared
sensitivity screen while leaving null-size uncertainty. Its pilot and fresh
experiments use new seed namespaces and are not pooled with the first design.
It is a revised design motivated by the earlier miss, not a replication of the
8,192-anchor design.

The public-screen pilot completed two searches at each epsilon. Results were
0/2 qualified at .9, 2/2 at 1.4, 1/2 at 1.5 (one inconclusive at cap), 0/2 at
1.6, and 0/2 at 2. Stationary reference means were approximately .8843, .6721,
.6747, .9185 and .0191. Their nonmonotonic pattern is consistent with fixed-L
resonance; every exact pair needs its own evidence. Pilot costs passed the
master's 50% margin check for 32 fresh searches per cell.

All 160 fresh public searches then completed in 2,963.546242 worker-seconds.
Every inventory check passed, and no fresh search was inconclusive. Each row
has 32 independent whole searches at L=5. The exact 95% qualification-rate
interval is [0,.10888] for 0/32 and [.89112,1] for 32/32.

| Epsilon | Fresh verified searches | Independent stationary acceptance mean |
| ---: | ---: | ---: |
| .9 | 0/32 | .888264 |
| 1.4 | 32/32 | .674524 |
| 1.5 | 32/32 | .672409 |
| 1.6 | 0/32 | .919645 |
| 2.0 | 0/32 | .021423 |

Reference Monte Carlo intervals and every measured/verification receipt are in
`screen-fresh-gpu-r1`. These are operational screen rates at the declared
Gaussian geometry and starts. They do not calibrate automatic preparation,
arbitrary targets, or nominal repeated-look coverage.

Three fresh GPU centered-funnel searches returned zero verified members. The
three noncentered fits retained 12, 28 and 19 verified members. Their selected
posteriors passed declared checks after 2,000 warmup transitions and 5,000,
7,000 and 3,500 retained transitions per chain, respectively. Three fits do not
establish a statistical ranking or family-wide reliability.

The longer saved-geometry experiment also finished: all three fresh searches
at L=18, epsilon=.2256942307263964 rejected the pair because of nonfinite
log-acceptance evidence. More evidence exposed a numerical veto; it did not
qualify the historical inconclusive setting. See
[the saved-pair report](artifacts/hmc-repair-master-2026-09-16/m8-r1/funnel-evidence-extension-gpu-r1/diagnostic.json).

All 64 fresh GPU stopping fits are complete. The two-, four- and eight-worker
resource stages passed, and the remaining 50 fits completed under
[the revised coordinator](artifacts/hmc-repair-master-2026-09-16/m8-r1/run_remaining_gpu_r3.py).
Individual design payloads and 64 unique seeds are
unchanged; only target interleaving and concurrency change. The old orchestration
parent was retired after the last acceptance child completed; no numerical
worker was killed or fit replaced. The retirement record preserves its worker
charges and explains the transition.
Native preparation failures are recorded candidate/fit failures, not permission
to discard a replication. Infrastructure failures stop the affected tranche for
localized repair under the remaining budget.

## Engineering checks and limitations

The GPU stopping tranche, including pilots, cost 43,574.79769107804 worker-seconds
against its 70,000-second reservation. Peak combined use during the final stage
was 3,781 MiB GPU and 100,208.98 MiB host, below the declared 8 GiB/100 GiB
bounds. All 64 fresh fits had available intervals and passed their declared
runtime checks. Their pointwise coverage is:

| Target | Stopped mean | Stopped median | Fixed mean | Fixed median |
| --- | ---: | ---: | ---: | ---: |
| Normal-conjugate | 30/32 | 29/32 | 32/32 | 29/32 |
| Beta-binomial | 32/32 | 29/32 | 32/32 | 32/32 |

Exact 95% intervals are [.79193,.99234] for 30/32, [.74977,.98023] for 29/32,
and [.89112,1] for 32/32. The saved
[GPU aggregation](artifacts/hmc-repair-master-2026-09-16/m8-r1/stopping-gpu-summary.json)
includes all planned fits, paired errors and original result hashes. GPU beta
stopped-mean empirical error SD / RMS reported MCSE was 1.010; the stopped-median
ratio was 1.128. These differ from the CPU descriptive values but neither
32-fit batch resolves a general coverage claim. No CPU/GPU ranking or pooled
coverage estimate is justified here.

All normal fits and 28 of 32 beta fits on each device stopped at the first
eligible 1,000-draw retained look. Every fit passed warmup at the 2,000 minimum.
Consequently this design mostly assesses accuracy at the minimum count; only
four fits per device exercise later stopping, and it supplies little evidence
about prolonged adaptation or repeated precision checks. Future calibration
needs predeclared regimes with materially later stopping and more datasets.

The separate GPU arithmetic check reproduced all 128 saved mean MCSE values
with maximum absolute difference 6.938893903907228e-18. Together with the CPU
check this validates the implemented formula on 256 saved arms, not stationarity
or coverage.

The eight-schools pilot retained 19 verified members and passed its selected
posterior/reference screen. The three fresh fits retained 15, 16 and 18 members.
All three selected posteriors passed the uncertainty-aware reference-mean
comparison; two passed the full declared screen. Fit 1 reached 10,000 retained
draws per chain with lugsail MCSE .1221 and .1198 for theta[3] and theta[8], above
the requested .1. Its warmup checks passed, but the precision cap remains a
failed posterior screen. Tuning membership was unaffected.

All four original zero-start regression fits failed bootstrap with nonfinite
log-acceptance records. The separately reviewed data-derived starting point
converged in three deterministic updates with maximum absolute score 2.19e-9.
All four no-hint fits from that point still failed bootstrap. Moving the start
near this stationary point did not resolve the geometric scaling problem.

The independent curvature check found a maximum negative-Hessian eigenvalue
of 6,322,058.5. The no-hint initializer's epsilon .31947 is about 401.63 times
the local quadratic stability limit .00079543. This calculation explains a
plausible failure mechanism; it is not a global stability theorem. The analytic
Hessian matched central differences of the checked score at two step lengths,
with maximum scaled error 5.25e-8. The first diagnostic command used a nonexistent
batch-method name; that failed harness attempt and its 2.8203 seconds are
preserved. The corrected check uses the adapter's actual batched
`log_prob_and_grad` endpoint and cost 2.7702 seconds.

A separate four-fit GPU diagnosis supplied that Hessian through the existing
public argument, using the same target and posterior/reference criteria. The
pilot completed preparation but returned zero verified candidates: all six
initial pairs requested larger epsilon, while their doubled-epsilon children
had nonfinite log acceptance. No finite opposing acceptance observation was
available for the current interval-refinement rule. The three fresh repetitions
retained 18, 13 and 10 members, and every preselected posterior passed both its
declared checks and the uncertainty-aware reference comparison. They used
2,000 warmup draws and respectively 1,000, 1,500 and 1,500 retained draws per
chain. The pilot remains a preparation success followed by a search failure;
it does not establish that no usable kernel exists between those tested steps.

All 16 matched-reference attempts, including both failed initialization arms,
cost 4,904.698596649338 GPU worker-seconds. The
[matched-fit audit](artifacts/hmc-repair-master-2026-09-16/m8-r1/posteriordb-summary.json)
checks distinct seeds, predeclared selection, candidate counts and reference
comparison arithmetic. Pilots and fresh fits remain separate. Three successful
curvature-assisted fits establish feasibility on this fixed dataset, not an
automatic preparation default or comparative superiority.

The saved-geometry interval diagnosis then measured 18 explicit interior pairs
in each of three fresh searches. All three independently verified exactly one
member, L=3 and epsilon=1.141019386629899; the other 17 pairs in each search
failed promotion. Mass and start-bank signatures exactly match the failed pilot.
Thus its coarse search missed an available member. This checks qualification
only, not that member's posterior accuracy. The
[interval audit](artifacts/hmc-repair-master-2026-09-16/m8-r1/regression-interior-summary.json)
preserves every pair, seed and input hash. The initial harness failed before HMC
because it supplied a zero repair reservation. Its 5.175907 seconds remain
charged; the corrected 109-unit allocation holds 108 units for the 18 pairs'
possible six looks and one unused required reservation, with repairs disabled.
Three corrected searches cost 400.635604 seconds. Combined matched-reference
and interval work cost 5,310.510108 seconds, below the 16,000-second reservation.

The full inference-validation suite passed 143 tests after the reporting change.
Two additional energy-veto/deadline tests passed after correcting the test's
patch target from a class attribute to its graph-instance attribute: 145
distinct validation tests now pass across those batches. The failed test
attempt and its wall time remain recorded. Tests exercise the actual public
endpoint, separate reference and tuning adapters, unique search scopes, exact
funnel Jacobian and score, source/seed receipts, and missing-fit denominators.

The final affected test batch passed 535 tests with one inherited tiny-bootstrap
skip. It includes the full inference-validation suite and the candidate,
preparation, warmup and public-dispatch regressions; overlapping earlier batches
must not be added to this count. Fifteen separate documentation-contract tests
also passed. The skip is `test_real_tiny_gaussian_windowed_mass_stage_returns_structured_result`,
whose small bootstrap allocation ends with `repair_budget_exhausted`; it is not
an additional successful numerical test.

The official `docs/main.pdf` has been rebuilt and installed (565 pages), with
coverage regenerated from 49 preserved reports. Physical pages 411, 413 and 432
were inspected for the new preparation/search explanations and coverage table;
the preceding build's pages 427/428 were also inspected. Verification is required
for each measurement survivor. The build retains the previously recorded missing
bibliography keys `Afshar2015`, `Gorinova2020` and `Pakman2014`. Installation
hashes and all 74 checked source hashes are archived in `guide-installation-r3.json`
and `guide-r3/build-manifest.json` under the M8 output root. All 15 documentation
contract tests passed again after the preparation explanation was added.

The terminal integrity audit checked 335 saved candidate inventories, 13 frozen
source snapshots, 177 indexed results and 568 tensor checksums without a
mismatch. Each verified member retains its own required receipt. All planned
outcomes are present, no indexed reservation remains, and the final process
check found no campaign worker. Frozen sources remain distinct from the live
checkout, including concurrent unrelated changes. These are engineering checks,
not posterior validation.

The pinned posteriordb checkout supplies matched data, Stan laws and ten-chain
references for noncentered eight-schools and `sblrc-blr` regression. Separate
SciPy density and analytic-score tests pass, including log-scale Jacobians and
model-coordinate ordering. Recomputed reference maximum R-hat values are
1.0004824 and 1.0002378. These finite references retain their Monte Carlo
uncertainty. The completed original fits and separate regression repair arms
are described above; reference preflight alone is not BayesFilter posterior
validation.

No `bayesfilter.external_posterior_reference.v1` bundle was found in the searched
MacroFinance plan/results directories, excluding copied source/test snapshots.
This does not prove none exists elsewhere. The MacroFinance comparison remains
unavailable. The two explicit posteriordb adapters do not make arbitrary external
catalog cells executable.

| Decision | Primary criterion | Veto evidence | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close M8 | All planned outcomes and terminal integrity checks complete | Failed preparation, empty sets and the eight-schools precision cap preserved | Broad reliability remains unmeasured | Carry forward exact remaining budget and prioritized repairs | Default readiness |
| Retain funnel coordinate repair | Three fresh GPU noncentered fits found verified settings and passed selected posterior checks | Centered searches and saved-pair extension remain rejected; CPU posterior caps preserved | Heavy model-coordinate tails and broader reliability | Preserve both coordinate outcomes and their uncertainty | Every qualified kernel yields adequate posterior draws |
| Retain revised defect test | 16,384-anchor .3 follow-on met the sensitivity criterion | Numerical checks passed; null error rate remains imprecise | Generalization beyond this defect and target | Retain independent original/follow-on results | First design was adequately powered or null size is precisely calibrated |
| Retain optional regression geometry input | Three fresh fits passed full posterior/reference checks | Original and data-start/no-hint arms all failed; geometry pilot found no member | Generalization and automated scale discovery | Diagnose startup scaling across targets without importing reference information | A universal Hessian or initializer default |
| Repair finite epsilon exploration next | All three exact-geometry interior searches verified a missed member | Invalid doubled parents remain rejected | Whether a bounded general proposal rule works across targets | Test proposals between healthy finite evidence and failed exploration without fabricating acceptance direction | Acceptance monotonicity or posterior quality of the new member |
| Preserve MCSE defaults | Independent arithmetic agrees on all 256 saved CPU/GPU arms | CPU beta concern and all misses remain visible | Only four fits per device exercised later stopping | Broaden dataset and stopping-time regimes before revising thresholds | Nominal or anytime coverage |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Failed preparation and unavailable posterior outputs retained; no fabricated qualification |
| Statistically supported ranking | None promoted; pointwise exploratory paired intervals are archived |
| Descriptive-only differences | Acceptance means, three-fit coordinate comparison, runtime and member counts |
| Default readiness | Not established; no default changed |
| Next evidence needed | Startup/search repairs, broader stopping regimes and full-procedure SBC; matched MacroFinance inputs, unknown-mode exploration and sequential-test wrapper validation remain open |

The strongest alternative explanation for the beta result is finite-sample
variance-estimator error or chance, since the fixed arm also misses intervals.
An independent arithmetic mismatch would invalidate the engineering conclusion;
fresh adequately replicated calibration could overturn the coverage concern.
The weakest evidence remains few target datasets and the missing MacroFinance
comparison. The local-curvature explanation would be weakened by failures after
an independently checked scaling repair; the exact saved-geometry interval
success already rules out absence of every useful candidate in that pilot's
continuous domain. These results reject individual candidates or expose test limits;
they do not reject the tuning architecture or the broader research direction.

## Terminal accounting and continuation

The [terminal reconciliation](artifacts/hmc-repair-master-2026-09-16/m8-r1/reconciliation-terminal.json)
includes failed attempts, all indexed workers, separately measured diagnostic,
test and build commands, direct JUnit durations, and the terminal audit's own
7.633027359028347 seconds exactly once. The CPU overhead allowance of 600 seconds
covers short unwrapped inspection, source freezing and supervision; it is not
a utilization measurement. Concurrent workers are charged by summed worker
wall time, not calendar time. No attempt or prior campaign charge is reset.

| Resource | M8 measured/JUnit seconds | M8 overhead allowance | Cumulative charged | Authorized total | Remaining seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU reference/engineering | 17,844.886235256454 | 600 | 103,901.0342446297 | 259,200 | 155,298.9657553703 |
| GPU | 56,751.053581654094 | 0 | 107,274.72379197675 | 172,800 | 65,525.27620802325 |

The remaining allowance is about 43.14 CPU hours and 18.20 GPU hours. No launched
work or reservation remains. The master now prioritizes startup-scale handling
and bounded exploration of the demonstrated omitted interval, followed by
stopping experiments that actually exercise later looks and multiple datasets.
Matched MacroFinance evidence, unknown-mode exploration, broader full-procedure
SBC, sequential invariance and remaining historical-code cleanup stay open.
The program has completed this campaign, not all of those research requirements.
