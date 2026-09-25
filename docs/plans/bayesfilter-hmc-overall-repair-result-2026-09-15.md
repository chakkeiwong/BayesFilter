# HMC tuning, equilibration and precision: execution result

Date: 2026-09-15. Baseline: `5139f151237e9764ebf34b4e8cf6ceee1e74a38d`.
The user authorized the [combined recommendation and reviewed plan](bayesfilter-hmc-overall-repair-plan-2026-09-15.md)
and its execution. The scoped repair is complete. Changes are in the worktree;
the twelve protected, unrelated tracked files retain their original hashes.

The public procedure now connects candidate-set tuning to a common posterior
assessment while preserving the selected member's numerical execution contract.
It reports discarded equilibration evidence and retained Monte Carlo precision
separately. R-hat, ESS and MCSE do not qualify, rank or remove tuning candidates.
Lugsail batch means is an optional mean-MCSE estimator. No finite set of these
diagnostics proves that burn-in is sufficient for an arbitrary posterior.

## Engineering closure

| Finding or phase | Implemented behavior | Evidence and boundary |
| --- | --- | --- |
| R1: posterior transition changes the target contract | The verified member supplies its original transition and health evaluator to the shared sequential loop. Scalar chains and the declared telemetry policy survive the bridge. | Actual tuning produces verified scalar and no-telemetry members and executes posterior chunks. These regressions mock the convergence summary only to isolate dispatch; the public Gaussian example exercises real assessments. |
| R2: shared invalidity breaks historical repair records | A repair child's earlier verification remains a historical fact; shared invalidity removes current replay eligibility. | Verified-repair/shared-invalidity round trip passes without granting a member. |
| R3: only root seeds were validated | Validate every bounded warmup/retained chunk seed, its int32 range, and collisions with other phases and all recorded tuning attempts. | Later-chunk overlap, overflow and tuning-collision regressions; old seed derivation preserved. |
| R4: declared target-domain failures pause peers | Store a typed candidate execution failure with its attempted seed/chunk history, charge the attempt, reject that pair and continue funded peers. | Fault-injected declared failure, healthy peer, checkpoint reload and member export pass. Unclassified and resource exceptions retain their infrastructure behavior; missing acceptance cannot fabricate a directional repair. |
| R5: released reservations strand affordable work | A positive reservation release reopens deferred work for the current invocation. | Deterministic same-budget completion regression and existing budget tests pass. |
| R6: broken example and incomplete guide | Repair the pilot allocation; document the exact acceptance predicate, retirement behavior and correct fixed-transport replacement. | Documentation/dispatch/registry tests; guide build and rendered-page review. The movement override was already rejected by its config. |
| R7: repeated evidence processing | Cache numerical analysis by current evidence content, skip unchanged-file parsing/writing, share evidence across compact member exports and validate ancestry iteratively. | Cache/mutation/corruption tests, portable and compact round trips, and a 1,050-predecessor test. The latter isolates ancestry with a health stub; numerical health is tested separately. Full content hashes, metadata and cumulative samples still grow with history. |
| P2: diagnostic arithmetic | Share rank, split, fold, R-hat and explicitly identified TFP positive-pairs ESS. Correct the rank denominator and take the square root of the variance ratio. Odd draw counts split the first and last halves. | Independent SciPy/published-formula checks exercise both diagnostic interfaces. Identity is `bayesfilter.hmc_diagnostic_math.v2`; historical thresholds must be reassessed from draws using this arithmetic. |
| P3: common assessment | Core numerical health and R-hat cannot be replaced by callbacks. Optional ESS floors, consecutive warmup checks and named scientific functionals apply to the shared, exact-transition and archive routes. | Sequential/NeuTra/tempered-lineage suites, callback regressions, policy mismatch and checkpoint replay without new native calls. |
| P4: estimator-specific accuracy | Add original-scale mean MCSE, ordinary/lugsail batch means, quantile-specific MCSE and explicit tolerances. Continue cumulative retained draws until declared checks pass or the cap is reached. | Independent formulas, finite-size calibration, precision-cap, negative-LRV, constant, insufficient-batch and rare-event regressions. |

The new public types are `HMCPosteriorAssessmentPolicy`, `HMCPrecisionPolicy`
and `HMCPrecisionTarget`; `run_hmc_posterior` accepts an explicitly selected,
verified exact member. Its decision wording is
`POSTERIOR_DECLARED_CHECKS_PASSED` or `POSTERIOR_DECLARED_CHECKS_NOT_MET`, with
`assessment_role='posterior_only'`. Historical wrappers retain their old names
for compatibility. The [executable example](../examples/hmc_posterior_precision.py)
shows actual tuning, member construction, discarded warmup, a mean target and a
90% quantile target, with durable numerical chunks.

`precision_not_requested` is explicit when no accuracy tolerance was declared.
A small R-hat alone does not establish an adequate number of retained draws.
Nonpositive or nonfinite per-chain long-run variance, insufficient complete
batches, or unavailable quantile information cannot grant precision. Complete
batch counts and raw long-run variances are reported. The pooled mean uses all
retained draws; terminal remainders omitted from batch-variance estimation are
recorded. Chains are combined as independent chain means, not concatenated as
one time series.

## Verification and provenance

The [run manifest](artifacts/hmc-overall-repair-2026-09-15/run-manifest.json)
contains the evidence paths, command records, Python/TF/TFP versions, source
hashes, seeds, device scopes and measured timings. All CPU numerical checks hid
GPUs before framework import. GPU work used trusted execution on host GPU 1,
an RTX 4080 SUPER, with memory growth configured and verified before device
initialization. No model-scale campaign or new numerical default was promoted.

| Check | Result | Interpretation |
| --- | --- | --- |
| Broad integration suite | 277 passed in 319.77 seconds | Candidate lifecycle, adapters, dispatch, documentation, diagnostics, NeuTra controllers, route policy and tempered lineage. |
| Final affected-route suite | 114 passed in 228.63 seconds | Covers the final posterior-summary compatibility change and two additional regressions. Overlaps the broad suite; counts are not additive. |
| Public CPU Gaussian example | Passed; 512 warmup and 512 retained draws per chain; 11.011 seconds | Explicit CPU/non-XLA reference fixture, not a recommended model allocation. |
| Public GPU/XLA Gaussian example | Passed; 512 warmup and 512 retained draws per chain; 46.703 seconds | Actual tuning-to-posterior execution with requested mean and quantile precision. |
| GPU/XLA lugsail arithmetic | Maximum absolute MCSE error `1.040834e-17` against an independent CPU formula; tolerance `1e-12`; 4.679 seconds | Same draws, float64, one trace and static `(512,4,2)` signature; HLO hash recorded. GPU allocator peak 100,352 bytes for this arithmetic fixture. |
| Complete guide book | 558-page PDF built; changed acceptance and diagnostic pages visually inspected | New citations resolve. Three unrelated existing citation keys remain unresolved: `Gorinova2020`, `Pakman2014`, `Afshar2015`; unrelated layout warnings remain. |

The CPU and GPU example results preserve their execution-time source closure.
Only `hmc_posterior_assessment.py` changed afterward: legacy callback/summary
fields are preserved without overriding core checks, and the generic wrapper's
decision wording now says posterior checks rather than kernel admission. The
final affected-route suite passed after this change. The sampling and estimator
implementations did not change, so the GPU arithmetic/sampling checks were not
rerun. Their saved JSON retains the earlier wrapper wording; it is historical
output, not an assertion about the final wording or replay eligibility under
changed source hashes.

An initial example attempt failed before tuning because a required
`repair_factor` argument was missing; its log is preserved. Earlier test
iterations exposed outdated mocked diagnostics, the registry expectation and
source drift while files were being edited. Those were debugging failures,
repaired before the recorded passing runs, not evidence against an HMC target.

## Bounded precision calibration

The comparator is the known long-run variance `(1+rho)/(1-rho)` of stationary,
unit-variance Gaussian AR(1) draws. There are 200 independent replications per
rho, four chains and 2,048 draws per chain, with seed 20260915. These are
diagnostic fixture choices. Ordinary batch means, lugsail, the runtime TFP
positive-pairs method and an independent source-translated Stan initial-monotone
reference process the same draws. See the [calibration JSON](artifacts/hmc-overall-repair-2026-09-15/precision-calibration.json)
for mean-ratio uncertainty intervals, empirical replicate standard deviations
and Wilson 95% coverage intervals.

Each table cell gives mean estimated MCSE / asymptotic truth, followed by the
fraction covered by a fixed-size normal interval. This comparison does not
calibrate sequential stopping.

| AR(1) rho | TFP positive pairs | Ordinary batch means | Lugsail batch means | Stan reference |
| --- | --- | --- | --- | --- |
| 0 | 1.012 / 0.945 | 1.002 / 0.940 | 1.003 / 0.935 | 1.006 / 0.945 |
| 0.8 | 1.022 / 0.975 | 0.954 / 0.970 | 1.049 / 0.980 | 1.009 / 0.975 |
| -0.5 | 1.010 / 0.965 | 1.015 / 0.965 | 0.987 / 0.955 | 0.990 / 0.960 |

Lugsail produced unavailable estimates in 2/200 antithetic replications; all
other cells had 200 valid estimates. Its mean MCSE uses the 198 valid
replications, while coverage counts unavailable estimates as failures among
all 200. For example, the antithetic lugsail coverage interval is
`[0.917, 0.976]`; the positive-persistence interval is `[0.950, 0.992]`.
These estimates remain uncertain. No predeclared paired superiority test
supports ranking the estimators, and no new default follows from this table.

The first Stan reference translation used divisor `N-lag`, following a stale
comment. Inspection of the actual official `autocovariance` body showed divisor
`N`. The preliminary script and results are retained in
`preliminary-comparator-invalid/` and explicitly excluded from evidence. The
corrected calibration was rerun; its timed computation was 1.020 seconds,
excluding framework imports. The runtime TFP estimator intentionally remains
different from Stan's initial-monotone method. The source/assumption review and
local paper/code inventory remain linked from the [burn-in review](bayesfilter-hmc-warmup-precision-repair-plan-2026-09-15.md).

## Terminal scientific audit and remaining work

The claimed mathematical targets are modern rank/folded split R-hat, the
declared mean long-run variance estimators and indicator/order-statistic
quantile MCSE. The first two agree with independently checked formulas within
the tested numerical tolerances. Quantile arithmetic agrees with the checked
Vehtari construction, using the explicitly identified TFP ESS component rather
than claiming exact Stan estimator parity. These are estimates under moment,
stationarity and mixing assumptions; sample diagnostics do not prove those
assumptions.

Stan's configured fast/slow/fast adaptation schedule is useful precedent for
organizing warmup. It does not provide a universal estimate of sufficient
burn-in. This repair retains the existing owner-policy warmup minimum/window/
cap of 2,000/1,000/10,000 and screening threshold 1.05 as operational settings.
Optional consecutive checks and ESS floors strengthen a declared assessment;
their target-specific false-pass rates have not been calibrated here.

The missed-mode regression makes the strongest alternative explanation
concrete: samples confined near one mode pass R-hat and precision while their
mean is far from the full target's mean. Meaningfully dispersed starts,
model-coordinate functionals, mode exploration and independent posterior
checks remain necessary. Heavy tails, unobserved rare events and inadequate
mixing can invalidate otherwise small MCSEs. Target-specific references or
replicated transient/multimodal checks could overturn a favorable diagnostic
interpretation. Fixed-size Gaussian calibration is the weakest basis for
generalizing to the user's scientific targets.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete this engineering repair | Reproduced defects and formulas have passing regressions; documented public execution works | Corruption, changed identity, invalid seeds and core failures still block the affected operation | Finite test coverage; large-history behavior unqualified | Use the declared public procedure and preserve model-specific evidence | Every possible tuning problem will succeed |
| Expose lugsail as an option | Formula and CPU/GPU arithmetic checks pass | Invalid per-chain variance cannot grant precision | Finite-sample noise, antithetic failures and unknown moments | Choose a method and tolerances before a model run; inspect validity and sensitivity | Lugsail is uniformly superior or a new default |
| Use explicit warmup/precision reports | Shared assessments and cumulative cap behavior work | Core health/convergence requirements cannot be bypassed | Missed modes and repeated-look false passes | Validate starts, functionals and downstream posterior agreement for the actual target | A diagnostic pass proves sufficient burn-in or stationarity |
| Keep structural work bounded | Repeated file I/O/numerical analysis and recursion defects repaired | Checksums and ancestry remain checked | Content hashing, metadata and cumulative arrays still grow with history | Measure host memory/restart costs before a larger storage refactor | Constant-cost restart or complete modernization of the historical tuner |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Corruption/invalidity regressions pass; two antithetic lugsail estimates are unavailable, not clipped to precision passes. |
| Statistically supported ranking | None; the calibration did not declare a paired superiority criterion. |
| Descriptive-only differences | The estimator ratios, coverage fractions and CPU/GPU example diagnostics describe these fixtures only. |
| Default-readiness | No new burn-in count, ESS floor, lugsail setting or stopping-rule default is established. Correct diagnostic arithmetic is a versioned bug fix. |
| Next evidence needed | Target-specific start/functional checks, replicated transient or multimodal tests, and uncertainty-aware method comparisons before any default promotion. |

Reasonable tuning problems can still exhaust a finite L/epsilon search, fail
geometry preparation, encounter candidate-local domain errors, run out of
evidence/compute, or fail fresh verification. Those outcomes preserve their
typed reasons and do not reject the research direction. Posterior diagnostic
failure remains separate from kernel qualification.

The remaining structural debt is the historical tuner and preparation modules,
some private cross-module calls, and history-dependent persistence/assessment
cost. The separate legacy Phase 29 warmup screen still rejects according to
configured drift heuristics. Its statistics omit covariance between adjacent
epoch means; their use does not establish calibrated z-test significance, and
they are not extra requirements of the common posterior assessment. The guide
now makes this compatibility boundary explicit.

Growing diagnostic windows, automatic new-start selection, automatic
batch-size selection, joint confidence regions and formal fixed-width stopping
are not implemented by this repair. The current implementation and guide state
these boundaries instead of implying a universal burn-in or precision guarantee.
