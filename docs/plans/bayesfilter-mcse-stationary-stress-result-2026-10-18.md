# Stationary MCSE stress calibration: completed results

18 October 2026, China time. All 128 replications in each of the six
predeclared conditions are complete. The numerical and provenance audits pass.
The experiment exposes finite-sample limitations of the existing posterior
precision assessment: strong dependence produces undercoverage, extreme
antithesis can make variance estimates unavailable, and the optional
per-chain maximum does not resolve the lognormal-square coverage failure.
No estimator is promoted and no running HMC policy changes.

The [plan](bayesfilter-mcse-stationary-stress-calibration-2026-10-17.md)
specifies the evidence contract and exact covariance derivations. Artifact
paths below are relative to
`artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/stationary-stress-20261017-01/`,
abbreviated A. The final result is `A/attempt-02/result.json`; all earlier
attempts, failures and costs are preserved.

## Design and implementation checks

Each independent unit consists of four stationary Gaussian AR(1) chains of
10000 draws. Correlations are 0, .99 and -.99. Gaussian and exp(Gaussian)
margins share each latent path; there are 768 condition records but only
384 independent latent four-chain units. The serious seed is 74117, with
streams [74117, correlation-group, replication]. The smoke used disjoint
seed 74116. The monitored quantities are X, X squared and the event |X|>2.
Their expectations are known exactly.

The frozen public `assess_posterior` computation is called at lengths
1000, 2000, ..., 10000 using original lugsail, the optional
`lugsail_max_batch_means`, plain batch means and autocorrelation MCSE.
All methods use the same paths. The retained criteria remain R-hat <=1.01,
bulk/tail ESS >=400 and MCSE/SD <=.02 for all declared quantities. Binary
quantities use the public quantity-aware rules. These are the inherited
operational criteria being examined, not established universal guarantees.
The known expectations never enter stopping.

The numerical source is the reviewed precision overlay on commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`; its only changed numerical file
is `hmc_precision.py`, SHA256
`d17eff6d5a37f6036469a96e09248045f96b6961bb6c7b548946e50a9c092270`.
The driver is frozen in `A/source-01`, SHA256
`3c0af2207ce32ab8a04295d1ec9727b8f9a41a1a90494a163b85c1dff52632e5`.
The environment uses Python from conda `tfgpu`, TensorFlow 2.20.0,
TensorFlow Probability 0.25.0, NumPy 2.1.3 and SciPy 1.17.1. GPUs are
deliberately hidden and two TensorFlow/OMP threads are specified. NumPy/SciPy
generate independent reference paths; TensorFlow evaluates the actual
assessment. Batch kernels use XLA; post-chunk rank/ESS diagnostics retain
their declared non-XLA graph exception. A no-device CUDA message is expected
in this deliberately CPU-only diagnostic and is not evidence of broken GPUs.

The first numerical attempt ended normally at its cooperative cap with
579 complete records: 97/97/97/96/96/96. Its reader then rejected equivalent
binomial intervals because an absolute comparison tolerance of 1e-13 was
too strict near zero. An exhaustive check of all counts with denominators
1..128 measured at most 4.998224056862455e-13 discrepancy between binomial
root inversion and beta quantiles. Only that reader comparison changed to
1e-12. The old reader, failed closeout and result remain intact. The corrected
reader passed the original prefix; the second numerical attempt added the
remaining 189 records without changing the driver, policy or streams. It
finished at 01:00:42 China time on 18 October.

The final independent reader, `A/source-02`, checks all 77 numerical source
hashes, all 768 record identities, all 579 carried records, analytic reference
variances, reported interval arithmetic and denominators. Its full 144-screen
report is `A/audit-02.json`. The additional `A/terminal-analysis-01.json`
replays all 768 fixed-endpoint reference paths and confirms their estimates,
reaggregates scalar summaries, and checks 1024 optional-maximum endpoints
including 12288 per-chain/quantity comparisons, activation flags and pooling.
It does not replay every intermediate assessment or first passing look.

The latest calibration/reader regression suite passes 72 tests. A previous
combined calibration/observer run passed 110 tests; these counts overlap
and must not be added. The observer repair corrects a stale per-method
capacity-wait label in the checkout; the running GPU snapshot is unchanged.

## Stopping and unavailable estimates

The table reports first passes by 10000 draws per chain, out of 128.
Nonstopping is retained, not omitted from the denominator.

| Margin, correlation | Lugsail | Optional maximum | Batch means | Autocorrelation |
| --- | ---: | ---: | ---: | ---: |
| Gaussian, 0 | 128 | 128 | 128 | 128 |
| Lognormal, 0 | 128 | 128 | 128 | 128 |
| Gaussian, .99 | 0 | 0 | 0 | 0 |
| Lognormal, .99 | 0 | 0 | 0 | 0 |
| Gaussian, -.99 | 0 | 0 | 0 | 0 |
| Lognormal, -.99 | 0 | 0 | 8 | 0 |

All independent conditions stopped at 1000 or 2000 draws. Among the eight
lognormal-antithetic batch-means stops, six occurred at 9000 and two at
10000. Their coordinate/square/event intervals missed the known means
2/4/3 times respectively. Eight conditional observations provide weak
information about stopped coverage; none passes the multiplicity-adjusted
undercoverage screen. This does not validate the stopping rule.

At the fixed endpoint, original lugsail reports no valid Gaussian-antithetic
coordinate interval in any of the 128 replications. The optional maximum
and plain batch means report 128; autocorrelation reports 56, leaving 72
unavailable. All other method/condition/quantity fixed intervals are available.
The square of an antithetic Gaussian process has positive serial dependence;
negative coordinate correlation does not make all monitored quantities easy.
No stops in a condition means that its stopped coverage is unmeasured, not
that coverage was correct.

## Coverage findings

For every condition, method, quantity and fixed/stopped endpoint, the plan
prespecified an exact binomial interval at alpha=.05/144. The lower endpoint
exceeding .05 falsifies nominal 5% miss probability for that reported
estimand under this design. Twelve of the 144 screens flag undercoverage;
all twelve concern fixed endpoints. Every listed denominator is 128 valid
intervals. The complete report also includes every unflagged or unavailable
screen and the pointwise intervals. These are not 144 independent experiments.

| Margin, correlation | Method | Quantity | Misses | Adjusted lower bound on miss probability |
| --- | --- | --- | ---: | ---: |
| Gaussian, .99 | Batch means | Coordinate | 24 | .0845 |
| Lognormal, .99 | Lugsail | Square | 23 | .0791 |
| Lognormal, .99 | Optional maximum | Square | 23 | .0791 |
| Lognormal, .99 | Batch means | Coordinate | 22 | .0738 |
| Lognormal, .99 | Batch means | Square | 27 | .1010 |
| Lognormal, .99 | Batch means | Event | 19 | .0585 |
| Lognormal, .99 | Autocorrelation | Square | 18 | .0536 |
| Gaussian, -.99 | Batch means | Square | 22 | .0738 |
| Gaussian, -.99 | Batch means | Event | 19 | .0585 |
| Lognormal, -.99 | Batch means | Coordinate | 20 | .0635 |
| Lognormal, -.99 | Batch means | Square | 23 | .0791 |
| Lognormal, -.99 | Batch means | Event | 21 | .0686 |

The lugsail and optional-maximum lognormal-square failures are identical:
23/128 misses, with adjusted interval [.0791, .3249]. Autocorrelation also
fails that screen, with 18/128 and [.0536, .2779]. The experiment therefore
does not establish an adequate replacement simply by switching among these
three choices. These are individual calibration failures, not a statistically
supported pairwise ranking. Paired differences and continuous diagnostics
remain exploratory; no default is selected.

## What explains the failures

The claimed numerical target is a variance estimate for the pooled stationary
mean. The TensorFlow batch implementation computes the documented finite
complete-batch formula and rejects nonpositive per-chain estimates. It is
not an exact finite-sample variance estimator for dependent paths.

The exact covariance derivation in the plan gives the unconditional
expectation of ordinary batch means and raw lugsail before positivity
screening. The independent analysis passed 24 small covariance-matrix trace
checks. At the fixed endpoint the inherited large and small batches are
100 and 33. Representative ratios of expected estimated variance to the
exact four-chain mean variance are:

| Margin, correlation, quantity | Batch means | Raw lugsail |
| --- | ---: | ---: |
| Any tested independent coordinate/square | 1.0000 | 1.0000 |
| Gaussian, .99, coordinate | .3666 | .5856 |
| Gaussian, .99, square | .5677 | .8674 |
| Lognormal, .99, coordinate | .4221 | .6614 |
| Lognormal, .99, square | .6247 | .9059 |
| Gaussian, -.99, coordinate | 1.6209 | -2.8933 |
| Gaussian, -.99, square | .5677 | .8674 |

The positive-dependence ratios show bias that is inherent in these batch
lengths, rather than an implementation error exposed by this experiment.
The negative raw expectation for the antithetic coordinate explains why a
positivity screen can return unavailable estimates. The optional maximum
prevents that particular negative estimate; it has no guarantee of unbiased
variance or calibrated intervals. Its nonlinear expectation is not given by
the table. The table also does not describe estimates conditional on passing
the positivity screen or variance conditional on a stopping time.

For lognormal squares, variance-estimator bias is only part of the issue.
Normal intervals can be inaccurate with skewed finite samples even when an
asymptotic variance estimate is sensible. Exact-reference errors and mean
variance ratios are descriptive here, not evidence separating those causes
completely. The next diagnostic must examine interval shape as well as batch
length. Autocorrelation's unavailable antithetic results also require a
separate comparison with the installed TFP estimator before attributing them
to BayesFilter arithmetic.

## Decision and remaining repair

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close this bounded calibration | All 768 predeclared records and terminal audits complete | Reader failure repaired; no source/prefix inconsistency | Intermediate first-pass calculations were not independently replayed | Preserve findings and proceed to root-cause diagnostics | Universal correctness of stopping |
| Keep default promotion open | Twelve adjusted fixed-interval failures; extreme-antithetic unavailability | Calibration veto for a general coverage claim | Dependent, skewed and data-dependent stopping behavior | Audit positive-pair ESS and assess batch/interval alternatives on fresh data | A preferred replacement or superiority ranking |
| Continue SIR independently | Its frozen numerical comparison remains valid as scoped | No continuation veto from this diagnostic study | Full candidate/posterior evidence and affordability | Monitor existing cohort; keep every incomplete/failed member | That MCSE calibration repaired SIR mixing |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Twelve fixed-interval undercoverage flags; specified unavailable estimates remain explicit |
| Statistically supported ranking | None; individual calibration tests do not establish pairwise superiority |
| Descriptive-only differences | Stop counts, paired variance differences, runtime and empirical variance ratios |
| Default-readiness | Not established for any replacement |
| Next evidence needed | Source-parity diagnosis, dependence/scale-sensitive precision proposals, fresh fixed/stopped calibration, and saved nonlinear-HMC quantity validation |

The next phase should first distinguish an antithetic ESS implementation
defect from behavior inherited from the installed estimator, using a preserved
failing seed, independent arithmetic and all affected public callers. Then
compare predeclared batch-length/variance estimators against exact covariance
controls and assess skew-aware interval procedures on fresh calibration data.
Never choose an estimator or batch length to make a failed retained HMC
member pass. Any optional policy must preserve tuning membership, source
identity, old failure records, explicit unavailable outcomes and bounded
compute. The ongoing SIR cohort continues under its frozen policy.

The strongest alternative explanation for the empirical failures is that a
few rare lognormal observations dominate 128 replications. That limits effect
size and ranking conclusions, but does not erase the prespecified adjusted
binomial failures or exact finite-batch bias. A faithful independent replay
that changed the miss counts, source identity or exact covariance arithmetic
would overturn the relevant conclusion. The weakest evidence concerns
stopped coverage under dependence, because most paths did not stop. This
experiment rejects broad calibration claims for the tested settings; it does
not reject HMC, the scientific targets or the entire estimator family.

## Accounting

`A/final-settlement-01.json` records 2422.0453545683995 measured
supervisor-inclusive seconds, including both numerical attempts, the failed
reader, repair checks, analytic explanation and terminal replay. It releases
1777.9546454316005 unused seconds from the 4200-second reservation. The
separate reconciled SGQF balance becomes 134872.4638433829 seconds; running
SIR caps do not change. The ledger does not double-count children, nested
analysis timers or observer idle waits. Short unmetered inspection/edit
operations are disclosed rather than presented as exact measured compute.
The experiment remains an explicit CPU reference exception, not GPU
performance evidence.
