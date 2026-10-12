# MCSE calibration: completed fixed and stopped reference study

12 October 2026. The [planned reference study](bayesfilter-mcse-stopping-calibration-2026-10-12.md)
is complete: **128 replications in each of six conditions**, with four
independent chains, 10000 draws per chain and three quantities per condition.
The result does not justify promoting a new precision default. It confirms
that the configured lugsail estimator need not be conservative for antithetic
chains, and that passing the operational stopping rule does not establish
nominal interval coverage.

The six conditions are stationary Gaussian AR(1) paths at rho=0,.9,-.9 and
their paired exact Student-t(5) marginal transforms. Every method sees the same
paths. There are 768 completed law/condition records from **384 independent
latent four-chain units**; the Gaussian and Student-t records sharing a seed
are paired observations. Coordinate means, squares and the event |x|>2 are
compared with their exact expectations. Reference generation uses diagnostic
NumPy/SciPy; the numerical precision and posterior assessment are the actual
TensorFlow routines in committed `e68cdf1c4`.

The first 1800-second allocation completed 717 records. A checked continuation
copied those records unchanged and computed only the remaining 51 using the
original predeclared seeds. It took 140.585811 seconds. Total calibration cost
is **1940.794223 CPU-worker seconds** (32.35 minutes), inside the same held
3600-second local allowance. The first cutoff is preserved as
`mcse-calibration-01`; the complete result is `mcse-calibration-02`. Finishing
the fixed replication count avoids interpreting a potentially outcome-dependent
wall-time prefix as a fixed-size binomial study. No sample or estimator was
selected to improve a result.

## Operational stopping outcomes

Each entry is the number of replications whose actual retained assessment first
passed by 10000 draws, out of 128. The assessment uses R-hat <=1.01, bulk/tail
ESS >=400 and MCSE/SD <=.02 for every quantity. It starts from stationary paths
and therefore does not assess warm-up or global exploration.

| Marginal and correlation | Lugsail | Plain batch means | Even-batch lugsail | Autocorrelation |
| --- | ---: | ---: | ---: | ---: |
| Gaussian, 0 | 128 | 128 | 128 | 128 |
| Student-t(5), 0 | 128 | 128 | 128 | 128 |
| Gaussian, .9 | 0 | 17 | 0 | 0 |
| Student-t(5), .9 | 0 | 28 | 0 | 1 |
| Gaussian, -.9 | 128 | 128 | 128 | 128 |
| Student-t(5), -.9 | 128 | 128 | 128 | 128 |

The zero counts under positive dependence are not evidence of a broken chain.
For the Gaussian coordinate at rho=.9, the exact fixed-length MCSE/SD at
10000 draws and four chains is **.02178417**, above the requested .02. A
larger stopping fraction can reflect a smaller uncertainty estimate; it does
not by itself establish better performance. This finite-count constraint is
separate from an estimator's interval calibration.

At the fixed endpoint, lugsail and even-batch lugsail lack a valid coordinate
variance in 2/128 antithetic Student-t replications. Both remain in the result;
their interval-miss rates are conditional on the 126 valid intervals. At their
first passing look all 128 paths have usable estimates, so earlier operational
success does not imply that a later estimate must remain available.

## Checked finite-sample explanation

The target is the variance of the mean across four independent stationary
Gaussian chains. For one chain let

`V(k) = [1 + 2 sum_(j=1)^(k-1) (1-j/k) rho^j]/k`.

This is the exact finite-k variance of its coordinate mean, obtained by
summing covariances. With batch length b and a=floor(n/b) complete batches,
the expected batch-means long-run variance estimate implemented in the library
is `b*a/(a-1) * [V(b)-V(a*b)]`. To see the centering term explicitly, the batch
means Y have `Var(Y_j)=V(b)` and `Var(Ybar)=V(a*b)`, so
`E[sum_j (Y_j-Ybar)^2] = a*V(b)-a*V(a*b)`.
Multiplying by b/(a-1) gives the expression above. The actual lugsail formula
with r=3,c=.5 is twice the b estimate minus the floor(b/3) estimate.

At n=10000 the two batch lengths are 100 and 33. Dividing the expected pooled
variance estimates by the exact pooled mean variance gives:

| Gaussian correlation | Plain batch means | Lugsail |
| --- | ---: | ---: |
| 0 | 1.000000 | 1.000000 |
| .9 | .905176 | 1.088796 |
| -.9 | 1.094645 | **.893587** |

These are deterministic finite-covariance calculations, checked against twelve
direct small covariance-matrix sums. In the antithetic control, the unmasked
lugsail variance estimate is downward biased by about **10.6%**. A claim that
this configuration is universally conservative would be wrong. This is not
a discrepancy between code and its stated formula: the formula itself has
this finite-sample behavior. The actual runtime additionally rejects any
nonpositive per-chain estimate. The table therefore describes the unmasked
formula's expectation, not the conditional expectation after that validity
screen, nor variance conditional on a stopping time.

The even-batch sensitivity changes some earlier looks, but both lugsail
policies have b=100 at the fixed endpoint and agree there exactly. It cannot
be credited with repairing that endpoint's bias.

## Interval diagnostics and interpretation

The study recorded normal 95% intervals at the fixed endpoint and at each
method's first passing look. Selected antithetic coordinate results illustrate
the remaining uncertainty. The complete result preserves all quantities and
conditions; these rows do not select an estimator.

| Condition and method | Endpoint | Misses / valid intervals | Pointwise 95% binomial interval for miss probability |
| --- | --- | ---: | ---: |
| Gaussian, lugsail | Fixed | 10/128 | .0381–.1390 |
| Gaussian, lugsail | Stopped | 12/128 | .0494–.1580 |
| Gaussian, autocorrelation | Fixed | 15/128 | .0671–.1859 |
| Student-t(5), lugsail | Fixed | 10/126; 2 unavailable | .0387–.1411 |
| Student-t(5), lugsail | Stopped | 12/128 | .0494–.1580 |
| Student-t(5), even-batch lugsail | Stopped | 17/128 | .0793–.2041 |

These pointwise intervals express replication uncertainty; they are not
adjusted for inspecting many conditions, quantities and estimators. The
within-chain normal intervals are CLT diagnostics and have no anytime-coverage
guarantee. Neither the selected rows nor a favorable raw stopping fraction
supports a method ranking. The antithetic findings and exact bias calculation
justify further estimator/stopping work; they do not establish that one of the
alternatives is a safe replacement for all posterior quantities.

## Evidence and next action

All files below are under
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01`.
`mcse-calibration-02/manifest.json` records the exact command, seed 72110,
source commit/hash, TensorFlow 2.20.0/TFP .25.0, NumPy 2.1.3/SciPy 1.17.1,
two CPU threads, deliberately hidden GPUs, CPU/XLA reference role, timing and
parent result hash. The driver repair is committed as `9075f1a42`.
Both attempts' raw results are also preserved as `result.json.gz`, with
lossless roundtrip checks and original JSON hashes in `mcse-archive.json`.
The full uncompressed JSON remains available at the audit's recorded paths.

The [full replication audit](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/mcse-analysis-02/result.json)
checks all 77 numerical source hashes, all 768 identities, the unchanged 717
carried records, summaries, the parent hash and cumulative compute charge.
The scripts and results in `mcse-analysis-01` and `mcse-analysis-02` preserve
the covariance derivation checks and terminal accounting. The continuation
repair passed 21 focused tests, including changes to source, seed, replication
count, law, library version and duplicate, incomplete or nonprefix records.
Together with the pricing/continuation checks there are 64 distinct passing
focused regressions; these engineering checks are separate from calibration.

An additional **14 deterministic CPU/XLA regressions passed** against the
immutable numerical snapshot in 3.553224 seconds including framework startup.
The TensorFlow batch kernel agrees with an independently summed covariance
oracle for independent, persistent and antithetic controls, including discarded
batch remainders. Cholesky columns provide exact second-moment quadrature, so
these tests incur no Monte Carlo uncertainty. They also preserve the n=10000
antithetic counterexample and reject a negative per-chain variance even when
the pooled estimate would be positive. The tests are in
`tests/test_hmc_precision_covariance_reference.py`; the source/test hashes and
JUnit output are in `mcse-kernel-regression-01`. There are now **78 distinct
passing focused regressions** across this phase. These checks support the
formula and guard behavior, not nominal interval coverage.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close the bounded reference experiment | All predeclared 128 identities per condition, actual assessment, exact-reference checks | No source or denominator mismatch; two invalid fixed-endpoint estimates preserved | Calibration beyond these stationary laws and finite budgets | Preserve the complete evidence and use it to design estimator/stopping repair | Universal coverage or warm-up sufficiency |
| Keep estimator promotion open | Finite antithetic bias and several elevated miss rates remain | No estimator is admitted as a replacement default | Multiplicity, dependence and finite-sample behavior across quantities | Compare specified alternatives on fresh fixed-count replications and actual saved HMC quantities | A preferred estimator or guaranteed stopped coverage |
| Continue scoped MAMS feasibility | Numerical source and original posterior rule remain frozen | Candidate/posterior checks retain their existing roles | Geometry and MCSE limitations can both explain delivery failures | Let completed prices determine the bounded allocation; diagnose failed quantities separately | That this calibration repaired MAMS mixing |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Source, stream and record audits pass. Two unavailable lugsail fixed-endpoint estimates are explicit availability failures. |
| Statistically supported ranking | None. No multiplicity-aware paired superiority analysis was predeclared. |
| Descriptive-only differences | Stop counts, stop lengths, observed error/variance ratios and cross-estimator differences. |
| Default-readiness | No estimator or stopping default is promoted. |
| Next evidence needed | A specified estimator repair with finite-covariance controls, fresh fixed/stopped replication and downstream HMC checks. |

Post-run red-team: stationary initialization removes the warm-up problem by
construction, and the chosen laws cannot test global exploration of a nonlinear
state-space posterior. Small finite-sample variance bias and selection at the
first passing look can explain elevated miss rates without an implementation
error. A fresh, properly powered comparison could overturn an apparent
empirical ordering, but cannot overturn the checked finite-Gaussian expectation
for the formula and settings used here.
