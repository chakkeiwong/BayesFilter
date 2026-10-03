# M21 controller confirmation and estimator diagnosis

The six predeclared controller cases completed all 2,400 replications, with
400 independent controller runs and independent fixed-count arms per case.
There were no implementation failures or hard chunk vetoes. This is evidence
for the controller with exact Gaussian AR(1) transitions; it does not exercise
HMC preparation or tuning. The separate public-HMC study remains necessary.

The frozen source is `m21-r1/source-r1`, based on `e6701cbf6`, with identity
`99f9fa5244e2000daf8424c3385764568b98188aa59446a5744dd6b98c6c0917`.
The later remote ESS repair is absent from these runs. Their source, settings,
seeds, commands, raw tensors and actual worker times remain under
`artifacts/hmc-repair-master-2026-09-16/m21-r1/`. The design and fixed inventory
are in the M21 design/confirmation plans and `confirmation-inventory.json`.
All numerical work was CPU reference work with GPUs intentionally hidden.

| Case | Posterior checks passed / 400 | Warmup caps | Stopped intervals covering stationary mean / 400 | Fixed lugsail coverage / 400 | Exact fixed Gaussian coverage / 400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent stationary | 400 | 0 | 374 | 372 | 377 |
| Stationary, rho 0.8 | 400 | 0 | 379 | 377 | 374 |
| Stationary, rho 0.98 | 225 | 41 | 336 | 379 | 385 |
| Common start 8, rho 0.98 | 230 | 51 | 326 | 372 | 379 |
| Starts (-8,-4,4,8), rho 0.995 | 0 | 398 | 2 | 293 | 373 |
| Common start 8, rho 0.9999 | 0 | 400 | 0 | 0 | 381 |

Unavailable intervals count as not covered in the planned denominator. An
interval from a retained-cap outcome remains a reported interval, even though
the posterior checks failed. Thus interval coverage and successful posterior
delivery are separate outcomes. Exact Gaussian intervals target the known
finite-count transient expectation; for the very slow common-start case that
expectation is far from the stationary mean zero. The exact intervals cannot
certify removal of initialization bias.

`confirmation-summary.json` records pointwise exact binomial intervals, all
caps and conditional as well as unconditional coverage. All six exact-oracle
screens passed their predeclared Bonferroni check. Independent and moderately
correlated cases passed the declared delivery and coverage screens. The slow
cases did not. For example, the stationary rho 0.98 case covered in 336/400
planned runs (95% interval 0.8003--0.8745), despite conditional coverage of
336/359 available intervals (0.9054--0.9590). Dropping warmup failures would
hide the delivery problem.

The rho 0.995 result triggered the planned fixed-count estimator diagnosis.
Using the stored fixed chains, the sqrt(n)=100 lugsail batch length covered
293/400 means and had median estimated/exact standard error 0.585. Its exact
expected untruncated variance estimate was only 0.3430 of the true finite-count
mean variance. Independent dense-covariance tests validate that expectation,
including the deterministic-start correction. This identifies finite-bandwidth
bias; it does not establish a wrong implementation of the lugsail formula.

Development comparisons at batch lengths 250 and 500 covered 349 and 367 of
400 means, with median estimated/exact standard-error ratios 0.813 and 0.961.
The existing TFP autocorrelation comparator covered 373/400 (95% interval
0.9033--0.9550), with ratio 0.987. These alternatives were examined on reused
evidence. They nominate independent confirmation; they do not support a
method ranking or a new default. A minimum batch count cannot by itself
establish adequate bandwidth, and lugsail cannot estimate burn-in bias.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept fixture engineering evidence | All 2,400 records and exact-oracle screens complete | No hard chunk veto | Only six Gaussian transition regimes | Preserve source and use fresh merged-source public fits | HMC pipeline validity |
| Keep slow-controller calibration open | Delivery/coverage screens fail | Caps are observed outcomes | Bandwidth bias versus readiness/finite-count limitations | Independently confirm nominated estimators before changing any default | Universal sufficient burn-in |
| Retain lugsail alternatives as hypotheses | Exact bandwidth diagnosis and development comparisons complete | No arithmetic mismatch found | Reused evidence and estimator variability | Fresh fixed-count calibration, then stopping calibration | Superiority or anytime coverage |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No numerical chunk veto or invalid oracle artifact; all planned outcomes retained |
| Statistically supported ranking | None; no paired method comparison was predeclared |
| Descriptive-only differences | Alternative bandwidths and autocorrelation comparator; observed durations |
| Default readiness | No default changed or promoted |
| Next evidence needed | Fresh estimator confirmation, merged-source public-HMC confirmation, and GPU repetition when capacity permits |

The six confirmation workers consumed 4737.1977903349325 seconds; the pilot
consumed 47.6015605708817. Reporting, estimator diagnosis and focused tests are
charged separately in the continuation ledger. Exact commands and environment
are preserved in `confirmation-cpu-r1/execution.json` and per-case manifests.
The strongest alternative explanation for the stopped coverage failures is
insufficient delivery under the fixed caps, rather than interval error alone.
The independent fixed arms and known transient laws distinguish those effects.
These results reject the tested adequacy claim for the slow cases; they do
not reject the research direction or invalidate viable HMC tuning siblings.
