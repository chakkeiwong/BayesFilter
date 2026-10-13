# Optional MCSE maximum: completed calibration and decision

13 October 2026. The optional `lugsail_max_batch_means` implementation passes
the numerical and public-controller checks. The fresh calibration is complete:
128 replications in each of six conditions, with no omitted completed records.
The maximum raises estimated variance in the antithetic control as intended,
but this study does **not** establish nominal interval coverage or justify
changing the default. Keep it available as an explicitly selected posterior
precision option. Tuning membership and the active fixed-HMC baseline are
unchanged.

The [controlling plan](bayesfilter-modern-hmc-comparator-precision-continuation-2026-10-13.md)
and [implementation/sensitivity note](bayesfilter-mcse-safeguard-progress-2026-10-13.md)
give the original question, assumptions, checks and commands. Artifacts below
are relative to
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/precision-repair-01/`.

## What was computed

For each chain and quantity, B is the ordinary complete-batch long-run
variance estimate and L is the existing lugsail estimate. The new option
uses `max(B,L)` before independent-chain pooling. Both components must be
finite and B positive. It reports both components and every activation; it
does not clip nonfinite results to an apparently usable variance. Independent
arithmetic oracles, the actual posterior controller and nine trusted GPU/XLA
parity cases verify the intended computation. This is a local extension,
not the literature lugsail formula or a proof of conservative variance.

Calibration used four independent stationary Gaussian AR(1) chains at
rho=0, .9 and -.9, and paired Student-t(5) marginal transforms of the same
latent paths. Coordinate, squared coordinate and the indicator |x|>2 were
assessed using the actual TensorFlow posterior rules at lengths
1000, 2000, ..., 10000. Every method saw the same paths. There are **768
law/condition records from 384 independent latent four-chain units**; the two
marginal laws sharing a seed are not independent replications. Seed 73113
was declared before the run. Stationary initialization makes this an estimator
and stopping study, not a test of burn-in or exploration of missing modes.

The first attempt reached its 1800-second cap with 704 complete records.
The checked continuation retained them unchanged and computed the remaining
64. All 128 replications per condition completed. The terminal audit checked
77 numerical source hashes, all replication identities, the 704 carried
records, 5163 present method/endpoints, interval arithmetic, per-chain maxima,
activation masks, pooling and summary reproduction. Both complete raw result
files have losslessly checked gzip archives.

## Results

The entries below are the number of paths whose declared posterior assessment
first passed by 10000 draws per chain, out of 128. The rules remain R-hat
<=1.01, bulk/tail ESS >=400 and MCSE/SD <=.02 for all three quantities.

| Marginal, rho | Original lugsail | Optional maximum | Plain batch means | Autocorrelation |
| --- | ---: | ---: | ---: | ---: |
| Gaussian, 0 | 128 | 128 | 128 | 128 |
| Student-t(5), 0 | 128 | 128 | 128 | 128 |
| Gaussian, .9 | 0 | 0 | 16 | 0 |
| Student-t(5), .9 | 1 | 1 | 26 | 0 |
| Gaussian, -.9 | 128 | 128 | 128 | 127 |
| Student-t(5), -.9 | 128 | 128 | 128 | 128 |

At rho=.9 the Gaussian coordinate's exact fixed-length MCSE/SD is about
.02178 at 10000 draws, above .02. A higher stopping fraction from another
estimator is therefore not evidence of better sampling or uncertainty.
Failure to reach this requested precision does not invalidate the stationary
simulation or require a tuning repair.

At the fixed endpoint, the following ratios compare the mean estimated
coordinate variance with its exact finite-length variance. They are descriptive
Monte Carlo estimates, not exact expectations.

| Gaussian rho | Original lugsail | Optional maximum | Plain batch means |
| --- | ---: | ---: | ---: |
| 0 | .9850 | 1.0357 | .9933 |
| .9 | 1.0949 | 1.0955 | .9064 |
| -.9 | .9118 | 1.1067 | 1.1005 |

The antithetic result is consistent with the mechanism: the maximum prevents
the lugsail subtraction from pulling the estimate below ordinary batch means.
It introduces upward estimation in this control. That tradeoff is preserved,
not treated as automatic improvement.

For the antithetic coordinate at the fixed endpoint, Gaussian interval misses
fall from 11/128 to 5/128 and Student-t misses from 10/128 to 6/128. The paired
95% conservative intervals for the change in miss probability are respectively
[-.1075, .0188] and [-.0856, .0267]. Both contain zero. **None of the reported
paired binary-miss intervals across the six conditions and three quantities
excludes zero.** No coverage ranking is established by this study.

There are also unfavorable absolute coverage observations. The IID
Student-t coordinate has 14/128 fixed-endpoint misses under both policies,
with pointwise 95% binomial interval [.0611, .1767]. This does not support a
universal 95% coverage claim. Post-run explanatory checks show that even a
normal interval using the known IID variance misses 13/128 on the same
paths; the analogous Gaussian known-variance interval misses 11/128. The
replication sample has unusually large mean errors. This is a concrete
alternative explanation to an estimator defect, not grounds to discard these
seeds or ignore the observed misses. The Student-t known-variance normal
interval is itself asymptotic. These checks were added after the run and
cannot become promotion evidence.

In the Gaussian antithetic condition, eight paths stop later under the maximum
and 120 at the same look. The mean delay is 85.94 transitions per chain, with
an approximate pointwise paired interval [23.75, 148.12]; this is a local
sampling-cost observation, not a method ranking. Five paths across the two IID
conditions and the antithetic Student-t condition stop earlier. All five have
a nonpositive raw lugsail component at the earlier maximum-policy stopping
look. They are availability recoveries: the maximum was not smaller than a
valid original estimate. The remaining Student-t conditions include some
later stops as well. The complete paired counts remain in the evidence.

Every original and maximum estimate is available at the fresh fixed endpoint.
Thus this fresh stream does not replicate the previous two unavailable
antithetic Student-t endpoints; it cannot quantify their unconditional
frequency. The deterministic negative-estimate test and the five earlier-look
recoveries exercise the repair mechanism directly.

## Uncertainty and reporting repairs

The simulation driver and its original summaries remain frozen. The terminal
analysis supersedes two summary interpretations without rerunning samples:

1. Paired binary-miss intervals use the probabilities of the two discordant
   outcomes. Each gets a 97.5% Clopper-Pearson interval; a union bound and
   subtraction give a conservative 95% interval for their difference. Unlike
   a generic Student-t calculation, this retains uncertainty when all observed
   pairs tie. It is pointwise across conditions, not multiplicity-adjusted.
2. For Student-t(5), E[X^4] exists but E[X^8] does not. Mean precision for X^2
   can therefore be defined while the across-replication variance of its
   quartic batch-variance statistic need not exist. Its paired estimated-
   variance change remains descriptive; the terminal report marks the
   Student-t uncertainty interval unsupported and omits it.

Continuous paired intervals are approximate. Comparisons on common valid
stopped endpoints are conditional on that subset; nonstoppers and unmatched
outputs remain explicit. None of these intervals is an anytime-valid
posterior interval or a proof that warm-up was sufficient.

## MAMS sensitivity, budget and next decision

Reassessment of all archived looks of the four completed MAMS members
reproduces the original precision calculations within 1.57e-13 and preserves
their first passing counts: 6000, 10000, 6000 and 3000. Maximum-policy MCSEs
increase by up to 10.4% across these endpoints. This is sensitivity on saved
draws only; original posterior/reference results are not reclassified.

Actual CPU-study charges, including subprocess startup, are 1986.120593
seconds for calibration/supervision, 30.844274 for MAMS sensitivity and
2.270380 for terminal audit, totaling **2019.235247 seconds**. The
3600-second reservation releases **1580.764753 seconds**, held pending the
next terminal refresh. The local-check reserve remains fully held, and the
active fixed-HMC allocation remains 79364.235006 seconds. No extra campaign
has been launched against the released balance.

`terminal-audit-01/result.json` contains corrected summaries and checks;
`terminal-audit-01/explanatory-checks.json` preserves the post-run alternative
explanation and stopping counts. `settlement-01.json` uses full supervisor
charges rather than the slightly smaller internal audit timer. Exact source,
commands, environment, device settings and seeds are in the attempt manifests,
source reproduction note and numerical checks.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain explicit optional maximum | Arithmetic, public-controller and GPU parity checks pass | No checked source, identity, pooling or nonfinite defect | Finite-sample calibration outside these controls | Keep default unchanged; retain reports | Conservative true variance or general robustness |
| Do not promote coverage/default claim | Calibration complete; no supported paired coverage ranking | IID misses and limited replication prevent broad assurance | Replication error, tails and repeated looks | Fresh targeted coverage work only under an updated budgeted plan | Nominal 95% or anytime coverage |
| Preserve completed MAMS evidence | All four original-policy replays match; stopping counts unchanged under sensitivity | No invalid archived tensor detected | Same target/seed and saved draws | Finish independently running fixed-HMC baseline | General method superiority or release |
| Continue master program | CPU phase complete; fixed-HMC still active | No terminal comparator error observed | Full comparator delivery and remaining N5/N8 evidence | Monitor run, settle terminal cost, refresh next phase | Master completion |

| Inference status | Finding |
| --- | --- |
| Hard veto evidence | No tested implementation/evidence defect; nonfinite components cannot grant precision |
| Statistically supported ranking | None for coverage; intervals are pointwise and limited in power |
| Descriptive-only differences | Variance ratios, most stopping differences and saved-MAMS sensitivities |
| Default-readiness | Not established; estimator remains optional |
| Next evidence needed | Completed classical comparator; broader independent model/seed evidence; calibrated posterior uncertainty before any default promotion |

Post-run skeptical conclusion: larger variances can widen intervals and delay
stopping without fixing missing modes or insufficient equilibration. The
unchanged saved-MAMS result is compatible with the repair but is not independent
validation. The strongest challenge to an estimator-specific explanation of
IID misses is the known-variance comparator on the same sample. Additional
independent replications could overturn either favorable or unfavorable
descriptive differences. These limits leave a sound optional implementation
and an unfinished statistical/release program.
