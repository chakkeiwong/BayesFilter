# M13 acceptance and saved-sample diagnostics

The first 9,000 posterior draws per chain met the agreed stopping rule. The
additional 3,750 saved draws raise a process-scale mixing concern supported by
traces, acceptance changes, autocorrelation, tail imbalance, and estimated Monte
Carlo error. This inspection adds no validation phase or stopping criterion and
produces no new samples.

## Acceptance and sampler health

| Chain | First 9,000 | Additional 3,750 | All 12,750 |
|---|---:|---:|---:|
| 1 | 80.58% | 81.97% | 80.99% |
| 2 | 81.04% | 37.09% | 68.12% |
| 3 | 81.74% | 68.48% | 77.84% |
| 4 | 78.38% | 85.84% | 80.57% |
| Average | 80.44% | 68.35% | 76.88% |

All stored finite-value, endpoint-status, movement, divergence, force-fallback,
fixed-kernel, checksum, and state/seed continuity checks passed. The largest
absolute proposal energy error was 62.884, below the declared energy-error veto
limit of 1,000; passing that limit does not establish low energy error. Longest
consecutive rejection runs were 12, 29, 16, and 12. Regular acceptance can coexist
with slow exploration.

## Precision and exploration

Parameters below are log process scale and log observation scale. ESS is pooled
across four chains. MCSE describes the posterior mean in log-parameter units;
it is not likelihood MCSE.

| Diagnostic | First 9,000: process / observation | All 12,750: process / observation |
|---|---|---|
| Maximum rank-normalized split/folded R-hat | 1.0074 / 1.0055 | 1.0542 / 1.0088 |
| Bulk ESS | 512 / 1,076 | 77 / 321 |
| Tail ESS | 324 / 1,956 | 30 / 483 |
| Posterior mean | 0.0765 / -0.1410 | 0.0303 / -0.1299 |
| Mean MCSE | 0.0112 / 0.0052 | 0.0418 / 0.0097 |
| Mean MCSE / posterior SD | 4.27% / 3.05% | 13.44% / 5.64% |

The [trace, running-mean, and autocorrelation figure](trace-running-mean-acf.png)
shows the later excursion. Chain 2's mean log process scale is 0.0913 in the
first 9,000 draws and -0.4793 in the additional 3,750. Its full-history process
scale autocorrelation is 0.625 at lag 100, 0.517 at lag 500, and 0.399 at lag
1,000; prefix values were 0.046, -0.017, and 0.031. Chain 3 also makes a later
lower-region excursion. These observations are descriptive, not independent
tests.

The pooled process-scale fifth percentile is -0.5987. Chains 1 and 4 never reach
that range; Chain 2 spends 17.77% of its draws there and Chain 3 spends 2.26%.
Observation-scale lower-tail occupancy ranges from 4.58% to 5.55%. These empirical
quantiles locate the imbalance; they create no new pass/fail threshold.

The pooled process-scale mean shifts only -0.0462 log units, about 0.15 of the
full-history posterior SD. Its point estimate is less sensitive than the tail
diagnostics. Adequacy depends on the intended inferential quantity and precision;
precise tail inference has weaker support. MCSE is itself estimated and can be
optimistic when exploration is poor.

## Other diagnostic tools

Trace/rank plots, running means and quantiles, autocorrelation, bulk/tail ESS,
MCSE, chain-distribution comparisons, and HMC numerical/energy diagnostics
address different failure modes. Formal within-chain stationarity tests are
also possible but cannot certify exploration of all posterior regions.
[Stan's diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html)
uses complementary diagnostics rather than acceptance alone.

This supplement computes traces, cumulative means, autocorrelation, tail
occupancy, and segment summaries alongside R-hat, ESS, and MCSE. Rank plots and
formal stationarity tests were not run. E-BFMI cannot be computed from the saved
chunk fields: they retain proposal energy differences, not the total Hamiltonian
energy series required. NUTS tree-depth diagnostics do not apply to this
fixed-leapfrog HMC run.

## Decision and uncertainty

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve 9,000-draw completion | Agreed R-hat rule met | Health passed | Future exploration can differ | Retain historical completion | Permanent convergence |
| Report later process-scale concern | No new promotion criterion | Numerical checks pass; full R-hat elevated | Lower-tail precision | Report complete saved evidence | Failure of surrogate HMC as a method |

| Inference status | Finding |
|---|---|
| Hard veto screen | No numerical, target-status, fallback, or continuity failure found |
| Viable candidate | Same frozen-endpoint surrogate-HMC candidate; lower-tail exploration unresolved |
| Statistically supported ranking | None; no methods compared |
| Descriptive-only differences | Acceptance, means, ACF, tail occupancy, and diagnostic estimates |
| Default-readiness | Not assessed; no exact-likelihood equivalence or all-model conclusion |
| Next evidence needed | Adequate lower-region exploration and precision for the intended estimand; no independent validation chain |

Post-run skeptical review: a stationary chain can make long correlated
excursions. The traces support a finite-run mixing concern, not proof that the
invariant target changed or a formal rejection of stationarity. Rare stationary
excursions are an alternative explanation; repeated exploration with stable tail
occupancy would reduce the concern. The weakest evidence is tail precision,
which depends heavily on one chain. These diagnostics do not determine whether
the frozen importance-likelihood posterior matches the original exact posterior.

## Execution

The command in [run-manifest.json](run-manifest.json) completed successfully in
6.93 seconds including interpreter startup. TensorFlow/TFP ran on CPU with GPUs
intentionally hidden before import; non-XLA diagnostic graphs are an explicit
reporting exception. Sample, original master, and tuning checksums matched;
archived state/seed continuity passed; recomputed R-hat, ESS, mean, MCSE, and
acceptance matched the saved report. The rendered figure was inspected.
Full values: [report.json](report.json). Reproduction: [analyze.py](analyze.py),
[command.log](command.log), [command receipt](command-receipt.json).

This saved-data assessment is complete. No master code or sampling rule changed.
Any future sampling continuation belongs to the existing master and the same
saved chains. It is not running because the previous sampling allocation
expired; this supplement requests no new continuation or validation phase.
