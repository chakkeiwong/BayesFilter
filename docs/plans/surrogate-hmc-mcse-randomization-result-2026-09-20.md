# Batch-dependence randomization calibration result

The candidate is **not calibrated**. It vetoed52/512 IID replications at one or
both planned looks:10.16%, Wilson95% interval7.83–13.08%. The predeclared
upper-bound limit was5%. The goal remains testing surrogate HMC with imperfect
full DCTR forces; this experiment tests a diagnostic needed to interpret those
chains, and supplies no new model-inference result.

The [plan](surrogate-hmc-mcse-randomization-2026-09-20.md) froze the comparator,
two looks, seeds and decision criteria. The complete2048-replication
[report](artifacts/surrogate-hmc-all-models-20260920-mcse-randomization-02/report.json),
per-replication JSONL and source snapshots preserve the evidence. The
[manifest](artifacts/surrogate-hmc-all-models-20260920-mcse-randomization-02/run-manifest.json)
records the exact command and CPU-only environment. GPU devices were
intentionally hidden. Retry02 used260.277849 CPU-worker seconds and228.416028
wall seconds. Attempt01 had a reporting failure after22replications and used
13.903043 CPU seconds; two focused test batches used19.579273 CPU seconds.
Total293.760165 CPU seconds, within the600-worker/60-test allocation.

| Situation | Any-look candidate veto | Both-look veto | Interval failures among replications with an admitted look | Decision |
|---|---:|---:|---:|---|
| IID normal | 52/512 | 3/512 | 5/509; Wilson upper2.28% | False-veto criterion fails; coverage passes |
| Stationary AR, rho=0.5 | 21/256 | 0/256 | 4/256; upper3.95% | Coverage passes |
| Stationary AR, rho=-0.5 | 22/256 | 1/256 | 4/255; upper3.96% | Coverage passes |
| Stationary AR, rho=0.9 | 256/256 | 230/256 | 1/26; upper18.89% | Insufficient admitted evidence; coverage criterion fails |
| Stationary AR, rho=0.99 | 256/256 | 256/256 | No admitted replication | Required dependence veto passes |
| Chain offsets | 256/256 | 256/256 | Not a stationary reference case | Required veto passes |
| Linear drift | 256/256 | 256/256 | Not a stationary reference case | Required veto passes |

All three required pathological-regime veto rates have Wilson lower98.52%,
above the95% criterion. The original fixed-lag screen vetoed456/512 IID
replications in this fresh run. That comparison does not make the candidate
acceptable: its full diagnostic still fails the declared standard.

The IID failure has a localized explanation. Forty-four replications had an
undefined function R-hat; ten had a randomization rejection, with two overlapping.
The undefined values occur in binary CDF summaries when the pooled counts of
zero and one match exactly: the pooled midpoint median is0.5, so every absolute
deviation is0.5. Folding therefore produces zero within- and between-chain
variance and the R-hat calculation is0/0. The current screen correctly rejects
the undefined statistic. A prospective, explicitly scoped treatment of this
uninformative folded component needs its own tests; no value is silently
imputed in this completed run.

For rho0.9, all replications failed ESS and R-hat at the shorter look; most still
failed at the longer look. Four chains of2000draws have only about421 effective
draws for the mean under the exact stationary AR law, near the400 minimum.
The observed1/26 interval-failure count does **not** establish undercoverage.
It leaves coverage unresolved because too few replications were admitted.
Mean batch-MCSE/oracle ratios at2000draws were0.983–1.002; these descriptive
averages cannot settle simultaneous interval coverage.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Do not promote this candidate | IID false-veto and rho0.9 coverage criteria fail | Pathological-regime protection passes | Binary folded degeneracy; insufficient admitted dependent samples | Test a scoped binary diagnostic repair and prospectively longer rho0.9 simulations using fresh seeds | No revised interpretation of closed LG1/C01 results; no rejection of surrogate HMC |

| Inference status | Finding |
|---|---|
| Hard veto screen | Current candidate fails its declared calibration; no runtime change |
| Statistically supported ranking | No general ranking of uncertainty methods or HMC forces |
| Descriptive-only differences | MCSE/oracle ratios and observed differences between screens |
| Default-readiness | Not ready; production equivalence and R-hat code remain unchanged |
| Next evidence needed | Fresh prospective calibration after a tested, documented repair, including sufficient admitted dependent draws |

Post-run red team: the normal/AR family is narrow, and a randomization failure
to reject does not prove independent batches or valid MCMC intervals. A repair
that only improves IID admission could still miss dependence. Both coverage
and pathological-regime vetoes must survive fresh testing. Gaussian calibration
alone cannot establish arbitrary-model sampler validity.
