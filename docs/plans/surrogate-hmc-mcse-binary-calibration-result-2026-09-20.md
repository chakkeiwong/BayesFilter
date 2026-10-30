# Binary-function diagnostic calibration result

The repaired candidate **passed all eight predeclared Gaussian calibration
screens** across 2,048 fresh replications. The result supports the scoped
diagnostic policy under the tested normal and AR laws. It does not establish
all-model surrogate-HMC validity or change any closed inference result.

The [plan](surrogate-hmc-mcse-binary-calibration-2026-09-20.md) defines the
exception for balanced binary CDF observables, the fresh seeds and the longer
rho 0.9 scope. The complete
[report](artifacts/surrogate-hmc-all-models-20260920-mcse-binary-calibration-01/report.json)
and per-replication JSONL preserve each look, diagnostic, raw folded statistic,
exception flag and interval check. The
[manifest](artifacts/surrogate-hmc-all-models-20260920-mcse-binary-calibration-01/run-manifest.json)
records the command, source snapshots, environment, seeds and accounting.
Source hashes were unchanged at completion.

| Situation | Draws per chain at two looks | Any-look veto | Both-look veto | Admitted interval failures; Wilson 95% upper bound | Criterion |
|---|---|---:|---:|---|---|
| IID normal | 1,000 / 2,000 | 8/512 | 0/512 | 10/512; 3.56% | False-veto and coverage pass |
| Stationary AR, rho 0.5 | 1,000 / 2,000 | 7/256 | 0/256 | 4/256; 3.95% | Coverage passes |
| Stationary AR, rho -0.5 | 1,000 / 2,000 | 6/256 | 0/256 | 3/256; 3.39% | Coverage passes |
| Stationary AR, rho 0.9 | 4,000 / 8,000 | 34/256 | 4/256 | 4/252; 4.01% | Coverage passes |
| Stationary AR, rho 0.99 | 1,000 / 2,000 | 256/256 | 256/256 | None admitted; coverage unassessed | Dependence veto passes |
| Chain offsets | 1,000 / 2,000 | 256/256 | 256/256 | No common stationary reference | Veto passes |
| Linear drift | 1,000 / 2,000 | 256/256 | 256/256 | No stationary reference | Veto passes |

The IID any-look veto rate was 1.56%, with Wilson 95% interval 0.79–3.05%;
its upper bound is below the declared 5% limit. All required coverage regimes
had at least 64 admitted replications and passed both unconditional and
admitted-conditional failure limits. For the three rejection controls, the
both-look veto-rate lower bound was 98.52%, above the required 95%. These are
per-regime bounds, not a simultaneous confidence statement across regimes.

The original runtime rule vetoed 467/512 IID replications in this run. The
binary repair preserves the raw folded result and flags the uninformative
component explicitly. It uses the bulk frequency diagnostic only in the stated
binary case and separately rejects constant or nonfinite raw observables.
The first focused test exposed roundoff that sometimes makes a constant fold's
computed R-hat appear finite. The policy therefore detects exact binary
balance from the data; it does not depend on observing a NaN.

The diagnostic adversaries were evaluated separately by regime. Naive IID
standard errors were close to the exact mean-variance oracle for IID draws,
but only about 0.58 of the oracle for rho 0.5 and 0.073 for rho 0.99. The
candidate's batch MCSE at the longer rho 0.9 look averaged 1.027–1.033 times
the oracle across coordinates. For rho 0.99 it remained only 0.733–0.738 times
the oracle; every such replication was vetoed. These ratios explain why
dependence protection remains necessary. They are descriptive averages, not
coverage guarantees or a ranking of general uncertainty estimators.

Eleven focused tests passed, including exact permutation enumeration, fixed
graph replay, binary scope and constant-series rejection, estimator parity
against the actual runtime at both existing looks, strict handling of
undefined fields, and an actual-worker 4,000/8,000-draw wiring smoke. Two
test batches consumed 32.786651 CPU seconds; the first contained the corrected
overly narrow NaN assertion. The calibration used 304.114897 CPU seconds and
264.829630 wall seconds. Total stage CPU use was 336.901548 seconds, within
the 900-worker/60-test allocation. GPU devices were intentionally hidden.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Keep this diagnostic as a viable prospective candidate | All Gaussian calibration criteria pass | Slow dependence, offsets and drift rejected | Untested target laws and function families; finite simulation uncertainty | Use explicit scope and fresh validation before any future runtime adoption; continue independent model preparation | No rescoring of LG1/C01, general MCMC validity or all-model surrogate-HMC success |

| Inference status | Finding |
|---|---|
| Hard veto screen | Candidate passes the declared calibration; old and first replacement failures remain preserved |
| Statistically supported ranking | No ranking of HMC forces, model performance or general MCSE methods |
| Descriptive-only differences | Runtime, batch-MCSE/oracle ratios and between-screen rate differences |
| Default-readiness | No runtime promotion; code remains in the diagnostic package |
| Next evidence needed | Scope-specific prospective validation and actual retained HMC comparisons under an explicitly declared uncertainty policy |

Post-run red team: longer rho 0.9 simulations changed the diagnostic scope;
they do not repair a short chain retrospectively. The negative controls remain
necessary because a randomization non-rejection does not prove independence.
Multimodal and heavy-tailed chains, different posterior functions and more
comparison families are not validated here. A fresh failure in those settings
would restrict this candidate without contradicting the narrow Gaussian result.
