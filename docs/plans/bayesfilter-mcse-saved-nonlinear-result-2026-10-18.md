# Saved nonlinear-HMC precision replay: terminal result

18 October 2026. This is a CPU-reference diagnostic on previously tuned,
already-stopped NL-PP-SGQF chains. It is not a new sampler run or a production
policy comparison. The six variance-estimator arms were frozen before the
concurrent synthetic interval study finished. The original tuning, delivery
and reference dispositions remain unchanged.

## Outcome and reconstruction repair

The full replay completed all 26 members and all 159 original retained
assessment windows. The independent audit passed 795 batch-arm checks,
reconstructed every member, checked all 1,157 bound inputs, and preserved the
complete member inventory. The maximum absolute discrepancies were
`1.4210854715202004e-14` for reconstructed quantities and
`3.979039320256561e-13` for original-report replay. They satisfy the unchanged
engineering tolerances, relative `1e-10` and absolute `1e-12`.

The failed first smoke identified a diagnostic-reader error. Committed chunks
store whitened active states `z`; the original posterior controller first
computes `q = center + z @ factor.T` and then the 18 named nonlinear
quantities. Both new diagnostic reconstructions initially omitted the affine
map. The repaired reader binds the saved geometry checksum and applies the
original composition. Its independent NumPy/SciPy reconstruction uses the same
mathematical map through a separate implementation. A shifted, nonsymmetric
fixture detects omitted maps, wrong centers and wrong transposes. Executable
parity also checks the original frozen public affine mapping.

This restores the offline reader to the computation performed by the original
sampler. It does not change that sampler. Failed freezes, tests, inventories
and the first smoke remain preserved and charged. The repaired frozen source
passed 60 focused tests, an independently audited original-member smoke, and
an independently audited tiny-cap check that retained all 26 denominator rows
while replaying none.

## Complete denominator and descriptive sensitivity

These are original dispositions, copied without reassessment. A delivery
requires both the original posterior and reference screens; a posterior pass
alone is insufficient.

| Cohort | Members replayed | Original looks | Original posterior passes | Original deliveries |
| --- | ---: | ---: | ---: | ---: |
| Fixed HMC | 8 | 50 | 7 | 7 |
| Randomized HMC | 8 | 46 | 6 | 5 |
| SNAPER-proposed randomized HMC | 6 | 38 | 5 | 5 |
| Fully refreshed MAMS | 4 | 25 | 4 | 4 |

Every look has 18 named quantities. The following counts and ratios describe
the saved windows, with no candidate or estimator ranking. The twice-square-
root arms use the predeclared `min(isqrt(4*n), n//20)` batch schedule; all other
arms use their original batch policy. Ratios compare each terminal quantity's
MCSE with its original lugsail MCSE when both are available and positive.

| Diagnostic arm | Available look quantities / 2,862 | Available terminal quantities / 468 | Terminal MCSE ratio: minimum / median / maximum |
| --- | ---: | ---: | --- |
| Original lugsail | 2,859 | 468 | 1 / 1 / 1 |
| Maximum of lugsail and batch means | 2,862 | 468 | 1 / 1.013 / 1.162 |
| Batch means | 2,862 | 468 | .827 / .977 / 1.162 |
| Autocorrelation | 2,862 | 468 | .840 / 1.001 / 1.319 |
| Twice-square-root lugsail | 2,789 | 465 | .714 / .991 / 1.297 |
| Twice-square-root maximum safeguard | 2,862 | 468 | .805 / 1.024 / 1.303 |

The longer-batch raw lugsail arm has 73 unavailable look-quantity estimates,
including three at terminal windows; the original arm has three, all before
terminal windows. Thus longer batches do not by themselves resolve variance
availability. The maximum safeguard has available estimates throughout this
inventory, but availability cannot establish accurate MCSEs or confidence
coverage. The ratios show sensitivity in both directions. These correlated
quantities and adaptively selected candidates are not independent replications;
their ranges, medians and maxima carry no statistical ranking inference.

## Decision and limits

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close the saved-data replay | Complete original inventory and faithful report replay passed | Source, geometry, retained-only reconstruction and independent batch checks passed | Saved, selected and stopped chains have no exact expectation oracle here | Interpret alongside the separately audited synthetic interval controls | MCSE calibration or convergence |
| Preserve all original decisions and defaults | No promotion was permitted | Longer-batch raw variance unavailability is preserved | Availability is not calibration; the finite-filter reference is not an exact posterior oracle | Use interval-study controls to distinguish variance scale from interval shape | Estimator replacement or superior sampler |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No unresolved replay-integrity veto. Negative/unavailable variances are retained findings. The failed first smoke was repaired before the single full attempt. |
| Statistically supported ranking | None; neither estimators nor samplers are ranked. |
| Descriptive-only differences | Availability counts and MCSE ratios on the fixed saved windows. |
| Default-readiness | Not established; no runtime, tuning or posterior default changed. |
| Next evidence needed | Terminal synthetic interval controls, broader matched model/seed evidence, and actual consumer/release qualification. |

The strongest alternative explanation for a favorable MCSE ratio is estimator
underestimation rather than greater sampler efficiency. The weakest evidence
is calibration on these adaptively selected saved endpoints: exact posterior
expectations and independent replications are absent. A separate controlled
study with known finite-length variance can overturn an interpretation about
variance scale; this replay cannot. Autocorrelation estimates use the public
TensorFlow authority and were not independently numerically reimplemented by
the terminal auditor. Independent checks cover the five batch-based arms.

## Provenance and settlement

Plan: [saved-data diagnostic](bayesfilter-mcse-saved-nonlinear-diagnostic-2026-10-18.md).
Let `R` be
`artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/saved-nonlinear-precision-20261018-01`.
The immutable numerical base is commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`, tree
`2a5277fac286732c5d94a2bd626ea72f39a772ec`, with the reviewed precision overlay.
Exact source identities, commands, environment, original seeds and wall times
are in `R/source-09`, `R/frozen-sources-09.json`, `R/design-02.json`,
`R/attempt-01/manifest.json` and the `R/command-*.json` receipts. GPU devices
were intentionally hidden before framework import; TensorFlow/TFP used CPU
XLA with two framework/OMP threads. There were no new random draws or seeds.

The complete scientific attempt took 22.870651 inclusive seconds and its
independent audit 5.129619 seconds. All setup, failed preflights, tests, smokes,
audits and terminal checks total `131.3668131614104` CPU-worker seconds from
the existing 1,800-second reservation. `R/final-settlement-01.json` returns
`1668.6331868385896` seconds to the same released SGQF pool. While the separate
7,200-second interval-study hold remains active, the unallocated balance is
`127520.20167178978` seconds. SIR caps, the separate 3,600-second hold and the
historical balance are unchanged. No new funds were granted or silently spent.

Terminal evidence is `R/attempt-01/result.json`, `R/audit-full-01.json`,
`R/summary-01.json` and `R/final-settlement-01.json`; the summary retains full
precision. No raw tensors were duplicated. Historical failures remain intact.
