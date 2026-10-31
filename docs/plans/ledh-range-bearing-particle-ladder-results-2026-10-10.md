# Range-bearing particle-count ladder

Both requested counts completed.

More particles are descriptively helpful in most tested cases, but they do not
resolve every case. At N4032, mean score error is 33–59% lower than N1008 in five
of six cells. In the difficult data18020 cell it is 25% higher. All twelve
reference-sensitive error intervals include zero, so this experiment establishes
no statistically supported particle-count ranking under its declared criterion.

The estimated sum of score-component variances at N4032 is 70–90% lower in those
five cells and approximately doubles in the remaining cell. Log-likelihood SD
falls in all six cells, by 39–82%. These variance comparisons are descriptive
across eight designs. A fixed-design surrogate does not draw fresh random score
noise at every HMC step; these numbers measure sensitivity to its frozen particle
design. Parameter-dependent approximation error can still affect inference.

Observed median time for one value plus the two-component score rises from
approximately 2.16s at N1008 to 6.46s at N2016 and 16.75s at N4032. Settings and GPU
contention differ, so these are descriptive costs, not an equal-runtime test.
This supports retaining more particles as an option, but does not establish
4032 as a necessary or preferred default. At N4032 the simple prior-proposal
comparator still has lower descriptive mean score error in four of six cells.

FP64/TF32-off, GPU/XLA, T20, state4/observation2/parameter2; eight paired designs per cell. Each N was tuned separately on the same disjoint calibration and validation protocol. These are filter score diagnostics preparing surrogate HMC; no HMC chain was run.

| Data | Point | N1008 mean score error / spread | N2016 mean score error / spread | N4032 mean score error / spread |
|---|---|---:|---:|---:|
| 1704 | truth | 1.274 / 1.397 | 0.627 / 0.678 | 0.523 / 0.654 |
| 1704 | difficult | 5.584 / 7.733 | 3.842 / 5.200 | 2.725 / 3.156 |
| 18020 | truth | 0.982 / 1.174 | 0.382 / 0.388 | 0.506 / 0.379 |
| 18020 | difficult | 1.400 / 1.388 | 1.601 / 2.574 | 1.754 / 1.962 |
| 18021 | truth | 1.640 / 1.969 | 1.316 / 1.758 | 0.755 / 0.660 |
| 18021 | difficult | 4.438 / 5.178 | 4.269 / 3.123 | 2.985 / 2.854 |

Spread is sqrt(sum of sample coordinate variances). It is across particle designs, not HMC MCSE. Centroid discrepancies, likelihood offsets/SD, every comparator, and full intervals are in ladder-report.json.

| Data | Point | N | New-minus-N1008 score error | Reference-sensitive interval | Verdict | Heuristics with lower descriptive mean |
|---|---|---:|---:|---|---|---|
| 1704 | 0 | 2016 | -0.646 | [-1.467, 0.174] | no_supported_ranking | none among valid arms |
| 1704 | 0 | 4032 | -0.750 | [-1.721, 0.220] | no_supported_ranking | none among valid arms |
| 1704 | 1 | 2016 | -1.742 | [-6.535, 3.051] | no_supported_ranking | prior, bootstrap_fisher |
| 1704 | 1 | 4032 | -2.859 | [-8.098, 2.380] | no_supported_ranking | prior |
| 18020 | 0 | 2016 | -0.600 | [-1.627, 0.427] | no_supported_ranking | none among valid arms |
| 18020 | 0 | 4032 | -0.477 | [-1.404, 0.451] | no_supported_ranking | prior |
| 18020 | 1 | 2016 | 0.201 | [-1.853, 2.255] | no_supported_ranking | none among valid arms |
| 18020 | 1 | 4032 | 0.355 | [-1.536, 2.245] | no_supported_ranking | prior |
| 18021 | 0 | 2016 | -0.324 | [-1.853, 1.204] | no_supported_ranking | none among valid arms |
| 18021 | 0 | 4032 | -0.885 | [-2.223, 0.453] | no_supported_ranking | none among valid arms |
| 18021 | 1 | 2016 | -0.168 | [-5.486, 5.150] | no_supported_ranking | bootstrap_fisher |
| 18021 | 1 | 4032 | -1.452 | [-6.507, 3.602] | no_supported_ranking | prior, bootstrap_fisher |

An interval spanning zero supports no ranking. These intervals are nominal, conditional and exploratory, following inspection of N1008 and with no multiple-comparison adjustment. Smaller point-estimate spread is descriptive only. Identical design-seed indices across N are coupled labels, not nested Gaussian draws. Retuning also changes settings, so this compares tuned particle budgets, not N with all controls held fixed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Requested extension complete | Per-cell errors and spreads above | Invalid/missing cells retained | Eight designs and reference bias | Assess materiality for the downstream target | Necessary final N |
| Default/HMC status unchanged | No default-precision/posterior test | Heuristic losses/incomplete screens above | Precision and posterior consequences | A separate tuned FP32 scope and actual HMC would answer those questions | Default readiness, posterior convergence, all-model improvement |

| Inference status | Assessment |
|---|---|
| Hard veto screen | See per-arm valid counts in JSON; no invalid-subset ranking |
| Statistically supported ranking | None: all twelve reference-sensitive intervals include zero |
| Descriptive differences | Spread ratios, log-likelihood SD, centroids, runtime and heuristic means |
| Default readiness | Not established: explicit FP64 diagnostic |
| Next evidence needed | Independent replication/reference refinement and downstream HMC materiality |

The strongest alternative explanation is that a larger particle count selects a different calibrated mixture, or that a few baseline outliers drive the contrast. Reference bias, few designs and previously inspected datasets limit the inference. Fresh data/particle designs reversing the result, or a refined reference revealing a persistent shift, would overturn an improvement interpretation. More particles do not automatically establish a correct posterior.

Artifacts: docs/plans/artifacts/ledh-range-bearing-particle-ladder-20261010-02. Plan: docs/plans/ledh-range-bearing-particle-ladder-2026-10-10.md. Numerical commands, source hashes, environment, verified GPU memory growth, allocator usage, seeds and elapsed times are in execution.json and per-worker manifests. Aggregate worker wall time: 5924.5s of 14400s; 11 of 20 workers.

Terminal review confirmed 192 unique rows per requested N, all 48 new-method and
48 previous-method rows finite and valid at each N, exact tuning-scope matches,
unchanged copied data/reference hashes, verified GPU memory growth, and the
RMSE/centroid/variance decomposition. The prior comparator has one invalid row at
N2016; the global-guide comparator has two invalid rows at N2016 and one at
N4032. Those failures remain in the evidence and are not dropped to rank a
successful subset. See terminal-diagnostic.py and terminal-diagnostic.json in
the artifact directory. Numerical preflight additionally passed 25 healthy and
report regressions, exact N1008 replay, blocked-transport parity, and analytical
score finite-difference checks at both requested particle counts.

The requested ladder is finished. FP32 retuning and downstream surrogate HMC
would be separate phases beyond this particle-count extension; neither was
silently launched. No unresolved execution failure prevents using this result.
