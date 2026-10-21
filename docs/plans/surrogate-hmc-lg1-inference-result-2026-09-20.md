# LG1 surrogate-HMC attempt4: first retained result

The full-DCTR surrogate-force arm completed four chains with 2000 retained
draws per chain against the exact Kalman posterior endpoint. **B/R is
INCONCLUSIVE**, not a detected discrepancy. All frozen mixing, function-R-hat,
MCSE sensitivity and batch-correlation screens passed. The confidence intervals
establish equivalence for the standardized mean, centered second moment and
predictive probability, but not all three CDF functions at their .02 margins.
The exact-force comparator A likewise remains INCONCLUSIVE at two CDF margins.

This tests the intended surrogate-HMC mechanism: an imperfect full canonical
DCTR field proposes moves and the independently bound exact endpoint performs
Metropolis correction. It does not test DCTR likelihood accuracy. The
finite-endpoint C/D,D/R,C/R comparisons are separate work.

Evidence: [sampling report](artifacts/surrogate-hmc-all-models-20260919-lg1-sample-01/P2-sample/report.json),
[archive inspection](artifacts/surrogate-hmc-all-models-20260919-lg1-sample-01/archive-inspection.json),
[run manifest](artifacts/surrogate-hmc-all-models-20260919-lg1-sample-01/run-manifest.json),
[executed source checksums](artifacts/surrogate-hmc-all-models-20260919-lg1-sample-01/executed-source/sha256.json).
The manifest preserves the exact master command, environment, GPU memory policy,
plan, source identities and independent tuning roots. All 16 retained chunk
archives passed checksum inspection; warmup and verification draws were excluded.

| Arm | Maximum parameter R-hat | Bulk ESS | Tail ESS | Acceptance range | Final result |
|---|---:|---:|---:|---:|---|
| A, exact force / exact endpoint | 1.00118 | 3585.75 | 2679.18 | .7245–.7575 | INCONCLUSIVE |
| B, DCTR force / exact endpoint | 1.00148 | 3843.85 | 3602.99 | .727–.739 | INCONCLUSIVE |

Acceptance, ESS and timing are descriptive. There is no supported efficiency
ranking. Both arms have zero divergences, force fallbacks and numerical endpoint
failures; maximum absolute energy error is below6.69. A used22.28 seconds and
B about1280 seconds inside the workers; compilation and unequal target costs
are included. The master sampling job charged1315.271 GPU seconds and
1315.066 CPU-worker seconds. Tuning and preparation costs remain charged
separately in the cumulative campaign ledger.

The intervals below use the frozen six-function, two-look multiplicity rule,
Student-t critical value with36 pooled batch degrees of freedom, the maximum
MCSE across three batch sizes, and the empirical reference allowance.

| B/R function | Difference | Lower | Upper | Margin | Equivalent |
|---|---:|---:|---:|---:|---|
| Standardized phi | .04215 | -.00465 | .08896 | .10 | Yes |
| Centered second moment | -.02426 | -.09490 | .04638 | .10 | Yes |
| CDF at R's .1 quantile | -.00863 | -.02317 | .00592 | .02 | No |
| CDF at R's .5 quantile | -.01050 | -.03581 | .01481 | .02 | No |
| CDF at R's .9 quantile | -.00862 | -.02272 | .00547 | .02 | No |
| Pr(y11>0 given y) | -.00950 | -.02016 | .00117 | .03 | Yes |

Every interval contains zero and overlaps its equivalence region. The
inconclusive decision reflects insufficient precision, not proof of equality
and not proof of bias. The stationary-chain CLT and approximately independent
batch means remain assumptions; refinement allowances are empirical rather
than rigorous integration bounds. One data realization and particle bank do
not establish cross-model or cross-dataset reliability.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| B remains viable | Three of six intervals inside margins; joint test inconclusive | All declared chain/MCSE screens pass | CDF MC error | Preserve two-look outcome; complete independent finite-endpoint comparisons | Posterior recovery established |
| A remains viable | Four of six intervals inside margins; joint test inconclusive | Screens pass | CDF MC error | Retain as descriptive comparator | Superiority or independent reference certification |
| Attempt4 used | No extra look authorized by this test | No continuation veto | Precision | C/D,D/R,C/R on frozen scope; later fresh-data attempt5 if justified | Research direction rejected |

| Inference status | Evidence |
|---|---|
| Hard veto screen | None fired in retained A/B; no missing archive tensors |
| Statistically supported ranking | None |
| Descriptive-only differences | Acceptance, ESS, runtime and observed mean differences |
| Default readiness | Not established; heuristic dominance is NOT_RUN |
| Next evidence needed | Independent finite posterior, separately tuned C, and eventual longer/fresh-data precision study |

Post-run red team: the strongest alternative explanation for observed mean
differences is Monte Carlo variation; intervals currently support it. A future
qualified interval wholly outside a frozen margin would overturn the present
absence of a detected discrepancy. The weakest evidence is the single
realization and modest CDF precision. The result invalidates neither the
harness, mathematical mechanism, target nor research direction.

Metadata errata: the original master scope card accidentally retained P1 and
`mechanics_only=true` and labeled `data_replicate=4`. The actual worker and
reference manifests identify inference attempt4, claim partition, replicate0.
The original artifacts are preserved; the master now writes these fields
correctly. Its generic `inference_blocker` text also remained stale after
successful inference and is corrected for future runs. Exit2 denotes incomplete
all-model coverage, not failure of this numerical job.
