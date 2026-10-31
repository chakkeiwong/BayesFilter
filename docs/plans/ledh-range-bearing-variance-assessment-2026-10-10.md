# Range-bearing comparison: variability and particle-count assessment

The new covariance-guided defensive-mixture filter has much smaller observed
score errors than the tuned previous method at the difficult parameter point.
The first two datasets have negative nominal paired error intervals, including
the declared reference-sensitivity expansion. The third has no supported
ranking because the previous method has an extreme, variable error tail.
Truth-point rankings remain unresolved. This is the existing T20, N1008,
FP64 diagnostic, not a surrogate-HMC posterior result.

## What was measured

The saved eight particle designs were inspected without rerunning or retuning
either filter. Score error is the Euclidean distance to the N65536
bootstrap/Fisher reference. SD below is across those eight designs, conditional
on each dataset and parameter point; it is not HMC-chain MCSE.

| Dataset, difficult point | Previous mean error (SD) | New mean error (SD) | Previous score RMS spread | New score RMS spread | New log-likelihood SD |
|---|---:|---:|---:|---:|---:|
| 1704 | 26.43 (15.01) | 5.58 (4.95) | 19.64 | 7.73 | 1.27 |
| 18020 | 22.54 (14.58) | 1.40 (0.94) | 21.18 | 1.39 | 0.55 |
| 18021 | 133.58 (169.29) | 4.44 (2.23) | 214.12 | 5.18 | 1.48 |

The new mean errors at the truth point are 1.27, 0.98, and 1.64, versus 1.25,
1.69, and 3.19 previously. Those differences do not establish a ranking.
For the new method the truth-point score RMS spread is 1.40, 1.17, and 1.97;
log-likelihood SD is 0.45, 0.36, and 0.73.

For score vectors g_i and reference g_ref, let g_bar be the eight-design mean
and S their sample covariance. The directly checked decomposition is

    mean_i ||g_i - g_ref||^2
      = ||g_bar - g_ref||^2 + (7/8) trace(S).

The new difficult-point centroid-reference discrepancies are 0.55, 1.02,
and 0.75, compared with RMS spreads 7.73, 1.39, and 5.18. Thus design variability
accounts for most observed squared error in datasets 1704 and 18021. This does
not prove low bias: the centroid itself has substantial eight-design uncertainty,
and the reference retains particle bias. The new score's maximum observed error
is 16.04 in the first difficult dataset. Minimum within-filter particle ESS there
is 42.49/1008, about 4.2%, an explanatory concentration diagnostic.

## Remaining weaknesses and next discriminating evidence

1. Residual design variability is substantial in two difficult datasets.
   N1008 has not been shown adequate for HMC or posterior inference.
2. The simple prior proposal has lower observed mean score error in four of six
   cells, including two difficult ones. These means are descriptive; nevertheless
   the declared broad-promotion screen remains unsatisfied. The more elaborate
   flow has not established uniform value beyond simpler proposals.
3. This is FP64/TF32-off evidence. Repaired default FP32/TF32 requires its own
   tuning and untouched evaluation. Optional new moment correction was disabled
   (correction_steps=0); the result does not validate that enhancement.
4. Coverage is three datasets, two parameter points, and T20. Independent
   reference bias is not bounded. No HMC chains or posterior-agreement checks
   were performed, and this scope does not establish all-model improvement.

A useful next extension is a scope-specific particle-count ladder
N1008 -> N2016 -> N4032, retaining the simple prior and bootstrap comparators.
Each count needs fresh scope-matched tuning on disjoint partitions; historical
controls can only be warm starts. Default precision should be checked first;
precision and N effects must be distinguished. The chunk-policy extents would
be K1008, K2016, K2016 respectively. This is a proposed extension beyond the
completed fixed-N plan, not an unfinished phase of that plan, and has not been
launched or assigned a new compute budget.

Under the explicit hypothesis variance is proportional to 1/N, doubling N
reduces SD to about 71%, and quadrupling N to about 50%. No ladder has verified
that hypothesis for these filters. More particles may not remove a persistent
approximation discrepancy; compare both spread and centroid-reference error.
More independent designs improve uncertainty in the comparison, not the
accuracy of an individual fixed-design filter. In HMC, a frozen particle design
defines a deterministic approximate target: this measured variability is across
possible designs, not fresh stochastic force noise at each transition.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve scoped comparison | Six complete paired cells | New/previous finite and valid 48/48 | Small-sample intervals and reference bias | Retain conditional evidence | Uniform superiority |
| N1008 sufficiency unestablished | No posterior materiality test | Broad-promotion screen unsatisfied | Design spread and approximation error | Proposed tuned particle-count ladder | Required final N |
| Default/HMC promotion withheld | No default-precision posterior evidence | Prior losses and incomplete guide screen | Precision, coverage, reference bias | Separate fresh default-precision and downstream HMC checks | All-model or posterior correctness |

| Inference status | Assessment |
|---|---|
| Hard veto screen | New/previous 48/48 valid; global guide invalid 5/48 |
| Statistically supported ranking | Exploratory conditional evidence in two difficult cells; sequential/multiple-look and reference assumptions remain |
| Descriptive differences | SDs, centroids, likelihood spread, ESS, remaining means |
| Default readiness | Not established by the FP64 diagnostic |
| Next evidence needed | Scope-tuned N ladder and default precision; actual HMC/posterior diagnostics for inference claims |

Post-run red team: eight designs may understate tail variability, and a small
centroid discrepancy can arise by chance. A refined reference or fresh design
replication showing a persistent shift would weaken the variance-dominance
explanation. Improved mean errors may largely reflect eliminating previous
outliers rather than benefits of the local flow beyond a simple prior proposal.

Evidence: artifacts/ledh-range-bearing-covariance-20261009-07/variance-diagnostics.json.
Reproducible diagnostic: the same directory's variance-inspection.py; Python
standard library only, no TensorFlow import or GPU use, no numerical campaign
budget expansion. Source hashes and the error-decomposition assertion are saved.
