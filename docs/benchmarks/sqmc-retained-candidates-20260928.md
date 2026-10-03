# SQMC candidate disposition — 2026-09-28

The owner reviewed the completed expanded Kalman comparison and directed that
all four routes be retained for further testing: IID dual-cap, inverse CDF,
repaired permutation, and the permutation coordinate-cap 0.97 ablation.

These are scope-tuned FP64 GPU/XLA comparison variants (TF32 off), not production
FP32/TF32 results. Their exact tuning records, actual scores, absolute errors,
invalid calibrations and terminal review remain in
`../plans/artifacts/sqmc-expanded-20260928/renewal-01/final-evidence-02/`.
No default, production, HMC or statistical-superiority promotion follows.

All final configurations passed their numerical validity checks. Inverse CDF
had the smallest observed mean score-vector error in four of eight scopes;
the cap-0.97 permutation did in three; IID did in one. The likelihood-error
ordering differs. None of these descriptive counts justifies elimination.

There are only two independent final datasets and two shared filter designs
per scope. The post-run dataset-conditional standard error calculated in chat
is the sample SD of the two design-averaged dataset errors divided by sqrt(2).
It has one degree of freedom and omits variability from drawing new filter
designs. It is not a reliable total SE, confidence interval or ranking test.
Future comparisons should use independent dataset/design pairs and paired
method differences, with predeclared uncertainty analysis.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain all four candidates | Complete descriptive Kalman comparison | No final numerical invalidity; conditional losses to IID block promotion | Few datasets, shared designs, selected numerical settings | Audit and execute the owner-requested KSC SV comparison under a new target-specific plan | A winner, default readiness or rejection of any route |

The archived Kalman result and its checksums are unchanged. This addendum records
the new owner decision and authorizes no compute beyond the user's request and
the existing bounded budget.
