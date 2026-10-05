# Corrected KSC comparison with all seven observation components — 2026-09-29

Configuration: FP64 TensorFlow GPU/XLA diagnostic comparison, TF32 off. The four particle routes retain their separately calibrated and validated N=1008 configurations. The new Gaussian-sum Kalman calculation is a deterministic reference with a checked quadrature resolution; it is not the production FP32/TF32 program or an HMC/default promotion. Exact particle tuning paths are preserved in the linked full tables.

**The main comparison now uses all seven KSC observation components.** The previous one-Gaussian Kalman column differentiated a different, moment-matched observation likelihood. Its numerical results remain historical heuristic evidence, but it was the wrong comparator to present as the central Kalman comparison for the full KSC mixture. That column and its headline comparison are superseded here.

The corrected reference applies a Kalman update for every observation component and retains the full Gaussian sum. Exact posterior branching grows as 7^T, so the long-horizon calculation uses fixed quadrature projection between updates. The final resolution has 1,201 state nodes and 8,407 Gaussian branches per later update; the first update has seven exact branches. All seven observation components are retained. This is a converged numerical approximation, not an exact seven-component posterior filter.

## Actual full-mixture reference results

These are means across the same eight saved datasets. The complete report gives every particle method beside these reference likelihoods and both score coordinates; the CSV files preserve individual datasets.

| T | Mean log likelihood | Mean gamma_raw score | Mean log_beta score |
| --- | ---: | ---: | ---: |
| 10 | -23.69426722 | -0.08041719 | 0.14076180 |
| 20 | -46.57656080 | 0.71034536 | -0.39021103 |
| 50 | -117.05459684 | -0.27051666 | 0.04123472 |
| 120 | -281.08122876 | 0.14771746 | -0.75825504 |

## Particle accuracy against the corrected reference

Entries are mean score-vector L2 error ± SE of that mean. Compute each dataset’s error before averaging; these are not norms of differences between the table’s mean scores.

| T | IID | Inverse CDF | Permutation .98 | Permutation .97 |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.21330 ± 0.04418 | 0.16295 ± 0.04293 | 0.16360 ± 0.04276 | 0.16504 ± 0.04311 |
| 20 | 0.35479 ± 0.08668 | 0.27694 ± 0.06177 | 0.27720 ± 0.06212 | 0.27645 ± 0.06307 |
| 50 | 0.46267 ± 0.08997 | 0.56875 ± 0.12536 | 0.57420 ± 0.12520 | 0.57172 ± 0.12599 |
| 120 | 1.54362 ± 0.40666 | 1.60816 ± 0.43979 | 1.58895 ± 0.43452 | 1.60149 ± 0.43352 |

All 128 original particle evaluations remain valid and are reused without changing observations, controls or random designs. Their error estimates change by at most 6.66e-15 because the original independent reference already integrated the full mixture. The correction changes the main comparator and the interpretation of the earlier Gaussian headline, not the particle findings.

The T=10 paired intervals comparing IID with each SQMC variant exclude zero in favor of SQMC. These are exploratory, unadjusted comparisons in one model regime. No interval establishes an ordering among the three SQMC variants, and every route-pair interval at T=20,50,120 includes zero. **Retain all four methods; no overall winner or default promotion is established.**

Sample SD and SE remain distinct: for inverse CDF at T=120, SD=1.243907 and SE=0.439788. They include both dataset and randomized-design variability. Fixed-dataset Monte Carlo uncertainty was not estimated. The full report also preserves signed coordinate means and their SEs, all absolute errors and paired intervals.

## Verification and terminal review

Eight CPU tests passed: exact seven-component enumeration at T=1,2 for three parameter points, plus finite differences on fine and deliberately coarse quadrature grids. The coarse case tests derivatives of projection normalization. The GPU call chain passed graph/XLA parity, both finite-difference coordinates and exact T=2 comparison. The GPU finite-difference discrepancy was 5.20e-11; graph and XLA likelihood/score outputs agreed exactly on the checked fixture.

Every saved dataset passed the four-level resolution/extent ladder. Against the independently implemented density-grid reference, the largest log-likelihood discrepancy was 5.12e-13 and the largest score-coordinate discrepancy was 9.77e-15. The largest refinement discrepancies were 5.68e-14 and 2.18e-14 respectively. Maximum projection-mass discrepancy across the ladder was 8.88e-16. These are observed numerical checks, not rigorous error bounds.

The reference source snapshot, GPU memory-growth/precision/device fields and original input hashes were verified. A separate arithmetic audit reconstructed all 128 particle errors from 320 actual coordinate rows and checked the 16 particle summaries and 24 paired intervals without importing the reporter’s statistics helpers. Peak TensorFlow allocator usage was 268,442,112 bytes. There were no invalid reference cases, infrastructure failures or retries.

Engineering correctness, numerical validity and scientific interpretation remain separate. The new reference passes its declared checks; finite particle scores still have the reported accuracy errors. One parameter regime, eight dataset/design pairs, N=1008 and a limited control search cannot establish broad superiority. Every original scope chose transport epsilon at the upper ladder endpoint; long-horizon tuning and particle convergence remain unresolved. The existing safeguard sensitivity evidence is preserved in the prior campaign, not reinterpreted as a non-harm certificate.

The strongest alternative explanation for particle differences is this restricted regime and numerical-control family. More independent pairs or fresh regimes could reverse the descriptive ordering. Independent reference formulations and short exact tests reduce the chance of a shared reference error; they do not prove finite-grid exactness for arbitrary data. Review was local, with executable checks; no independent agent was launched.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept corrected comparison | Full mixture retained; all datasets converge and agree independently | No reference validity/derivative veto | Empirical quadrature convergence | Use corrected tables | No exact finite-component long-horizon claim |
| Retain all four particle methods | Actual scores/errors and uncertainty preserved | Conditional heuristic promotion vetoes remain in the full report | Small replication, one regime and finite controls | Fresh long-horizon tuning and broader replication | No general winner or default readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Original particle validity and new reference checks pass; conditional heuristic losses still prevent promotion in those cases |
| Statistically supported ranking | Exploratory T10 SQMC-versus-IID intervals only; no within-SQMC or overall ranking |
| Descriptive-only differences | Other means, likelihood errors, timing and tails |
| Default readiness | Not established |
| Next evidence needed | Fresh long-horizon tuning, particle convergence, more pairs, fixed-dataset replications and broader regimes |

## Material changes and accounting

Added an independent analytical Gaussian-sum Kalman reference, exact-mixture/derivative tests, a bounded GPU comparison runner, and a corrected report assembler. The runner reuses original particle evidence, preserves all seven observation components and records quadrature convergence. The old one-Gaussian helper and archived results remain explicitly historical/explanatory; candidate execution and scientific defaults were not changed.

The correction charged 51.327502 GPU-owning wall seconds. Aggregate use is 6.795053/12 GPU hours, leaving 5.204947 hours, including all earlier use and the unchanged prior-hook reserve. The elapsed deadline remains 2026-09-30T16:15:40.010888+00:00. CPU-only tests, reporting and commit checks consume no GPU budget. No HMC, packages, push, publication or scientific/default promotion.

[Full actual-score, likelihood, error, uncertainty and convergence tables](../plans/artifacts/sqmc-ksc-full-mixture-20260929/final-evidence-01/report.md) · [Individual score coordinates](../plans/artifacts/sqmc-ksc-full-mixture-20260929/final-evidence-01/scores.csv) · [Individual likelihoods](../plans/artifacts/sqmc-ksc-full-mixture-20260929/final-evidence-01/values.csv) · [Plan and derivation](../plans/sqmc-ksc-full-mixture-correction-20260929.md).
