# SQMC master-program campaign summary — 2026-09-29

**The requested expanded Kalman and KSC comparisons are complete. Retain all
four methods; no overall winner or scientific/default promotion is established.**

Configuration: these are FP64 TensorFlow GPU/XLA diagnostic comparisons with
TF32 off, not the production FP32/TF32 program. Each particle route uses its
own scope-specific calibration, validation and frozen final controls. Exact
tuning paths and settings are recorded in the linked detailed reports and
manifests. These comparisons do not establish HMC readiness, control transfer
across scopes, or production performance.

## Completed scope and evidence

| Campaign | Completed evaluation | Reference and authoritative result |
| --- | --- | --- |
| Expanded Gaussian models | P44 d=3 at T=10,120, N=1008; full A/full SPD Q at d=3,10 and T=2,10,120, N=1020. All eight scopes and four routes completed: 128 final cells and 8,480 actual score coordinates with absolute errors. | Matched exact Kalman likelihoods and scores; [expanded result and terminal review links](sqmc-expanded-results-20260926.md). |
| KSC stochastic volatility | T=10,20,50,120, N=1008; eight shared dataset/design pairs per horizon and four routes. All 128 particle evaluations retained, with both gamma_raw and log_beta scores. | Full seven-component observation-mixture Gaussian-sum Kalman reference; [corrected result](sqmc-ksc-full-mixture-corrected-results-20260929.md) and [complete values, scores, errors and uncertainty](../plans/artifacts/sqmc-ksc-full-mixture-20260929/final-evidence-01/report.md). |

The four routes are IID, Hilbert inverse CDF, Hilbert permutation with coordinate
cap .98, and its cap .97 ablation. Comparisons concern the separately tuned
configurations, so differences cannot be attributed solely to ancestry ordering.
The Gaussian-model Q scores use lower-Cholesky coordinates, not raw symmetric
covariance entries.

The KSC correction replaces the prominently presented single moment-matched
Gaussian comparator, which computed a different observation likelihood. The
corrected calculation applies every one of the seven observation components.
Long-horizon posterior branching requires quadrature projection: the final
resolution uses 1,201 state nodes and 8,407 Gaussian branches after the first
update. This is a convergence-checked numerical reference, not an exact
seven-component posterior filter. The independent reference used for the
original particle errors already integrated the full mixture, so the corrected
particle errors change by at most 6.66e-15. Original evidence remains preserved.

## Findings and uncertainty

All final particle configurations passed the declared numerical validity
checks. In the expanded comparison, IID had smaller observed score error in
27 of 96 SQMC comparisons; these are dependent comparisons sharing datasets.
Two independent final datasets per scope do not support a statistical ranking.

For KSC, exploratory paired intervals favor each SQMC variant over IID at T=10.
They do not distinguish the three SQMC variants. Every route-pair interval at
T=20,50,120 includes zero. Inverse CDF is a promising candidate, not an
established overall winner. Error norms and absolute errors are calculated per
dataset before averaging; subtracting displayed mean scores can hide cancellation.
Reported SDs describe variability across dataset/design pairs; SEs describe
uncertainty in their means. Fixed-dataset Monte Carlo uncertainty was not measured.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close the requested campaign | All requested scopes, actual values, scores and errors preserved | Final validity checks pass; historical failed calibration remains recorded | Limited datasets, regimes and numerical-control search | Use the final reports; do not rerun completed research | Entire multi-model master program completed |
| Retain all four methods | No supported overall ranking | Conditional losses to simpler comparators veto promotion in those cases | Finite particle count and tuning scope | Plan fresh long-horizon tuning, particle convergence and broader replication | Inverse CDF or any other method is universally best |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Final numerical validity passes; recorded conditional heuristic losses still block promotion. |
| Statistically supported ranking | Exploratory KSC T=10 SQMC-versus-IID comparisons only; no within-SQMC or overall ordering. |
| Descriptive-only differences | Remaining means, error magnitudes, tails and timings. |
| Default readiness | Not established; scientific defaults are unchanged. |
| Next evidence needed | Fresh tuning, more independent pairs, fixed-dataset replications, particle convergence and broader regimes. |

## Material changes, verification and accounting

The expanded runner now checks complete data/design pairing, every score
coordinate, consistent primal values, provenance and model conditioning. Its
controller preserves versioned attempts and budget accounting. Invalid initial
full-d3 calibration candidates were retained and repaired through fresh
scope-specific tuning without relaxing validity guards.

The KSC work repaired the analytical score tangent and added the independent
full-mixture Gaussian-sum reference, exact short-horizon and finite-difference
checks, a bounded GPU runner, and corrected uncertainty reporting. The reference
passed eight CPU tests, GPU analytical/finite-difference/exact-enumeration checks,
and four quadrature resolutions/extents on all 32 saved datasets. A separate
arithmetic audit reconstructed every particle error and checked all summary and
paired-interval calculations. The correction had no invalid reference cases,
infrastructure failures or retries. Research commits are 479a4616, b7ed96ec
and c2ae4eb0; their complete logs and versioned evidence remain preserved.

The [final budget ledger](../plans/artifacts/sqmc-ksc-full-mixture-20260929/budget.json)
records 24,462.190031 charged GPU-owning seconds: 6.795053 of the aggregate
12 GPU hours, leaving 5.204947 hours. This includes earlier pilots, checks,
repairs and the unchanged conservative prior-hook reserve; linked prior ledgers
must not be added again. The elapsed deadline is
2026-09-30T16:15:40.010888+00:00. No research worker remains active. Documentation,
Git integration and explicitly CPU-only commit checks require no further GPU run.

Terminal review is local review plus executable checks, not independent external
review. The strongest alternative explanation for apparent method differences is
the limited dataset/regime and tuning coverage; further replication could reverse
the descriptive ordering. Numerical agreement between independent references
supports the reported comparisons, not arbitrary-data exactness or HMC validity.
The wider master program's other models and HMC objectives remain outside this
completed campaign. The owner has separately authorized merging this completed
work into main, synchronizing origin/main and updating sqmc-development from main.
