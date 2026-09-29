# SQMC master-program campaign summary — 2026-09-29

**The requested expanded Kalman and KSC comparisons are complete. Retain all
four methods; no overall winner or scientific/default promotion is established.**

Configuration: these are FP64 TensorFlow GPU/XLA diagnostic comparisons with
TF32 off, not the production FP32/TF32 program. Each particle route in the original comparisons used its
own scope-specific calibration, validation and frozen final controls. The later
discrepancy investigation froze those controls across changed scopes and labels
all new cells UNTUNED diagnostics; it grants no tuning admission. Exact
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
uncertainty in their means. Fixed-dataset Monte Carlo uncertainty was not measured in that original
comparison. The completed follow-up below now reports it separately.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close the requested campaign | All requested scopes, actual values, scores and errors preserved | Final validity checks pass; historical failed calibration remains recorded | Limited datasets, regimes and numerical-control search | Use the final reports; do not rerun completed research | Entire multi-model master program completed |
| Retain all four methods | No supported overall ranking | Conditional losses to simpler comparators veto promotion in those cases | Finite particle count and tuning scope | Test reset distribution preservation, then fresh tuning and broader replication | Inverse CDF or any other method is universally best |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | Final numerical validity passes; recorded conditional heuristic losses still block promotion. |
| Statistically supported ranking | Exploratory KSC T=10 SQMC-versus-IID comparisons only; no within-SQMC or overall ordering. |
| Descriptive-only differences | Remaining means, error magnitudes, tails and timings. |
| Default readiness | Not established; scientific defaults are unchanged. |
| Next evidence needed | Reset/design repair evidence, fresh tuning and untouched broader multi-dataset comparisons; the limited fixed-data and particle ladders are now complete. |

## Completed discrepancy follow-up

The [reviewed analysis and full results](sqmc-ksc-discrepancy-results-20260929.md)
separate finite-program derivative correctness from full-mixture score accuracy.
Two retrospective T=120 cases, eight random designs and N=1,008/2,016/4,032 gave
192 valid particle replications. All 48 branch-matched derivative checks and
144 numerical-control cells passed; the large conditional mean score error
persisted on dataset 213006. At N=4,032 its mean gamma error remained 1.56–1.84,
with coordinate SEs 0.14–0.16. This does not establish asymptotic bias or rank
methods. All four also lost to the Gaussian Kalman heuristic on this case.

Eight canonical traces identified a shared local distortion: reset plus
higher-moment correction preserved mean and variance but reduced average
kurtosis from 2.77–2.82 to 1.32–1.34. Exact seven-component integration on the
same before/after clouds showed changes in the next predictive likelihood and
score. This is a measured local reset effect, not a decomposition of the whole
120-step error or proof that reset is the only cause. The next discriminating
study should test residual-design richness and distribution preservation before
simply increasing particles or solver iterations again.

All 376 saved numerical evaluations were valid, all 21 GPU worker attempts
completed without retries, eight focused CPU tests passed, and the terminal
evidence audit passed 20 checks. Full values, scores, absolute errors, SD/SE,
covariance matrices, exploratory intervals and plots are linked from the result.
No filtering runtime, model, safeguard, package/environment or default changed.
All four methods remain research candidates; none receives accuracy/default
promotion from these diagnostics.

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

The [current budget ledger](../plans/artifacts/sqmc-ksc-discrepancy-20260929/budget.json)
records 31543.211557 charged GPU-owning seconds: 8.762003 of the
aggregate 12 GPU hours, leaving 3.237997 hours. This includes the immutable
prior total of 24,462.190031 seconds and 7081.021525 seconds for the
follow-up, including worker initialization and compilation. The old reserve is
already included; do not add predecessor ledgers again. The elapsed deadline
remains 2026-09-30T16:15:40.010888+00:00. No research worker remains active.
CPU-only tests and reporting intentionally hid GPUs and used no GPU budget.

Terminal review is local review plus executable checks, not independent external
review. The strongest alternative explanation for apparent method differences is
the limited dataset/regime and tuning coverage; further replication could reverse
the descriptive ordering. Numerical agreement between independent references
supports the reported comparisons, not arbitrary-data exactness or HMC validity.
The wider master program's other models and HMC objectives remain outside this
completed campaign. The original comparison was integrated with main and origin/main before this
follow-up began at commit 023e1061. The follow-up is recorded on
sqmc-development; it does not repeat that merge/push operation.
