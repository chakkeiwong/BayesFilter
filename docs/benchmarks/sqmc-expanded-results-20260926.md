# Expanded SQMC Kalman comparison — complete

All eight requested scopes and four routes are complete: **128 valid final cells and 8,480 score coordinates**, each with the actual matched Kalman score, particle score and absolute error. P44 uses N=1008; the full A/full SPD Q models use N=1020. The terminal executable audit found no missing cells or engineering consistency failures.

**IID has a smaller observed score error in 27 of the 96 SQMC route/data/design comparisons.** These conditional losses veto promotion in those situations. All final candidates pass the numerical validity screen and remain available for diagnosis, but the comparison does not support a general SQMC advantage. The 96 comparisons share datasets and are not independent trials.

[Detailed comparison and paired plots](../plans/artifacts/sqmc-expanded-20260928/renewal-01/final-evidence-02/report/report.md) · [Every actual score and absolute error](../plans/artifacts/sqmc-expanded-20260928/renewal-01/final-evidence-02/report/scores.csv) · [Actual likelihoods and errors](../plans/artifacts/sqmc-expanded-20260928/renewal-01/final-evidence-02/report/likelihoods.csv) · [Terminal review](../plans/artifacts/sqmc-expanded-20260928/renewal-01/final-evidence-02/terminal-review.md)

## Descriptive Kalman comparisons

Each entry below is the mean score-vector L2 error over two final datasets crossed with two filter designs. There is no confidence interval or statistically supported ranking. Coordinate scales differ across models, so these magnitudes should be compared within a row. The detailed tables preserve all individual coordinates and pairs.

| Scope | IID | Inverse CDF | Permutation | Permutation cap .97 |
|---|---:|---:|---:|---:|
| P44 d=3, T=10 | 0.0772176 | 0.00525146 | 0.00888736 | 0.00847798 |
| P44 d=3, T=120 | 0.168087 | 0.0126038 | 0.0155137 | 0.0101323 |
| Full d=3, T=2 | 0.427525 | 0.323399 | 0.308004 | 0.307794 |
| Full d=3, T=10 | 0.457723 | 0.330518 | 0.302479 | 0.293541 |
| Full d=3, T=120 | 1.60621 | 0.782603 | 0.851894 | 0.839536 |
| Full d=10, T=2 | 4.61754 | 4.36268 | 5.41456 | 5.40664 |
| Full d=10, T=10 | 4.81459 | 4.819 | 6.04861 | 5.96964 |
| Full d=10, T=120 | 10.4493 | 8.4198 | 10.3125 | 9.41929 |

Mean absolute log-likelihood errors for the same final cells are:

| Scope | IID | Inverse CDF | Permutation | Permutation cap .97 |
|---|---:|---:|---:|---:|
| P44 d=3, T=10 | 0.0251458 | 0.00137358 | 0.000730114 | 0.000581389 |
| P44 d=3, T=120 | 0.0512718 | 0.00303762 | 0.00123169 | 0.000889246 |
| Full d=3, T=2 | 0.0419645 | 0.0265645 | 0.0260339 | 0.0260709 |
| Full d=3, T=10 | 0.0548891 | 0.023328 | 0.0192543 | 0.0238048 |
| Full d=3, T=120 | 0.235139 | 0.0525003 | 0.0825038 | 0.0729839 |
| Full d=10, T=2 | 0.174825 | 0.136288 | 0.2321 | 0.23431 |
| Full d=10, T=10 | 0.253434 | 0.176291 | 0.306361 | 0.270063 |
| Full d=10, T=120 | 0.931343 | 0.334121 | 0.261992 | 0.522648 |

All SQMC final cells have smaller observed score error than the zero-score and first-observation-only heuristics. Matched IID is a stronger adversary: observed SQMC losses occur in full d=3 at T=2 and T=10, and in full d=10 at every tested horizon. These are promotion vetoes, not infrastructure failures or grounds to reject the entire SQMC research direction. P44 and full d=3 at T=120 have no observed SQMC loss to IID in these particular final cells.

The main permutation route uses coordinate cap .98; the ablation uses .97. All P44 and full-d3 routes selected eight flow steps. Full-d10 routes selected two steps except inverse CDF at T=120, which selected eight. P44 retained epsilon .4; full-d3 IID selected 25.6 and its SQMC routes selected 102.4; all full-d10 routes selected 25.6. Thus full-d3 route comparisons also change smoothing, and the full-d10 T=120 inverse-CDF comparison also changes flow resolution. They compare separately tuned configurations and cannot isolate ancestry ordering.

## Material changes

The runner now evaluates the complete 2×2 Cartesian product of final data and filter seeds, compares the first-observation-only heuristic with the full-horizon Kalman score, verifies coordinate counts and consistent primal values across analytical directions, and records observation/random-design hashes, numerical source dependencies and model conditioning. Oracle failures and resource exceptions stop the run; invalid particle candidates are labeled separately. Completed units can be reused after reporting repairs without repeating numerical work. The controller uses versioned outputs, worker time limits, interrupted-worker cleanup, aggregate compute accounting and the existing retry limit.

The original full-d3 calibration failed the transport mass guards. A replay with identical observations and random designs reproduced the failure on CPU and GPU. Increasing the balance count through 192 did not meet the unchanged mass criterion. In the shared reset, Sinkhorn and balance counts enter the same update loop, and additive normalization denominators can leave a residual. This explains why additional iterations need not remove the residual; it does not prove that mechanism explains every possible invalid candidate.

A bounded ladder of the existing smoothing control was added for each full-model scope and route: epsilon 0.4, 1.6, 6.4, 25.6 and 102.4. Calibration selects the smallest candidate passing full-horizon validity, column discrepancy at most 1e-6 and row discrepancy at most 1e-8 on both fresh calibration datasets. All-coordinate flow-step tuning and separate validation follow, and controls are frozen before final evaluation. The original validity guards were not relaxed. Epsilon changes the finite transport; the shared numerical kernel and analytical recursion were unchanged during this repair.

The diagnostic epsilon became a scalar graph input to avoid repeated compilation. The bounded GPU check verified agreement with the static diagnostic and one trace across the ladder. A first cost projection did not fit the remaining budget and therefore did not authorize a full launch; the revised implementation passed parity and the conservative affordability check.

The report now checks every exported coordinate and error against the saved worker result. Five deliberate corruptions—estimated score, exact score, error, missing coordinate and duplicate coordinate—were rejected. A report-only CSV field-list bug found during this check was repaired and the audit rerun. The numerical campaign was unaffected.

## Validation and uncertainty

The initial focused CPU regression passed 47 tests. Two targeted repair runs passed 9 and 7 tests; these counts overlap. Checks cover all-coordinate model-callback finite differences, small-fixture finite-program directional checks, matched Kalman timing, shared-evaluator wiring, seed separation and failure classification. Kalman finite differences cover all 17 coordinates at d=3 and 11 selected coordinates at d=10. Sixteen GPU graph/XLA parity cases passed; the maximum absolute score difference was 2.8422e-14. Full 157-coordinate finite differences at N=1020 and T=120 were not performed.

The exact oracle differentiates the marginal likelihood of the matched predict-first Gaussian state-space model. The particle score uses analytical recursion for the finite particle program. Finite-program checks support implementation consistency; they do not establish equality with the exact Kalman score. Finite particle count, flow integration, transport smoothing and piecewise ancestry/order choices remain relevant to the observed errors.

These are FP64 GPU/XLA reference comparisons with TF32 disabled and verified memory growth on an RTX 4080 SUPER. They do not assess FP32/TF32 production accuracy. The models use full observations, stable transitions, moderately conditioned covariance matrices and one fixed generating parameter per model. Near-singular covariance, high persistence, nonlinear targets, partial observations, MLE neighborhoods and HMC were not evaluated.

P44 retains the original partitions, including two previously inspected pilot pairs. Full-model repair uses calibration seeds 199001/199002, validation 200001 and final data 201001/201002 crossed with filter designs 202001/202002. The generic manifest sentence about reused pilot pairs was inherited by repair units; their saved partitions and row labels correctly identify fresh data. The sentence applies only to P44, and the original manifests are preserved.

Each scope has only two independent final datasets. Filter repetitions do not create additional independent datasets. Means, individual errors and conditional loss counts are descriptive. The full-covariance scores use lower-Cholesky Q coordinates, with log diagonal and unconstrained off-diagonal entries; they are not gradients with respect to raw symmetric Q entries.

## Failed attempts and budget

[Original calibration outputs](../plans/artifacts/sqmc-expanded-20260928/renewal-01/original-calibration-rejections-01/README.md) preserve 36 evaluations and 612 returned coordinates, including 21 invalid evaluations. Invalid zero returns are guard sentinels rather than actual gradients, and their errors are unavailable. The original run stopped deliberately for repair. Repair-check 01 had mismatched regenerated inputs and used one infrastructure retry. Checks 02 and 03 were numerical candidate rejections. Check 04 passed validity but failed the affordability condition; check 05 passed both. All attempts and complete logs remain preserved.

Total charged GPU-process wall time is **21,377.330 seconds (5.938 hours)** of 12 hours, leaving **21,822.670 seconds (6.062 hours)**. This includes a conservative 1,918-second charge for earlier expanded pilots: their failed-process end times were unavailable, so the entire first-start-to-last-finish interval was charged. It is not a measurement of device-active utilization.

The renewed window began 2026-09-27 17:25:28.470267 UTC. Numerical execution finished at 23:48:25.469708 UTC, 6.3825 hours later, before the 14-hour deadline of 2026-09-28 07:25:28.470267 UTC. No unit exceeded two infrastructure retries. The final repair campaign needed no infrastructure retry. The [budget ledger](../plans/artifacts/sqmc-expanded-20260928/renewal-01/budget.json) preserves exact commands, logs, timestamps and charges.

## Decision and terminal review

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete the requested descriptive comparison | All 32 units, 128 final cells and 8,480 coordinates recorded and audited | No final numerical invalidity or provenance failure; 27 conditional losses to IID | Two independent datasets; finite resolution and separately tuned controls | Preserve results; investigate the specific full-model losses before any new campaign | General superiority, default readiness, HMC or production admission |

| Inference status | Finding |
|---|---|
| Hard veto screen | All final configurations pass numerical validity; original rejected calibrations remain rejected; conditional IID losses veto promotion in those situations |
| Viable candidates | All four final routes in each scope remain numerically viable for further investigation |
| Statistically supported ranking | None |
| Descriptive-only differences | Every likelihood, score error, runtime, mean and conditional loss count |
| Default readiness | Not established |
| Next evidence needed | Predeclared multi-dataset replication with paired uncertainty analysis and target-specific numerical calibration; no further run authorized by this result note |

The strongest alternative explanations for apparent route differences are the particular datasets and random designs, selected smoothing/flow resolution, and finite-program approximation. Independent replication could reverse the descriptive ordering; errors that persist under numerical refinement would instead point to a more durable limitation. The weakest evidence is the small number of independent datasets and narrow model regime.

The [terminal review](../plans/artifacts/sqmc-expanded-20260928/renewal-01/final-evidence-02/terminal-review.md) checks completeness, worker-to-report equality, source/data/design/tuning hashes, coordinate identity, exact-oracle alignment, GPU/XLA provenance, invalidity labels, conditional heuristics and interpretation. It is a Codex review with executable checks, not an independent external review. The recorded decision is to accept the completed descriptive comparison and withhold method/default promotion.

No package or environment was changed, and no merge, push, publication or HMC run was performed. Existing unrelated work and all earlier evidence were preserved.
