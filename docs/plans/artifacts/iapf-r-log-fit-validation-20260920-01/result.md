# Frozen R log-fit validation

Bounded stage complete.

The numerical core is frozen from the previous log-fit repair. Each worker executes captured sources. The reporting-only append change passed real-run numerical parity. The fitting objective is unweighted squared log-target error, different from paper Eq.15. All runs are independent R CPU references.

| Attempt | d / repeats | Data seed | Mean ratio to Kalman | Bootstrap95% | Screen | Heuristic veto |
|---|---:|---:|---:|---|---|---|
| attempt01-d20-a | 20 / 32 | 70000020 | 0.996404 | [0.976202, 1.016881] | pass (conditional_32_repeat_screen) | no_observed_veto |
| attempt02-d20-b | 20 / 32 | 71000020 | 1.009890 | [0.985510, 1.033173] | pass (conditional_32_repeat_screen) | no_observed_veto |
| attempt03-d40-pilot | 40 / 4 | 72000040 | 0.875362 | [0.711744, 1.020850] | pass (diagnostic_range_only) | no_observed_veto |
| attempt04-d80-pilot | 80 / 4 | 72000080 | incomplete | N/A | veto | not interpreted |
| attempt05-d40-follow-on | 40 / 16 | 73000040 | 1.004931 | [0.961774, 1.051352] | pass (diagnostic_range_only) | no_observed_veto |
| attempt06-d80-diagnostic | 80 / diagnostic | 72000080 | see scoped report | N/A | exact replay and oracle pass | no default promotion |
| attempt07-floor-diagnostic | 80 / diagnostic | 72000080 | see scoped report | N/A | three exact replays; floor intervention recorded | no default promotion |
| attempt08-positive-floor-repair | 80 / 1 | [70000020, 71000020, 72000080, 74000080] | incomplete | N/A | veto | not interpreted |
| attempt09-positive-floor-retry | 80 / diagnostic | [70000020, 71000020, 72000080, 74000080] | see scoped report | N/A | healthy parity, tail margins, d20 non-harm and d80 feasibility pass | no default promotion |

Intervals resample independent filter repetitions conditional on each fixed dataset (2,000 bootstrap resamples, seed=data_seed+900). A d20 screen requires its entire interval inside [0.9,1.1]. Fewer than32 repetitions provide descriptive feasibility evidence only. These intervals do not quantify variation across datasets or support a method ranking.

| Attempt / situation | iAPF log-prefix MSE | Fully adapted | BPF | SIS |
|---|---:|---:|---:|---:|
| attempt01-d20-a / ordinary | 0.00493654 | 0.0337159 | 1387.73 | 1.24901e+06 |
| attempt01-d20-a / large_innovation | 0.00386275 | 0.0221979 | 804.668 | 600049 |
| attempt02-d20-b / ordinary | 0.00582264 | 0.034752 | 1769.35 | 1.44214e+06 |
| attempt02-d20-b / large_innovation | 0.00908647 | 0.0284224 | 1250.34 | 886703 |
| attempt03-d40-pilot / ordinary | 0.0451943 | 0.111461 | 101477 | 6.05001e+06 |
| attempt03-d40-pilot / large_innovation | 0.0341854 | 0.0823549 | 83820.2 | 4.52416e+06 |
| attempt05-d40-follow-on / ordinary | 0.0229552 | 0.213872 | 106745 | 6.69373e+06 |
| attempt05-d40-follow-on / large_innovation | 0.0147465 | 0.183369 | 81698.1 | 5.17945e+06 |

The conditional heuristic comparison is a veto screen, not a statistical superiority claim. SIS likelihood ratios may underflow to zero; their finite log errors remain visible and zero sample variance of underflowed ratios is not accuracy evidence.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Investigate rejected scope | Dataset screens above | See rejected attempts | Dataset coverage and higher-dimensional replication | Complete longer high-dimensional validation before published-scale runs | Paper Eq.15 replication; TF/score/HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | ['attempt04-d80-pilot'] |
| Statistically supported ranking | None; fixed, unequal particle/computing budgets |
| Descriptive-only differences | Method means, SD, runtime, extreme ratios and conditional MSE |
| Default-readiness | Not established; optional independent reference |
| Next evidence needed | More repetitions/datasets, author-setting resolution, equal-cost comparison if ranking is sought |

Worker budget: 1474.773800/1800 seconds, 9 process launches (eight planned plus one bounded infrastructure retry); 325.226200 seconds remaining. Completed repetitions: 84; QR fits: 50400. Mechanics and focused checks are separate; see checkpoint.

Post-run red-team: easy datasets and linear-Gaussian structure may explain favorable evidence. A rejected fit or likelihood discrepancy on untouched data would overturn advancement in that scope. The weakest evidence is higher-dimensional replication and unknown author fitting/floor/controller settings. Engineering parity does not imply numerical accuracy; conditional Kalman agreement does not imply nonlinear, score, HMC, KDM or canonical LEDH validity.

Continuation: [floor diagnosis and positive-floor repair](floor-result.md) records the analytic cause, intervention results, safeguards, repair outcomes and remaining gaps. The original power2 d80 candidate remains rejected. A positive-floor repair is assessed separately; a logging/status infrastructure failure is not a candidate rejection.
