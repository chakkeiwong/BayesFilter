# Fresh SIR diagnostic results

Status: budget_partial; 26/96 rows.
This is the current optional finite covariance-mixture consumer, not an admitted canonical route.
No SIR cSMC/KDM fitting or guidance is tested. Reference errors are conditional on a saved approximate Zhao–Cui solution.

## Data seed 26100831

Reference score: [-157.11039650174231, 104.57831048996057, 56.86070139879417]; between-fit SE: [1.0730237734705952, 1.6595666145057564, 0.05650986375427891].
Rank/support/approximation bias is unbounded; these SEs are not total oracle uncertainty.

| Arm | n | Mean score | Score MCSE | Coordinate RMSE | Mean L2 error |
|---|---:|---|---|---|---:|
| requested5 | 3 | -149.4996, 116.2436, 57.4181 | 33.9627, 21.4590, 5.2029 | 48.6297, 32.5124, 7.3792 | 58.1107 |
| baseline40 | 3 | -207.6012, 143.7611, 60.9464 | 34.3750, 16.8744, 4.9011 | 70.0900, 45.8779, 8.0458 | 72.8357 |
| prior-map | 2 | -153.9601, 130.4370, 61.5539 | 51.6671, 27.1042, 0.8734 | 51.7630, 37.4608, 4.7738 | 63.5203 |
| defensive-global | 2 | 341358.0337, -123621.3585, -202214.7724 | 234114.7482, 77402.0070, 58544.9557 | 414055.9249, 145942.3795, 210573.8003 | 424250.1895 |
| defensive-local | 2 | -163.8208, 133.9124, 62.4218 | 61.9684, 31.4779, 0.8415 | 62.3307, 43.0273, 5.6244 | 73.7282 |
| equal-mixture | 2 | 305.6151, -106.1744, -118.1476 | 1184.1277, 427.0659, 306.5196 | 1271.3274, 476.2373, 352.9620 | 1297.5260 |

Paired requested5 minus comparator L2 errors (negative favors requested5):

- baseline40: -14.7250; simultaneous interval None.
- prior-map: -1.2889; simultaneous interval None.
- defensive-global: -424187.9580; simultaneous interval None.
- defensive-local: -11.4967; simultaneous interval None.
- equal-mixture: -1235.2946; simultaneous interval None.

## Data seed 26100832

Reference score: [-135.19081394778084, 56.69628763246438, 5.8874743984982905]; between-fit SE: [1.554121718621559, 1.9114904388346012, 0.0015390605298644111].
Rank/support/approximation bias is unbounded; these SEs are not total oracle uncertainty.

| Arm | n | Mean score | Score MCSE | Coordinate RMSE | Mean L2 error |
|---|---:|---|---|---|---:|
| requested5 | 2 | -134.4702, 64.9024, 2.2867 | 70.2752, 0.0360, 1.4095 | 70.2789, 8.2062, 3.8668 | 70.8590 |
| baseline40 | 2 | -181.3115, 80.5482, 8.9347 | 71.3356, 0.4810, 5.9883 | 84.9464, 23.8568, 6.7191 | 77.6318 |
| prior-map | 2 | -150.7871, 71.8478, 4.4389 | 62.9511, 3.7193, 1.7911 | 64.8543, 15.6013, 2.3036 | 65.2092 |
| defensive-global | 2 | 5147.4348, -1867.3644, 4622.5740 | 9473.6560, 3724.5365, 3428.4133 | 10846.9485, 4192.1572, 5750.4619 | 11222.1779 |
| defensive-local | 2 | -156.2011, 76.3776, 3.4562 | 66.6122, 1.8567, 0.7849 | 69.8471, 19.7687, 2.5548 | 69.9671 |
| equal-mixture | 2 | -120.6111, 71.2966, -15.8893 | 36.5720, 1.6620, 0.2942 | 39.3710, 14.6946, 21.7787 | 45.7755 |

Paired requested5 minus comparator L2 errors (negative favors requested5):

- baseline40: -6.7727; simultaneous interval None.
- prior-map: 5.6498; simultaneous interval None.
- defensive-global: -11151.3188; simultaneous interval None.
- defensive-local: 0.8920; simultaneous interval None.
- equal-mixture: 25.0835; simultaneous interval None.

## Decision

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Diagnostic only; no promotion | False | hard veto=False; heuristic=not_checked | [3, 2] completed primary pairs by dataset; fixed approximate reference | Inspect conditional score errors before any SIR guide-fitting campaign | Exact-oracle, canonical, default or cSMC/KDM improvement |

## Inference status

| Question | Status |
|---|---|
| Hard veto screen | budget_partial; all completed rows valid |
| Statistically supported ranking | None: incomplete eight-pair design; no confidence intervals or ranking issued |
| Descriptive-only differences | Score means/MCSE, coordinate RMSE, likelihood and trace diagnostics |
| Default-readiness | Not evaluated; controls are untuned and route is noncanonical |
| Next evidence needed | SIR-specific fitted-guidance design, untouched validation and stronger reference/replication |

## Post-run red team

The strongest alternative explanation is approximation error in the fixed reference combined with seed noise. More replications and an independent, more accurate reference are needed to establish a conditional ranking. The small replication count limits interpretation, especially for extreme score errors. A failed current setting is not evidence against the future-guidance research direction.

Provenance and commands: `manifest.json`; fresh complete rows and traces: `rows.jsonl`; analytical/FD smoke: `smoke.json`; structured analysis: `analysis.json`.
