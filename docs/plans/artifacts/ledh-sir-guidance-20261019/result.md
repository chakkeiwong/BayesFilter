# SIR guidance comparison — 19 October 2026

The equal-mixture guided candidates are worse than unguided on both fixed
SIR datasets. All four prespecified paired bootstrap intervals lie above zero
for the change in squared score error: conditional evidence of harm, not gain.
Cheap prior/local proposals have much smaller observed errors. The global
comparator has one genuine covariance-reset failure.

All 96 keys were evaluated; 95 are numerically valid. The failed global row
is preserved, not imputed or omitted for survivor ranking. All primary
guidance contrasts retain eight valid pairs on each dataset.

| Arm | Data 26100831 score RMSE | Data 26100832 score RMSE |
|---|---:|---:|
| Unguided, equal mixture | 1334.5664 | 123.2970 |
| cSMC quadratic, equal mixture | 8524.2062 | 1730.4322 |
| KDM-inspired kernel, equal mixture | 58237.4284 | 606.0270 |
| Physical prior | 116.0418 | 86.6344 |
| Defensive local | 127.6919 | 80.9164 |
| Defensive global | 1445125.0155 | Invalid: one failed seed |

RMSE is the square root of mean squared L2 score error against the saved
approximate Zhao–Cui reference. These eight-seed summaries do not support
statistical rankings among the simple methods. The reference is not exact
and its uncertainty is not propagated.

| Guided minus unguided squared error | Data | 98.75% paired interval |
|---|---:|---:|
| Quadratic | 26100831 | [2002904.85, 163655370.48] |
| Kernel | 26100831 | [831450.43, 13551941142.60] |
| Quadratic | 26100832 | [74357.41, 9816549.04] |
| Kernel | 26100832 | [109115.89, 577104.77] |

## Decision

| Field | Finding |
|---|---|
| Decision | Reject equal-mixture guidance candidates for promotion |
| Primary criterion | Four conditional comparisons show harm, not gain |
| Veto diagnostics | Global reset fails at time 4; every complex arm fails component-wise heuristic dominance |
| Main uncertainty | Eight seeds, one guide fit per dataset, heavy tails and approximate reference |
| Next justified action | Independent beta calibration and matched-beta controls on fresh seeds |
| Not concluded | Rejection of cSMC/KDM as a direction, canonical LEDH accuracy, or original neural KDM performance |

| Inference status | Finding |
|---|---|
| Hard veto screen | 95 valid; global/data832/seed261019405 invalid and independently reproduced |
| Statistically supported ranking | Approximate bootstrap supports conditional guided harm in four prespecified contrasts |
| Descriptive-only differences | RMSE, component errors, timing and prior/local rankings |
| Default-readiness | Rejected; this consumer is not canonical LEDH |
| Next evidence needed | Allocation repair, independent fits/data and canonical consumer evidence |

The result rejects the candidate allocations, not the data, analytical
derivative implementation or research direction. Both guided arms complete
their trajectories and pass finite-program derivative checks. A correct
derivative of a poor finite likelihood estimate can be inaccurate as a model
score. The global-map errors make allocation a plausible failure mechanism;
independent beta fitting tests it without calibrating against reference scores.

The strongest alternative explanation is distorted message approximation:
39/49 and 37/49 quadratic information matrices needed projection, while
kernels were active at only 10 and 9 times. Allocation may not rescue bad
guidance. Independent fits or a canonical consumer could overturn these
conditional conclusions. No OT sweep follows from this result.

Evidence: attempt-04/{manifest.json,rows.jsonl,analysis.json,result.md};
reset-diagnostic-01/diagnostic.json; earlier attempts retained. First-stage
GPU charge: 4,350.290199399 seconds including the 50-second preflight reserve.
Remaining owner ceiling: 38,849.709800601 seconds. Activate the 9,000-second
beta stage under ../../ledh-sir-beta-repair-20261019.md.
