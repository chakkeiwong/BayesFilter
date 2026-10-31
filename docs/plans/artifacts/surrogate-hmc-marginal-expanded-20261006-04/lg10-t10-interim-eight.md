# Current-source LGSSM d=10, T=10 interim result

Scientific question: preserve or reduce likelihood and analytical-score errors while developing the components needed to test surrogate HMC. This is the predeclared eight-dataset look, not the final scope result.

Eight fresh datasets, 32 paired designs each, three parameter points, N=1024: all 1,536 candidate/baseline evaluations are valid. Exact FP64 Kalman values and reference scores supply the comparator. Controls were frozen on separate data before these datasets.

| Parameter | Absolute value error, ancestor → marginal | Value difference interval | Scaled score error, ancestor → marginal | Score difference interval |
|---|---|---|---|---|
| (1, 0) | 0.26426 → 0.16156 | [-0.24650, 0.04111] | 0.11997 → 0.05310 | [-0.16356, 0.02981] |
| (0.8, -0.5) | 0.32089 → 0.23687 | [-0.25297, 0.08492] | 0.11378 → 0.06537 | [-0.10731, 0.01050] |
| (1.2, 0.3) | 0.48640 → 0.15545 | [-0.57135, -0.09055] | 0.47312 → 0.16044 | [-0.55886, -0.06650] |

These are marginal-minus-ancestor error intervals with the predeclared family/look correction and reference allowance; the score scale is 0.1 times physical-score RMS. The two approximate sensitivity checks also pass wherever the reported criterion passes. This look supports non-inferiority for five of six cells, including all score cells. At (1.2,0.3), both intervals support lower error. At (0.8,-0.5), value non-inferiority is still inconclusive and triggers the already planned extension to sixteen datasets.

The simple Kalman filter retains lower observed likelihood error in every regime. That is a promotion veto for computational advantage over the simple-filter set; it does not invalidate the controlled weight-law comparison. Equal-runtime matching is incomplete. Neither this result nor particle-weight ESS establishes HMC convergence.

Saved diagnostics show median within-run minimum particle ESS/N of 0.0333→0.4191, 0.0457→0.8535 and 0.0131→0.3071 at the three points. These are descriptive mechanism checks only. Every marginal run at the second and third points used selective precision at least once; none did at the first. Saved extrema do not show the per-time frequency. The repair preserves a valid filter through cases the ordinary covariance symmetry guard would reject.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue the same master | Five of six non-inferiority cells pass; one remains inconclusive | Numerical validity passes; simple-filter advantage promotion veto remains | Eight independent datasets and approximation of interval coverage | Sixteen-dataset look, already executing | All-model superiority or HMC readiness |

| Inference status | Interpretation |
|---|---|
| Hard veto screen | No invalid paired filter or reference at this look |
| Statistically supported ranking | Lower value and score errors at (1.2,0.3), under the declared approximate interval procedure |
| Descriptive-only differences | Particle ESS, other point estimates, cap/repair frequencies and timing |
| Default-readiness | Not established; source04 is an optional component investigation |
| Next evidence needed | Finish the sixteen-dataset look and subsequent declared scopes; later downstream surrogate-HMC testing |

Post-run red-team note: the most favorable point may not generalize across horizons, dimensions or generating regimes. The remaining value interval can widen or move when eight more datasets arrive; retain this look rather than replacing it. Same-process finite-program correctness does not resolve the documented separate-process FP32/TF32 reproduction discrepancy.

Evidence: scopes/lg10-t10-n1024/look-8.json and diagnostic-reports/001/diagnostics.json. Live continuation: campaign.json and next-phase.json.
