# Initialization isolation: results

Target-aware QR initialization repairs the exact-diagonal-Gaussian failure in the shared TensorFlow fitter. Across 30 targets and both objective scales, every CPU and GPU fit recovers the known center/covariance within 1e-9. Existing cloud-moment outputs reproduce the preceding saved trajectories within 1e-9. This is a concrete optional repair for the diagnosed initialization problem.

The fresh downstream screen does not support promotion. Five of 24 adaptive runs fail their declared caps, and 35 of the 38 completed fitted/heldout evaluations have a larger absolute log-likelihood error than at least one matched-count heuristic. These are single-draw conditional observations, not a statistically supported ranking. The optional QR procedure is viable for further study; it has not solved general guide fitting or filter accuracy.

## What changed and what is proved

The shared fitter now exposes cloud_moments/log_quadratic initialization and native/initial_peak density-objective units. Defaults are unchanged. QR fits log b_i=c+beta^T z_i+gamma^T z_i^2. For a diagonal Gaussian, mu_z=-beta/(2 gamma) and variance_z=-1/(2 gamma) recover its parameters when the design has full rank and negative quadratic coefficients. The representable fixtures have those properties.

The optional initial-peak scale subtracts a fixed log-density constant before exponentiation and multiplies objective and analytical gradient by the same constant. It leaves the mathematical minimizers unchanged while changing finite stopping/optimization. The scale is never recomputed as a function of the current parameters. Relative-shape optimization retains its separate identity.

Rank/concavity failures reject the strict QR candidate; there is no hidden moment fallback or ridge. Initialization validity, relative triangular-factor margin, clipping, initial shape error and fixed log scale are retained through the actual recursive and adaptive consumer. The whole iAPF configuration is already part of tuning identity, and accounting now checks the two reported controls against it. The public consumer wiring and mismatched-report rejection are executable regression checks.

## Fresh GPU consumer screen

Model: the existing six-parameter linear-Gaussian fixture with non-diagonal observation map/covariance, d=o in {2,5,10}, T4, initial N128, cap1024, seeds81/82. The paper-study model is different. Guides are learned on the fitted observations; reuse on independently generated heldout observations is explicitly a transfer diagnostic. Each heuristic uses the arm's realized final particle count. The comparisons are exact Kalman, ordinary BPF, the same consumer with a constant guide, and the one-step optimal Gaussian proposal.

| d | seed | initialization | objective units | outcome |
|---|---|---|---|---|
| 2 | 81 | cloud_moments | native | complete, N=128 |
| 2 | 81 | cloud_moments | initial_peak | complete, N=128 |
| 2 | 81 | log_quadratic | native | complete, N=256 |
| 2 | 81 | log_quadratic | initial_peak | complete, N=256 |
| 2 | 82 | cloud_moments | native | complete, N=128 |
| 2 | 82 | cloud_moments | initial_peak | complete, N=128 |
| 2 | 82 | log_quadratic | native | complete, N=256 |
| 2 | 82 | log_quadratic | initial_peak | complete, N=256 |
| 5 | 81 | cloud_moments | native | complete, N=256 |
| 5 | 81 | cloud_moments | initial_peak | complete, N=256 |
| 5 | 81 | log_quadratic | native | complete, N=256 |
| 5 | 81 | log_quadratic | initial_peak | complete, N=256 |
| 5 | 82 | cloud_moments | native | complete, N=512 |
| 5 | 82 | cloud_moments | initial_peak | complete, N=512 |
| 5 | 82 | log_quadratic | native | complete, N=256 |
| 5 | 82 | log_quadratic | initial_peak | iAPF density fit invalid or unconverged at iteration 4 |
| 10 | 81 | cloud_moments | native | iAPF particle cap exhausted at iteration 11 |
| 10 | 81 | cloud_moments | initial_peak | iAPF iteration cap exhausted before convergence |
| 10 | 81 | log_quadratic | native | iAPF iteration cap exhausted before convergence |
| 10 | 81 | log_quadratic | initial_peak | iAPF particle cap exhausted at iteration 11 |
| 10 | 82 | cloud_moments | native | complete, N=512 |
| 10 | 82 | cloud_moments | initial_peak | complete, N=512 |
| 10 | 82 | log_quadratic | native | complete, N=128 |
| 10 | 82 | log_quadratic | initial_peak | complete, N=256 |

The d5/seed82/QR+peak failure is a finite iteration-budget failure, not invalid rank or nonconcavity: the fit remains valid, one time step reaches 2000 steps with projected gradient 3.39e-7 versus 1e-7. Its shape residual is 6.18e-5. Other steps can reduce density loss while worsening shape (one rises from .568 to .877), so extending this one optimizer cap would not establish good guides. All four d10/seed81 arms hit particle or adaptive-iteration caps. These candidate rejections leave the scientific direction open.

Four prespecified derivative checks are available for successful seed81 QR+peak runs (d2 and d5, fitted and heldout). All retain ancestor and mixture labels at the selected centered-difference pair and agree within 1.15e-8, below the declared 2e-5 threshold. The d10 derivative checks are unavailable because fitting/adaptation failed; they are not silently counted as passes. The score is the derivative of a fixed-guide finite program, not the marginal-model score.

## Decision and inference

| decision | primary criterion | veto status | uncertainty | next action | limits |
|---|---|---|---|---|---|
| Retain explicit optional initialization/scale controls | Exact-target recovery and saved-default parity pass CPU/GPU | Known-good non-harm and rank/concavity guards pass | Non-diagonal/non-Gaussian targets | Examine floor strength and guide-family error in actual consumers | No default or paper-faithfulness promotion |
| Reject immediate downstream promotion | Fresh screen not passed | Five caps; conditional heuristic losses | Two seeds, different realized N and optimizer basins | Diagnose finite-floor mixture and preserve failures | No supported ranking or rejection of iAPF as a method |

| inference status | finding |
|---|---|
| Hard veto screen | Declared fitting/adaptation caps fail in 5/24 cells; failed derivative/rank/source arithmetic not observed in accepted runs |
| Statistically supported ranking | None |
| Descriptive-only differences | Conditional errors, runtimes, iterations, realized counts and heuristic losses |
| Default readiness | Defaults unchanged; optional controls only |
| Next evidence needed | Mechanism diagnosis followed by disjoint calibration/validation and uncertainty-aware filter comparisons |

Terminal skeptical review: exact diagonal recovery is mathematically informative but favorable to QR by construction. Actual backward targets include cross-coordinate dependence and a positive floor; they need not lie in that family. Larger particle counts or more optimizer iterations may rescue individual caps but cannot prove guide quality. A different floor may restore useful twisting or worsen importance weights; measure its mixture probabilities and effects before claiming a cause. All earlier strict TF32 vetoes and paper-model/author-code gaps remain.

Evidence: launch manifests bind source snapshots, commands, environment and times; Stage A stores all pointwise fits; Stage B stores complete accepted/rejected diagnostics and actual final random inputs for completed cases. Tests and verification are recorded separately.
