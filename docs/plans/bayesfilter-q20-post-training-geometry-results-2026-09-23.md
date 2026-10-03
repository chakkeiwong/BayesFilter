# Saved NeuTra geometry results

The diagnostic ran successfully on the original map and all four continued
maps. All 32 target rows per map were valid and all scores were finite. The
continued maps have smaller observed Gaussian-score residuals on this bank,
but substantial residuals and the original scale saturation remain. No map
has yet demonstrated adequate training for stable HMC estimation.

The controlling [plan](bayesfilter-q20-post-training-geometry-2026-09-23.md)
specified one common 32-point standard-normal bank. The quantity is
`r(z) = grad_z[log pi_beta(T(z)) + log|det J_T(z)|] + z`, including the full
log-Jacobian gradient, at beta one. RMS below is
`sqrt(mean over points and coordinates of r(z)^2)`. It would be zero for an
exact standard-normal transformed target. Standard normal coordinates have
unit population RMS, so these residuals are not close to that exact identity.
No finite cutoff was used to declare HMC success or failure.

| Map | Lifetime updates | Score residual RMS | Maximum row residual norm | Centered log-density residual RMS |
| --- | ---: | ---: | ---: | ---: |
| Shared starting map | 1,024 | 6.0245 | 26.9030 | 4.5284 |
| Continued cap-10 control | 2,048 | 4.4592 | 19.5659 | 2.5592 |
| Repaired clipping, depth 2 | 2,048 | 3.7195 | 15.3875 | 2.3691 |
| Repaired clipping, depth 4 | 2,048 | 2.2147 | 7.3211 | 1.9840 |
| Repaired clipping, batch 128 | 1,280 | 4.4858 | 22.9022 | 3.5628 |

These are descriptive measurements on one small common bank. The depth-4 map
has the smallest observed RMS here; no statistically supported ranking is
established. Its second latent coordinate has RMS residual 4.1334, so its
aggregate reduction does not imply near-Gaussian geometry in every direction.
The common latent draws reach different physical points under different maps.
None of these base-draw measurements establishes coverage of all posterior modes.

The original second-stage final scale remains near its -2 bound in every
continued map: mean log scale is -1.9874, -1.9936, -1.9716 and -1.9859,
respectively. All 32 points have slope below 0.1 for that output. The extra
depth-4 stages have no such slope alerts on this bank. Saturation is a training
repair signal; finite imperfect geometry does not prove NeuTra or HMC fails.

## Decision and remaining work

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the geometry diagnostic | All five immutable maps checked on the same declared bank | All scores finite; all target rows valid; sources and inputs match | Sparse local coverage | Carry reports alongside exact saved checkpoints | Global coverage or full training adequacy |
| Retain continued maps as viable HMC-trial candidates | Previous training assessments plus current numerical geometry screen pass | No new numerical rejection; saturation remains a repair signal, and control still over-clips | Whether residual geometry permits stable sampling | Fresh map-specific public fixed-transport tuning with identity latent mass | Verified kernel, converged chains or posterior estimate |
| Keep the training question open | Continued learning previously observed; current residuals are nonzero | No training-convergence evidence | Further learning, parameterization and seed sensitivity | Use downstream tuning outcomes to decide continuation or scale/capacity repair | A completed optimizer budget means a correctly trained map |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No numerical veto on the declared probe bank; all four continued maps remain viable |
| Statistically supported ranking | None |
| Descriptive-only differences | Residuals, scale summaries and observed changes from the parent |
| Default-readiness | Not established |
| Next evidence needed | Fresh fixed-transport HMC tuning, chain convergence, precision and posterior/reference checks; mode coverage and training robustness remain open |

The separate workspace posterior-checker issue for binary `positive_theta_2`
tail ESS remains unresolved. This diagnostic neither changes that checker nor
relaxes posterior checks. Imperfect whitening alone is not a reason to prohibit
a bounded HMC trial; its actual trajectories are the required downstream test.

## Execution and terminal review

The trusted GPU1 run used TensorFlow 2.20.0, float64, batch-native target/score
evaluation and XLA with one graph trace per map. Memory growth was verified
before device initialization. Numerical target, bridge, map restoration and
pullback code came from the frozen training source
`/tmp/BayesFilter-q20-training-repair-20260923-r1`; the separately frozen probe
wrapper was the module verified in the preceding engineering tests. No map,
optimizer history, training random stream or old result was overwritten.

The existing supervisor charged **66.020697 seconds** to both the campaign and
diagnostic ledgers. Remaining balances are **144386.382924 seconds (40.1073
hours)** for the campaign and **418.808365 seconds (6.9801 minutes)** for
diagnostics. The run completed on its first attempt within the 450-second cap.

[Full results](artifacts/q20-post-training-geometry-2026-09-23/run-01/worker-01/result.json),
[run manifest](artifacts/q20-post-training-geometry-2026-09-23/run-01/worker-01/manifest.json),
[terminal review](artifacts/q20-post-training-geometry-2026-09-23/run-01/terminal-review.json)
and [accounting](artifacts/q20-post-training-geometry-2026-09-23/run-01/accounting-01.json)
preserve the exact commands, checkpoint hashes, target/source identity, seeds,
device configuration, timings and per-map observations.

Post-run skeptical review: learning one local region could lower both reverse
KL and these probe residuals while missing important posterior modes. This is
the strongest alternative explanation to useful global transport learning.
Fresh HMC failures or inconsistent dispersed starts would overturn readiness;
successful posterior checks would demonstrate usefulness despite imperfect
Gaussianity. The weakest evidence remains one training stream and only 32 base
points. The diagnostic is complete; reliable posterior estimation is not.

Active checkpoint: five saved maps now have geometry reports at the paths
above. The old completed training result's balances are historical; the current
campaign ledger includes the new diagnostic charge. Next is fresh HMC tuning
and resolution of the separate posterior-checker issue, with map continuation
or parameterization repair guided by the resulting evidence.
