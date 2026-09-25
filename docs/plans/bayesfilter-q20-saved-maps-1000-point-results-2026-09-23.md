# Saved q20 maps: full 1,000-point diagnostic results

The new standard procedure has now run on the actual saved maps on GPU1.
Control, repaired clipping, and depth four each completed all 1,000 points.
All 3,000 completed target/score evaluations were valid and finite. Their score
residuals remain substantial. The batch-128 map paused after 440 points when
the diagnostic allocation reached its stopping margin; it has no full report.
The optional parent-map rerun was not started.

The [plan](bayesfilter-q20-saved-maps-1000-point-plan-2026-09-23.md) fixes the
target, checkpoints, bank, numerical procedure, interpretation and budget.
These are actual saved-map results, distinct from the earlier software tests.

## Numerical results

Each completed map used the same 1,000 standard-normal latent points with seed
`(1843264305,1053964650)` and fifty native batches of twenty. The reported norm
is `||grad_z log pi_z(z)+z||`, including the full log-Jacobian score. The companion
quantity is `r(z)=log pi_z(z)+||z||^2/2`. An exact standard-normal pullback would
have zero score residual and constant r. Quantiles use the original lower-index
convention; these are vector norms, not per-coordinate RMS values.

| Quantity | Continued-training control | Repaired clipping, depth 2 | Repaired clipping, depth 4 |
| --- | ---: | ---: | ---: |
| Valid points | 1,000/1,000 | 1,000/1,000 | 1,000/1,000 |
| Minimum norm | 0.3638 | 0.7363 | 0.3300 |
| Median norm | 6.1216 | 5.4109 | 4.0640 |
| Mean norm | 7.0121 | 5.9718 | 4.0881 |
| Vector-norm RMS | 8.2024 | 6.8458 | 4.5192 |
| p95 | 15.1123 | 12.2537 | 7.1818 |
| p99 | 23.2728 | 17.3788 | 9.0059 |
| Maximum norm | 33.4371 | 28.5069 | 13.4745 |
| Fraction exceeding 1 | 98.7% | 99.2% | 95.9% |
| Fraction exceeding norm(z) | 97.4% | 97.0% | 86.2% |
| Range of r, log units | 31.3093 | 22.6333 | 13.9826 |
| Per-coordinate RMS | 4.1012 | 3.4229 | 2.2596 |

The depth-four map has descriptively smaller residuals on this bank. A ranking
across training runs is not established: there is one training stream per arm,
the reused seed was previously inspected, and no ranking criterion with
appropriate uncertainty analysis was declared. Common base points also map to
different physical parameter values under different maps.

The depth-four map is still far from the exact Gaussian score identity on these
points: 86.2% of residual norms exceed the corresponding Gaussian score norm.
Its second-coordinate RMS residual is 4.2638. The original second-stage final
scale remains near its lower bound: mean log scales are -1.98735, -1.99358 and
-1.97141 for control, clipping repair and depth four. For that scale output,
all 1,000 points have tanh-parameterization slope below the existing 0.1 alert.
This is a repair signal, not proof that other layers cannot compensate or that
the NeuTra research direction fails.

## Completion, accounting and remaining work

| Map | Saved points | Full diagnostic |
| --- | ---: | --- |
| Control | 1,000 | Complete |
| Repaired clipping | 1,000 | Complete |
| Depth four | 1,000 | Complete |
| Batch 128 | 440 | Incomplete; no full-bank statistics issued |
| Shared parent | 0 in this phase | Optional rerun not started |

The worker ended cooperatively after 409.6722 seconds and exited successfully
with scientific completion status `partial_budget`. Including supervision,
**411.096951625 seconds** were charged to both ledgers. The remaining balances
are **143975.28597232018 campaign seconds** and **7.711412891243526 diagnostic
seconds**. The stage's remaining margin is insufficient to resume the last map.
The pending user question requests two additional diagnostic minutes under the
unchanged total campaign limit. No increase has been applied and no unfunded
retry has run. The 22 completed batches for batch 128 are preserved for exact
continuation; only 560 target rows remain to evaluate.

The [run manifest](artifacts/q20-saved-maps-1000-point-2026-09-23/run-01/worker-01/manifest.json),
[results](artifacts/q20-saved-maps-1000-point-2026-09-23/run-01/worker-01/result.json),
[accounting](artifacts/q20-saved-maps-1000-point-2026-09-23/run-01/accounting-01.json)
and [independent statistic audit](artifacts/q20-saved-maps-1000-point-2026-09-23/run-01/statistics-audit-01.json)
preserve inputs and observations. Each map's `*-points/` directory contains
checksummed per-batch latent points, residual vectors and r values.

## Validation and terminal review

Trusted execution used TensorFlow 2.20.0, float64, GPU1, XLA and one numerical
graph trace per completed map. Memory growth was set before import and verified
before logical GPU initialization. The allocator recorded a 268899328-byte
peak. The target, score and map restoration code stayed on the original frozen
training source; the diagnostic was an exact copy of the v2 module verified by
the software tests. Every completed map's restored parameters were unchanged.

An independent Python standard-library audit read the saved per-point values,
checked their file hashes, and reproduced the norm statistics, exceedance
fractions, r range and global centered-r RMS to the existing relative/absolute
1e-12 engineering tolerance. This verifies reporting, not an independent
derivation of the target score. The earlier finite-difference score checks
remain separate evidence with their original scope.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept three full post-training reports | Exactly 1,000 saved, valid points per map and checked global statistics | No numerical or identity veto | Base points may miss posterior modes | Carry these reports with their exact checkpoints | Adequate training, posterior validity or HMC convergence |
| Keep the whitening problem open | Residuals remain substantial relative to the Gaussian score | Scale saturation remains a repair trigger; finite size is explanatory | Whether this geometry permits stable downstream sampling | Use fresh fixed-transport HMC evidence or targeted training repair under the campaign plan | NeuTra as a method has failed |
| Pause the fourth report | Only 440/1,000 points saved | Diagnostic budget stopping margin reached | Remaining 560 points unevaluated | Resume saved batches if the pending two-minute allocation is approved | The partial map passed the full procedure |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No numerical veto among the three fully evaluated maps; the fourth has incomplete evidence |
| Statistically supported ranking | None |
| Descriptive-only differences | Residual norms, tails, exceedance fractions, density variation and scale summaries |
| Default-readiness | No trained-map or HMC promotion follows from these reports |
| Next evidence needed | Complete the last report, then target-specific downstream sampling and posterior/reference checks for any quality claim |

The strongest alternative explanation for smaller residuals is a map that fits
one accessible region while missing posterior mass elsewhere. More base draws
cannot by themselves resolve that. Independent posterior/reference evidence and
dispersed-chain behavior could overturn a favorable interpretation of local
geometry. The run validates three complete diagnostic reports and stops the
fourth for budget, without invalidating the implementation, target or method.
