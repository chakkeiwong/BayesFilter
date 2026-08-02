# Retuned Higher-Moment GenUT Value And Score Comparison

Date: 2026-07-23

Candidate artifact:
`docs/benchmarks/artifacts/higher_moment_genut_retuning_20260723/attempt03/result.json`

Prior artifact:
`docs/benchmarks/artifacts/higher_moment_contract_e_regression_20260723/attempt02/result.json`

Both artifacts use identical claim observations and identical 16 particle
seeds within each scope. Candidate-minus-prior intervals are paired Student-t
95% intervals. Dense/Kalman references are post-run diagnostics and were not
used in tuning.

## LGSSM T=2

| Quantity | Retuned mean [95% CI] | Prior mean [95% CI] | Oracle | Retuned error | Paired abs-error change [95% CI] |
|---|---:|---:|---:|---:|---:|
| value | -8.819203 [-8.913964, -8.724441] | -8.820236 [-8.914672, -8.725800] | -8.862151 | 0.042948 | 0.000669 [-0.000519, 0.001857] |
| phi1 | 3.956353 [3.676056, 4.236650] | 3.955536 [3.675235, 4.235837] | 3.828014 | 0.128339 | 0.000600 [-0.004658, 0.005857] |
| phi2 | -0.373449 [-0.429467, -0.317431] | -0.374019 [-0.431688, -0.316349] | -0.384181 | 0.010733 | -0.001342 [-0.005789, 0.003104] |
| phi3 | -0.048188 [-0.097188, 0.000812] | -0.049051 [-0.098015, -0.000087] | -0.083087 | 0.034899 | 0.000849 [-0.001147, 0.002845] |
| q_scale | 4.660078 [4.284009, 5.036148] | 4.657738 [4.277207, 5.038269] | 4.417213 | 0.242866 | -0.004717 [-0.015384, 0.005950] |
| r_scale | 11.057809 [10.429424, 11.686195] | 11.060646 [10.434589, 11.686703] | 11.138136 | -0.080326 | 0.005533 [-0.002451, 0.013517] |

No value or score coordinate has a statistically supported paired accuracy
change at this scope.

## LGSSM T=10

| Quantity | Retuned mean [95% CI] | Prior mean [95% CI] | Oracle | Retuned error | Paired abs-error change [95% CI] |
|---|---:|---:|---:|---:|---:|
| value | -32.141328 [-32.240861, -32.041795] | -32.140096 [-32.241053, -32.039138] | -32.052616 | -0.088713 | 0.000575 [-0.004340, 0.005491] |
| phi1 | 11.139410 [10.746293, 11.532527] | 11.159604 [10.765178, 11.554029] | 11.279994 | -0.140584 | 0.001717 [-0.034573, 0.038008] |
| phi2 | -0.291848 [-0.476980, -0.106715] | -0.288911 [-0.476689, -0.101134] | -0.304041 | 0.012194 | -0.004763 [-0.015627, 0.006100] |
| phi3 | -1.335269 [-1.449967, -1.220572] | -1.339795 [-1.453923, -1.225666] | -1.305039 | -0.030231 | 0.001972 [-0.005613, 0.009556] |
| q_scale | 9.681445 [9.026667, 10.336223] | 9.688007 [9.041469, 10.334544] | 9.488660 | 0.192785 | 0.002029 [-0.046693, 0.050751] |
| r_scale | 14.601965 [13.893109, 15.310821] | 14.601880 [13.879695, 15.324066] | 14.068336 | 0.533630 | -0.019757 [-0.048205, 0.008690] |

No value or score coordinate has a statistically supported paired accuracy
change at this scope.

## LGSSM T=50

| Quantity | Retuned mean [95% CI] | Prior mean [95% CI] | Oracle | Retuned error | Paired abs-error change [95% CI] |
|---|---:|---:|---:|---:|---:|
| value | -136.070522 [-136.360734, -135.780311] | -136.064044 [-136.345591, -135.782497] | -136.075975 | 0.005452 | 0.019202 [0.002585, 0.035820] |
| phi1 | 5.708890 [5.342751, 6.075028] | 5.719227 [5.332309, 6.106144] | 5.655446 | 0.053444 | -0.028134 [-0.084943, 0.028676] |
| phi2 | -4.016671 [-4.318592, -3.714751] | -4.024100 [-4.310154, -3.738047] | -3.835057 | -0.181615 | 0.030039 [-0.006717, 0.066795] |
| phi3 | 0.214474 [-0.027456, 0.456405] | 0.220750 [-0.013684, 0.455185] | 0.302362 | -0.087887 | 0.017989 [-0.005460, 0.041439] |
| q_scale | -2.269385 [-3.300499, -1.238271] | -2.214523 [-3.271355, -1.157690] | -1.917176 | -0.352209 | -0.041507 [-0.127786, 0.044772] |
| r_scale | 4.416392 [2.656778, 6.176006] | 4.402989 [2.701998, 6.103981] | 4.354276 | 0.062116 | 0.103720 [-0.030086, 0.237527] |

The retuned 16-seed value mean is closer to the oracle, but the paired
per-seed absolute value error is statistically worse. No score-coordinate
accuracy change is statistically supported.

## Fresh Transformed SV T=50

| Quantity | Retuned mean [95% CI] | Prior mean [95% CI] | Dense reference | Retuned error | Paired abs-error change [95% CI] |
|---|---:|---:|---:|---:|---:|
| value | -116.813925 [-116.864370, -116.763479] | -116.804210 [-116.856213, -116.752208] | -116.799295 | -0.014630 | -0.000270 [-0.006829, 0.006289] |
| theta_gamma | -0.839342 [-0.898670, -0.780014] | -0.812440 [-0.872376, -0.752503] | -0.852029 | 0.012687 | 0.000896 [-0.014301, 0.016093] |
| theta_log_beta | -2.269701 [-2.380521, -2.158880] | -2.283505 [-2.395205, -2.171806] | -2.231567 | -0.038133 | -0.001429 [-0.013640, 0.010781] |

The score mean moved closer to the dense reference on both coordinates, but
the paired absolute-error intervals include zero. Improvement is descriptive,
not statistically supported.

## Predator-Prey T=20

There is no exact score oracle for this scope.

| Quantity | Retuned mean [95% CI] | Prior mean [95% CI] | Paired value/score change [95% CI] |
|---|---:|---:|---:|
| value | -103.166317 [-103.334445, -102.998188] | -103.164259 [-103.333152, -102.995365] | -0.002058 [-0.005789, 0.001673] |
| r | -22.102998 [-22.799004, -21.406991] | -22.110306 [-22.809023, -21.411588] | 0.007308 [-0.010340, 0.024956] |
| K | 1.198074 [1.134805, 1.261343] | 1.199750 [1.136987, 1.262513] | -0.001676 [-0.004048, 0.000696] |
| a | -0.001608 [-0.003450, 0.000234] | -0.001464 [-0.003205, 0.000276] | -0.000143 [-0.000407, 0.000121] |
| s | -3.166723 [-3.446947, -2.886498] | -3.166701 [-3.441714, -2.891687] | -0.000022 [-0.014979, 0.014935] |
| u | -0.604427 [-1.072761, -0.136094] | -0.644450 [-1.092250, -0.196650] | 0.040022 [-0.024673, 0.104718] |
| v | 0.111219 [-0.468901, 0.691339] | 0.159891 [-0.393378, 0.713160] | -0.048672 [-0.128304, 0.030960] |

All paired intervals include zero. The two candidates are statistically
indistinguishable under this evidence.

## SIR Boundary

The trial preserved the historical fixed-parameter SGQF SIR value
`-691.3692068263654`; it did not execute the parameterized score-capable
Austria SIR GenUT route. No SIR GenUT value/score comparison can be claimed
from this artifact.

## Decision

| Evidence question | Result |
|---|---|
| Did moment matching improve? | Yes, for the emitted selected diagonal moment residuals. |
| Did score accuracy improve statistically? | No supported improvement on any oracle-backed coordinate. |
| Did value accuracy improve statistically? | No; LGSSM `T=50` paired absolute value error regressed. |
| Did predator-prey change materially? | No statistically supported paired change; no oracle exists. |
| Is the retuned route ready for promotion? | No. |

These comparisons do not establish exact likelihood, exact score, HMC
readiness, default readiness, leaderboard promotion, or method superiority.
