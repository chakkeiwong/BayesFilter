# Cubature/GenUT Versus Previous LGSSM Runs

Date: 2026-07-21

## Direct Answer

The comparison metric is now the same as the previous Contract E runs:

- same dataset seed `81100` and canonical 3-dimensional observation target;
- same six outputs: `value, phi1, phi2, phi3, q_scale, r_scale`;
- same physical-to-HMC score transformation;
- same per-coordinate relative error normalization;
- same 16-seed sample SD/SE and simultaneous critical value
  `3.036283222821165`;
- same frozen screen margins: `0.001` for value and `0.05` for each score.

The read-only comparison artifact is:

`docs/benchmarks/artifacts/lgssm_cubature_genut_previous_metric_comparison_20260721.json`

## T=50 Comparison

Entries are mean relative error followed by the simultaneous interval.

| Arm | Value | phi1 | phi2 | phi3 | q_scale | r_scale | Screen |
|---|---|---|---|---|---|---|---|
| Contract E, `N=5000` | `+0.148% [0.112%,0.185%]` | `-0.207% [-2.295%,1.881%]` | `-0.308% [-1.800%,1.183%]` | `-10.972% [-30.018%,8.073%]` | `-9.912% [-17.716%,-2.107%]` | `-1.814% [-4.408%,0.779%]` | `screen_fail` |
| Contract E, `N=10000` | `+0.174% [0.150%,0.197%]` | `-0.288% [-1.439%,0.863%]` | `+0.051% [-1.049%,1.152%]` | `+0.371% [-18.327%,19.068%]` | `-15.899% [-22.004%,-9.794%]` | `-3.611% [-5.239%,-1.983%]` | `screen_fail` |
| Cubature/GenUT, `N=1008` | `-0.006% [-0.240%,0.227%]` | `-2.414% [-16.664%,11.836%]` | `-7.390% [-15.610%,0.831%]` | `+3.855% [-95.355%,103.065%]` | `+3.915% [-94.474%,102.304%]` | `+22.876% [-29.295%,75.047%]` | `inconclusive` |

## Interpretation

The new route is descriptively closer on mean value error and mean `q_scale`
error than both prior arms. That is not a supported improvement claim because:

- the new route has `N=1008`, while the prior arms have `N=5000` and `N=10000`;
- prior arms use tuned Contract E controls (`epsilon=0.5`, 20 Sinkhorn steps,
  balance steps 5 or 8, XLA); the new diagnostic uses `epsilon=2`, 8 Sinkhorn
  steps, no JIT, and Cubature/GenUT residual injection;
- the random streams are not paired common-random-number streams. Reusing seed
  labels does not pair them: the prior preparation uses Philox keys
  `[seed, domain_tag]` and float64 raw draws cast to float32, while the new
  wrapper uses keys `[seed, horizon]` and `[seed, horizon+100]` with a different
  particle shape;
- the new score dispersion is much larger. At `T=50`, its SD is about 6.4x,
  6.8x, 5.5x, 5.2x, 12.6x, and 20.1x the N=5000 SD for value, `phi1`, `phi2`,
  `phi3`, `q_scale`, and `r_scale`, respectively.

Therefore the correct classification is:

> Same metric and same target: yes. Evidence that the new algorithm improves the
> previous method: unsupported. The result is a viable feasibility diagnostic,
> but the T=50 screen is inconclusive and the controls/particle count are not
> matched.

## What Would Establish Improvement

Run a predeclared matched comparison at the same `N`, horizon, observations,
16 seed IDs, precision/backend, and resource policy. Tune each method within its
own scope using disjoint calibration/validation data, freeze controls, then
compare paired per-seed six-coordinate error rows. A supported improvement would
require a declared paired interval/test for the difference, not only a smaller
descriptive mean.

## Evidence Status

| Evidence class | Status |
|---|---|
| Metric identity | Passed |
| Target identity | Passed |
| Prior N=5000 screen | `screen_fail` |
| Prior N=10000 screen | `screen_fail` |
| New Cubature/GenUT screen | `inconclusive` |
| Statistically supported ranking | None |
| Exact filtering validity | Not established |
| Nonlinear/NAWM conclusion | Not made |
