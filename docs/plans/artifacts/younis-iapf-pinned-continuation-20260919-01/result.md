# Pinned replication passes; fresh weak cases expose fitting and accuracy failures

2026-09-19. The device correction is confirmed, but the fresh weakly nonlinear datasets still lose to simple filters. Widening the fitting box leaves the problematic bound contact unresolved. The inherited stopping threshold is also mathematically uninformative. The candidate is not promoted.

## pinned (pinned01)

Primary comparisons: 4/4 pass. Nonlinear bias screens: 4/4 pass.

**Heuristic losses:** none on the four nonlinear datasets.

| Dataset | Regime | Ancestor MSE | New MSE | UKF MSE | New minus ancestor interval | Fit bound |
|---|---|---:|---:|---:|---|---|
| 1490 | affine | 0.0016023 | 0.0006483 | 0.0000000 | [-0.0013486, -0.0006023] | False |
| 1500 | weak | 0.0049922 | 0.0015068 | 0.0018787 | [-0.0048072, -0.0022339] | False |
| 1501 | weak | 0.0061980 | 0.0014747 | 0.0077080 | [-0.0064325, -0.0031813] | False |
| 1510 | curved | 0.0022972 | 0.0013259 | 0.0259029 | [-0.0015851, -0.0004208] | False |
| 1511 | curved | 0.0083948 | 0.0038193 | 0.3450186 | [-0.0070923, -0.0023497] | False |

[Manifest](pinned01/manifest.json), [raw results](pinned01/results.json), [frozen calibration](pinned01/frozen-calibration.json), [log](pinned01.log).

## fresh (fresh01)

Primary comparisons: 3/4 pass. Nonlinear bias screens: 4/4 pass.

**Heuristic losses:** 1900: ekf, ukf; 1901: ukf.

| Dataset | Regime | Ancestor MSE | New MSE | UKF MSE | New minus ancestor interval | Fit bound |
|---|---|---:|---:|---:|---|---|
| 1900 | weak | 0.0864440 | 0.0853189 | 0.0138890 | [-0.0216411, 0.0211390] | True |
| 1901 | weak | 0.0024756 | 0.0010790 | 0.0005113 | [-0.0019779, -0.0008493] | False |
| 1910 | curved | 0.0026663 | 0.0016243 | 0.0104953 | [-0.0018310, -0.0004339] | False |
| 1911 | curved | 0.0039317 | 0.0017840 | 0.2292166 | [-0.0034808, -0.0010343] | False |

[Manifest](fresh01/manifest.json), [raw results](fresh01/results.json), [frozen calibration](fresh01/frozen-calibration.json), [log](fresh01.log).

## wide (wide01)

Primary comparisons: 3/4 pass. Nonlinear bias screens: 4/4 pass.

**Heuristic losses:** 1900: ukf; 1901: ukf.

| Dataset | Regime | Ancestor MSE | New MSE | UKF MSE | New minus ancestor interval | Fit bound |
|---|---|---:|---:|---:|---|---|
| 1900 | weak | 0.0526371 | 0.0462637 | 0.0138890 | [-0.0175583, 0.0039384] | True |
| 1901 | weak | 0.0020511 | 0.0011670 | 0.0005113 | [-0.0014543, -0.0003739] | False |
| 1910 | curved | 0.0028361 | 0.0012787 | 0.0104953 | [-0.0024762, -0.0007779] | False |
| 1911 | curved | 0.0040760 | 0.0017494 | 0.2292166 | [-0.0034874, -0.0013372] | False |

[Manifest](wide01/manifest.json), [raw results](wide01/results.json), [frozen calibration](wide01/frozen-calibration.json), [log](wide01.log).

## Interpretation and decision

The paired wider-box check leaves the three interior proposal fits exactly unchanged. Dataset 1900 still contacts a bound, and its first-step predictive shape residual remains large. Widening is rejected as a complete fitting repair; it passes the narrower interior non-harm check. Dataset 1901 also loses to UKF with an interior fit, so removing bound contacts alone cannot resolve the weak-case accuracy gap. See the [mathematical diagnosis](fitting-diagnosis.md) for the stopping-rule derivation and score limits.

All completed attempts verify a single physical RTX5080, memory growth, GPU:0 particle/control outputs, one trace per kernel, frozen coefficients, numerical references and preserved source hashes. Independent Python arithmetic reproduces corrected scores and MSE summaries. This repairs the previous device-selection evidence gap; it does not retroactively change the previous run.

The controls subtract independent-calibration linear projections of exactly centered ancestor statistics and same-law Gaussian moment contrasts. They preserve the raw normalized Fisher estimator's finite-N bias under the stated law and arithmetic assumptions. They are not derivatives of the finite likelihood program.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep diagnostic candidate only | Pinned 4/4; fresh and wider-box 3/4 each | Weak-case heuristic failures, persistent bound contact and vacuous fitting stop; affine Kalman still wins | Fixed short scalar datasets and one calibration each | Repair fitting-cloud/stopping protocol on separate calibration data; retain UKF | Default, population, LEDH or HMC readiness |
| Close this planned campaign | Three planned stages executed | Eight fitting attempts exhausted | Larger-cloud repair remains untested | Write a fresh bounded fitting protocol before more fitting | Whole-master completion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical/device checks pass for completed runs; observed heuristic losses and bound contacts are listed above |
| Statistically supported ranking | Only per-dataset new-versus-matched-ancestor comparisons with negative primary upper endpoints, conditional on frozen fits |
| Descriptive-only differences | Percent reductions, variance, heuristic mean differences, timings and comparisons across stages |
| Default readiness | Not established; inherited tiny fitting cloud and limited model/horizon remain |
| Next evidence needed | Resolve any bound/heuristic failure, then independently repeated fitting/calibration and longer horizons |

Budget used: 3/4 launches, 3068/8000 charged filter calls, 8/8 adaptive fits, 137.859234/1800 driver seconds. Fitting reserves eight calls per attempt; actual fitting calls are separately retained in each fit record.

Post-run red-team: chosen T2 scalar cases and one calibration can exaggerate general usefulness. A fresh-data success cannot certify the proposal fitting protocol; a heuristic loss is a candidate failure, not evidence against Fisher's identity or the entire iAPF/KDM/LEDH direction. Calibration uncertainty and data-population uncertainty remain outside the intervals. Do not retune against final streams.

[Plan](../../younis-iapf-pinned-continuation-2026-09-19.md); [analysis](analysis.json); [conditional errors](conditional-errors.csv); [CPU checks](cpu-tests01.log).
