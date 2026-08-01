# GenUT Predator-Prey Leaderboard Continuation Result

Date: 2026-07-22

Status: `VALUE_COMPATIBLE_SCORE_N_STABLE_DIAGNOSTIC_ONLY`

## Value

| Route | N | Mean | SD | 95% CI |
|---|---:|---:|---:|---:|
| GenUT | 96 | -103.084606 | 0.775194 | [-103.497678, -102.671535] |
| GenUT | 384 | -103.367252 | 0.372564 | [-103.565777, -103.168727] |
| GenUT | 1002 | -103.161827 | 0.316176 | [-103.330305, -102.993349] |
| Bootstrap PF reference | 65536 | -103.141449 | 0.038219 | [-103.161815, -103.121084] |
| Bootstrap PF reference | 262144 | -103.137676 | 0.012751 | [-103.144471, -103.130881] |

## Score Stability

| Coordinate | GenUT N=1002 mean | 95% CI | N=1002 minus N=384 95% CI | Stable |
|---|---:|---:|---:|---:|
| r | -22.147850 | [-22.847029, -21.448670] | [-1.426341, 1.095016] | True |
| K | 1.198696 | [1.136246, 1.261147] | [-0.009153, 0.181634] | True |
| a | -0.001523 | [-0.003205, 0.000159] | [-0.002470, 0.004122] | True |
| s | -3.151173 | [-3.416852, -2.885493] | [-0.718397, 0.245884] | True |
| u | -0.640241 | [-1.076820, -0.203662] | [-0.895604, 0.689583] | True |
| v | 0.154255 | [-0.384672, 0.693182] | [-0.853367, 1.101859] | True |

## Same-Target Triangulation

The principal-square-root UKF value is `-103.137862`, compared with the refined
bootstrap-PF mean `-103.137676`. Its score is analytical but remains an
approximation diagnostic, not truth.

| Coordinate | GenUT N=1002 95% CI | UKF analytical score | UKF inside interval |
|---|---:|---:|---:|
| r | [-22.847029, -21.448670] | -21.763646 | True |
| K | [1.136246, 1.261147] | 1.175923 | True |
| a | [-0.003205, 0.000159] | -0.001066 | True |
| s | [-3.416852, -2.885493] | -3.148735 | True |
| u | [-1.076820, -0.203662] | -0.703709 | True |
| v | [-0.384672, 0.693182] | 0.246536 | True |

This cross-method agreement supports score consistency at the tested dataset
and parameter point. It does not prove accuracy because GenUT and UKF may share
Gaussian-closure error.

## Decision

GenUT N=1002 minus refined PF value: `-0.024151` with 95% interval `[-0.192766, 0.144464]`.

GenUT passed the declared value-compatibility and score-N-stability screens. The row remains unadmitted because no independent score truth authority or leaderboard integration exists.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Predator-prey GenUT value | True | Passed engineering/reference checks | Bootstrap PF is refined, not exact | Preserve as candidate value evidence | No exact likelihood or superiority |
| Predator-prey GenUT score | N-stable=True | Recursive same-scalar audit passed in tuning | No independent marginal-score oracle | Build an independent analytical score authority or stronger consistency ladder | No score truth or HMC readiness |
| Leaderboard admission | Not admitted | Identity valid; evidence incomplete | Score truth and integration schema absent | Close cross-cutting leaderboard gaps | No default change |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed |
| Statistically supported ranking | None; this is compatibility and stability evidence |
| Descriptive-only differences | UKF comparisons, runtimes, and per-rung means outside declared intervals; SGQF is target-mismatched and excluded |
| Default readiness | Failed; GenUT remains experimental |
| Next evidence needed | Independent analytical score validation, leaderboard wiring, and high-dimensional memory repair |

## Post-Run Red Team

The strongest alternative explanation is that two biased approximations agree at one dataset and parameter point. The result would be overturned by PF refinement drift, fresh-DGP value disagreement, or persistent score movement at larger N. The generic fixed-SGQF number from the pre-correction artifact is not evidence: its loop transitions before `y0`, contrary to the canonical target timing, and is now blocked by the leaderboard runner. The weakest evidence remains score truth, not finite execution.
