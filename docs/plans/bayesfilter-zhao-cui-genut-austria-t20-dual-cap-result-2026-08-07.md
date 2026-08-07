# Austria SIR T20 Pairwise/Dual-Cap GenUT Result

Date: 2026-08-07

Plan: `docs/plans/bayesfilter-zhao-cui-genut-austria-t20-dual-cap-plan-2026-08-07.md`
Artifact: `docs/benchmarks/artifacts/zhao_cui_genut_austria_t20_dual_cap_20260807/attempt01/result.json`
## Verdict

The T20 dual-cap arms are finite and pass the declared program-validity gates,
but they do not provide a value-stable replacement for the existing route. The
smooth coordinate cap changes most standardized coordinates at T20 and shifts
the value by roughly `1.5-1.6` log units, or `9.6-10.2` baseline MCSEs. The
observed score standard deviations fall by about 98.5-99.2%, but that is a
descriptive effect of changing the finite program, not evidence of lower score
bias or a more accurate score.

The historical uncapped pairwise arm is finite on the 16 claim seeds but fails
the disjoint calibration score-additivity gate on one tuning row (`residual
128.0`), so it is retained as a comparator and not promoted. The diagonal-only
baseline passes calibration and claim hard gates. No arm is promoted to a
default, HMC, NeuTra, or scientific score route.

## Scope And Missing Teacher

| Field | Value |
|---|---|
| Model | `austria_sir_T20` |
| Horizon | 20 observations |
| State / observation / parameter dimensions | 18 / 9 / 3 |
| Particles | 1008 |
| Event order | `x0_then_transition_before_y1_to_y20` |
| Source observation hash | `cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07` |
| Runtime | FP32, TF32 enabled, TensorFlow GPU/XLA |
| Seeds | tuning `98301,98302`; claim `98201..98216` |
| Strict T20 bounded teacher | unavailable; the existing bounded sampled teacher is only T1/T2 |

## Controls

All arms use epsilon 8, Sinkhorn/balance 16/16, ridge `1e-5`, four diagonal
steps at strength `0.2`, and four pairwise steps at strength `0.02` unless
otherwise stated.

| Arm | Pairwise | Coordinate cap | Radial cap |
|---|---|---|---|
| diagonal-only | off | off | off |
| pairwise-only | on | off | off |
| dual `b=.90` | on | `b=.90`, `p=8` | off |
| dual `b=.95` | on | `b=.95`, `p=8` | off |
| dual `b=.98` | on | `b=.98`, `p=8` | off |
| dual `b=.98` + radial | on | `b=.98`, `p=8` | RMS cap 2 |

## Claim Results

Values and scores are means with sample SD over the 16 common particle seeds.
The intervals in the JSON are descriptive 95% t intervals.

| Arm | Value mean (SD) | Score 0 mean (SD) | Score 1 mean (SD) | Score 2 mean (SD) | Hard valid |
|---|---:|---:|---:|---:|---|
| diagonal-only | `-683.3638 (0.6367)` | `-865.923 (3435.63)` | `170.885 (1272.44)` | `114.981 (301.97)` | yes |
| pairwise-only | `-682.1039 (0.5647)` | `-16.905 (33.94)` | `-108.702 (17.99)` | `15.152 (19.72)` | yes on claim, no on calibration |
| dual `b=.90` | `-681.8023 (0.5403)` | `38.909 (41.01)` | `-106.622 (18.95)` | `9.719 (3.00)` | yes |
| dual `b=.95` | `-681.8348 (0.6430)` | `27.279 (37.63)` | `-103.514 (12.42)` | `10.111 (3.41)` | yes |
| dual `b=.98` | `-681.7467 (0.5554)` | `39.318 (32.53)` | `-109.040 (19.42)` | `11.323 (2.33)` | yes |
| dual `b=.98` + radial | `-681.8232 (0.7039)` | `33.383 (37.18)` | `-106.884 (14.03)` | `10.230 (2.33)` | yes |

Relative to diagonal-only, value shifts are respectively `+1.2599`, `+1.5615`,
`+1.5291`, `+1.6172`, and `+1.5407`. The diagonal-only value MCSE is `0.1592`,
so these are `7.9`, `9.8`, `9.6`, `10.2`, and `9.7` MCSEs. Every paired value
difference is positive across the 16 seeds for the four dual-cap arms and the
pairwise arm.

Score SD ratios relative to diagonal-only:

| Arm | Score 0 SD ratio | Score 1 SD ratio | Score 2 SD ratio |
|---|---:|---:|---:|
| pairwise-only | `0.00988` | `0.01414` | `0.06531` |
| dual `b=.90` | `0.01194` | `0.01489` | `0.00995` |
| dual `b=.95` | `0.01095` | `0.00976` | `0.01130` |
| dual `b=.98` | `0.00947` | `0.01526` | `0.00771` |
| dual `b=.98` + radial | `0.01082` | `0.01103` | `0.00773` |

These are descriptive seed-dispersion ratios only. The score means also move
by hundreds of units relative to the diagonal route, so variance reduction
cannot be interpreted as accuracy improvement.

## Numerical Diagnostics

All six arms are finite and program-valid on the claim observations. The
uncapped pairwise calibration failure is localized to tuning seed `98301`, with
score-increment sum residual `128.0`; its claim residual maximum is only
`1.53e-5`. This is why the candidate is not promoted despite finite claim rows.

For the dual caps, the maximum claim cap-active fractions are approximately
`80.45%` (`b=.90`), `79.24%` (`b=.95`), `79.01%` (`b=.98`), and `78.57%`
(`b=.98` plus radial). Mean coordinate displacement is `0.1802`, `0.1709`,
`0.1660`, and `0.1654`, respectively. The largest pre-cap standardized
coordinate is about `8-10`, while the post-cap maximum is the selected bound.
Thus the cap is not tail-only in this T20 route.

Internal same-program finite-difference checks at `h=1e-3` were computed for
the uncapped pairwise and `b=.98` arms on claim seed `98201`. Maximum absolute
residuals were `66.83` and `258.06`, respectively. These diagnostics indicate
finite-precision/conditioning sensitivity and are not an external score oracle.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep diagonal-only as T20 reference | baseline hard gates pass | none | score variance remains huge | use as reference only | correctness or HMC readiness |
| Reject uncapped pairwise as promoted T20 arm | calibration score-additivity gate fails at seed 98301 | candidate veto | localized numerical instability | do not transfer T2 controls; retune T20 only if explicitly justified | pairwise idea universally rejected |
| Reject dual-cap as value-stable repair | value shift 9.6-10.2 baseline MCSEs; cap active on ~79-80% coordinates | value-stability veto | no exact T20 score authority | treat as finite exploratory diagnostic, not default | score bias, posterior quality |
| Radial cap addition | no meaningful value/dispersion advantage over `b=.98` | no promotion | one radial setting | no further radial sweep under this contract | radial cap generally useless |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | diagonal and all claim arms finite; pairwise fails one calibration additivity row |
| Statistically supported ranking | none; 16 seeds and no exact score authority |
| Descriptive-only differences | all value/score means, SDs, cap activity, and FD diagnostics |
| Default readiness | not established; dual cap is not promoted |
| Next evidence needed | a valid T20 teacher or independent observed-data score authority, plus a value-preserving repair |

## Artifact And Manifest

The machine-readable result and six per-arm checkpoints are under
`docs/benchmarks/artifacts/zhao_cui_genut_austria_t20_dual_cap_20260807/attempt01/`.
The manifest records the TensorFlow environment, GPU devices, memory-growth
policy, XLA/TF32 settings, seeds, plan, command, and source hashes.
