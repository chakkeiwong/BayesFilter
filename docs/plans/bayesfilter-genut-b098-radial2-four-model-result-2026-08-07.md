# GenUT b=.98 + Radial-2 Four-Model Result

Date: 2026-08-07

Plan: `docs/plans/bayesfilter-genut-b098-radial2-four-model-plan-2026-08-07.md`
Artifact: `docs/benchmarks/artifacts/genut_b098_radial2_four_model_20260807/attempt01/result.json`

## Verdict

The `b=.98 + radial RMS cap 2` scheme is finite on all four model scopes, but
the evidence does not support it as a universal improvement or new default.
The radial cap is an exact structural no-op for one-dimensional KSC SV. It has
small, mostly uncertainty-scale effects on LGSSM and predator-prey. It has a
material effect on Austria SIR, where it moves the score closer to the same-
target UKF/SGQF approximations, but increases value and score-0 dispersion and
does not remove the large value shift introduced by the coordinate cap.

The defensible interpretation is scope-specific:

- KSC: `b=.98` and `b=.98 + radial 2` are identical; the radial control cannot
  act because the state dimension is one.
- LGSSM: radial changes are tiny and not statistically separated from zero in
  four of five score coordinates; the exact Kalman score remains the authority.
- Predator-prey: radial changes are small, with paired intervals covering zero;
  no exact nonlinear score authority exists.
- Austria SIR: radial moves score means toward UKF/SGQF and lowers the
  approximate-reference score distance, but claim-level value shift and score
  dispersion remain material. There is no exact T20 score authority.

## Scope And Runtime

| Field | Value |
|---|---|
| Models | LGSSM `T=50`, KSC SV `T=10`, predator-prey `T=20`, Austria SIR `T=20` |
| Particles | `N=1008` |
| Seeds | claim `98201..98216`; tuning `98401,98402`; Austria tuning `98301,98302` |
| Backend | TensorFlow FP32, TF32 enabled, GPU/XLA, memory growth verified |
| Wall time | `527.1 s` |
| Cap | coordinate `b=.98,p=8`; pairwise radial RMS cap `2.0` |

## Claim Summaries

Means and SDs are over the 16 common claim seeds. They are descriptive.

### LGSSM `T=50`

| Arm | Value mean (SD) | Score means |
|---|---:|---|
| Diagonal | `-136.3335 (0.4680)` | `[5.795, -4.050, 0.240, -1.984, 5.538]` |
| Pairwise | `-136.3325 (0.4708)` | `[5.784, -4.032, 0.221, -2.056, 5.528]` |
| Coordinate cap | `-136.3330 (0.4695)` | `[5.690, -3.989, 0.198, -2.116, 5.536]` |
| Dual cap | `-136.3301 (0.4681)` | `[5.708, -3.990, 0.203, -2.104, 5.504]` |

The exact Kalman reference is value `-136.0760`, score
`[5.6554,-3.8351,0.3024,-1.9172,4.3543]`. Score Euclidean distance to that
reference was `1.213` for coordinate-only and `1.180` for dual cap. This is a
small descriptive change, not a statistically supported improvement. The
dual-minus-coordinate paired 95% intervals covered zero for the first four
score coordinates; score 4 had a small negative interval, but this does not
establish broad accuracy improvement. The coordinate cap was active on about
74% of coordinates, so it is not tail-only in this scope.

### KSC SV `T=10`

| Arm | Value mean (SD) | Score means |
|---|---:|---|
| Diagonal | `-19.95395 (0.04760)` | `[-0.6944, 0.6077]` |
| Pairwise | `-19.95395 (0.04760)` | `[-0.6944, 0.6077]` |
| Coordinate cap | `-19.95785 (0.04894)` | `[-0.7068, 0.5755]` |
| Dual cap | `-19.95785 (0.04894)` | `[-0.7068, 0.5755]` |

The pairwise and radial controls are exact no-ops at state dimension one. The
coordinate cap moves the result away from the dense transformed-mixture
diagnostic: its value is `-19.95628` and score is approximately
`[-0.70567,0.63549]`; the diagonal value/score is closer in score distance
(`0.00272`) than the capped/dual value (`0.03700`). The internal FD check
passed, but this is not evidence for the coordinate cap.

### Predator-prey `T=20`

| Arm | Value mean (SD) | Score means |
|---|---:|---|
| Diagonal | `-102.7395 (0.2923)` | `[-27.775,0.0777,-0.0875,1.042,18.367,-23.651]` |
| Pairwise | `-102.7450 (0.3072)` | `[-27.805,0.0723,-0.0877,1.041,18.434,-23.735]` |
| Coordinate cap | `-102.7268 (0.3063)` | `[-27.769,0.0759,-0.0871,1.016,18.262,-23.525]` |
| Dual cap | `-102.7277 (0.3055)` | `[-27.748,0.0755,-0.0870,1.015,18.244,-23.502]` |

Against the prior SGQF diagnostic, score distance was `1.173` for coordinate-
only and `1.142` for dual; against the prior Zhao-Cui diagnostic it was `5.241`
and `5.215`. These are descriptive diagnostic gaps, not accuracy evidence.
The paired dual-minus-coordinate intervals covered zero for all six score
coordinates and value. The radial cap slightly increased score SDs in every
coordinate and did not establish a benefit.

### Austria SIR `T=20`

| Arm | Value mean (SD) | Score means |
|---|---:|---|
| Diagonal | `-683.3638 (0.6367)` | `[-865.923,170.885,114.981]` |
| Pairwise | `-682.1039 (0.5647)` | `[-16.905,-108.702,15.152]` |
| Coordinate cap | `-681.7467 (0.5554)` | `[39.318,-109.040,11.323]` |
| Dual cap | `-681.8232 (0.7039)` | `[33.383,-106.884,10.230]` |

The same-target UKF diagnostic is value `-681.6886`, score
`[29.184,-106.963,9.327]`; SGQF is value `-682.3480`, score
`[28.739,-106.659,9.431]`. The dual arm's score gaps were `[+4.20,+0.08,+0.90]`
to UKF and `[+4.64,-0.23,+0.80]` to SGQF, compared with approximately
`[+10.13,-2.08,+2.00]` and `[+10.58,-2.38,+1.89]` for coordinate-only.
Thus radial cap 2 moves the score descriptively closer to both approximate
references.

However, relative to coordinate-only, the dual arm changes the value mean by
`-0.0765` and increases value SD by `26.7%`; score-0 SD increases by `14.3%`.
The paired 95% intervals for the radial-minus-coordinate changes cover zero:
value `[-0.242,0.089]`, score 0 `[-22.79,10.92]`, score 1 `[-7.90,12.21]`,
score 2 `[-2.42,0.23]`. The coordinate cap remains active on about `78.6%`
of coordinates. Relative to diagonal-only, the dual value shift is `+1.541`
log units, about `9.7` diagonal baseline MCSEs. The uncapped pairwise arm passes
claim rows but fails one calibration additivity row and is not promoted.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Universal radial-cap default | no cross-model statistical support | reject promotion | few-seed stochastic evidence and no nonlinear oracle | keep scope-specific arms only | universal superiority |
| KSC dual cap | exact structural no-op for pairwise/radial | viable but irrelevant | cap changes dense-reference agreement | use diagonal/reference route | radial benefit |
| LGSSM dual cap | finite; tiny exact-reference movement | viable diagnostic | FD sensitivity and residual score gap | retain Kalman as authority | exact-score improvement |
| Predator-prey dual cap | finite; paired radial changes cover zero | viable diagnostic | no exact score authority | retain coordinate-only as simpler comparator | nonlinear accuracy |
| Austria dual cap | finite; closer to UKF/SGQF descriptively | value/dispersion veto for default | UKF/SGQF are approximate only | use as exploratory proposal arm, retain no-radial control | score correctness |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | All four baselines and all claim arms are finite/program-valid; Austria uncapped pairwise fails calibration only |
| Statistically supported ranking | None across arms or models |
| Descriptive-only differences | all values, score means/SDs, reference distances, cap activity, and FD diagnostics |
| Default readiness | not established; no universal radial-cap default |
| Next evidence needed | target-specific independent score authorities and larger paired replications if a default decision is required |

## Internal FD Diagnostic

The `h=1e-3` same-program finite-difference check passed only for KSC. Maximum
absolute residuals for the diagonal/coordinate/dual arms were:

| Model | Diagonal | Coordinate cap | Dual cap |
|---|---:|---:|---:|
| LGSSM | `0.181` | `0.849` | `0.575` |
| KSC | `0.00120` | `0.000511` | `0.000511` |
| Predator-prey | `0.461` | `0.456` | `0.582` |
| Austria SIR | `79.74` | `258.06` | `201.85` |

These checks are explanatory only in this campaign. They do not establish that
the reported score is correct; conversely, the matching baseline failures show
that the small-step FP32/TF32 diagnostic is not an isolated radial-cap failure.

## Artifact Integrity

The JSON result, Markdown summary, and sixteen model/arm checkpoints are under
`docs/benchmarks/artifacts/genut_b098_radial2_four_model_20260807/attempt01/`.
The manifest records git commit, command, environment, GPU/XLA/TF32 settings,
memory growth, seeds, plan, and source hashes.
