# Pairwise Radial-Cap Zhao-Cui/Austria Diagnostic Result

Date: 2026-08-06

Plan:
`docs/plans/bayesfilter-pairwise-radial-cap-zhao-cui-austria-diagnostic-plan-2026-08-06.md`

Primary artifact:
`docs/benchmarks/artifacts/pairwise_radial_cap_diagnostic_20260806/three_seed_attempt01/result.json`

Replay artifacts:

- `docs/benchmarks/artifacts/pairwise_radial_cap_diagnostic_20260806/smoke_attempt01/result.json`
- `docs/benchmarks/artifacts/pairwise_radial_cap_diagnostic_20260806/smoke_attempt02_isolated_4080/result.json`

Status: `MECHANISM_SUPPORTED_REPLAY_VETO_UNRESOLVED`

## Outcome

The smooth per-particle radial cap works as implemented. Its manual total JVP
matches independent TensorFlow forward autodiff, it composes with explicit
Zhao-Cui squared-TT pairwise targets, and the post-cap particle RMS is strictly
below the declared cap. Mean and covariance restoration remain correct.

On Austria SIR, the cap is active and sharply limits the largest pairwise
correction direction. More importantly, it substantially reduces an unexpected
cross-process score instability. For the same particle seed `98201`, source
hash, code hashes, FP32/TF32/XLA controls, and RTX 4080 execution, the uncapped
score varied across three processes by approximately `(266.7,109.1,137.5)`.
At cap `2`, those coordinate ranges fell to `(6.7,5.7,4.5)`.

This is a promising mechanism result, not a cap selection or promotion. The
same-program replay failure is a hard numerical-validity concern for the
underlying Austria score route. The three-seed table is descriptive and cannot
support a stochastic ranking while cross-process replay remains unresolved.

## Claimed Target And Computed Quantity

| Item | Verdict |
|---|---|
| Claimed target | Total JVP of the finite reset/filter program after adding the declared smooth radial cap |
| Quantity computed | Raw empirical weighted pairwise third/fourth-moment targets followed by the existing correction, per-row radial cap, restandardization, and recursive finite likelihood score |
| Equality | Correct on FP64 reference/autodiff fixtures; cross-process FP32/TF32/XLA replay is not stable enough to establish numerical identity for Austria |
| Zhao-Cui role | An explicit squared-TT target fixture verifies composition; the Austria run itself uses empirical particle targets |
| Austria recursive Zhao-Cui teacher | Not executed because its existing recursive TT-fit veto fires before reset evaluation |
| Source classification | The radial cap is an `extension_or_invention`, not a Zhao-Cui source-faithful operation |

## Implemented Formula

For the globally normalized affine-projected pairwise direction `v_n`, the
candidate executes

```text
q_n^2 = mean_j(v_nj^2)
s_n   = (1 + q_n^2 / c^2)^(-1/2)
v_n   <- s_n v_n
```

and propagates the total tangent of both `v_n` and `s_n`. A zero cap control
selects the original route without applying this operation.

## Engineering Evidence

CPU-hidden reference tests:

```text
5 passed, 25 deselected
```

The selected checks include disabled-cap exact parity, strict RMS-bound
enforcement, affine mean/covariance restoration, generic capped manual-JVP
parity, a finite full-filter call, and capped composition with Zhao-Cui
squared-TT pairwise targets.

GPU evidence:

| Item | Result |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| TensorFlow | 2.19.1 |
| dtype / TF32 / XLA | FP32 / enabled / compiled |
| memory policy | growth verified before logical-device initialization |
| three-seed rows | `12/12` finite and program-valid |
| score-increment additivity | maximum observed residual `1.53e-5` |
| mean restoration | maximum observed residual `6.10e-5` |
| TensorFlow allocator peak | `72,429,568` bytes |
| three-seed wall time | `163.99 s` |

Primary artifact SHA-256:
`722e521d36f30b03933e55841e33b3982f1eef92f20ded20711ae9270eaa35c7`

## Three-Seed Descriptive Results

These rows use common seeds `98201..98203`. Sample SDs are descriptive only.

| Arm | Value mean (SD) | Score mean | Score SD | Maximum post-cap RMS | Minimum scale |
|---|---:|---:|---:|---:|---:|
| uncapped | `-681.9049 (0.5065)` | `(-33.23,-98.65,13.12)` | `(47.70,11.80,5.74)` | `26.887` | `1.000` |
| cap 8 | `-681.9274 (0.3611)` | `(-6.26,-117.70,10.71)` | `(25.78,7.53,1.79)` | `7.190` | `0.438` |
| cap 4 | `-682.2411 (0.4063)` | `(26.08,-136.78,17.12)` | `(40.21,32.08,6.79)` | `3.891` | `0.232` |
| cap 2 | `-682.1101 (0.4026)` | `(-15.51,-121.91,13.42)` | `(16.05,12.58,4.54)` | `1.992` | `0.092` |

The cap trades moment correction for bounded influence. Mean normalized
pairwise residual objectives were about `0.111--0.160` uncapped,
`0.136--0.167` at cap 8, `0.138--0.189` at cap 4, and `0.146--0.176` at cap 2.
Residual worsening is expected and prevents interpreting a lower score spread
as free improvement.

## Replay Diagnostic

Seed `98201` was evaluated in three separate processes. The first exposed both
GPUs but placed the graph on the 4080; the latter two exposed only the 4080.
The second and third processes still disagreed materially, so RTX 5080
visibility is not the complete explanation.

| Arm | Value range | Score-coordinate ranges | Post-cap RMS range |
|---|---:|---:|---:|
| uncapped | `0.1730` | `(266.66,109.08,137.52)` | `22.25--26.89` |
| cap 8 | `0.2337` | `(19.00,7.55,2.70)` | `5.98--7.19` |
| cap 4 | `0.2731` | `(58.78,27.23,6.23)` | `3.61--3.89` |
| cap 2 | `0.1201` | `(6.72,5.67,4.52)` | `1.93--1.99` |

The uncapped replay defect is much larger in the score than in the scalar
value, consistent with recursive derivative amplification. Cap 2 most
consistently suppresses it in this diagnostic. Cap 4 is not monotone in score
stability and therefore a smaller numerical cap is not automatically better in
every sense.

The historical July artifact for the same uncapped seed reported value
`-681.56897` and score `(3.84,-118.80,12.48)`, close to but not identical to
the new isolated replay `(2.26,-119.35,11.76)`. This reinforces that the replay
problem predates the cap rather than being created by it.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain radial cap as an opt-in diagnostic | formula, bound, affine restoration, and manual JVP pass | no engineering veto | exact influence/value tradeoff | preserve implementation and use it in replay localization | no default selection |
| Treat cap 2 as the leading replay-stability hypothesis | strongest cross-process score-range reduction | ranking not supported | only one repeated seed and three processes | compare cap 2, cap 8, and uncapped under a deterministic arithmetic/reference ladder | no superiority claim |
| Do not interpret the three-seed SD ranking | all rows finite but only three seeds | hard replay-validity concern | XLA/TF32/reduction sensitivity versus unstable nonlinear recursion | localize replay under FP64/no-TF32 and deterministic execution before a 16-seed claim | no score-accuracy claim |
| Keep recursive Austria Zhao-Cui blocked | its prior TT-fit validity veto was not repaired or bypassed | unchanged prior veto | whether a valid teacher representation can be built | treat this cap as reset-side evidence only | no Austria Zhao-Cui feasibility claim |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | cap implementation passes; Austria cross-process score replay fails |
| Statistically supported ranking | none |
| Descriptive-only differences | all three-seed means/SDs, residuals, values, scores, and runtimes |
| Viable candidates | uncapped remains the baseline; cap 8 and cap 2 remain diagnostic candidates |
| Default readiness | not established |
| Next evidence needed | deterministic/FP64/no-TF32 replay localization, then scope-specific cap tuning and the existing 16 common seeds only if replay passes |

## Post-Run Red Team

The strongest alternative explanation is not that the cap improves the true
score, but that it damps an already numerically unstable recursive tangent. The
data support damping: raw moment residuals worsen while replay score ranges
shrink. They do not establish reduced bias or closer agreement to an exact
Austria score.

The result would be overturned if a deterministic or FP64 reference showed
that the capped score is systematically farther from the derivative of its own
finite scalar, or if repeated stable processes showed no influence reduction.
The weakest evidence is score accuracy. The strongest evidence is the checked
cap algebra and the direct reduction of realized correction and replay tails.
