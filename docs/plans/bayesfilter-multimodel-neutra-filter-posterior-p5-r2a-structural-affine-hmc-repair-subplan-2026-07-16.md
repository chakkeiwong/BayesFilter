# P5 R2A Subplan: STR-UKF Mode/Hessian Affine HMC Repair

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `REVIEWED_READY_FOR_EXECUTION`

## Objective And Entry Conditions

Construct one checked target-bound affine coordinate system for typed
`STR-UKF` identity
`e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`,
then repeat fixed-kernel nomination and sequential plain HMC with fresh seeds.

Entry evidence is the admitted R1B identity and the preserved source-coordinate
attempt at `phase-p5/STR-UKF/plain-hmc/attempt-02`. That attempt rejected its
selected kernel after one sequential-warm-up energy-error divergence. It did
not hit the R-hat cap and produced no retained sample. All source warm-up is
tuning-only and excluded from inference.

## Evidence Contract

| Field | Frozen R2A contract |
| --- | --- |
| Question | Does a checked target-bound posterior-mode/Hessian affine chart make the unchanged structural UKF posterior sampleable under the original comparator gates? |
| Baseline | rejected source-coordinate fixed kernel from R2; it is a geometry diagnostic, not posterior evidence |
| Geometry start | pooled mean of finite source warm-up, used only as a mode-locator start hypothesis |
| Geometry authority | central finite difference of the exact batch-native source score at a checked local posterior mode; steps `1e-4` and `5e-5` |
| Geometry admission | finite/status-valid mode and stencils; score infinity norm `<=1e-4`; raw negative Hessian positive definite; two-step relative Hessian gap `<=1e-3`; affine round-trip/value/score gaps `<=1e-10` |
| Coordinate map | `theta = center + z @ factor.T`, where `factor=chol(inverse(negative Hessian))`, including the constant log-absolute determinant |
| Kernel nomination | among finite health-valid probes, maximize minimum rank-normalized bulk ESS in source coordinates; grid order breaks ties |
| Comparator pass | separate fresh warm-up; recent-window modern R-hat `<=1.05`; retained modern R-hat `<=1.01`; minimum bulk ESS `>=1000`; minimum tail ESS `>=400`; all health/status gates clear |
| Hard vetoes | identity/hash drift, invalid mode/stencil, non-SPD or unstable Hessian, affine chain-rule failure, nonfinite/status/energy/movement failure, or 10,000-draw cap |
| Explanatory only | acceptance, short-probe diagnostics, mode location, Hessian spectrum, runtime, truth distance, posterior summaries |
| Not concluded | NeuTra quality, structural UKF exactness, truth recovery, calibration, superiority, robustness, or readiness |

Modern R-hat remains the maximum of rank-normalized split and folded
rank-normalized split R-hat. Failed source warm-up is never pooled with the
fresh affine warm-up or retained sample.

## Runtime Design And Defaults

- Mode locator: at most eight damped Newton iterations, infinity-norm trust
  radius `1.0`, line multipliers `(1, 0.5, 0.25, 0.125, 0.0625, 0)`.
- Stagnation may stop only with step infinity norm `<=1e-5`, nonnegative value
  change within `1e-8`, and the terminal mode/curvature gates still passing.
- HMC: four chains initialized at fixed dispersed offsets in the affine `z`
  chart; eight leapfrog steps; step grid `(0.05,0.10,0.20,0.30,0.40,0.50)`.
- Each probe has 64 burn-in transitions and 128 draws.
- Probe root `(20260716,18100)`, warm-up root `(20260716,18201)`, retained root
  `(20260716,18301)`.
- Warm-up chunks/minimum/window/maximum are `1000/2000/1000/10000` per chain.
- Retained chunks/minimum/maximum are `2000/4000/10000` per chain.
- GPU/XLA, float64 target arithmetic, TF32 enabled, and TensorFlow memory growth
  remain unchanged.

The finite-difference steps and Newton settings are inherited from the admitted
P4 SGQF geometry procedure and are hypotheses here, not universal defaults.
The two-step Hessian gate is the early diagnostic for step-size sensitivity.
The raw-SPD and terminal-score gates prevent a regularized saddle from being
misrepresented as a posterior mode.

## Artifacts And Checks

1. Verify both recursive source ledgers, reconstruct the typed identity, and
   replay independent posterior recomposition.
2. Write the full locator ledger, both terminal Hessian stencils, raw spectrum,
   covariance/factor, affine checks, manifest, and recursive hashes under a
   fresh geometry root.
3. Only if geometry passes, load it through its recursive hash ledger and run
   fresh probes and the shared sequential controller under a separate fresh
   HMC root.
4. Preserve latent and source-coordinate warm-up/retained chunks separately.
5. Run focused unit tests and recursive ledger verification after execution.

## Handoff And Stops

On comparator pass, move only `STR-UKF` to `COMPARATOR_ADMITTED`, write the R2
result, and draft target-specific dense-IAF training. If geometry fails, record
`COMPARATOR_BLOCKED_GEOMETRY` and do not sample. If fresh affine HMC fails a
health or convergence gate, record `COMPARATOR_BLOCKED`; do not relax gates or
reuse warm-up. R2 stops after this one affine repair, three identical
infrastructure failures, or the remaining six-GPU-hour comparator budget.

## Skeptical Pre-Execution Audit

Decision: `PASS_AFTER_REVISION`.

The first draft would have used the divergent warm-up covariance and would
have described the source failure as an R-hat geometry failure. That was not
supported by the artifact. This revision uses the failed archive only for a
mode-start hypothesis, requires an independently evaluated terminal Hessian,
and records the actual energy-error veto. The target, posterior, thresholds,
and caps do not change. Short probes nominate only; only fresh sequential
health, warm-up, retained convergence, and ESS can admit the comparator.

