# P5 R2A Result: STR-UKF Affine Geometry Gate

Date: 2026-07-16

Status: `COMPARATOR_BLOCKED_GEOMETRY`

## Decision

The target-bound mode/Hessian geometry attempt is rejected because the mode
locator reached its eight-iteration cap with score infinity norm
`0.0050403880`, above the predeclared `0.0001` gate. The affine HMC runner was
not launched.

The terminal point was finite and status-valid. Its two finite-difference
Hessians were stable, the raw negative Hessian was positive definite without
eigenvalue clipping, and the affine wrapper checks passed. Those facts make the
geometry diagnostically promising, but they do not override the failed mode
criterion.

## Binding Evidence

- Artifact root:
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/affine-geometry/attempt-01`
- Result SHA-256:
  `0ee609c414673d3dc3f797aa135ae2de349cec5be39df18241b6de584a5f12d9`
- Recursive ledger SHA-256:
  `4ac945f4dcaa86d8551ebcf820297195998607ff8a859b5270cd552189059ada`
- Target signature:
  `e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665`
- Trusted GPU/XLA wall time: `45.5263` seconds.
- Terminal score infinity norm: `0.0050403880`; required `<=0.0001`.
- Terminal Hessian relative step gap: `3.53828e-7`; required `<=0.001`.
- Raw negative-Hessian eigenvalues:
  `(1.93994, 5.54851, 13.51241, 48.64731, 293.75120)`.
- Raw precision condition number: `151.423`; no eigenvalue was clipped.
- Affine round-trip gap: `1.80411e-16`.
- Affine value and score chain-rule gaps: zero.

## Interpretation

The claimed target was the frozen five-probit structural UKF posterior. The
quantity computed was its exact batch-native source score and a central
finite-difference Hessian at the locator's terminal point. The point is not an
admitted posterior mode under the plan because its score failed the frozen
tolerance. Therefore the computed covariance is heuristic tuning evidence only
and is not an admitted mass matrix, posterior covariance, or HMC comparator.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| block R2 comparator and close `STR-UKF` locally | terminal mode score failed | geometry admission veto fired; target/status/curvature checks otherwise clear | whether a larger predeclared locator budget would converge without changing the scientific target | continue independent P6; any future P5 re-entry starts with a refreshed geometry plan | comparator validity, HMC convergence, NeuTra quality, filter exactness, calibration, robustness, readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | failed at the terminal mode-score gate |
| Statistically supported ranking | none |
| Descriptive-only differences | score trajectory, curvature spectrum, condition number, runtime |
| Default readiness | not established |
| Next evidence needed | a prospectively reviewed geometry repair and fresh HMC; no current P5 budget is silently extended |

## Post-Run Red Team

The strongest alternative explanation is that eight Newton iterations were too
few and the stable terminal curvature is already adequate for HMC. That is
plausible but untested. Launching HMC anyway would convert an explanatory
diagnostic into an admission criterion after seeing the result, so the program
does not do that. The result weakens only this frozen affine repair, not the
admitted target identity or the broader research direction.

