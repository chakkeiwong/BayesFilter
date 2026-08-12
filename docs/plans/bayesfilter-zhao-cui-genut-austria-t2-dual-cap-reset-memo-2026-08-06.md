# Zhao-Cui GenUT Austria T2 Dual-Cap Reset Memo

Date: 2026-08-06

## State

- Added an opt-in smooth coordinatewise cap after the final bounded-coordinate
  restandardization, with complete total JVP and support diagnostics.
- Existing radial correction-direction cap remains available and was crossed
  with the coordinate cap.
- Focused cap/JVP/restoration tests pass (`5 passed` in the targeted subset).
- Dual-cap T2 campaign passed on GPU/XLA with all 18 crossed validation rows
  finite and all nine FD coordinates passing.
- Selected scope-specific control: radial cap disabled, coordinate cap `b=0.98`,
  power `p=8`, diagonal strength `0`, pairwise strength `0.02`, four steps.
- Teacher-to-particle SD ratios were `0.0469--0.1001`, below the declared `0.5`
  screen for value and all three score coordinates.
- Post-run review found and corrected an explanatory moment-residual
  standardization defect. The successful artifact remains eligible for support,
  affine, FD, value/score, cap-activity, and teacher-sensitivity evidence, but
  its stored post-cap moment-residual fields are ineligible.

## Verdict

The dual-cap route is eligible as a T2 candidate for constructing and validating
teachers at later times. It is not a default and does not establish score
accuracy. The cap changes the heuristic higher-moment target; the result shows
finite, differentiable, teacher-robust behavior under the declared T2 scope.

## Required next work

1. Construct strict Zhao-Cui bounded teachers for each `T=3..20`.
2. Run focused per-time support/JVP/calibration checks before assembling T20.
3. Tune the dual-cap controls for the actual T20 scope using disjoint calibration
   and validation partitions.
4. Run untouched multi-seed T20 validation before considering proposal-training
   use. Do not infer T20 performance from T2.

## Nonclaims

No exact nonlinear Austria score, likelihood improvement, unbiasedness, posterior
correctness, HMC/NeuTra readiness, or repository default readiness is established.
