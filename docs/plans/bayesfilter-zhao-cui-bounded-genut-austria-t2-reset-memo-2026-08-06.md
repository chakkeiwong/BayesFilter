# Zhao-Cui Bounded GenUT Austria T2 Reset Memo

Date: 2026-08-06

## Current state

- Four independent 128-sample T1/T2 Zhao-Cui bounded teachers exist under
  `docs/benchmarks/artifacts/zhao_cui_bounded_genut_austria_t2_crossed_validation_20260806/`.
- They strictly reload, use eight distinct teacher time seeds, bind one issuer
  identity set, and have ESS near `127.7/128`.
- The filter now exposes raw and scale-normalized post-teacher physical affine
  mean/covariance residuals plus maximum corrected bounded coordinate.
- Focused affected tests passed before the campaigns: `41 passed`.
- Fixed-diagonal calibration failed all 18 rows at the bounded-chart veto.
- Joint diagonal/pairwise calibration failed all 72 rows at the same veto.
- No validation teacher or validation particle seed was evaluated. The planned
  3-teacher by 6-particle sensitivity comparison remains untouched.
- T20 remains unavailable because valid teachers T3 through T20 do not exist.

## Scientific verdict

The current active bounded-coordinate correction is wrong as an admissible
finite program for this calibration scope because it can produce `|u|>=1`
before an inverse map that is defined only on `(-1,1)`. The exact invalid graph
is correctly vetoed; clipping is only a finite failure-path device and must not
be treated as a result.

This rejects the tested correction composition, not the independent Zhao-Cui
teacher artifacts or the idea of teacher-controlled higher moments. The next
repair must preserve compact support by construction and propagate its total
JVP. Do not rerun broader seeds, validation, T20, NeuTra, or HMC until that
engineering/mathematical gate passes.

## Next smallest discriminating work

1. Derive a boundary-preserving update, with `tanh(atanh(u)+delta)` as the first
   candidate and explicit treatment of near-boundary conditioning.
2. Add FP64 reference and TensorFlow forward-accumulator parity tests for its
   total JVP, disabled-update identity, support preservation, and physical
   affine restoration.
3. Run a tiny T2 GPU/XLA calibration smoke on the existing calibration teacher
   and particles only.
4. If a nontrivial candidate passes, create a fresh tuning plan and then consume
   the still-untouched validation teachers/seeds under the existing `0.5`
   teacher-to-particle SD criterion.
5. Only after T2 promotion, construct and validate teachers T3 through T20.

## Nonclaims

There is no T2 value/score improvement result, no teacher-sensitivity result,
no exact Austria score, no T20 evidence, and no HMC/NeuTra or default readiness.
