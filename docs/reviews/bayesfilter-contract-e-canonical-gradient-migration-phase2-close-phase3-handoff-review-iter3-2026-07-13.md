# Phase 2 Close And Phase 3 Handoff Review, Iteration 3

Date: 2026-07-13

Reviewer: fresh bounded Codex substitute.

## Findings

1. The zero-`Y+`, zero-`Xi`, zero-output fixture annihilated covariance,
   Cholesky, solve, affine, and ridge derivative paths.
2. Dense residual-noise preparation contains non-dyadic `sqrt(8/7)`, so it could
   not be included in a symbolic dyadic-exact claim.
3. Tangent/cotangent property descriptions did not freeze exact arrays or
   expected path contributions before implementation.

## Verdict

`VERDICT: REVISE`

## Repair

The primary fixture now uses `Xi=R`, giving `particles=R` and active internal
derivative paths. Exact rational tangents, cotangent, JVP/VJP arrays,
intermediate contributions, and duality scalar are frozen in a machine-readable
preimplementation certificate. Dense noise preparation is separately classified
as an executed float64 replay identity rather than symbolic dyadic arithmetic.
