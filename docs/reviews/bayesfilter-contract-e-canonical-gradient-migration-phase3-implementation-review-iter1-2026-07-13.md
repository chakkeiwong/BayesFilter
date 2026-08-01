# Phase 3 Implementation Review, Iteration 1

Date: 2026-07-13

Reviewer: fresh bounded Codex substitute.

## Finding

The accepted exact fixtures used identity or scalar-identity Cholesky/affine
maps. They could not expose noncommuting order or transpose errors in Cholesky,
triangular-solve, affine JVP, or affine VJP paths. The nonidentity fixture checked
only finiteness and positive factor diagonals.

No formula defect was found by direct inspection.

## Verdict

`VERDICT: REVISE`

## Repair

An independently derived exact-rational noncommuting certificate is frozen
before its repair execution. It has distinct noncommuting target/injected
Cholesky factors, a nonsymmetric nonidentity affine map, full per-input JVP/VJP
expectations including ridge, and an exact duality scalar.
