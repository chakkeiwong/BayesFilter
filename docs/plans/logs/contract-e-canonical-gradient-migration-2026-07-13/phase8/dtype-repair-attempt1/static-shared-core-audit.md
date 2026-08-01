# Rung 0A Static Shared-Core Audit

Date: 2026-07-14

Verdict: `PASS`

- Both factory dtypes call `canonical_value_and_score_core`, which calls the
  same `_canonical_primal_core` and `_canonical_manual_jvp_core`.
- `dtype` changes the input signature, prepared tensor dtype, constants, casts,
  tensor arrays, and zero/fill tensors only.
- AST inspection found no dtype-dependent conditional in the canonical,
  streaming-composition, or cloud-reset cores. The canonical module has only an
  input dtype allow-list. The shared annealed-transport dependency has one
  compatibility conditional that casts a non-floating epsilon to its historical
  default; both canonical float32/float64 routes pass floating epsilon and take
  the same branch.
- There is no dtype-conditioned algorithm, Sinkhorn cadence, reset policy,
  derivative composition, or active-set policy.
- The primal and manual-JVP factory graph count is one for each checked dtype.
- The static result is supported, but not replaced, by the source-bound
  float64 and float32 execution certificates in this directory.

This audit does not establish large-shape numerical adequacy, GPU behavior,
Kalman equivalence, formal FD promotion, or HMC readiness.
