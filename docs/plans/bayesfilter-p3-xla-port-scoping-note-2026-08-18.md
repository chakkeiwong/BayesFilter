# P3 XLA Port — Scoping Note (pre-implementation) — 2026-08-18

Status: `SCOPING` (read-only survey; no code changed). Plan phase: P3
(revision 4, Section 6). Gate: eager-vs-XLA parity 1e-12 on value AND
score; memory-growth policy (V11); throughput recorded without
feasibility language (V12).

## Current state (surveyed 2026-08-18)

No `tf.function`/`jit_compile` anywhere in the highdim engine stack.
The compute kernels are XLA-friendly (einsum chains: 18 in
retained_quadratic_form_tf, 7 in the adjoint engine; QR/Cholesky/
triangular_solve in the fitters), but the ENGINE LOOPS are Python-side
with per-step host syncs.

Host-sync / graph-break inventory (must move out of any compiled scope
or become tensor-side):

1. `_fixed_als_fit(_traced)`: `float(solve.scaled_augmented_condition_number)`
   per update + veto `raise` (squared_tt_engine_v0_tf.py:186/245).
   XLA route: compute condition tensor-side, accumulate a status tensor,
   check once per step outside the compiled fit (fail-closed preserved).
2. Value engine diagnostics: `float(...numpy())` per step
   (log_increment, gram_condition, weighted_fit_rms). XLA route: return
   stacked tensors, materialize at the artifact boundary (backend rule
   compliant).
3. `RetainedQuadraticForm.__post_init__` asserts call `.numpy()`
   (retained_quadratic_form_tf.py:231-241). XLA route: a compiled inner
   step must bypass dataclass construction (carry (prefix cores, E, Zc)
   as a tensor tuple) and construct the checked object only at
   host boundaries.
4. Adjoint engine: eigvalsh veto `float(...)` (line ~186) plus the trace
   list-of-dicts. XLA route: compile per-update kernels (design build,
   scaled solve, adjoint nodes) first; keep the sweep loop in Python —
   the P2A cost profile showed per-update linear algebra dominates.
5. `_gauss_rows` uses numpy at setup: frozen inputs, diagnostic-only
   exception — acceptable as precomputed constants fed into compiled
   fns (no runtime numpy in compiled paths).

## Recommended port order (smallest-diagnostic-first)

- P3.1: DONE (2026-08-18): `jit_compile` solve-kernel parity probe
  (`docs/benchmarks/run_p31_xla_solve_parity_20260818.py`, log
  `/tmp/p31_xla_parity.log`): worst rel vs `_solve_scaled_augmented_ridge`
  1.5e-14 over 6 fixtures incl. column-scale 1e-6..1e-8 arms; eager and
  XLA kernels agree at the same level; 0 retraces over 5 same-shape
  calls. Gate 1e-12 PASS with two orders of margin — the QR-lowering
  risk did not materialize at the solution-vector level.
- P3.2: DONE (2026-08-18): design assembly + Gram chains compile AS-IS
  (`docs/benchmarks/run_p32_xla_design_gram_parity_20260818.py`, log
  `/tmp/p32_xla_parity.log`): `FixedTTFitter._build_design_matrix` on the
  mixed transition basis (incl. the discrete branch axis) at core
  indices {first, branch, last}: worst XLA rel 9.4e-16;
  prefix/suffix Gram chains ~1.5e-16. Gate 1e-12 PASS; no compile
  failures — the repo's compute kernels need no rewrite, so P3.3
  reduces to hoisting the inventoried host syncs out of the compiled
  step.
- P3.3: whole-step compiled fit (fit + retention tensors), Python
  filter loop retained; value parity 1e-12 vs eager on n in {1,2}
  fixtures + one ladder cell.
- P3.4: adjoint node kernels (same order as the reverse sweep);
  score parity 1e-12 via I-P2-4-style cross-check under XLA.
- Wall/RSS measured per rung on the SAME fixture as the T=120 stress
  (artifact-first; the n=4 ladder feasibility decision consumes these
  numbers).

## Risks

- QR under XLA: `tf.linalg.qr` lowering differences may move solutions
  at the 1e-14..1e-13 level; the 1e-12 parity gate absorbs this, but
  bit-identity claims (value-engine equality asserts at 1e-12) must be
  re-checked, not assumed.
- Dynamic branch_count (boundary_rank+1 changes between t=0/t>=1 and
  potentially across steps if ranks vary): compile per distinct shape
  signature; count retraces (V12 telemetry).
- float64 XLA on GPU is slow on consumer devices; CPU XLA and GPU
  float64 both need measured walls before any claim (V12). TF32/f32
  routes are OUT OF SCOPE for P3 (dtype is part of scope identity).
