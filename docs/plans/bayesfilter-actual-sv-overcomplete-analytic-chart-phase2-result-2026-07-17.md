# Actual-SV Overcomplete Analytical Chart Phase 2 Result

Date: 2026-07-17

Status: `PASS_PHASE_2_IMPLEMENTATION`

## Implemented Route

The experimental Actual-SV lane now has:

- a nonthrowing diagonal-P Pearson projection using one `q x q` Cholesky solve;
- explicit single- and multi-direction JVPs and an explicit VJP;
- TensorFlow-only weighted-quantile, duplicate-fill, Voronoi, and Pearson
  preparation;
- a separate fixed-capacity forward recursion with `[T-1,K]` prepared tensors;
- a complete two-direction manual total JVP for Actual-SV dynamics, affine LEDH
  proposal, target corrections, continuation recursion, normalized fourth
  feature, projection, and terminal contribution; and
- XLA-default forward, manual-score, and independent autodiff-oracle factories.

The candidate projection never constructs or solves a dense `K x K` precision
matrix.  The filter and continuation recursions use `tf.while_loop`.  Historical
square, dense-P, and Python-unrolled functions were not redefined.

## Verification

Focused command:

```text
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 pytest -q tests/highdim/test_ledh_contract_e_tp_diagonal_kkt.py tests/highdim/test_ledh_contract_e_tp_actual_sv_overcomplete.py tests/highdim/test_ledh_contract_e_tp_primitives.py tests/highdim/test_ledh_contract_e_tp_derivatives.py tests/highdim/test_ledh_contract_e_tp_scalar_sv_loop.py
```

Result: `40 passed` in 45.02 seconds.  The two warnings are pre-existing
TensorFlow Probability deprecation warnings.

The tests cover dense-KKT equality, exact features, square-limit equality,
invalid fail-closed behavior, JVP/autodiff, VJP/autodiff, vectorized JVP versus
single directions, teacher and continuation tangents with moving points,
complete recursive manual score, historical regressions, source-call-graph
guarding, and compiled functional loops.

At the frozen `T=2,K=5` smoke, manual versus autodiff score differed by at most
`4.54081217071689e-14`.  At the `T=10,K=7` CPU-XLA smoke, it differed by at most
`6.397105067890152e-13`, with symmetric relative difference
`7.457627472995128e-13`.  The route-scoped source guard passed with no reachable
Python loop, and the concrete graph contains functional loops.

## Phase 3 Observations Already Preserved

While obtaining an executable `T=10` implementation witness, deterministic
center preparation showed:

- `K=5`: negative reference at time 2;
- `K=6`: negative reference at time 2;
- `K=7`: center preparation and all nine design points pass at `T=10`;
- `K=7..23`: center preparation passes at `T=10`; and
- `K=24` and `K=25`: zero Voronoi/reference mass at time 0, violating strict
  positivity.

These are capacity observations, not implementation-test outcomes.  They are
preserved under `phase-03-capacity` and do not yet select `K=7`, because the
`T=100` and `T=1000` design gates remain.

The frozen endpoint pilot cannot produce warm evaluation timings at `K=5` or
`K=25` because both endpoint preparations fail mathematically before runtime.
This is a pilot-design limitation, not a timeout or implementation failure.
Phase 3 therefore uses the measured executable `K=7` graph plus conservative
unamortized per-horizon costs and must remain within the original five
core-hour budget; it does not reinterpret endpoint failure as timing data.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Pass Phase 2 | Specialized value/JVP route is correct against dense and autodiff oracles and compiles with functional loops | No implementation veto fired | Long-horizon chart feasibility remains unknown | Continue staged `K=7` design ladder | No held-out, full-horizon, scientific-equivalence, GPU, HMC, canonical, or leaderboard claim |
