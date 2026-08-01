# Contract E--TP Clean-XLA Loop Repair Result

metadata_date: 2026-07-15
status: PASS_CLEAN_XLA_COMPILE_EXPLOSION_REPAIRED
plan: `docs/plans/bayesfilter-contract-e-tp-clean-xla-loop-repair-plan-2026-07-15.md`
artifact_root: `docs/benchmarks/artifacts/contract_e_tp_clean_xla_loop_repair_20260715/`

## Verdict

The XLA compilation issue is solved for the frozen float64 LGSSM Contract E--TP
`finite_lookahead=8` route. The compiled factory now uses one batched fixed-lag
`tf.while_loop` and one fixed-shape filtering `tf.while_loop`, with static
`T=1,2` edge dispatch. `T=10` and `T=50` produce the same graph-body size and
compile in seconds rather than the prior `T=50` 28-minute static unroll.

The repair is execution-equivalent to the same finite scalar and total score
within machine-roundoff-scaled bounds; bitwise equality is claimed only where
reported below. It does not repair the remaining float32/TF32 or nonlinear-
model XLA gaps, and it does not change Contract E--TP's experimental admission
status.

## Implementation

- `_finite_lookahead_information_parameters_loop` evaluates all starting
  windows in one batched information recursion with a maximum of eight
  functional-loop iterations and terminal masks.
- `contract_e_tp_lgssm_finite_lookahead_loop_core` executes a separate initial
  shape-changing step, one fixed-shape intermediate functional loop, and a
  separate terminal no-projection step.
- `make_contract_e_tp_lgssm_score_informed_recursive_tf` now selects the loop
  core for finite-lookahead execution. The historical unrolled function remains
  as a parity oracle, not the XLA factory route.
- Cartesian parent/innovation points and weights now use broadcasting and
  reshape rather than dynamic `repeat`/`tile`, preserving parent-major ordering
  while making the reverse loop XLA-compilable.
- Static source and graph tests require functional loops and reject Python
  time/window unrolling in the clean route.

No NumPy or SciPy computation entered the gradient-bearing runtime. SciPy
remains confined to offline chart preparation.

## Numerical Parity

The loop and unrolled finite programs were compared directly.

| Check | Result |
| --- | --- |
| Fixed-lookahead values | matrices exact; vectors within `8.9e-16` through `T=10` and `4.4e-16` at `T=50` |
| Fixed-lookahead aggregate Jacobian | maximum absolute difference `9.1e-13` at `T=50` |
| Full `T=50` objective | exact equality |
| Full `T=50` total score | maximum absolute difference `4.0e-14` |
| Full `T=50` increment history | maximum absolute difference `2.2e-15` |
| Full `T=50` final log weights | maximum absolute difference `2.31e-12` |
| Validity history | exact equality |

The permanent focused suite passed `36 passed, 2 warnings`. The warnings are
TensorFlow Probability `distutils` deprecations.

## Graph Topology

The controlling final-source graph audit is `graph_audit_final.json`, SHA-256
`c3765e31ce0a1b8ee5cf4523ef141f3c1a86555b87c0669d34fbf3736549b7b3`.

| Metric | `T=10` | `T=50` | Ratio |
| --- | ---: | ---: | ---: |
| Top-level nodes | `4,014` | `4,014` | `1.0` |
| Function-library nodes | `3,712` | `3,712` | `1.0` |
| Function count | `36` | `36` | `1.0` |
| Functional `While` operations | `4` | `4` | `1.0` |
| GraphDef bytes | approximately `1.24 MB` | approximately `1.24 MB` | `0.9992` |

All predeclared topology gates pass. The audit was regenerated after the
dynamic-repeat and `T=2` edge repairs, so these counts correspond to the final
source used for closeout.

## Trusted GPU/XLA Results

| Horizon | Old compile + first | New compile + first | Compile reduction | Old warm | New warm | Warm ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `T=10` | `51.2585 s` | `11.6156 s` | `4.41x` | `0.2330 s` | `0.4329 s` | `1.86x` slower |
| `T=50` | `1694.9407 s` | `14.1293 s` | `119.96x` | `1.5158 s` | `2.7621 s` | `1.82x` slower |

The `T=50/T=10` new compile-plus-first ratio is `1.22`, compared with the old
`33.07`. Both outputs are GPU-backed, XLA-compiled, finite, chart-valid, and
agree with their controlling CPU results at roundoff:

- `T=10` maximum score difference `5.33e-14`, artifact SHA-256
  `f450f5dd214f67253659bac6cf164f03a1352ffe8391fc536bd43bb32880caf0`;
- `T=50` maximum score difference `2.32e-13`, artifact SHA-256
  `5b1810a2505960659bece7072b4aa7e2106dfcb2c8b4f6c3736743e1ce0a618b`.

The loop route trades some warmed throughput for bounded compilation. `ptxas`
still reports approximately 1.7 KB of register spill traffic in one loop fusion.
That is a future kernel-throughput diagnostic, not evidence that compile graph
explosion remains.

## Compiled Fail-Closed Result

The controlling negative artifact is
`invalid_chart_gpu_attempt2/result.json`, SHA-256
`f745e5747ad6b45905b9a4dc7f089dfa347f26b88115ce828f460fc9901e3759`.

A negative row scale made the input chart invalid without relying on an eager
assertion. Compiled XLA returned `valid_history=[false,false]`; final particles,
objective, and score were all nonfinite through numerical poisoning. The
compiled route therefore remains fail-closed when XLA ignores assertion
operators.

## Attempt And Repair Record

1. Offline `T=2,3` chart generation attempt initially failed because the caller
   pre-created a directory that the non-overwrite script owns. Fresh per-rung
   directories repaired the artifact layout without changing charts.
2. Trusted `T=10` attempt 1 failed during XLA conversion because reverse
   autodiff of dynamic `tf.repeat` generated a reshape whose shape was not a
   compile-time constant. Broadcasting plus reshape repaired the same Cartesian
   product. CPU parity passed before retry.
3. Trusted `T=10` attempt 2 passed.
4. Trusted `T=50` attempt 1 passed.
5. Invalid-chart attempt 1 exposed a zero-iteration `T=2` reverse TensorList
   compiler defect. Static `T=2` dispatch now omits the intermediate loop, as
   predeclared by the plan. CPU parity passed before retry.
6. Invalid-chart attempt 2 passed.

All attempts remained within the reviewed scientific target, hardware class,
and compute/attempt budget. Prior artifacts were not overwritten.

## Decision And Inference Status

| Item | Status |
| --- | --- |
| Same finite scalar and total score | execution-equivalent within roundoff; objective exact in the `T=50` direct comparison |
| Python-unrolled XLA time/window route | removed from the finite-lookahead factory |
| Functional-loop graph gate | pass |
| Trusted GPU/XLA compilation | pass at `T=10,50` |
| Compile explosion | repaired for frozen LGSSM float64 route |
| Warmed throughput | slower; descriptive tradeoff, not optimized |
| Compiled fail-closed gate | pass |
| Float32/TF32 production readiness | blocked, not tested |
| Nonlinear-model clean-XLA readiness | blocked, not tested |
| Canonical/default/leaderboard/HMC readiness | false |

## Execution Review And Red Team

The strongest risk is that the new loop and old unrolled route share the same
mathematical error. This topology repair therefore uses old/new parity only for
implementation identity; the independent Kalman comparisons remain the
scientific oracle and are unchanged.

The strongest alternative explanation for the compile improvement is compiler
cache reuse. It does not explain the result: `T=10` and `T=50` ran in separate
fresh processes and each recorded its own compile-plus-first call, while the
static graph audit independently shows constant graph bodies.

The weakest performance result is warmed runtime, which regressed about 1.8x.
The compile problem is solved, but future throughput work should inspect loop
fusion and register spills without reintroducing horizon unrolling. No claim of
overall speed superiority is made.

No float32/TF32, nonlinear full-horizon, HMC, canonical, default, or leaderboard
readiness follows from this repair.
