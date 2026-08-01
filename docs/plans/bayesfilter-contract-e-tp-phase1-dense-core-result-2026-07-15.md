# Contract E--TP Phase 1 Dense Core Result

metadata_date: 2026-07-15
phase: 1
status: PASS_PHASE1_DENSE_CORE
master_plan: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`

## Outcome

The experimental dense TensorFlow projection core is implemented in
`bayesfilter/highdim/ledh_contract_e_tp_tf.py`. It accepts model-owned finite
teacher points, unnormalized log weights, and feature values. It has no teacher
construction, model assumptions, active-set search, clipping, or canonical
Contract E wiring.

The core provides:

- stable log-weight normalization and dense feature reduction;
- a frozen square chart with the explicit tangent solve;
- a separately selected overcomplete equality-constrained KKT chart;
- manual JVP and VJP implementations with separate adjoints for teacher
  points, log weights/correction factors, feature values/anchor features,
  reference weights, and KKT precision;
- condition-number and dtype-roundoff diagnostics; and
- fail-closed mass-feature, finite-input, unique-index, rank, residual,
  positive-weight, and KKT-precision checks.

The public wrappers default to `tf.function(jit_compile=True)`. XLA execution
and blockwise storage are Phase 2 gates, not claims from this phase.

## Evidence Contract Result

| Question | Result |
| --- | --- |
| Does the owned core reproduce the independent 6561-to-7 LGSSM witness? | Pass: weights, feature identity, condition number, and minimum weight reproduce the frozen artifact. |
| Does the same differentiable primal preserve retained-feature tangents and the next finite value/score? | Pass: manual JVP, TensorFlow forward AD, reverse AD, and same-program FD agree. |
| Are all declared derivative owners represented? | Pass at the generic finite-teacher boundary: points, log weights/corrections, features including anchor dependence, reference weights, and KKT precision. Model-specific parent/innovation ownership remains for adapters. |
| Does a failed chart silently change the target? | No: duplicate, rank-deficient, negative square, and negative KKT fixtures raise. No clipping or runtime active-set selection exists. |

## Numerical Evidence

For the frozen two-dimensional LGSSM witness:

| Diagnostic | Value |
| --- | ---: |
| Active indices | `(108, 221, 2317, 2402, 2474, 3942, 4001)` |
| Minimum student weight | `0.00029612211860157757` |
| Scaled feature-matrix condition number | `84.26064554729527` |
| Feature-value maximum residual | `1.304512053934559e-15` |
| Feature-tangent maximum residual | `4.052314039881821e-15` |
| Student-minus-teacher total value | `0.0` |
| Student-minus-teacher total score maximum residual | `1.1102230246251565e-16` |
| Largest AD versus centered-FD residual | `1.827806100918039e-10` |

The implementation residual gate is derived from matrix condition, dtype
epsilon, and the standard `gamma_n` roundoff factor. The existing
`0.05*sqrt(p)` rule appears only in the same-scalar directional FD test; it is
not used for cross-method or oracle agreement.

## Attempt And Repair Record

Attempt 1 produced three engineering test failures:

1. TensorFlow's zero-tolerance `assert_near` uses a strict predicate and
   rejected even exactly displayed symmetric matrices.
2. A TensorFlow gather adjoint was represented as `IndexedSlices`, while the
   test passed that sparse object directly to NumPy.
3. The KKT derivative test was consequently stopped by the same symmetry
   assertion before derivative comparison.

The symmetry check was replaced by a dtype-roundoff bound, and tests now
densify sparse TensorFlow adjoints before comparison. Neither repair changed
the projection, positivity/rank criteria, derivative formulas, or scientific
target. Attempt 2 passed.

## Checks Actually Run

CPU-only choice: `CUDA_VISIBLE_DEVICES=-1` was set before TensorFlow import.

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-contract-e-tp pytest -q \
  tests/highdim/test_ledh_contract_e_tp_primitives.py \
  tests/highdim/test_ledh_contract_e_tp_derivatives.py
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-contract-e-tp python \
  docs/benchmarks/contract_e_score_aware_teacher_projection_2d_lgssm.py
CUDA_VISIBLE_DEVICES=-1 python -m compileall -q \
  bayesfilter/highdim/ledh_contract_e_tp_tf.py \
  tests/highdim/test_ledh_contract_e_tp_primitives.py \
  tests/highdim/test_ledh_contract_e_tp_derivatives.py
git diff --check -- <Phase 1 paths>
```

Results: `11 passed`; independent witness `PASS`; compilation and diff hygiene
passed. TensorFlow emitted CUDA plugin-registration startup messages despite
the explicit CPU-only setting; they are not GPU evidence.

## Decision And Handoff

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 1 | Pass | No active chart, derivative, or target veto | Streaming transpose and XLA behavior not yet checked | Implement Phase 2 streaming reductions and dense parity | No recursive, nonlinear, GPU, canonical, leaderboard, or HMC validity |

Phase 1 gate: `PASS`. Phase 2 may add blockwise teacher reduction and its
transpose/JVP composition. It must preserve the same finite target and may not
wire the experimental route into models, the leaderboard, canonical Contract
E, or HMC.
