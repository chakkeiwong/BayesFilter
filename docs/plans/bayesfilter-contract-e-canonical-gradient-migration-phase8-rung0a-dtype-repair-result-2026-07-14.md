# Phase 8 Rung 0A Result: Canonical Full-Filter Dtype Repair

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status:
`DTYPE_SHARED_CORE_REPAIR_PASSED_FORMAL_PHASE1_FD_SUBGATE_INCONCLUSIVE_PRIMARY_SHAPE_BLOCKED`

## Outcome

The canonical full-filter LGSSM graph now supports `tf.float32` and `tf.float64`
through the same primal and manual-JVP cores. `tf.float64` remains the
backward-compatible factory default. The repair removed unconditional float64
casts only from the streaming helpers reached by this canonical route and did
not add a dtype-conditioned algorithm, cadence, reset rule, branch policy, or
derivative composition.

The observed float32 score mismatch was an operation-order defect in the manual
JVP, not a missing mathematical derivative. TensorFlow's forward derivative of
the final affine flow assembly grouped the two matrix-product tangents before
adding the post-mean tangent. The manual path grouped the first matrix term with
the post-mean tangent. A second mismatch came from algebraically equivalent
division/quotient-rule order in geometry and the mandatory row quotient. These
forms differ by float32 rounding and the reset carries the difference into the
next time step. Shared, dtype-independent operation orders were found that make
the complete checked score bitwise equal in both dtypes while leaving the
finite primal unchanged.

## Evidence

| Check | Result |
| --- | --- |
| Focused canonical full-filter tests | `11 passed`; float32 and float64 per-batch and aggregate manual JVP equal forward autodiff bitwise |
| Shared streaming/reset regression tests | `30 passed` |
| Fresh float64 exact derivative | `ZERO_ULP_SAME_PRIVATE_PRIMAL_CORE_PASSED` |
| Float64 CPU-XLA | finite, repeatable, branch-identical, one concrete callable |
| Phase 5 v2 preservation | objective hex, all five score hex values, and branch hash reproduce exactly |
| Float32 CPU-XLA | finite, repeatable, branch-identical at every checked endpoint, one concrete callable |
| Float32 explanatory FD | all three frozen Phase 5 fixture steps pass `0.05*sqrt(5)`; maximum relative errors are `0.000172974`, `0.000125961`, and `0.00422226` |
| Historical six-route payload | bitwise identical; SHA-256 remains `e97ef467de3932339ff837b565c029df8363a75195276027e0cc60d65b34a24f` |
| Historical inventory/kernel audit | zero unclassified hits, all roots present, kernel AST hashes unchanged |
| Static checks | Python compilation and scoped `git diff --check` pass |

The float32-versus-float64 center differences are explanatory only: objective
`5.726233854419061e-07`; score differences are approximately
`[-4.37e-08,-6.69e-08,-7.67e-09,-4.08e-07,-1.14e-08]`. No cross-dtype
scientific tolerance is inferred from these values.

## Formal FD Qualification

The float32 CPU-XLA certificate replays the frozen Phase 5 three-step fixture
ladder and verifies endpoint separation indirectly through distinct endpoint
calls, chart validity, and exact branch identity. It does not execute the Phase
1 seven-multiplier plateau selection, and the Phase 1 callable-specific endpoint
value and score error bounds remain undeclared. Consequently the formal Phase 1
FD subgate is `INCONCLUSIVE_BLOCKED`. The three passing relative screens are
engineering diagnostics only and cannot override that missing contract.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept shared-core dtype repair | Both dtypes use one core; exact tiny derivatives and float64 preservation pass | Cleared | Larger-shape accumulation and GPU behavior | Build the lower-rung oracle harness | Kalman agreement or production feasibility |
| Close formal Rung 0A FD | Phase 1 plateau/error-bound design executed | Blocked/inconclusive | Callable-specific endpoint and score bounds | Resolve before advancement past Rung 0A | Same-callable FD promotion |
| Execute `T=50,N=10000` | Owner-frozen margin, scale, interval, seed, power/count and tuning design | Blocked | Human scientific decision | Obtain owner amendment only after lower-rung harness is ready | Scientific equivalence |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Shared-core dtype, exact tiny derivative, repeatability, branch, finiteness, and historical preservation checks pass |
| Statistically supported ranking | None; no stochastic scientific comparison was run |
| Descriptive-only differences | Float32/float64 center differences and three-step FD errors |
| Default-readiness | Not established |
| Next evidence needed | Formal FD contract resolution, lower-rung Kalman harness, then owner-frozen primary statistical design |

## Artifacts

Evidence is under
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/dtype-repair-attempt1/`.
The run manifest binds commands, source hashes, artifact hashes, CPU-hidden
status, and nonclaims.

## Post-Run Red Team

The strongest alternative explanation is that exact tiny-fixture equality was
engineered by operation-order matching and may not control forward error at
large `N,T`. That is true and is why this record closes only the shared-core
dtype/derivative wiring defect. A nonzero same-core derivative result on a new
branch-stable fixture, a changed Phase 5 hex value, or a dtype-conditioned
algorithm would overturn this close.
