# Phase 7 Subplan: Trusted GPU XLA Method-Isolated Gates

Date: 2026-07-11
Status: `DRAFT_AWAITING_PHASE6_HANDOFF`

## Phase Objective

Determine independently whether analytical value/score, batched value, and
batched reverse-mode gradient compile and run on trusted GPU/XLA.

## Entry Conditions

- Phase 6 common correctness gates pass. CPU XLA either passes or records a
  reviewed CPU-backend-only failure that does not invalidate the smallest GPU gate.
- Managed-session GPU trust requirements are encoded in artifacts.
- Exact method-isolated commands and timeouts were refreshed/reviewed at Phase 6 close.

## Required Artifacts

- Trusted GPU visibility/provenance artifact.
- Float32 analytical-only preflight.
- Float32 batched value-only and corrected gradient preflights.
- Float64 preflights only after float32 localization.
- Per-method JSON/Markdown/log outputs; Phase 7 result; refreshed Phase 8 subplan.

## Required Checks, Tests, And Reviews

- Record logical/physical GPU, trust basis `owner_designated_managed_session_visible_gpu_trusted`, TF32, dtype, JIT, XLA flags, TensorFlow/CUDA-visible provenance, source hashes.
- Run smallest `dim=10,P=50,T=120,B=1` method-isolated arms first.
- If GPU autodiff fails, distinguish value forward, gradient construction, XLA lowering/codegen, and execution.
- Progress to `B=4/16` only after smallest arm passes.
- Finite/dtype/shape/parity checks and material review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Which repaired score methods are independently viable on the target GPU/XLA path? |
| Baseline | Historical combined preflight that failed only in row-loop autodiff and blocked all GPU work. |
| Primary criterion | Analytical-only and corrected autodiff are assessed independently with complete provenance; passing arms are finite and parity-valid. |
| Vetoes | Untrusted GPU context, combined-arm coupling, XLA failure, non-finite output, parity failure, or missing settings. |
| Explanatory only | Compile/warm time, memory, and layout error text. |
| Not concluded | Production/default/HMC/posterior readiness or superiority. |

## Forbidden Claims And Actions

- Do not let autodiff failure erase analytical GPU evidence.
- Do not treat non-elevated failure outside managed trust as hardware failure.
- Do not run the full grid before Phase 8 review.

## Exact Next-Phase Handoff Conditions

- Phase 7 result classifies each method/dtype/batch gate separately.
- Phase 8 grid is prospectively narrowed to viable arms without rewriting failure history.
- Replication count, uncertainty method, time budget, and exact artifacts are refreshed and reviewed in Phase 8.

## Stop Conditions

- Trusted GPU visibility/provenance fails.
- A hard veto fires and no declared repair remains.
- Five review rounds fail to converge.

## Mandatory Phase-End Sequence

1. Run required checks.
2. Write Phase 7 result.
3. Refresh Phase 8 subplan.
4. Review and repair Phase 8 before advancing.
