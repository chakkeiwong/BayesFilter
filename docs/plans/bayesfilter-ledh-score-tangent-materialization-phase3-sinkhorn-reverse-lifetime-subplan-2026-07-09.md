# Phase 3 Subplan: LGSSM Sinkhorn Reverse Lifetime Repair

Date: 2026-07-09

Status: `READY_FOR_CODEX_REVIEW_BEFORE_EXECUTION`

## Phase Objective

Repair the remaining LGSSM memory-style score blocker by reducing finite
Sinkhorn reverse/VJP tensor lifetime at the full production Sinkhorn step
count, without changing the realized finite-`N` LEDH
`observed_data_log_likelihood_estimator` / `log_likelihood` scalar, parameter
order, seeds, transport policy, or score admission criteria.

## Entry Conditions Inherited From Previous Phase

- Phase 0 contract gate passed and rejects generic `exact_reference` full
  admission.
- Phase 1 predator-prey/actual-SV wrappers now default to memory-style
  reverse/VJP route IDs.
- Phase 2 fixed-SIR has a new memory-style route ID and old
  `manual_total_vjp*` aliases remain historical.
- Combined focused gate passed:
  `79 passed, 2 warnings`.
- Prior LGSSM memory-style rungs emitted at `N=256,T=3,Sinkhorn=2` and
  `N=1000,T=10,Sinkhorn=2`.
- Prior trusted GPU attempts at `N=1000,T=10,Sinkhorn=10` and
  `N=10000,T=50,Sinkhorn=10` exceeded the reviewed `14000 MiB` score-memory
  budget and emitted no artifact.

## Required Artifacts

- Implementation diff, if repair proceeds:
  `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- LGSSM runner diff only if the trace proves retained time history is the
  active blocker:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- Focused tests:
  - `tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py`
  - `tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py`
  - `tests/test_ledh_lgssm_manual_score_phase4.py`
  - `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
  - `tests/highdim/test_ledh_score_contract_phase1.py`
- Trusted GPU artifacts or blocker records under `docs/plans/artifacts/`.
- Phase 3 result:
  `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase3-sinkhorn-reverse-lifetime-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

Review:

- Claude review is not retried for this program because the approval reviewer
  rejected the bounded Claude artifact disclosure path.
- Use a fresh Codex read-only review and record its verdict before GPU rungs.

Pre-implementation trace:

1. Inventory retained state in
   `_filterflow_streaming_finite_sinkhorn_potentials_vjp_total`.
2. Estimate memory scaling of `running_ta`, `a_y_ta`, `b_x_ta`, `a_x_ta`,
   `b_y_ta`, and per-call `_filterflow_streaming_softmin_vjp` block stacks.
3. Decide whether the smallest valid repair is:
   - a specialized softmin VJP mode that avoids unused cotangent block stacks;
   - checkpoint/replay over finite Sinkhorn steps;
   - or LGSSM time-history checkpointing after Sinkhorn lifetime is proven
     not dominant.

CPU-hidden tests after implementation:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_softmin_vjp_matches_dense_and_tiny_autodiff \
  tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py::test_streaming_transport_from_potentials_vjp_matches_manual_and_tiny_autodiff \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Trusted GPU rungs, in order, stopping on first no-artifact or memory-over-budget
failure:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 1000 \
  --time-steps 10 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --annealed-scaling 0.9 \
  --annealed-convergence-threshold 0.001 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 256 \
  --col-chunk-size 256 \
  --particle-chunk-size 256 \
  --score-mode manual-reverse \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-score-tangent-materialization-phase3-lgssm-score-only-n1000-t10-s10-2026-07-09.json
```

Only if the first rung emits under budget:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 10000 \
  --time-steps 50 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --annealed-scaling 0.9 \
  --annealed-convergence-threshold 0.001 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode manual-reverse \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-score-tangent-materialization-phase3-lgssm-score-only-n10000-t50-seed81120-2026-07-09.json
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can memory-style LGSSM score avoid the full Sinkhorn-step memory/no-artifact failure while preserving the same finite-`N` LEDH scalar? |
| Baseline/comparator | Prior Phase 1R `Sinkhorn=2` success and `Sinkhorn=10` no-artifact/memory-over-budget blocker. |
| Primary criterion | Trusted GPU `N=1000,T=10,Sinkhorn=10` score-only emits under `14000 MiB`; then single-seed `N=10000,T=50,Sinkhorn=10` emits under the same budget. |
| Correctness criterion | Focused VJP oracle and LGSSM manual score tests pass; no production autodiff or exact Kalman substitution is introduced. |
| Admission criterion | No score-only rung is admitted. Full admission still requires fixed-seed score+FD aggregation through `validate_ledh_score_artifact(..., require_admitted=True)`. |
| Veto diagnostics | Target drift; exact Kalman score substitution; parameter-order change; `GradientTape`/`ForwardAccumulator`; stopped partials; nonfinite score; no artifact; memory over budget. |
| Explanatory diagnostics | Runtime, peak memory, Sinkhorn step count, chunk sizes, retained state ledger, and whether remaining memory scales with Sinkhorn steps or time steps. |
| Not concluded | Full score admission, full leaderboard completion, HMC readiness, posterior correctness, scientific superiority, or cross-model readiness. |

## Forbidden Claims And Actions

- Do not claim score admission from score-only diagnostics.
- Do not run the `N=10000,T=50` rung unless the `N=1000,T=10,Sinkhorn=10`
  rung emits under budget.
- Do not use `compact-sensitivity` as the Phase 3 command path, even though the
  runner currently maps that compatibility alias to memory-style internally.
- Do not replace LEDH finite estimator score with exact Kalman score.
- Do not change seeds, `N`, `T`, transport settings, parameter order, target
  scalar, output tensor field, or memory budget after seeing results.

## Exact Next-Phase Handoff Conditions

If both GPU rungs emit under budget, draft the score+FD aggregation re-entry
subplan. If `N=1000,T=10,Sinkhorn=10` still fails, write a blocker result that
identifies the next smallest repair surface. If `N=1000,T=10,Sinkhorn=10`
passes but `N=10000,T=50` fails, hand off to LGSSM time-history checkpointing
or lower-level XLA allocator diagnosis based on the retained-state trace.

## Stop Conditions

Stop if review finds an unpatched material flaw, focused correctness tests
fail, GPU memory exceeds budget, no artifact is emitted, or continuing would
require package installation, network/data fetches, credentials, destructive
git actions, or changing pass/fail criteria after seeing results.

## Skeptical Audit Before Execution

- Wrong baseline checked: exact Kalman likelihood is not a score target.
- Proxy metric checked: score-only emission is diagnostic, not admission.
- Hidden assumption checked: route taxonomy is no longer the active blocker;
  prior evidence points to full Sinkhorn-step reverse lifetime.
- Environment checked: GPU rungs require trusted/escalated execution.
- Artifact sufficiency checked: result or blocker record is mandatory.

Audit status: `PASS_FOR_CODEX_REVIEW_BEFORE_IMPLEMENTATION`.
