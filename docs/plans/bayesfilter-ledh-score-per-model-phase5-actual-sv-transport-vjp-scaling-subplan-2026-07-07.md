# Phase 5 Repair Subplan: Actual-SV Transport VJP Scaling

metadata_date: 2026-07-07
status: `DRAFT_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-repair-transport-vjp-scaling

## Phase Objective

Repair or redesign the actual-SV no-tape streaming transport VJP path so a
single value/score reverse pass can scale beyond the current `T=20,N=1024`
blocker, without changing the admitted forward scalar or weakening score
correctness semantics.

## Entry Conditions Inherited From Previous Phase

- Same-forward-scalar streaming-flow parity is repaired at tiny scale.
- CPU-hidden tiny value-score-only smoke works and is non-admission.
- Trusted GPU value-score-only `T=20,N=1024` was manually interrupted after
  near-budget memory and long runtime.
- Tracebacks identify the blocker inside manual streaming finite transport
  total pullback and column-normalizer VJP.
- Full actual-SV score remains not admitted.

## Required Artifacts

Input artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-scaling-repair-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-value-score-only-tiny-smoke-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-gpu-ladder-t5-n256-2026-07-07.json`

Code/tests:

- `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- `docs/benchmarks/benchmark_p8p_parameterized_sir_gradient.py`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- transport VJP tests under `tests/highdim/` if new helpers are added.

Expected artifacts:

- transport VJP scaling result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-transport-vjp-scaling-result-2026-07-07.md`;
- focused diagnostic outputs for small and moderate shapes;
- refreshed full-row score admission subplan or explicit blocker;
- review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-transport-vjp-scaling-review-bundle-2026-07-07.md`.

## Required Checks/Tests/Reviews

Before implementation:

- bounded read-only review of this transport VJP scaling subplan.

After implementation:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Add focused transport VJP tests if new helper symbols are introduced.

Any TensorFlow GPU diagnostic must use trusted/escalated execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the manual streaming transport VJP be made memory/runtime-bounded enough for moderate and eventually full actual-SV score runs while preserving the same forward scalar? |
| Baseline/comparator | Current `transport_ad_mode=full` manual streaming finite total pullback, tiny FD diagnostics, value-score-only smoke, and interrupted `T=20,N=1024` run. |
| Primary criterion | A reviewed VJP path passes small-shape correctness checks and a moderate trusted GPU no-FD diagnostic without near-budget memory or impractical runtime. |
| Veto diagnostics | Tape/autodiff, stopped partials, forward scalar change, hidden dense `N x N` materialization at full scale, nonfinite cotangents, validator overclaim, or runtime-only admission. |
| Explanatory diagnostics | Stage timing, peak memory, block counts, TensorArray payload, per-block recompute cost, tiny FD errors. |
| Not concluded | Full score admission, HMC readiness, posterior correctness, scientific superiority, or all-model leaderboard completion. |
| Artifact | Transport VJP scaling result plus refreshed full-row subplan or blocker. |

## Step-By-Step Plan

1. Read and map the current transport VJP call chain:
   - `_manual_transport_vjp_tf`;
   - `_filterflow_manual_streaming_finite_transport_total_pullback`;
   - `_filterflow_streaming_transport_from_potentials_vjp`;
   - `_filterflow_streaming_column_log_normalizer`;
   - `_filterflow_streaming_finite_sinkhorn_potentials_vjp_total`.
2. Identify retained tensors, TensorArrays, and any hidden dense or repeated
   block loops that explain the `T=20,N=1024` near-budget memory.
3. Add a tiny standalone transport VJP diagnostic that does not run the full
   actual-SV filter, with fixed particles/log weights/upstream.
4. Compare candidate VJP modes at tiny shape:
   - current total pullback;
   - blockwise VJP finite route if compatible with no-stopped-partial policy;
   - a recompute/checkpointed variant if implementable locally.
5. If a candidate changes derivative semantics, stop and review; do not
   silently use stopped-scale/partial derivatives.
6. If a candidate preserves semantics, add focused tests and a moderate GPU
   diagnostic.
7. Write result and refresh full-row admission subplan or blocker.

## Forbidden Claims/Actions

- Do not run full `N=10000,T=1000` in this phase.
- Do not use `transport_ad_mode=stabilized` or stopped-scale derivatives as
  admission evidence unless a reviewed plan explicitly changes the target
  derivative contract.
- Do not use tape, `ForwardAccumulator`, hidden autodiff, or stopped partials.
- Do not change the forward scalar or transport primal to fit the VJP.
- Do not promote timing/memory diagnostics to correctness or admission.
- Do not broaden validator semantics without separate reviewed artifact
  contract.

## Exact Next-Phase Handoff Conditions

A refreshed full-row actual-SV score admission subplan may start only if:

- a transport VJP scaling result exists;
- local checks pass;
- moderate no-FD GPU diagnostic is no longer near-budget/impractical, or a
  blocker explicitly states why not;
- read-only review agrees the next handoff is boundary-safe.

## Stop Conditions

Stop and write a blocker result if:

- total-pullback memory/runtime cannot be reduced without stopped partials or
  forward-scalar changes;
- blockwise/recompute variants fail tiny correctness;
- no candidate avoids near-budget memory at moderate shape;
- review finds a material issue that does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Current total pullback and repaired same-forward scalar remain the baseline. |
| Proxy promotion | Moderate GPU timing/memory is a gate, not correctness/admission. |
| Missing stop condition | Stop on stopped partials, forward changes, or failed tiny VJP correctness. |
| Hidden assumption | Audit TensorArrays and block loops before patching. |
| Stale context | Starts from the no-FD `T=20,N=1024` blocker, not the earlier FD-only blocker. |
| Environment mismatch | GPU diagnostics require trusted execution. |
| Useless artifact | Result must refresh admission subplan or explicitly block. |
