# LEDH Shared Compact Score Memory Root-Cause Repair Plan

Date: 2026-07-09

Status: `DRAFT_FOR_REVIEW`

## Objective

Trace and repair the shared causes of the LEDH compact score memory/runtime
blocker before any renewed full `N=10000` score admission attempt. The immediate
target is the no-tape compact forward-sensitivity route used by the LEDH score
scripts, while preserving the same realized finite-`N`
`observed_data_log_likelihood_estimator` / `log_likelihood` scalar.

## Root-Cause Trace

The current implementation fixed the old time-history memory problem by using
per-time loops, but it still carries full particle sensitivities and builds
large per-block tangent intermediates:

- LGSSM compact score carries `running_d_particles` and
  `running_d_log_weights` through the `tf.while_loop` in
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`.
- The LGSSM resampling step calls
  `_filterflow_manual_streaming_finite_transport_value_and_jvp_total` and
  receives a full `d_transported` tensor.
- Fixed-SIR, actual-SV, predator-prey, generalized-SV, and KSC-SV score scripts
  have analogous `_compact_forward_transport_jvp_tf` wrappers that call the
  same shared transport JVP and then carry `next_d_particles`.
- The shared hotspot is
  `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`.
  In particular:
  - `_filterflow_streaming_softmin_jvp` builds a 5D `d_diff`/`d_cost` path for
    each row/column block.
  - `_filterflow_streaming_column_log_normalizer_jvp` repeats the same 5D
    pattern for column normalization tangents.
  - `_filterflow_streaming_transport_from_potentials_jvp` builds the largest
    5D `d_weighted` block before reducing over columns.
  - The function also stacks `tangent_blocks` into `d_transported`; this final
    full tangent state is part of the current forward-sensitivity API and is
    smaller than the per-block 5D products, but it confirms the route is not a
    reduce-only score recurrence yet.

This is an implementation/tensor-lifetime issue, not evidence that the target
scalar is wrong. A later reduce-only recurrence may still be necessary if the
shared contraction repair is insufficient at `N=10000,T=50`.

## Similar-Issue Inventory

| Surface | File | Shared issue |
| --- | --- | --- |
| LGSSM compact score | `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py` | carries `d_particles`; calls shared total JVP; full `d_transported` returned |
| fixed-SIR compact score | `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py` | same shared total JVP and `d_transported` carry |
| actual-SV compact score | `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py` | same shared total JVP and `d_transported` carry |
| predator-prey compact score | `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py` | same shared total JVP and `d_transported` carry |
| generalized-SV compact score | `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py` | same shared total JVP and `d_transported` carry |
| KSC-SV compact score | `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py` | same shared total JVP and `d_transported` carry |
| shared Sinkhorn softmin JVP | `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py` | 5D `d_diff` product can be contracted directly |
| shared column normalizer JVP | same | 5D `d_diff` product can be contracted directly |
| shared transport-from-potentials JVP | same | 5D `d_weighted` product can be reduced by direct contractions |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the shared compact no-tape transport JVP be made materially less memory-hungry without changing the LEDH score target scalar? |
| Baseline/comparator | Current compact JVP outputs and tiny same-scalar FD checks. |
| Primary engineering criterion | Replace avoidable 5D broadcast products in the shared transport JVP with equivalent direct reductions/einsums, while all focused compact score tests still pass. |
| Correctness criterion | Tiny deterministic transport JVP directional-oracle tests and LGSSM compact score tests remain numerically unchanged within existing tolerances. |
| Full-scale diagnostic criterion | After focused tests pass, run bounded GPU rungs in order: tiny score-only, intermediate score-only, then one `N=10000,T=50` LGSSM score-only diagnostic. |
| Veto diagnostics | Target scalar drift; changed parameter order; exact-Kalman substitution; use of `GradientTape` or `ForwardAccumulator` on production route; stopped partial derivative; FD mismatch; nonfinite score; no artifact from a rung. |
| Explanatory diagnostics | Runtime, peak GPU memory, chunk sizes, tensor source audit, and whether final `d_transported` remains as a known current-API carry. |
| Not concluded | Full leaderboard admission, HMC readiness, posterior correctness, scientific superiority, or that a reduce-only score recurrence is unnecessary. |

## Phases

### Phase 0: Trace And Baseline Audit

Objective: freeze the root cause inventory and verify that the planned commands
answer the stated question.

Steps:

1. Inspect all compact score scripts for calls to the shared total JVP and
   carried particle sensitivities.
2. Inspect the shared transport JVP for avoidable large broadcast products.
3. Record the affected surfaces and forbidden nonclaims in this plan.
4. Skeptically audit the plan for wrong baselines, proxy promotion criteria,
   stale runbook assumptions, and missing stop conditions.

Exit condition: this plan identifies the shared code surface and avoids
claiming score admission from any smoke result.

### Phase 1: Shared Kernel Contraction Repair

Objective: remove avoidable 5D block temporaries from the shared compact JVP
without changing returned values.

Steps:

1. In `_filterflow_streaming_softmin_jvp`, compute `d_cost` by direct
   contractions against `d_query_block` and `d_key_block` instead of
   materializing `d_diff`.
2. Apply the same direct contraction in
   `_filterflow_streaming_column_log_normalizer_jvp`.
3. In `_filterflow_streaming_transport_from_potentials_jvp`, compute
   `d_increment` as the sum of:
   - `tf.einsum("brc,bcdp->brdp", transport_block, d_particle_block)`;
   - a transport-logit contribution contracting `transport_block`,
     `particle_block`, and `d_log_transport`.
4. Keep the public helper signatures and returned tensors unchanged.
5. Add source-level regression checks that the old 5D `d_weighted` pattern does
   not reappear in the shared compact JVP.

Exit condition: focused CPU tests pass and source audit confirms no production
autodiff.

### Phase 2: Focused Correctness And Source Tests

Objective: prove the repair preserves current compact behavior on deterministic
small cases.

Required checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_compact_transport_jvp.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Exit condition: tests pass, or a result record states the smallest failing
symbol and no GPU rung is launched.

### Phase 3: LGSSM Score-Only Rung Ladder

Objective: test whether the shared kernel repair resolves the current
`N=10000,T=50` LGSSM score-only no-artifact blocker.

Rungs:

1. Trusted GPU `N=256,T=3` score-only smoke.
2. Trusted GPU `N=1000,T=10` score-only diagnostic.
3. Trusted GPU `N=10000,T=50` single-seed score-only diagnostic.

Exit condition: each rung emits its expected artifact before the next rung
starts, or a blocker result is written.

### Phase 4: Cross-Model Smoke Coverage

Objective: ensure the shared repair does not regress other compact score
wrappers.

Steps:

1. Run the existing fixed-SIR, actual-SV, predator-prey, generalized-SV, and
   KSC-SV compact score contract tests that are already small enough for local
   CPU execution.
2. If a model-specific compact score fails, write a model-specific blocker and
   do not reinterpret the shared repair as globally admitted.

Exit condition: all focused cross-model tests pass, or model-specific blockers
are recorded.

### Phase 5: Reduce-Only Design Decision

Objective: decide whether the remaining blocker requires a true reduce-only
score recurrence.

Decision rule:

- If the shared contraction repair lets LGSSM emit at full score-only scale,
  return to the existing admission runbook for score+FD aggregation.
- If the full rung still fails, write a new subplan for a reduce-only
  recurrence that eliminates the need to return full `d_transported` except in
  tiny equivalence tests.

## Review Plan

Use Claude as read-only reviewer if local policy and tooling allow a bounded
packet-only review. Claude has no edit, execution, approval, scientific-claim,
or authority role. If Claude is unavailable or policy-blocked, use a fresh
Codex review and record the limitation.

Review must check:

- same-target scalar preservation;
- all affected model surfaces are identified;
- Phase 1 is a value-preserving tensor-lifetime repair;
- full-scale rungs are diagnostic and not admission;
- stop conditions are sufficient.

## Stop Conditions

Stop before implementation or next rung if:

- review finds a material unpatched flaw;
- tests show a numerical mismatch with the current compact route;
- any production route uses forbidden autodiff;
- a GPU rung fails to emit;
- continuing requires changing the target scalar, parameter order, seeds,
  `N/T` identity, or admission criteria after seeing results;
- continuing requires package installation, network/data fetches, credentials,
  destructive git actions, or unrelated dirty-worktree edits.

## Skeptical Audit

- Wrong baseline checked: exact Kalman likelihood and historical manual VJP are
  not valid baselines for admission.
- Proxy metric checked: score-only emission is diagnostic, not admission.
- Hidden assumption checked: removing 5D temporaries may not be enough; Phase 5
  keeps the reduce-only recurrence as the next planned repair.
- Environment checked: GPU rungs require trusted execution.
- Artifact sufficiency checked: each meaningful run must preserve logs or write
  a blocker result.

Audit status: `PASS_FOR_REVIEW_BEFORE_IMPLEMENTATION`.
