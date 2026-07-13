# LEDH Memory-Style Score Phase 1R Subplan: Transport VJP Memory Repair

Date: 2026-07-09

Status: `READY_FOR_REVIEW_BEFORE_EXECUTION`

## Phase Objective

Repair the remaining memory blocker in the memory-style LEDH score path by
reducing full-tensor scatter/update lifetime inside the streaming transport VJP
pullbacks, without changing the realized finite-`N` LEDH
`observed_data_log_likelihood_estimator` / `log_likelihood` scalar.

The immediate target is the LGSSM memory-style score route, but the shared
transport VJP is also used by fixed-SIR full mode, predator-prey, and actual-SV
manual score routes.

## Entry Conditions Inherited From Previous Phase

- Phase 1 default LGSSM score wiring passed focused CPU tests.
- Tiny trusted GPU `N=256,T=3` score-only emitted memory-style provenance.
- Trusted GPU `N=1000,T=10` memory-style score-only exceeded the memory budget
  and emitted no artifact.
- Old compact forward-sensitivity route is historical/tiny-only.
- Full admission still requires same-scalar FD, memory gate, runtime/device
  gate, fixed-seed aggregation, and score artifact validation.

## Root-Cause Trace

The current reverse/VJP score route removed the old parameter-axis particle
tangent carry, but it still has two memory-heavy surfaces:

1. Shared transport pullback:
   `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
   contains VJP helpers that repeatedly update full `[batch,N,D]` or
   `[batch,N]` accumulators with `tf.tensor_scatter_nd_add`.

   Hot symbols:

   - `_filterflow_streaming_softmin_vjp`
   - `_filterflow_streaming_transport_from_potentials_vjp`
   - `_scatter_axis1_add_3d`
   - `_scatter_axis1_add_2d`

   These helpers are no-tape and mathematically reverse-mode, but the repeated
   scatter-add pattern can allocate full accumulator copies for each block.

2. LGSSM reverse score history:
   `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py` stores
   full per-time `scalar_tas` and `flow_tas`, including several
   `[batch,N,D,D]` flow auxiliary tensors. This may still need checkpointing or
   recomputation after the shared transport VJP repair.

The first repair target is the shared transport pullback because the interrupted
stack reached `_scatter_axis1_add_3d` inside the transport VJP.

## Required Artifacts

- Updated transport implementation:
  `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- Updated or added focused tests:
  - `tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py`
  - `tests/test_ledh_compact_transport_jvp.py`
- Phase 1R result record:
  `docs/plans/bayesfilter-ledh-memory-style-score-phase1r-transport-vjp-memory-repair-result-2026-07-09.md`
- If the GPU rung remains blocked, a precise blocker record with the smallest
  remaining stack/symbol and no admission claim.

## Required Checks, Tests, And Reviews

Review:

- Use Claude read-only review only if the bounded review command is locally
  permitted.
- If Claude is unavailable or policy-blocked, use a fresh Codex read-only
  review and record the limitation.

CPU-hidden tests:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py \
  tests/test_ledh_compact_transport_jvp.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

GPU rungs, trusted/escalated only:

1. LGSSM memory-style score-only `N=256,T=3`.
2. LGSSM memory-style score-only `N=1000,T=10`.
3. Only if rung 2 emits under the reviewed memory budget, attempt one-seed
   LGSSM memory-style score-only `N=10000,T=50`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can the memory-style transport VJP avoid repeated full scatter/update allocation while preserving the same finite-`N` LEDH score? |
| Baseline/comparator | Current no-tape total VJP tests and tiny transport JVP directional-oracle tests. |
| Primary criterion | Focused CPU tests pass and the `N=1000,T=10` memory-style LGSSM score-only rung emits under the reviewed memory budget. |
| Correctness criterion | Tiny transport VJP/JVP tests match tape directional oracles within existing tolerances; LGSSM manual score tests still pass. |
| Veto diagnostics | Target scalar drift; exact Kalman substitution; changed parameter order; production `GradientTape`/`ForwardAccumulator`; stopped partials; FD mismatch in any checked rung; nonfinite score; no artifact; memory over budget. |
| Explanatory diagnostics | Runtime, peak GPU memory, chunk sizes, output devices, whether scatter helpers remain on the hot path. |
| Not concluded | Full score admission, all-model score readiness, HMC readiness, posterior correctness, scientific superiority, or that LGSSM flow-history checkpointing is unnecessary. |

## Implementation Steps

1. Rewrite `_filterflow_streaming_softmin_vjp` to avoid repeated full
   scatter-add updates:
   - first row pass computes `row_logsum`, `d_query` blocks, row epsilon
     cotangents, and stores compact row-logsum blocks;
   - second column pass accumulates `d_key` and `d_values` per column block and
     writes column blocks to TensorArrays;
   - stack row/column blocks once at function exit.
2. Rewrite `_filterflow_streaming_transport_from_potentials_vjp` similarly:
   - first column pass computes and stores `s_block`, `d_particle_block`, and
     `d_logw_block`;
   - row pass computes `d_query` and `d_f` blocks using the stored
     column-block `s`;
   - column pass computes `d_key` blocks;
   - stack blocks once instead of repeated full scatter-add.
3. Keep all public helper signatures and returned tensors unchanged.
4. Add or update source sentinels that forbid the new implementation from
   calling `_scatter_axis1_add_3d` / `_scatter_axis1_add_2d` from those two VJP
   helpers.
5. Run the focused CPU-hidden test command.
6. If CPU tests pass, run trusted GPU rungs in sequence and stop on first
   no-artifact or memory-over-budget rung.
7. Write the Phase 1R result record and either hand off to full score admission
   re-entry or to an LGSSM flow-history checkpointing subplan.

## Forbidden Claims And Actions

- Do not claim score admission from score-only diagnostics.
- Do not replace LEDH score with exact Kalman score.
- Do not change seeds, `N`, `T`, parameter order, or target output field after
  seeing results.
- Do not use production `GradientTape`, `ForwardAccumulator`, stopped partials,
  or old historical forward-sensitivity strings as full-admission evidence.
- Do not run full `N=10000,T=50` before `N=1000,T=10` emits under budget.
- Do not edit unrelated HMC, Neutra, or leaderboard files during this repair.

## Exact Next-Phase Handoff Conditions

Handoff to score-admission re-entry only if:

- CPU-hidden focused tests pass;
- trusted GPU `N=1000,T=10` emits under budget;
- trusted GPU one-seed `N=10000,T=50` score-only emits under budget;
- result record states that the output is diagnostic-only until same-scalar FD
  and fixed-seed aggregation are run.

Handoff to a new LGSSM flow-history checkpointing subplan if:

- transport VJP tests pass but the `N=1000,T=10` or `N=10000,T=50` rung still
  exceeds memory or emits no artifact, and the remaining stack points outside
  the repaired scatter/update surface.

## Stop Conditions

Stop and write a blocker result if:

- review finds an unpatched material flaw;
- any focused correctness test fails;
- source sentinels find production autodiff or old scatter helpers in repaired
  VJP symbols;
- any GPU rung does not emit, exceeds memory budget, or shows nonfinite score;
- continuing would require changing scalar identity, seed identity, parameter
  order, runtime policy, package installation, network access, or unrelated
  dirty-worktree edits.

## Skeptical Audit Before Execution

- Wrong baseline checked: exact Kalman is not a score target here.
- Proxy metric checked: score-only emission is diagnostic, not admission.
- Hidden assumption checked: reverse/VJP avoids parameter tangents but may still
  allocate full scatter workspaces; this plan targets that actual stack.
- Environment checked: all GPU rungs require trusted execution.
- Artifact sufficiency checked: a result or blocker record is mandatory.

Audit status: `PASS_FOR_READ_ONLY_REVIEW_BEFORE_IMPLEMENTATION`.
