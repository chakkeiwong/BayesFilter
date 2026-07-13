# Phase 5 Repair Subplan: Actual-SV Streaming-Flow Parity VJP

metadata_date: 2026-07-07
status: `DRAFT_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-repair-streaming-flow-parity

## Phase Objective

Repair the actual-SV score route so its forward scalar uses the same
streaming-flow algorithm as the admitted value route before any full-row
score/memory run.

The score target remains:

```text
row_id = zhao_cui_sv_actual_nongaussian_T1000
target_scalar = observed_data_log_likelihood_estimator
target_output_tensor_field = log_likelihood
target_observation_policy = transformed_actual_sv_log_y_square
theta_coordinate_system = synthetic_unconstrained
score_parameter_names = [gamma_unconstrained, log_beta]
```

## Entry Conditions Inherited From Previous Phase

- Actual-SV tiny score diagnostic passed using a matrix-flow aux route.
- Full actual-SV score is blocked because the matrix-flow score route differs
  from the admitted streaming-flow value route by about `2.21e-05` at
  `T=2,N=64`.
- No full actual-SV score/memory run has been launched.

## Required Artifacts

Input artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-result-2026-07-07.md`
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`

Code/tests:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`
- `experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_streaming_tf.py`
- `experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`

Expected artifacts:

- updated score adapter and tests;
- parity repair result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-flow-parity-repair-result-2026-07-07.md`;
- refreshed full-row score/memory subplan or explicit blocker.

## Required Checks/Tests/Reviews

Required local checks after repair:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

New required tests:

- same-forward-scalar parity between the value route and score route at
  `T=2,N=64`;
- same-forward-scalar parity for at least two particle chunk sizes;
- no-autodiff sentinel for the streaming-flow score route;
- all-coordinate FD after parity passes;
- artifact remains `tiny_score_diagnostic_not_admitted` until full memory gate.

Review:

- bounded read-only review of this repair subplan before implementation;
- bounded read-only review of the repair result and refreshed next subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the score route differentiate the exact same streaming-flow finite-`N` scalar used by the admitted actual-SV value artifact? |
| Baseline/comparator | `benchmark_ledh_same_target_actual_sv_value.py` using `streaming_tf.batched_ledh_flow_streaming_particles_tf`. |
| Primary criterion | Score-route forward scalar matches value-route forward scalar at tested tiny scales within strict numerical tolerance before FD score checks are used for admission evidence. |
| Veto diagnostics | Matrix-flow route accepted as same algorithm; parity mismatch; tape/autodiff; stopped partial; target substitution; chunk-order mismatch; padding affects unpadded particles; nonfinite output. |
| Explanatory diagnostics | Per-chunk differences, dtype, chunk sizes, FD errors, memory estimate, runtime. |
| Not concluded | Full score admission, HMC readiness, posterior correctness, runtime ranking, or scientific superiority. |
| Artifact | Repair result, tests, refreshed score artifact or blocker. |

## Step-By-Step Plan

1. Add a forward-only parity test comparing the admitted value route with the
   current score route. Keep it failing or xfail only if the blocker result is
   being written; do not hide the mismatch.
2. Implement a streaming-flow-with-aux wrapper that mirrors
   `batched_ledh_flow_streaming_particles_tf`:
   - same chunk size;
   - same padded slicing;
   - same call to `batched_ledh_flow_core_tf`;
   - retained aux only for each real block.
3. Implement streaming-flow VJP by reversing the retained blocks and calling
   the existing matrix-flow VJP per block.
4. Replace actual-SV score route flow call with the streaming-flow-with-aux
   primitive.
5. Rerun same-forward-scalar parity tests.
6. Rerun no-autodiff sentinel and all-coordinate FD tiny tests.
7. Write repair result.
8. Refresh the full-row score/memory subplan if parity and tiny FD pass;
   otherwise write a blocker result.

## Forbidden Claims/Actions

- Do not claim same algorithm from algebraic similarity alone.
- Do not run full `N=10000,T=1000` before parity and tiny FD pass.
- Do not use tape, `ForwardAccumulator`, hidden autodiff, or stopped partials.
- Do not substitute KSC, raw Gaussian, or augmented-noise target evidence.
- Do not relax parity tolerance after seeing mismatches without review.
- Do not claim HMC readiness, posterior correctness, runtime ranking,
  scientific superiority, or all-algorithm comparison.

## Exact Next-Phase Handoff Conditions

The full-row score/memory gate may resume only if:

- same-forward-scalar parity passes;
- no-tape tiny all-coordinate FD passes after parity repair;
- local checks pass;
- a refreshed full-row score/memory subplan exists and is reviewed.

## Stop Conditions

Stop and write a blocker result if:

- streaming-flow aux/VJP cannot be implemented without changing the value
  algorithm;
- parity remains nonzero beyond strict tolerance;
- no-tape provenance becomes ambiguous;
- tiny FD fails after parity repair;
- review finds a material issue that does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Compare directly to the admitted streaming-flow value route. |
| Proxy promotion | FD score checks are not considered until forward parity passes. |
| Missing stop condition | Stop on parity mismatch, no-tape ambiguity, or target substitution. |
| Hidden assumption | Chunking and padding are explicitly tested. |
| Stale context | Starts from the current blocker result and tiny diagnostic. |
| Environment mismatch | CPU-hidden tiny parity tests first; trusted GPU only later. |
| Useless artifact | Full admission remains impossible until validator replay with `require_admitted=True`. |
