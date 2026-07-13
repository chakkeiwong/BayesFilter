# LEDH Score Tangent-Materialization Root-Cause Repair Plan

Date: 2026-07-09

Status: `READY_FOR_REVIEW_THEN_EXECUTION`

## Objective

Trace and repair the remaining LEDH score paths that still materialize or carry
full parameter-axis particle tangent tensors such as
`[batch, N, state_dim, param_dim]` at leaderboard scale. The production default
score route must be memory-style reverse/VJP or an equivalent reduce-only
recurrence of the same realized finite-`N` LEDH
`observed_data_log_likelihood_estimator`. The compact forward-sensitivity route
is historical/diagnostic only for full score admission.

## Root-Cause Trace

The earlier shared transport contraction repair removed avoidable blockwise
5D broadcast temporaries in `annealed_transport_tf.py`. Current source tests
already reject the old `d_weighted` / `d_diff` patterns in
`tests/test_ledh_compact_transport_jvp.py`.

The remaining problem is different: the compact score API still returns and
propagates full particle tangents. In
`experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`,
`_filterflow_streaming_transport_from_potentials_jvp` creates
`tangent_blocks`, stacks them, and returns `d_transported` with shape
approximately `[batch, N, state_dim, param_dim]`. The LGSSM compact route in
`docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py` carries
`running_d_particles` and `running_d_log_weights` through the filtering loop.
Fixed-SIR, actual-SV, predator-prey, generalized-SV, and KSC-SV compact score
wrappers have the same pattern through their `_compact_forward_transport_jvp_tf`
helpers.

Important current-state correction: the LGSSM runner now labels the default
score route as `memory_style_reverse_vjp_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`
and routes score-only diagnostics through `_manual_value_and_score_from_components`.
Older Phase 2W wording that calls the current default full attempt a compact
forward-sensitivity attempt is stale. The plan must not chase that stale label.

Model inventory:

| Model | Current executable score paths | Required action |
| --- | --- | --- |
| LGSSM | Memory-style reverse/VJP exists and is default route; compact forward sensitivity still exists as historical helper. | Tighten tests and run bounded score-only/memory rungs later. |
| Fixed SIR | Manual total VJP exists through the parameterized SIR harness, but contract currently treats its route string as historical. Compact route still exists. | Introduce a new memory-style fixed-SIR route ID, keep old `manual_total_vjp*` historical, wire artifacts to the new ID only when all-parameter correctness and memory evidence exist. |
| Predator-prey | Manual reverse/VJP exists, but `_manual_value_and_score_across_seeds` currently calls compact forward sensitivity and emits compact provenance. | Wire across-seed/default diagnostics to manual reverse/VJP and update tests. |
| Actual SV | Manual reverse/VJP exists, but `_manual_value_and_score_across_seeds` currently calls compact forward sensitivity and emits compact provenance. | Wire across-seed/default diagnostics to manual reverse/VJP and update tests. |
| Generalized SV | Only compact forward sensitivity exists. | Keep tiny-only/diagnostic; derive a memory-style route in a later phase before full admission. |
| KSC SV | Only compact forward sensitivity exists. | Keep tiny-only/diagnostic; derive a KSC-specific memory-style route in a later phase before full admission. |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can all LEDH score defaults avoid full parameter-axis particle tangent propagation where a memory-style reverse/VJP route already exists, while preserving the same finite-`N` LEDH scalar? |
| Baseline/comparator | Current compact forward-sensitivity helpers and tiny same-scalar FD tests; current LGSSM memory-style reverse/VJP route. |
| Primary criterion for this plan | Contract/tests and model wrappers make memory-style reverse/VJP the default for LGSSM, fixed-SIR, predator-prey, and actual-SV where available; generalized-SV/KSC remain blocked from full admission. |
| Correctness criterion | Focused CPU-hidden tiny tests pass: no production `GradientTape`/`ForwardAccumulator`, same-scalar finite-difference diagnostics pass where already available, and compact helper remains diagnostic only. |
| Full-scale diagnostic criterion | After focused tests pass, trusted GPU score-only rungs are allowed only under the model-specific phase gates. For LGSSM, `N=10000,T=50,Sinkhorn=10` is forbidden until the `N=1000,T=10,Sinkhorn=10` memory-style blocker rung emits under budget. Score-only emission is never admission. |
| Memory gate | Full admission requires `memory_diagnostics.n10000_memory_pass == true`, `memory_diagnostics.source` naming the measured source (`score_gpu_memory_info_after`, `max_per_seed_score_gpu_memory_info_after`, or an explicitly reviewed trusted-GPU memory artifact), finite `peak_mib`, finite `budget_mib`, and `peak_mib <= budget_mib`. The default per-row budget is `14000.0 MiB` unless a reviewed subplan sets a stricter row-specific budget before the run. |
| Admission criterion | A row is admitted only when full fixed-seed score+FD artifact validates with `validate_ledh_score_artifact(..., require_admitted=True)`, passes the numeric memory gate above, and uses same-scalar finite-difference correctness. Generic `exact_reference` correctness is not sufficient for full admission because exact Kalman or other non-LEDH baselines are not the realized finite-`N` LEDH scalar. |
| Veto diagnostics | Target scalar drift; exact-Kalman substitution; production autodiff; stopped partial derivative; compact forward-sensitivity full admission; nonfinite score; parameter order mismatch; no artifact; memory over budget. |
| Explanatory diagnostics | Runtime, GPU peak memory, chunk sizes, route IDs, whether history/records are stored, and whether compact tangent helpers remain present for tiny diagnostics. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, exact likelihood equality, or full leaderboard completion. |

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Exact Kalman and non-LEDH scores are not comparators for admission; use same finite-`N` LEDH scalar and existing tiny FD checks. |
| Proxy promoted | Unit tests and score-only rungs are engineering diagnostics, not full admission. |
| Stale context | Phase 2W wording is corrected: current LGSSM default uses memory-style route despite a legacy CLI mode name. |
| Hidden assumption | Existing manual reverse/VJP routes for predator-prey and actual-SV are checked by FD tests before default wiring. |
| Artifact insufficiency | Every material phase writes a result record and names exact handoff conditions. Memory artifacts must include source, peak MiB, budget MiB, and pass/fail derivation. |
| Boundary risk | Generalized-SV and KSC are not promoted until a reviewed memory-style derivation/implementation exists. |

Audit status: `PASS_FOR_BOUNDED_REVIEW_BEFORE_EXECUTION`.

## Phase 0: Contract And Route Taxonomy Repair

### Phase Objective

Make the shared score contract distinguish three route classes:
memory-style admissible, compact forward-sensitivity historical/diagnostic, and
old `manual_total_vjp*` historical aliases. Add model-specific memory-style
route IDs only for routes that have an executable reverse/VJP implementation.

### Entry Conditions

- The current contract admits only LGSSM memory-style provenance.
- Compact forward-sensitivity provenance is already historical for full
  admission.
- Fixed-SIR, predator-prey, and actual-SV have manual reverse/VJP code but old
  route strings are currently historical.

### Required Artifacts

- `bayesfilter/highdim/ledh_score_contract.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`
- `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`
- Phase 0 result:
  `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase0-contract-result-2026-07-09.md`

### Required Checks/Reviews

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py -q
```

### Evidence Contract

Pass if full admission accepts only memory-style route IDs and rejects compact
or old historical aliases, while tiny diagnostics can still preserve historical
provenance. Full admission must also reject bare memory flags without numeric
`peak_mib <= budget_mib` evidence and reject generic `exact_reference`
correctness unless a later reviewed contract proves the reference is the same
realized finite-`N` LEDH scalar.

### Forbidden Claims/Actions

- Do not claim model score admission.
- Do not run GPU rungs.
- Do not change target scalar, seeds, parameter order, or value artifacts.
- Do not use exact Kalman, exact nonlinear likelihood, or any non-LEDH
  reference as full-admission score correctness.

### Next-Phase Handoff

Phase 1 may start only if contract tests pass and the result records the exact
new memory-style route IDs.

### Stop Conditions

Stop if the contract cannot distinguish memory-style route IDs from old
historical aliases without weakening admission validation.

## Phase 1: Predator-Prey And Actual-SV Default Wrapper Repair

### Phase Objective

Wire predator-prey and actual-SV across-seed/default score diagnostics to the
existing manual reverse/VJP routes instead of compact forward sensitivity.

### Entry Conditions

Phase 0 contract taxonomy passed.

### Required Artifacts

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- Phase 1 result:
  `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase1-predator-actual-sv-result-2026-07-09.md`

### Required Checks/Reviews

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py -q
```

### Evidence Contract

Pass if tiny default score wrappers run under the no-autodiff sentinel, emit the
memory-style route IDs, preserve same-scalar finite-difference checks, and the
compact helper is tested only as diagnostic/historical.

### Forbidden Claims/Actions

- Do not claim `N=10000` memory pass.
- Do not remove compact helper code needed for tiny equivalence diagnostics.
- Do not make generalized-SV/KSC share actual-SV route IDs.

### Next-Phase Handoff

Phase 2 may start only if both model tests pass and the result records any
remaining model-specific risks.

### Stop Conditions

Stop if manual reverse/VJP default wiring changes the tiny value scalar or
fails FD checks.

## Phase 2: Fixed-SIR Memory-Style Reclassification

### Phase Objective

Introduce a fixed-SIR memory-style route ID distinct from the old
`manual_total_vjp*` historical alias, and allow full admission only for that new
route when all-parameter FD correctness and `N=10000` memory evidence exist.

### Entry Conditions

Phase 0 contract taxonomy passed. Existing fixed-SIR memory artifact remains
directional-only/tiny unless all-parameter correctness is present.

### Required Artifacts

- `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
- Phase 2 result:
  `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase2-fixed-sir-result-2026-07-09.md`

### Required Checks/Reviews

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

### Evidence Contract

Pass if old fixed-SIR `manual_total_vjp*` artifacts remain historical/tiny, the
new memory-style provenance is accepted by the contract only with full
correctness and memory gates, and no parameterized diagnostic row can be
admitted.

### Forbidden Claims/Actions

- Do not turn the old directional memory artifact into full admission.
- Do not admit the parameterized SIR diagnostic row.
- Do not weaken the all-parameter correctness requirement.

### Next-Phase Handoff

Phase 3 may start only if fixed-SIR contract tests pass and a result record
states that no new `N=10000` full admission was claimed.

### Stop Conditions

Stop if route reclassification admits old historical artifacts or allows
directional-only evidence as all-parameter correctness.

## Phase 3: LGSSM Sinkhorn Reverse Lifetime Repair

### Phase Objective

Repair the remaining LGSSM memory-style score blocker by reducing finite
Sinkhorn reverse/VJP tensor lifetime at the full production Sinkhorn step
count, not by re-running the already blocked full `N=10000,T=50` command.
The route remains the current memory-style reverse/VJP route, and the old
compact forward-sensitivity route remains historical/diagnostic only.

### Entry Conditions

Phases 0-2 passed. Prior trusted GPU evidence already shows:

- `N=256,T=3,Sinkhorn=2` and `N=1000,T=10,Sinkhorn=2` score-only emitted with
  memory-style provenance.
- `N=1000,T=10,Sinkhorn=10` and `N=10000,T=50,Sinkhorn=10` exceeded the
  reviewed `14000 MiB` score-memory budget and emitted no artifact.

### Required Artifacts

- Phase 3 subplan:
  `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase3-sinkhorn-reverse-lifetime-subplan-2026-07-09.md`
- Implementation diff, if repair proceeds:
  `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- LGSSM runner diff only if the trace proves retained time history is the
  active blocker:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- Trusted GPU score-only artifacts or blocker records under
  `docs/plans/artifacts/`.
- Phase 3 result:
  `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase3-sinkhorn-reverse-lifetime-result-2026-07-09.md`

### Required Checks/Reviews

Trace and repair the finite-Sinkhorn reverse lifetime first. After focused
tests pass, run rungs in order, stopping on first failure to emit or
memory-over-budget result:

1. `N=1000,T=10,Sinkhorn=10` score-only.
2. `N=10000,T=50,Sinkhorn=10` single-seed score-only only if rung 1 emits
   under budget.

Trusted GPU execution is required by project policy.

### Evidence Contract

Pass if the repaired route emits durable score-only artifacts with memory-style
route ID, finite score, and peak memory below `14000 MiB` for both rungs above.
Rungs remain diagnostic until same-scalar FD and fixed-seed aggregation pass.

### Forbidden Claims/Actions

- Do not admit score-only artifacts.
- Do not run the `N=10000,T=50,Sinkhorn=10` rung unless
  `N=1000,T=10,Sinkhorn=10` emits under budget.
- Do not change CLI seeds, `N`, `T`, transport policy, Sinkhorn settings,
  memory budget, or parameter order after seeing results.
- Do not describe compact forward sensitivity as the current LGSSM default.

### Next-Phase Handoff

If both rungs emit under budget, draft the score+FD aggregation subplan. If
`N=1000,T=10,Sinkhorn=10` still fails, write a blocker result naming the next
smallest Sinkhorn reverse-lifetime repair. If rung 1 passes but
`N=10000,T=50,Sinkhorn=10` fails, hand off to LGSSM time-history checkpointing
or lower-level XLA allocator diagnosis based on the retained-state trace.

### Stop Conditions

Stop if review finds an unpatched material flaw, focused tests fail, a GPU
rung fails to emit, memory exceeds budget, or the emitted route ID is not
memory-style.

## Phase 4: Generalized-SV And KSC Reduce-Only Derivation Gate

### Phase Objective

Prevent full admission for generalized-SV and KSC until a reviewed
model-specific memory-style reverse/VJP or reduce-only recurrence exists.

### Entry Conditions

The available generalized-SV and KSC paths are compact forward-sensitivity only.

### Required Artifacts

- Derivation/implementation subplans for generalized-SV and KSC, or blocker
  results preserving tiny-only status.

### Required Checks/Reviews

Existing tiny tests must continue to pass, but they are not admission evidence.

### Evidence Contract

Pass only when each model has either a reviewed memory-style derivation and
implementation plan or a blocker that explicitly prevents full score admission.

### Forbidden Claims/Actions

- Do not borrow actual-SV route IDs for generalized-SV or KSC.
- Do not claim compact forward-sensitivity is production memory-style.
- Do not admit KSC as exact native actual-SV likelihood.

### Next-Phase Handoff

Only models with reviewed memory-style implementations may enter full
`N=10000` score rungs.

### Stop Conditions

Stop if a model would require changing the target scalar or using production
autodiff to proceed.

## Review Plan

Use Claude as bounded read-only reviewer via
`~/python/claudecodex/scripts/claude_review_gate.sh`. Claude is not an
execution authority. If Claude is unavailable or the review gate fails, replace
the review with a fresh Codex review and record the limitation.

Review must check:

- whether the root cause separates old 5D broadcast temporaries from the
  remaining `[B,N,D,P]` tangent carry;
- whether the route taxonomy blocks compact/historical full admission;
- whether model phases avoid promoting generalized-SV/KSC prematurely;
- whether evidence and stop conditions are sufficient.

## Stop Conditions For The Program

Stop before further execution if:

- review returns a material unpatched flaw;
- focused tests fail and cannot be repaired without changing the scalar;
- a change would require package installation, network/data fetches,
  credentials, destructive git actions, or unrelated dirty-worktree edits;
- full-scale rungs would cross runtime/model-file/funding/product/scientific
  claim boundaries without explicit approval.
