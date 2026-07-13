# LEDH Memory-Style Score Root-Cause And Repair Plan

Date: 2026-07-09

Status: `READY_FOR_REVIEW_THEN_EXECUTION`

## Phase Objective

Trace the remaining `N=10000` LEDH score memory blocker, identify all score
paths with the same tensor-lifetime issue, and repair the default score wiring
so full-score attempts use a memory-style no-tape reverse/VJP recurrence rather
than the old full forward-sensitivity tangent route.

## Root-Cause Trace

The failing full LGSSM score path is not a procedural runbook issue.  The
current default score route still calls
`_compact_value_and_score_from_components` in
`docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`.  That route
carries:

- `running_d_particles` with shape approximately
  `[batch, N, state_dim, param_dim]`;
- `running_d_log_weights` with shape approximately `[batch, N, param_dim]`;
- transport JVP potential tangents `d_alpha`, `d_beta`;
- `d_transported` from
  `annealed_transport_tf._filterflow_streaming_transport_from_potentials_jvp`.

The shared contraction patch removed avoidable 5D broadcast temporaries, but
the route still propagates the parameter axis through every particle at full
scale.  At `N=10000,T=50`, the trusted GPU run reached about
`15740 MiB / 16376 MiB`, exceeded the reviewed `14000 MiB` budget, and emitted
no artifact.

The repo already contains memory-style no-tape reverse/VJP score mechanisms:

- LGSSM:
  `benchmark_ledh_same_target_lgssm_m3_t50_value._manual_value_and_score_from_components`
  uses `_manual_transport_vjp_tf` and transport pullbacks.  It avoids
  parameter-axis particle tangents, but is labeled historical/diagnostic and
  full admission is blocked by `_score_admission_decision`.
- Fixed SIR:
  `benchmark_ledh_same_target_fixed_sir_score._fixed_sir_manual_score_diagnostic`
  delegates to `benchmark_p8p_parameterized_sir_gradient._manual_value_and_score_from_components`.
  This is the known memory-style path that produced the earlier `N=10000`
  score-memory artifact.
- Predator-prey and actual-SV:
  both have explicit `_manual_value_and_score_from_components` reverse/VJP
  routines, but their across-seed default score wrappers still call compact
  forward-sensitivity helpers in places.

Similar full-tangent issue sites remain in:

- `benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `benchmark_ledh_same_target_fixed_sir_score.py`
- `benchmark_ledh_same_target_actual_sv_score.py`
- `benchmark_ledh_same_target_predator_prey_score.py`
- `benchmark_ledh_same_target_generalized_sv_score.py`
- `benchmark_ledh_same_target_ksc_sv_score.py`

All of those compact score helpers have the same shape pattern:
`d_particles`, `d_post_flow`, `_compact_forward_transport_jvp_tf`, and
`d_transported`.

## Phase Sequence

### Phase 0: Contract And Route Taxonomy Repair

Objective: make the score contract name the admissible property correctly:
same finite-`N` scalar, no production autodiff, full same-scalar correctness,
and memory gate.  Do not require "compact forward sensitivity" as the only
admissible mechanism after that mechanism has failed the memory gate.

Required artifacts:

- updated `bayesfilter/highdim/ledh_score_contract.py`;
- updated tests in `tests/highdim/test_ledh_score_contract_phase1.py` and
  `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`;
- route constants for memory-style LGSSM score in the LGSSM runner.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py -q
```

Handoff: Phase 1 starts only if old forward-sensitivity routes are no longer
the only full-admission provenance and old historical `manual_total_vjp*`
routes remain blocked unless explicitly reclassified.

### Phase 1: LGSSM Default Score Wiring

Objective: route LGSSM score-mode through the memory-style reverse/VJP
recurrence by default while keeping the full forward-sensitivity route as
historical/diagnostic.

Required artifacts:

- update `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`;
- update LGSSM score tests to require memory-style route for admission;
- keep tiny equivalence test against the old compact route.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

GPU rungs:

1. trusted `N=256,T=3` score-only;
2. trusted `N=1000,T=10` score-only;
3. trusted `N=10000,T=50` single-seed score-only.

Handoff: Phase 2 starts only if the full single-seed LGSSM memory-style route
emits under budget, or the result records a precise blocker.

### Phase 2: Fixed-SIR Default Reclassification

Objective: promote the existing fixed-SIR memory-style score path from
historical wording to default memory-style wording, without changing its scalar
or using the parameterized diagnostic row as the admitted row.

Required artifacts:

- fixed-SIR route constant update;
- contract tests showing the earlier `manual_total_vjp` string is historical,
  while the new fixed-SIR memory-style route is admissible only with
  all-parameter same-scalar correctness and memory evidence.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

### Phase 3: Predator-Prey And Actual-SV Default Wiring

Objective: make the across-seed score wrappers use existing reverse/VJP
memory-style routines instead of compact forward-sensitivity routines.

Required artifacts:

- predator-prey and actual-SV score runner updates;
- tests proving tiny objective/score correctness still pass and source
  sentinels do not reject the intended reverse/VJP memory route.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py -q
```

### Phase 4: Generalized-SV And KSC-SV Memory-Style Derivation Gate

Objective: do not paper over missing reverse/VJP routines for generalized-SV
and KSC-SV.  Either derive/implement model-specific reverse/VJP routines or
write a blocker that keeps compact forward-sensitivity as tiny-only until a
memory route exists.

Checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py -q
```

### Phase 5: Full Score Re-entry

Objective: resume the score admission runbook only after the default route no
longer materializes full parameter-axis particle tangents at full scale.

Required evidence:

- full single-seed score-only emission under budget;
- same-scalar FD split diagnostics;
- fixed-seed shard aggregation through `validate_ledh_score_artifact(...,
  require_admitted=True)`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can LEDH score computation use the same finite-`N` scalar while avoiding full `[batch,N,state_dim,param_dim]` tangent propagation at full score scale? |
| Baseline | Current compact forward-sensitivity route plus the Phase shared-kernel result that failed `N=10000,T=50` memory/artifact emission. |
| Primary criterion | LGSSM `N=10000,T=50` single-seed score-only route emits under the reviewed memory budget using the memory-style no-tape route. |
| Correctness criterion | Tiny memory-style score matches current compact score and same-scalar FD. |
| Admission criterion | No row is admitted until full fixed-seed score+FD aggregation validates against the admitted value artifact. |
| Veto diagnostics | target scalar drift; exact Kalman substitution; production `GradientTape` or `ForwardAccumulator`; stopped partials; FD mismatch; nonfinite score; memory over budget; no artifact; row/seed/parameter-order mismatch. |
| Explanatory diagnostics | runtime, peak memory, chunk sizes, route id, score output devices, and whether route still contains `d_particles[..., param]` at full scale. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, exact Kalman score equality, or cross-model full admission. |

## Forbidden Claims And Actions

- Do not claim admission from score-only diagnostics.
- Do not replace the finite LEDH estimator score with exact Kalman likelihood.
- Do not change seeds, `N`, `T`, transport settings, parameter order, or the
  target output field after seeing results.
- Do not use production `GradientTape`, `ForwardAccumulator`, stopped partials,
  or old historical route strings as full-admission evidence.
- Do not move generalized-SV or KSC-SV to full score admission until they have
  a reviewed memory-style reverse/VJP route or a reviewed equivalent recurrence.

## Skeptical Audit Before Execution

- Wrong baseline checked: exact Kalman is not a valid score target here.
- Proxy metric checked: score-only emission is a memory diagnostic, not
  admission.
- Hidden assumption checked: reverse/VJP memory style must still compute the
  same realized finite-`N` scalar and pass tiny equivalence/FD tests.
- Environment checked: all GPU/CUDA score rungs require trusted execution.
- Artifact sufficiency checked: each material phase must write a result or
  blocker, not just rely on terminal output.

Audit status: `READY_FOR_BOUNDED_READ_ONLY_REVIEW`.
