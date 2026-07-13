# Phase 2 Subplan: LGSSM Score

metadata_date: 2026-07-07
status: `DRAFT_PENDING_PHASE1_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 2

## Phase Objective

Admit, or explicitly block, the LEDH score for:

```text
benchmark_lgssm_exact_oracle_m3_T50
```

The score is the no-tape total derivative of the same realized finite-`N`
LEDH estimator admitted by the value artifact:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

The parameter vector and order are:

```text
phi1, phi2, phi3, q_scale, r_scale
```

in coordinate system:

```text
physical_benchmark_exact_oracle
```

## Entry Conditions Inherited From Previous Phase

- Phase 0 froze six eligible value rows and excluded the parameterized SIR
  diagnostic row.
- Phase 1 score schema locally passed.
- Phase 1 result and this Phase 2 subplan must pass read-only review before
  execution.
- No model score has been admitted by the score program before this phase.

## Required Artifacts

Source value artifact:

- `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json`

Current implementation/test artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `tests/test_ledh_lgssm_manual_score_phase4.py`
- `bayesfilter/highdim/ledh_score_contract.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`

Expected Phase 2 artifacts:

- score artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-score-artifact-2026-07-07.json`
- score markdown summary:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-score-artifact-2026-07-07.md`
- full-run log:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-n10000-run-2026-07-07.log`
- Phase 2 result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-result-2026-07-07.md`
- Phase 3 fixed-SIR subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-subplan-2026-07-07.md`
- Phase 2 review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase2-lgssm-review-bundle-2026-07-07.md`

Expected implementation additions or repairs:

- adapter or writer that converts the LGSSM score runner output into the Phase
  1 score artifact schema;
- tests that validate the LGSSM score artifact against the Phase 1 score
  contract;
- if needed, a repair to make the active LGSSM runner's full-row identity use
  `N=10000`, matching the admitted value artifact.

## Required Checks/Tests/Reviews

Preflight/static checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  bayesfilter/highdim/ledh_score_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py
```

Existing tiny score checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

New or refreshed artifact-contract check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py -q
```

Full-row score run, only after preflight and tiny checks pass. This is a
trusted GPU/CUDA/XLA run and must be executed with escalated/trusted
permissions:

```text
python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --device-scope visible \
  --expect-device-kind gpu \
  --device /GPU:0 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --num-particles 10000 \
  --time-steps 50 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --transport-ad-mode full \
  --row-chunk-size 512 \
  --col-chunk-size 512 \
  --particle-chunk-size 256 \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float64 \
  --tf32-mode disabled \
  --warmups 0 \
  --repeats 1 \
  --score-fd-step 1.0e-5 \
  --output docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-raw-run-2026-07-07.json \
  --markdown-output docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-raw-run-2026-07-07.md
```

After the raw run, write or refresh the schema-level score artifact and run:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Diff hygiene:

```text
git diff --check -- \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  bayesfilter/highdim/ledh_score_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-result-2026-07-07.md \
  docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-subplan-2026-07-07.md
```

Review:

- bounded read-only review of material implementation diffs, Phase 2 result,
  and Phase 3 subplan before Phase 3 execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the LGSSM row produce an admitted no-tape total derivative of the same finite-`N` LEDH `log_likelihood` scalar as the admitted value artifact? |
| Baseline/comparator | Admitted LGSSM value artifact, exact Kalman value reference for context only, compact no-autodiff score route, same-scalar finite differences with fixed randomness. |
| Primary criterion | Full LGSSM score is admitted only if a schema artifact passes `validate_ledh_score_artifact(..., require_admitted=True)` against the admitted N=10000 value artifact; same-scalar FD status is pass; score vector is finite and ordered `[phi1, phi2, phi3, q_scale, r_scale]`; route provenance is `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`; value/score route is `same_route_value_score`; no tape/ForwardAccumulator/stopped partial is used; and full-run memory gate passes. |
| Veto diagnostics | Active full-row identity still uses N=1000; score artifact fails schema validation; score/value target mismatch; wrong observation policy; wrong theta coordinate; wrong parameter order; nonfinite score; FD failure; tape/autodiff/stopped partial route; GPU/device mismatch for full run; memory budget failure; raw runner status claims admission but schema artifact does not. |
| Explanatory diagnostics | Exact Kalman value comparison, runtime, compile time, memory peak, FD absolute/relative error, per-seed score, and device placement. |
| Not concluded | Exact Kalman score equality, HMC readiness, posterior correctness, scientific superiority, runtime ranking, all-algorithm comparison, or nonlinear-row validity. |
| Artifact | Phase 2 score artifact JSON/Markdown, raw-run artifact/log, tests, Phase 2 result, and Phase 3 subplan. |

## Step-By-Step Plan

1. Re-read the admitted LGSSM value artifact and active LGSSM score runner.
2. Preflight the target identity:
   - row id must be `benchmark_lgssm_exact_oracle_m3_T50`;
   - `num_particles` must be `10000`;
   - `time_steps` must be `50`;
   - batch seeds must be `81120..81124`;
   - target scalar must be `observed_data_log_likelihood_estimator`;
   - output field must be `log_likelihood`;
   - target observation policy must be `lgssm_gaussian_observation_density`;
   - theta coordinate system must be `physical_benchmark_exact_oracle`;
   - parameter order must be `[phi1, phi2, phi3, q_scale, r_scale]`.
3. Repair or override the active runner full-row identity if it still uses
   `FULL_ROW_NUM_PARTICLES = 1000`.
4. Add a schema adapter or writer for LGSSM score artifacts. The adapter must
   not transform the score numerically; it may only normalize metadata into
   the Phase 1 schema.
5. Add or refresh tests:
   - full-row identity is N=10000, not N=1000;
   - full-mode manual streaming dispatch uses the total-VJP transport route
     despite the legacy CLI constant name;
   - raw LGSSM score output can be normalized into the score artifact schema;
   - score artifact validates against the admitted value artifact;
   - stale N=1000 or tiny diagnostics cannot validate as admitted full score;
   - no-tape static/runtime sentinels remain active.
6. Run CPU-hidden tiny and schema checks.
7. Run the trusted GPU full-row score command only after the tiny/schema gates
   pass.
8. Validate the resulting score artifact with
   `validate_ledh_score_artifact(..., require_admitted=True)`.
9. Write the Phase 2 result with command, environment, memory, score, FD
   diagnostics, admission decision, nonclaims, and exact artifacts.
10. Draft the Phase 3 fixed-SIR subplan.
11. Run bounded read-only review of the Phase 2 result and Phase 3 subplan.

## Forbidden Claims/Actions

- Do not admit an LGSSM score from the old `N=1000` full-row identity.
- Do not admit a raw runner artifact unless the Phase 1 score schema validates
  it against the N=10000 value artifact.
- Do not use `GradientTape`, `ForwardAccumulator`, hidden autodiff, or stopped
  partial derivatives for the admitted score route.
- Do not use the historical manual-reverse route as the default admitted route.
- Do not claim exact Kalman score equality; the target is the LEDH finite-`N`
  estimator score.
- Do not claim HMC readiness, posterior correctness, scientific superiority,
  runtime ranking, all-algorithm comparison, or nonlinear-row validity.
- Do not proceed to full GPU run if the tiny/schema gates fail.

## Exact Next-Phase Handoff Conditions

Phase 3 fixed-SIR may start only if:

- Phase 2 result exists;
- LGSSM score is either admitted by schema validation and result review, or an
  explicit blocker result explains why it is not admitted;
- Phase 3 fixed-SIR subplan exists;
- bounded read-only review agrees that the LGSSM decision is boundary-safe and
  that Phase 3 does not inherit stale LGSSM assumptions.

## Stop Conditions

Stop and write a blocker result if:

- active LGSSM full-row identity cannot be made to match the admitted N=10000
  value artifact;
- the score artifact cannot preserve row id, target scalar, output field,
  target observation policy, theta coordinate system, and parameter order;
- no-tape provenance cannot be established;
- same-scalar FD fails beyond the predeclared tolerance;
- score is nonfinite;
- the full GPU run fails for memory/device/runtime reasons after focused
  repair;
- review finds a material issue that does not converge after five rounds;
- a human approval boundary is reached.
