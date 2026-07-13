# Phase 5 Result: Actual-SV Forward Scalar Admission

metadata_date: 2026-07-07
status: `PASSED_ACTUAL_SV_FORWARD_SCALAR_ADMITTED_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
phase: 5

## Phase Objective

Build and admit the exact transformed actual-SV LEDH observed-data forward
scalar row:

```text
zhao_cui_sv_actual_nongaussian_T1000
```

Target scalar: `observed_data_log_likelihood_estimator`, reported as
`log_likelihood`.

This phase was forward-scalar-only. It did not implement or admit scores.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Phase 5 actual-SV full-row value artifact is locally admitted. |
| Primary criterion status | Passed locally: canonical artifact validates with `validate_ledh_forward_scalar_artifact(..., expected_row_id=ACTUAL_SV_ROW_ID, require_admitted=True)`. |
| Veto diagnostic status | No raw Gaussian target correction, no KSC finite mixture, no augmented-noise Gaussian closure, no positive transform offset, no tiny-artifact admission, no score evidence, and no runtime-only admission. |
| Main uncertainty | This is finite-N LEDH value evidence only; it does not certify score correctness, posterior correctness, HMC readiness, or scientific superiority. |
| Next justified action | Review this Phase 5 result and the Phase 6 generalized-SV subplan; if review agrees, execute Phase 6. |
| What is not concluded | No score admission, score correctness, generalized-SV admission, KSC admission, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Entry Conditions

- Phase 1 executable schema guard passed.
- Phase 2 LGSSM artifact validates locally.
- Phase 3 fixed SIR artifact validates locally under `sir_log_scale_theta`.
- Phase 4 predator-prey artifact validates locally with `require_admitted=True`.
- Phase 5 exact transformed actual-SV tiny adapter smoke passed with
  `admission_status=tiny_executed_not_full_row`.
- Phase 5 full-row subplan passed bounded read-only review before the full run.

## Skeptical Plan Audit

| Risk checked | Result |
| --- | --- |
| Wrong baseline | The full run used the exact transformed actual-SV adapter route that passed tiny smoke; it did not borrow KSC, raw-Gaussian, predator-prey, SIR, or LGSSM evidence. |
| Proxy metric promoted | Runtime, memory, finite output, and GPU output were explanatory until the canonical artifact passed schema validation and replay. |
| Missing stop conditions | The subplan stopped on wrong target, nonfinite output, wrong row settings, target-density substitution, OOM/runtime failure, score creep, and approval boundaries. |
| Hidden assumptions | Full admission required explicit `--run-scope full-row-admission` and exact seeds, `T`, `N`, theta, and target policy. |
| Artifact mismatch | The result is a canonical Phase 1 schema artifact plus replay tests, not stdout or a leaderboard rebuild. |

Audit status: passed after the full-row metadata repair described below.

## Implementation

Patched and used:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`
- `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py`
- `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py`

The runner now has two explicit scopes:

- `tiny-smoke`, which emits `tiny_executed_not_full_row` and rejects accidental
  full-row settings;
- `full-row-admission`, which requires exact `T=1000`, `N=10000`, seeds
  `[81120,81121,81122,81123,81124]`, theta
  `[0.2533471031357997,-0.916290731874155]`, and finite output before emitting
  `n10000_same_target_value_admitted`.

The actual-SV target bridge is:

```text
z_t = log(y_t^2)
z_t - 2 log(beta) - x_t ~ log(chi_square_1)
```

The LEDH flow uses a Gaussianized log-square observation surface only as the
proposal surface. The target correction uses
`exact_log_chi_square_log_density`, not the raw Gaussian callback, not KSC, and
not an augmented-noise Gaussian closure.

## Canonical Artifact

Wrote canonical Phase 1 schema artifacts:

- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.md`

The canonical artifact records:

- `row_id = zhao_cui_sv_actual_nongaussian_T1000`;
- `admission_status = n10000_same_target_value_admitted`;
- `theta_coordinate_system = synthetic_unconstrained`;
- `theta_values = [0.2533471031357997, -0.916290731874155]`;
- `target_observation_policy = transformed_actual_sv_log_y_square`;
- `flow_observation_policy = gaussianized_exact_log_square_actual_sv_flow_observation`;
- `target_observation_density = exact_log_chi_square_log_density`;
- `target_density_used_for_correction = true`;
- `transform_offset = 0.0`;
- `T = 1000`;
- `N = 10000`;
- `batch_seeds = [81120, 81121, 81122, 81123, 81124]`;
- `log_likelihood_by_seed =
  [-2290.10205078125, -2289.888916015625, -2289.83154296875,
  -2289.517333984375, -2290.427490234375]`;
- `average_log_likelihood_by_seed =
  [-2.29010205078125, -2.289888916015625, -2.28983154296875,
  -2.289517333984375, -2.290427490234375]`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- `finite_output = true`.

Validation:

```text
validate_ledh_forward_scalar_artifact(
    artifact,
    expected_row_id=ACTUAL_SV_ROW_ID,
    require_admitted=True,
)
```

passed during artifact generation and during the mandatory replay test.

## Execution Evidence

Trusted GPU full-row command:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  --run-scope full-row-admission \
  --time-steps 1000 \
  --num-particles 10000 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --row-chunk-size 512 \
  --col-chunk-size 512 \
  --particle-chunk-size 512 \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --device /GPU:0 \
  --expect-device-kind gpu \
  --output docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.md
```

Result:

- passed;
- XLA compiled on `/device:GPU:0`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- compile plus first call: `1105.1444279109128` seconds;
- warm call: `1066.3097243178636` seconds;
- finite output and schema validation passed.

## Metadata Repair

After the full-row run, the artifact still carried the tiny-smoke nonclaim
`not full actual-SV row admission`. That was wrong relative to the admitted
full-row artifact. I repaired the runner to emit separate nonclaim sets for
tiny and full-row scopes, repaired the full JSON/markdown metadata without
changing numerical values, and added a replay assertion that admitted full-row
artifacts must not carry the tiny-only nonclaim.

Read-only review then found the first repair was incomplete: the top-level
`nonclaims` were clean, but the cached
`validator_normalized_core.nonclaims` inside the JSON still carried the
tiny-only nonclaim. I repaired that cached block to match the top-level
nonclaims and strengthened the full-row replay test to assert equality between
top-level and cached normalized nonclaims.

## Local Checks

Compile check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py
```

Result: passed.

Through-Phase-5 replay check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py -q
```

Result:

```text
35 passed, 2 warnings in 2.69s
```

Diff hygiene:

```text
git diff --check -- \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-full-row-subplan-2026-07-07.md \
  docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.md \
  docs/plans/bayesfilter-ledh-forward-scalar-per-model-visible-execution-ledger-2026-07-07.md
```

Result: passed.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the exact transformed actual-SV LEDH adapter produce a full-row executable same-target observed-data likelihood artifact? |
| Answer | Yes, locally. The canonical full-row artifact is admitted under the shared Phase 1 schema. |
| Baseline/comparator | Phase 5 tiny adapter smoke, `make_actual_sv_forward_contract(...)`, `StochasticVolatilitySSM`, `exact_transformed_sv_observations`, `exact_log_chi_square_log_density`, and Phase 1 schema validator. |
| Primary criterion | Passed locally: full-row JSON artifact validates with `require_admitted=True`, has finite `log_likelihood_by_seed`, row id `zhao_cui_sv_actual_nongaussian_T1000`, theta coordinate `synthetic_unconstrained`, theta values `[0.2533471031357997,-0.916290731874155]`, `T=1000`, `N=10000`, seeds `[81120,81121,81122,81123,81124]`, `target_observation_policy=transformed_actual_sv_log_y_square`, `target_density_used_for_correction=true`, and GPU output device. |
| Veto diagnostics | No tiny artifact was admitted; full run did not use raw Gaussian observation likelihood as target correction; KSC finite mixture was not used; augmented-noise Gaussian closure was not used; transform offset was `0.0`; target density was used for correction; theta/seeds/T/N matched; replay test reads disk artifact; no score fields were used as value evidence; runtime/memory were explanatory only. |
| Explanatory diagnostics | Runtime, compile time, memory, ESS placeholders in value-only mode, warm-call timing, and Monte Carlo variability across seeds. |
| Not concluded | No score admission, score correctness, generalized-SV admission, KSC admission, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Phase 6 Handoff

Phase 6 generalized-SV may begin only after read-only review agrees with this
Phase 5 result and the Phase 6 subplan.

Phase 6 must not reuse actual-SV target evidence. It must use the generalized
SV row target:

```text
source_route_prior_mean_generalized_sv
```

and must treat any log-square Gaussianized observation as an LEDH proposal
surface only, not as the target likelihood.

## Nonclaims

- No score route is implemented or admitted.
- This is not generalized-SV, KSC, LGSSM, SIR, or predator-prey evidence.
- This is not KSC surrogate likelihood evidence.
- This is not raw Gaussian observation likelihood evidence.
- This is not augmented-noise Gaussian-closure evidence.
- No leaderboard is rebuilt.
- No HMC readiness, posterior correctness, scientific superiority, or runtime
  ranking is claimed.
