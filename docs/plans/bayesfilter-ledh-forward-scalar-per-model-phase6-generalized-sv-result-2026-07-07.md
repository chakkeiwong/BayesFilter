# Phase 6 Result: Generalized-SV Forward Scalar Admission

metadata_date: 2026-07-07
status: `PASSED_GENERALIZED_SV_FORWARD_SCALAR_ADMITTED_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
phase: 6

## Phase Objective

Build and admit the LEDH observed-data forward scalar row:

```text
zhao_cui_generalized_sv_synthetic_from_estimated_values
```

Target scalar: `observed_data_log_likelihood_estimator`, reported as
`log_likelihood`.

This phase was forward-scalar-only. It did not implement or admit scores.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Phase 6 generalized-SV full-row value artifact is locally admitted. |
| Primary criterion status | Passed locally: canonical artifact validates with `validate_ledh_forward_scalar_artifact(..., expected_row_id=GENERALIZED_SV_ROW_ID, require_admitted=True)`. |
| Veto diagnostic status | No actual-SV target borrowing, no KSC mixture borrowing, no native generalized-SV dense-fixture substitution, no SP500 benchmark observations, no author-default truth substitution, no log-square proposal surface promoted to target likelihood, no tiny-artifact admission, no score evidence, and no runtime-only admission. |
| Main uncertainty | This is finite-N LEDH value evidence only; it does not certify score correctness, posterior correctness, HMC readiness, Zhao-Cui source-faithfulness, scientific superiority, or runtime ranking. |
| Next justified action | Review this Phase 6 result and the Phase 7 KSC-SV subplan; if review agrees, execute Phase 7. |
| What is not concluded | No score admission, score correctness, KSC admission, actual-SV admission, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Entry Conditions

- Phase 1 executable schema guard passed.
- Phase 2 LGSSM artifact validates locally.
- Phase 3 fixed SIR artifact validates locally under `sir_log_scale_theta`.
- Phase 4 predator-prey artifact validates locally with `require_admitted=True`.
- Phase 5 actual-SV artifact validates locally with `require_admitted=True`.
- Phase 5 result and Phase 6 subplan passed read-only review before Phase 6
  execution.

## Skeptical Plan Audit

| Risk checked | Result |
| --- | --- |
| Wrong baseline | The generalized-SV runner used `make_generalized_sv_forward_contract(...)` and the source-route prior-mean row, not actual-SV, KSC, native dense, SP500, or author-default substitutes. |
| Proxy metric promoted | Runtime, memory, finite output, and GPU output were explanatory until the canonical artifact passed schema validation and replay. |
| Missing stop conditions | The subplan stopped on wrong target, wrong timing convention, nonfinite output, wrong row settings, target-density substitution, OOM/runtime failure, score creep, and approval boundaries. |
| Hidden assumptions | Full admission required explicit `--run-scope full-row-admission` and exact seeds, `T`, `N`, theta, target policy, and source-route timing convention. |
| Artifact mismatch | The result is a canonical Phase 1 schema artifact plus replay tests, not stdout or a leaderboard rebuild. |

Audit status: passed for Phase 6 local admission.

## Implementation

Patched and used:

- `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py`
- `tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py`
- `tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py`

The runner has two explicit scopes:

- `tiny-smoke`, which emits `tiny_executed_not_full_row` and rejects accidental
  full-row settings;
- `full-row-admission`, which requires exact `T=1008`, `N=10000`, seeds
  `[81120,81121,81122,81123,81124]`, theta
  `[1.0824113944610982,-2.076793740349318,0.0]`, and finite output before
  emitting `n10000_same_target_value_admitted`.

The generalized-SV target bridge is:

```text
y_t | x_t ~ Normal(0, exp(tau * x_t))
```

with source-route prior-mean physical truth:

```text
gamma = 0.8604651162790697
tau = 0.12533141373155005
mu = 0.0
```

The implementation mirrors the existing dataset timing convention: initial
particles are stationary previous states; every recorded observation, including
`t=0`, first applies the AR(1) transition before target weighting.

The LEDH flow uses `log(y_t^2 + 1e-6)` only as a proposal surface. The target
correction uses `raw_zero_mean_generalized_sv_prior_mean_normal_log_density`.

## Canonical Artifact

Wrote canonical Phase 1 schema artifacts:

- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.md`

The canonical artifact records:

- `row_id = zhao_cui_generalized_sv_synthetic_from_estimated_values`;
- `admission_status = n10000_same_target_value_admitted`;
- `theta_coordinate_system = source_route_active_transformed_prior_mean`;
- `theta_values = [1.0824113944610982, -2.076793740349318, 0.0]`;
- `target_observation_policy = source_route_prior_mean_generalized_sv`;
- `flow_observation_policy = log_square_gaussian_surrogate_for_ledh_flow_only`;
- `target_observation_density = raw_zero_mean_generalized_sv_prior_mean_normal_log_density`;
- `target_density_used_for_correction = true`;
- `flow_observation_transform = log(y_t^2 + 1e-6)`;
- `T = 1008`;
- `N = 10000`;
- `batch_seeds = [81120, 81121, 81122, 81123, 81124]`;
- `log_likelihood_by_seed =
  [-1438.90966796875, -1438.8360595703125, -1438.908935546875,
  -1439.0206298828125, -1438.9456787109375]`;
- `average_log_likelihood_by_seed =
  [-1.427489749968998, -1.427416725764199, -1.427489023359995,
  -1.427599831232949, -1.4275254749116444]`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- `finite_output = true`.

Validation:

```text
validate_ledh_forward_scalar_artifact(
    artifact,
    expected_row_id=GENERALIZED_SV_ROW_ID,
    require_admitted=True,
)
```

passed during artifact generation and during the mandatory replay test.

## Execution Evidence

Trusted GPU full-row command:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py \
  --run-scope full-row-admission \
  --time-steps 1008 \
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
  --output docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.md
```

Result:

- passed;
- XLA compiled on `/device:GPU:0`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- compile plus first call: `1187.9543538549915` seconds;
- warm call: `1048.247488206951` seconds;
- finite output and schema validation passed.

## Local Checks

Compile check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py
```

Result: passed.

Focused Phase 6 replay:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py -q
```

Result:

```text
5 passed, 2 warnings in 5.02s
```

Through-Phase-6 replay check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py -q
```

Result:

```text
40 passed, 2 warnings in 2.64s
```

Diff hygiene:

```text
git diff --check -- \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py \
  docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.md \
  docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json \
  docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md \
  docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json
```

Result: passed.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the source-route prior-mean generalized-SV LEDH adapter produce a full-row executable same-target observed-data likelihood artifact? |
| Answer | Yes, locally. The canonical full-row artifact is admitted under the shared Phase 1 schema. |
| Baseline/comparator | `make_generalized_sv_forward_contract(...)`, `_generalized_sv_prior_mean_dataset(...)`, source-route DPF generalized-SV callback convention, and Phase 1 schema validator. |
| Primary criterion | Passed locally: full-row JSON artifact validates with `require_admitted=True`, has finite `log_likelihood_by_seed`, row id `zhao_cui_generalized_sv_synthetic_from_estimated_values`, theta coordinate `source_route_active_transformed_prior_mean`, theta values `[1.0824113944610982,-2.076793740349318,0.0]`, `T=1008`, `N=10000`, seeds `[81120,81121,81122,81123,81124]`, `target_observation_policy=source_route_prior_mean_generalized_sv`, `target_density_used_for_correction=true`, raw zero-mean generalized-SV target observation density, and GPU output device. |
| Veto diagnostics | No tiny artifact was admitted; actual-SV evidence was not used; KSC evidence was not used; native generalized-SV dense fixture was not used; SP500 returns were not used as benchmark observations; author defaults were not used as truth; log-square Gaussianized observation was proposal-only; target density was used for correction; theta/seeds/T/N matched; replay test reads disk artifact; no score fields were used as value evidence; runtime/memory were explanatory only. |
| Explanatory diagnostics | Runtime, compile time, memory, warm-call timing, and Monte Carlo variability across seeds. |
| Not concluded | No score admission, score correctness, KSC admission, actual-SV admission, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Phase 7 Handoff

Phase 7 KSC-SV may begin only after read-only review agrees with this Phase 6
result and the Phase 7 subplan.

Phase 7 must not reuse actual-SV or generalized-SV target evidence. It must use
the KSC surrogate row target:

```text
ksc_log_chi_square_gaussian_mixture_surrogate
```

and must treat the finite Gaussian mixture as the declared target likelihood
for that row, not as exact native actual-SV likelihood.

## Nonclaims

- No score route is implemented or admitted.
- This is not actual-SV, KSC, LGSSM, SIR, or predator-prey evidence.
- This is not KSC surrogate likelihood evidence.
- This is not native generalized-SV dense fixture evidence.
- No leaderboard is rebuilt.
- No HMC readiness, posterior correctness, scientific superiority, or runtime
  ranking is claimed.
