# Phase 7 Result: KSC-SV Forward Scalar Admission

metadata_date: 2026-07-07
status: `PASSED_KSC_SV_FORWARD_SCALAR_ADMITTED_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
phase: 7

## Phase Objective

Build and admit the LEDH observed-data forward scalar row:

```text
zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000
```

Target scalar: `observed_data_log_likelihood_estimator`, reported as
`log_likelihood`.

This phase was forward-scalar-only. It did not implement or admit scores.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Phase 7 KSC-SV finite-mixture surrogate full-row value artifact is locally admitted. |
| Primary criterion status | Passed locally: canonical artifact validates with `validate_ledh_forward_scalar_artifact(..., expected_row_id=KSC_SV_ROW_ID, require_admitted=True)`. |
| Veto diagnostic status | No exact actual-SV target reuse, no generalized-SV target reuse, no raw Gaussian callback target, no augmented-noise Gaussian closure, no proposal-only KSC mixture, no tiny-artifact admission, no score evidence, and no runtime-only admission. |
| Main uncertainty | This is finite-N LEDH value evidence for the declared KSC finite-mixture surrogate row only; it does not certify exact native actual-SV likelihood, score correctness, posterior correctness, HMC readiness, Zhao-Cui source-faithfulness, scientific superiority, or runtime ranking. |
| Next justified action | Review this Phase 7 result and the Phase 8 value-integration subplan; if review agrees, execute Phase 8. |
| What is not concluded | No score admission, score correctness, exact native actual-SV likelihood, generalized-SV admission, actual-SV admission, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Entry Conditions

- Phase 1 executable schema guard passed.
- Phase 2 LGSSM artifact validates locally.
- Phase 3 fixed SIR artifact validates locally under `sir_log_scale_theta`.
- Phase 4 predator-prey artifact validates locally with `require_admitted=True`.
- Phase 5 actual-SV artifact validates locally with `require_admitted=True`.
- Phase 6 generalized-SV artifact validates locally with
  `require_admitted=True`.
- Phase 6 result and Phase 7 subplan passed read-only review before Phase 7
  execution.

## Skeptical Plan Audit

| Risk checked | Result |
| --- | --- |
| Wrong baseline | The KSC runner used `make_ksc_sv_forward_contract(...)`, transformed observations `log(y_t^2 + 1e-8)`, and the finite KSC Gaussian-mixture density as the row target. It did not borrow actual-SV, generalized-SV, LGSSM, SIR, or predator-prey evidence. |
| Proxy metric promoted | Runtime, memory, finite output, and GPU output were explanatory until the canonical artifact passed schema validation and replay. |
| Missing stop conditions | The subplan stopped on exact actual-SV target reuse, generalized-SV reuse, raw Gaussian target reuse, missing transform offset, wrong row settings, nonfinite output, score creep, and approval boundaries. |
| Hidden assumptions | Full admission required explicit `--run-scope full-row-admission` and exact seeds, `T`, `N`, theta, transform offset, target policy, and KSC mixture target flags. |
| Artifact mismatch | The result is a canonical Phase 1 schema artifact plus replay tests, not stdout or a score-inclusive leaderboard rebuild. |

Audit status: passed for Phase 7 local admission.

## Implementation

Patched and used:

- `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py`
- `tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py`
- `tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py`
- `scripts/run_gpu_benchmark.sh`

The runner has two explicit scopes:

- `tiny-smoke`, which emits `tiny_executed_not_full_row` and rejects accidental
  full-row admission;
- `full-row-admission`, which requires exact `T=1000`, `N=10000`, seeds
  `[81120,81121,81122,81123,81124]`, theta
  `[0.2533471031357997,-0.916290731874155]`, transform offset `1e-8`, and
  finite output before emitting `n10000_same_target_value_admitted`.

The KSC target bridge is:

```text
z_t = log(y_t^2 + 1e-8)
z_t | x_t = log(beta^2) + x_t + epsilon_t
epsilon_t ~ finite 7-component KSC Gaussian mixture
```

The finite KSC Gaussian mixture is the declared target likelihood for this row.
It is not exact native actual-SV likelihood. The LEDH flow uses a Gaussianized
KSC log-square surface only as the proposal surface. The target correction uses
`finite_ksc_log_chi_square_gaussian_mixture_log_density`.

## Canonical Artifact

Wrote canonical Phase 1 schema artifacts:

- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.md`

The canonical artifact records:

- `row_id = zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`;
- `admission_status = n10000_same_target_value_admitted`;
- `theta_coordinate_system = synthetic_unconstrained`;
- `theta_values = [0.2533471031357997, -0.916290731874155]`;
- `target_observation_policy = ksc_log_chi_square_gaussian_mixture_surrogate`;
- `flow_observation_policy = gaussianized_ksc_log_square_surrogate_flow_observation`;
- `target_observation_density = finite_ksc_log_chi_square_gaussian_mixture_log_density`;
- `target_density_used_for_correction = true`;
- `target_transform = log_y_square_plus_offset`;
- `transform_offset = 1e-8`;
- `mixture_component_count = 7`;
- `ksc_mixture_is_target_likelihood = true`;
- `actual_sv_exact_log_chi_square_target_used = false`;
- `generalized_sv_target_used = false`;
- `legacy_raw_gaussian_callback_used = false`;
- `T = 1000`;
- `N = 10000`;
- `batch_seeds = [81120, 81121, 81122, 81123, 81124]`;
- `log_likelihood_by_seed =
  [-2288.165771484375, -2287.877685546875, -2287.852294921875,
  -2287.529296875, -2288.34375]`;
- `average_log_likelihood_by_seed =
  [-2.288165771484375, -2.287877685546875, -2.287852294921875,
  -2.287529296875, -2.28834375]`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- `finite_output = true`;
- streaming transport with `dense_transport_matrix_materialized = false`.

Validation:

```text
validate_ledh_forward_scalar_artifact(
    artifact,
    expected_row_id=KSC_SV_ROW_ID,
    require_admitted=True,
)
```

passed during artifact generation and during the mandatory replay test.

## Execution Evidence

Trusted GPU tiny-smoke command was run through the approved wrapper:

```text
bash scripts/run_gpu_benchmark.sh ledh_phase7_ksc_tiny_smoke
```

Tiny-smoke result:

- `admission_status = tiny_executed_not_full_row`;
- `T = 4`;
- `N = 128`;
- seed `[81120]`;
- `log_likelihood_by_seed = [-7.610705852508545]`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- finite output passed;
- not admitted as a full row.

Trusted GPU full-row command was run through the approved wrapper:

```text
bash scripts/run_gpu_benchmark.sh ledh_phase7_ksc_full_row
```

Full-row result:

- passed;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- compile plus first call: `1054.9561004990246` seconds;
- warm call: `1054.2882441349793` seconds;
- finite output and schema validation passed.

## Local Checks

Compile check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py
```

Result: passed.

Focused Phase 7 replay:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py -q
```

Result:

```text
5 passed, 2 warnings
```

Through-Phase-7 replay check:

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
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py -q
```

Result:

```text
45 passed, 2 warnings
```

Diff hygiene:

```text
git diff --check -- \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py \
  scripts/run_gpu_benchmark.sh \
  docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.md \
  docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json \
  docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md \
  docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json
```

Result: passed.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the declared KSC finite Gaussian-mixture surrogate SV row produce an executable same-target observed-data likelihood artifact under LEDH? |
| Answer | Yes, locally. The canonical full-row artifact is admitted under the shared Phase 1 schema. |
| Baseline/comparator | `make_ksc_sv_forward_contract(...)`, `KSCMixtureTransformedSVSSM`, `ksc_1998_log_chi_square_mixture()`, `transformed_sv_observations(..., offset=1e-8)`, the actual-SV source observations transformed as `log(y_t^2 + 1e-8)`, and Phase 1 schema validator. |
| Primary criterion | Passed locally: full-row JSON artifact validates with `require_admitted=True`, has finite `log_likelihood_by_seed`, row id `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`, theta coordinate `synthetic_unconstrained`, theta values `[0.2533471031357997,-0.916290731874155]`, `T=1000`, `N=10000`, seeds `[81120,81121,81122,81123,81124]`, target policy `ksc_log_chi_square_gaussian_mixture_surrogate`, finite KSC mixture target density, transform offset `1e-8`, target-density correction, and GPU output device. |
| Veto diagnostics | No tiny artifact was admitted; exact actual-SV target density was not used; generalized-SV target density was not used; raw Gaussian callback was not used as target; KSC mixture was target likelihood, not proposal-only; transform offset matched `1e-8`; theta/seeds/T/N matched; replay test reads disk artifact; no score fields were used as value evidence; runtime/memory were explanatory only. |
| Explanatory diagnostics | Runtime, compile time, memory, warm-call timing, and Monte Carlo variability across seeds. |
| Not concluded | No score admission, score correctness, exact native actual-SV likelihood, actual-SV admission, generalized-SV admission, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Phase 8 Handoff

Phase 8 value integration may begin only after read-only review agrees with
this Phase 7 result and the Phase 8 subplan.

Phase 8 must consume only admitted forward-scalar artifacts and explicit
blocker/diagnostic records. It must not reuse the old score-inclusive
leaderboard builder unchanged. It must not treat the legacy parameterized SIR
diagnostic row as a main leaderboard row.

## Nonclaims

- No score route is implemented or admitted.
- This is not exact native actual-SV likelihood evidence.
- This is not actual-SV, generalized-SV, LGSSM, SIR, or predator-prey evidence.
- No score-inclusive leaderboard is rebuilt.
- No HMC readiness, posterior correctness, scientific superiority, or runtime
  ranking is claimed.
