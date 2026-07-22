# PP-UKF True HMC Validation Preflight Result

Date: 2026-07-22

Plan: `docs/plans/bayesfilter-pp-ukf-true-hmc-validation-plan-2026-07-22.md`

Artifact: `docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-02/`

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Preflight accepted; retained campaign blocked | Candidate/source/controller bindings and focused tests pass | Full-run budget and fresh-partition veto active | ESS/reference thresholds are not yet declared; no draws exist | Amend plan with budget, untouched validation partition, and thresholds, then launch ten candidates sequentially | No posterior, convergence, sampler-ranking, or scientific claim |

The preflight reconstructed the required unranked set
`L=(5,9,12,13,14,17,18,19,24,25)`. Primaries `L=(5,9,13,18,25)` use
independently tuned epsilon values. Coverage points `L=(12,14,17,19,24)`
inherit their parent epsilon bit-for-bit. No acceptance, ESS, runtime, or tail
ranking was performed.

The artifact records target signature
`d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`, transport
SHA-256 `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`,
fixed identity metric, four chains, warm-up minimum 2,000, retained minimum
1,000, warm-up R-hat limit 1.05, retained R-hat limit 1.01, XLA enabled, and
the TFP native-divergence limitation. Native TFP divergence is
`not_exposed_by_tfp_hamiltonian_monte_carlo`, not zero.

## Inference Status

| Evidence class | Status | Interpretation |
| --- | --- | --- |
| Hard veto screen | Preflight passed; full campaign vetoed by missing fresh partition and budget | No candidate has been admitted or rejected on HMC behavior |
| Statistically supported ranking | None | No retained draws or uncertainty analysis exist |
| Descriptive-only differences | Epsilon/provenance differences only | These are frozen controls, not performance evidence |
| Default readiness | Not evaluated | The preflight cannot support a default change |
| Next evidence needed | New budget, untouched claim partition, declared ESS/reference gates, and the ten-candidate sequential run | Preserve each candidate's warm-up and retained artifacts separately |

## Checks Executed

- `python -m py_compile docs/benchmarks/preflight_pp_ukf_true_hmc_validation_20260722.py tests/test_pp_ukf_true_hmc_preflight.py`
- `pytest -q tests/test_neutra_hmc.py tests/test_frozen_kernel_validation.py tests/test_pp_ukf_statistical_compatibility_guard_repair_driver.py tests/test_pp_ukf_true_hmc_preflight.py` -> `14 passed`
- Trusted `nvidia-smi` probe -> one RTX 4080 SUPER, 16,376 MiB total, 2,866 MiB used, 29% utilization at probe time.
- Trusted TensorFlow probe with `TF_FORCE_GPU_ALLOW_GROWTH=true` -> TensorFlow 2.19.1, one visible physical GPU, verified memory growth, logical `/device:GPU:0`.
- Preflight command -> `status=blocked_before_sampling_missing_fresh_partition_and_budget`.

## Post-Run Red Team

The strongest alternative explanation is that the remaining budget estimate is
overly conservative or that an existing validation partition is merely named
differently. That does not justify sampling: the artifact currently contains
no repository-issued fresh partition signature and no prospective ESS/reference
thresholds. A later result could overturn this blocker only by supplying those
two inputs and a bounded budget; it cannot be overturned by the preflight's
successful configuration checks.
