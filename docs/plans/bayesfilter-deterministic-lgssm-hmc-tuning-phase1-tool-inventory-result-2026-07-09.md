# Phase 1 Result: Tool Inventory And API Binding

Date: 2026-07-09

Status: `PASSED`

## Scope

Phase 1 performed read-only source inspection of existing BayesFilter tuning
tools. It did not run tuning, HMC, NeuTra training, or GPU/CUDA commands.

## API Inventory

| Decision point | Existing tool | Source anchor | Binding decision |
| --- | --- | --- | --- |
| Local quadratic geometry initializer | `LowRankSPDQuadraticGeometryConfig`, `fit_low_rank_spd_quadratic_geometry` | `bayesfilter/inference/quadratic_geometry.py:31`, `bayesfilter/inference/quadratic_geometry.py:193` | Use directly in deterministic driver Phase B. |
| Geometry payload and arrays | `LowRankSPDQuadraticGeometryResult.payload(include_arrays=True)` | `bayesfilter/inference/quadratic_geometry.py:114`, `bayesfilter/inference/quadratic_geometry.py:159` | Persist JSON with arrays, diagnostics, and stable hash. |
| Precision-to-covariance mass conversion | `covariance_from_precision` | `bayesfilter/inference/mass_matrix.py:68` | Use accepted quadratic precision to create mass covariance. |
| Negative Hessian adapter if available | `covariance_from_negative_hessian` | `bayesfilter/inference/mass_matrix.py:112` | Optional deterministic input path; not required for Phase 2 schema. |
| Whitening/factor construction | `whitening_from_covariance` | `bayesfilter/inference/mass_matrix.py:205` | Use only for deterministic coordinate transforms when required by HMC driver. |
| Geometry-derived HMC initializer | `HMCGeometryInitializationConfig`, `HMCGeometryInitializationResult` | `bayesfilter/inference/hmc_kernel_tuning.py:1305`, `bayesfilter/inference/hmc_kernel_tuning.py:1393` | Use through `tune_hmc_kernel` or direct stage only if driver needs explicit artifacting. |
| Serious staged kernel tuning | `HMCKernelTuningConfig.serious`, `tune_hmc_kernel` | `bayesfilter/inference/hmc_kernel_tuning.py:3238`, `bayesfilter/inference/hmc_kernel_tuning.py:3538`, `bayesfilter/inference/hmc_kernel_tuning.py:7098` | Use as the owner of step size, leapfrog, mass, screen, verification, and repair mechanics. |
| Fixed-mass budget ladder | `FixedMassHMCTuningBudgetLadderConfig`, `run_fixed_mass_hmc_tuning_budget_ladder` | `bayesfilter/inference/hmc_budget_ladder.py:104`, `bayesfilter/inference/__init__.py:113` | Use only through existing staged tuning unless explicit lower-level driver evidence is needed. |
| Operational HMC diagnostics | `summarize_hmc_diagnostics`, `screen_hmc_diagnostics`, `classify_hmc_screen` | `bayesfilter/inference/hmc_diagnostics.py:64`, `bayesfilter/inference/hmc_diagnostics.py:133`, `bayesfilter/inference/hmc_diagnostics.py:211` | Use for role-labeled final diagnostics and fail-closed screens. |
| Public exports | `bayesfilter.inference.__init__` exports budget ladder and staged tuning APIs | `bayesfilter/inference/__init__.py:113`, `bayesfilter/inference/__init__.py:128` | Prefer public imports from `bayesfilter.inference` where available. |

## Binding Rules For The Deterministic Driver

1. The driver must call `HMCKernelTuningConfig.serious(...)` with
   `chain_execution_mode="tf_function"` and `use_xla=True`.
2. Existing config defaults are not sufficient: `HMCKernelTuningConfig` and
   `FixedMassHMCTuningBudgetLadderConfig` both default `use_xla=False` for
   generality. The driver must override and verify XLA metadata.
3. The driver must not expose manual post-result tuning choices. If the staged
   tuner fails or hits caps, the result is a deterministic failure/blocker.
4. Geometry/mass artifacts are initializer evidence only. They cannot promote
   posterior recovery or convergence.
5. Final recovery remains Phase 8 only: all parameters must have
   `R_hat <= 1.01`, ESS floors satisfied, and posterior mean within
   `3 * posterior_sd` of truth.

## Local Checks

- Confirmed relevant APIs are exported from `bayesfilter.inference`.
- Confirmed quadratic geometry, mass-matrix, staged kernel tuning, budget
  ladder, and HMC diagnostic modules exist.
- Confirmed source-level nonclaims in geometry and diagnostics prevent geometry
  or bounded screens from being treated as convergence evidence.

## Skeptical Audit

| Risk | Verdict |
| --- | --- |
| Wrong baseline | Controlled by Phase 2 config and Phase 3 fixture; not decided in Phase 1. |
| Proxy metric promoted | Controlled: geometry, mass, acceptance, compile success, and diagnostics remain non-promoting. |
| Missing stop condition | Phase 6-8 subplans stop on missing approval, XLA fallback, cap failure, and incomplete artifacts. |
| Agent hand-tuning | Blocked by binding to deterministic driver plus existing staged tuner. |
| Environment mismatch | Driver must override `use_xla=True` and record CPU-hidden/GPU status. |

## Decision Table

| Field | Decision |
| --- | --- |
| Phase decision | `PASS_TO_PHASE2` |
| Primary criterion status | Existing BayesFilter tools cover required tuning decision points |
| Veto diagnostic status | No manual-only tuning gap found |
| Main uncertainty | Burn-in/sampling controller may need new deterministic driver code, but not a new agent decision process |
| Next justified action | Define config schema and create fixed JSON config skeleton |
| What is not concluded | No tuning success, HMC readiness, posterior convergence, recovery, runtime feasibility, or scientific claim |
