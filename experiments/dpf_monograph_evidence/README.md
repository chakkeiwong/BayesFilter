# DPF monograph implementation evidence harness

This directory is a clean-room, non-production evidence lane for Chapter 26
implementation diagnostics. It exists to define bounded fixtures, schemas,
validators, and result artifacts for IE2--IE8.

## Scope

- Allowed imports: Python standard library and modules within
  `experiments.dpf_monograph_evidence`.
- Forbidden imports: production `bayesfilter`, student-baseline modules, and
  controlled-baseline experiment modules.
- CPU-only is mandatory unless a later accepted transition plan explicitly
  supersedes it.

## Layout

- `fixtures/`: fixture contracts and placeholder fixture assets.
- `diagnostics/`: canonical diagnostic metadata.
- `reports/`: reserved for future generated summaries.
- `runners/validate_results.py`: bounded validator and import-boundary scan.
- `results.py`: schema enums, canonical IDs, manifest rules, and validation.

## Canonical diagnostic ids

- `linear_gaussian_recovery`
- `synthetic_affine_flow`
- `pfpf_algebra_parity`
- `soft_resampling_bias`
- `sinkhorn_residual`
- `learned_map_residual`
- `hmc_value_gradient`
- `posterior_sensitivity_summary`

## Coverage statuses

Every result artifact must include a machine-readable coverage object keyed by
all canonical diagnostic ids. Allowed status values are:

- `missing`
- `blocked`
- `deferred`
- `passed`

## Required run-manifest policy

Every result JSON must embed a run manifest with these canonical keys:

- `command`
- `branch`
- `commit`
- `dirty_state_summary`
- `python_version`
- `package_versions`
- `cpu_only`
- `cuda_visible_devices`
- `gpu_devices_visible`
- `gpu_hidden_before_import`
- `pre_import_cuda_visible_devices`
- `pre_import_gpu_hiding_assertion`
- `seed_policy`
- `wall_clock_cap_seconds`
- `started_at_utc`
- `ended_at_utc`
- `artifact_paths`

CPU-only runs must set `CUDA_VISIBLE_DEVICES=-1` before scientific imports, and
manifest evidence must show `gpu_hidden_before_import=true`.

## Fixture contracts

The harness reserves fixture contracts for:

- linear-Gaussian recovery;
- synthetic affine flow;
- two-particle soft-resampling bias;
- small Sinkhorn problem;
- learned-map teacher/student residual;
- fixed scalar HMC target.

These fixtures are intentionally small, explicit, and independent of production
BayesFilter internals.
