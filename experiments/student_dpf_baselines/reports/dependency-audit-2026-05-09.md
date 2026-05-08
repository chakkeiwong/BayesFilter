# Student DPF Baseline Dependency Audit

## Date

2026-05-09

## Scope

This audit covers the pinned student snapshots:

- `2026MLCOE` at `020cfd7f2f848afa68432e95e6c6e747d3d2402d`;
- `advanced_particle_filter` at `d2a797c330e11befacbb736b5c86b8d03eb4a389`.

The audit is import-only and dependency-read-only.  No packages were installed.

## Environment Observed

```text
Python: 3.13.13
TensorFlow: 2.20.0
TensorFlow Probability: 0.25.0
```

Package availability by import spec:

| Package | Available |
| --- | --- |
| `numpy` | yes |
| `scipy` | yes |
| `matplotlib` | no |
| `tensorflow` | yes |
| `tensorflow_probability` | yes |
| `pytest` | yes |
| `numba` | no |

## Source Dependency Records

| Source | Dependency files found | Notes |
| --- | --- | --- |
| `2026MLCOE` | `README.md` | README mentions `pip install -r requirements.txt`, but no requirements file was found in the vendored snapshot during the max-depth audit. |
| `advanced_particle_filter` | `requirements.txt`, `README.md`, `notebooks/README.md` | Requirements list NumPy/SciPy/matplotlib, TensorFlow/TFP, Jupyter/ipykernel, pytest, and optional numba. README states TF side was tested with TF 2.16 and TFP 0.24. |

## Import Smoke Results

| Source | Import context | Status | Interpretation |
| --- | --- | --- | --- |
| `2026MLCOE` | Added `vendor/2026MLCOE` and `vendor/2026MLCOE/src` to `sys.path`; imported `filters.classical`, `filters.particle`, `filters.flow_filters`, `models.classic_ssm`. | ok | Core package imports in the current environment. |
| `advanced_particle_filter` NumPy side | Added `vendor` to `sys.path`; imported package plus `filters.kalman`, `filters.particle`, `filters.edh`, `models.linear_gaussian`. | ok | NumPy/classical side imports in the current environment. |
| `advanced_particle_filter` TensorFlow side | Added `vendor` to `sys.path`; imported `tf_filters.kalman`, `tf_filters.differentiable_particle`, `tf_models.linear_gaussian`. | ok | TensorFlow side imports with TF 2.20.0 / TFP 0.25.0, though upstream README says tested with TF 2.16 / TFP 0.24. |

## Conflict and Runtime Risks

| Risk | Source | Severity | Mitigation |
| --- | --- | --- | --- |
| Missing `matplotlib` | both likely for plotting scripts | medium | Avoid plotting scripts for first reproduction, or classify plotting examples as blocked until dependency is available. |
| Missing `numba` | `advanced_particle_filter` optional | low | Avoid optional accelerated path. |
| Missing requirements file | `2026MLCOE` | medium | Infer dependencies from imports; start with non-plotting, core examples. |
| Python version newer than student README assumptions | both | medium | Record any failures as environment incompatibilities; do not patch student code before reproduction. |
| TF/TFP version mismatch from README | `advanced_particle_filter` TF side | medium | Run minimal smoke examples first; do not interpret HMC/DPF numerical behavior as production evidence. |
| Large/generated artifacts in vendored snapshots | both | medium | Keep during first reproducibility pass; decide cleanup only after baseline report. |

## Decision

Both student snapshots are classified as `runnable_for_smoke_import`.

Next justified phase: reproduce the smallest original examples that avoid
missing plotting dependencies where possible.  If original examples require
`matplotlib`, classify the plotting part as dependency-blocked and continue
with non-plotting or test-based reproduction.
