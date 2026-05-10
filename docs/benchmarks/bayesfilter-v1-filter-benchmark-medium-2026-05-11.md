# BayesFilter v1 Filter CPU Benchmark

Purpose: record CPU-only timing and process-memory metadata for BayesFilter v1 filtering candidates.

## Claim Scope

Benchmark artifact only.  Not a client switch-over, GPU, XLA-GPU, or HMC readiness claim.

## Configuration

```text
repeats = 2
timesteps = 12
state_dim = 4
observation_dim = 3
parameter_dim = 3
dtype = float64
```

## Environment

```text
python = 3.11.14
platform = Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
tensorflow = 2.19.1
cuda_visible_devices = -1
logical_devices = [{'name': '/device:CPU:0', 'device_type': 'CPU'}]
```

CPU-only benchmark harness; GPU/XLA-GPU claims require separate escalated probes and matching shapes.

Process-level RSS snapshots are recorded before and after each benchmark row.  max_rss is process high-water RSS, so row deltas are diagnostic metadata, not isolated allocation profiles.

## Result

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-filter-benchmark-medium-2026-05-11.json`.

| Benchmark | Backend | Mode | First s | Steady s | RSS delta MB | Max RSS delta MB | Points | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| linear_qr_value | tf_qr | eager | 0.9849 | 0.0024 | 51.7500 | 51.7500 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | eager | 57.5890 | 0.0214 | 3335.7656 | 3335.7656 |  | ok |
| linear_svd_value | tf_svd | eager | 1.9792 | 0.0025 | -991.4492 | 3.2500 |  | ok |
| svd_cubature_value | tf_svd_cubature | eager | 0.0676 | 0.0441 | 1.0000 | 0.0000 | 6 | ok |
| svd_ukf_value | tf_svd_ukf | eager | 0.0465 | 0.0453 | 0.0000 | 0.0000 | 7 | ok |
| svd_cut4_value | tf_svd_cut4 | eager | 0.0504 | 0.0452 | 0.5000 | 0.0000 | 14 | ok |
| linear_qr_value | tf_qr | graph | 0.3425 | 0.0024 | 19.7500 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | graph | 11.0857 | 0.0021 | 1459.7891 | 486.3398 |  | ok |
| linear_svd_value | tf_svd | graph | 0.3434 | 0.0020 | -79.8711 | 0.2500 |  | ok |
| svd_cubature_value | tf_svd_cubature | graph | 1.5059 | 0.0032 | -902.8320 | 0.0000 | 6 | ok |
| svd_ukf_value | tf_svd_ukf | graph | 1.1135 | 0.0025 | 22.3242 | 0.0000 | 7 | ok |
| svd_cut4_value | tf_svd_cut4 | graph | 0.9892 | 0.0034 | 63.2188 | 0.0000 | 14 | ok |

## Interpretation

Rows with `status = ok` completed for the declared fixed shape.  First-call timing includes tracing/initialization effects for graph mode.  Steady timing uses calls after the first observation.  Memory fields are process-level diagnostics and should not be interpreted as isolated per-backend allocation profiles.

This artifact does not certify MacroFinance/DSGE switch-over, GPU/XLA-GPU readiness, or HMC readiness.
