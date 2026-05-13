# BayesFilter v1 Filter CPU Benchmark

Purpose: record CPU-only timing and process-memory metadata for BayesFilter v1 filtering candidates.

## Claim Scope

Benchmark artifact only.  Not a client switch-over, GPU, XLA-GPU, or HMC readiness claim.

## Configuration

```text
repeats = 2
timesteps = 4
state_dim = 2
observation_dim = 2
parameter_dim = 2
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

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.json`.

| Benchmark | Backend | Mode | First s | Steady s | RSS delta MB | Max RSS delta MB | Points | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| linear_qr_value | tf_qr | eager | 0.4476 | 0.0015 | 28.2500 | 28.2500 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | eager | 9.6282 | 0.0054 | 539.7109 | 539.7109 |  | ok |
| linear_svd_value | tf_svd | eager | 0.5085 | 0.0017 | -145.6094 | 3.2500 |  | ok |
| svd_cubature_value | tf_svd_cubature | eager | 0.0396 | 0.0155 | 1.2500 | 0.0000 | 6 | ok |
| svd_ukf_value | tf_svd_ukf | eager | 0.0156 | 0.0171 | 0.0000 | 0.0000 | 7 | ok |
| svd_cut4_value | tf_svd_cut4 | eager | 0.0229 | 0.0176 | 0.2500 | 0.0000 | 14 | ok |
| linear_qr_value | tf_qr | graph | 0.1804 | 0.0012 | 7.0000 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | graph | 1.8546 | 0.0014 | 234.9727 | 94.6133 |  | ok |
| linear_svd_value | tf_svd | graph | 0.2311 | 0.0011 | 0.2500 | 0.2500 |  | ok |
| svd_cubature_value | tf_svd_cubature | graph | 1.2768 | 0.0011 | -216.2070 | 4.0000 | 6 | ok |
| svd_ukf_value | tf_svd_ukf | graph | 0.3480 | 0.0015 | 9.6328 | 0.0000 | 7 | ok |
| svd_cut4_value | tf_svd_cut4 | graph | 0.7648 | 0.0012 | -1.6367 | 0.0000 | 14 | ok |

## Interpretation

Rows with `status = ok` completed for the declared fixed shape.  First-call timing includes tracing/initialization effects for graph mode.  Steady timing uses calls after the first observation.  Memory fields are process-level diagnostics and should not be interpreted as isolated per-backend allocation profiles.

This artifact does not certify MacroFinance/DSGE switch-over, GPU/XLA-GPU readiness, or HMC readiness.
