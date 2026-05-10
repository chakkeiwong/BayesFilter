# BayesFilter v1 Filter Benchmark

Purpose: record CPU-only timing and process-memory metadata for BayesFilter v1 filtering candidates.

## Claim Scope

Benchmark artifact only.  Not a client switch-over or HMC readiness claim.  GPU/XLA-GPU claims require escalated device probe evidence and matching labeled artifacts.

## Configuration

```text
repeats = 2
timesteps = 4
state_dim = 2
observation_dim = 2
parameter_dim = 2
dtype = float64
device_scope = cpu
graph_warmup_calls = 0
benchmark_selector = all
modes = ('eager', 'graph')
shape_ladder = None
```

## Environment

```text
python = 3.11.14
platform = Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
tensorflow = 2.19.1
benchmark_device_scope = cpu
cuda_visible_devices = -1
logical_devices = [{'name': '/device:CPU:0', 'device_type': 'CPU'}]
```

CPU is the default benchmark scope.  Device scope 'visible' is allowed only after escalated GPU probes and does not by itself certify XLA-GPU readiness.

Process-level RSS snapshots are recorded before and after each benchmark row.  max_rss is process high-water RSS, so row deltas are diagnostic metadata, not isolated allocation profiles.

## Result

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-11-smoke.json`.

| Benchmark | Backend | Mode | T | n | m | p | Warmup s | First s | Steady s | RSS delta MB | Max RSS delta MB | Points | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| linear_qr_value | tf_qr | eager | 4 | 2 | 2 |  |  | 0.8179 | 0.0018 | 29.0000 | 29.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | eager | 4 | 2 | 2 | 2 |  | 9.5482 | 0.0056 | 540.5156 | 540.5156 |  | ok |
| linear_svd_value | tf_svd | eager | 4 | 2 | 2 |  |  | 0.4961 | 0.0017 | -146.3164 | 2.5000 |  | ok |
| svd_cubature_value | tf_svd_cubature | eager | 4 | 2 | 1 |  |  | 0.0539 | 0.0182 | 1.2500 | 0.0000 | 6 | ok |
| svd_ukf_value | tf_svd_ukf | eager | 4 | 2 | 1 |  |  | 0.0166 | 0.0180 | 0.0000 | 0.0000 | 7 | ok |
| svd_cut4_value | tf_svd_cut4 | eager | 4 | 2 | 1 |  |  | 0.0222 | 0.0160 | 0.5000 | 0.0000 | 14 | ok |
| linear_qr_value | tf_qr | graph | 4 | 2 | 2 |  |  | 0.1792 | 0.0009 | 6.5000 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | graph | 4 | 2 | 2 | 2 |  | 1.5914 | 0.0015 | 234.7656 | 94.1992 |  | ok |
| linear_svd_value | tf_svd | graph | 4 | 2 | 2 |  |  | 0.6372 | 0.0009 | 0.0000 | 0.0000 |  | ok |
| svd_cubature_value | tf_svd_cubature | graph | 4 | 2 | 1 |  |  | 0.9079 | 0.0011 | -214.8906 | 4.2500 | 6 | ok |
| svd_ukf_value | tf_svd_ukf | graph | 4 | 2 | 1 |  |  | 0.6119 | 0.0011 | -1.1875 | 0.0000 | 7 | ok |
| svd_cut4_value | tf_svd_cut4 | graph | 4 | 2 | 1 |  |  | 0.3984 | 0.0017 | 9.5078 | 0.0000 | 14 | ok |

## Interpretation

Rows with `status = ok` completed for the declared fixed shape.  First-call timing includes tracing/initialization effects unless graph warmup calls were requested.  Steady timing uses calls after the first measured observation.  Memory fields are process-level diagnostics and should not be interpreted as isolated per-backend allocation profiles.

This artifact does not certify MacroFinance/DSGE switch-over, GPU/XLA-GPU readiness, or HMC readiness.
