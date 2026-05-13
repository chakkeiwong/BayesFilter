# BayesFilter v1 Filter Benchmark

Purpose: record CPU-only timing and process-memory metadata for BayesFilter v1 filtering candidates.

## Claim Scope

Benchmark artifact only.  Not a client switch-over or HMC readiness claim.  GPU/XLA-GPU claims require escalated device probe evidence and matching labeled artifacts.

## Configuration

```text
repeats = 2
timesteps = 8
state_dim = 2
observation_dim = 2
parameter_dim = 2
dtype = float64
device_scope = cpu
graph_warmup_calls = 0
benchmark_selector = linear
modes = ('eager', 'graph')
shape_ladder = v1_cpu_diagnostic
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

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-filter-benchmark-ladder-2026-05-11.json`.

| Benchmark | Backend | Mode | T | n | m | p | Warmup s | First s | Steady s | RSS delta MB | Max RSS delta MB | Points | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| linear_qr_value | tf_qr | eager | 4 | 2 | 2 |  |  | 0.4710 | 0.0017 | 28.2500 | 28.2500 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | eager | 4 | 2 | 2 | 2 |  | 10.2084 | 0.0059 | 540.0508 | 540.0508 |  | ok |
| linear_svd_value | tf_svd | eager | 4 | 2 | 2 |  |  | 0.5063 | 0.0015 | -145.1445 | 3.0000 |  | ok |
| linear_qr_value | tf_qr | graph | 4 | 2 | 2 |  |  | 0.1816 | 0.0009 | 5.2500 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | graph | 4 | 2 | 2 | 2 |  | 1.8434 | 0.0009 | 234.0312 | 91.1367 |  | ok |
| linear_svd_value | tf_svd | graph | 4 | 2 | 2 |  |  | 0.1824 | 0.0014 | 0.0000 | 0.0000 |  | ok |
| linear_qr_value | tf_qr | eager | 8 | 2 | 2 |  |  | 0.3527 | 0.0019 | -148.3711 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | eager | 8 | 2 | 2 | 2 |  | 16.8422 | 0.0097 | 989.0781 | 840.7070 |  | ok |
| linear_svd_value | tf_svd | eager | 8 | 2 | 2 |  |  | 0.3343 | 0.0018 | -296.7969 | 0.0000 |  | ok |
| linear_qr_value | tf_qr | graph | 8 | 2 | 2 |  |  | 0.1594 | 0.0014 | 9.7500 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | graph | 8 | 2 | 2 | 2 |  | 3.3099 | 0.0015 | 459.3398 | 172.2930 |  | ok |
| linear_svd_value | tf_svd | graph | 8 | 2 | 2 |  |  | 0.2248 | 0.0014 | 0.0000 | 0.0000 |  | ok |
| linear_qr_value | tf_qr | eager | 12 | 4 | 3 |  |  | 0.5777 | 0.0027 | -437.1094 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | eager | 12 | 4 | 3 | 3 |  | 57.3580 | 0.0215 | 3324.9648 | 2887.8555 |  | ok |
| linear_svd_value | tf_svd | eager | 12 | 4 | 3 |  |  | 0.5264 | 0.0028 | -994.7031 | 0.0000 |  | ok |
| linear_qr_value | tf_qr | graph | 12 | 4 | 3 |  |  | 0.2578 | 0.0025 | 17.7500 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | graph | 12 | 4 | 3 | 3 |  | 11.2559 | 0.0024 | 1372.9922 | 476.1133 |  | ok |
| linear_svd_value | tf_svd | graph | 12 | 4 | 3 |  |  | 0.2526 | 0.0029 | 0.0000 | 0.0000 |  | ok |
| linear_qr_value | tf_qr | eager | 16 | 4 | 3 |  |  | 2.7067 | 0.0038 | -1333.6328 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | eager | 16 | 4 | 3 | 3 |  | 77.7314 | 0.0267 | 4392.3750 | 2978.6680 |  | ok |
| linear_svd_value | tf_svd | eager | 16 | 4 | 3 |  |  | 0.7299 | 0.0030 | -1311.2383 | 0.0000 |  | ok |
| linear_qr_value | tf_qr | graph | 16 | 4 | 3 |  |  | 0.3651 | 0.0028 | 22.7500 | 0.0000 |  | ok |
| linear_qr_score_hessian | tf_qr_sqrt | graph | 16 | 4 | 3 | 3 |  | 15.5678 | 0.0037 | 1853.8164 | 656.6484 |  | ok |
| linear_svd_value | tf_svd | graph | 16 | 4 | 3 |  |  | 0.4005 | 0.0023 | 0.0000 | 0.0000 |  | ok |

## Interpretation

Rows with `status = ok` completed for the declared fixed shape.  First-call timing includes tracing/initialization effects unless graph warmup calls were requested.  Steady timing uses calls after the first measured observation.  Memory fields are process-level diagnostics and should not be interpreted as isolated per-backend allocation profiles.

This artifact does not certify MacroFinance/DSGE switch-over, GPU/XLA-GPU readiness, or HMC readiness.
