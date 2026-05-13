# BayesFilter v1 Filter Benchmark

Purpose: record CPU-only timing and process-memory metadata for BayesFilter v1 filtering candidates.

## Claim Scope

Benchmark artifact only.  Not a client switch-over or HMC readiness claim.  GPU/XLA-GPU claims require escalated device probe evidence and matching labeled artifacts.

## Configuration

```text
repeats = 1
timesteps = 4
state_dim = 2
observation_dim = 2
parameter_dim = 2
dtype = float64
device_scope = visible
graph_warmup_calls = 0
benchmark_selector = linear_value
modes = ('xla',)
shape_ladder = None
```

## Environment

```text
python = 3.11.14
platform = Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
tensorflow = 2.19.1
benchmark_device_scope = visible
cuda_visible_devices = None
logical_devices = [{'name': '/device:CPU:0', 'device_type': 'CPU'}, {'name': '/device:GPU:0', 'device_type': 'GPU'}]
```

CPU is the default benchmark scope.  Device scope 'visible' is allowed only after escalated GPU probes and does not by itself certify XLA-GPU readiness.

Process-level RSS snapshots are recorded before and after each benchmark row.  max_rss is process high-water RSS, so row deltas are diagnostic metadata, not isolated allocation profiles.

## Result

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-filter-benchmark-xla-visible-2026-05-11.json`.

| Benchmark | Backend | Mode | T | n | m | p | Warmup s | First s | Steady s | RSS delta MB | Max RSS delta MB | Points | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| linear_qr_value | tf_qr | xla | 4 | 2 | 2 |  |  | 1.4656 | 1.4656 | 138.9961 | 138.9961 |  | ok |
| linear_svd_value | tf_svd | xla | 4 | 2 | 2 |  |  | 1.1214 | 1.1214 | 21.9727 | 21.9727 |  | ok |

## Interpretation

Rows with `status = ok` completed for the declared fixed shape.  First-call timing includes tracing/initialization effects unless graph warmup calls were requested.  Steady timing uses calls after the first measured observation.  Memory fields are process-level diagnostics and should not be interpreted as isolated per-backend allocation profiles.

This artifact does not certify MacroFinance/DSGE switch-over, GPU/XLA-GPU readiness, or HMC readiness.
