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
graph_warmup_calls = 1
benchmark_selector = score
modes = ('graph',)
shape_ladder = v1_state_observation_ladder
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

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-qr-score-state-observation-ladder-2026-05-11.json`.

| Benchmark | Backend | Mode | T | n | m | p | Warmup s | First s | Steady s | RSS delta MB | Max RSS delta MB | Points | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| linear_qr_score | tf_qr_sqrt_score | graph | 8 | 2 | 2 | 2 | 4.0891 | 0.0042 | 0.0040 | 221.2500 | 221.2500 |  | ok |
| linear_qr_score | tf_qr_sqrt_score | graph | 8 | 3 | 2 | 2 | 2.7100 | 0.0050 | 0.0050 | 130.0078 | 130.0078 |  | ok |
| linear_qr_score | tf_qr_sqrt_score | graph | 8 | 4 | 3 | 2 | 3.0901 | 0.0040 | 0.0041 | 81.7422 | 81.7422 |  | ok |

## Interpretation

Rows with `status = ok` completed for the declared fixed shape.  First-call timing includes tracing/initialization effects unless graph warmup calls were requested.  Steady timing uses calls after the first measured observation.  Memory fields are process-level diagnostics and should not be interpreted as isolated per-backend allocation profiles.

This artifact does not certify MacroFinance/DSGE switch-over, GPU/XLA-GPU readiness, or HMC readiness.
