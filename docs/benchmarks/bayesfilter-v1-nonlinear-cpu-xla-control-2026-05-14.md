# BayesFilter V1 Nonlinear GPU/XLA Diagnostic

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-nonlinear-cpu-xla-control-2026-05-14.json`.

## Claim Scope

Diagnostic timing for one fixed nonlinear Model B shape.  Not a correctness certificate, not a broad speedup claim, and not a production behavior change.

## Environment

```text
tensorflow = 2.19.1
device_scope = cpu
cuda_visible_devices = -1
logical_devices = [{'name': '/device:CPU:0', 'device_type': 'CPU'}]
```

## Rows

| Backend | Device | Mode | T | Points | Branch OK | Warmup s | First s | Steady s | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tf_svd_cut4 | cpu | eager | 24 | 14 | 3/3 |  | 0.087636 | 0.088768 | ok |
| tf_svd_cut4 | cpu | graph | 24 | 14 | 3/3 | 2.852207 | 0.005564 | 0.004820 | ok |
| tf_svd_cut4 | cpu | xla | 24 | 14 | 3/3 | 3.930447 | 0.000751 | 0.000391 | ok |

## Interpretation Rule

Only rows with `status = ok` and matching branch metadata can be used for timing interpretation.  Failed or skipped rows are diagnostic evidence only.
