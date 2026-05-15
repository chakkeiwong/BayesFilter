# BayesFilter V1 Nonlinear GPU/XLA Diagnostic

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-nonlinear-gpu-xla-diagnostic-2026-05-14.json`.

## Claim Scope

Diagnostic timing for one fixed nonlinear Model B shape.  Not a correctness certificate, not a broad speedup claim, and not a production behavior change.

## Environment

```text
tensorflow = 2.19.1
device_scope = visible
cuda_visible_devices = None
logical_devices = [{'name': '/device:CPU:0', 'device_type': 'CPU'}, {'name': '/device:GPU:0', 'device_type': 'GPU'}]
```

## Rows

| Backend | Device | Mode | T | Points | Branch OK | Warmup s | First s | Steady s | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tf_svd_cut4 | cpu | eager | 24 | 14 | 3/3 |  | 0.149513 | 0.086636 | ok |
| tf_svd_cut4 | cpu | graph | 24 | 14 | 3/3 | 2.888792 | 0.005392 | 0.005458 | ok |
| tf_svd_cut4 | cpu | xla | 24 | 14 | 3/3 | 6.725224 | 0.020841 | 0.022035 | ok |
| tf_svd_cut4 | gpu | eager | 24 | 14 | 3/3 |  | 0.329029 | 0.305209 | ok |
| tf_svd_cut4 | gpu | graph | 24 | 14 | 3/3 | 2.039923 | 0.066121 | 0.061104 | ok |
| tf_svd_cut4 | gpu | xla | 24 | 14 | 3/3 | 6.351294 | 0.022169 | 0.022038 | ok |

## Interpretation Rule

Only rows with `status = ok` and matching branch metadata can be used for timing interpretation.  Failed or skipped rows are diagnostic evidence only.
