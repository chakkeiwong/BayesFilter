# BayesFilter V1 Model B T=3 GPU/XLA Scaling Diagnostic

The JSON file is authoritative: `/home/chakwong/BayesFilter/docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.json`.

## Claim Scope

Diagnostic timing for exact Model B nonlinear sigma-point value cells at T=3 only.  Not a broad Model B/C, GPU, XLA, or default-policy speedup claim.

## Environment

```text
tensorflow = 2.19.1
device_scope = visible
cuda_visible_devices = None
logical_devices = [{'name': '/device:CPU:0', 'device_type': 'CPU'}, {'name': '/device:GPU:0', 'device_type': 'GPU'}]
```

## Rows

| Model | Backend | T | Device | Mode | Branch | Steady s | Status |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| model_b_nonlinear_accumulation | tf_svd_cubature | 3 | cpu | graph | 3/3 | 0.001646 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 3 | cpu | xla | 3/3 | 0.000819 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 3 | gpu | graph | 3/3 | 0.005359 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 3 | gpu | xla | 3/3 | 0.000811 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 3 | cpu | graph | 3/3 | 0.001629 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 3 | cpu | xla | 3/3 | 0.001394 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 3 | gpu | graph | 3/3 | 0.004964 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 3 | gpu | xla | 3/3 | 0.001396 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 3 | cpu | graph | 3/3 | 0.002160 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 3 | cpu | xla | 3/3 | 0.002218 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 3 | gpu | graph | 3/3 | 0.007809 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 3 | gpu | xla | 3/3 | 0.002190 | ok |

## Interpretation Rule

Only rows with `status = ok` and matching branch metadata can be used for timing interpretation.  Failed, skipped, or branch-blocked rows are diagnostic evidence only.

The JSON rows additionally record comparator id, path, dtype, static shape, seed policy, tolerance, finite/shape status, trust label, CPU/GPU policy, promotion/continuation/repair labels, command, environment, artifact path, and non-implication text. Those row fields are authoritative for audit; this Markdown table is a compact viewing aid.
