# BayesFilter V1 Model B/C GPU/XLA Scaling Diagnostic

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-model-bc-gpu-xla-scaling-2026-05-15.json`.

## Claim Scope

Diagnostic timing for exact Model B/C nonlinear sigma-point shapes only.  Not a broad speedup claim.

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
| model_b_nonlinear_accumulation | tf_svd_cubature | 8 | cpu | graph | 3/3 | 0.003045 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 8 | cpu | xla | 3/3 | 0.003305 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 8 | gpu | graph | 3/3 | 0.019369 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 8 | gpu | xla | 3/3 | 0.007478 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 8 | cpu | graph | 3/3 | 0.003411 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 8 | cpu | xla | 3/3 | 0.005567 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 8 | gpu | graph | 3/3 | 0.035424 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 8 | gpu | xla | 3/3 | 0.005116 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 8 | cpu | graph | 3/3 | 0.002702 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 8 | cpu | xla | 3/3 | 0.005034 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 8 | gpu | graph | 3/3 | 0.034096 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 8 | gpu | xla | 3/3 | 0.005794 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 16 | cpu | graph | 3/3 | 0.004795 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 16 | cpu | xla | 3/3 | 0.013830 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 16 | gpu | graph | 3/3 | 0.052235 | ok |
| model_b_nonlinear_accumulation | tf_svd_cubature | 16 | gpu | xla | 3/3 | 0.009923 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 16 | cpu | graph | 3/3 | 0.004698 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 16 | cpu | xla | 3/3 | 0.012501 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 16 | gpu | graph | 3/3 | 0.052157 | ok |
| model_b_nonlinear_accumulation | tf_svd_ukf | 16 | gpu | xla | 3/3 | 0.014626 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 16 | cpu | graph | 3/3 | 0.004037 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 16 | cpu | xla | 3/3 | 0.012385 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 16 | gpu | graph | 3/3 | 0.049213 | ok |
| model_b_nonlinear_accumulation | tf_svd_cut4 | 16 | gpu | xla | 3/3 | 0.011808 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | 8 | cpu | graph | 3/3 | 0.002582 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | 8 | cpu | xla | 3/3 | 0.002707 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | 8 | gpu | graph | 3/3 | 0.029501 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | 8 | gpu | xla | 3/3 | 0.002810 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | 8 | cpu | graph | 3/3 | 0.003002 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | 8 | cpu | xla | 3/3 | 0.003529 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | 8 | gpu | graph | 3/3 | 0.034108 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | 8 | gpu | xla | 3/3 | 0.004073 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | 8 | cpu | graph | 3/3 | 0.002740 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | 8 | cpu | xla | 3/3 | 0.011417 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | 8 | gpu | graph | 3/3 | 0.040755 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | 8 | gpu | xla | 3/3 | 0.010692 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | 16 | cpu | graph | 3/3 | 0.004119 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | 16 | cpu | xla | 3/3 | 0.016998 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | 16 | gpu | graph | 3/3 | 0.053260 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | 16 | gpu | xla | 3/3 | 0.008179 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | 16 | cpu | graph | 3/3 | 0.004241 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | 16 | cpu | xla | 3/3 | 0.021621 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | 16 | gpu | graph | 3/3 | 0.047798 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | 16 | gpu | xla | 3/3 | 0.007669 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | 16 | cpu | graph | 3/3 | 0.004286 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | 16 | cpu | xla | 3/3 | 0.007369 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | 16 | gpu | graph | 3/3 | 0.060656 | ok |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | 16 | gpu | xla | 3/3 | 0.008451 | ok |

## Interpretation Rule

Only rows with `status = ok` and matching branch metadata can be used for timing interpretation.  Failed, skipped, or branch-blocked rows are diagnostic evidence only.

