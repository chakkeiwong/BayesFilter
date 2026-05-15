# BayesFilter V1 Nonlinear Performance NP1 CPU-only Smoke Artifact

Authoritative JSON artifact: `docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.json`.

## Manifest

- Command: `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py --requested-device cpu --repeats 1 --output docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.json --markdown-output docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.md`
- Git commit: `81c647b157612a233dde1a9fbd847647a5b03b8f`
- Dirty worktree: `True`
- Python / TF / TFP: `3.11.14` / `2.19.1` / `0.25.0`
- Visibility policy: `cpu_only_hidden_gpu_before_tensorflow_import` with `CUDA_VISIBLE_DEVICES=-1`

## Rows

| Row ID | Role | Model | Backend | Path | T | Points | Branch precheck | First s | Steady s | Skip |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| branch-precheck-tf_svd_cubature-model-a-tiny | branch_precheck | model_a_affine_gaussian_structural_oracle | tf_svd_cubature | score | 2 | 6 | pass | n/a | n/a |  |
| value-timing-tf_svd_cubature-model-a-tiny-return-filtered-false | value_timing | model_a_affine_gaussian_structural_oracle | tf_svd_cubature | value | 2 | 6 | n/a | 0.015386 | 0.008996 |  |
| value-timing-tf_svd_cubature-model-a-tiny-return-filtered-true | value_timing | model_a_affine_gaussian_structural_oracle | tf_svd_cubature | value | 2 | 6 | n/a | 0.008033 | 0.007434 |  |
| score-timing-tf_svd_cubature-model-a-tiny | score_timing | model_a_affine_gaussian_structural_oracle | tf_svd_cubature | score | 2 | 6 | pass | 0.022196 | 0.022409 |  |
| branch-precheck-tf_svd_ukf-model-a-tiny | branch_precheck | model_a_affine_gaussian_structural_oracle | tf_svd_ukf | score | 2 | 7 | pass | n/a | n/a |  |
| value-timing-tf_svd_ukf-model-a-tiny-return-filtered-false | value_timing | model_a_affine_gaussian_structural_oracle | tf_svd_ukf | value | 2 | 7 | n/a | 0.007809 | 0.009005 |  |
| value-timing-tf_svd_ukf-model-a-tiny-return-filtered-true | value_timing | model_a_affine_gaussian_structural_oracle | tf_svd_ukf | value | 2 | 7 | n/a | 0.007920 | 0.008304 |  |
| score-timing-tf_svd_ukf-model-a-tiny | score_timing | model_a_affine_gaussian_structural_oracle | tf_svd_ukf | score | 2 | 7 | pass | 0.024285 | 0.025269 |  |
| branch-precheck-tf_svd_cut4-model-a-tiny | branch_precheck | model_a_affine_gaussian_structural_oracle | tf_svd_cut4 | score | 2 | 14 | pass | n/a | n/a |  |
| value-timing-tf_svd_cut4-model-a-tiny-return-filtered-false | value_timing | model_a_affine_gaussian_structural_oracle | tf_svd_cut4 | value | 2 | 14 | n/a | 0.008828 | 0.008688 |  |
| value-timing-tf_svd_cut4-model-a-tiny-return-filtered-true | value_timing | model_a_affine_gaussian_structural_oracle | tf_svd_cut4 | value | 2 | 14 | n/a | 0.008556 | 0.009088 |  |
| score-timing-tf_svd_cut4-model-a-tiny | score_timing | model_a_affine_gaussian_structural_oracle | tf_svd_cut4 | score | 2 | 14 | pass | 0.024767 | 0.023373 |  |
| skipped-cut4-point-cap-synthetic | skipped | model_a_affine_gaussian_structural_oracle | tf_svd_cut4 | value | 2 | 1044 | not_applicable | n/a | n/a | cut4_point_cap |

## Scope boundary

NP1 CPU-only smoke rows do not certify broad speedups, default backend policy, GPU/XLA support, exact nonlinear likelihood quality for Models B-C, or HMC/Hessian readiness.
