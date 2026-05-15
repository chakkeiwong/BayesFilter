# BayesFilter v1 Nonlinear Filter Benchmark

The JSON file is authoritative: `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-14.json`.

## Claim Scope

CPU-only benchmark.  Model A uses an exact linear-Gaussian Kalman reference.  Models B-C use dense one-step Gaussian projection references only, so their full log-likelihoods are recorded as filter outputs, not exact-error evidence.

## Rows

| Model | Backend | Reference | Loglik Error | First-Step Error | Points | Value Branch OK | Score Branch | Score OK | Steady Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| model_a_affine_gaussian_structural_oracle | tf_svd_cubature | exact_linear_gaussian_kalman | 4.441e-16 | 2.776e-16 | 6 | 3/3 | smooth_simple_spectrum_no_active_floor | 3/3 | 0.020068 |
| model_a_affine_gaussian_structural_oracle | tf_svd_ukf | exact_linear_gaussian_kalman | 4.441e-16 | 2.776e-16 | 7 | 3/3 | smooth_simple_spectrum_no_active_floor | 3/3 | 0.018551 |
| model_a_affine_gaussian_structural_oracle | tf_svd_cut4 | exact_linear_gaussian_kalman | 3.331e-16 | 1.110e-16 | 14 | 3/3 | smooth_simple_spectrum_no_active_floor | 3/3 | 0.021537 |
| model_b_nonlinear_accumulation | tf_svd_cubature | dense_one_step_projection_only | n/a | 1.681e-02 | 6 | 3/3 | smooth_simple_spectrum_no_active_floor | 3/3 | 0.014269 |
| model_b_nonlinear_accumulation | tf_svd_ukf | dense_one_step_projection_only | n/a | 1.681e-02 | 7 | 3/3 | smooth_simple_spectrum_no_active_floor | 3/3 | 0.011751 |
| model_b_nonlinear_accumulation | tf_svd_cut4 | dense_one_step_projection_only | n/a | 3.643e-03 | 14 | 3/3 | smooth_simple_spectrum_no_active_floor | 3/3 | 0.012268 |
| model_c_autonomous_nonlinear_growth | tf_svd_cubature | dense_one_step_projection_only | n/a | 4.926e-02 | 6 | 3/3 | structural_fixed_support_no_active_floor | 3/3 | 0.008045 |
| model_c_autonomous_nonlinear_growth | tf_svd_ukf | dense_one_step_projection_only | n/a | 1.490e-01 | 7 | 3/3 | structural_fixed_support_no_active_floor | 3/3 | 0.007492 |
| model_c_autonomous_nonlinear_growth | tf_svd_cut4 | dense_one_step_projection_only | n/a | 3.391e-02 | 14 | 3/3 | structural_fixed_support_no_active_floor | 3/3 | 0.008541 |

## Interpretation

CUT4 point counts are larger than cubature and UKF.  This artifact is designed to test whether that larger rule improves the small nonlinear moment-projection rows enough to justify further GPU/XLA or HMC-specific work.
