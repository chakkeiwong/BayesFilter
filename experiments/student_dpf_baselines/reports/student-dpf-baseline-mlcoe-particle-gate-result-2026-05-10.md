# Student DPF baseline MP1 MLCOE particle-gate result

## Date

2026-05-10

## Scope

This report covers the MP1 MLCOE BPF adapter gate in the quarantined
student DPF experimental-baseline stream.  It is comparison-only
evidence and does not promote student code into production.

## Implementation Summary

| Implementation | Runs | OK | Failed | Max mean RMSE vs Kalman | Max covariance RMSE vs Kalman | Min avg ESS | Median runtime seconds | Likelihood runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE | 45 | 45 | 0 | 0.586727 | 0.125279 | 5.82374 | 0.145955 | 0 |
| advanced_particle_filter | 45 | 45 | 0 | 0.275898 | 0.116073 | 10.1708 | 0.0110125 | 45 |

## Particle Summary

| Implementation / fixture / particles | Runs | OK | Median mean RMSE vs Kalman | Median covariance RMSE vs Kalman | Median avg ESS | Min avg ESS | Median resampling count | Median runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE/lgssm_1d_long/N=128 | 3 | 3 | 0.0465493 | 0.0140584 | 42.4862 | 41.3552 | 5 | 0.149811 |
| 2026MLCOE/lgssm_1d_long/N=512 | 3 | 3 | 0.0287934 | 0.0133222 | 172.793 | 169.181 | 6 | 0.156615 |
| 2026MLCOE/lgssm_1d_long/N=64 | 3 | 3 | 0.0644844 | 0.0174186 | 22.5391 | 22.3348 | 5 | 0.148288 |
| 2026MLCOE/lgssm_1d_short/N=128 | 3 | 3 | 0.0515137 | 0.0220594 | 53.8269 | 23.9793 | 1 | 0.11258 |
| 2026MLCOE/lgssm_1d_short/N=512 | 3 | 3 | 0.0281619 | 0.0320257 | 136.784 | 131.867 | 1 | 0.112491 |
| 2026MLCOE/lgssm_1d_short/N=64 | 3 | 3 | 0.089343 | 0.0294203 | 17.0132 | 13.0221 | 1 | 0.111614 |
| 2026MLCOE/lgssm_cv_2d_long/N=128 | 3 | 3 | 0.248179 | 0.0788928 | 29.0608 | 27.7576 | 17 | 0.146988 |
| 2026MLCOE/lgssm_cv_2d_long/N=512 | 3 | 3 | 0.129103 | 0.0386711 | 108.96 | 104.003 | 18 | 0.162046 |
| 2026MLCOE/lgssm_cv_2d_long/N=64 | 3 | 3 | 0.441545 | 0.0991054 | 15.4869 | 13.2692 | 17 | 0.143704 |
| 2026MLCOE/lgssm_cv_2d_low_noise/N=128 | 3 | 3 | 0.14834 | 0.0185813 | 13.8447 | 13.3088 | 32 | 0.147336 |
| 2026MLCOE/lgssm_cv_2d_low_noise/N=512 | 3 | 3 | 0.0610863 | 0.0090668 | 54.1162 | 52.1021 | 29 | 0.162718 |
| 2026MLCOE/lgssm_cv_2d_low_noise/N=64 | 3 | 3 | 0.27174 | 0.0229581 | 6.98139 | 5.82374 | 34 | 0.147436 |
| 2026MLCOE/lgssm_cv_2d_short/N=128 | 3 | 3 | 0.242595 | 0.076884 | 25.8035 | 23.0914 | 4 | 0.115795 |
| 2026MLCOE/lgssm_cv_2d_short/N=512 | 3 | 3 | 0.0982208 | 0.0345515 | 91.9719 | 83.7155 | 5 | 0.120072 |
| 2026MLCOE/lgssm_cv_2d_short/N=64 | 3 | 3 | 0.276938 | 0.0761828 | 13.0698 | 13.0194 | 4 | 0.116134 |
| advanced_particle_filter/lgssm_1d_long/N=128 | 3 | 3 | 0.0303328 | 0.0137739 | 79.9997 | 79.4647 | 19 | 0.0109828 |
| advanced_particle_filter/lgssm_1d_long/N=512 | 3 | 3 | 0.0141104 | 0.00423154 | 315.099 | 314.22 | 17 | 0.0129325 |
| advanced_particle_filter/lgssm_1d_long/N=64 | 3 | 3 | 0.050659 | 0.0238661 | 39.7058 | 39.6885 | 16 | 0.0107625 |
| advanced_particle_filter/lgssm_1d_short/N=128 | 3 | 3 | 0.0308422 | 0.0324997 | 77.0727 | 72.0284 | 3 | 0.0020486 |
| advanced_particle_filter/lgssm_1d_short/N=512 | 3 | 3 | 0.0161229 | 0.0114404 | 301.403 | 295.559 | 3 | 0.00230355 |
| advanced_particle_filter/lgssm_1d_short/N=64 | 3 | 3 | 0.0509307 | 0.0583119 | 37.8451 | 35.9716 | 3 | 0.00203338 |
| advanced_particle_filter/lgssm_cv_2d_long/N=128 | 3 | 3 | 0.163523 | 0.0572587 | 43.5292 | 43.2161 | 39 | 0.0124444 |
| advanced_particle_filter/lgssm_cv_2d_long/N=512 | 3 | 3 | 0.0862924 | 0.0299221 | 173.985 | 170.586 | 39 | 0.0225642 |
| advanced_particle_filter/lgssm_cv_2d_long/N=64 | 3 | 3 | 0.267923 | 0.0678306 | 22.2596 | 21.9301 | 39 | 0.0114953 |
| advanced_particle_filter/lgssm_cv_2d_low_noise/N=128 | 3 | 3 | 0.0840566 | 0.0138317 | 20.791 | 17.0158 | 50 | 0.0129035 |
| advanced_particle_filter/lgssm_cv_2d_low_noise/N=512 | 3 | 3 | 0.0459685 | 0.00708515 | 83.2624 | 82.3349 | 50 | 0.0234311 |
| advanced_particle_filter/lgssm_cv_2d_low_noise/N=64 | 3 | 3 | 0.114903 | 0.020026 | 10.6534 | 10.1708 | 50 | 0.0120851 |
| advanced_particle_filter/lgssm_cv_2d_short/N=128 | 3 | 3 | 0.154446 | 0.0547586 | 40.3947 | 37.036 | 11 | 0.00355349 |
| advanced_particle_filter/lgssm_cv_2d_short/N=512 | 3 | 3 | 0.0566948 | 0.0231219 | 173.721 | 161.852 | 10 | 0.00493733 |
| advanced_particle_filter/lgssm_cv_2d_short/N=64 | 3 | 3 | 0.169939 | 0.0672833 | 22.3238 | 20.28 | 10 | 0.00327535 |

## Cross-Student Particle Comparison

| Fixture / particles | Groups | Median mean RMSE | Median covariance RMSE | Median ESS difference | Median runtime ratio MLCOE/advanced |
| --- | ---: | ---: | ---: | ---: | ---: |
| lgssm_1d_long/N=128 | 3 | 0.055695 | 0.0157717 | -38.1095 | 13.6722 |
| lgssm_1d_long/N=512 | 3 | 0.0315213 | 0.0112354 | -145.918 | 12.0043 |
| lgssm_1d_long/N=64 | 3 | 0.0784865 | 0.0227825 | -17.371 | 13.8968 |
| lgssm_1d_short/N=128 | 3 | 0.068597 | 0.0310276 | -22.855 | 56.2938 |
| lgssm_1d_short/N=512 | 3 | 0.0361848 | 0.0222332 | -158.775 | 49.6176 |
| lgssm_1d_short/N=64 | 3 | 0.102621 | 0.0440558 | -20.8319 | 54.891 |
| lgssm_cv_2d_long/N=128 | 3 | 0.286972 | 0.0949645 | -15.0447 | 11.8116 |
| lgssm_cv_2d_long/N=512 | 3 | 0.160315 | 0.0539623 | -61.6265 | 7.18157 |
| lgssm_cv_2d_long/N=64 | 3 | 0.450618 | 0.120946 | -6.60717 | 12.6353 |
| lgssm_cv_2d_low_noise/N=128 | 3 | 0.161667 | 0.0219726 | -6.94629 | 11.2874 |
| lgssm_cv_2d_low_noise/N=512 | 3 | 0.0725569 | 0.0115977 | -31.1603 | 6.94455 |
| lgssm_cv_2d_low_noise/N=64 | 3 | 0.29059 | 0.0285583 | -3.78837 | 12.3273 |
| lgssm_cv_2d_short/N=128 | 3 | 0.263011 | 0.0998661 | -17.3033 | 32.5864 |
| lgssm_cv_2d_short/N=512 | 3 | 0.120755 | 0.0443007 | -81.7492 | 24.3192 |
| lgssm_cv_2d_short/N=64 | 3 | 0.343967 | 0.106494 | -7.59264 | 35.4369 |

## Hypothesis Results

### h1_mlcoe_bpf_adapter_feasibility

Supported.  MLCOE BPF ran through the quarantined adapter without vendored-code edits and returned particle trajectories plus ESS diagnostics.

### h2_diagnostic_extraction

Supported.  MLCOE BPF exposes enough state, weight, and ESS data for particle mean, weighted covariance, ESS, runtime, and threshold-inferred resampling diagnostics.  Likelihood remains unavailable.

### h3_linear_stress_particle_behavior

Partially supported.  MLCOE BPF produced interpretable low-noise stress diagnostics, but the degradation pattern was weaker than the pre-specified qualitative threshold.

### h4_cross_student_particle_comparison

Supported as comparison-only evidence.  Matched fixture, seed, and particle-count groups were summarized without treating student agreement as correctness and without fabricating MLCOE likelihoods.

## Interpretation

MLCOE BPF likelihood fields remain null because the vendored BPF path
does not expose a defensible likelihood estimator.  MLCOE resampling
counts are threshold-inferred from the documented `ess < 0.1 * N`
condition and should not be treated as direct event logs.
