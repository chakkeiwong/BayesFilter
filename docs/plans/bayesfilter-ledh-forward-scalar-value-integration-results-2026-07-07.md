# LEDH Forward-Scalar Value Integration

- JSON artifact: `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`
- Schema: `bayesfilter.highdim.ledh_forward_scalar_value_integration.v1`
- Target scalar: `observed_data_log_likelihood_estimator`
- Output tensor field: `log_likelihood`
- Score integration: `blocked_out_of_scope_forward_scalar_only`
- Runtime cross-ranking allowed: `False`
- Main row count: `6`

| Row | Mean Log Likelihood | MCSE | Target Policy | Source Artifact |
| --- | ---: | ---: | --- | --- |
| benchmark_lgssm_exact_oracle_m3_T50 | -135.96 | 0.00593341 | lgssm_gaussian_observation_density | `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json` |
| zhao_cui_spatial_sir_austria_j9_T20 | -902.83 | 0.208222 | fixed_sir_infectious_components_gaussian_observation_density | `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json` |
| zhao_cui_predator_prey_T20 | -169.868 | 0.423501 | additive_gaussian_predator_prey | `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json` |
| zhao_cui_sv_actual_nongaussian_T1000 | -2289.95 | 0.151 | transformed_actual_sv_log_y_square | `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json` |
| zhao_cui_generalized_sv_synthetic_from_estimated_values | -1438.92 | 0.0299775 | source_route_prior_mean_generalized_sv | `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json` |
| zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 | -2287.95 | 0.140231 | ksc_log_chi_square_gaussian_mixture_surrogate | `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json` |

## Diagnostic Rows

- `zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale`: `excluded_from_main_value_leaderboard`. legacy scoped parameterized SIR diagnostic row; no separate Phase 8 main-row admission artifact

## Nonclaims

- value-only integration from admitted forward-scalar artifacts
- not score admission
- not score correctness
- not all-algorithm comparison evidence
- not runtime ranking evidence
- not HMC readiness evidence
- not posterior correctness evidence
- not scientific superiority evidence
- KSC row is finite-mixture surrogate target evidence, not exact native actual-SV likelihood
