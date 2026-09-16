"""Repository-owned score-study provider catalogue (no numerical imports)."""
from .contracts import Estimator, Model, Proposal, Registry, Target


def default_registry() -> Registry:
    regular = ("regular_euclidean",)
    targets = {x.id: x for x in (
        Target("model_score", regular, "exact_model_normalizer"),
        Target("finite_grid_score", regular, "fixed_grid_normalizer"),
        Target("finite_program_score", regular, "finite_program_normalizer"),
        Target("finite_difference_score", regular, "symmetric_difference_of_finite_program"),
        Target("kdm_expectation_gradient", regular, "mixture_expectation"),
        Target("integrated_kdm_program_score", regular, "convolved_finite_program"),
        Target("frozen_iwsg_log_program_score", regular, "frozen_proposal_raw_importance_log_program"),
    )}
    proposals = {x.id: x for x in (
        Proposal("kalman", regular, "exact_gaussian_integration"),
        Proposal("grid_reference", regular, "scalar_trapezoidal_filter_with_refinement"),
        Proposal("ekf", regular, "extended_gaussian_moment_filter"),
        Proposal("local_linear", regular, "local_linear_gaussian_proposal_with_physical_f_g_over_q"),
        Proposal("bootstrap", regular, "transition_prior"),
        Proposal("prior_sis", regular, "transition_prior_without_resampling"),
        Proposal("ukf", regular, "gaussian_moment_filter"),
        Proposal("adapted", regular, "conditional_gaussian"),
        Proposal("adapted_sis", regular, "conditional_gaussian_without_resampling"),
        Proposal("ledh", regular, "canonical_ledh_pfpf_contract_e"),
        Proposal("integrated_kdm", regular, "ledh_convolved_observation_factors"),
        Proposal("resampling_kdm", regular, "contract_e_then_raw_full_mixture_iwsg"),
        Proposal("kdm", regular, "corrected_kernel_mixture", "blocked", "0D KDM/trace/callback identities"),
        Proposal("sgqf", regular, "sgqf_moments_shared_ledh_contract_e"),
        Proposal("kdm_covariance", regular, "independent_conditioned_gaussian_mixture_moments_shared_ledh"),
        Proposal("twist", regular, "gaussian_power_psi_apf_with_initial_terminal_correction"),
        Proposal("fitted_twist", regular, "offline_log_quadratic_frozen_gaussian_floor_psi_apf_local_adaptation"),
        Proposal("iapf", regular, "bounded_diagonal_density_fit_adaptive_psi_apf_with_particle_initialization"),
    )}
    estimators = {x.id: x for x in (
        Estimator("exact_gaussian", "model_score", ("kalman",), "bayesfilter.score_study.adapters:evaluate_gaussian", (), True),
        Estimator("symmetric_fd", "finite_difference_score", ("kalman", "ukf", "bootstrap", "adapted", "prior_sis", "adapted_sis", "ledh"), "bayesfilter.score_study.fd_adapter_tf:evaluate_finite_difference", (), True, derivative="symmetric_fd_original_value_endpoint"),
        Estimator("analytical_filter", "finite_program_score", ("bootstrap", "ukf", "adapted", "prior_sis", "adapted_sis", "ledh", "sgqf", "kdm_covariance", "twist", "fitted_twist", "iapf"), "bayesfilter.score_study.adapters:evaluate_gaussian", (), True),
        Estimator("integrated_kdm_analytical", "integrated_kdm_program_score", ("integrated_kdm",), "bayesfilter.score_study.adapters:evaluate_gaussian", (), True),
        Estimator("resampling_iwsg_analytical", "frozen_iwsg_log_program_score", ("resampling_kdm",), "bayesfilter.score_study.adapters:evaluate_gaussian", (), True),
        Estimator("iwsg", "kdm_expectation_gradient", ("kdm",), None, (), True, prerequisite="0D"),
        Estimator("nonlinear_reference", "finite_grid_score", ("grid_reference",), "bayesfilter.score_study.nonlinear_adapter:evaluate_nonlinear", (), True, models=("nonlinear_scalar",)),
        Estimator("nonlinear_analytical", "finite_program_score", ("ekf", "ukf", "bootstrap", "prior_sis", "local_linear", "ledh", "sgqf", "kdm_covariance"), "bayesfilter.score_study.nonlinear_adapter:evaluate_nonlinear", (), True, models=("nonlinear_scalar",)),
    )}
    from dataclasses import replace
    estimators = {key: (value if value.models else replace(value, models=("gaussian_all_parameters",)))
                  for key,value in estimators.items()}
    return Registry(
        models={"gaussian_all_parameters": Model("gaussian_all_parameters", "regular_euclidean",
                                                 "lebesgue", True, "analytical_kalman"),
                "nonlinear_scalar": Model("nonlinear_scalar", "regular_euclidean", "lebesgue", True,
                                          "checked_mesh_and_domain_grid_reference")},
        targets=targets, proposals=proposals, estimators=estimators,
        tuning={"ledh": "0C repository-issued scoped selection and consumption"})
