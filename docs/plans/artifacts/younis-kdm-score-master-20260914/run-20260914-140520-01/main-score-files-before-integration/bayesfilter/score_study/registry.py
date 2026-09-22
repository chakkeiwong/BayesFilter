"""Repository-owned score-study provider catalogue (no numerical imports)."""
from .contracts import Estimator, Model, Proposal, Registry, Target


def default_registry() -> Registry:
    regular = ("regular_euclidean",)
    targets = {x.id: x for x in (
        Target("model_score", regular, "exact_model_normalizer"),
        Target("finite_program_score", regular, "finite_program_normalizer"),
        Target("kdm_expectation_gradient", regular, "mixture_expectation"),
    )}
    proposals = {x.id: x for x in (
        Proposal("kalman", regular, "exact_gaussian_integration"),
        Proposal("bootstrap", regular, "transition_prior"),
        Proposal("prior_sis", regular, "transition_prior_without_resampling"),
        Proposal("ukf", regular, "gaussian_moment_filter"),
        Proposal("adapted", regular, "conditional_gaussian"),
        Proposal("adapted_sis", regular, "conditional_gaussian_without_resampling"),
        Proposal("ledh", regular, "canonical_ledh_pfpf_contract_e"),
        Proposal("kdm", regular, "corrected_kernel_mixture", "blocked", "0D KDM/trace/callback identities"),
        Proposal("sgqf", regular, "sgqf_guided_corrected_gaussian", "blocked", "0E moment lifecycle and call-chain checks"),
        Proposal("twist", regular, "corrected_auxiliary_twist", "blocked", "0E ancestor/state/telescoping identity"),
        Proposal("iapf", regular, "fitted_corrected_auxiliary_twist", "blocked", "0E fixed twist and fitted controls"),
    )}
    estimators = {x.id: x for x in (
        Estimator("exact_gaussian", "model_score", ("kalman",), "bayesfilter.score_study.adapters:evaluate_gaussian", (), True),
        Estimator("analytical_filter", "finite_program_score", ("bootstrap", "ukf", "adapted", "prior_sis", "adapted_sis", "ledh", "sgqf", "twist", "iapf"), "bayesfilter.score_study.adapters:evaluate_gaussian", (), True),
        Estimator("iwsg", "kdm_expectation_gradient", ("kdm",), None, (), True, prerequisite="0D"),
    )}
    return Registry(
        models={"gaussian_all_parameters": Model("gaussian_all_parameters", "regular_euclidean",
                                                 "lebesgue", True, "analytical_kalman")},
        targets=targets, proposals=proposals, estimators=estimators,
        tuning={"ledh": "0C repository-issued scoped selection and consumption"})
