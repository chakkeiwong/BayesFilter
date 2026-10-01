"""Lazy public adapters; reference backends load only on explicit access."""

from importlib import import_module

_EXPORT_MODULES = {
    'BGSBatchPosteriorAdapter': 'bayesfilter.adapters.bgs',
    'BGSConstrainedLikelihoodResult': 'bayesfilter.adapters.bgs',
    'BGSPosteriorAdapter': 'bayesfilter.adapters.bgs',
    'BGSPosteriorComponents': 'bayesfilter.adapters.bgs',
    'constrained_log_prior_and_score': 'bayesfilter.adapters.bgs',
    'bgs_log_abs_det_jacobian': 'bayesfilter.adapters.bgs',
    'bgs_theta_from_unconstrained': 'bayesfilter.adapters.bgs',
    'bgs_unconstrained_from_theta': 'bayesfilter.adapters.bgs',
    'DSGEStructuralAdapterGateResult': 'bayesfilter.adapters.dsge',
    'dsge_structural_adapter_gate': 'bayesfilter.adapters.dsge',
    'DerivativeCoverageMetadata': 'bayesfilter.adapters.macrofinance',
    'FiniteDifferenceOracleMetadata': 'bayesfilter.adapters.macrofinance',
    'IdentificationEvidenceMetadata': 'bayesfilter.adapters.macrofinance',
    'CrossCurrencyDerivativeGateResult': 'bayesfilter.adapters.macrofinance',
    'LargeScaleAdaptationGateResult': 'bayesfilter.adapters.macrofinance',
    'MacroFinanceHMCBackendComparisonResult': 'bayesfilter.adapters.macrofinance',
    'MacroFinanceHMCDiagnosticGateResult': 'bayesfilter.adapters.macrofinance',
    'MacroFinanceDerivativeResult': 'bayesfilter.adapters.macrofinance',
    'MacroFinanceHMCGateResult': 'bayesfilter.adapters.macrofinance',
    'MacroFinanceHMCReadinessResult': 'bayesfilter.adapters.macrofinance',
    'MacroFinanceLikelihoodResult': 'bayesfilter.adapters.macrofinance',
    'ObservationMaskMetadata': 'bayesfilter.adapters.macrofinance',
    'ParameterUnitMetadata': 'bayesfilter.adapters.macrofinance',
    'ReadinessBlockerMetadata': 'bayesfilter.adapters.macrofinance',
    'ProductionExposureGateResult': 'bayesfilter.adapters.macrofinance',
    'SparseBackendPolicyMetadata': 'bayesfilter.adapters.macrofinance',
    'compare_macrofinance_hmc_backend_diagnostics': 'bayesfilter.adapters.macrofinance',
    'evaluate_cross_currency_derivative_gate': 'bayesfilter.adapters.macrofinance',
    'evaluate_large_scale_adaptation_gate': 'bayesfilter.adapters.macrofinance',
    'evaluate_macrofinance_hmc_diagnostic_gate': 'bayesfilter.adapters.macrofinance',
    'evaluate_macrofinance_hmc_gate': 'bayesfilter.adapters.macrofinance',
    'evaluate_macrofinance_hmc_readiness': 'bayesfilter.adapters.macrofinance',
    'evaluate_macrofinance_provider_derivatives': 'bayesfilter.adapters.macrofinance',
    'evaluate_macrofinance_provider_likelihood': 'bayesfilter.adapters.macrofinance',
    'evaluate_production_exposure_gate': 'bayesfilter.adapters.macrofinance',
    'extract_derivative_coverage_metadata': 'bayesfilter.adapters.macrofinance',
    'extract_finite_difference_oracle_metadata': 'bayesfilter.adapters.macrofinance',
    'extract_identification_evidence_metadata': 'bayesfilter.adapters.macrofinance',
    'extract_observation_mask_metadata': 'bayesfilter.adapters.macrofinance',
    'extract_parameter_unit_metadata': 'bayesfilter.adapters.macrofinance',
    'extract_readiness_blocker_metadata': 'bayesfilter.adapters.macrofinance',
    'extract_sparse_backend_policy_metadata': 'bayesfilter.adapters.macrofinance',
    'macrofinance_lgssm_to_bayesfilter': 'bayesfilter.adapters.macrofinance',
}
_EXPORT_ATTRIBUTES = {
    'bgs_log_abs_det_jacobian': 'log_abs_det_jacobian',
    'bgs_theta_from_unconstrained': 'theta_from_unconstrained',
    'bgs_unconstrained_from_theta': 'unconstrained_from_theta',
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), _EXPORT_ATTRIBUTES.get(name, name))
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [  # noqa: RUF022 - Preserve the published export order.
    "BGSBatchPosteriorAdapter",
    "BGSConstrainedLikelihoodResult",
    "BGSPosteriorAdapter",
    "BGSPosteriorComponents",
    "DerivativeCoverageMetadata",
    "DSGEStructuralAdapterGateResult",
    "FiniteDifferenceOracleMetadata",
    "IdentificationEvidenceMetadata",
    "CrossCurrencyDerivativeGateResult",
    "LargeScaleAdaptationGateResult",
    "MacroFinanceHMCBackendComparisonResult",
    "MacroFinanceHMCDiagnosticGateResult",
    "MacroFinanceDerivativeResult",
    "MacroFinanceHMCGateResult",
    "MacroFinanceHMCReadinessResult",
    "MacroFinanceLikelihoodResult",
    "ObservationMaskMetadata",
    "ParameterUnitMetadata",
    "ReadinessBlockerMetadata",
    "ProductionExposureGateResult",
    "SparseBackendPolicyMetadata",
    "compare_macrofinance_hmc_backend_diagnostics",
    "dsge_structural_adapter_gate",
    "evaluate_cross_currency_derivative_gate",
    "evaluate_large_scale_adaptation_gate",
    "evaluate_macrofinance_hmc_diagnostic_gate",
    "evaluate_macrofinance_hmc_gate",
    "evaluate_macrofinance_hmc_readiness",
    "evaluate_macrofinance_provider_derivatives",
    "evaluate_macrofinance_provider_likelihood",
    "evaluate_production_exposure_gate",
    "extract_derivative_coverage_metadata",
    "extract_finite_difference_oracle_metadata",
    "extract_identification_evidence_metadata",
    "extract_observation_mask_metadata",
    "extract_parameter_unit_metadata",
    "extract_readiness_blocker_metadata",
    "extract_sparse_backend_policy_metadata",
    "macrofinance_lgssm_to_bayesfilter",
    "constrained_log_prior_and_score",
    "bgs_log_abs_det_jacobian",
    "bgs_theta_from_unconstrained",
    "bgs_unconstrained_from_theta",
]
