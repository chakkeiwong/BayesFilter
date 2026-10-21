"""Generic Bayesian SSM contracts."""

from importlib import import_module

from bayesfilter.ssm.contracts import (
    ApproximationSemantics,
    BayesianSSMProblem,
    DeterministicTargetPolicy,
    FilterProgram,
    FrozenTransportBinding,
    InvalidSSMContract,
    LogJacobianConvention,
    ParameterChart,
    ParameterPrior,
    PriorLogDensityAuthority,
    PriorSupportPolicy,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    TargetCoordinateConvention,
    stable_filter_program_signature,
    stable_frozen_transport_signature,
    stable_parameter_chart_signature,
    stable_prior_signature,
    stable_problem_signature,
    stable_ssm_target_signature,
    validate_ssm_target_contract,
)

_TARGET_BUILDER_EXPORTS = {
    "BatchRankPolicy",
    "BatchValueScoreFn",
    "GenericSSMPosteriorAdapter",
    "InvalidSSMTargetBuilderContract",
    "SSMTargetBuilderMetadata",
    "TARGET_BUILDER_NONCLAIMS",
    "XLA_TARGET_BUILDER_NONCLAIMS",
    "build_ssm_posterior_adapter",
    "stable_ssm_posterior_adapter_signature",
}
_FILTER_REGISTRY_EXPORTS = {
    "FilterProgramDescriptor",
    "FilterProgramRegistry",
    "FilterRegistryDecision",
    "InvalidFilterRegistryContract",
    "build_filter_program_registry",
    "stable_filter_descriptor_signature",
}

__all__ = [
    "ApproximationSemantics",
    "BatchRankPolicy",
    "BatchValueScoreFn",
    "BayesianSSMProblem",
    "DeterministicTargetPolicy",
    "FilterProgram",
    "FrozenTransportBinding",
    "FilterProgramDescriptor",
    "FilterProgramRegistry",
    "FilterRegistryDecision",
    "GenericSSMPosteriorAdapter",
    "InvalidFilterRegistryContract",
    "InvalidSSMContract",
    "InvalidSSMTargetBuilderContract",
    "LogJacobianConvention",
    "ParameterChart",
    "ParameterPrior",
    "PriorLogDensityAuthority",
    "PriorSupportPolicy",
    "SSMDataSignature",
    "SSMStaticShape",
    "SSMTargetContract",
    "SSMTargetBuilderMetadata",
    "TARGET_BUILDER_NONCLAIMS",
    "XLA_TARGET_BUILDER_NONCLAIMS",
    "TargetCoordinateConvention",
    "build_filter_program_registry",
    "build_ssm_posterior_adapter",
    "stable_filter_descriptor_signature",
    "stable_filter_program_signature",
    "stable_frozen_transport_signature",
    "stable_parameter_chart_signature",
    "stable_ssm_posterior_adapter_signature",
    "stable_prior_signature",
    "stable_problem_signature",
    "stable_ssm_target_signature",
    "validate_ssm_target_contract",
]


def __getattr__(name: str):
    if name in _TARGET_BUILDER_EXPORTS:
        module = import_module("bayesfilter.ssm.target_builder")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _FILTER_REGISTRY_EXPORTS:
        module = import_module("bayesfilter.ssm.filter_registry")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'bayesfilter.ssm' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
