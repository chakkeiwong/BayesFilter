"""Public windowed mass-adaptation compatibility surface."""

from bayesfilter.inference.hmc_kernel_tuning import (
    HMCWindowedMassStageConfig,
    HMCWindowedMassStageResult,
    run_hmc_windowed_mass_stage,
)
from bayesfilter.inference.hmc_tuning import (
    WindowedMassAdaptationConfig,
    WindowedMassAdaptationResult,
    build_windowed_warmup_schedule,
    run_windowed_mass_adaptation_diagnostic,
    validate_windowed_shrinkage_target,
    welford_covariance,
)

__all__ = [
    "HMCWindowedMassStageConfig",
    "HMCWindowedMassStageResult",
    "WindowedMassAdaptationConfig",
    "WindowedMassAdaptationResult",
    "build_windowed_warmup_schedule",
    "run_hmc_windowed_mass_stage",
    "run_windowed_mass_adaptation_diagnostic",
    "validate_windowed_shrinkage_target",
    "welford_covariance",
]
