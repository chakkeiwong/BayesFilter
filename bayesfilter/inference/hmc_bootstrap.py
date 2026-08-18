"""Public bootstrap-stage compatibility surface."""

from bayesfilter.inference.hmc_kernel_tuning import (
    BOOTSTRAP_SCREEN_NONCLAIMS,
    HMCBootstrapRepairRound,
    HMCBootstrapScreenConfig,
    HMCBootstrapScreenResult,
    _BootstrapFixedMassLatentValueScoreAdapter,
    run_hmc_bootstrap_screen,
)

BootstrapFixedMassAdapter = _BootstrapFixedMassLatentValueScoreAdapter

__all__ = [
    "BOOTSTRAP_SCREEN_NONCLAIMS",
    "BootstrapFixedMassAdapter",
    "HMCBootstrapRepairRound",
    "HMCBootstrapScreenConfig",
    "HMCBootstrapScreenResult",
    "run_hmc_bootstrap_screen",
]
