"""Public bootstrap-stage compatibility surface."""

from bayesfilter.inference.hmc_kernel_tuning import (
    BOOTSTRAP_SCREEN_NONCLAIMS,
    HMCBootstrapRepairRound,
    HMCBootstrapScreenConfig,
    HMCBootstrapScreenResult,
    _BootstrapFixedMassLatentValueScoreAdapter,
    _build_bootstrap_fixed_mass_adapter,
    run_hmc_bootstrap_screen,
)

BootstrapFixedMassAdapter = _BootstrapFixedMassLatentValueScoreAdapter

# Public alias for the bootstrap fixed-mass adapter builder. The adapter
# signature payload embeds the historical private class path
# ("...hmc_kernel_tuning._BootstrapFixedMassLatentValueScoreAdapter"), which
# this alias preserves byte-identically: it is the same function object.
build_bootstrap_fixed_mass_adapter = _build_bootstrap_fixed_mass_adapter

__all__ = [
    "BOOTSTRAP_SCREEN_NONCLAIMS",
    "BootstrapFixedMassAdapter",
    "HMCBootstrapRepairRound",
    "HMCBootstrapScreenConfig",
    "HMCBootstrapScreenResult",
    "build_bootstrap_fixed_mass_adapter",
    "run_hmc_bootstrap_screen",
]
