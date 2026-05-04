"""Optional adapters for external state-space projects."""

from bayesfilter.adapters.macrofinance import (
    MacroFinanceDerivativeResult,
    MacroFinanceHMCReadinessResult,
    MacroFinanceLikelihoodResult,
    evaluate_macrofinance_hmc_readiness,
    evaluate_macrofinance_provider_derivatives,
    evaluate_macrofinance_provider_likelihood,
    macrofinance_lgssm_to_bayesfilter,
)

__all__ = [
    "MacroFinanceDerivativeResult",
    "MacroFinanceHMCReadinessResult",
    "MacroFinanceLikelihoodResult",
    "evaluate_macrofinance_hmc_readiness",
    "evaluate_macrofinance_provider_derivatives",
    "evaluate_macrofinance_provider_likelihood",
    "macrofinance_lgssm_to_bayesfilter",
]
