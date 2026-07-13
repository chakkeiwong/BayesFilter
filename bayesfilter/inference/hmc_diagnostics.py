"""Operational HMC diagnostics that avoid convergence claims."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HMCDiagnosticSummary:
    acceptance_rate: float | None
    divergence_status: str
    divergence_count: int | None
    split_rhat: tuple[float, ...]
    ess: tuple[float, ...]
    diagnostic_status: str
    nonclaims: tuple[str, ...] = (
        "operational diagnostics only; no posterior convergence claim",
    )


@dataclass(frozen=True)
class HMCLogAcceptSummary:
    finite_count: int
    nonfinite_count: int
    max_abs_log_accept_ratio: float | None
    divergence_count: int | None
    divergence_threshold: float
    divergence_criterion: str
    status: str


@dataclass(frozen=True)
class HMCScreenResult:
    passed: bool
    checks: Mapping[str, bool]
    diagnostic_roles: Mapping[str, str]
    unavailable_diagnostics: tuple[str, ...]
    nonclaims: tuple[str, ...] = (
        "bounded HMC screen only; no posterior convergence claim",
        "no sampler superiority claim",
        "no default sampler readiness claim",
    )


@dataclass(frozen=True)
class HMCFailureClassification:
    failure_class: str | None
    diagnostic_role: str
    screen_veto: str | None = None
    screen_failed_checks: tuple[str, ...] = ()
    interpretation: str = ""
    next_repair: str | None = None
    nonclaims: tuple[str, ...] = (
        "classification of bounded diagnostics only",
        "no posterior convergence claim",
        "no sampler superiority claim",
    )


def summarize_hmc_diagnostics(
    *,
    is_accepted: Any | None = None,
    divergences: Any | None = None,
    split_rhat: Any | None = None,
    ess: Any | None = None,
) -> HMCDiagnosticSummary:
    acceptance_rate = None
    if is_accepted is not None:
        accepted = np.asarray(is_accepted, dtype=bool)
        acceptance_rate = float(np.mean(accepted)) if accepted.size else np.nan
    if divergences is None:
        divergence_status = "unavailable"
        divergence_count = None
    else:
        divergence_array = np.asarray(divergences, dtype=bool)
        divergence_status = "available"
        divergence_count = int(np.sum(divergence_array))
    rhat_values = _float_tuple(split_rhat)
    ess_values = _float_tuple(ess)
    finite_parts = []
    if acceptance_rate is not None:
        finite_parts.append(np.isfinite(acceptance_rate))
    finite_parts.extend(np.isfinite(value) for value in rhat_values)
    finite_parts.extend(np.isfinite(value) for value in ess_values)
    diagnostic_status = "finite" if all(finite_parts or [True]) else "nonfinite"
    return HMCDiagnosticSummary(
        acceptance_rate=acceptance_rate,
        divergence_status=divergence_status,
        divergence_count=divergence_count,
        split_rhat=rhat_values,
        ess=ess_values,
        diagnostic_status=diagnostic_status,
    )


def summarize_log_accept_ratios(
    log_accept_ratio: Any | None,
    *,
    divergence_threshold: float = 1000.0,
) -> HMCLogAcceptSummary:
    if log_accept_ratio is None:
        return HMCLogAcceptSummary(
            finite_count=0,
            nonfinite_count=0,
            max_abs_log_accept_ratio=None,
            divergence_count=None,
            divergence_threshold=float(divergence_threshold),
            divergence_criterion="unavailable",
            status="unavailable",
        )
    values = np.asarray(log_accept_ratio, dtype=float)
    finite_mask = np.isfinite(values)
    finite_count = int(np.sum(finite_mask))
    nonfinite_count = int(values.size - finite_count)
    finite_values = values[finite_mask]
    max_abs = None if finite_count == 0 else float(np.max(np.abs(finite_values)))
    divergence_count = None if finite_count == 0 else int(np.sum(np.abs(finite_values) > float(divergence_threshold)))
    return HMCLogAcceptSummary(
        finite_count=finite_count,
        nonfinite_count=nonfinite_count,
        max_abs_log_accept_ratio=max_abs,
        divergence_count=divergence_count,
        divergence_threshold=float(divergence_threshold),
        divergence_criterion="abs(log_accept_ratio) > threshold",
        status="available",
    )


def screen_hmc_diagnostics(
    *,
    sample_chain_returned: bool,
    hmc_error_absent: bool = True,
    required_arrays_finite: bool | None = None,
    log_accept_ratio: Any | None = None,
    divergences: Any | None = None,
    acceptance_rate_by_chain: Any | None = None,
    fixed_kernel_used: bool | None = None,
    num_adaptation_steps_zero: bool | None = None,
    latent_initial_scale_zero: bool | None = None,
    use_xla_false: bool | None = None,
    compile_chain_with_xla_false: bool | None = None,
    divergence_threshold: float = 1000.0,
    diagnostic_role: str = "bounded HMC screen",
) -> HMCScreenResult:
    log_accept = summarize_log_accept_ratios(
        log_accept_ratio,
        divergence_threshold=divergence_threshold,
    )
    unavailable: list[str] = []
    checks: dict[str, bool] = {
        "sample_chain_returned": bool(sample_chain_returned),
        "hmc_error_absent": bool(hmc_error_absent),
    }
    if required_arrays_finite is None:
        unavailable.append("required_arrays_finite")
        checks["required_arrays_finite"] = False
    else:
        checks["required_arrays_finite"] = bool(required_arrays_finite)
    if divergences is None:
        unavailable.append("zero_divergences")
        checks["zero_divergences"] = False
    else:
        divergence_array = np.asarray(divergences, dtype=bool)
        checks["zero_divergences"] = int(np.sum(divergence_array)) == 0
    if log_accept.status == "unavailable":
        unavailable.append("log_accept_nonfinite_count_zero")
        checks["log_accept_nonfinite_count_zero"] = False
    else:
        checks["log_accept_nonfinite_count_zero"] = log_accept.nonfinite_count == 0
    if acceptance_rate_by_chain is None:
        unavailable.extend(
            [
                "acceptance_rate_by_chain_finite",
                "acceptance_rate_by_chain_strictly_between_0_05_and_0_99",
            ]
        )
        checks["acceptance_rate_by_chain_finite"] = False
        checks["acceptance_rate_by_chain_strictly_between_0_05_and_0_99"] = False
    else:
        acceptance = np.asarray(acceptance_rate_by_chain, dtype=float)
        finite_acceptance = bool(acceptance.size and np.all(np.isfinite(acceptance)))
        checks["acceptance_rate_by_chain_finite"] = finite_acceptance
        checks["acceptance_rate_by_chain_strictly_between_0_05_and_0_99"] = bool(
            finite_acceptance and np.all((acceptance > 0.05) & (acceptance < 0.99))
        )
    optional_checks = {
        "fixed_kernel_used": fixed_kernel_used,
        "num_adaptation_steps_zero": num_adaptation_steps_zero,
        "latent_initial_scale_zero": latent_initial_scale_zero,
        "use_xla_false": use_xla_false,
        "compile_chain_with_xla_false": compile_chain_with_xla_false,
    }
    for key, value in optional_checks.items():
        if value is not None:
            checks[key] = bool(value)
    roles = _screen_diagnostic_roles(checks)
    return HMCScreenResult(
        passed=all(checks.values()),
        checks=checks,
        diagnostic_roles={**roles, "screen": diagnostic_role},
        unavailable_diagnostics=tuple(unavailable),
    )


def classify_hmc_screen(
    screen: HMCScreenResult,
    *,
    hmc_error: Mapping[str, Any] | None = None,
    acceptance_rate_by_chain: Any | None = None,
    fixed_kernel_used: bool = False,
    num_adaptation_steps_zero: bool = False,
    step_size: float | None = None,
    num_leapfrog_steps: int | None = None,
    max_abs_log_accept_ratio: float | None = None,
) -> HMCFailureClassification:
    if hmc_error is not None:
        return HMCFailureClassification(
            failure_class="hmc_execution_error",
            diagnostic_role="continuation_veto",
            interpretation="HMC failed before bounded screen diagnostics could be trusted.",
            next_repair="repair execution error before interpreting sampler diagnostics",
        )
    if screen.passed:
        return HMCFailureClassification(
            failure_class=None,
            diagnostic_role="screen_passed_not_convergence",
            interpretation=(
                "The bounded HMC screen passed; this is not posterior convergence "
                "or sampler promotion evidence."
            ),
        )
    failed = tuple(key for key, value in screen.checks.items() if not bool(value))
    hard_failure_keys = {
        "sample_chain_returned",
        "hmc_error_absent",
        "required_arrays_finite",
        "zero_divergences",
        "log_accept_nonfinite_count_zero",
        "acceptance_rate_by_chain_finite",
    }
    if any(key in hard_failure_keys for key in failed):
        return HMCFailureClassification(
            failure_class="hmc_hard_veto_before_promotion",
            diagnostic_role="hard_veto",
            screen_failed_checks=failed,
            interpretation=(
                "The bounded HMC screen hit execution, nonfinite-array, divergence, "
                "nonfinite-log-acceptance, or unavailable/invalid acceptance diagnostics "
                "before any promotion interpretation is valid."
            ),
            next_repair="repair hard diagnostic failure before tuning or promotion claims",
        )
    acceptance = (
        np.asarray(acceptance_rate_by_chain, dtype=float)
        if acceptance_rate_by_chain is not None
        else np.asarray([], dtype=float)
    )
    acceptance_veto = "acceptance_rate_by_chain_strictly_between_0_05_and_0_99" in failed
    if (
        fixed_kernel_used
        and num_adaptation_steps_zero
        and acceptance.size
        and np.all(np.isfinite(acceptance))
        and acceptance_veto
        and np.all(acceptance >= 0.99)
    ):
        return HMCFailureClassification(
            failure_class="fixed_kernel_conservative_acceptance_veto",
            diagnostic_role="promotion_veto_repair_trigger",
            screen_veto="acceptance_rate_by_chain_strictly_between_0_05_and_0_99",
            screen_failed_checks=failed,
            interpretation=(
                "The fixed-kernel trace materialized cleanly but per-chain acceptance "
                "was degenerate near one. This is a tuning/envelope veto, not posterior "
                "convergence, DGP failure, or sampler superiority evidence."
            ),
            next_repair="run a reviewed bounded tuning repair before promotion claims",
        )
    return HMCFailureClassification(
        failure_class="hmc_screen_veto_after_trace_materialization",
        diagnostic_role="promotion_veto",
        screen_failed_checks=failed,
        interpretation=(
            "The HMC trace materialized but failed one or more bounded-screen checks. "
            "Inspect failed checks before assigning a model, target, or sampler cause."
        ),
        next_repair="classify failed checks under a reviewed repair plan",
    )


def _float_tuple(values: Any | None) -> tuple[float, ...]:
    if values is None:
        return tuple()
    return tuple(float(value) for value in np.asarray(values, dtype=float).ravel())


def _screen_diagnostic_roles(checks: Mapping[str, bool]) -> Mapping[str, str]:
    roles: dict[str, str] = {}
    for key in checks:
        if key in {
            "sample_chain_returned",
            "hmc_error_absent",
            "required_arrays_finite",
            "zero_divergences",
            "log_accept_nonfinite_count_zero",
            "acceptance_rate_by_chain_finite",
        }:
            roles[key] = "hard_veto"
        elif key == "acceptance_rate_by_chain_strictly_between_0_05_and_0_99":
            roles[key] = "promotion_veto"
        else:
            roles[key] = "explanatory_or_contract"
    return roles
