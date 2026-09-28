"""Explicit v2 fixed-trajectory HMC tuning fixture.

This module is intentionally not re-exported from ``bayesfilter.inference`` or
the top-level ``bayesfilter`` package. It keeps the experimental fixed-
trajectory tuner separate from the legacy shared HMC tuning-policy API while a
more complete BayesFilter tuning procedure is developed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


FIXED_TRAJECTORY_HMC_V2_ACCEPTANCE_BAND = (0.65, 0.75)
FIXED_TRAJECTORY_HMC_V2_NONCLAIMS = (
    "fixed-trajectory HMC v2 tuning candidate only",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default sampler readiness claim",
)


@dataclass(frozen=True)
class FixedTrajectoryHMCV2CandidateResult:
    """One finite-grid fixed-trajectory HMC v2 tuning candidate."""

    step_size: float
    num_leapfrog_steps: int
    trajectory_length: float
    mass_policy: str
    acceptance_rate: float | None
    log_accept_ratio_finite: bool
    finite_sample_count: int
    nonfinite_sample_count: int
    outcome: str
    vetoes: tuple[str, ...] = ()

    def payload(self) -> Mapping[str, Any]:
        return {
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "trajectory_length": self.trajectory_length,
            "mass_policy": self.mass_policy,
            "acceptance_rate": self.acceptance_rate,
            "log_accept_ratio_finite": self.log_accept_ratio_finite,
            "finite_sample_count": self.finite_sample_count,
            "nonfinite_sample_count": self.nonfinite_sample_count,
            "outcome": self.outcome,
            "vetoes": self.vetoes,
        }


@dataclass(frozen=True)
class FixedTrajectoryHMCV2TuningResult:
    """Structured first-slice v2 tuning artifact for a tiny Gaussian fixture."""

    policy_label: str
    selected_step_size: float | None
    selected_num_leapfrog_steps: int | None
    selected_trajectory_length: float | None
    mass_policy: str
    acceptance_band: tuple[float, float]
    candidate_results: tuple[FixedTrajectoryHMCV2CandidateResult, ...]
    diagnostics: Mapping[str, Any]
    vetoes: tuple[str, ...]
    nonclaims: tuple[str, ...] = FIXED_TRAJECTORY_HMC_V2_NONCLAIMS

    @property
    def selected_candidate(self) -> FixedTrajectoryHMCV2CandidateResult | None:
        for candidate in self.candidate_results:
            if (
                self.selected_step_size == candidate.step_size
                and self.selected_num_leapfrog_steps == candidate.num_leapfrog_steps
                and self.mass_policy == candidate.mass_policy
                and candidate.outcome == "passed_screen"
            ):
                return candidate
        return None

    @property
    def passed(self) -> bool:
        return self.selected_candidate is not None and not self.vetoes

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy_label": self.policy_label,
            "selected_step_size": self.selected_step_size,
            "selected_num_leapfrog_steps": self.selected_num_leapfrog_steps,
            "selected_trajectory_length": self.selected_trajectory_length,
            "mass_policy": self.mass_policy,
            "acceptance_band": self.acceptance_band,
            "candidate_results": [
                candidate.payload() for candidate in self.candidate_results
            ],
            "diagnostics": self.diagnostics,
            "vetoes": self.vetoes,
            "nonclaims": self.nonclaims,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_default_readiness": False,
        }


def run_tiny_gaussian_fixed_trajectory_hmc_tuning_v2(
    *,
    step_size_candidates: tuple[float, ...] = (0.5, 0.8, 1.0),
    num_leapfrog_step_candidates: tuple[int, ...] = (2,),
    mass_policy: str = "identity",
    requested_kernel: str = "hmc",
    initial_state: Any = (3.0, -2.0),
    num_results: int = 16,
    num_burnin_steps: int = 0,
    seed: tuple[int, int] = (20260612, 7),
    acceptance_band: tuple[float, float] = FIXED_TRAJECTORY_HMC_V2_ACCEPTANCE_BAND,
) -> FixedTrajectoryHMCV2TuningResult:
    """Select an explicit v2 fixed-trajectory HMC candidate.

    This first v2 slice evaluates finite, caller-supplied fixed-trajectory
    candidates with identity mass on a tiny Gaussian fixture. The closed
    acceptance band is a tuning promotion screen only, separate from the legacy
    broad fixed-kernel diagnostic screen.
    """

    kernel_label = str(requested_kernel).lower()
    if kernel_label in {"nuts", "no_u_turn", "no_u_turn_sampler", "tfp_nuts"}:
        raise ValueError(
            "NUTS is reference/diagnostic only, not a tuning/default remedy; "
            "fixed-trajectory HMC tuning must use HamiltonianMonteCarlo."
        )
    if kernel_label not in {"hmc", "hamiltonian_monte_carlo"}:
        raise ValueError("fixed-trajectory HMC v2 tuning supports only HMC")
    if str(mass_policy) != "identity":
        raise ValueError(
            "fixed-trajectory HMC v2 first slice supports mass_policy='identity' "
            "only; empirical/windowed mass adaptation is not implemented"
        )

    lower, upper = _validate_acceptance_band(acceptance_band)
    steps = _validate_positive_float_grid("step_size_candidates", step_size_candidates)
    leapfrogs = _validate_positive_int_grid(
        "num_leapfrog_step_candidates",
        num_leapfrog_step_candidates,
    )
    draws = int(num_results)
    burnin = int(num_burnin_steps)
    if draws <= 0:
        raise ValueError("num_results must be positive")
    if burnin < 0:
        raise ValueError("num_burnin_steps must be non-negative")

    import tensorflow as tf
    import tensorflow_probability as tfp

    state = tf.convert_to_tensor(initial_state, dtype=tf.float64)
    tfm = tfp.mcmc

    def target_log_prob(x: Any) -> Any:
        values = tf.convert_to_tensor(x, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1)

    start = time.perf_counter()
    candidates: list[FixedTrajectoryHMCV2CandidateResult] = []
    for step_size in steps:
        for leapfrog_count in leapfrogs:
            kernel = tfm.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                step_size=tf.constant(step_size, dtype=tf.float64),
                num_leapfrog_steps=leapfrog_count,
            )

            def trace_fn(_state: Any, kernel_results: Any) -> Mapping[str, Any]:
                return {
                    "is_accepted": kernel_results.is_accepted,
                    "log_accept_ratio": kernel_results.log_accept_ratio,
                }

            try:
                samples, trace = tfm.sample_chain(
                    num_results=draws,
                    num_burnin_steps=burnin,
                    current_state=state,
                    kernel=kernel,
                    trace_fn=trace_fn,
                    seed=tf.constant(tuple(int(item) for item in seed), dtype=tf.int32),
                )
                sample_array = np.asarray(samples.numpy(), dtype=float)
                accepted = np.asarray(trace["is_accepted"].numpy(), dtype=bool)
                log_accept_ratio = np.asarray(
                    trace["log_accept_ratio"].numpy(),
                    dtype=float,
                )
                acceptance_rate = (
                    float(np.mean(accepted)) if accepted.size else float("nan")
                )
                finite_rows = np.all(np.isfinite(sample_array), axis=-1)
                log_accept_finite = bool(np.all(np.isfinite(log_accept_ratio)))
                nonfinite_count = int(np.sum(~finite_rows))
                finite_count = int(np.sum(finite_rows))
                if nonfinite_count:
                    outcome = "rejected_nonfinite_target"
                    vetoes = ("nonfinite_sample",)
                elif not log_accept_finite:
                    outcome = "rejected_energy_instability"
                    vetoes = ("nonfinite_log_accept_ratio",)
                elif not np.isfinite(acceptance_rate):
                    outcome = "blocked_missing_diagnostics"
                    vetoes = ("acceptance_rate_nonfinite",)
                elif acceptance_rate < lower:
                    outcome = "rejected_accept_low"
                    vetoes = ("acceptance_below_closed_promotion_band",)
                elif acceptance_rate > upper:
                    outcome = "rejected_accept_high"
                    vetoes = ("acceptance_above_closed_promotion_band",)
                else:
                    outcome = "passed_screen"
                    vetoes = ()
            except Exception as exc:  # pragma: no cover - TFP runtime fault path.
                acceptance_rate = None
                log_accept_finite = False
                finite_count = 0
                nonfinite_count = 0
                outcome = "blocked_missing_diagnostics"
                vetoes = (f"hmc_execution_error:{type(exc).__name__}",)

            candidates.append(
                FixedTrajectoryHMCV2CandidateResult(
                    step_size=step_size,
                    num_leapfrog_steps=leapfrog_count,
                    trajectory_length=float(step_size * leapfrog_count),
                    mass_policy=str(mass_policy),
                    acceptance_rate=acceptance_rate,
                    log_accept_ratio_finite=log_accept_finite,
                    finite_sample_count=finite_count,
                    nonfinite_sample_count=nonfinite_count,
                    outcome=outcome,
                    vetoes=tuple(vetoes),
                )
            )

    elapsed = time.perf_counter() - start
    selected = _select_fixed_trajectory_candidate(candidates, lower, upper)
    result_vetoes: tuple[str, ...] = ()
    if selected is None:
        result_vetoes = ("no_candidate_in_closed_acceptance_promotion_band",)
    diagnostics = {
        "runtime": "tfp.mcmc.HamiltonianMonteCarlo",
        "target": "tiny_standard_gaussian_fixture",
        "candidate_count": len(candidates),
        "passed_candidate_count": sum(
            candidate.outcome == "passed_screen" for candidate in candidates
        ),
        "selection_rule": "closest_acceptance_to_closed_band_midpoint_then_grid_order",
        "acceptance_band_role": "tuning_promotion_screen_only",
        "legacy_fixed_kernel_screen_band": "(0.05, 0.99) separate broad screen",
        "api_surface": "explicit v2 module only; no package-level re-export",
        "elapsed_s": elapsed,
    }
    return FixedTrajectoryHMCV2TuningResult(
        policy_label="fixed_trajectory_hmc_tuning_v2_first_slice",
        selected_step_size=None if selected is None else selected.step_size,
        selected_num_leapfrog_steps=(
            None if selected is None else selected.num_leapfrog_steps
        ),
        selected_trajectory_length=None if selected is None else selected.trajectory_length,
        mass_policy=str(mass_policy),
        acceptance_band=(lower, upper),
        candidate_results=tuple(candidates),
        diagnostics=diagnostics,
        vetoes=result_vetoes,
    )


def _validate_acceptance_band(band: tuple[float, float]) -> tuple[float, float]:
    if len(tuple(band)) != 2:
        raise ValueError("acceptance_band must contain exactly two values")
    lower, upper = (float(band[0]), float(band[1]))
    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise ValueError("acceptance_band values must be finite")
    if not (0.0 < lower <= upper < 1.0):
        raise ValueError("acceptance_band must satisfy 0 < lower <= upper < 1")
    return lower, upper


def _validate_positive_float_grid(name: str, values: Any) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any((not np.isfinite(value)) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain only positive finite values")
    return result


def _validate_positive_int_grid(name: str, values: Any) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain only positive integer values")
    return result


def _select_fixed_trajectory_candidate(
    candidates: tuple[FixedTrajectoryHMCV2CandidateResult, ...]
    | list[FixedTrajectoryHMCV2CandidateResult],
    lower: float,
    upper: float,
) -> FixedTrajectoryHMCV2CandidateResult | None:
    midpoint = 0.5 * (lower + upper)
    passing = [
        candidate
        for candidate in candidates
        if candidate.outcome == "passed_screen"
        and candidate.acceptance_rate is not None
        and lower <= candidate.acceptance_rate <= upper
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda candidate: (
            abs(float(candidate.acceptance_rate) - midpoint),
            candidate.step_size,
            candidate.num_leapfrog_steps,
        ),
    )
