"""Exact moments and bounded compilation of frozen trajectory requests.

This proposer helper does not change ``FrozenTransitionSpec`` or issue tuning
authority. In the existing uniform law, the *integer* count is uniform on
``{1, ..., K}``, so its mean differs from half the maximum duration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .hmc_transition_spec import FrozenTransitionSpec, HMCTransitionProfile


@dataclass(frozen=True)
class CompiledMeanDuration:
    requested_mean_duration: float
    frozen: FrozenTransitionSpec
    realized_mean_duration: float
    realized_second_moment: float
    quantization_error: float
    max_steps_cap: int

    def payload(self) -> dict[str, object]:
        return {
            "schema": "bayesfilter.compiled_mean_trajectory_duration.v1",
            "requested_mean_duration": self.requested_mean_duration,
            "realized_mean_duration": self.realized_mean_duration,
            "realized_second_moment": self.realized_second_moment,
            "quantization_error": self.quantization_error,
            "max_steps_cap": self.max_steps_cap,
            "frozen_transition_spec": self.frozen.payload(),
            "rounding": "nearest_count_ties_up",
            "tuning_artifact_authority": False,
        }


def compile_mean_duration(*, profile: HMCTransitionProfile, epsilon: float,
                          requested_mean_duration: float,
                          max_steps_cap: int) -> CompiledMeanDuration:
    """Compile a physical-duration request into the existing integer law.

    Unsupported requests fail explicitly: there is no clipping at K=1 or at
    the declared count cap. A caller may separately propose a new epsilon.
    """
    if not isinstance(profile, HMCTransitionProfile):
        raise TypeError("profile must be HMCTransitionProfile")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if (type(requested_mean_duration) not in (int, float)
            or not math.isfinite(requested_mean_duration)
            or requested_mean_duration <= 0):
        raise ValueError("requested mean duration must be finite and positive")
    if type(max_steps_cap) is not int or not 1 <= max_steps_cap < 2**31 - 1:
        raise ValueError("max_steps_cap must fit the frozen int32 count law")
    eps, target = float(epsilon), float(requested_mean_duration)
    if target < eps:
        raise ValueError("requested mean duration is below the one-step minimum")
    floating_count = (2 * target / eps - 1 if profile.trajectory == "uniform_duration"
                      else target / eps)
    if not math.isfinite(floating_count) or floating_count >= 2**53:
        raise ValueError("duration/count ratio cannot be rounded reliably")
    count = math.floor(floating_count + 0.5)
    if not 1 <= count <= max_steps_cap:
        raise ValueError("requested duration exceeds the declared step-count cap")
    spec = profile.freeze(epsilon=eps, max_steps=count)
    if profile.trajectory == "uniform_duration":
        mean = eps * (count + 1) / 2
        second = eps * eps * (count + 1) * (2 * count + 1) / 6
    else:
        mean = eps * count
        second = mean * mean
    if not all(math.isfinite(v) for v in (mean, second)):
        raise ValueError("duration moments overflow")
    return CompiledMeanDuration(target, spec, mean, second, mean - target,
                                max_steps_cap)
