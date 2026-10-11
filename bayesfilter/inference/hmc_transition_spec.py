"""Framework-free specifications for optional corrected transition profiles.

These specifications issue no tuning authority. The shared numerical binding
owns coordinates, target, evidence and source identity. Every profile here uses
a full independent momentum/direction refresh at each outer transition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping


PROFILE_SCHEMA = "bayesfilter.hmc_transition_profile.v1"
SPEC_SCHEMA = "bayesfilter.frozen_transition_spec.v1"


@dataclass(frozen=True)
class HMCTransitionProfile:
    """An explicit optional kernel family, with no inferred tuning defaults.

    ``uniform_duration`` draws an IID uniform duration in [0, epsilon * L)
    for each chain/transition, then uses max(1, ceil(duration / epsilon)).
    L is the maximum step count. All coefficients and geometry are frozen.
    ``mams`` denotes the full-refresh isokinetic velocity-Verlet composition
    (force half-step, position step, force half-step), not a persistent kernel.
    """

    family: str
    trajectory: str

    def __post_init__(self) -> None:
        if self.family not in {"hmc", "mams"}:
            raise ValueError("unsupported transition family")
        if self.trajectory not in {"fixed", "uniform_duration"}:
            raise ValueError("unsupported trajectory law")

    @property
    def integrator(self) -> str:
        return "leapfrog" if self.family == "hmc" else "isokinetic_velocity_verlet"

    def payload(self) -> dict[str, Any]:
        return {"schema": PROFILE_SCHEMA, **asdict(self),
                "integrator": self.integrator,
                "auxiliary_refresh": "independent_full_each_transition",
                "length_randomness": "independent_chain_transition_stateless",
                "state_codec": "position_only_full_refresh_v1",
                "status": "experimental"}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HMCTransitionProfile":
        if not isinstance(payload, Mapping) or payload.get("schema") != PROFILE_SCHEMA:
            raise ValueError("transition profile schema mismatch")
        result = cls(payload["family"], payload["trajectory"])
        if result.payload() != dict(payload):
            raise ValueError("transition profile metadata mismatch")
        return result

    def freeze(self, *, epsilon: float, max_steps: int) -> "FrozenTransitionSpec":
        return FrozenTransitionSpec(self, epsilon, max_steps)


@dataclass(frozen=True)
class FrozenTransitionSpec:
    profile: HMCTransitionProfile
    epsilon: float
    max_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile, HMCTransitionProfile):
            raise TypeError("profile must be HMCTransitionProfile")
        if type(self.epsilon) not in (int, float) or not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if type(self.max_steps) is not int or not 1 <= self.max_steps < 2**31 - 1:
            raise ValueError("max_steps must be positive with int32 space for endpoint work")
        object.__setattr__(self, "epsilon", float(self.epsilon))
        if not math.isfinite(self.epsilon * self.max_steps):
            raise ValueError("trajectory duration overflows")

    @property
    def max_duration(self) -> float:
        return self.epsilon * self.max_steps

    @property
    def conservative_gradient_work_per_transition(self) -> int:
        # At most L+1 value/score evaluations, plus an independently computed
        # status evaluation at each point when no fused status path exists.
        # The runner reports actual calls separately. Masked batch rows count.
        return 2 * (self.max_steps + 1)

    def payload(self) -> dict[str, Any]:
        return {"schema": SPEC_SCHEMA, "profile": self.profile.payload(),
                "epsilon": self.epsilon, "max_steps": self.max_steps,
                "max_duration": self.max_duration,
                "rounding": "max_1_ceil_uniform_times_max_steps" if
                    self.profile.trajectory == "uniform_duration" else "fixed_count",
                "log_mh_ratio": "delta_log_density_plus_kinetic_correction" if
                    self.profile.family == "hmc" else "delta_log_density_minus_sphere_work"}

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenTransitionSpec":
        if not isinstance(payload, Mapping) or payload.get("schema") != SPEC_SCHEMA:
            raise ValueError("transition specification schema mismatch")
        result = cls(HMCTransitionProfile.from_payload(payload["profile"]),
                     payload["epsilon"], payload["max_steps"])
        if result.payload() != dict(payload):
            raise ValueError("transition specification metadata mismatch")
        return result


def candidate_gradient_work(candidate: Any) -> int:
    """Conservative native work per row, including legacy candidate payloads."""
    payload = candidate if isinstance(candidate, Mapping) else candidate.payload()
    spec = payload.get("transition_spec")
    if spec is None:
        return payload["leapfrog_steps"] + 1
    frozen = FrozenTransitionSpec.from_payload(spec)
    if frozen.epsilon != payload["epsilon"] or frozen.max_steps != payload["leapfrog_steps"]:
        raise ValueError("candidate settings disagree with transition specification")
    return frozen.conservative_gradient_work_per_transition
