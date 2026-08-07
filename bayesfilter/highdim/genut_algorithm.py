"""Repository-owned GenUT algorithm-family selection.

Numerical transport and diagonal-moment controls remain scope-tuned.  This
module selects the structural higher-moment family without changing explicit
historical option semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


GENUT_DEFAULT_ALGORITHM = "dual_cap"
GENUT_ALGORITHMS = (
    "dual_cap",
    "coordinate_cap",
    "pairwise",
    "diagonal",
    "none",
    "bounded_teacher",
    "projected_cumulant",
)


@dataclass(frozen=True)
class GenUTAlgorithmSelection:
    """Resolved structural controls for one GenUT family."""

    requested_name: str
    algorithm: str
    pairwise_moment_correction_steps: int = 0
    pairwise_moment_strength: float = 0.0
    pairwise_moment_floor: float = 1.0e-5
    pairwise_particle_rms_cap: float = 0.0
    coordinatewise_standardized_cap: float = 0.0
    coordinatewise_standardized_cap_power: int = 8
    disable_diagonal_correction: bool = False
    requires_route_specific_inputs: bool = False

    def apply(self, scope_controls: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the selected family to already scope-tuned controls."""

        controls = dict(scope_controls)
        if self.disable_diagonal_correction:
            controls["higher_moment_correction_steps"] = 0
            controls["higher_moment_strength"] = 0.0
        controls.update(
            {
                "pairwise_moment_correction_steps": (
                    self.pairwise_moment_correction_steps
                ),
                "pairwise_moment_strength": self.pairwise_moment_strength,
                "pairwise_moment_floor": self.pairwise_moment_floor,
                "pairwise_particle_rms_cap": self.pairwise_particle_rms_cap,
                "coordinatewise_standardized_cap": (
                    self.coordinatewise_standardized_cap
                ),
                "coordinatewise_standardized_cap_power": (
                    self.coordinatewise_standardized_cap_power
                ),
            }
        )
        return controls


def resolve_genut_algorithm(
    name: str = "default",
    *,
    pairwise_strength: float = 0.02,
    pairwise_steps: int = 4,
    pairwise_floor: float = 1.0e-5,
) -> GenUTAlgorithmSelection:
    """Resolve a stable public name without choosing scope-specific tuning."""

    requested = str(name).strip().lower()
    algorithm = GENUT_DEFAULT_ALGORITHM if requested == "default" else requested
    if algorithm not in GENUT_ALGORITHMS:
        choices = ", ".join(("default", *GENUT_ALGORITHMS))
        raise ValueError(f"unknown GenUT algorithm {name!r}; choose one of {choices}")
    if pairwise_steps < 0 or pairwise_strength < 0.0 or pairwise_floor <= 0.0:
        raise ValueError("invalid pairwise GenUT controls")
    pairwise_active = algorithm in {"dual_cap", "coordinate_cap", "pairwise"}
    return GenUTAlgorithmSelection(
        requested_name=requested,
        algorithm=algorithm,
        pairwise_moment_correction_steps=pairwise_steps if pairwise_active else 0,
        pairwise_moment_strength=pairwise_strength if pairwise_active else 0.0,
        pairwise_moment_floor=pairwise_floor,
        pairwise_particle_rms_cap=2.0 if algorithm == "dual_cap" else 0.0,
        coordinatewise_standardized_cap=(
            0.98 if algorithm in {"dual_cap", "coordinate_cap"} else 0.0
        ),
        coordinatewise_standardized_cap_power=8,
        disable_diagonal_correction=algorithm == "none",
        requires_route_specific_inputs=algorithm
        in {"bounded_teacher", "projected_cumulant"},
    )


__all__ = [
    "GENUT_ALGORITHMS",
    "GENUT_DEFAULT_ALGORITHM",
    "GenUTAlgorithmSelection",
    "resolve_genut_algorithm",
]
