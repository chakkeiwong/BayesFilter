"""Framework-free target and reference capabilities for diagnostic experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceSpec:
    kind: str
    quantities: tuple[str, ...]
    source: str
    uncertainty: str
    independent_implementation: str


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    family: str
    dimension: int
    support: str
    parameters: tuple[str, ...]
    reference: ReferenceSpec
    generative: bool = False
    finite_mean: bool = True
    finite_variance: bool = True
    available: bool = True
    unavailable_reason: str | None = None
    # Numerical target availability and posterior-adapter availability are
    # separate; unsupported routes must fail during suite planning.
    pipeline_available: bool = True
    pipeline_unavailable_reason: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


_EXACT = ReferenceSpec("analytic_and_iid", ("density", "score", "draws", "mean", "cdf"),
    "explicit laws in targets.py; independent formulas in references/analytic.py",
    "analytical quantities exact; reference draws have finite Monte Carlo uncertainty",
    "TF target formulas versus separately written SciPy/NumPy diagnostic references")
_DRAW = ReferenceSpec("iid", ("density", "score", "draws"), _EXACT.source,
    "iid Monte Carlo uncertainty must be included", _EXACT.independent_implementation)
_EXTERNAL = ReferenceSpec("external", (), "caller-supplied checked reference bundle",
    "requires quantity-specific uncertainty and target identity", "requires dependency declaration")

TARGETS = {
    item.target_id: item for item in (
        TargetSpec("gaussian", "quadratic", 2, "R2", ("x", "y"), _EXACT),
        TargetSpec("rotated_gaussian", "quadratic", 2, "R2", ("x", "y"), _EXACT),
        TargetSpec("banana", "nonlinear_transform", 2, "R2", ("x", "y"), _DRAW),
        TargetSpec("funnel", "hierarchical", 3, "R3", ("scale", "x", "y"), _DRAW),
        TargetSpec("student_t", "heavy_tail", 2, "R2", ("x", "y"), _EXACT),
        TargetSpec("cauchy", "heavy_tail", 2, "R2", ("x", "y"),
                   ReferenceSpec("analytic_and_iid", ("density", "score", "draws", "cdf"),
                                 _EXACT.source, _EXACT.uncertainty, _EXACT.independent_implementation),
                   finite_mean=False, finite_variance=False),
        TargetSpec("mixture", "multimodal", 2, "R2", ("x", "y"), _EXACT),
        TargetSpec("gamma", "positive", 1, "positive", ("rate",), _EXACT),
        TargetSpec("beta", "bounded", 1, "unit_interval", ("probability",), _EXACT),
        TargetSpec("dirichlet", "simplex", 2, "simplex3", ("p0", "p1", "p2"), _DRAW),
        TargetSpec("normal_conjugate", "conjugate", 1, "R", ("theta",), _EXACT, generative=True),
        TargetSpec("beta_binomial", "conjugate", 1, "unit_interval", ("probability",),
                   _EXACT, generative=True),
        TargetSpec("lgssm_location", "state_space", 1, "R", ("location",), _EXACT, generative=True),
        TargetSpec("eight_schools", "hierarchical", 10, "R9_x_positive", (), _EXTERNAL,
                   available=False, unavailable_reason="needs a versioned posteriordb target/reference bundle"),
        TargetSpec("regression", "regression", 0, "declared_by_bundle", (), _EXTERNAL,
                   available=False, unavailable_reason="needs a matched target/data/prior and posterior reference"),
        TargetSpec("macrofinance", "consumer", 0, "declared_by_consumer", (), _EXTERNAL,
                   available=False, unavailable_reason="consumer target factory and scope-specific reference required"),
    )
}


def get_target(target_id: str) -> TargetSpec:
    try:
        return TARGETS[target_id]
    except KeyError as exc:
        raise ValueError(f"unknown validation target: {target_id}") from exc
