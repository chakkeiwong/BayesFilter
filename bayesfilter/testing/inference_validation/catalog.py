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
_SSM_MECHANICS_ONLY = ReferenceSpec(
    "reference_unavailable",
    ("density", "score"),
    "independent filter-mechanics references are available per profile; no general posterior draw oracle is bundled",
    "posterior accuracy remains unavailable until a profile-specific integration/reference bundle is checked",
    "QR Kalman recursion or deterministic sigma-point implementation is independent of the TensorFlow target wrapper",
)
_SSM_LGSSM_GRID = ReferenceSpec(
    "numerical_grid_iid",
    ("density", "score", "draws"),
    "independent scalar Kalman recursion with a bounded two-dimensional quadrature grid",
    "finite iid grid draws plus explicit resolution/domain sensitivity; no rigorous integration error bound",
    "NumPy/SciPy recursion and quadrature are separate from the TensorFlow QR adapter",
)

TARGETS = {
    item.target_id: item for item in (
        TargetSpec("gaussian", "quadratic", 2, "R2", ("x", "y"), _EXACT),
        TargetSpec("rotated_gaussian", "quadratic", 2, "R2", ("x", "y"), _EXACT),
        TargetSpec("banana", "nonlinear_transform", 2, "R2", ("x", "y"), _DRAW),
        TargetSpec("funnel", "hierarchical", 3, "R3", ("scale", "x", "y"), _DRAW),
        TargetSpec("funnel_noncentered", "hierarchical", 3, "R3", ("scale", "x", "y"), _DRAW),
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
        TargetSpec("ssm_lgssm_qr", "state_space", 2, "R2", ("rho_unconstrained", "log_measurement_noise"), _SSM_LGSSM_GRID),
        TargetSpec("ssm_lgssm_near_unit", "state_space", 2, "R2", ("rho_unconstrained", "log_measurement_noise"), _SSM_LGSSM_GRID),
        TargetSpec("ssm_lgssm_small_noise", "state_space", 2, "R2", ("rho_unconstrained", "log_measurement_noise"), _SSM_LGSSM_GRID),
        TargetSpec("ssm_lgssm_long_horizon", "state_space", 2, "R2", ("rho_unconstrained", "log_measurement_noise"), _SSM_LGSSM_GRID),
        TargetSpec("ssm_nonlinear", "state_space_nonlinear", 3, "R3", ("rho", "sigma", "beta"), _SSM_MECHANICS_ONLY),
        TargetSpec("ssm_nonlinear_long_horizon", "state_space_nonlinear", 3, "R3", ("rho", "sigma", "beta"), _SSM_MECHANICS_ONLY),
        TargetSpec("eight_schools", "hierarchical", 10, "R9_x_positive", (), _EXTERNAL,
                   available=False, unavailable_reason="needs a versioned posteriordb target/reference bundle"),
        TargetSpec("regression", "regression", 0, "declared_by_bundle", (), _EXTERNAL,
                   available=False, unavailable_reason="needs a matched target/data/prior and posterior reference"),
        TargetSpec("macrofinance", "consumer", 0, "declared_by_consumer", (), _EXTERNAL,
                   available=False, unavailable_reason="consumer target factory and scope-specific reference required"),
    )
}

# Import only framework-free model metadata during CLI planning.
from .ssm_campaign_profiles import PROFILES as _CAMPAIGN_PROFILES
for _profile in _CAMPAIGN_PROFILES.values():
    TARGETS[_profile.target] = TargetSpec(
        _profile.target, "state_space_nonlinear" if _profile.family == "nonlinear" else "state_space",
        len(_profile.raw_names), "R", _profile.names,
        (ReferenceSpec("reference_unavailable", ("density", "score"),
                       "independent multivariate Kalman reference; no joint posterior oracle",
                       "finite difference and numerical covariance tolerances",
                       "NumPy/SciPy filtering and stationary covariance")
         if _profile.reference == "unavailable" else
         ReferenceSpec("analytic_and_iid" if _profile.reference == "analytic" else "numerical_grid_iid",
                       ("density", "score", "draws"),
                       "references/ssm.py: independent same-target filtering and integration",
                       "grid resolution/domain checks required; no rigorous integration error bound",
                       "NumPy/SciPy reference, independent of TF filters")))


def get_target(target_id: str) -> TargetSpec:
    try:
        return TARGETS[target_id]
    except KeyError as exc:
        raise ValueError(f"unknown validation target: {target_id}") from exc
