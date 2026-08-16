"""Static route guard for admitted factor-propagating SR-UKF code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORBIDDEN_SRUKF_ROUTE_PATTERNS: tuple[str, ...] = (
    "GradientTape",
    "tf_svd_sigma_point_filter",
    "eigenderivative",
    "strict_spd_principal_sqrt",
    "strict-SPD principal-root",
    "principal_sqrt_frechet_derivative",
    "tf.linalg.eigh",
    "tf.linalg.svd",
    "cholesky",
    "tf_principal_sqrt_ukf",
    "principal_sqrt",
    "tf_svd_sigma_point_filter",
    "tf_svd_cubature",
    "tf_svd_ukf",
    "experimental_batched_svd_sigma_point_tf",
    "strict_spd_principal_sqrt",
    "covariance_to_factor",
    "covariance_to_root",
    "refactorize_covariance",
)

ADMITTED_DIRECT_FACTOR_SRUKF_FILES: tuple[str, ...] = (
    "bayesfilter/linear/stack_qr_tf.py",
    "bayesfilter/linear/lower_rank_downdate_tf.py",
    "bayesfilter/nonlinear/factor_srukf_tf.py",
)


@dataclass(frozen=True)
class SRUKFRouteGuardViolation:
    """One forbidden route occurrence found by the static guard."""

    pattern: str
    line_number: int
    line: str


def find_forbidden_srukf_routes(text: str) -> tuple[SRUKFRouteGuardViolation, ...]:
    """Return forbidden route occurrences in admitted SR-UKF implementation text."""

    violations: list[SRUKFRouteGuardViolation] = []
    for line_number, line in enumerate(str(text).splitlines(), start=1):
        for pattern in FORBIDDEN_SRUKF_ROUTE_PATTERNS:
            if pattern.casefold() in line.casefold():
                violations.append(
                    SRUKFRouteGuardViolation(
                        pattern=pattern,
                        line_number=line_number,
                        line=line.strip(),
                    )
                )
    return tuple(violations)


def assert_no_forbidden_srukf_routes(
    paths: Iterable[str | Path],
) -> tuple[SRUKFRouteGuardViolation, ...]:
    """Raise if any admitted SR-UKF source path contains a forbidden route."""

    all_violations: list[SRUKFRouteGuardViolation] = []
    for path_like in paths:
        path = Path(path_like)
        violations = find_forbidden_srukf_routes(path.read_text(encoding="utf-8"))
        # The historical factor-prototype route is intentionally retained as
        # a diagnostic comparator; its existing Cholesky refactorization is
        # not part of the admitted direct-factor source boundary.
        if path.name == "srukf_factor_tf.py":
            violations = tuple(v for v in violations if v.pattern.casefold() != "cholesky")
        all_violations.extend(violations)
    if all_violations:
        formatted = "; ".join(
            f"{violation.pattern}@{violation.line_number}: {violation.line}"
            for violation in all_violations
        )
        raise ValueError(f"forbidden_srukf_route_detected: {formatted}")
    return tuple()
