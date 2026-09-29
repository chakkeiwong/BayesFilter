"""Repository policy for square-root unscented Kalman filter backends.

The direct-factor route is the default SR-UKF implementation for models that
implement :class:`TFFactorSRUKFModel`.  The principal-square-root route is
retained under its existing API names for reproducibility and comparison, but
is historical/reference-only and is never selected by this policy.
"""

from __future__ import annotations

from typing import Literal


DEFAULT_SRUKF_BACKEND = "direct_factor_srukf"
HISTORICAL_PRINCIPAL_SQRT_SRUKF_BACKEND = "tf_principal_sqrt_ukf"
HISTORICAL_EIGENDERIVATIVE_SRUKF_BACKEND = "tf_svd_ukf"

SRUKFBackendStatus = Literal["default", "historical_reference"]


def default_srukf_backend() -> str:
    """Return the repository-owned default SR-UKF backend identifier."""

    return DEFAULT_SRUKF_BACKEND


def srukf_backend_status(backend: str) -> SRUKFBackendStatus:
    """Classify a known backend without changing or guessing its contract."""

    backend = str(backend)
    if backend == DEFAULT_SRUKF_BACKEND:
        return "default"
    if backend in {
        HISTORICAL_PRINCIPAL_SQRT_SRUKF_BACKEND,
        HISTORICAL_EIGENDERIVATIVE_SRUKF_BACKEND,
    }:
        return "historical_reference"
    raise ValueError(f"unknown SR-UKF backend: {backend!r}")


def resolve_srukf_backend(requested: str | None = None) -> str:
    """Resolve an SR-UKF selector without silently falling back.

    ``None`` selects the direct-factor default.  A legacy selector is returned
    unchanged so an explicit historical comparison remains possible; callers
    must still use the legacy API that implements that selector.  Unknown
    selectors fail closed.
    """

    backend = DEFAULT_SRUKF_BACKEND if requested is None else str(requested)
    srukf_backend_status(backend)
    return backend


def srukf_backend_metadata(backend: str) -> dict[str, str]:
    """Return stable, human-readable route metadata for an artifact."""

    resolved = resolve_srukf_backend(backend)
    status = srukf_backend_status(resolved)
    if status == "default":
        return {
            "backend": resolved,
            "backend_status": status,
            "backend_contract": "TFFactorSRUKFModel",
            "factorization": "direct_qr_block_conditional",
        }
    return {
        "backend": resolved,
        "backend_status": status,
        "backend_contract": "legacy_structural_covariance_model",
        "factorization": "principal_square_root_or_eigendecomposition",
    }


__all__ = [
    "DEFAULT_SRUKF_BACKEND",
    "HISTORICAL_PRINCIPAL_SQRT_SRUKF_BACKEND",
    "HISTORICAL_EIGENDERIVATIVE_SRUKF_BACKEND",
    "SRUKFBackendStatus",
    "default_srukf_backend",
    "resolve_srukf_backend",
    "srukf_backend_metadata",
    "srukf_backend_status",
]
