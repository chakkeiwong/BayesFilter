"""Public geometry and mass-construction compatibility surface."""

from bayesfilter.inference.hmc import PrecomputedMassArtifact
from bayesfilter.inference.hmc_kernel_tuning import (
    GEOMETRY_INITIALIZATION_NONCLAIMS,
    HMCGeometryInitializationConfig,
    HMCGeometryInitializationResult,
    initialize_hmc_kernel_geometry,
)
from bayesfilter.inference.mass_matrix import (
    MassMatrixResult,
    covariance_from_negative_hessian,
    covariance_from_precision,
    regularize_covariance,
    regularize_precision,
    whitening_from_covariance,
)

__all__ = [
    "GEOMETRY_INITIALIZATION_NONCLAIMS",
    "HMCGeometryInitializationConfig",
    "HMCGeometryInitializationResult",
    "MassMatrixResult",
    "PrecomputedMassArtifact",
    "covariance_from_negative_hessian",
    "covariance_from_precision",
    "initialize_hmc_kernel_geometry",
    "regularize_covariance",
    "regularize_precision",
    "whitening_from_covariance",
]
