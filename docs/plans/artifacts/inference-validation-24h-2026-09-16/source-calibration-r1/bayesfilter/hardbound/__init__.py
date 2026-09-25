"""Hard-bound (kink-target) shadow-rate model tools.

Program: docs/plans/hardbound-kink-hmc-master-program-2026-08-21.md.
Runtime code is TensorFlow/TFP float64; NumPy appears only in the
``reference_numpy`` diagnostic module and in tests.
"""

from bayesfilter.hardbound.dns_curve_tf import (
    dns_loadings,
    gauss_legendre_unit,
    yield_curve,
)
from bayesfilter.hardbound.model_tf import (
    FIXTURE,
    HardBoundFixture,
    observation_log_density,
    observation_mean,
    simulate,
)

__all__ = [
    "dns_loadings",
    "gauss_legendre_unit",
    "yield_curve",
    "FIXTURE",
    "HardBoundFixture",
    "observation_log_density",
    "observation_mean",
    "simulate",
]
