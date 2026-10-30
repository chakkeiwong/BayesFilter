"""
Find-Mode-Then-Sample pipeline for SVSSM Bayesian calibration.

Three phases:
  1. MAP via Adam + DPF-soft (adam_mle.py)
  2. Laplace covariance via finite-difference Hessian (laplace.py)
  3. Optional: whitened local HMC around the mode (whitened_hmc.py)

The key idea is that each phase sidesteps the DPF gradient-magnitude bias
differently:
  - Adam's running-second-moment normalization cancels the ~25-30% attenuation
  - Laplace uses only log-likelihood *values* (not gradients)
  - Local HMC doesn't give attenuation distance to accumulate over trajectories

See mle/run_mle_laplace.py for the driver.
"""

from .adam_mle import run_adam_mle, AdamMLEResult
from .laplace import compute_laplace_covariance, LaplaceResult
from .whitened_hmc import run_whitened_hmc, WhitenedHMCResult
from .preconditioned_hmc import run_preconditioned_hmc, PreconditionedHMCResult
# validate_adam_mle defines _unpack_eager / _log_prior_eager that
# validate_fd_hessian imports, so it must be loaded first.
from .validate_adam_mle import (
    validate_adam_mle, run_adam_on_kalman, AdamValidationResult,
)
from .validate_fd_hessian import (
    validate_fd_hessian, kalman_logpost_hessian, convergence_sweep,
)
