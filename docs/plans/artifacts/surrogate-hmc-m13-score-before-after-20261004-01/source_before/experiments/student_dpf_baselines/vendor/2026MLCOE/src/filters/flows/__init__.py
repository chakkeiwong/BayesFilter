"""
Flows Package
=============
The `flows` package is the core mathematical engine of the Particle Flow framework.
It encapsulates the velocity field definitions (dz/dlambda) used to migrate
particle ensembles from a prior distribution toward a posterior distribution.

Architecture & Modularity:
-------------------------
By decoupling the 'drift' logic from the sequential filtering 'shell', this
package allows for independent testing, optimization, and extension of
migration algorithms. This satisfies industrial requirements for modular
boundaries and facilitates research into different homotopy trajectories
without altering the underlying filter infrastructure.

Core Methodology:
----------------
The migration is governed by the general homotopy-based ODE:
    dz/d_lambda = f(z, lambda)
where 'z' is the state, 'lambda' is the homotopy parameter [0, 1], and
'f' is the velocity field.

Available Modules:
------------------
1. **daum_huang**: Implements the Exact Daum-Huang (EDH) and Local Exact
   Daum-Huang (LEDH) velocity fields. These rely on analytic solutions to
   the log-posterior gradient flow, using either global or local Jacobians.

2. **dai_homotopy**: Implements the 'LEDHOptimizedFlow' which solves a
   Boundary Value Problem (BVP) via a shooting method to find a schedule
   beta(lambda) that minimizes numerical stiffness (Dai et al., 2022).

3. **stochastic**: Contains the 'StochasticParticleFlow' (SPF) based on
   Theorem 2.1. This module is used for stiffness validation and
   monitoring the condition number of the Hessian matrix (M) throughout
    the migration path.

4. **kernel**: Implements the 'KernelFlow' which utilizes a model-free
   Reproducing Kernel Hilbert Space (RKHS) embedding. This moves the
   ensemble as a collective unit, minimizing KL-divergence without
   requiring analytical Jacobians of the observation model.

Usage for Teammates:
-------------------
To integrate a new drift formula:
    1. Define the velocity calculation in a new module within this package.
    2. Expose the class in this `__init__.py` file.
    3. Call the `compute_velocity` or equivalent method from the `step()`
       function in the `filters/flow_filters.py` module.

Author: Joowon Lee
Date: 2026-02-26
"""

from .EDH import EDHSolver
from .LEDH import LEDHSolver
from .dai_homotopy import LEDHOptimizedFlow
from .stochastic import StochasticParticleFlow
from .kernel import KernelFlow
