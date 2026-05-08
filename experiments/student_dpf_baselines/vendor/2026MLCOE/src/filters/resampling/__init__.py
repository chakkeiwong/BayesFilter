"""
Resampling Package
==================
This package implements differentiable resampling algorithms for Sequential
Monte Carlo (SMC) methods. These algorithms are essential for training
neural state-space models, as they allow gradients to flow from the
marginal likelihood objective back to the transition and emission parameters.

Available Modules:
------------------
1. **soft**: Implements Soft Resampling, which uses a convex combination of
   standard resampling and identity to maintain a differentiable signal.

2. **optimal_transport**: Implements entropy-regularized Optimal Transport
   (EOT) via the Sinkhorn algorithm, enabling smooth particle migration.

3. **transformer**: Implements the Particle Transformer, which utilizes
   multi-head attention to learn the optimal resampling mapping in a
   data-driven manner.

Author: Joowon Lee
Date: 2026-02-26
"""

from .optimal_transport import SinkhornResampler
from .soft import SoftResampler
from .transformer import TransformerResampler
