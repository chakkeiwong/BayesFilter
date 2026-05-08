"""
Particle Filtering Library.

A NumPy-based library for state space filtering with:
- Kalman filters (KF, EKF, UKF)
- Particle filters (Bootstrap/SIR)
- Particle flow filters (EDH, LEDH, kernel flow)
"""

from . import models
from . import filters
from . import simulation
from . import utils

__version__ = "0.1.0"
