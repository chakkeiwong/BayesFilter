"""
Inference Module
================
Exposes PHMC and PMMH engines for parameter estimation.

Author: Joowon Lee
Date: 2026-02-27
"""

from .phmc import hmc_pfpf
from .pmmh import pmmh_bpf

# Expose DTYPE for convenience, but import it from its true home
from src.filters.classical import DTYPE
