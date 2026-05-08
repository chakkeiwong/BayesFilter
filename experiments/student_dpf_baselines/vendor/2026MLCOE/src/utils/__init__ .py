"""
Utilities Package
=================
This package provides supporting tools for the filtering and inference pipeline.
It includes mathematical metrics (RMSE, OMAT, ESS), hardware diagnostic tools
(memory/CPU tracking), and standardized plotting functions to ensure consistent
reporting of results.

Modules:
--------
1. **metrics**: Theoretical and empirical performance indicators.
2. **diagnostics**: System-level resource tracking (CPU/RAM).
3. **plotting**: Visualization templates for state tracking and error analysis.

Author: Joowon Lee
Date: 2026-02-26
"""

from .metrics import rmse, compute_ess, compute_omat
from .diagnostics import get_process_stats
from .plotting import setup_plotting_style, plot_state_tracking
