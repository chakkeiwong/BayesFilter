#!/usr/bin/env python3
"""Compatibility entry point for repaired SQMC diagnostic tuning.

The former Fisher/HMC proxies and legacy in-sample verdicts were invalid.
The implementation at f3995a06 remains in Git as historical evidence.
This entry point now uses run_sqmc_tuning.py arguments and writes calibration,
independent validation and untouched diagnostics. It issues no performance,
HMC, or equivalence verdict. See sqmc-repair-results-20260925.md.
"""
from run_sqmc_tuning import main

if __name__ == '__main__':
    raise SystemExit(main())
