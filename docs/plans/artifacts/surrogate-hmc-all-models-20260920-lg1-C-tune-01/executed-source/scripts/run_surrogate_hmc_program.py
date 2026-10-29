#!/usr/bin/env python3
"""Repository-wide surrogate-HMC master entry point; see --help."""
import sys
import os
from pathlib import Path

# Orchestration and --dry-run must not trigger the optional test bootstrap.
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayesfilter.inference.surrogate_hmc_program import main

if __name__ == "__main__":
    raise SystemExit(main())
