#!/usr/bin/env python3
"""Continue the existing surrogate-HMC master from its saved M13 state."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayesfilter.inference.surrogate_hmc_resume_program import main

if __name__ == "__main__":
    raise SystemExit(main())
