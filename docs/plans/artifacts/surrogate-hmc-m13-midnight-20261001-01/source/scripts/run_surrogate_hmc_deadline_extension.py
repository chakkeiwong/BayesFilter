#!/usr/bin/env python3
"""Continue the active M13 master through the owner-authorized midnight limit."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bayesfilter.inference.surrogate_hmc_deadline_extension import main

if __name__ == "__main__":
    raise SystemExit(main())
