#!/usr/bin/env python3
"""Active sequential-recovery phases of the surrogate-HMC master."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bayesfilter.inference.surrogate_hmc_recovery_program import main
if __name__=="__main__":
    raise SystemExit(main())
