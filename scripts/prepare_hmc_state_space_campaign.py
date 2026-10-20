"""Prepare, qualify, price and run the reviewed state-space validation campaign.

Run --help for the executable stages. GPU stages are deliberately explicit;
preparing reference data never starts or waits for C1 or a GPU worker.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

from bayesfilter.testing.inference_validation.ssm_campaign import main

if __name__ == "__main__":
    main()
