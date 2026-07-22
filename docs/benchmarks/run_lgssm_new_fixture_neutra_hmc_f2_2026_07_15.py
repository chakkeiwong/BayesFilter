#!/usr/bin/env python3
"""CLI for NeuTra F2 new-fixture HMC admission and confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("tune-and-admit", "confirm"))
    args = parser.parse_args()
    from bayesfilter.testing import lgssm_new_fixture_neutra_hmc_f2_tf as campaign
    result = campaign.run_f2_tuning_and_admission() if args.stage == "tune-and-admit" else campaign.run_f2_confirmation()
    print(json.dumps({key: result[key] for key in ("passed", "decision", "artifact_hash")}, sort_keys=True))
    return 0 if result["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
