#!/usr/bin/env python3
"""CLI for the reviewed NeuTra same-fixture third-seed robustness arm."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("tune-and-admit", "confirm"))
    args = parser.parse_args()
    from bayesfilter.testing import lgssm_neutra_robustness_s1_tf as campaign

    result: Any = (
        campaign.run_s1_tuning_and_admission()
        if args.stage == "tune-and-admit"
        else campaign.run_s1_confirmation()
    )
    print(json.dumps(_summary(result), sort_keys=True))
    return 0 if result.get("passed") is True else 1


def _summary(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: value[key]
        for key in ("candidate_id", "passed", "decision", "artifact_hash")
        if key in value
    }


if __name__ == "__main__":
    raise SystemExit(main())
