#!/usr/bin/env python3
"""CLI for the reviewed exact-target LGSSM NeuTra validation campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "contract",
            "phase3",
            "phase3-score-parity",
            "phase4-candidate",
            "phase4-finalize",
            "phase5-candidate",
            "phase5-finalize",
            "phase6-candidate",
            "phase6-finalize",
        ),
    )
    parser.add_argument(
        "--candidate",
        choices=("affine_control", "dense_seed1201", "dense_seed1202"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    from bayesfilter.testing import lgssm_neutra_serious_validation_tf as campaign

    if args.stage == "contract":
        result: Any = {"path": str(campaign.write_campaign_contract())}
    elif args.stage == "phase3":
        result = campaign.run_phase3_gpu_canary()
    elif args.stage == "phase3-score-parity":
        result = campaign.run_phase3_gpu_score_parity_addendum()
    elif args.stage == "phase4-candidate":
        _require_candidate(args.candidate)
        result = campaign.run_phase4_gpu_candidate(args.candidate)
    elif args.stage == "phase4-finalize":
        result = campaign.finalize_phase4()
    elif args.stage == "phase5-candidate":
        _require_candidate(args.candidate)
        result = campaign.run_phase5_candidate(args.candidate)
    elif args.stage == "phase5-finalize":
        result = campaign.finalize_phase5()
    elif args.stage == "phase6-candidate":
        _require_candidate(args.candidate)
        result = campaign.run_phase6_candidate(args.candidate)
    else:
        result = campaign.finalize_phase6()
    print(json.dumps(_summary(result), sort_keys=True))
    return 0


def _require_candidate(value: str | None) -> None:
    if value is None:
        raise ValueError("--candidate is required for candidate stages")


def _summary(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    keys = (
        "path",
        "phase",
        "candidate_id",
        "passed",
        "admitted",
        "decision",
        "artifact_hash",
        "viable_candidates",
        "admitted_candidates",
    )
    return {key: value[key] for key in keys if key in value}


if __name__ == "__main__":
    raise SystemExit(main())
