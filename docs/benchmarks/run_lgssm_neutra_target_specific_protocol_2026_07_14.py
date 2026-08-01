#!/usr/bin/env python3
"""CLI for the reviewed target-specific LGSSM NeuTra protocol."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "contract",
            "train",
            "screen-finalize",
            "phase4-finalize",
            "phase5-candidate",
            "phase5-finalize",
            "phase6-candidate",
            "phase6-finalize",
        ),
    )
    parser.add_argument("--job-kind", choices=("smoke", "screen", "final"))
    parser.add_argument("--job-id")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--artifact-root")
    parser.add_argument("--selected-recipe")
    parser.add_argument("--step-override", type=int)
    parser.add_argument(
        "--candidate",
        choices=("affine_control", "dense_seed1201", "dense_seed1202"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()

    if args.stage == "train":
        from bayesfilter.testing import lgssm_neutra_strict_training_tf as campaign

        if args.resume_checkpoint is not None:
            raise ValueError(
                "graph-native training uses terminal-only checkpoints and does not "
                "support implicit infrastructure resume"
            )
        if args.job_kind is None or args.job_id is None:
            raise ValueError("train requires --job-kind and --job-id")
        result: Any = campaign.run_gpu_training_job(
            job_kind=args.job_kind,
            job_id=args.job_id,
            artifact_root=args.artifact_root,
            selected_recipe_path=args.selected_recipe,
            step_override=args.step_override,
        )
        print(json.dumps(_summary(result), sort_keys=True))
        return 0

    if args.artifact_root is not None or args.selected_recipe is not None or args.step_override is not None:
        raise ValueError(
            "artifact-root, selected-recipe, and step-override are train-stage options only"
        )
    from bayesfilter.testing import lgssm_neutra_target_specific_protocol_tf as campaign

    if args.stage == "contract":
        result: Any = campaign.write_campaign_contract()
    elif args.stage == "screen-finalize":
        result = campaign.finalize_screen()
    elif args.stage == "phase4-finalize":
        result = campaign.finalize_phase4()
    elif args.stage == "phase5-candidate":
        if args.candidate is None:
            raise ValueError("candidate stage requires --candidate")
        result = campaign.run_phase5_candidate(args.candidate)
    elif args.stage == "phase5-finalize":
        result = campaign.finalize_phase5()
    elif args.stage == "phase6-candidate":
        if args.candidate is None:
            raise ValueError("candidate stage requires --candidate")
        result = campaign.run_phase6_candidate(args.candidate)
    else:
        result = campaign.finalize_phase6()
    print(json.dumps(_summary(result), sort_keys=True))
    return 0


def _summary(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    keys = (
        "path",
        "contract_hash",
        "phase",
        "job_kind",
        "job_id",
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
