#!/usr/bin/env python3
"""CLI for NeuTra F1 new-fixture target-specific GPU training."""

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
    parser.add_argument("stage", choices=("train", "finalize-screen"))
    parser.add_argument("--job-kind", choices=("screen", "final"))
    parser.add_argument("--recipe-id")
    args = parser.parse_args()
    from bayesfilter.testing import lgssm_new_fixture_neutra_training_f1_tf as campaign

    if args.stage == "train":
        if args.job_kind is None or args.recipe_id is None:
            raise ValueError("train requires --job-kind and --recipe-id")
        result = campaign.run_training_job(job_kind=args.job_kind, recipe_id=args.recipe_id)
    else:
        result = campaign.finalize_screen()
    print(json.dumps({key: result[key] for key in result if key in {"passed", "decision", "recipe_id", "selected_recipe_id", "artifact_hash"}}, sort_keys=True))
    return 0 if result["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
