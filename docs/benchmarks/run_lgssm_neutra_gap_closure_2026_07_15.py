#!/usr/bin/env python3
"""CLI for the TensorFlow-only LGSSM NeuTra gap-closure campaign."""

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
            "phase0-local",
            "phase0-gaussian",
            "phase0-frozen-smoke",
            "post-validate-training",
            "frozen-probe",
            "frozen-finalize",
            "tune-candidate",
            "tune-finalize",
            "sequential-candidate",
            "sequential-finalize",
            "confirm-candidate",
            "confirm-finalize",
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument(
        "--device-mode", choices=("trusted_gpu_xla", "cpu_hidden_xla")
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    from bayesfilter.testing import lgssm_neutra_gap_closure_tf as campaign

    if args.stage == "phase0-local":
        result: Any = campaign.phase0_local_checks()
    elif args.stage == "phase0-gaussian":
        result = campaign.run_gaussian_phase0_diagnostic(output_path=args.output)
    elif args.stage == "phase0-frozen-smoke":
        result = campaign.run_screen_frozen_phase0_smoke(output_path=args.output)
    elif args.stage == "post-validate-training":
        if args.output is None or args.attempt_root is None or args.job_id is None:
            raise ValueError(
                "post-validate-training requires --output, --attempt-root, and --job-id"
            )
        result = campaign.post_validate_completed_training_attempt(
            args.attempt_root,
            expected_job_id=args.job_id,
            output_path=args.output,
        )
    elif args.stage == "frozen-probe":
        if args.job_id is None or args.device_mode is None:
            raise ValueError("frozen-probe requires --job-id and --device-mode")
        result = campaign.run_fresh_frozen_objective_probe(
            args.job_id,
            device_mode=args.device_mode,
            output_path=args.output,
        )
    elif args.stage == "frozen-finalize":
        result = campaign.finalize_frozen_objective_validation(output_path=args.output)
    elif args.stage == "tune-candidate":
        if args.job_id is None:
            raise ValueError("tune-candidate requires --job-id")
        result = campaign.run_hmc_tuning_candidate(
            args.job_id,
            output_path=args.output,
        )
    elif args.stage == "tune-finalize":
        result = campaign.finalize_hmc_tuning(output_path=args.output)
    elif args.stage == "sequential-candidate":
        if args.job_id is None:
            raise ValueError("sequential-candidate requires --job-id")
        result = campaign.run_corrected_sequential_hmc_candidate(
            args.job_id,
            output_path=args.output,
        )
    elif args.stage == "sequential-finalize":
        result = campaign.finalize_corrected_sequential_hmc(output_path=args.output)
    elif args.stage == "confirm-candidate":
        if args.job_id is None:
            raise ValueError("confirm-candidate requires --job-id")
        result = campaign.run_confirmatory_hmc_candidate(
            args.job_id,
            output_path=args.output,
        )
    else:
        result = campaign.finalize_confirmatory_hmc(output_path=args.output)
    print(json.dumps(_summary(result), sort_keys=True))
    return 0 if result.get("passed") is True else 1


def _summary(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    keys = ("passed", "decision", "artifact_hash", "selected_recipe")
    return {key: value[key] for key in keys if key in value}


if __name__ == "__main__":
    raise SystemExit(main())
