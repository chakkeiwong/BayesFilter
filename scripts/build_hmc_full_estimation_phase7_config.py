#!/usr/bin/env python
"""Build a direct fresh-run Phase 7 config from passed tuning artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_identity import (  # noqa: E402
    canonical_artifact_payload_hash,
)
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (  # noqa: E402
    PHASE7_CONFIG_SCHEMA_V3,
    DeterministicLGSSMPhase7Config,
    build_phase7_live_identity_bundle,
    validate_phase7_v3_inputs,
)
from docs.benchmarks import (  # noqa: E402
    run_multidim_lgssm_serious_hmc_tuning_2026_07_09 as tuning_driver,
)


PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-deterministic-lgssm-hmc-full-estimation-rerun-plan-2026-07-13.md"
)
DEFAULT_TUNING_CONFIG = ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_full_estimation_rerun_2026_07_13.json"
)
DEFAULT_OUTPUT = ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_full_estimation_phase7_2026_07_13.json"
)
SOURCE_KEYS = (
    "fixture",
    "xla_compile",
    "geometry",
    "mass",
    "kernel",
    "private_replay",
    "source_tuning_config",
    "source_contract",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--tuning-config", type=Path, default=DEFAULT_TUNING_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def build_phase7_v3_payload(
    *,
    tuning_config_path: str | Path,
    output_path: str | Path,
) -> Mapping[str, Any]:
    tuning_config = tuning_driver.DeterministicLGSSMHMCConfig.load(
        tuning_config_path
    )
    if not tuning_config.no_overwrite:
        raise ValueError("fresh Phase 7 builder requires no-overwrite tuning config")
    artifact_root = tuning_config.artifact_root
    output = Path(output_path)
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 7 config: {output}")
    paths = {
        "fixture": tuning_config.fixture_path,
        "xla_compile": tuning_config.xla_compile_path,
        "geometry": tuning_config.geometry_path,
        "mass": tuning_config.mass_path,
        "kernel": tuning_config.kernel_tuning_path,
        "private_replay": tuning_config.private_tuning_replay_path,
        "source_tuning_config": Path(tuning_config.path),
        "source_contract": tuning_config.source_contract_path,
    }
    if tuple(paths) != SOURCE_KEYS:
        raise AssertionError("builder source inventory drift")
    snapshots = {name: _json_snapshot(path) for name, path in paths.items()}
    kernel = snapshots["kernel"]["payload"]
    private_replay = snapshots["private_replay"]["payload"]
    if kernel.get("passed") is not True or kernel.get("final_status") != "passed":
        raise ValueError("fresh Phase 7 config requires passed kernel tuning")
    if private_replay.get("tuning_payload", {}).get(
        "tune_verify_repair_loop", {}
    ).get("passed") is not True:
        raise ValueError("fresh Phase 7 config requires passed private tuning loop")
    root_relative = artifact_root.resolve().relative_to(ROOT)
    campaign_root = root_relative / "phase7_campaign"
    references = {
        name: {
            "path": str(path.resolve().relative_to(ROOT)),
            "schema": snapshots[name]["payload"]["schema"],
            "file_sha256": snapshots[name]["file_sha256"],
            "byte_count": snapshots[name]["byte_count"],
            "artifact_hash": snapshots[name]["payload"].get("artifact_hash"),
        }
        for name, path in paths.items()
    }
    placeholder = "sha256:" + "0" * 64
    payload: dict[str, Any] = {
        "schema": PHASE7_CONFIG_SCHEMA_V3,
        "config_id": "multidim_lgssm_full_estimation_phase7_2026_07_13",
        "plan_path": PLAN_PATH,
        "source_tuning_config_path": str(
            Path(tuning_config.path).resolve().relative_to(ROOT)
        ),
        "source_tuning_config_hash": tuning_config.hash,
        "artifact_root": str(root_relative),
        "execution": {
            "worker_count": 2,
            "chains_per_worker": 2,
            "root_seed": [20260713, 701],
            "cuda_visible_devices": "-1",
            "jit_compile": True,
            "use_xla": True,
            "chain_execution_mode": "tf_function",
            "compile_workers_sequentially": True,
            "wall_time_cap_seconds": 28800,
            "thread_environment": {
                "TF_NUM_INTRAOP_THREADS": "8",
                "TF_NUM_INTEROP_THREADS": "1",
                "OMP_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "burnin": {
            "initial_results_per_chain": 2000,
            "extension_results_per_chain": 1000,
            "check_window_results_per_chain": 1000,
            "max_results_per_chain": 16000,
        },
        "retained": {
            "initial_results_per_chain": 4000,
            "extension_results_per_chain": 2000,
            "check_interval_results_per_chain": 2000,
            "max_results_per_chain": 40000,
        },
        "diagnostics": {
            "rhat_max": 1.01,
            "bulk_ess_min": 1000.0,
            "tail_ess_min": 400.0,
            "all_parameters_required": True,
            "coordinate_system": (
                "raw_lgssm_parameters_after_two_mass_transforms"
            ),
            "rhat_definition": (
                "max(rank-normalized split R-hat, "
                "folded rank-normalized split R-hat)"
            ),
        },
        "artifacts": {
            "public_result": str(campaign_root / "result.json"),
            "public_progress": str(campaign_root / "progress.json"),
            "private_replay": references["private_replay"]["path"],
            "private_retained_samples": str(
                campaign_root / "private" / "retained_samples.npz"
            ),
        },
        "governed_source_references": references,
        "expected_identities": {
            "transition_identity_hash": placeholder,
            "serious_execution_contract_hash": placeholder,
            "smoke_execution_contract_hash": placeholder,
            "selection_provenance_hash": placeholder,
            "complete_tuning_payload_hash": placeholder,
        },
        "fresh_run_policy": {
            "historical_identity_inputs_allowed": False,
            "migration_certificates_required": False,
            "approval_manifests_required": False,
            "no_overwrite": True,
            "tuning_root_seed": [20260709, 501],
            "serious_root_seed": [20260713, 701],
        },
        "nonclaims": [
            "fresh direct local academic HMC campaign only",
            "tuning pass is a kernel handoff screen, not posterior convergence evidence",
            "single-fixture recovery does not establish calibration or generality",
            "not sampler superiority evidence",
            "not GPU, production, or default readiness evidence",
        ],
    }
    provisional = DeterministicLGSSMPhase7Config(payload=payload, path=output)
    provisional.validate()
    live = build_phase7_live_identity_bundle(provisional)
    payload["expected_identities"] = {
        "transition_identity_hash": live.transition.identity_hash,
        "serious_execution_contract_hash": live.serious_execution.identity_hash,
        "smoke_execution_contract_hash": live.smoke_execution.identity_hash,
        "selection_provenance_hash": live.provenance.identity_hash,
        "complete_tuning_payload_hash": canonical_artifact_payload_hash(
            private_replay["tuning_payload"]
        ),
    }
    final = DeterministicLGSSMPhase7Config(payload=payload, path=output)
    final.validate()
    return payload


def write_and_validate_phase7_v3(
    *,
    tuning_config_path: str | Path,
    output_path: str | Path,
) -> Mapping[str, Any]:
    output = Path(output_path)
    output = output if output.is_absolute() else ROOT / output
    payload = build_phase7_v3_payload(
        tuning_config_path=tuning_config_path,
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 7 config: {output}")
    _write_exclusive_json(output, payload)
    loaded = DeterministicLGSSMPhase7Config.load(output)
    try:
        preflight = validate_phase7_v3_inputs(loaded)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {
        "config_path": str(output.relative_to(ROOT)),
        "config_file_sha256": _file_sha256(output),
        "config_hash": loaded.hash,
        "preflight_artifact_hash": preflight["artifact_hash"],
        "expected_identities": dict(loaded.payload["expected_identities"]),
    }


def _json_snapshot(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return {
        "payload": payload,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = write_and_validate_phase7_v3(
        tuning_config_path=args.tuning_config,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
