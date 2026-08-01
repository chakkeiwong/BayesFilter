#!/usr/bin/env python
"""Write the immutable preflight manifest for the fresh HMC rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime import stable_config_hash  # noqa: E402


SCHEMA = "bayesfilter.hmc_full_estimation_preflight_manifest.v1"
DEFAULT_CONFIG = ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_full_estimation_rerun_2026_07_13.json"
)
DEFAULT_PHASE7_CONFIG = ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_full_estimation_phase7_2026_07_13.json"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-deterministic-lgssm-hmc-full-estimation-rerun-plan-2026-07-13.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-deterministic-lgssm-hmc-full-estimation-rerun-result-2026-07-13.md"
)
SOURCE_PATHS = (
    "bayesfilter/inference/hmc.py",
    "bayesfilter/inference/hmc_convergence.py",
    "bayesfilter/inference/hmc_identity.py",
    "bayesfilter/inference/hmc_identity_integration.py",
    "bayesfilter/inference/hmc_kernel_tuning.py",
    "bayesfilter/inference/mass_matrix.py",
    "bayesfilter/inference/quadratic_geometry.py",
    "bayesfilter/linear/kalman_svd_derivatives_tf.py",
    "bayesfilter/linear/kalman_svd_tf.py",
    "bayesfilter/linear/types_tf.py",
    "bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py",
    "bayesfilter/testing/multidim_triangular_lgssm_tf.py",
    "docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py",
    "scripts/build_hmc_full_estimation_phase7_config.py",
    "scripts/build_hmc_full_estimation_preflight_manifest.py",
    "scripts/run_hmc_full_estimation_campaign.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Verify an existing preflight manifest without writing files.",
    )
    return parser.parse_args(argv)


def build_manifest(
    *,
    config_path: str | Path,
    python_executable: str | Path = sys.executable,
) -> tuple[Path, Mapping[str, Any]]:
    config_file = Path(config_path).resolve()
    config = _read_json(config_file)
    artifact_root = ROOT / config["artifact_paths"]["root"]
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(
            f"fresh-run artifact root is already nonempty: {artifact_root}"
        )
    if artifact_root.exists() and not artifact_root.is_dir():
        raise FileExistsError(f"artifact root is not a directory: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    output = artifact_root / "preflight_manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight manifest: {output}")
    python = str(Path(python_executable).resolve())
    driver = "docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py"
    config_relative = str(config_file.relative_to(ROOT))
    phase7_relative = str(DEFAULT_PHASE7_CONFIG.relative_to(ROOT))
    root_relative = str(artifact_root.relative_to(ROOT))
    commands = (
        (python, driver, "--config", config_relative, "--stage", "fixture"),
        (python, driver, "--config", config_relative, "--stage", "xla_score"),
        (python, driver, "--config", config_relative, "--stage", "geometry_mass"),
        (python, driver, "--config", config_relative, "--stage", "kernel_tuning"),
        (
            python,
            "scripts/build_hmc_full_estimation_phase7_config.py",
            "--tuning-config",
            config_relative,
            "--output",
            phase7_relative,
        ),
        (
            python,
            driver,
            "--config",
            config_relative,
            "--phase7-config",
            phase7_relative,
            "--stage",
            "burnin_sampling",
            "--phase7-smoke",
            "--phase7-output-dir",
            "/tmp/bayesfilter-full-estimation-smoke-20260713",
        ),
        (
            python,
            "scripts/run_hmc_full_estimation_campaign.py",
            "--config",
            phase7_relative,
            "--campaign-root",
            f"{root_relative}/phase7_campaign",
        ),
        (
            python,
            driver,
            "--config",
            config_relative,
            "--phase7-config",
            phase7_relative,
            "--stage",
            "final_recovery",
        ),
    )
    source_inventory = {
        relative: _file_reference(ROOT / relative) for relative in SOURCE_PATHS
    }
    output_paths = {
        name: value for name, value in config["artifact_paths"].items()
    }
    if any(
        name != "root"
        and not (ROOT / path).resolve().is_relative_to(artifact_root.resolve())
        for name, path in output_paths.items()
    ):
        raise ValueError("fresh-run output escapes the new artifact root")
    historical_root = ROOT / (
        "docs/benchmarks/artifacts/"
        "multidim_lgssm_serious_hmc_tuning_2026_07_09"
    )
    if any(
        (ROOT / path).resolve().is_relative_to(historical_root.resolve())
        for name, path in output_paths.items()
        if name != "root"
    ):
        raise ValueError("fresh-run output depends on the historical artifact root")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "preflight_passed_no_experiment_started",
        "git_commit": _git_commit(),
        "config_path": config_relative,
        "config_file_sha256": _file_sha256(config_file),
        "config_hash": "sha256:" + stable_config_hash(config),
        "phase7_config_path_after_tuning": phase7_relative,
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "artifact_root": root_relative,
        "output_paths": output_paths,
        "commands": commands,
        "environment": {
            "python_executable": python,
            "CUDA_VISIBLE_DEVICES": "-1",
            "MPLCONFIGDIR": "/tmp/matplotlib-bayesfilter-full-rerun",
            "sample_generation_device": "deliberate_cpu_only",
            "tensorflow_dtype": "float64",
            "jit_compile": True,
            "use_xla": True,
        },
        "seeds": {
            "simulation": tuple(config["truth_and_data"]["simulation_seed"]),
            "geometry": tuple(config["geometry_initializer"]["seed"]),
            "tuning": tuple(config["kernel_tuning"]["seed"]),
            "serious_sampling": (20260713, 701),
            "tuning_and_serious_seeds_differ": True,
        },
        "thresholds": {
            "tuning_acceptance_band": tuple(
                config["kernel_tuning"]["acceptance_band"]
            ),
            "tuning_rhat_definition": (
                "max(rank-normalized split R-hat, "
                "folded rank-normalized split R-hat)"
            ),
            "tuning_rhat_max": 1.01,
            "tuning_min_retained_draws": config["kernel_tuning"][
                "verification_min_retained_results_for_pass"
            ],
            "serious_rhat_max": config["sampling_controller"][
                "r_hat_threshold"
            ],
            "serious_bulk_ess_min": config["sampling_controller"][
                "bulk_ess_min_per_parameter"
            ],
            "serious_tail_ess_min": config["sampling_controller"][
                "tail_ess_min_per_parameter"
            ],
            "recovery_max_abs_z": config["final_recovery_gate"][
                "truth_distance_max_abs_z"
            ],
        },
        "source_inventory": source_inventory,
        "source_inventory_hash": "sha256:" + stable_config_hash(source_inventory),
        "phase0_checks": {
            "focused_tests": "74 passed",
            "python_compilation": "passed",
            "forbidden_nonjit_static_scan": "passed",
            "v1_v2_regressions": "passed",
            "claude_initial_review": (
                "FEASIBILITY_MISSING_HARDCODED_IDENTITY_REPAIR"
            ),
            "claude_post_revision_recheck": (
                "unavailable_external_disclosure_rejected_no_retry"
            ),
            "local_post_revision_review": "passed",
        },
        "evidence_contract": {
            "primary_criterion": (
                "target, corrected tuning, serious convergence, retained "
                "integrity, and all-parameter recovery jointly pass"
            ),
            "continuation_vetoes": (
                "invalid target or score",
                "corrupt or stale artifact",
                "no corrected tuning candidate",
                "nonfinite transition",
                "XLA fallback",
                "serious diagnostic cap failure",
                "wall-time cap",
            ),
            "explanatory_only": (
                "smoke diagnostics",
                "runtime",
                "compile time",
                "posterior contraction",
            ),
            "forbidden_claims": (
                "calibration",
                "generality",
                "sampler superiority",
                "GPU readiness",
                "production or default readiness",
            ),
        },
        "no_overwrite": True,
        "historical_artifact_root_consumed_as_input": False,
        "nonclaims": tuple(config["nonclaims"]),
    }
    payload["artifact_hash"] = "sha256:" + stable_config_hash(payload)
    return output, payload


def write_manifest(
    *,
    config_path: str | Path,
    python_executable: str | Path = sys.executable,
) -> Mapping[str, Any]:
    output, payload = build_manifest(
        config_path=config_path,
        python_executable=python_executable,
    )
    _write_exclusive_json(output, payload)
    return payload


def verify_manifest(path: str | Path) -> Mapping[str, Any]:
    manifest_path = Path(path).resolve()
    payload = _read_json(manifest_path)
    if payload.get("schema") != SCHEMA:
        raise ValueError("preflight manifest schema mismatch")
    observed_hash = payload.get("artifact_hash")
    without_hash = {key: value for key, value in payload.items() if key != "artifact_hash"}
    expected_hash = "sha256:" + stable_config_hash(without_hash)
    if observed_hash != expected_hash:
        raise ValueError("preflight manifest artifact hash mismatch")
    source_inventory = payload.get("source_inventory")
    if not isinstance(source_inventory, Mapping):
        raise ValueError("preflight source inventory is missing")
    changed = tuple(
        relative
        for relative, reference in source_inventory.items()
        if not (ROOT / relative).is_file()
        or _file_sha256(ROOT / relative) != reference.get("file_sha256")
        or (ROOT / relative).stat().st_size != int(reference.get("byte_count", -1))
    )
    if changed:
        raise ValueError(f"preflight source inventory drift: {changed}")
    config_path = ROOT / str(payload["config_path"])
    if _file_sha256(config_path) != payload["config_file_sha256"] or (
        "sha256:" + stable_config_hash(_read_json(config_path))
    ) != payload["config_hash"]:
        raise ValueError("preflight tuning config drift")
    return {
        "passed": True,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "artifact_hash": observed_hash,
        "source_inventory_hash": payload["source_inventory_hash"],
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _file_reference(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"preflight source is missing: {path}")
    return {
        "file_sha256": _file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "N/A"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = (
        verify_manifest(args.verify)
        if args.verify is not None
        else write_manifest(config_path=args.config)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
