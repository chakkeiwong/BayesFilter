#!/usr/bin/env python
"""Run one no-overwrite direct academic Phase 7 full-estimation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime import stable_config_hash  # noqa: E402
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (  # noqa: E402
    PHASE7_CONFIG_SCHEMA_V3,
    DeterministicLGSSMPhase7Config,
    run_phase7,
    validate_phase7_v3_inputs,
)


MANIFEST_SCHEMA = "bayesfilter.hmc_full_estimation_campaign_manifest.v1"
TERMINAL_SCHEMA = "bayesfilter.hmc_full_estimation_campaign_terminal.v1"
FAILURE_SCHEMA = "bayesfilter.hmc_full_estimation_campaign_launcher_failure.v1"
DEFAULT_CONFIG = ROOT / (
    "docs/benchmarks/configs/"
    "multidim_lgssm_full_estimation_phase7_2026_07_13.json"
)
REQUIRED_IMPLEMENTATION_SOURCES = (
    "bayesfilter/inference/batched_value_score.py",
    "bayesfilter/inference/hmc.py",
    "bayesfilter/inference/hmc_budget_ladder.py",
    "bayesfilter/inference/hmc_convergence.py",
    "bayesfilter/inference/hmc_diagnostics.py",
    "bayesfilter/inference/hmc_identity.py",
    "bayesfilter/inference/hmc_identity_integration.py",
    "bayesfilter/inference/hmc_kernel_tuning.py",
    "bayesfilter/inference/hmc_tuning.py",
    "bayesfilter/inference/mass_matrix.py",
    "bayesfilter/inference/posterior_adapter.py",
    "bayesfilter/inference/quadratic_geometry.py",
    "bayesfilter/linear/kalman_svd_derivatives_tf.py",
    "bayesfilter/linear/kalman_svd_tf.py",
    "bayesfilter/linear/types_tf.py",
    "bayesfilter/runtime/runner.py",
    "bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py",
    "bayesfilter/testing/multidim_triangular_lgssm_tf.py",
    "docs/benchmarks/run_multidim_lgssm_serious_hmc_tuning_2026_07_09.py",
    "scripts/run_hmc_full_estimation_campaign.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args(argv)


def build_run_manifest(
    *,
    config: DeterministicLGSSMPhase7Config,
    preflight: Mapping[str, Any],
    command: Sequence[str],
    campaign_root: Path,
) -> Mapping[str, Any]:
    import tensorflow as tf
    import tensorflow_probability as tfp

    inventory = _loaded_repository_source_inventory()
    payload = {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": config.payload["config_id"],
        "git_commit": _git_commit(),
        "command": tuple(str(item) for item in command),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "tfp_version": tfp.__version__,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "conda_prefix": os.environ.get("CONDA_PREFIX", "N/A"),
        "device": "cpu_only_cuda_visible_devices_minus_1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "float64",
        "jit_compile": True,
        "use_xla": True,
        "config_path": str(config.path.resolve().relative_to(ROOT)),
        "config_file_sha256": _file_sha256(config.path),
        "config_hash": config.hash,
        "source_tuning_config_hash": config.payload["source_tuning_config_hash"],
        "data_version": config.payload["governed_source_references"]["fixture"][
            "file_sha256"
        ],
        "root_seed": tuple(config.payload["execution"]["root_seed"]),
        "worker_count": config.worker_count,
        "chains_per_worker": config.chains_per_worker,
        "thread_environment": dict(
            config.payload["execution"]["thread_environment"]
        ),
        "wall_time_cap_seconds": config.payload["execution"][
            "wall_time_cap_seconds"
        ],
        "wall_time_seconds": "recorded_in_campaign_terminal",
        "campaign_root": str(campaign_root.relative_to(ROOT)),
        "artifact_paths": dict(config.payload["artifacts"]),
        "plan_path": config.payload["plan_path"],
        "preflight_artifact_hash": preflight["artifact_hash"],
        "transition_identity_hash": config.payload["expected_identities"][
            "transition_identity_hash"
        ],
        "serious_execution_contract_hash": config.payload[
            "expected_identities"
        ]["serious_execution_contract_hash"],
        "governed_source_references": dict(
            config.payload["governed_source_references"]
        ),
        "implementation_source_inventory": inventory,
        "implementation_inventory_hash": (
            "sha256:" + stable_config_hash(inventory)
        ),
        "fresh_run_policy": dict(config.payload["fresh_run_policy"]),
        "trust_basis": "local_direct_academic_cpu_xla_execution",
        "nonclaims": tuple(config.payload["nonclaims"]),
    }
    return _with_hash(payload)


def run_campaign(
    *,
    config_path: str | Path,
    campaign_root: str | Path,
    command: Sequence[str],
) -> Mapping[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("full-estimation campaign requires CUDA_VISIBLE_DEVICES=-1")
    config = DeterministicLGSSMPhase7Config.load(config_path)
    if config.payload.get("schema") != PHASE7_CONFIG_SCHEMA_V3:
        raise ValueError("full-estimation campaign requires Phase 7 V3 config")
    root = Path(campaign_root).resolve()
    expected_root = config.artifact_path("public_result").parent.resolve()
    if root != expected_root:
        raise ValueError("campaign root must match the Phase 7 V3 artifact contract")
    if root.exists():
        raise FileExistsError(f"campaign root already exists: {root}")
    preflight = validate_phase7_v3_inputs(config)
    root.mkdir(parents=False, exist_ok=False)
    private_root = root / "private"
    private_root.mkdir(mode=0o700)
    manifest_path = root / "run_manifest.json"
    terminal_path = root / "campaign_terminal.json"
    failure_path = root / "launcher_failure.json"
    manifest = build_run_manifest(
        config=config,
        preflight=preflight,
        command=command,
        campaign_root=root,
    )
    _write_exclusive_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        result = dict(run_phase7(config, smoke=False))
    except BaseException as error:
        elapsed = time.monotonic() - started
        if not isinstance(error, Exception):
            raise
        failure = _with_hash(
            {
                "schema": FAILURE_SCHEMA,
                "passed": False,
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "elapsed_seconds": elapsed,
                "config_hash": config.hash,
                "run_manifest_artifact_hash": manifest["artifact_hash"],
                "nonclaims": tuple(config.payload["nonclaims"]),
            }
        )
        _write_exclusive_json(failure_path, failure)
        terminal = _with_hash(
            {
                "schema": TERMINAL_SCHEMA,
                "passed": False,
                "classification": "launcher_or_uncaught_infrastructure_failure",
                "elapsed_seconds": elapsed,
                "run_manifest_artifact_hash": manifest["artifact_hash"],
                "controller_terminal_artifact_hash": None,
                "launcher_failure_artifact_hash": failure["artifact_hash"],
                "result_file_sha256": None,
                "private_samples_file_sha256": None,
                "implementation_sources_unchanged": (
                    _verify_source_inventory(
                        manifest["implementation_source_inventory"]
                    )
                ),
                "nonclaims": tuple(config.payload["nonclaims"]),
            }
        )
        _write_exclusive_json(terminal_path, terminal)
        return terminal
    elapsed = time.monotonic() - started
    result_path = config.artifact_path("public_result")
    private_path = config.artifact_path("private_retained_samples")
    if not result_path.is_file():
        raise RuntimeError("controller returned without a terminal result artifact")
    sources_unchanged = _verify_source_inventory(
        manifest["implementation_source_inventory"]
    )
    controller_passed = result.get("passed") is True
    terminal = _with_hash(
        {
            "schema": TERMINAL_SCHEMA,
            "passed": bool(controller_passed and sources_unchanged),
            "classification": (
                "strict_sampling_pass"
                if controller_passed and sources_unchanged
                else "implementation_source_drift_continuation_veto"
                if controller_passed
                else "sampling_candidate_rejected_or_continuation_veto"
            ),
            "elapsed_seconds": elapsed,
            "run_manifest_artifact_hash": manifest["artifact_hash"],
            "controller_terminal_artifact_hash": result.get("artifact_hash"),
            "launcher_failure_artifact_hash": None,
            "result_file_sha256": _file_sha256(result_path),
            "private_samples_file_sha256": (
                _file_sha256(private_path) if private_path.is_file() else None
            ),
            "implementation_sources_unchanged": sources_unchanged,
            "nonclaims": tuple(config.payload["nonclaims"]),
        }
    )
    _write_exclusive_json(terminal_path, terminal)
    return terminal


def _loaded_repository_source_inventory() -> Mapping[str, Mapping[str, Any]]:
    paths: set[Path] = {ROOT / relative for relative in REQUIRED_IMPLEMENTATION_SOURCES}
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value is None:
            continue
        path = Path(value).resolve()
        if path.suffix == ".py" and path.is_relative_to(ROOT):
            paths.add(path)
    inventory = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"required implementation source is missing: {path}")
        raw = path.read_bytes()
        inventory[str(path.relative_to(ROOT))] = {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
    return inventory


def _verify_source_inventory(
    inventory: Mapping[str, Mapping[str, Any]],
) -> bool:
    for relative, reference in inventory.items():
        path = ROOT / relative
        if not path.is_file() or _file_sha256(path) != reference.get(
            "file_sha256"
        ) or path.stat().st_size != int(reference.get("byte_count", -1)):
            return False
    return True


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite campaign artifact: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _with_hash(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(payload)
    result["artifact_hash"] = "sha256:" + stable_config_hash(result)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _invocation() -> tuple[str, ...]:
    return (
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    terminal = run_campaign(
        config_path=args.config,
        campaign_root=args.campaign_root,
        command=_invocation(),
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if terminal.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
