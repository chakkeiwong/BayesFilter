#!/usr/bin/env python3
"""Run P5 STR-UKF plain HMC in an admitted target-bound affine chart."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_plain_hmc as base
from docs.benchmarks import run_multimodel_neutra_p5_structural_plain_hmc as structural
from docs.benchmarks.run_multimodel_neutra_p5_structural_affine_geometry import (
    PLAN_PATH,
    StructuralAffineTargetAdapter,
    _affine_checks,
    _rebuild_identity,
)


CELL_ID = "STR-UKF"
STEP_SIZES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
NUM_LEAPFROG_STEPS = 8
PROBE_RESULTS = 128
PROBE_BURNIN = 64
PROBE_SEED = (20260716, 18100)
WARMUP_SEED = (20260716, 18201)
RETAINED_SEED = (20260716, 18301)
NONCLAIMS = (
    "same-target STR-UKF affine plain-HMC comparator repair only",
    "mode/Hessian geometry and probe diagnostics are tuning evidence only",
    "acceptance is explanatory only",
    "no NeuTra quality, structural UKF exactness, calibration, robustness, superiority, or readiness claim",
)


def run_campaign(*, geometry_root: Path, output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"structural affine HMC root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime import atomic_write_json
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    import tensorflow_probability as tfp

    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
    )
    from bayesfilter.inference.neutra_campaign import CampaignCellLedger, SeparateCampaignArchive
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig,
        SequentialNeuTraHMCConfig,
        run_batched_hmc,
        run_sequential_neutra_hmc,
    )
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_PARAMETER_NAMES,
    )

    geometry, geometry_reference = _load_geometry(geometry_root)
    adapter, identity, identity_reference = _rebuild_identity(tf)
    if geometry["target_identity"]["target_signature"] != identity.target_signature:
        raise base.P4PlainHMCError("structural geometry target signature drift")
    center = tf.constant(geometry["final_geometry"]["center"], tf.float64)
    factor = tf.constant(
        geometry["final_geometry"]["cholesky_factor"], tf.float64
    )
    affine = StructuralAffineTargetAdapter(
        base_adapter=adapter,
        center=center,
        factor=factor,
        target_signature=identity.target_signature,
        geometry_sha256=geometry_reference["result_sha256"],
    )
    initial_theta = center[None, :] + tf.constant(structural.INITIAL_OFFSETS, tf.float64)
    initial_z = affine.inverse_batch(initial_theta)
    checks = _affine_checks(tf, affine, initial_z)
    if checks["passed"] is not True:
        raise base.P4PlainHMCError("structural affine wrapper replay failed")

    thresholds = RankNormalizedHMCThresholds(
        rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
    )
    probe_rows = []
    for grid_index, step_size in enumerate(STEP_SIZES):
        probe = run_batched_hmc(
            adapter=affine,
            initial_state=initial_z,
            config=BatchedHMCConfig(
                num_results=PROBE_RESULTS,
                num_burnin_steps=PROBE_BURNIN,
                step_size=step_size,
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                seed=(PROBE_SEED[0], PROBE_SEED[1] + grid_index),
            ),
        )
        source_samples = affine.forward(probe["samples"])
        diagnostics = rank_normalized_hmc_diagnostics(
            source_samples,
            parameter_names=STRUCTURAL_PARAMETER_NAMES,
            thresholds=thresholds,
        )
        health = probe["diagnostics"]
        eligible = bool(
            health["health_passed"] is True
            and diagnostics["input_all_finite"] is True
            and diagnostics["diagnostics_all_finite"] is True
        )
        probe_rows.append(
            {
                "grid_index": grid_index,
                "step_size": step_size,
                "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
                "trajectory_length": step_size * NUM_LEAPFROG_STEPS,
                "seed": (PROBE_SEED[0], PROBE_SEED[1] + grid_index),
                "eligible": eligible,
                "nomination_metric": "minimum_rank_normalized_bulk_ess",
                "minimum_bulk_ess": diagnostics.get("min_bulk_ess"),
                "minimum_tail_ess": diagnostics.get("min_tail_ess"),
                "maximum_modern_rhat": diagnostics.get("max_rhat"),
                "acceptance_rate": health["acceptance_rate"],
                "health": health,
                "short_chain_diagnostics": diagnostics,
            }
        )
        atomic_write_json(
            output_root / "probe_progress.json",
            {
                "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_hmc_probe_progress.v1",
                "completed_probe_count": len(probe_rows),
                "total_probe_count": len(STEP_SIZES),
                "probe_rows": probe_rows,
                "scientific_role": "checkpoint_only_not_comparator_admission",
            },
        )
    selected = base._select_probe(probe_rows)
    atomic_write_json(
        output_root / "tuning_selection.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_hmc_selection.v1",
            "probe_rows": probe_rows,
            "selected_probe": selected,
            "selection_rule": "health_valid_then_maximum_minimum_rank_normalized_bulk_ess_grid_order_tie_break",
            "acceptance_role": "explanatory_only",
        },
    )

    sequential = None
    if selected is not None:
        archive = SeparateCampaignArchive(
            output_root=output_root / "samples", identity=identity, adapter=adapter
        )

        def archive_with_progress(**kwargs: Any) -> Mapping[str, Any]:
            payload = archive(**kwargs)
            atomic_write_json(
                output_root / "sequential_progress.json",
                {
                    "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_hmc_progress.v1",
                    "latest_stage": kwargs["stage"],
                    "latest_chunk_index": kwargs["chunk_index"],
                    "latest_seed": kwargs["seed"],
                    "latest_cumulative": kwargs["cumulative"],
                    "latest_archive": payload,
                    "scientific_role": "checkpoint_only_not_comparator_admission",
                },
            )
            return payload

        sequential = run_sequential_neutra_hmc(
            adapter=affine,
            initial_state=initial_z,
            model_transform=affine.forward,
            parameter_names=STRUCTURAL_PARAMETER_NAMES,
            config=SequentialNeuTraHMCConfig(
                step_size=float(selected["step_size"]),
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                warmup_seed=WARMUP_SEED,
                retained_seed=RETAINED_SEED,
                warmup_chunk_results=1000,
                warmup_min_results=2000,
                warmup_check_window_results=1000,
                warmup_max_results=10000,
                warmup_rhat_max=1.05,
                retained_chunk_results=2000,
                retained_min_results=4000,
                retained_max_results=10000,
                retained_rhat_max=1.01,
            ),
            retained_diagnostic_fn=lambda draws: rank_normalized_hmc_diagnostics(
                draws,
                parameter_names=STRUCTURAL_PARAMETER_NAMES,
                thresholds=thresholds,
            ),
            archive_callback=archive_with_progress,
        )

    passed = bool(sequential is not None and sequential["passed"] is True)
    public_sequential = (
        None
        if sequential is None
        else {key: value for key, value in sequential.items() if not key.startswith("private_")}
    )
    posterior = (
        base._posterior_summary(
            tf, sequential["private_retained_raw"], STRUCTURAL_PARAMETER_NAMES
        )
        if sequential is not None and sequential["retained_results_per_chain"] > 0
        else None
    )
    ledger = CampaignCellLedger(
        {
            "cells": [
                {
                    "cell_id": CELL_ID,
                    "state": "POSTERIOR_IDENTITY_ADMITTED",
                    "target_signature": identity.mathematical_target_signature,
                }
            ]
        },
        required_candidate_families=("plain_dense_iaf",),
        event_path=output_root / "cell_events.jsonl",
    )
    ledger.transition(
        cell_id=CELL_ID,
        new_state="COMPARATOR_ADMITTED" if passed else "COMPARATOR_BLOCKED",
        evidence_path=str(output_root / "result.json"),
        target_identity=identity if passed else None,
    )
    base._write_new_json(output_root / "cell_ledger.json", ledger.payload())
    result = {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_hmc.v1",
        "program_id": structural.PROGRAM_ID,
        "cell_id": CELL_ID,
        "completed": True,
        "passed": passed,
        "decision": (
            "ADMIT_STR_UKF_AFFINE_PLAIN_HMC_COMPARATOR"
            if passed
            else "BLOCK_STR_UKF_AFFINE_PLAIN_HMC_COMPARATOR"
        ),
        "terminal_state": "COMPARATOR_ADMITTED" if passed else "COMPARATOR_BLOCKED",
        "identity_reference": identity_reference,
        "geometry_reference": geometry_reference,
        "target_identity": identity.payload(),
        "affine_adapter_signature": affine.adapter_signature(),
        "affine_checks": checks,
        "initial_theta": base._json_ready(initial_theta),
        "initial_z": base._json_ready(initial_z),
        "probe_rows": probe_rows,
        "selected_probe": selected,
        "sequential_run": public_sequential,
        "posterior_summary": posterior,
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": NONCLAIMS,
    }
    base._write_new_json(output_root / "result.json", result)
    base._write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            geometry_root=geometry_root,
            output_root=output_root,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            target_signature=identity.target_signature,
            wall_time=time.monotonic() - started,
        ),
    )
    hashes = {
        str(path.relative_to(output_root)): base._file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    base._write_new_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_hmc_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result


def _load_geometry(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    result_path = root / "result.json"
    hashes_path = root / "artifact_hashes.json"
    declared = base._read_mapping(hashes_path)["artifacts"]
    for relative_path, expected in declared.items():
        if base._file_sha256(root / relative_path) != expected:
            raise base.P4PlainHMCError(f"structural geometry hash mismatch: {relative_path}")
    result = base._read_mapping(result_path)
    if not (
        result.get("passed") is True
        and result.get("raw_precision_spd") is True
        and result.get("terminal_hessian_stable") is True
        and result.get("score_gate_passed") is True
        and result.get("affine_checks", {}).get("passed") is True
        and result.get("final_geometry") is not None
    ):
        raise base.P4PlainHMCError("structural affine geometry was not admitted")
    return result, {
        "root": str(root),
        "result_sha256": base._file_sha256(result_path),
        "artifact_hashes_sha256": base._file_sha256(hashes_path),
        "role": "target_bound_posterior_mode_hessian_hmc_tuning_geometry",
    }


def _run_manifest(
    *,
    geometry_root: Path,
    output_root: Path,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    target_signature: str,
    wall_time: float,
) -> Mapping[str, Any]:
    git_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_affine_hmc_manifest.v1",
        "program_id": structural.PROGRAM_ID,
        "cell_id": CELL_ID,
        "git_commit": git_commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p5_structural_affine_hmc.py "
            f"--geometry-root {geometry_root} --output-root {output_root}"
        ),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_executable": sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "data_version": "chapter18b-structural-T100-seed-20260716-15001",
        "random_seeds": {
            "probe_root": PROBE_SEED,
            "warmup_root": WARMUP_SEED,
            "retained_root": RETAINED_SEED,
        },
        "target_signature": target_signature,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(wall_time),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_campaign(geometry_root=args.geometry_root, output_root=args.output_root)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "passed": result["passed"],
                "decision": result["decision"],
                "selected_step_size": (
                    None
                    if result["selected_probe"] is None
                    else result["selected_probe"]["step_size"]
                ),
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

