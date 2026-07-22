#!/usr/bin/env python3
"""Run the P5 structural same-target plain-HMC comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_plain_hmc as base


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
CELL_ID = "STR-UKF"
IDENTITY_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p5/STR-UKF/r1b-identity/cpu-attempt-02"
)
EXPECTED_TYPED_SIGNATURE = (
    "e8d78a8ee12245fee2e6c4c739d9dc03d672e8dd9a96bfbd492b426a72e1c665"
)
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p5-r2-structural-plain-hmc-subplan-2026-07-16.md"
)
STEP_SIZES = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
NUM_LEAPFROG_STEPS = 8
PROBE_RESULTS = 128
PROBE_BURNIN = 64
PROBE_SEED = (20260716, 16000)
WARMUP_SEED = (20260716, 16101)
RETAINED_SEED = (20260716, 16201)
INITIAL_OFFSETS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.10, -0.10, 0.08, -0.08, 0.06),
    (-0.10, 0.10, -0.08, 0.08, -0.06),
    (0.16, 0.08, -0.12, -0.10, 0.12),
)
NONCLAIMS = (
    "same-target structural UKF plain-HMC comparator only",
    "probe ESS nominates a kernel but does not establish convergence",
    "acceptance is explanatory only",
    "no NeuTra, filter exactness, truth recovery, calibration, robustness, or readiness claim",
)


def _audit_points(tf: Any, structural_truth_source: Any) -> Any:
    with tf.device("/CPU:0"):
        truth = structural_truth_source()
        eye = 0.5 * tf.eye(5, dtype=tf.float64)
        tails = tf.constant(
            [[1.5, -1.0, 1.2, -1.4, 0.8], [-1.3, 1.4, -1.1, 1.0, -1.5]],
            tf.float64,
        )
        return tf.concat(
            [
                truth[None, :],
                tf.zeros([1, 5], tf.float64),
                truth[None, :] + eye,
                truth[None, :] - eye,
                tails,
            ],
            axis=0,
        )


def run_campaign(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"structural HMC output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    import tensorflow_probability as tfp

    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
    )
    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        SeparateCampaignArchive,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
        run_campaign_plain_hmc,
    )
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig,
        SequentialNeuTraHMCConfig,
        run_sequential_neutra_hmc,
    )
    from bayesfilter.runtime import atomic_write_json
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_PARAMETER_NAMES,
        StructuralUKFLikelihoodRecomposer,
        generate_frozen_structural_dataset_tf,
        make_structural_ukf_neutra_adapter,
        structural_source_probit_jacobian_value_score,
        structural_source_uniform_prior_value_score,
        structural_truth_source,
    )

    source_reference = base._verify_source_root(IDENTITY_ROOT)
    expected_identity = base._read_mapping(IDENTITY_ROOT / "target_identity.json")
    registry = base._read_mapping(IDENTITY_ROOT / "repaired_registry.json")
    registry_hash = base._file_sha256(IDENTITY_ROOT / "repaired_registry.json")
    _states, observations = generate_frozen_structural_dataset_tf()
    adapter = make_structural_ukf_neutra_adapter(observations=observations)
    audit_points = _audit_points(tf, structural_truth_source)
    recomposer = StructuralUKFLikelihoodRecomposer(adapter)
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=audit_points,
        prior_value_score_fn=structural_source_uniform_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=structural_source_probit_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    identity = issue_typed_neutra_target_identity(
        program_id=PROGRAM_ID,
        scope_kind="model_cell",
        scope_id=CELL_ID,
        adapter=adapter,
        recomposition=recomposition,
        registry_row=registry,
        registry_artifact_sha256=registry_hash,
    )
    require_typed_neutra_target(identity, adapter=adapter)
    if base._json_ready(identity.payload()) != expected_identity:
        raise RuntimeError("structural typed identity payload drifted before HMC")
    if identity.target_signature != EXPECTED_TYPED_SIGNATURE:
        raise RuntimeError("structural typed target signature mismatch")

    initial_state = audit_points[0][None, :] + tf.constant(INITIAL_OFFSETS, tf.float64)
    thresholds = RankNormalizedHMCThresholds(
        rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
    )
    probe_rows = []
    for grid_index, step_size in enumerate(STEP_SIZES):
        probe = run_campaign_plain_hmc(
            identity=identity,
            adapter=adapter,
            initial_state=initial_state,
            config=BatchedHMCConfig(
                num_results=PROBE_RESULTS,
                num_burnin_steps=PROBE_BURNIN,
                step_size=step_size,
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                seed=(PROBE_SEED[0], PROBE_SEED[1] + grid_index),
            ),
        )
        diagnostics = rank_normalized_hmc_diagnostics(
            probe["samples"],
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
                "schema": "bayesfilter.multimodel_neutra_p5_structural_hmc_probe_progress.v1",
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
            "schema": "bayesfilter.multimodel_neutra_p5_structural_hmc_selection.v1",
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
                    "schema": "bayesfilter.multimodel_neutra_p5_structural_hmc_progress.v1",
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
            adapter=adapter,
            initial_state=initial_state,
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
        "schema": "bayesfilter.multimodel_neutra_p5_structural_plain_hmc.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "completed": True,
        "passed": passed,
        "decision": "ADMIT_STR_UKF_PLAIN_HMC_COMPARATOR" if passed else "BLOCK_STR_UKF_SOURCE_GEOMETRY",
        "terminal_state": "COMPARATOR_ADMITTED" if passed else "COMPARATOR_BLOCKED",
        "source_identity": source_reference,
        "target_identity": identity.payload(),
        "initial_state": base._json_ready(initial_state),
        "probe_rows": probe_rows,
        "selected_probe": selected,
        "sequential_run": public_sequential,
        "posterior_summary": posterior,
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": NONCLAIMS,
    }
    base._write_new_json(output_root / "result.json", result)
    manifest = {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_plain_hmc_manifest.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip(),
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_executable": sys.executable,
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "random_seeds": {
            "probe_root": PROBE_SEED,
            "warmup_root": WARMUP_SEED,
            "retained_root": RETAINED_SEED,
        },
        "target_signature": identity.target_signature,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": time.monotonic() - started,
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }
    base._write_new_json(output_root / "run_manifest.json", manifest)
    hashes = {
        str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    base._write_new_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p5_structural_plain_hmc_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_campaign(args.output_root)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "passed": result["passed"],
                "selected_step_size": None if result["selected_probe"] is None else result["selected_probe"]["step_size"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
