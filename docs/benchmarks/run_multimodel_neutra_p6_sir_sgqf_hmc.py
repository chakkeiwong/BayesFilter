#!/usr/bin/env python3
"""Run the P6 SIR-SGQF same-target Laplace-affine HMC comparator."""

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
from typing import Any, Mapping, Sequence

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_plain_hmc as base
from docs.benchmarks.run_multimodel_neutra_p6_sir_sgqf_geometry import (
    AffineMassTargetAdapter,
)


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
CELL_ID = "SIR-SGQF"
IDENTITY_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p6/SIR-SGQF/r1b-identity/gpu-attempt-02"
)
IDENTITY_RESULT_SHA256 = "5cca9efae6147dbdcbd5ad12d0371451b58b6d26cc879ad1c267c0f40d100ea2"
TYPED_SIGNATURE = "0e7921dbd1a2c9a943674b16fd10ccd8b68e1c889e9ae8269a06e0359a750fbc"
GEOMETRY_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p6/SIR-SGQF/laplace-geometry/attempt-02"
)
GEOMETRY_RESULT_SHA256 = "cbe82fe175991c549ed1c7c309a03a719be372e040ea755e358589deeb2c6d67"
PLAN_FILE = (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p6-r2-sir-sgqf-comparator-subplan-2026-07-16.md"
)
STEP_SIZES = (0.025, 0.05, 0.10, 0.20, 0.30, 0.40)
NUM_LEAPFROG_STEPS = 8
PROBE_RESULTS = 128
PROBE_BURNIN = 64
PROBE_SEED = (20260716, 16300)
PROBE_SOURCE_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p6/SIR-SGQF/plain-hmc-affine/attempt-01"
)
PROBE_SOURCE_SHA256 = "76e204264d38a51079be8866a39b01e038ffc86030666371b748b89bd3b0a5be"
TUNING_VERIFICATION_RESULTS = 1000
TUNING_VERIFICATION_BURNIN = 1000
TUNING_VERIFICATION_RHAT_MAX = 1.01
TUNING_VERIFICATION_SEED = (20260716, 26401)
WARMUP_SEED = (20260716, 26501)
RETAINED_SEED = (20260716, 26601)
INITIAL_Z = (
    (0.0, 0.0, 0.0),
    (0.10, -0.10, 0.08),
    (-0.10, 0.10, -0.08),
    (0.16, 0.08, -0.12),
)
NONCLAIMS = (
    "same-target SIR-SGQF level-2 plain-HMC comparator only",
    "Laplace geometry and short probes are tuning evidence only",
    "acceptance is explanatory only",
    "no NeuTra, SGQF exactness, superiority, calibration, forecasting, robustness, or readiness claim",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _verify_root(root: Path, result_sha256: str) -> Mapping[str, Any]:
    if base._file_sha256(root / "result.json") != result_sha256:
        raise RuntimeError(f"result hash mismatch: {root}")
    declared = base._read_mapping(root / "artifact_hashes.json")["artifacts"]
    for relative_path, expected in declared.items():
        if base._file_sha256(root / relative_path) != expected:
            raise RuntimeError(f"artifact hash mismatch: {root / relative_path}")
    return {
        "root": str(root),
        "result_sha256": result_sha256,
        "artifact_hashes_sha256": base._file_sha256(root / "artifact_hashes.json"),
    }


def _ordered_probe_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Order healthy short probes for independent tuning verification."""

    eligible = [row for row in rows if row.get("eligible") is True]
    return tuple(
        dict(row)
        for row in sorted(
            eligible,
            key=lambda row: (
                float(row["maximum_modern_rhat"]),
                -float(row["minimum_bulk_ess"]),
                int(row["grid_index"]),
            ),
        )
    )


def _tuning_verification_admitted(
    *, health: Mapping[str, Any], modern_rhat: Mapping[str, Any]
) -> bool:
    return bool(
        health.get("health_passed") is True
        and modern_rhat.get("input_all_finite") is True
        and modern_rhat.get("diagnostics_all_finite") is True
        and modern_rhat.get("passed") is True
    )


def _archive_tuning_verification(
    *,
    tf: Any,
    atomic_write_json: Any,
    output_root: Path,
    candidate: Mapping[str, Any],
    latent_samples: Any,
    model_samples: Any,
    seed: tuple[int, int],
    target_signature: str,
) -> Mapping[str, Any]:
    destination = output_root / f"candidate-{int(candidate['grid_index']):04d}"
    if destination.exists():
        raise FileExistsError(f"tuning archive destination exists: {destination}")
    destination.mkdir(parents=True)
    latent = tf.convert_to_tensor(latent_samples, tf.float64)
    model = tf.convert_to_tensor(model_samples, tf.float64)
    if latent.shape != model.shape or latent.shape.rank != 3:
        raise ValueError("tuning archives must share [draw, chain, parameter] shape")
    latent_path = destination / "latent.tensor"
    model_path = destination / "model.tensor"
    tf.io.write_file(str(latent_path), tf.io.serialize_tensor(latent))
    tf.io.write_file(str(model_path), tf.io.serialize_tensor(model))
    metadata = {
        "schema": "bayesfilter.multimodel_neutra_p6_tuning_verification_archive.v1",
        "grid_index": int(candidate["grid_index"]),
        "step_size": float(candidate["step_size"]),
        "seed": seed,
        "target_signature": target_signature,
        "sample_shape": tuple(int(item) for item in latent.shape),
        "latent_path": str(latent_path),
        "model_path": str(model_path),
        "excluded_from_posterior": True,
        "scientific_role": "fixed_kernel_tuning_admission_only",
    }
    atomic_write_json(destination / "metadata.json", metadata)
    return metadata


def run_campaign(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"HMC output root must be fresh: {output_root}")
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
        rank_normalized_split_rhat_summary,
    )
    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        SeparateCampaignArchive,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig,
        SequentialNeuTraHMCConfig,
        run_batched_hmc,
        run_sequential_neutra_hmc,
    )
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        SIR_PARAMETER_NAMES,
        SIRSGQFLikelihoodRecomposer,
        generate_frozen_sir_dataset_tf,
        make_sir_sgqf_neutra_adapter,
        sir_identity_chart_jacobian_value_score,
        sir_prior_value_score,
    )

    identity_reference = _verify_root(IDENTITY_ROOT, IDENTITY_RESULT_SHA256)
    geometry_reference = _verify_root(GEOMETRY_ROOT, GEOMETRY_RESULT_SHA256)
    geometry = base._read_mapping(GEOMETRY_ROOT / "result.json")
    if geometry.get("passed") is not True or geometry.get("geometry") is None:
        raise RuntimeError("SIR-SGQF geometry is not admitted")
    expected_identity = base._read_mapping(IDENTITY_ROOT / "target_identity.json")
    registry = base._read_mapping(IDENTITY_ROOT / "repaired_registry.json")
    registry_hash = base._file_sha256(IDENTITY_ROOT / "repaired_registry.json")
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    adapter = make_sir_sgqf_neutra_adapter(observations=observations)
    audit_points = tf.concat(
        [
            tf.zeros([1, 3], tf.float64),
            0.5 * tf.eye(3, dtype=tf.float64),
            -0.5 * tf.eye(3, dtype=tf.float64),
            tf.eye(3, dtype=tf.float64),
            -tf.eye(3, dtype=tf.float64),
        ],
        axis=0,
    )
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=audit_points,
        prior_value_score_fn=sir_prior_value_score,
        likelihood_value_score_fn=SIRSGQFLikelihoodRecomposer(adapter).__call__,
        jacobian_value_score_fn=sir_identity_chart_jacobian_value_score,
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
        raise RuntimeError("SIR-SGQF identity drift before HMC")
    if identity.target_signature != TYPED_SIGNATURE:
        raise RuntimeError("SIR-SGQF typed signature mismatch")
    if geometry["target_identity"]["target_signature"] != identity.target_signature:
        raise RuntimeError("geometry target identity drift")

    center = tf.constant(geometry["geometry"]["center"], tf.float64)
    factor = tf.constant(geometry["geometry"]["cholesky_factor"], tf.float64)
    affine = AffineMassTargetAdapter(
        base_adapter=adapter,
        center=center,
        factor=factor,
        target_signature=identity.target_signature,
        mass_artifact_sha256=GEOMETRY_RESULT_SHA256,
    )
    initial_z = tf.constant(INITIAL_Z, tf.float64)
    initial_theta = affine.forward(initial_z)
    tf.debugging.assert_near(
        affine.inverse_batch(initial_theta), initial_z, atol=1.0e-10, rtol=1.0e-10
    )
    thresholds = RankNormalizedHMCThresholds(
        rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
    )
    probe_source_path = PROBE_SOURCE_ROOT / "tuning_selection.json"
    if base._file_sha256(probe_source_path) != PROBE_SOURCE_SHA256:
        raise RuntimeError("attempt-01 probe source hash mismatch")
    probe_source = base._read_mapping(probe_source_path)
    probe_rows = list(probe_source["probe_rows"])
    if tuple(int(row["grid_index"]) for row in probe_rows) != tuple(
        range(len(STEP_SIZES))
    ):
        raise RuntimeError("attempt-01 probe grid index drift")
    if tuple(float(row["step_size"]) for row in probe_rows) != STEP_SIZES:
        raise RuntimeError("attempt-01 probe step grid drift")
    probe_source_reference = {
        "path": str(probe_source_path),
        "sha256": PROBE_SOURCE_SHA256,
        "reuse_scope": "valid_short_probe_rows_only",
        "invalidated_attempt01_selection_reused": False,
        "attempt01_samples_reused": False,
    }
    atomic_write_json(output_root / "probe_reuse.json", probe_source_reference)

    candidate_order = _ordered_probe_candidates(probe_rows)
    verification_rows = []
    selected = None
    for candidate in candidate_order:
        grid_index = int(candidate["grid_index"])
        seed = (
            TUNING_VERIFICATION_SEED[0],
            TUNING_VERIFICATION_SEED[1] + grid_index,
        )
        verification = run_batched_hmc(
            adapter=affine,
            initial_state=initial_z,
            config=BatchedHMCConfig(
                num_results=TUNING_VERIFICATION_RESULTS,
                num_burnin_steps=TUNING_VERIFICATION_BURNIN,
                step_size=float(candidate["step_size"]),
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                seed=seed,
            ),
        )
        model_samples = affine.forward(verification["samples"])
        modern_rhat = rank_normalized_split_rhat_summary(
            model_samples, rhat_max=TUNING_VERIFICATION_RHAT_MAX
        )
        health = verification["diagnostics"]
        admitted = _tuning_verification_admitted(
            health=health, modern_rhat=modern_rhat
        )
        archive = _archive_tuning_verification(
            tf=tf,
            atomic_write_json=atomic_write_json,
            output_root=output_root / "tuning-verification",
            candidate=candidate,
            latent_samples=verification["samples"],
            model_samples=model_samples,
            seed=seed,
            target_signature=identity.target_signature,
        )
        row = {
            "grid_index": grid_index,
            "step_size": float(candidate["step_size"]),
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "seed": seed,
            "probe_nomination_rank": len(verification_rows),
            "probe_minimum_bulk_ess": candidate["minimum_bulk_ess"],
            "health": health,
            "modern_rhat": modern_rhat,
            "archive": archive,
            "admitted": admitted,
            "acceptance_role": "explanatory_only",
            "scientific_role": "fixed_kernel_tuning_admission_only",
        }
        verification_rows.append(row)
        atomic_write_json(
            output_root / "tuning_verification_progress.json",
            {
                "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_hmc_tuning_verification_progress.v1",
                "completed_candidate_count": len(verification_rows),
                "total_candidate_count": len(candidate_order),
                "verification_rows": verification_rows,
                "scientific_role": "checkpoint_only_not_comparator_admission",
            },
        )
        if admitted:
            selected = {**dict(candidate), "tuning_verification": row}
            break
    atomic_write_json(
        output_root / "tuning_selection.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_hmc_selection.v2",
            "probe_source_reference": probe_source_reference,
            "probe_rows": probe_rows,
            "candidate_order": candidate_order,
            "tuning_verification_rows": verification_rows,
            "selected_probe": selected,
            "selection_rule": "short_probe_lowest_modern_rhat_then_bulk_ess_order_then_disjoint_1000_burnin_1000_draw_modern_rhat_at_most_1.01_first_pass",
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
                    "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_hmc_progress.v1",
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
            parameter_names=SIR_PARAMETER_NAMES,
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
                parameter_names=SIR_PARAMETER_NAMES,
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
            tf, sequential["private_retained_raw"], SIR_PARAMETER_NAMES
        )
        if sequential is not None and sequential["retained_results_per_chain"] > 0
        else None
    )
    result = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_hmc.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "completed": True,
        "passed": passed,
        "decision": "ADMIT_SIR_SGQF_PLAIN_HMC_COMPARATOR" if passed else "BLOCK_SIR_SGQF_PLAIN_HMC_COMPARATOR",
        "terminal_state": "COMPARATOR_ADMITTED" if passed else "COMPARATOR_BLOCKED",
        "identity_reference": identity_reference,
        "geometry_reference": geometry_reference,
        "probe_source_reference": probe_source_reference,
        "target_identity": identity.payload(),
        "affine_adapter_signature": affine.adapter_signature(),
        "initial_theta": base._json_ready(initial_theta),
        "initial_z": base._json_ready(initial_z),
        "probe_rows": probe_rows,
        "tuning_verification_rows": verification_rows,
        "selected_probe": selected,
        "sequential_run": public_sequential,
        "posterior_summary": posterior,
        "elapsed_seconds": time.monotonic() - started,
        "blocker_stage": None if passed else ("TUNING" if selected is None else "COMPARATOR"),
        "nonclaims": NONCLAIMS,
    }
    base._write_new_json(output_root / "result.json", result)
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
    base._write_new_json(
        output_root / "run_manifest.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_hmc_manifest.v1",
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
            "data_version": "parameterized Austria SIR seed 81120 y1:y20",
            "random_seeds": {
                "probe_source_root": PROBE_SEED,
                "tuning_verification_root": TUNING_VERIFICATION_SEED,
                "warmup_root": WARMUP_SEED,
                "retained_root": RETAINED_SEED,
            },
            "target_signature": identity.target_signature,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(output_root),
            "plan_file": PLAN_FILE,
            "result_file": str(output_root / "result.json"),
            "nonclaims": NONCLAIMS,
        },
    )
    hashes = {
        str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    base._write_new_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_hmc_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result


def main() -> None:
    run_campaign(_args().output_root)


if __name__ == "__main__":
    main()
