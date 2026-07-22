"""Run the P4 PP-UKF affine-mass plain-HMC comparator repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


CELL_ID = "PP-UKF"
SOURCE_ROOT = base.PHASE_ROOT / (
    "PP-UKF/plain-hmc/attempt-02-20260715T133100Z"
)
SOURCE_RESULT_SHA256 = (
    "d0f894257ec3c93a26e2900327637ea17e69d0a50c2ce87417e6c704c9a726f2"
)
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p4-pp-ukf-affine-hmc-repair-subplan-2026-07-15.md"
)
STEP_SIZES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
NUM_LEAPFROG_STEPS = 8
PROBE_RESULTS = 128
PROBE_BURNIN = 64
PROBE_SEED = (20260715, 9000)
WARMUP_SEED = (20260715, 9101)
RETAINED_SEED = (20260715, 9201)
RELATIVE_EIGENVALUE_FLOOR = 1.0e-6
ABSOLUTE_EIGENVALUE_FLOOR = 1.0e-8
NONCLAIMS = (
    "same-target PP-UKF affine-mass plain-HMC comparator repair only",
    "failed identity-mass warm-up is tuning evidence only",
    "probe ESS nominates but does not establish convergence",
    "acceptance is explanatory only",
    "no NeuTra quality, UKF exactness, superiority, calibration, or readiness claim",
)


class AffineMassTargetAdapter:
    """Exact affine coordinate change around a typed batch-native target."""

    def __init__(
        self,
        *,
        base_adapter: Any,
        center: Any,
        factor: Any,
        target_signature: str,
        mass_artifact_sha256: str,
    ) -> None:
        import tensorflow as tf

        self.base_adapter = base_adapter
        self.center = tf.convert_to_tensor(center, tf.float64)
        self.factor = tf.convert_to_tensor(factor, tf.float64)
        if self.center.shape != (6,) or self.factor.shape != (6, 6):
            raise base.P4PlainHMCError("affine mass center/factor shape mismatch")
        sign, log_abs_det = tf.linalg.slogdet(self.factor)
        if float(sign.numpy()) <= 0.0 or not bool(
            tf.math.is_finite(log_abs_det).numpy()
        ):
            raise base.P4PlainHMCError("affine mass factor must have positive determinant")
        self.log_abs_det = log_abs_det
        self.parameter_dim = 6
        self.target_signature = str(target_signature)
        self.mass_artifact_sha256 = str(mass_artifact_sha256)

    def forward(self, z: Any) -> Any:
        import tensorflow as tf

        values = tf.convert_to_tensor(z, tf.float64)
        return self.center + tf.tensordot(values, self.factor, axes=[[-1], [1]])

    def inverse(self, theta: Any) -> Any:
        import tensorflow as tf

        values = tf.convert_to_tensor(theta, tf.float64) - self.center
        return tf.linalg.triangular_solve(
            self.factor, tf.transpose(values), lower=True
        , adjoint=False)

    def inverse_batch(self, theta: Any) -> Any:
        import tensorflow as tf

        values = tf.convert_to_tensor(theta, tf.float64) - self.center
        return tf.transpose(
            tf.linalg.triangular_solve(
                self.factor, tf.transpose(values), lower=True
            )
        )

    def log_prob_and_grad(self, z: Any) -> tuple[Any, Any]:
        import tensorflow as tf

        values = tf.convert_to_tensor(z, tf.float64)
        theta = self.forward(values)
        raw_value, raw_score = self.base_adapter.log_prob_and_grad(theta)
        latent_score = tf.tensordot(raw_score, self.factor, axes=[[-1], [0]])
        return raw_value + self.log_abs_det, latent_score

    def target_status_telemetry(self, z: Any) -> Mapping[str, Any]:
        return self.base_adapter.target_status_telemetry(self.forward(z))

    def adapter_signature(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": "bayesfilter.multimodel_neutra_p4_affine_mass_target.v1",
                    "target_signature": self.target_signature,
                    "mass_artifact_sha256": self.mass_artifact_sha256,
                    "coordinate_program": "theta=center+z@factor.T",
                    "score_program": "score_z=score_theta@factor",
                    "constant_log_abs_det_included": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def run_campaign(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"affine repair output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime import atomic_write_json
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

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
    )
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig,
        SequentialNeuTraHMCConfig,
        run_batched_hmc,
        run_sequential_neutra_hmc,
    )
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PP_PARAMETER_NAMES,
        PredatorPreyUKFLikelihoodRecomposer,
        generate_frozen_predator_prey_dataset_tf,
        make_predator_prey_ukf_neutra_adapter,
        source_six_probit_jacobian_value_score,
        source_uniform_prior_value_score,
    )

    source_reference = _verify_source_attempt()
    source_identity_root = base.IDENTITY_ROOTS[CELL_ID]
    base._verify_source_root(source_identity_root)
    identity_source_result = base._read_mapping(source_identity_root / "result.json")
    expected_identity = base._read_mapping(source_identity_root / "target_identity.json")
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)
    audit_points = tf.constant(identity_source_result["audit_points"], tf.float64)
    recomposer = PredatorPreyUKFLikelihoodRecomposer(adapter)
    registry = base._read_mapping(source_identity_root / "repaired_registry.json")
    registry_hash = base._file_sha256(source_identity_root / "repaired_registry.json")
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=audit_points,
        prior_value_score_fn=source_uniform_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=source_six_probit_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    identity = issue_typed_neutra_target_identity(
        program_id=base.PROGRAM_ID,
        scope_kind="model_cell",
        scope_id=CELL_ID,
        adapter=adapter,
        recomposition=recomposition,
        registry_row=registry,
        registry_artifact_sha256=registry_hash,
    )
    require_typed_neutra_target(identity, adapter=adapter)
    if base._json_ready(identity.payload()) != expected_identity:
        raise base.P4PlainHMCError("PP-UKF identity drift in affine repair")

    mass_payload, center, covariance, factor = _build_mass_artifact(tf)
    mass_path = output_root / "affine_mass.json"
    base._write_new_json(mass_path, mass_payload)
    mass_sha256 = base._file_sha256(mass_path)
    affine = AffineMassTargetAdapter(
        base_adapter=adapter,
        center=center,
        factor=factor,
        target_signature=identity.target_signature,
        mass_artifact_sha256=mass_sha256,
    )
    initial_theta = audit_points[0][None, :] + tf.constant(
        base.INITIAL_OFFSETS, tf.float64
    )
    initial_z = affine.inverse_batch(initial_theta)
    _check_affine_wrapper(tf, affine, initial_z)

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
        model_samples = affine.forward(probe["samples"])
        diagnostic = rank_normalized_hmc_diagnostics(
            model_samples, parameter_names=PP_PARAMETER_NAMES, thresholds=thresholds
        )
        health = probe["diagnostics"]
        eligible = bool(
            health["health_passed"] is True
            and diagnostic["input_all_finite"] is True
            and diagnostic["diagnostics_all_finite"] is True
        )
        probe_rows.append(
            {
                "grid_index": grid_index,
                "step_size": step_size,
                "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
                "trajectory_length": step_size * NUM_LEAPFROG_STEPS,
                "seed": (PROBE_SEED[0], PROBE_SEED[1] + grid_index),
                "eligible": eligible,
                "minimum_bulk_ess": diagnostic.get("min_bulk_ess"),
                "minimum_tail_ess": diagnostic.get("min_tail_ess"),
                "maximum_modern_rhat": diagnostic.get("max_rhat"),
                "acceptance_rate": health["acceptance_rate"],
                "health": health,
                "short_chain_diagnostics": diagnostic,
            }
        )
        atomic_write_json(
            output_root / "probe_progress.json",
            {
                "schema": "bayesfilter.multimodel_neutra_p4_affine_probe_progress.v1",
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
            "schema": "bayesfilter.multimodel_neutra_p4_affine_tuning_selection.v1",
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
                    "schema": "bayesfilter.multimodel_neutra_p4_affine_sequential_progress.v1",
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
            parameter_names=PP_PARAMETER_NAMES,
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
                draws, parameter_names=PP_PARAMETER_NAMES, thresholds=thresholds
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
            tf, sequential["private_retained_raw"], PP_PARAMETER_NAMES
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
        "schema": "bayesfilter.multimodel_neutra_p4_pp_ukf_affine_hmc_repair.v1",
        "program_id": base.PROGRAM_ID,
        "cell_id": CELL_ID,
        "completed": True,
        "passed": passed,
        "decision": (
            "ADMIT_PP_UKF_AFFINE_PLAIN_HMC_COMPARATOR"
            if passed
            else "BLOCK_PP_UKF_AFFINE_PLAIN_HMC_COMPARATOR"
        ),
        "terminal_state": "COMPARATOR_ADMITTED" if passed else "COMPARATOR_BLOCKED",
        "source_attempt": source_reference,
        "target_identity": identity.payload(),
        "affine_mass": {
            "path": str(mass_path),
            "sha256": mass_sha256,
            "adapter_signature": affine.adapter_signature(),
        },
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
            "schema": "bayesfilter.multimodel_neutra_p4_affine_hmc_artifact_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result


def _build_mass_artifact(tf: Any) -> tuple[Mapping[str, Any], Any, Any, Any]:
    metadata_path = SOURCE_ROOT / "samples/warmup/cumulative/metadata.json"
    metadata = base._read_mapping(metadata_path)
    tensor_path = Path(str(metadata["model_path"]))
    if base._file_sha256(tensor_path) != base._file_sha256(
        SOURCE_ROOT / "samples/warmup/cumulative/model.tensor"
    ):
        raise base.P4PlainHMCError("source cumulative warm-up tensor path mismatch")
    draws = tf.io.parse_tensor(tf.io.read_file(str(tensor_path)), out_type=tf.float64)
    draws = tf.ensure_shape(draws, (10000, 4, 6))
    chain_center = tf.reduce_mean(draws, axis=0)
    residual = draws - chain_center[None, :, :]
    denominator = tf.cast(10000 * 4 - 4, tf.float64)
    covariance_raw = tf.einsum("nci,ncj->ij", residual, residual) / denominator
    covariance_raw = 0.5 * (covariance_raw + tf.transpose(covariance_raw))
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance_raw)
    largest = tf.reduce_max(eigenvalues)
    floor = tf.maximum(
        tf.constant(ABSOLUTE_EIGENVALUE_FLOOR, tf.float64),
        tf.constant(RELATIVE_EIGENVALUE_FLOOR, tf.float64) * largest,
    )
    regularized_eigenvalues = tf.maximum(eigenvalues, floor)
    covariance = tf.einsum(
        "ik,k,jk->ij", eigenvectors, regularized_eigenvalues, eigenvectors
    )
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    factor = tf.linalg.cholesky(covariance)
    center = tf.reduce_mean(draws, axis=(0, 1))
    payload = {
        "schema": "bayesfilter.multimodel_neutra_p4_affine_mass.v1",
        "cell_id": CELL_ID,
        "target_signature": base.EXPECTED_TYPED_SIGNATURES[CELL_ID],
        "source_attempt": str(SOURCE_ROOT),
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_metadata_sha256": base._file_sha256(metadata_path),
        "source_tensor_sha256": base._file_sha256(tensor_path),
        "source_draw_shape": tuple(int(item) for item in draws.shape),
        "source_role": "failed_identity_mass_warmup_tuning_only",
        "estimator": "within_chain_centered_pooled_covariance",
        "denominator": 10000 * 4 - 4,
        "center": base._json_ready(center),
        "raw_covariance": base._json_ready(covariance_raw),
        "raw_eigenvalues": base._json_ready(eigenvalues),
        "raw_condition_number": float((eigenvalues[-1] / eigenvalues[0]).numpy()),
        "absolute_eigenvalue_floor": ABSOLUTE_EIGENVALUE_FLOOR,
        "relative_eigenvalue_floor": RELATIVE_EIGENVALUE_FLOOR,
        "realized_eigenvalue_floor": float(floor.numpy()),
        "regularized_eigenvalues": base._json_ready(regularized_eigenvalues),
        "regularized_covariance": base._json_ready(covariance),
        "cholesky_factor": base._json_ready(factor),
        "coordinate_program": "theta=center+z@factor.T",
        "inverse_program": "solve(factor, (theta-center).T).T",
        "score_program": "score_z=score_theta@factor",
        "constant_log_abs_det_included": True,
        "warmup_excluded_from_posterior": True,
    }
    return payload, center, covariance, factor


def _check_affine_wrapper(tf: Any, adapter: AffineMassTargetAdapter, z: Any) -> None:
    theta = adapter.forward(z)
    tf.debugging.assert_near(adapter.inverse_batch(theta), z, atol=1.0e-10, rtol=1.0e-10)
    raw_value, raw_score = adapter.base_adapter.log_prob_and_grad(theta)
    value, score = adapter.log_prob_and_grad(z)
    tf.debugging.assert_near(value, raw_value + adapter.log_abs_det, atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_near(
        score,
        tf.tensordot(raw_score, adapter.factor, axes=[[-1], [0]]),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def _verify_source_attempt() -> Mapping[str, Any]:
    result_path = SOURCE_ROOT / "result.json"
    if base._file_sha256(result_path) != SOURCE_RESULT_SHA256:
        raise base.P4PlainHMCError("identity-mass source result hash mismatch")
    hashes = base._read_mapping(SOURCE_ROOT / "artifact_hashes.json")["artifacts"]
    for relative_path, expected in hashes.items():
        if base._file_sha256(SOURCE_ROOT / relative_path) != expected:
            raise base.P4PlainHMCError(f"source attempt hash mismatch: {relative_path}")
    result = base._read_mapping(result_path)
    sequential = result["sequential_run"]
    if not (
        result["passed"] is False
        and sequential["warmup_cap_hit"] is True
        and sequential["warmup_results_per_chain"] == 10000
        and sequential["retained_results_per_chain"] == 0
        and not sequential["hard_vetoes"]
    ):
        raise base.P4PlainHMCError("source attempt is not the frozen healthy cap failure")
    return {
        "root": str(SOURCE_ROOT),
        "result_sha256": SOURCE_RESULT_SHA256,
        "artifact_hashes_sha256": base._file_sha256(SOURCE_ROOT / "artifact_hashes.json"),
        "role": "identity_mass_warmup_tuning_only",
    }


def _run_manifest(
    *,
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
        "schema": "bayesfilter.multimodel_neutra_p4_affine_hmc_run_manifest.v1",
        "program_id": base.PROGRAM_ID,
        "cell_id": CELL_ID,
        "git_commit": git_commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p4_pp_ukf_affine_hmc_repair.py "
            f"--output-root {output_root}"
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
        "data_version": "zhao_cui_predator_prey_T20 seed 81104",
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
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_campaign(args.output_root)
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
