#!/usr/bin/env python3
"""Run P5 R1B structural posterior recomposition and typed identity admission."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
CELL_ID = "STR-UKF"
PLAN_FILE = (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p5-r1b-structural-identity-subplan-2026-07-16.md"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _audit_points(structural_truth_source: Any) -> tf.Tensor:
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


def _fd(adapter: Any, points: tf.Tensor, step: float) -> tf.Tensor:
    columns = []
    epsilon = tf.constant(step, tf.float64)
    for coordinate in range(5):
        basis = tf.one_hot(coordinate, 5, dtype=tf.float64)[None, :]
        plus = adapter.log_prob(points + epsilon * basis)
        minus = adapter.log_prob(points - epsilon * basis)
        columns.append((plus - minus) / (2.0 * epsilon))
    return tf.stack(columns, axis=1)


def _mutated_contract_signatures(adapter: Any, stable_signature: Any) -> Mapping[str, Any]:
    from bayesfilter.ssm import FilterProgram, ParameterChart, ParameterPrior, SSMDataSignature

    contract = adapter.contract
    baseline = stable_signature(contract)

    changed_data = dataclasses.replace(
        contract,
        problem=dataclasses.replace(
            contract.problem,
            data_signature=SSMDataSignature(
                dataset_id=contract.problem.data_signature.dataset_id,
                observation_shape=contract.problem.data_signature.observation_shape,
                data_hash="sha256:" + "0" * 64,
            ),
        ),
    )
    filter_manifest = dict(contract.filter_program.filter_manifest)
    filter_manifest.update(
        {
            "filter_id": "chapter18b-artificial-k-noise-negative-control",
            "integration_space": "lagged_state_plus_epsilon_plus_eta_k",
            "artificial_k_noise_allowed": True,
            "artificial_k_variance": 0.04,
        }
    )
    filter_manifest["filter_hash"] = "sha256:" + _semantic_hash(
        {key: value for key, value in filter_manifest.items() if key != "filter_hash"}
    )
    artificial_filter = FilterProgram(
        filter_id="chapter18b-artificial-k-noise-negative-control",
        required_model_capabilities=(
            "quadratic_structural_transition",
            "two_dimensional_artificial_innovation",
        ),
        deterministic_target_policy="deterministic",
        approximation_semantics="deterministic_approximation",
        filter_manifest=filter_manifest,
    )
    changed_artificial = dataclasses.replace(contract, filter_program=artificial_filter)

    chart_manifest = dict(contract.chart.transform_manifest)
    chart_manifest["upper"] = tuple(
        float(value) + (0.01 if index == 4 else 0.0)
        for index, value in enumerate(chart_manifest["upper"])
    )
    chart_manifest["transform_hash"] = "sha256:" + _semantic_hash(
        {key: value for key, value in chart_manifest.items() if key != "transform_hash"}
    )
    changed_chart = dataclasses.replace(
        contract,
        chart=ParameterChart(
            parameter_names=contract.chart.parameter_names,
            unconstrained_dim=5,
            constrained_shape=(5,),
            transform_manifest=chart_manifest,
            log_jacobian_convention="included_in_chart",
        ),
    )
    prior_manifest = dict(contract.prior.prior_manifest)
    prior_manifest["parameter_order"] = ("sigma", "rho", "phi", "gamma", "R")
    prior_manifest["prior_hash"] = "sha256:" + _semantic_hash(
        {key: value for key, value in prior_manifest.items() if key != "prior_hash"}
    )
    changed_prior = dataclasses.replace(
        contract,
        prior=ParameterPrior(
            prior_manifest=prior_manifest,
            support_policy="enforced_by_transform",
            log_density_authority="graph_native",
        ),
    )
    rows = {
        "changed_observation_hash": stable_signature(changed_data),
        "artificial_k_noise_route": stable_signature(changed_artificial),
        "changed_R_chart_upper": stable_signature(changed_chart),
        "changed_prior_order": stable_signature(changed_prior),
    }
    return {
        "baseline_mathematical_signature": baseline,
        "mutated_signatures": rows,
        "all_mutations_rejected_by_signature": all(value != baseline for value in rows.values()),
        "artificial_route_innovation_dim": 2,
        "artificial_route_eligible_for_baseline_registry": False,
    }


def main() -> None:
    args = _args()
    if args.output_root.exists():
        raise FileExistsError(f"R1B output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    memory_policy = None
    if args.device == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise RuntimeError("CPU R1B requires CUDA_VISIBLE_DEVICES=-1 before import")
    else:
        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)

    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        NeuTraCampaignError,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_FINAL_OBSERVATION_SHA256,
        STRUCTURAL_FINAL_STATE_SHA256,
        StructuralUKFLikelihoodRecomposer,
        generate_frozen_structural_dataset_tf,
        make_structural_ukf_neutra_adapter,
        structural_source_probit_jacobian_value_score,
        structural_source_uniform_prior_value_score,
        structural_truth_source,
    )

    states, observations = generate_frozen_structural_dataset_tf()
    state_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(states).numpy())).hexdigest()
    observation_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(observations).numpy())).hexdigest()
    if state_hash != STRUCTURAL_FINAL_STATE_SHA256 or observation_hash != STRUCTURAL_FINAL_OBSERVATION_SHA256:
        raise RuntimeError("frozen structural dataset hash mismatch")
    adapter = make_structural_ukf_neutra_adapter(observations=observations)
    points = _audit_points(structural_truth_source)

    @tf.function(input_signature=[tf.TensorSpec([14, 5], tf.float64)], jit_compile=True)
    def compiled(theta: tf.Tensor):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    value, score, status = compiled(points)
    reversed_value, reversed_score, reversed_status = compiled(tf.reverse(points, axis=(0,)))
    permutation_value_gap = tf.reduce_max(tf.abs(value - tf.reverse(reversed_value, axis=(0,))))
    permutation_score_gap = tf.reduce_max(tf.abs(score - tf.reverse(reversed_score, axis=(0,))))
    permutation_status_equal = tf.reduce_all(
        tf.equal(status["status_code"], tf.reverse(reversed_status["status_code"], axis=(0,)))
    )
    fd_fine = _fd(adapter, points, 5.0e-6)
    fd_coarse = _fd(adapter, points, 1.0e-5)
    analytic_fd_gap = tf.reduce_max(tf.abs(score - fd_fine))
    fd_step_gap = tf.reduce_max(tf.abs(fd_fine - fd_coarse))

    recomposer = StructuralUKFLikelihoodRecomposer(adapter)
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=points,
        prior_value_score_fn=structural_source_uniform_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=structural_source_probit_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    mathematical_signature = stable_ssm_target_signature(adapter.contract)
    substitutions = _mutated_contract_signatures(adapter, stable_ssm_target_signature)
    missing_jacobian_value, missing_jacobian_score = recomposer(points)
    prior_value, prior_score = structural_source_uniform_prior_value_score(points)
    missing_jacobian_gap = tf.reduce_max(
        tf.abs(value - (missing_jacobian_value + prior_value))
    )
    duplicated_jacobian_value, duplicated_jacobian_score = (
        structural_source_probit_jacobian_value_score(points)
    )
    duplicated_jacobian_gap = tf.reduce_max(
        tf.abs(
            value
            - (
                missing_jacobian_value
                + prior_value
                + 2.0 * duplicated_jacobian_value
            )
        )
    )
    substitutions.update(
        {
            "missing_jacobian_value_gap": missing_jacobian_gap,
            "duplicated_jacobian_value_gap": duplicated_jacobian_gap,
            "missing_jacobian_rejected": missing_jacobian_gap > 1.0e-6,
            "duplicated_jacobian_rejected": duplicated_jacobian_gap > 1.0e-6,
        }
    )

    primary_pass = bool(
        tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
        and float(permutation_value_gap.numpy()) <= 1.0e-10
        and float(permutation_score_gap.numpy()) <= 1.0e-10
        and bool(permutation_status_equal.numpy())
        and float(analytic_fd_gap.numpy()) <= 3.0e-5
        and float(fd_step_gap.numpy()) <= 1.0e-5
        and recomposition.passed
        and substitutions["all_mutations_rejected_by_signature"]
        and bool(substitutions["missing_jacobian_rejected"].numpy())
        and bool(substitutions["duplicated_jacobian_rejected"].numpy())
    )
    identity_payload = None
    registry_row = None
    if primary_pass:
        registry_row = {
            "schema": "bayesfilter.multimodel_neutra_p5_structural_registry.v1",
            "program_id": PROGRAM_ID,
            "cell_id": CELL_ID,
            "previous_state": "TARGET_BLOCKED",
            "state": "VALUE_SCORE_ADMITTED",
            "target_signature": mathematical_signature,
            "dataset_sha256": observation_hash,
            "repair_evidence": (
                "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
                "p5-structural-target-design-result-2026-07-16.md"
            ),
            "remaining_blockers": [],
        }
        registry_path = args.output_root / "repaired_registry.json"
        _write(registry_path, registry_row)
        identity = issue_typed_neutra_target_identity(
            program_id=PROGRAM_ID,
            scope_kind="model_cell",
            scope_id=CELL_ID,
            adapter=adapter,
            recomposition=recomposition,
            registry_row=registry_row,
            registry_artifact_sha256=_hash(registry_path),
        )
        require_typed_neutra_target(identity, adapter=adapter)
        identity_payload = identity.payload()
        _write(args.output_root / "target_identity.json", identity_payload)
        _write(args.output_root / "recomposition.json", recomposition.payload())
        ledger = CampaignCellLedger(
            {"cells": [registry_row]},
            required_candidate_families=("plain_dense_iaf",),
            event_path=args.output_root / "cell_events.jsonl",
        )
        ledger.transition(
            cell_id=CELL_ID,
            new_state="POSTERIOR_IDENTITY_ADMITTED",
            evidence_path=str(args.output_root / "target_identity.json"),
            target_identity=identity,
        )
        _write(args.output_root / "cell_ledger.json", ledger.payload())

    passed = identity_payload is not None
    result = {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_r1b.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "device_intent": args.device,
        "passed": passed,
        "decision": "ADMIT_STR_UKF_POSTERIOR_IDENTITY" if passed else "BLOCK_STR_UKF_R1B",
        "dataset": {
            "state_sha256": state_hash,
            "observation_sha256": observation_hash,
            "horizon": 100,
            "seed": [20260716, 15001],
        },
        "audit_points": points,
        "value": value,
        "score": score,
        "status": status,
        "permutation_value_gap": permutation_value_gap,
        "permutation_score_gap": permutation_score_gap,
        "permutation_status_equal": permutation_status_equal,
        "analytic_fine_fd_gap": analytic_fd_gap,
        "fine_coarse_fd_step_gap": fd_step_gap,
        "recomposition": recomposition.payload(),
        "substitution_negatives": substitutions,
        "mathematical_target_signature": mathematical_signature,
        "target_identity": identity_payload,
        "registry_row": registry_row,
        "memory_policy": memory_policy,
        "output_devices": [value.device, score.device, status["status_code"].device],
        "jit_compile": True,
        "elapsed_seconds": time.perf_counter() - started,
        "nonclaims": [
            "typed posterior identity and value-score admission only",
            "no HMC, NeuTra, filter exactness, global identifiability, calibration, or readiness claim",
        ],
    }
    _write(args.output_root / "result.json", result)
    manifest = {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_r1b_manifest.v1",
        "program_id": PROGRAM_ID,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "device_intent": args.device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "jit_compile": True,
        "gpu_memory_policy": memory_policy,
        "target_signature": None if identity_payload is None else identity_payload["target_signature"],
        "plan_file": PLAN_FILE,
        "result_file": str(args.output_root / "result.json"),
        "wall_time_seconds": result["elapsed_seconds"],
    }
    _write(args.output_root / "run_manifest.json", manifest)
    hashes = {
        str(path.relative_to(args.output_root)): _hash(path)
        for path in sorted(args.output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write(
        args.output_root / "artifact_hashes.json",
        {"schema": "bayesfilter.multimodel_neutra_p5_structural_r1b_hashes.v1", "artifacts": hashes},
    )


if __name__ == "__main__":
    main()
