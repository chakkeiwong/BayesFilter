#!/usr/bin/env python3
"""Run P6 R1B SIR-SGQF recomposition and typed-identity admission."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
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
CELL_ID = "SIR-SGQF"
PLAN_FILE = (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p6-r1b-sir-sgqf-identity-subplan-2026-07-16.md"
)
TARGET_DESIGN_RESULT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p6/SIR-common/target-design/cpu-attempt-04/result.json"
)
TARGET_DESIGN_SHA256 = "5d0d73f302b160b9f1277cd4ab5ef22ad53200f2c156cf6395d1e6a4ba0f9852"
GPU_CANARY = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p6/SIR-common/gpu-canary/attempt-02/gpu_canary.json"
)
GPU_CANARY_SHA256 = "51d61ea606521fe553555792ff771c1810424344bdcae2c300e42344731716b9"
MATHEMATICAL_TARGET_SIGNATURE = (
    "43968c975409021dcabe931081f0d1efaaae431b5b9245929a5786fe566e545d"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--cpu-result", type=Path)
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
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _audit_points() -> tf.Tensor:
    with tf.device("/CPU:0"):
        eye = tf.eye(3, dtype=tf.float64)
        zero = tf.zeros([1, 3], tf.float64)
        return tf.concat([zero, 0.5 * eye, -0.5 * eye, eye, -eye], axis=0)


def _stencil(points: tf.Tensor, step: float) -> tf.Tensor:
    offsets = tf.constant(step, tf.float64) * tf.eye(3, dtype=tf.float64)
    plus = points[:, None, :] + offsets[None, :, :]
    minus = points[:, None, :] - offsets[None, :, :]
    return tf.reshape(tf.concat([plus, minus], axis=1), [-1, 3])


def _fd(adapter: Any, points: tf.Tensor, step: float) -> tf.Tensor:
    values = _stencil(points, step)
    row_count = int(values.shape[0])
    chunks = [
        adapter.log_prob(values[6 * index : 6 * (index + 1)])
        for index in range(row_count // 6)
    ]
    reshaped = tf.reshape(tf.concat(chunks, axis=0), [int(points.shape[0]), 6])
    return (reshaped[:, :3] - reshaped[:, 3:]) / (2.0 * step)


def _gap(left: tf.Tensor, right: tf.Tensor) -> Mapping[str, tf.Tensor]:
    difference = tf.abs(left - right)
    scale = tf.maximum(tf.ones_like(left), tf.maximum(tf.abs(left), tf.abs(right)))
    return {
        "maximum_absolute_gap": tf.reduce_max(difference),
        "maximum_scale_normalized_gap": tf.reduce_max(difference / scale),
    }


def _mutated_contract_signatures(adapter: Any, stable_signature: Any) -> Mapping[str, Any]:
    from bayesfilter.ssm import FilterProgram, ParameterPrior, SSMDataSignature

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
    prior_manifest = dict(contract.prior.prior_manifest)
    prior_manifest["scale"] = (1.0, 1.0, 1.0)
    prior_manifest["prior_hash"] = "sha256:" + _semantic_hash(
        {key: value for key, value in prior_manifest.items() if key != "prior_hash"}
    )
    changed_prior = dataclasses.replace(
        contract,
        prior=ParameterPrior(
            prior_manifest=prior_manifest,
            support_policy="unbounded",
            log_density_authority="graph_native",
        ),
    )
    filter_manifest = dict(contract.filter_program.filter_manifest)
    filter_manifest["time_order"] = "observe_y0_then_transition"
    filter_manifest["filter_hash"] = "sha256:" + _semantic_hash(
        {key: value for key, value in filter_manifest.items() if key != "filter_hash"}
    )
    changed_time = dataclasses.replace(
        contract,
        filter_program=FilterProgram(
            filter_id=contract.filter_program.filter_id,
            required_model_capabilities=contract.filter_program.required_model_capabilities,
            deterministic_target_policy=contract.filter_program.deterministic_target_policy,
            approximation_semantics=contract.filter_program.approximation_semantics,
            filter_manifest=filter_manifest,
        ),
    )
    observation_manifest = dict(contract.problem.model_manifest)
    observation_manifest["theta_scaling"] = (
        "kappa=0.1*exp(theta0)",
        "nu=18*exp(theta1)",
        "R=100*exp(theta2)*I9",
    )
    observation_manifest["model_hash"] = "sha256:" + _semantic_hash(
        {key: value for key, value in observation_manifest.items() if key != "model_hash"}
    )
    changed_observation = dataclasses.replace(
        contract,
        problem=dataclasses.replace(contract.problem, model_manifest=observation_manifest),
    )
    rows = {
        "changed_observation_hash": stable_signature(changed_data),
        "changed_prior_scale": stable_signature(changed_prior),
        "changed_time_order": stable_signature(changed_time),
        "changed_observation_covariance_exponent": stable_signature(changed_observation),
    }
    return {
        "baseline_mathematical_signature": baseline,
        "mutated_signatures": rows,
        "all_mutations_rejected_by_signature": all(value != baseline for value in rows.values()),
    }


def _verify_entry() -> Mapping[str, Any]:
    if _hash(TARGET_DESIGN_RESULT) != TARGET_DESIGN_SHA256:
        raise RuntimeError("P6 target-design result hash mismatch")
    if _hash(GPU_CANARY) != GPU_CANARY_SHA256:
        raise RuntimeError("P6 GPU canary hash mismatch")
    design = json.loads(TARGET_DESIGN_RESULT.read_text(encoding="utf-8"))
    canary = json.loads(GPU_CANARY.read_text(encoding="utf-8"))
    cell = design["cells"][CELL_ID]
    if not cell["passed"] or cell["target_signature"] != MATHEMATICAL_TARGET_SIGNATURE:
        raise RuntimeError("SIR-SGQF target design is not admitted")
    if not canary["cells"][CELL_ID]["passed"]:
        raise RuntimeError("SIR-SGQF trusted GPU canary is not admitted")
    return {
        "target_design_result": str(TARGET_DESIGN_RESULT),
        "target_design_sha256": TARGET_DESIGN_SHA256,
        "gpu_canary": str(GPU_CANARY),
        "gpu_canary_sha256": GPU_CANARY_SHA256,
    }


def main() -> None:
    args = _args()
    if args.output_root.exists():
        raise FileExistsError(f"R1B output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    entry = _verify_entry()
    memory_policy = None
    if args.device == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise RuntimeError("CPU R1B requires CUDA_VISIBLE_DEVICES=-1")
        if args.cpu_result is not None:
            raise RuntimeError("CPU R1B must not receive --cpu-result")
    else:
        if args.cpu_result is None:
            raise RuntimeError("GPU R1B requires --cpu-result")
        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)

    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        SIR_OBSERVATION_SHA256,
        SIR_STATE_SHA256,
        SIRSGQFLikelihoodRecomposer,
        generate_frozen_sir_dataset_tf,
        make_sir_sgqf_neutra_adapter,
        sir_identity_chart_jacobian_value_score,
        sir_prior_value_score,
        sir_sgqf_likelihood_value_score_status,
    )

    states, observations, _all = generate_frozen_sir_dataset_tf()
    state_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(states).numpy())).hexdigest()
    observation_hash = hashlib.sha256(bytes(tf.io.serialize_tensor(observations).numpy())).hexdigest()
    if state_hash != SIR_STATE_SHA256 or observation_hash != SIR_OBSERVATION_SHA256:
        raise RuntimeError("frozen SIR dataset hash mismatch")
    adapter = make_sir_sgqf_neutra_adapter(observations=observations)
    mathematical_signature = stable_ssm_target_signature(adapter.contract)
    if mathematical_signature != MATHEMATICAL_TARGET_SIGNATURE:
        raise RuntimeError("SIR-SGQF mathematical signature drift")
    points = _audit_points()
    eager_value, eager_score, eager_status = adapter.neutra_batch_log_prob_and_grad_status(points)

    @tf.function(input_signature=[tf.TensorSpec([13, 3], tf.float64)], jit_compile=True)
    def compiled(theta: tf.Tensor):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    value, score, status = compiled(points)
    reversed_value, reversed_score, reversed_status = compiled(tf.reverse(points, axis=(0,)))
    permutation_value_gap = tf.reduce_max(tf.abs(value - tf.reverse(reversed_value, axis=(0,))))
    permutation_score_gap = tf.reduce_max(tf.abs(score - tf.reverse(reversed_score, axis=(0,))))
    permutation_status_equal = tf.reduce_all(
        tf.equal(status["status_code"], tf.reverse(reversed_status["status_code"], axis=(0,)))
    )
    eager_replay_value, eager_replay_score, eager_replay_status = (
        adapter.neutra_batch_log_prob_and_grad_status(points)
    )
    eager_replay_passed = bool(
        tf.reduce_all(tf.equal(eager_value, eager_replay_value)).numpy()
        and tf.reduce_all(tf.equal(eager_score, eager_replay_score)).numpy()
        and tf.reduce_all(
            tf.equal(eager_status["status_code"], eager_replay_status["status_code"])
        ).numpy()
    )
    fd_fine = _fd(adapter, points, 5.0e-5)
    fd_coarse = _fd(adapter, points, 1.0e-4)
    analytic_fd = _gap(eager_score, fd_fine)
    fd_step = _gap(fd_fine, fd_coarse)
    xla_value = _gap(value, eager_value)
    xla_score = _gap(score, eager_score)

    recomposer = SIRSGQFLikelihoodRecomposer(adapter)
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=points,
        prior_value_score_fn=sir_prior_value_score,
        likelihood_value_score_fn=recomposer.__call__,
        jacobian_value_score_fn=sir_identity_chart_jacobian_value_score,
        value_tolerance=1.0e-9,
        score_tolerance=1.0e-8,
    )
    substitutions = _mutated_contract_signatures(adapter, stable_ssm_target_signature)
    likelihood_value, _likelihood_score = recomposer(points)
    prior_value, _prior_score = sir_prior_value_score(points)
    omitted_prior_gap = tf.reduce_max(tf.abs(eager_value - likelihood_value))
    duplicated_prior_gap = tf.reduce_max(
        tf.abs(eager_value - (likelihood_value + 2.0 * prior_value))
    )
    substitutions.update(
        {
            "omitted_prior_value_gap": omitted_prior_gap,
            "duplicated_prior_value_gap": duplicated_prior_gap,
            "omitted_prior_rejected": omitted_prior_gap > 1.0e-6,
            "duplicated_prior_rejected": duplicated_prior_gap > 1.0e-6,
            "zero_chart_jacobian_numerically_detectable": False,
            "zero_chart_jacobian_convention_signature_bound": True,
        }
    )

    source_functions = (
        adapter.log_prob_and_grad,
        sir_sgqf_likelihood_value_score_status,
        sir_prior_value_score,
        sir_identity_chart_jacobian_value_score,
    )
    source = "\n".join(inspect.getsource(function) for function in source_functions)
    static_source = {
        "inspected_functions": [function.__qualname__ for function in source_functions],
        "active_numpy": "import numpy" in source or "np." in source,
        "host_callback": "numpy_function" in source or "py_function" in source,
        "python_algorithmic_time_or_batch_loop": "for time" in source or "for sample" in source,
    }
    static_source["passed"] = not any(
        static_source[key]
        for key in ("active_numpy", "host_callback", "python_algorithmic_time_or_batch_loop")
    )

    primary_pass = bool(
        tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
        and tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
        and float(permutation_value_gap.numpy()) <= 1.0e-10
        and float(permutation_score_gap.numpy()) <= 1.0e-10
        and bool(permutation_status_equal.numpy())
        and eager_replay_passed
        and float(analytic_fd["maximum_absolute_gap"].numpy()) <= 5.0e-3
        and float(analytic_fd["maximum_scale_normalized_gap"].numpy()) <= 5.0e-4
        and float(fd_step["maximum_absolute_gap"].numpy()) <= 5.0e-3
        and float(fd_step["maximum_scale_normalized_gap"].numpy()) <= 5.0e-4
        and float(xla_value["maximum_scale_normalized_gap"].numpy()) <= 1.0e-8
        and float(xla_score["maximum_scale_normalized_gap"].numpy()) <= 1.0e-7
        and recomposition.passed
        and substitutions["all_mutations_rejected_by_signature"]
        and bool(substitutions["omitted_prior_rejected"].numpy())
        and bool(substitutions["duplicated_prior_rejected"].numpy())
        and bool(static_source["passed"])
    )
    registry_row = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_registry.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "previous_state": "TARGET_BLOCKED",
        "state": "VALUE_SCORE_ADMITTED",
        "target_signature": mathematical_signature,
        "dataset_sha256": observation_hash,
        "selected_sparse_level": 2,
        "repair_evidence": str(TARGET_DESIGN_RESULT),
        "remaining_blockers": [],
    }
    registry_path = args.output_root / "repaired_registry.json"
    _write(registry_path, registry_row)
    identity = None
    identity_payload = None
    if primary_pass:
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

    cpu_replay = None
    if args.device == "gpu":
        cpu_replay = json.loads(args.cpu_result.read_text(encoding="utf-8"))
        if _jsonable(identity_payload) != cpu_replay["target_identity"]:
            primary_pass = False
        cpu_value = tf.constant(cpu_replay["compiled_value"], tf.float64)
        cpu_score = tf.constant(cpu_replay["compiled_score"], tf.float64)
        cpu_value_parity = _gap(value, cpu_value)
        cpu_score_parity = _gap(score, cpu_score)
        cpu_status_equal = bool(
            tf.reduce_all(
                tf.equal(
                    status["status_code"],
                    tf.constant(cpu_replay["compiled_status"]["status_code"], tf.int32),
                )
            ).numpy()
        )
        primary_pass = bool(
            primary_pass
            and cpu_value_parity["maximum_scale_normalized_gap"] <= 1.0e-8
            and cpu_score_parity["maximum_scale_normalized_gap"] <= 1.0e-7
            and cpu_status_equal
            and all("GPU" in item.device.upper() for item in (value, score, status["status_code"]))
        )
    else:
        cpu_value_parity = None
        cpu_score_parity = None
        cpu_status_equal = None

    passed = bool(primary_pass and identity_payload is not None)
    if passed:
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
    result = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_r1b.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "device_intent": args.device,
        "passed": passed,
        "decision": "ADMIT_SIR_SGQF_POSTERIOR_IDENTITY" if passed else "BLOCK_SIR_SGQF_R1B",
        "entry_evidence": entry,
        "dataset": {
            "state_sha256": state_hash,
            "observation_sha256": observation_hash,
            "horizon": 20,
            "seed": 81120,
            "time_order": "transition_then_observe_y1_through_y20",
        },
        "audit_points": points,
        "eager_value": eager_value,
        "eager_score": eager_score,
        "compiled_value": value,
        "compiled_score": score,
        "compiled_status": status,
        "permutation_value_gap": permutation_value_gap,
        "permutation_score_gap": permutation_score_gap,
        "permutation_status_equal": permutation_status_equal,
        "eager_replay_passed": eager_replay_passed,
        "analytic_fine_fd_gap": analytic_fd,
        "fine_coarse_fd_gap": fd_step,
        "compiled_eager_value_parity": xla_value,
        "compiled_eager_score_parity": xla_score,
        "recomposition": recomposition.payload(),
        "substitution_negatives": substitutions,
        "static_source_audit": static_source,
        "mathematical_target_signature": mathematical_signature,
        "target_identity": identity_payload,
        "registry_row": registry_row,
        "memory_policy": memory_policy,
        "cpu_result_path": None if args.cpu_result is None else str(args.cpu_result),
        "cpu_result_sha256": None if args.cpu_result is None else _hash(args.cpu_result),
        "cpu_value_parity": cpu_value_parity,
        "cpu_score_parity": cpu_score_parity,
        "cpu_status_equal": cpu_status_equal,
        "output_devices": [value.device, score.device, status["status_code"].device],
        "jit_compile": True,
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "elapsed_seconds": time.perf_counter() - started,
        "nonclaims": [
            "typed posterior identity and value-score admission only",
            "no HMC, NeuTra, filter exactness, calibration, forecasting, robustness, or readiness claim",
        ],
    }
    _write(args.output_root / "result.json", result)
    manifest = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_r1b_manifest.v1",
        "program_id": PROGRAM_ID,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
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
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_sgqf_r1b_hashes.v1",
            "artifacts": hashes,
        },
    )


if __name__ == "__main__":
    main()
