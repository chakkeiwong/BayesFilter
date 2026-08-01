#!/usr/bin/env python3
"""Execute the reviewed P6 parameterized-SIR target-design gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
PLAN_FILE = (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p6-sir-target-design-subplan-2026-07-16.md"
)
RESULT_NOTE = (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p6-sir-target-design-result-2026-07-16.md"
)
PRIOR_PREDICTIVE_SEED = (20260716, 16201)
PF_SEED = (20260716, 16202)
FD_FINE = 5.0e-5
FD_COARSE = 1.0e-4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-canary-only", action="store_true")
    parser.add_argument("--cpu-result", type=Path)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _audit_points() -> tf.Tensor:
    eye = tf.eye(3, dtype=tf.float64)
    zero = tf.zeros([1, 3], tf.float64)
    return tf.concat([zero, 0.5 * eye, -0.5 * eye, eye, -eye], axis=0)


def _design_points() -> tf.Tensor:
    axis = tf.constant([-math.log(2.0), 0.0, math.log(2.0)], tf.float64)
    mesh = tf.meshgrid(axis, axis, axis, indexing="ij")
    return tf.reshape(tf.stack(mesh, axis=-1), [27, 3])


def _stencil(points: tf.Tensor, step: float) -> tf.Tensor:
    offsets = tf.constant(step, tf.float64) * tf.eye(3, dtype=tf.float64)
    plus = points[:, None, :] + offsets[None, :, :]
    minus = points[:, None, :] - offsets[None, :, :]
    return tf.reshape(tf.concat([plus, minus], axis=1), [-1, 3])


def _fd_from_values(values: tf.Tensor, row_count: int, step: float) -> tf.Tensor:
    reshaped = tf.reshape(values, [row_count, 6])
    return (reshaped[:, :3] - reshaped[:, 3:]) / (2.0 * step)


def _curvature_from_scores(
    scores: tf.Tensor, row_count: int, step: float
) -> tf.Tensor:
    reshaped = tf.reshape(scores, [row_count, 6, 3])
    jacobian = tf.transpose(
        (reshaped[:, :3, :] - reshaped[:, 3:, :]) / (2.0 * step),
        [0, 2, 1],
    )
    curvature = -jacobian
    return 0.5 * (curvature + tf.linalg.matrix_transpose(curvature))


def _gap_metrics(left: tf.Tensor, right: tf.Tensor) -> Mapping[str, tf.Tensor]:
    difference = tf.abs(left - right)
    scale = tf.maximum(tf.ones_like(left), tf.maximum(tf.abs(left), tf.abs(right)))
    return {
        "maximum_absolute_gap": tf.reduce_max(difference),
        "maximum_scale_normalized_gap": tf.reduce_max(difference / scale),
    }


def _evaluate_stencil_in_xla_chunks(
    compiled_chunk: Callable[[tf.Tensor], Any], stencil: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Evaluate complete six-row stencils with one bounded XLA batch graph."""

    values = tf.convert_to_tensor(stencil, tf.float64)
    row_count = int(values.shape[0])
    if row_count % 6 != 0:
        raise ValueError("central stencil row count must be divisible by six")
    chunk_count = row_count // 6

    # This is diagnostic scheduling over fixed design points, not an algorithm,
    # sample-generation, training-step, or HMC-transition loop.
    chunks = [compiled_chunk(values[6 * index : 6 * (index + 1)]) for index in range(chunk_count)]
    return (
        tf.concat([chunk[0] for chunk in chunks], axis=0),
        tf.concat([chunk[1] for chunk in chunks], axis=0),
        tf.concat([chunk[2] for chunk in chunks], axis=0),
    )


def _evaluate_stencil_in_eager_chunks(
    evaluate: Callable[[tf.Tensor], Any], stencil: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Evaluate complete six-row stencils in one consistent eager mode."""

    values = tf.convert_to_tensor(stencil, tf.float64)
    row_count = int(values.shape[0])
    if row_count % 6 != 0:
        raise ValueError("central stencil row count must be divisible by six")
    chunks = [
        evaluate(values[6 * index : 6 * (index + 1)])
        for index in range(row_count // 6)
    ]
    return (
        tf.concat([chunk[0] for chunk in chunks], axis=0),
        tf.concat([chunk[1] for chunk in chunks], axis=0),
        tf.concat(
            [chunk[2]["valid_pre_regularized_score"] for chunk in chunks], axis=0
        ),
    )


def _filter_components(kind: str, observations: tf.Tensor):
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        make_sir_sgqf_neutra_adapter,
        make_sir_ukf_neutra_adapter,
        sir_sgqf_likelihood_value_score_status,
        sir_ukf_likelihood_value_score_status,
    )

    if kind == "SIR-UKF":
        adapter = make_sir_ukf_neutra_adapter(observations=observations)

        def evaluate(theta: tf.Tensor):
            return sir_ukf_likelihood_value_score_status(
                theta, observations=observations
            )

        return adapter, evaluate
    if kind == "SIR-SGQF":
        adapter = make_sir_sgqf_neutra_adapter(observations=observations)

        def evaluate(theta: tf.Tensor):
            return sir_sgqf_likelihood_value_score_status(
                theta,
                observations=observations,
                nodes=adapter.nodes,
                weights=adapter.weights,
            )

        return adapter, evaluate
    raise ValueError(f"unknown filter kind: {kind}")


def _static_source_audit(kind: str) -> Mapping[str, Any]:
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        sir_rk4_transition_value_state_source_jacobians,
        sir_sgqf_likelihood_value_score_status,
        sir_ukf_likelihood_value_score_status,
    )

    functions = [sir_rk4_transition_value_state_source_jacobians]
    functions.append(
        sir_ukf_likelihood_value_score_status
        if kind == "SIR-UKF"
        else sir_sgqf_likelihood_value_score_status
    )
    sources = "\n".join(inspect.getsource(function) for function in functions)
    forbidden = {
        "numpy": "numpy" in sources or "np." in sources,
        "host_callback": "numpy_function" in sources or "py_function" in sources,
        "python_time_loop": "for time" in sources or "for index" in sources,
        "python_batch_loop": "for row" in sources or "for sample" in sources,
    }
    return {
        "inspected_functions": [function.__name__ for function in functions],
        "forbidden_patterns": forbidden,
        "passed": not any(forbidden.values()),
    }


def _run_filter_design(kind: str, observations: tf.Tensor) -> Mapping[str, Any]:
    from bayesfilter.ssm import stable_ssm_target_signature

    adapter, evaluate = _filter_components(kind, observations)

    @tf.function(input_signature=[tf.TensorSpec([6, 3], tf.float64)], jit_compile=True)
    def compiled_stencil_chunk(theta: tf.Tensor):
        value, score, status = evaluate(theta)
        return value, score, status["valid_pre_regularized_score"]

    audit_points = _audit_points()
    design_points = _design_points()
    eager_value, eager_score, eager_status = evaluate(audit_points)

    fd_rows: dict[str, Any] = {}
    fd_values: dict[float, tf.Tensor] = {}
    for label, step in (("fine", FD_FINE), ("coarse", FD_COARSE)):
        stencil_value, _stencil_score, stencil_valid = (
            _evaluate_stencil_in_eager_chunks(
                evaluate, _stencil(audit_points, step)
            )
        )
        fd = _fd_from_values(stencil_value, 13, step)
        fd_values[step] = fd
        gap = _gap_metrics(eager_score, fd)
        fd_rows[label] = {
            "step": step,
            "finite_difference_score": fd,
            "analytic_gap": gap,
            "all_status_valid": tf.reduce_all(stencil_valid),
        }
    fd_step_gap = _gap_metrics(fd_values[FD_FINE], fd_values[FD_COARSE])
    fd_passed = bool(
        (
            fd_rows["fine"]["analytic_gap"]["maximum_absolute_gap"] <= 5.0e-3
        ).numpy()
        and (
            fd_rows["fine"]["analytic_gap"]["maximum_scale_normalized_gap"]
            <= 5.0e-4
        ).numpy()
        and (fd_step_gap["maximum_absolute_gap"] <= 5.0e-3).numpy()
        and (fd_step_gap["maximum_scale_normalized_gap"] <= 5.0e-4).numpy()
        and all(bool(row["all_status_valid"].numpy()) for row in fd_rows.values())
    )

    curvature_rows: dict[str, Any] = {}
    curvature: dict[float, tf.Tensor] = {}
    curvature_status_valid = True
    for label, step in (("fine", FD_FINE), ("coarse", FD_COARSE)):
        _value, stencil_score, stencil_valid = _evaluate_stencil_in_xla_chunks(
            compiled_stencil_chunk, _stencil(design_points, step)
        )
        matrix = _curvature_from_scores(stencil_score, 27, step)
        singular_values = tf.linalg.svd(matrix, compute_uv=False)
        rank = tf.reduce_sum(
            tf.cast(
                singular_values
                > 1.0e-8 * tf.maximum(singular_values[:, :1], 1.0e-300),
                tf.int32,
            ),
            axis=1,
        )
        curvature[step] = matrix
        status_valid = tf.reduce_all(stencil_valid)
        curvature_status_valid = curvature_status_valid and bool(status_valid.numpy())
        curvature_rows[label] = {
            "step": step,
            "curvature": matrix,
            "eigenvalues": tf.linalg.eigvalsh(matrix),
            "singular_values": singular_values,
            "numerical_rank": rank,
            "all_rank_three": tf.reduce_all(tf.equal(rank, 3)),
            "all_status_valid": status_valid,
        }
    curvature_difference = tf.linalg.norm(
        curvature[FD_FINE] - curvature[FD_COARSE], axis=[1, 2]
    )
    curvature_scale = tf.maximum(
        tf.ones([27], tf.float64),
        tf.maximum(
            tf.linalg.norm(curvature[FD_FINE], axis=[1, 2]),
            tf.linalg.norm(curvature[FD_COARSE], axis=[1, 2]),
        ),
    )
    curvature_relative_gap = curvature_difference / curvature_scale
    curvature_passed = (
        curvature_status_valid
        and bool(curvature_rows["fine"]["all_rank_three"].numpy())
        and bool(curvature_rows["coarse"]["all_rank_three"].numpy())
        and bool(tf.reduce_all(curvature_relative_gap <= 5.0e-3).numpy())
    )

    permutation = tf.constant([12, 0, 7, 4, 10, 2, 9, 6, 1, 11, 3, 8, 5], tf.int32)
    permuted_value, permuted_score, permuted_status = evaluate(
        tf.gather(audit_points, permutation)
    )
    permutation_passed = bool(
        tf.reduce_all(tf.equal(permuted_value, tf.gather(eager_value, permutation))).numpy()
        and tf.reduce_all(tf.equal(permuted_score, tf.gather(eager_score, permutation))).numpy()
        and tf.reduce_all(
            tf.equal(
                permuted_status["status_code"],
                tf.gather(eager_status["status_code"], permutation),
            )
        ).numpy()
    )
    replay_value, replay_score, replay_status = evaluate(audit_points)
    replay_passed = bool(
        tf.reduce_all(tf.equal(replay_value, eager_value)).numpy()
        and tf.reduce_all(tf.equal(replay_score, eager_score)).numpy()
        and tf.reduce_all(
            tf.equal(replay_status["status_code"], eager_status["status_code"])
        ).numpy()
    )

    @tf.function(
        input_signature=[tf.TensorSpec([13, 3], tf.float64)], jit_compile=True
    )
    def compiled(theta: tf.Tensor):
        return evaluate(theta)

    xla_value, xla_score, xla_status = compiled(audit_points)
    xla_value_metrics = _gap_metrics(xla_value, eager_value)
    xla_score_metrics = _gap_metrics(xla_score, eager_score)
    xla_passed = bool(
        (xla_value_metrics["maximum_scale_normalized_gap"] <= 1.0e-8).numpy()
        and (xla_score_metrics["maximum_scale_normalized_gap"] <= 1.0e-7).numpy()
        and tf.reduce_all(
            tf.equal(xla_status["status_code"], eager_status["status_code"])
        ).numpy()
    )
    static_audit = _static_source_audit(kind)
    base_passed = bool(
        tf.reduce_all(eager_status["valid_pre_regularized_score"]).numpy()
        and tf.reduce_all(tf.math.is_finite(eager_value)).numpy()
        and tf.reduce_all(tf.math.is_finite(eager_score)).numpy()
    )
    passed = all(
        (
            base_passed,
            fd_passed,
            curvature_passed,
            permutation_passed,
            replay_passed,
            xla_passed,
            bool(static_audit["passed"]),
        )
    )
    return {
        "cell_id": kind,
        "decision": "TARGET_DESIGN_READY_FOR_R1B" if passed else "TARGET_DESIGN_BLOCKED",
        "passed": passed,
        "target_signature": stable_ssm_target_signature(adapter.contract),
        "adapter_signature": adapter.adapter_signature(),
        "audit_points": audit_points,
        "design_points": design_points,
        "eager": {"value": eager_value, "score": eager_score, "status": eager_status},
        "score_finite_difference": {
            "rows": fd_rows,
            "fine_coarse_gap": fd_step_gap,
            "passed": fd_passed,
        },
        "observed_curvature": {
            "rows": curvature_rows,
            "fine_coarse_relative_frobenius_gap": curvature_relative_gap,
            "passed": curvature_passed,
            "nonclaim": "not Fisher information; eigenvalue signs are explanatory",
        },
        "batch_permutation_passed": permutation_passed,
        "deterministic_replay_passed": replay_passed,
        "cpu_xla": {
            "jit_compile": True,
            "value": xla_value,
            "score": xla_score,
            "status": xla_status,
            "value_parity": xla_value_metrics,
            "score_parity": xla_score_metrics,
            "output_devices": [xla_value.device, xla_score.device],
            "passed": xla_passed,
        },
        "static_source_audit": static_audit,
        "parity_reference": {
            "theta": audit_points[:2],
            "value": eager_value[:2],
            "score": eager_score[:2],
            "status_code": eager_status["status_code"][:2],
        },
    }


def _run_prior_predictive() -> Mapping[str, Any]:
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        sir_prior_predictive_tf,
    )

    @tf.function(jit_compile=True)
    def compiled():
        return dict(
            sir_prior_predictive_tf(
                batch_size=4096,
                horizon=20,
                seed=tf.constant(PRIOR_PREDICTIVE_SEED, tf.int32),
            )
        )

    result = compiled()
    combined = tf.concat([result["states"], result["observations"]], axis=2)
    finite = tf.reduce_all(tf.math.is_finite(combined), axis=[1, 2])
    susceptible_valid = tf.reduce_all(result["states"][:, :, 0::2] >= 0.0, axis=[1, 2])
    magnitude = tf.reduce_max(tf.abs(combined), axis=[1, 2])
    valid = tf.logical_and(finite, magnitude <= 1.0e6)
    sorted_magnitude = tf.sort(magnitude)
    indices = tf.constant([2047, 3890, 4054, 4095], tf.int32)
    valid_fraction = tf.reduce_mean(tf.cast(valid, tf.float64))
    return {
        "batch_size": 4096,
        "horizon": 20,
        "seed": PRIOR_PREDICTIVE_SEED,
        "jit_compile": True,
        "valid_count": tf.reduce_sum(tf.cast(valid, tf.int32)),
        "valid_fraction": valid_fraction,
        "finite_fraction": tf.reduce_mean(tf.cast(finite, tf.float64)),
        "nonnegative_susceptible_fraction": tf.reduce_mean(
            tf.cast(susceptible_valid, tf.float64)
        ),
        "nonnegative_susceptible_role": "explanatory_support_telemetry_only",
        "magnitude_q50_q95_q99_max": tf.gather(sorted_magnitude, indices),
        "passed": valid_fraction >= 0.99,
    }


def _run_pf_reference(observations: tf.Tensor) -> Mapping[str, Any]:
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        sir_bootstrap_pf_log_likelihood_tf,
    )

    @tf.function(jit_compile=True)
    def compiled():
        return sir_bootstrap_pf_log_likelihood_tf(
            tf.zeros([3], tf.float64),
            observations=observations,
            particle_count=4096,
            replicate_count=4,
            seed=tf.constant(PF_SEED, tf.int32),
        )

    estimates = compiled()
    return {
        "particle_count": 4096,
        "replicate_count": 4,
        "seed": PF_SEED,
        "jit_compile": True,
        "truth_log_likelihood_estimates": estimates,
        "mean": tf.reduce_mean(estimates),
        "standard_deviation": tf.math.reduce_std(estimates),
        "all_finite": tf.reduce_all(tf.math.is_finite(estimates)),
        "role": "explanatory_and_gross_target_veto_only",
        "nonclaim": "four PF replications cannot rank or validate UKF/SGQF",
    }


def _run_negative_substitutions(
    states: tf.Tensor,
    observations: tf.Tensor,
    all_observations: tf.Tensor,
    cells: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    from bayesfilter.highdim.models import (
        parameterized_zhao_cui_sir_austria_model,
        zhao_cui_sir_austria_local_complete_data_log_density_xla,
    )
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        SIR_OBSERVATION_SHA256,
        SIR_WRONG_TIME_ORDER_SHA256,
        make_sir_ukf_neutra_adapter,
        sir_prior_value_score,
    )

    failures: dict[str, bool] = {}
    try:
        make_sir_ukf_neutra_adapter(observations=all_observations[:20])
        failures["wrong_time_order_rejected"] = False
    except ValueError:
        failures["wrong_time_order_rejected"] = True
    _wrong_states, wrong_all = parameterized_zhao_cui_sir_austria_model().base_model.simulate(
        final_time=20, seed=81121
    )
    try:
        make_sir_ukf_neutra_adapter(observations=tf.convert_to_tensor(wrong_all[1:21], tf.float64))
        failures["wrong_seed_rejected"] = False
    except ValueError:
        failures["wrong_seed_rejected"] = True
    try:
        sir_prior_value_score(tf.zeros([1, 3], tf.float32))
        failures["wrong_dtype_rejected"] = False
    except ValueError:
        failures["wrong_dtype_rejected"] = True
    prior_canonical = {"family": "Normal", "mean": 0.0, "scale": 0.5}
    prior_wrong = {"family": "Normal", "mean": 0.0, "scale": 1.0}
    observation_canonical = {"R": "100*exp(2*theta2)*I9"}
    observation_wrong = {"R": "100*exp(theta2)*I9"}
    failures["prior_scale_identity_changed"] = _semantic_hash(prior_canonical) != _semantic_hash(prior_wrong)
    failures["observation_covariance_identity_changed"] = _semantic_hash(observation_canonical) != _semantic_hash(observation_wrong)
    failures["filter_identity_changed"] = cells["SIR-UKF"]["target_signature"] != cells["SIR-SGQF"]["target_signature"]
    local_complete = zhao_cui_sir_austria_local_complete_data_log_density_xla(
        tf.zeros([3], tf.float64), states, all_observations
    )
    filter_truth = {
        key: value["eager"]["value"][0] for key, value in cells.items()
    }
    failures["local_complete_data_scalar_distinct"] = all(
        bool(tf.not_equal(local_complete, value).numpy()) for value in filter_truth.values()
    )
    return {
        "canonical_observation_sha256": SIR_OBSERVATION_SHA256,
        "wrong_time_order_sha256": SIR_WRONG_TIME_ORDER_SHA256,
        "checks": failures,
        "local_complete_data_value_at_truth": local_complete,
        "filter_likelihood_values_at_truth": filter_truth,
        "passed": all(failures.values()),
        "nonclaim": "identity-change checks do not issue the R1B typed posterior identity",
    }


def _run_sgqf_cloud() -> Mapping[str, Any]:
    from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_level2_axis_cloud

    cloud = tf_fixed_sgqf_level2_axis_cloud(18)
    mean = tf.einsum("r,rd->d", cloud.weights, cloud.points)
    covariance = tf.einsum("r,ri,rj->ij", cloud.weights, cloud.points, cloud.points)
    cloud_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(cloud.points).numpy())
        + bytes(tf.io.serialize_tensor(cloud.weights).numpy())
    ).hexdigest()
    passed = (
        cloud.point_count == 37
        and cloud.negative_weight_count == 1
        and bool(tf.reduce_max(tf.abs(mean)).numpy() <= 1.0e-14)
        and bool(tf.reduce_max(tf.abs(covariance - tf.eye(18, dtype=tf.float64))).numpy() <= 1.0e-14)
    )
    return {
        "point_count": cloud.point_count,
        "negative_weight_count": cloud.negative_weight_count,
        "weight_total": tf.reduce_sum(cloud.weights),
        "mean_maximum_absolute_gap": tf.reduce_max(tf.abs(mean)),
        "covariance_maximum_absolute_gap": tf.reduce_max(
            tf.abs(covariance - tf.eye(18, dtype=tf.float64))
        ),
        "cloud_sha256": cloud_hash,
        "stored_tensor_bytes": 37 * 18 * 8 + 37 * 8,
        "per_batch_point_state_bytes": 37 * 18 * 8,
        "low_dimensional_parity_test": "tests/test_fixed_sgqf_tf.py and focused P6 suite",
        "passed": passed,
    }


def _run_gpu_canary(cpu_result_path: Path | None) -> Mapping[str, Any]:
    policy = configure_tensorflow_gpu_memory_growth(tf)
    devices = tf.config.list_logical_devices("GPU")
    if not devices:
        raise RuntimeError("trusted GPU canary requires a visible TensorFlow GPU")
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        generate_frozen_sir_dataset_tf,
    )

    _states, observations, _all = generate_frozen_sir_dataset_tf()
    theta = _audit_points()[:2]
    cpu_result = None
    if cpu_result_path is not None:
        cpu_result = json.loads(cpu_result_path.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    for kind in ("SIR-UKF", "SIR-SGQF"):
        _adapter, evaluate = _filter_components(kind, observations)

        @tf.function(
            input_signature=[tf.TensorSpec([2, 3], tf.float64)], jit_compile=True
        )
        def compiled(values: tf.Tensor):
            return evaluate(values)

        value, score, status = compiled(theta)
        baseline = None if cpu_result is None else cpu_result["cells"][kind]["parity_reference"]
        value_metrics = {
            "maximum_absolute_gap": tf.constant(float("nan"), tf.float64),
            "maximum_scale_normalized_gap": tf.constant(float("nan"), tf.float64),
        }
        score_metrics = dict(value_metrics)
        status_equal = False
        if baseline is not None:
            value_metrics = _gap_metrics(
                value, tf.constant(baseline["value"], tf.float64)
            )
            score_metrics = _gap_metrics(
                score, tf.constant(baseline["score"], tf.float64)
            )
            status_equal = bool(
                tf.reduce_all(
                    tf.equal(
                        status["status_code"],
                        tf.constant(baseline["status_code"], tf.int32),
                    )
                ).numpy()
            )
        passed = bool(
            tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
            and tf.reduce_all(tf.math.is_finite(value)).numpy()
            and tf.reduce_all(tf.math.is_finite(score)).numpy()
            and all("GPU" in name.upper() for name in (value.device, score.device))
            and baseline is not None
            and (
                value_metrics["maximum_scale_normalized_gap"] <= 1.0e-8
            ).numpy()
            and (
                score_metrics["maximum_scale_normalized_gap"] <= 1.0e-7
            ).numpy()
            and status_equal
        )
        rows[kind] = {
            "passed": passed,
            "value": value,
            "score": score,
            "status": status,
            "value_cpu_parity": value_metrics,
            "score_cpu_parity": score_metrics,
            "status_cpu_equal": status_equal,
            "output_devices": [value.device, score.device],
        }
    return {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_gpu_canary.v1",
        "program_id": PROGRAM_ID,
        "cells": rows,
        "memory_policy": policy,
        "logical_gpus": [device.name for device in devices],
        "jit_compile": True,
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "passed": all(row["passed"] for row in rows.values()),
    }


def _artifact_hashes(output_root: Path, names: list[str]) -> None:
    _write_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_hashes.v1",
            "artifacts": {name: _sha256(output_root / name) for name in names},
        },
    )


def main() -> None:
    args = _parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if args.gpu_canary_only:
        result = _run_gpu_canary(args.cpu_result)
        result = {**result, "elapsed_seconds": time.perf_counter() - started}
        _write_json(output_root / "gpu_canary.json", result)
        manifest = {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_gpu_canary_manifest.v1",
            "program_id": PROGRAM_ID,
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "cpu_result_path": None if args.cpu_result is None else str(args.cpu_result),
            "cpu_result_sha256": None if args.cpu_result is None else _sha256(args.cpu_result),
            "device_intent": "trusted GPU/XLA target-design parity canary",
            "memory_policy": result["memory_policy"],
            "tf32_enabled": result["tf32_enabled"],
            "wall_time_seconds": result["elapsed_seconds"],
            "output_root": str(output_root),
            "plan_file": PLAN_FILE,
        }
        _write_json(output_root / "run_manifest.json", manifest)
        _artifact_hashes(output_root, ["gpu_canary.json", "run_manifest.json"])
        return

    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        SIR_OBSERVATION_SHA256,
        SIR_STATE_SHA256,
        SIR_WRONG_TIME_ORDER_SHA256,
        generate_frozen_sir_dataset_tf,
        sir_bootstrap_pf_log_likelihood_tf,
        sir_prior_predictive_tf,
        sir_rk4_transition_value,
        sir_rk4_transition_value_state_source_jacobians,
        sir_sgqf_likelihood_value_score_status,
        sir_ukf_likelihood_value_score_status,
    )

    _write_json(
        output_root / "attempt_status.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_attempt_status.v1",
            "status": "STARTED",
            "program_id": PROGRAM_ID,
        },
    )
    states, observations, all_observations = generate_frozen_sir_dataset_tf()
    dataset = {
        "seed": 81120,
        "state_shape": states.shape.as_list(),
        "observation_shape": observations.shape.as_list(),
        "state_sha256": _tensor_sha256(states),
        "observation_sha256": _tensor_sha256(observations),
        "wrong_time_order_sha256": _tensor_sha256(all_observations[:20]),
        "expected_hashes_match": (
            _tensor_sha256(states) == SIR_STATE_SHA256
            and _tensor_sha256(observations) == SIR_OBSERVATION_SHA256
            and _tensor_sha256(all_observations[:20]) == SIR_WRONG_TIME_ORDER_SHA256
        ),
        "time_order": "x0_prior_then_transition_and_observe_y1_through_y20",
        "minimum_susceptible_state": tf.reduce_min(states[:, 0::2]),
        "fixture_clipping_inactive": tf.reduce_all(states[:, 0::2] >= 0.0),
    }
    _write_json(output_root / "dataset.json", dataset)
    prior_predictive = _run_prior_predictive()
    _write_json(output_root / "prior_predictive.json", prior_predictive)
    sgqf_cloud = _run_sgqf_cloud()
    _write_json(output_root / "sgqf_cloud.json", sgqf_cloud)
    cells = {
        kind: _run_filter_design(kind, observations)
        for kind in ("SIR-UKF", "SIR-SGQF")
    }
    for kind, row in cells.items():
        _write_json(output_root / f"{kind.lower()}-target-design.json", row)
    pf_reference = _run_pf_reference(observations)
    _write_json(output_root / "pf_reference.json", pf_reference)
    substitutions = _run_negative_substitutions(
        states, observations, all_observations, cells
    )
    _write_json(output_root / "negative_substitutions.json", substitutions)
    common_passed = bool(
        dataset["expected_hashes_match"]
        and dataset["fixture_clipping_inactive"].numpy()
        and prior_predictive["passed"].numpy()
        and sgqf_cloud["passed"]
        and pf_reference["all_finite"].numpy()
        and substitutions["passed"]
    )
    decisions = {
        kind: (
            "TARGET_DESIGN_READY_FOR_R1B"
            if common_passed and row["passed"]
            else "TARGET_DESIGN_BLOCKED"
        )
        for kind, row in cells.items()
    }
    decisions["SIR-ZC"] = "TARGET_BLOCKED_MISSING_OBSERVED_DATA_SCORE_ROUTE"
    elapsed = time.perf_counter() - started
    result = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_target_design.v1",
        "program_id": PROGRAM_ID,
        "plan_file": PLAN_FILE,
        "result_note": RESULT_NOTE,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "device_intent": "deliberate CPU-only target design with CUDA hidden; trusted GPU/XLA canary separate",
        "dataset": dataset,
        "prior_predictive": prior_predictive,
        "sgqf_cloud": sgqf_cloud,
        "cells": cells,
        "pf_reference": pf_reference,
        "negative_substitutions": substitutions,
        "decisions": decisions,
        "common_passed": common_passed,
        "gpu_canary_required_for_admission": True,
        "typed_target_signature_issued": False,
        "source_hashes": {
            "sir_prior_predictive_tf": _source_sha256(sir_prior_predictive_tf),
            "sir_bootstrap_pf_log_likelihood_tf": _source_sha256(sir_bootstrap_pf_log_likelihood_tf),
            "sir_rk4_transition_value": _source_sha256(sir_rk4_transition_value),
            "sir_rk4_transition_value_state_source_jacobians": _source_sha256(sir_rk4_transition_value_state_source_jacobians),
            "sir_ukf_likelihood_value_score_status": _source_sha256(sir_ukf_likelihood_value_score_status),
            "sir_sgqf_likelihood_value_score_status": _source_sha256(sir_sgqf_likelihood_value_score_status),
        },
        "nonclaims": [
            "no typed posterior identity issued in target-design rung",
            "no HMC convergence or NeuTra training claim",
            "no filter exactness, superiority, epidemiological calibration, forecasting, robustness, or readiness claim",
            "observed curvature is not Fisher information or global identifiability evidence",
            "PF estimates are explanatory and cannot rank the deterministic filters",
        ],
        "elapsed_seconds": elapsed,
    }
    _write_json(output_root / "result.json", result)
    manifest = {
        "schema": "bayesfilter.multimodel_neutra_p6_sir_target_design_manifest.v1",
        "program_id": PROGRAM_ID,
        "git_commit": result["git_commit"],
        "command": " ".join(sys.argv),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_executable": sys.executable,
        "cpu_gpu_status": result["device_intent"],
        "random_seeds": {
            "dataset": 81120,
            "prior_predictive": PRIOR_PREDICTIVE_SEED,
            "particle_filter": PF_SEED,
        },
        "wall_time_seconds": elapsed,
        "output_root": str(output_root),
        "plan_file": PLAN_FILE,
        "result_file": str(output_root / "result.json"),
        "result_note": RESULT_NOTE,
    }
    _write_json(output_root / "run_manifest.json", manifest)
    artifact_names = [
        "dataset.json",
        "prior_predictive.json",
        "sgqf_cloud.json",
        "sir-ukf-target-design.json",
        "sir-sgqf-target-design.json",
        "pf_reference.json",
        "negative_substitutions.json",
        "result.json",
        "run_manifest.json",
    ]
    _artifact_hashes(output_root, artifact_names)
    _write_json(
        output_root / "attempt_status.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p6_sir_attempt_status.v1",
            "status": "COMPLETED",
            "program_id": PROGRAM_ID,
            "decisions": decisions,
            "result_sha256": _sha256(output_root / "result.json"),
        },
    )


if __name__ == "__main__":
    main()
