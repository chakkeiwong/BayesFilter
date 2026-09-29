"""Component audit for filterflow regularized transport semantics."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_REFERENCE_DTYPE,
    FILTERFLOW_UPSTREAM_BASE_COMMIT,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
THETA = 0.5
EPSILONS = (0.25, 0.5, 0.75)
HORIZON = 150
NUM_REALIZATIONS = 100
NUM_PARTICLES = 25
DATA_SEED = 111
FILTER_SEED = 555
SCALING = 0.9
THRESHOLD = 1e-3
MAX_ITER = 100
MATCH_TOLERANCE = 1e-4
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_transport_component_audit_2026-05-30.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-transport-component-audit-2026-05-30.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
UPSTREAM_FILTERFLOW_COMMIT = FILTERFLOW_UPSTREAM_BASE_COMMIT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    filterflow_status = _filterflow_status()
    reference = _run_filterflow_subprocess()
    variant_rows = _run_candidate_variants(reference)
    full_vs_from_potentials = _full_vs_from_potentials(reference)
    decision = _decision(reference, variant_rows, full_vs_from_potentials)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "Component-level audit of filterflow transport semantics on one frozen LGSSM resampling state",
        "plan_path": "docs/plans/bayesfilter-dpf-filterflow-transport-component-audit-plan-2026-05-30.md",
        "filterflow_status": filterflow_status,
        "filterflow_command": reference["command"],
        "frozen_state": reference["frozen_state"],
        "filterflow_reference": reference["reference_rows"],
        "filterflow_transport_vs_transport_from_potentials": full_vs_from_potentials,
        "candidate_variants": variant_rows,
        "interpretation": _interpretation(variant_rows),
        "settings": {
            "theta": THETA,
            "epsilons": list(EPSILONS),
            "horizon": HORIZON,
            "num_realizations": NUM_REALIZATIONS,
            "num_particles": NUM_PARTICLES,
            "data_seed": DATA_SEED,
            "filter_seed": FILTER_SEED,
            "scaling": SCALING,
            "threshold": THRESHOLD,
            "max_iter": MAX_ITER,
            "resampling_threshold": "relative Neff/ESS 0.5",
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_transport_component_audit_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _run_filterflow_subprocess() -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        raise RuntimeError(f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_subprocess_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "filterflow component subprocess failed\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_COMPONENT_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_COMPONENT_JSON_END")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"filterflow JSON sentinels missing from stdout:\n{stdout[-4000:]}")
    raw = stdout[start + len("FILTERFLOW_COMPONENT_JSON_BEGIN"):end].strip()
    payload = json.loads(raw)
    payload["command"] = (
        f"CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <structured filterflow transport component script>"
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_subprocess_script() -> str:
    return textwrap.dedent(
        f"""
        import json
        import os
        import tensorflow as tf
        np = __import__("numpy")

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from filterflow.resampling.differentiable.regularized_transport.plan import (
            transport,
            transport_from_potentials,
        )
        from filterflow.resampling.differentiable.regularized_transport.sinkhorn import sinkhorn_potentials
        from filterflow.resampling.differentiable.regularized_transport.utils import cost, diameter, max_min
        from scripts.simple_linear_common import get_data

        THETA = {THETA!r}
        EPSILONS = {list(EPSILONS)!r}
        T = {HORIZON}
        BATCH_SIZE = {NUM_REALIZATIONS}
        N = {NUM_PARTICLES}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}
        SCALING = {SCALING!r}
        THRESHOLD = {THRESHOLD!r}
        MAX_ITER = {MAX_ITER}

        transition_matrix = 0.5 * np.eye(2, dtype=np.float64)
        transition_covariance = np.eye(2, dtype=np.float64)
        observation_matrix = np.eye(2, dtype=np.float64)
        observation_covariance = 0.1 * np.eye(2, dtype=np.float64)
        rng = np.random.RandomState(seed=DATA_SEED)
        data, _ = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T,
            rng,
            dtype=np.float64,
        )
        initial_particles = rng.normal(0., 1., [BATCH_SIZE, N, 2]).astype(np.float64)
        transition_covariance_chol = tf.linalg.cholesky(tf.convert_to_tensor(transition_covariance, dtype=tf.float64))
        observation_covariance_chol = tf.linalg.cholesky(tf.convert_to_tensor(observation_covariance, dtype=tf.float64))
        transition_var = tf.Variable(np.diag([THETA, THETA]).astype(np.float64), trainable=True, dtype=tf.float64)
        resampler = RegularisedTransform(epsilon=0.5, scaling=SCALING, convergence_threshold=THRESHOLD)
        criterion = NeffCriterion(0.5, True)
        smc = make_filter(
            tf.convert_to_tensor(observation_matrix, dtype=tf.float64),
            transition_var,
            observation_covariance_chol,
            transition_covariance_chol,
            resampler,
            criterion,
        )

        state = State(tf.constant(initial_particles, dtype=tf.float64))
        capture = None
        for t in range(T):
            flags, ess = criterion.apply(state)
            flags_np = flags.numpy()
            ess_np = ess.numpy()
            if bool(np.any(flags_np)):
                batch_index = int(np.argmin(ess_np))
                capture = {{
                    "time_index": int(t),
                    "batch_index": batch_index,
                    "triggered_count": int(np.sum(flags_np)),
                    "min_ess": float(np.min(ess_np)),
                    "selected_ess": float(ess_np[batch_index]),
                    "threshold": float(0.5 * N),
                }}
                break
            state = smc.propose_and_weight(
                state,
                tf.constant(data[t]),
                tf.constant(t, dtype=tf.int32),
                seed=tf.constant([FILTER_SEED, t + 1], dtype=tf.int32),
            )
        if capture is None:
            raise RuntimeError("no LGSSM resampling-trigger state found")

        x = state.particles[capture["batch_index"]:capture["batch_index"] + 1]
        logw = state.log_weights[capture["batch_index"]:capture["batch_index"] + 1]
        float_n = tf.cast(N, tf.float64)
        log_n = tf.math.log(float_n)
        uniform_log_weight = -log_n * tf.ones_like(logw)
        dimension = tf.cast(x.shape[-1], tf.float64)
        centered_x = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
        diameter_value = diameter(x, x)
        scale = tf.reshape(diameter_value, [-1, 1, 1]) * tf.sqrt(dimension)
        scaled_x = centered_x / tf.stop_gradient(scale)
        cost_matrix = cost(scaled_x, scaled_x)
        epsilon0_filterflow_range = max_min(scaled_x, scaled_x) ** 2
        epsilon0_max_cost = tf.reduce_max(cost_matrix, axis=[1, 2])

        rows = []
        for eps in EPSILONS:
            eps_tensor = tf.constant(eps, dtype=tf.float64)
            transport_matrix = transport(
                x,
                logw,
                eps_tensor,
                tf.constant(SCALING, dtype=tf.float64),
                tf.constant(THRESHOLD, dtype=tf.float64),
                tf.constant(MAX_ITER, dtype=tf.int32),
                tf.constant(N, dtype=tf.int32),
            )
            alpha, beta, _, _, iterations = sinkhorn_potentials(
                logw,
                scaled_x,
                uniform_log_weight,
                scaled_x,
                eps_tensor,
                tf.constant(SCALING, dtype=tf.float64),
                tf.constant(THRESHOLD, dtype=tf.float64),
                tf.constant(MAX_ITER, dtype=tf.int32),
            )
            from_potentials = transport_from_potentials(scaled_x, alpha, beta, eps_tensor, logw, float_n)
            rows.append({{
                "epsilon": float(eps),
                "transport_matrix": transport_matrix.numpy().astype(float).tolist(),
                "transport_from_potentials_matrix": from_potentials.numpy().astype(float).tolist(),
                "transported_particles": tf.linalg.matmul(transport_matrix, x).numpy().astype(float).tolist(),
                "transport_from_potentials_particles": tf.linalg.matmul(from_potentials, x).numpy().astype(float).tolist(),
                "alpha": alpha.numpy().astype(float).tolist(),
                "beta": beta.numpy().astype(float).tolist(),
                "iterations": int(iterations.numpy()),
                "row_sums": tf.reduce_sum(transport_matrix, axis=2).numpy().astype(float).tolist(),
                "column_sums": tf.reduce_sum(transport_matrix, axis=1).numpy().astype(float).tolist(),
            }})

        payload = {{
            "status": "executed",
            "dtype": "{FILTERFLOW_REFERENCE_DTYPE}",
            "python": os.sys.version.split()[0],
            "tensorflow": tf.__version__,
            "frozen_state": {{
                **capture,
                "particles": x.numpy().astype(float).tolist(),
                "log_weights": logw.numpy().astype(float).tolist(),
                "weights": tf.exp(logw).numpy().astype(float).tolist(),
                "cost_scale": scale.numpy().astype(float).reshape(-1).tolist(),
                "scaled_particles": scaled_x.numpy().astype(float).tolist(),
                "cost_matrix": cost_matrix.numpy().astype(float).tolist(),
                "epsilon0_filterflow_range": epsilon0_filterflow_range.numpy().astype(float).tolist(),
                "epsilon0_max_cost": epsilon0_max_cost.numpy().astype(float).tolist(),
            }},
            "reference_rows": rows,
        }}
        print("FILTERFLOW_COMPONENT_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_COMPONENT_JSON_END")
        """
    )


def _run_candidate_variants(reference: dict[str, Any]) -> list[dict[str, Any]]:
    x = tf.constant(reference["frozen_state"]["particles"], dtype=DTYPE)
    logw = tf.constant(reference["frozen_state"]["log_weights"], dtype=DTYPE)
    rows = []
    variants = [
        ("legacy_axis_row_epsilon0_max_cost", "row", "max_cost"),
        ("axis_column_epsilon0_max_cost", "column", "max_cost"),
        ("axis_row_epsilon0_filterflow_range", "row", "filterflow_range"),
        ("axis_column_epsilon0_filterflow_range", "column", "filterflow_range"),
    ]
    for ref_row in reference["reference_rows"]:
        epsilon = float(ref_row["epsilon"])
        reference_matrix = tf.constant(ref_row["transport_matrix"], dtype=DTYPE)
        reference_from_potentials = tf.constant(ref_row["transport_from_potentials_matrix"], dtype=DTYPE)
        reference_particles = tf.constant(ref_row["transported_particles"], dtype=DTYPE)
        reference_alpha = tf.constant(ref_row["alpha"], dtype=DTYPE)
        reference_beta = tf.constant(ref_row["beta"], dtype=DTYPE)
        for variant_id, logw_axis, epsilon0_mode in variants:
            candidate = _candidate_transport(
                x,
                logw,
                epsilon,
                logw_axis=logw_axis,
                epsilon0_mode=epsilon0_mode,
            )
            rows.append(
                {
                    "variant_id": variant_id,
                    "epsilon": epsilon,
                    "logw_axis": logw_axis,
                    "epsilon0_mode": epsilon0_mode,
                    "matrix_vs_filterflow_transport": _tensor_comparison(candidate["matrix"], reference_matrix),
                    "matrix_vs_filterflow_transport_from_potentials": _tensor_comparison(
                        candidate["matrix"],
                        reference_from_potentials,
                    ),
                    "transported_particles_vs_filterflow": _tensor_comparison(
                        candidate["transported_particles"],
                        reference_particles,
                    ),
                    "alpha_vs_filterflow": _tensor_comparison(candidate["alpha"], reference_alpha),
                    "beta_vs_filterflow": _tensor_comparison(candidate["beta"], reference_beta),
                    "iterations": candidate["iterations"],
                    "filterflow_iterations": ref_row["iterations"],
                    "epsilon0": _to_float_list(candidate["epsilon0"]),
                    "row_sum_max_abs_residual_vs_one": _float(
                        tf.reduce_max(tf.abs(tf.reduce_sum(candidate["matrix"], axis=2) - 1.0))
                    ),
                    "column_sum_max_abs_residual_vs_n_weights": _float(
                        tf.reduce_max(
                            tf.abs(
                                tf.reduce_sum(candidate["matrix"], axis=1)
                                - tf.cast(NUM_PARTICLES, DTYPE) * tf.exp(logw)
                            )
                        )
                    ),
                    "finite_matrix": _finite(candidate["matrix"]),
                    "finite_particles": _finite(candidate["transported_particles"]),
                }
            )
    return rows


def _candidate_transport(
    x: tf.Tensor,
    logw: tf.Tensor,
    epsilon: float,
    *,
    logw_axis: str,
    epsilon0_mode: str,
) -> dict[str, tf.Tensor | int]:
    scaled_x = _filterflow_scaled_x(x)
    cost = 0.5 * _pairwise_squared(scaled_x)
    uniform_log_weight = -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)) * tf.ones_like(logw)
    epsilon0 = _epsilon0(cost, scaled_x, epsilon0_mode)
    alpha, beta, iterations = _candidate_potentials(
        logw,
        uniform_log_weight,
        cost,
        epsilon,
        epsilon0,
    )
    matrix = _candidate_transport_from_potentials(
        cost,
        alpha,
        beta,
        epsilon,
        logw,
        logw_axis=logw_axis,
    )
    transported = tf.linalg.matmul(matrix, x)
    return {
        "matrix": matrix,
        "transported_particles": transported,
        "alpha": alpha,
        "beta": beta,
        "iterations": iterations,
        "epsilon0": epsilon0,
    }


def _candidate_potentials(
    log_alpha: tf.Tensor,
    log_beta: tf.Tensor,
    cost: tf.Tensor,
    epsilon: float,
    epsilon0: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, int]:
    batch_size = tf.shape(cost)[0]
    epsilon_target = tf.ones([batch_size], dtype=DTYPE) * tf.constant(epsilon, dtype=DTYPE)
    epsilon_running = tf.reshape(tf.cast(epsilon0, DTYPE), [batch_size])
    continue_flag = tf.ones([batch_size], dtype=tf.bool)
    scaling_factor = tf.constant(SCALING * SCALING, dtype=DTYPE)
    threshold = tf.constant(THRESHOLD, dtype=DTYPE)
    a = _softmin(epsilon_running, cost, log_alpha)
    b = _softmin(epsilon_running, cost, log_beta)
    i = 0
    while i < MAX_ITER - 1 and bool(tf.reduce_all(continue_flag).numpy()):
        running_col = epsilon_running[:, None]
        continue_col = continue_flag[:, None]
        at = tf.where(continue_col, _softmin(epsilon_running, cost, log_alpha + b / running_col), a)
        bt = tf.where(continue_col, _softmin(epsilon_running, cost, log_beta + a / running_col), b)
        a_new = 0.5 * (a + at)
        b_new = 0.5 * (b + bt)
        a_diff = tf.reduce_max(tf.abs(a_new - a), axis=1)
        b_diff = tf.reduce_max(tf.abs(b_new - b), axis=1)
        local_continue = tf.logical_or(a_diff > threshold, b_diff > threshold)
        new_epsilon = tf.maximum(epsilon_running * scaling_factor, epsilon_target)
        continue_flag = tf.logical_or(new_epsilon < epsilon_running, local_continue)
        a, b = a_new, b_new
        epsilon_running = new_epsilon
        i += 1
    epsilon_col = epsilon_target[:, None]
    final_a = _softmin(epsilon_target, cost, log_alpha + tf.stop_gradient(b) / epsilon_col)
    final_b = _softmin(epsilon_target, cost, log_beta + tf.stop_gradient(a) / epsilon_col)
    return final_a, final_b, i + 2


def _candidate_transport_from_potentials(
    cost: tf.Tensor,
    f: tf.Tensor,
    g: tf.Tensor,
    epsilon: float,
    logw: tf.Tensor,
    *,
    logw_axis: str,
) -> tf.Tensor:
    log_n = tf.math.log(tf.cast(NUM_PARTICLES, DTYPE))
    temp = (f[:, :, None] + g[:, None, :] - cost) / tf.constant(epsilon, dtype=DTYPE)
    temp = temp - tf.reduce_logsumexp(temp, axis=1, keepdims=True) + log_n
    if logw_axis == "column":
        temp = temp + logw[:, None, :]
    elif logw_axis == "row":
        temp = temp + logw[:, :, None]
    else:
        raise ValueError(logw_axis)
    return tf.exp(temp)


def _filterflow_scaled_x(x: tf.Tensor) -> tf.Tensor:
    dimension = tf.cast(tf.shape(x)[2], DTYPE)
    centered = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    std = tf.math.reduce_std(x, axis=1)
    diameter = tf.reduce_max(std, axis=1)
    diameter = tf.where(diameter == 0.0, tf.ones_like(diameter), diameter)
    scale = diameter * tf.sqrt(dimension)
    return centered / tf.stop_gradient(scale[:, None, None])


def _epsilon0(cost: tf.Tensor, scaled_x: tf.Tensor, mode: str) -> tf.Tensor:
    if mode == "filterflow_range":
        coordinate_range = tf.reduce_max(scaled_x, axis=[1, 2]) - tf.reduce_min(scaled_x, axis=[1, 2])
        return coordinate_range * coordinate_range
    if mode == "max_cost":
        return tf.reduce_max(cost, axis=[1, 2])
    raise ValueError(mode)


def _softmin(epsilon: tf.Tensor, cost: tf.Tensor, f: tf.Tensor) -> tf.Tensor:
    epsilon = tf.reshape(tf.cast(epsilon, DTYPE), [-1])
    temp = f[:, None, :] - cost / epsilon[:, None, None]
    return -epsilon[:, None] * tf.reduce_logsumexp(temp, axis=2)


def _pairwise_squared(x: tf.Tensor) -> tf.Tensor:
    diff = x[:, :, None, :] - x[:, None, :, :]
    return tf.reduce_sum(diff * diff, axis=3)


def _full_vs_from_potentials(reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for ref_row in reference["reference_rows"]:
        transport_matrix = tf.constant(ref_row["transport_matrix"], dtype=DTYPE)
        from_potentials = tf.constant(ref_row["transport_from_potentials_matrix"], dtype=DTYPE)
        transport_particles = tf.constant(ref_row["transported_particles"], dtype=DTYPE)
        from_potentials_particles = tf.constant(ref_row["transport_from_potentials_particles"], dtype=DTYPE)
        rows.append(
            {
                "epsilon": ref_row["epsilon"],
                "matrix": _tensor_comparison(transport_matrix, from_potentials),
                "transported_particles": _tensor_comparison(transport_particles, from_potentials_particles),
                "transport_row_sum_max_abs_residual_vs_one": _float(
                    tf.reduce_max(tf.abs(tf.reduce_sum(transport_matrix, axis=2) - 1.0))
                ),
                "transport_column_sum_max_abs_residual_vs_n_weights": _float(
                    tf.reduce_max(
                        tf.abs(
                            tf.reduce_sum(transport_matrix, axis=1)
                            - tf.cast(NUM_PARTICLES, DTYPE)
                            * tf.constant(reference["frozen_state"]["weights"], dtype=DTYPE)
                        )
                    )
                ),
            }
        )
    return rows


def _decision(
    reference: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    full_vs_from_potentials: list[dict[str, Any]],
) -> str:
    if reference["status"] != "executed":
        return "blocked_filterflow_component_not_executed"
    if reference["frozen_state"]["selected_ess"] > reference["frozen_state"]["threshold"]:
        return "blocked_not_a_resampling_state"
    if any(row["matrix"]["max_abs_diff"] > MATCH_TOLERANCE for row in full_vs_from_potentials):
        return "blocked_filterflow_transport_from_potentials_mismatch"
    exact_rows = [row for row in variant_rows if row["variant_id"] == "axis_column_epsilon0_filterflow_range"]
    legacy_rows = [row for row in variant_rows if row["variant_id"] == "legacy_axis_row_epsilon0_max_cost"]
    exact_ok = all(row["matrix_vs_filterflow_transport"]["max_abs_diff"] <= MATCH_TOLERANCE for row in exact_rows)
    legacy_ok = all(row["matrix_vs_filterflow_transport"]["max_abs_diff"] <= MATCH_TOLERANCE for row in legacy_rows)
    if exact_ok and not legacy_ok:
        return "transport_formula_mismatch_identified"
    if exact_ok and legacy_ok:
        return "transport_formula_legacy_already_matched"
    return "blocked_exact_transport_reconstruction_failed"


def _interpretation(variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in variant_rows:
        by_variant.setdefault(row["variant_id"], []).append(row)
    summary = {}
    for variant_id, rows in by_variant.items():
        summary[variant_id] = {
            "max_matrix_abs_diff": max(row["matrix_vs_filterflow_transport"]["max_abs_diff"] for row in rows),
            "max_particle_abs_diff": max(row["transported_particles_vs_filterflow"]["max_abs_diff"] for row in rows),
            "all_matrix_within_tolerance": all(
                row["matrix_vs_filterflow_transport"]["max_abs_diff"] <= MATCH_TOLERANCE for row in rows
            ),
            "all_particles_within_tolerance": all(
                row["transported_particles_vs_filterflow"]["max_abs_diff"] <= MATCH_TOLERANCE for row in rows
            ),
        }
    return {
        "summary_by_variant": summary,
        "likely_mismatch": (
            "The dominant mismatch is the log-weight axis in transport_from_potentials. "
            "The failed audit-only mirror used row-axis log weights; filterflow uses column-axis log "
            "weights, giving row sums near one and column sums near N times the source weights. "
            "The max-cost epsilon-start variant changes potentials and iteration counts on this frozen "
            "state, but still matches the transported particles once the log-weight axis is corrected."
        ),
    }


def _tensor_comparison(a: tf.Tensor, b: tf.Tensor) -> dict[str, float]:
    diff = tf.cast(a, DTYPE) - tf.cast(b, DTYPE)
    return {
        "max_abs_diff": _float(tf.reduce_max(tf.abs(diff))),
        "rmse": _float(tf.sqrt(tf.reduce_mean(diff * diff))),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    validate_filterflow_reference_status(payload["filterflow_status"])
    if payload["frozen_state"]["selected_ess"] > payload["frozen_state"]["threshold"]:
        raise RuntimeError("frozen state did not trigger resampling")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if payload["decision"] != "transport_formula_mismatch_identified":
        raise RuntimeError(f"unexpected decision: {payload['decision']}")
    exact_rows = [
        row
        for row in payload["candidate_variants"]
        if row["variant_id"] == "axis_column_epsilon0_filterflow_range"
    ]
    if len(exact_rows) != len(EPSILONS):
        raise RuntimeError("missing exact variant rows")
    if any(row["matrix_vs_filterflow_transport"]["max_abs_diff"] > MATCH_TOLERANCE for row in exact_rows):
        raise RuntimeError("exact filterflow formula reconstruction did not match")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow Transport Component Audit

## Decision

`{payload['decision']}`

## Frozen State

- Time index: `{payload['frozen_state']['time_index']}`
- Batch index: `{payload['frozen_state']['batch_index']}`
- Selected ESS: `{payload['frozen_state']['selected_ess']:.6g}`
- ESS threshold: `{payload['frozen_state']['threshold']:.6g}`
- Triggered batches: `{payload['frozen_state']['triggered_count']}`

## Filterflow Transport Versus `transport_from_potentials`

{_full_vs_from_potentials_table(payload['filterflow_transport_vs_transport_from_potentials'])}

## Candidate Variants

{_variant_table(payload['candidate_variants'])}

## Interpretation

{payload['interpretation']['likely_mismatch']}

## Non-Implications

{_non_implications_markdown()}
"""


def _full_vs_from_potentials_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| eps | matrix max diff | particle max diff | row residual | column residual |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {eps:.2f} | {md:.3e} | {pd:.3e} | {rr:.3e} | {cr:.3e} |".format(
                eps=row["epsilon"],
                md=row["matrix"]["max_abs_diff"],
                pd=row["transported_particles"]["max_abs_diff"],
                rr=row["transport_row_sum_max_abs_residual_vs_one"],
                cr=row["transport_column_sum_max_abs_residual_vs_n_weights"],
            )
        )
    return "\n".join(lines)


def _variant_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| variant | eps | matrix max diff | particle max diff | alpha diff | beta diff | iterations | row residual | column residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{variant}` | {eps:.2f} | {md:.3e} | {pd:.3e} | {ad:.3e} | {bd:.3e} | {it}/{fit} | {rr:.3e} | {cr:.3e} |".format(
                variant=row["variant_id"],
                eps=row["epsilon"],
                md=row["matrix_vs_filterflow_transport"]["max_abs_diff"],
                pd=row["transported_particles_vs_filterflow"]["max_abs_diff"],
                ad=row["alpha_vs_filterflow"]["max_abs_diff"],
                bd=row["beta_vs_filterflow"]["max_abs_diff"],
                it=row["iterations"],
                fit=row["filterflow_iterations"],
                rr=row["row_sum_max_abs_residual_vs_one"],
                cr=row["column_sum_max_abs_residual_vs_n_weights"],
            )
        )
    return "\n".join(lines)


def _filterflow_status() -> dict[str, Any]:
    return {
        "branch": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "HEAD"]),
        "status": _git(["git", "-C", str(FILTERFLOW_PATH), "status", "--short", "--branch"]),
        "diff_summary": _git(["git", "-C", str(FILTERFLOW_PATH), "diff", "--stat"]) or "clean",
        "upstream_base": UPSTREAM_FILTERFLOW_COMMIT,
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


def _to_float_list(value: tf.Tensor) -> list[float]:
    return [float(item) for item in tf.reshape(tf.cast(value, tf.float64), [-1]).numpy().tolist()]


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, tf.float64).numpy())


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, DTYPE))).numpy())


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No HMC readiness is concluded.",
        "No posterior correctness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No claim that finite relaxed OT is categorical PF is concluded.",
        "No claim that the BayesFilter outer matched audit runner is fixed is concluded.",
        "No claim that patched filterflow is untouched upstream code is concluded.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


if __name__ == "__main__":
    raise SystemExit(main())
