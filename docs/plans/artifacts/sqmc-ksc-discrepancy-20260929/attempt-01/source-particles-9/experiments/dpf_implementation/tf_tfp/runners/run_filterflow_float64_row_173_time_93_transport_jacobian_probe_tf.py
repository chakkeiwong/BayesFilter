"""Probe row-173/time-93 transport-matrix Jacobian against float64 FilterFlow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
import textwrap
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_row_173_vjp_decomposition_tf as row_vjp,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r3_float64_trace_replay_tf as r3,
)
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
    FILTERFLOW_BRANCH_MARKER,
    reference_policy,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-time-93-transport-jacobian-probe-plan-2026-06-03.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-time-93-transport-jacobian-probe-result-2026-06-03.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_row_173_time_93_transport_jacobian_probe_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-row-173-time-93-transport-jacobian-probe-2026-06-03.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
FROZEN_INPUT_PATH = OUTPUT_DIR / "dpf_filterflow_float64_row_173_time_93_transport_jacobian_input_2026-06-03.json"

TARGET_TIME_INDEX = 93
EPSILON = row_vjp.EPSILON
SCALING = row_vjp.SCALING
CONVERGENCE_THRESHOLD = row_vjp.CONVERGENCE_THRESHOLD
MAX_ITERATIONS = row_vjp.MAX_ITERATIONS
NUM_PARTICLES = row_vjp.NUM_PARTICLES
GRADIENT_TOLERANCE = 2e-4


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
    payload["reproducibility_digest"] = stable_digest(_compact_for_digest(payload))
    write_json(JSON_PATH, payload)
    markdown = _markdown(payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    reference_status = r3._filterflow_status()
    validate_filterflow_reference_status(reference_status, marker_path=FILTERFLOW_MARKER_PATH)
    frozen = _frozen_time_93_inputs()
    filterflow = _filterflow_transport_jacobian_subprocess(frozen)
    if filterflow.get("status") != "executed":
        return _blocked_payload(reference_status, frozen, filterflow)
    bayesfilter = _bayesfilter_transport_jacobian(frozen)
    comparison = _compare(filterflow, bayesfilter)
    decision = _decision(comparison)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_frozen_transport_jacobian_probe",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "model_contract": _model_contract(),
        "frozen_source_summary": frozen["summary"],
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "comparison": comparison,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_time_93_transport_jacobian_probe_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _blocked_payload(
    reference_status: dict[str, Any],
    frozen: dict[str, Any],
    filterflow: dict[str, Any],
) -> dict[str, Any]:
    blocker = filterflow.get("blocker", "unknown FilterFlow transport-Jacobian blocker")
    comparison = {"status": "blocked", "blocker": blocker}
    return {
        "decision": "filterflow_float64_row_173_time_93_transport_jacobian_blocked",
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "blocker": blocker,
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "model_contract": _model_contract(),
        "frozen_source_summary": frozen["summary"],
        "filterflow_probe": filterflow,
        "bayesfilter_probe": None,
        "comparison": comparison,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_time_93_transport_jacobian_probe_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(
            "filterflow_float64_row_173_time_93_transport_jacobian_blocked",
            comparison,
        ),
        "non_implications": _non_implications(),
    }


def _frozen_time_93_inputs() -> dict[str, Any]:
    config = row_vjp.RunConfig(
        target_time_index=TARGET_TIME_INDEX,
        tag="row-173-time-93-jacobian-source",
        plan_path=row_vjp.PLAN_PATH,
        result_path=row_vjp.RESULT_PATH,
        json_path=row_vjp.JSON_PATH,
        report_path=row_vjp.REPORT_PATH,
    )
    filterflow_vjp = row_vjp._filterflow_vjp_subprocess(config)
    if filterflow_vjp.get("status") != "executed":
        raise RuntimeError(filterflow_vjp.get("blocker", "FilterFlow VJP source failed"))
    particles = filterflow_vjp["value_tensors"]["pre_particles"]
    log_weights = filterflow_vjp["value_tensors"]["pre_log_weights"]
    upstream = filterflow_vjp["gradient_tensors"]["transport_matrix"]
    frozen = {
        "particles": particles,
        "log_weights": log_weights,
        "transport_upstream": upstream,
        "summary": {
            "source": "filterflow_row_173_time_93_vjp_subprocess",
            "target_time_index": TARGET_TIME_INDEX,
            "theta": row_vjp.THETA,
            "transport_upstream_clip_fraction": _clip_fraction_json(upstream),
            "particles_shape": _shape_json(particles),
            "log_weights_shape": _shape_json(log_weights),
            "upstream_shape": _shape_json(upstream),
        },
    }
    write_json(FROZEN_INPUT_PATH, frozen)
    return frozen


def _filterflow_transport_jacobian_subprocess(frozen: dict[str, Any]) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing FilterFlow env python: {FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_probe_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "FilterFlow transport-Jacobian subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_TRANSPORT_JACOBIAN_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_TRANSPORT_JACOBIAN_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "FilterFlow transport-Jacobian JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(
        completed.stdout[start + len("FILTERFLOW_TRANSPORT_JACOBIAN_JSON_BEGIN"):end].strip()
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_probe_script() -> str:
    return textwrap.dedent(
        f"""
        import inspect
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import tensorflow as tf

        from filterflow.resampling.differentiable.regularized_transport.plan import transport
        from filterflow.resampling.differentiable.regularized_transport.sinkhorn import sinkhorn_potentials
        from filterflow.resampling.differentiable.regularized_transport.utils import cost, diameter

        DTYPE = tf.float64
        INPUT_PATH = {str(FROZEN_INPUT_PATH)!r}
        EPSILON = tf.constant({EPSILON!r}, dtype=DTYPE)
        SCALING = tf.constant({SCALING!r}, dtype=DTYPE)
        THRESHOLD = tf.constant({CONVERGENCE_THRESHOLD!r}, dtype=DTYPE)
        MAX_ITER = tf.constant({MAX_ITERATIONS}, dtype=tf.int32)
        N = tf.constant({NUM_PARTICLES}, dtype=tf.int32)

        def tensor(value):
            return tf.convert_to_tensor(value, dtype=DTYPE)

        def to_json(value):
            return tf.cast(value, tf.float64).numpy().tolist()

        def field(value):
            return {{
                "shape": [int(v) for v in value.shape],
                "finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
                "max_abs": float(tf.reduce_max(tf.abs(value)).numpy()),
                "sum": float(tf.reduce_sum(value).numpy()),
            }}

        def internals(x, logw):
            dtype = x.dtype
            float_n = tf.cast(N, dtype)
            uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(logw)
            dimension = tf.cast(x.shape[-1], dtype)
            centered_x = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
            diameter_value = diameter(x, x)
            scale = tf.reshape(diameter_value, [-1, 1, 1]) * tf.sqrt(dimension)
            scaled_x = centered_x / tf.stop_gradient(scale)
            alpha, beta, _ax, _by, iterations = sinkhorn_potentials(
                logw,
                scaled_x,
                uniform_log_weight,
                scaled_x,
                EPSILON,
                SCALING,
                THRESHOLD,
                MAX_ITER,
            )
            cost_matrix = cost(scaled_x, scaled_x)
            fg = tf.expand_dims(alpha, 2) + tf.expand_dims(beta, 1)
            temp = (fg - cost_matrix) / EPSILON
            temp = temp - tf.reduce_logsumexp(temp, 1, keepdims=True) + tf.math.log(float_n)
            temp = temp + tf.expand_dims(logw, 1)
            manual_transport = tf.exp(temp)
            custom_transport = transport(x, logw, EPSILON, SCALING, THRESHOLD, MAX_ITER, N)
            return {{
                "centered_particles": centered_x,
                "diameter_value": diameter_value,
                "scale": scale,
                "scaled_particles": scaled_x,
                "alpha": alpha,
                "beta": beta,
                "iterations": tf.cast(iterations, DTYPE),
                "cost_matrix": cost_matrix,
                "manual_transport_matrix": manual_transport,
                "custom_transport_matrix": custom_transport,
            }}

        with open(INPUT_PATH, "r", encoding="utf-8") as handle:
            frozen = json.load(handle)
        particles = tensor(frozen["particles"])
        log_weights = tensor(frozen["log_weights"])
        upstream = tensor(frozen["transport_upstream"])
        clipped_upstream = tf.clip_by_value(upstream, -1.0, 1.0)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([particles, log_weights])
            fields = internals(particles, log_weights)
            custom_transport_matrix = fields["custom_transport_matrix"]
            manual_transport_matrix = fields["manual_transport_matrix"]
        custom_grad_particles = tape.gradient(custom_transport_matrix, particles, upstream)
        custom_grad_log_weights = tape.gradient(custom_transport_matrix, log_weights, upstream)
        manual_grad_particles = tape.gradient(manual_transport_matrix, particles, clipped_upstream)
        manual_grad_log_weights = tape.gradient(manual_transport_matrix, log_weights, clipped_upstream)
        del tape

        payload = {{
            "status": "executed",
            "backend": "executable_filterflow_subprocess",
            "settings": {{
                "epsilon": float(EPSILON.numpy()),
                "scaling": float(SCALING.numpy()),
                "convergence_threshold": float(THRESHOLD.numpy()),
                "max_iter": int(MAX_ITER.numpy()),
                "num_particles": int(N.numpy()),
                "dtype": "float64",
            }},
            "summaries": {{
                name: field(value)
                for name, value in fields.items()
            }},
            "gradient_summaries": {{
                "custom_grad_particles": field(custom_grad_particles),
                "custom_grad_log_weights": field(custom_grad_log_weights),
                "manual_grad_particles": field(manual_grad_particles),
                "manual_grad_log_weights": field(manual_grad_log_weights),
            }},
            "tensors": {{
                "centered_particles": to_json(fields["centered_particles"]),
                "diameter_value": to_json(fields["diameter_value"]),
                "scale": to_json(fields["scale"]),
                "scaled_particles": to_json(fields["scaled_particles"]),
                "alpha": to_json(fields["alpha"]),
                "beta": to_json(fields["beta"]),
                "iterations": to_json(fields["iterations"]),
                "cost_matrix": to_json(fields["cost_matrix"]),
                "manual_transport_matrix": to_json(fields["manual_transport_matrix"]),
                "custom_transport_matrix": to_json(fields["custom_transport_matrix"]),
                "custom_grad_particles": to_json(custom_grad_particles),
                "custom_grad_log_weights": to_json(custom_grad_log_weights),
                "manual_grad_particles": to_json(manual_grad_particles),
                "manual_grad_log_weights": to_json(manual_grad_log_weights),
            }},
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
        }}
        print("FILTERFLOW_TRANSPORT_JACOBIAN_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_TRANSPORT_JACOBIAN_JSON_END")
        """
    )


def _bayesfilter_transport_jacobian(frozen: dict[str, Any]) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        particles = tf.convert_to_tensor(frozen["particles"], dtype=DTYPE)
        log_weights = tf.convert_to_tensor(frozen["log_weights"], dtype=DTYPE)
        upstream = tf.convert_to_tensor(frozen["transport_upstream"], dtype=DTYPE)
        clipped_upstream = tf.clip_by_value(upstream, -1.0, 1.0)
        eps = tf.constant(EPSILON, DTYPE)
        scaling = tf.constant(SCALING, DTYPE)
        threshold = tf.constant(CONVERGENCE_THRESHOLD, DTYPE)
        max_iter = tf.constant(MAX_ITERATIONS, tf.int32)
        num_particles = tf.shape(particles)[1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([particles, log_weights])
            fields = _bayesfilter_internals(
                particles,
                log_weights,
                eps,
                scaling,
                threshold,
                max_iter,
                num_particles,
            )
            transport_matrix = fields["transport_matrix"]
        grad_particles = tape.gradient(transport_matrix, particles, clipped_upstream)
        grad_log_weights = tape.gradient(transport_matrix, log_weights, clipped_upstream)
        del tape
        tensors = {
            **{name: _json(value) for name, value in fields.items()},
            "manual_grad_particles": _json(grad_particles),
            "manual_grad_log_weights": _json(grad_log_weights),
        }
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "settings": {
                "epsilon": EPSILON,
                "scaling": SCALING,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iter": MAX_ITERATIONS,
                "num_particles": NUM_PARTICLES,
                "dtype": "float64",
            },
            "summaries": {name: _field(value) for name, value in fields.items()},
            "gradient_summaries": {
                "manual_grad_particles": _field(grad_particles),
                "manual_grad_log_weights": _field(grad_log_weights),
            },
            "tensors": tensors,
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_internals(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    eps: tf.Tensor,
    scaling: tf.Tensor,
    threshold: tf.Tensor,
    max_iter: tf.Tensor,
    num_particles: tf.Tensor,
) -> dict[str, tf.Tensor]:
    centered = particles - tf.stop_gradient(tf.reduce_mean(particles, axis=1, keepdims=True))
    scale = annealed_transport_tf._filterflow_scale(particles)
    scaled = centered / tf.stop_gradient(scale[:, None, None])
    float_n = tf.cast(num_particles, DTYPE)
    uniform_log = -tf.math.log(float_n) * tf.ones_like(log_weights)
    alpha, beta, _ax, _by, iterations = annealed_transport_tf._filterflow_exact_sinkhorn_potentials(
        log_weights,
        scaled,
        uniform_log,
        scaled,
        eps,
        scaling,
        threshold,
        max_iter,
    )
    cost_matrix = annealed_transport_tf._filterflow_exact_cost(scaled, scaled)
    transport_matrix = annealed_transport_tf._filterflow_exact_transport_from_potentials(
        scaled,
        alpha,
        beta,
        eps,
        log_weights,
        float_n,
    )
    return {
        "centered_particles": centered,
        "scale": scale[:, None, None],
        "scaled_particles": scaled,
        "alpha": alpha,
        "beta": beta,
        "iterations": tf.cast(iterations, DTYPE),
        "cost_matrix": cost_matrix,
        "transport_matrix": transport_matrix,
    }


def _compare(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    field_map = {
        "centered_particles": ("centered_particles", "centered_particles"),
        "scale": ("scale", "scale"),
        "scaled_particles": ("scaled_particles", "scaled_particles"),
        "alpha": ("alpha", "alpha"),
        "beta": ("beta", "beta"),
        "iterations": ("iterations", "iterations"),
        "cost_matrix": ("cost_matrix", "cost_matrix"),
        "transport_matrix": ("custom_transport_matrix", "transport_matrix"),
        "manual_transport_matrix": ("manual_transport_matrix", "transport_matrix"),
        "grad_particles": ("custom_grad_particles", "manual_grad_particles"),
        "manual_grad_particles": ("manual_grad_particles", "manual_grad_particles"),
        "grad_log_weights": ("custom_grad_log_weights", "manual_grad_log_weights"),
        "manual_grad_log_weights": ("manual_grad_log_weights", "manual_grad_log_weights"),
    }
    for row_name, (ff_name, bf_name) in field_map.items():
        rows[row_name] = _tensor_delta(
            filterflow["tensors"][ff_name],
            bayesfilter["tensors"][bf_name],
        )
    first_delta = _first_delta(rows)
    status = "compared"
    return {
        "status": status,
        "first_delta_over_tolerance": first_delta,
        "rows": rows,
        "interpretation": _interpret(first_delta, rows),
    }


def _tensor_delta(left: Any, right: Any) -> dict[str, Any]:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    if len(left_flat) != len(right_flat):
        return {
            "shape_match": False,
            "finite": False,
            "max_abs_delta": None,
            "sum_delta": None,
            "left_max_abs": None,
            "right_max_abs": None,
        }
    deltas = [float(lhs) - float(rhs) for lhs, rhs in zip(left_flat, right_flat, strict=True)]
    left_values = [float(value) for value in left_flat]
    right_values = [float(value) for value in right_flat]
    return {
        "shape_match": True,
        "finite": all(tf.math.is_finite(tf.constant(value, DTYPE)).numpy() for value in left_values + right_values),
        "max_abs_delta": max(abs(value) for value in deltas) if deltas else 0.0,
        "sum_delta": sum(deltas),
        "left_max_abs": max(abs(value) for value in left_values) if left_values else 0.0,
        "right_max_abs": max(abs(value) for value in right_values) if right_values else 0.0,
    }


def _flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]


def _first_delta(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for name, row in rows.items():
        if (
            not row["shape_match"]
            or not row["finite"]
            or row["max_abs_delta"] is None
            or row["max_abs_delta"] > GRADIENT_TOLERANCE
        ):
            return {"status": "delta", "field": name, "row": row, "tolerance": GRADIENT_TOLERANCE}
    return {"status": "no_delta", "tolerance": GRADIENT_TOLERANCE}


def _interpret(first_delta: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    if first_delta["status"] == "no_delta":
        return "transport_jacobian_matches_on_recorded_fields"
    field = first_delta["field"]
    if field in {"grad_particles", "manual_grad_particles"}:
        return "implicit_transport_particle_jacobian_difference"
    if rows["manual_grad_particles"]["max_abs_delta"] <= GRADIENT_TOLERANCE:
        return "custom_gradient_difference_not_raw_transport_arithmetic"
    return f"first_recorded_transport_field_difference_{field}"


def _decision(comparison: dict[str, Any]) -> str:
    if comparison.get("status") != "compared":
        return "filterflow_float64_row_173_time_93_transport_jacobian_blocked"
    if comparison["first_delta_over_tolerance"]["status"] == "no_delta":
        return "filterflow_float64_row_173_time_93_transport_jacobian_match"
    return "filterflow_float64_row_173_time_93_transport_jacobian_difference_localized"


def _field(value: tf.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(dim) for dim in value.shape],
        "finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
        "max_abs": _float(tf.reduce_max(tf.abs(value))),
        "sum": _float(tf.reduce_sum(value)),
    }


def _json(value: tf.Tensor) -> Any:
    return tf.cast(value, DTYPE).numpy().tolist()


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _clip_fraction_json(value: Any) -> float:
    flat = _flatten(value)
    return sum(1.0 for item in flat if abs(item) > 1.0) / max(1, len(flat))


def _shape_json(value: Any) -> list[int]:
    shape = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else None
    return shape


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _compact_side(side: dict[str, Any] | None) -> dict[str, Any] | None:
    if side is None or side.get("status") != "executed":
        return side
    return {
        "status": side["status"],
        "backend": side["backend"],
        "settings": side["settings"],
        "summaries": side["summaries"],
        "gradient_summaries": side["gradient_summaries"],
        "cpu_only_manifest": side["cpu_only_manifest"],
        "stderr_excerpt": side.get("stderr_excerpt", ""),
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "primary_comparator": "local executable float64 FilterFlow checkout",
        "primary_question": "frozen row-173/time-93 transport Jacobian agreement",
        "primary_pass": "transport matrix particle/log-weight gradients agree under same upstream adjoint",
        "diagnostic_tolerance": GRADIENT_TOLERANCE,
        "correctness_status": "difference audit only",
        "mathematical_correctness": "not_concluded",
    }


def _model_contract() -> dict[str, Any]:
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "mesh_index": row_vjp.MESH_INDEX,
        "theta": row_vjp.THETA,
        "target_time_index": TARGET_TIME_INDEX,
        "epsilon": EPSILON,
        "scaling": SCALING,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "max_iter": MAX_ITERATIONS,
        "num_particles": NUM_PARTICLES,
        "dtype": "float64",
    }


def _decision_table(decision: str, comparison: dict[str, Any]) -> list[dict[str, str]]:
    if decision == "filterflow_float64_row_173_time_93_transport_jacobian_match":
        primary = "transport Jacobian rows match on recorded fields"
        next_action = "look outside transport Jacobian only if full-path residual remains"
        veto = "none"
    elif decision == "filterflow_float64_row_173_time_93_transport_jacobian_difference_localized":
        primary = f"first delta: {comparison.get('first_delta_over_tolerance')}"
        next_action = "patch only after the differing executable FilterFlow arithmetic is identified"
        veto = "frozen value/source path stayed valid"
    else:
        primary = f"blocked: {comparison.get('blocker')}"
        next_action = "resolve blocker before interpreting transport Jacobian"
        veto = "blocked"
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": veto,
            "main_uncertainty": "single frozen row/time transport Jacobian probe",
            "next_justified_action": next_action,
            "not_concluded": ", ".join(_non_implications()),
        }
    ]


def _non_implications() -> list[str]:
    return [
        "correctness of either implementation",
        "analytic gradient correctness",
        "posterior correctness",
        "production readiness",
        "public API readiness",
        "HMC readiness",
        "general nonlinear-SSM validity",
        "DSGE/NAWM validation",
        "monograph claim",
    ]


def _markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Result: Row 173 Time 93 Transport-Jacobian Probe",
            "",
            "## Decision",
            "",
            f"`{payload['decision']}`",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                "| {decision} | {primary_criterion_status} | {veto_diagnostic_status} | {main_uncertainty} | {next_justified_action} | {not_concluded} |".format(
                    **row
                )
                for row in payload["decision_table"]
            ],
            "",
            "## Model Contract",
            "",
            _json_block(payload["model_contract"]),
            "",
            "## Comparison",
            "",
            _json_block(payload["comparison"]),
            "",
            "## Frozen Source",
            "",
            _json_block(payload["frozen_source_summary"]),
            "",
            "## FilterFlow Probe",
            "",
            _json_block(payload["filterflow_probe"]),
            "",
            "## BayesFilter Probe",
            "",
            _json_block(payload["bayesfilter_probe"]),
            "",
            "## Verification",
            "",
            "- CPU-only manifest recorded `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import.",
            "- This artifact is a difference audit only.",
        ]
    )


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _validate_payload(payload: dict[str, Any]) -> None:
    required = [
        "decision",
        "comparison",
        "filterflow_probe",
        "bayesfilter_probe",
        "frozen_source_summary",
        "run_manifest",
    ]
    for key in required:
        if key not in payload:
            raise ValueError(f"missing payload key: {key}")
    if payload["comparison"].get("status") not in {"compared", "blocked"}:
        raise ValueError("unexpected comparison status")


def _compact_for_digest(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    compact.pop("reproducibility_digest", None)
    if "run_manifest" in compact:
        manifest = dict(compact["run_manifest"])
        manifest.pop("wall_time_seconds", None)
        compact["run_manifest"] = manifest
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
