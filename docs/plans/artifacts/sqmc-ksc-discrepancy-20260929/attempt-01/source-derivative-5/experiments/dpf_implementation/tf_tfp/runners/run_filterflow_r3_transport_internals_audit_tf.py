"""Selected-row R3 RegularisedTransform internals audit.

This is a difference audit against the executable filterflow reference. It
does not claim either implementation is mathematically correct.
"""

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
    run_filterflow_r3_proposal_trace_replay_tf as r3,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-r3-transport-internals-audit-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-r3-transport-internals-audit-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_r3_transport_internals_audit_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-r3-transport-internals-audit-2026-06-03.md"
SELECTED_ROWS = (7, 16, 43, 79)
FIELD_ORDER = (
    "input_particles",
    "input_log_weights",
    "centered_particles",
    "diameter_value",
    "scale",
    "scaled_particles",
    "cost_xy",
    "cost_yx",
    "cost_xx",
    "cost_yy",
    "sinkhorn_scale",
    "epsilon_0",
    "scaling_factor",
    "alpha",
    "beta",
    "iterations",
    "transport_matrix",
    "transported_particles",
)
TRACE_TOLERANCE = r3.TRACE_TOLERANCE


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
    markdown = _markdown(payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    initial_fingerprint = continuation._filterflow_fingerprint()
    filterflow_trace = r3._filterflow_trace_subprocess()
    if filterflow_trace.get("status") != "executed":
        return _blocked_payload(
            "filterflow_r3_transport_internals_filterflow_blocker",
            filterflow_trace.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow_trace,
        )
    if not filterflow_trace["trace_validation"]["official_trace_match"]:
        return _blocked_payload(
            "filterflow_r3_transport_internals_trace_validation_failed",
            "external R3 trace did not reproduce official filterflow",
            initial_fingerprint,
            filterflow_trace,
        )
    rows = [
        filterflow_trace["trace_rows"][index]
        for index in SELECTED_ROWS
    ]
    filterflow_internals = _filterflow_internals_subprocess(
        rows,
        filterflow_trace["model"],
    )
    if filterflow_internals.get("status") != "executed":
        return _blocked_payload(
            "filterflow_r3_transport_internals_filterflow_blocker",
            filterflow_internals.get("blocker", "unknown filterflow internals blocker"),
            initial_fingerprint,
            filterflow_trace,
            filterflow_internals,
        )
    bayesfilter_internals = _bayesfilter_internals(rows, filterflow_trace["model"])
    comparison = _compare_internals(filterflow_internals, bayesfilter_internals)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(comparison, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "Which selected-row 2D RegularisedTransform internal first "
            "differs between BayesFilter and executable filterflow?"
        ),
        "evidence_contract": {
            "primary_comparator": "local executable patched filterflow checkout",
            "primary_question": "cross_implementation_difference_only",
            "mathematical_correctness": "not_concluded",
            "selected_rows": list(SELECTED_ROWS),
        },
        "model_contract": r3._model_contract(),
        "selected_rows": list(SELECTED_ROWS),
        "field_order": list(FIELD_ORDER),
        "trace_validation": filterflow_trace["trace_validation"],
        "filterflow_internals": _compact_internals(filterflow_internals),
        "bayesfilter_internals": _compact_internals(bayesfilter_internals),
        "comparison": comparison,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": {"trace_replay": TRACE_TOLERANCE},
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r3_transport_internals_audit_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications()
        + [
            "No correctness claim is made for either implementation.",
            "No smoothness-surface gradient correctness is concluded.",
            "No transport residual correctness claim is concluded.",
        ],
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
    filterflow_trace: dict[str, Any],
    filterflow_internals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "blocker": blocker,
        "trace_validation": filterflow_trace.get("trace_validation"),
        "filterflow_internals": filterflow_internals,
        "comparison": {"status": "blocked", "blocker": blocker},
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r3_transport_internals_audit_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _filterflow_internals_subprocess(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, Any]:
    if not r3.FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {r3.FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(r3.FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(r3.FILTERFLOW_ENV_PYTHON), "-c", _filterflow_internals_script()],
        input=json.dumps({"rows": rows, "model": model}),
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow internals subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_R3_INTERNALS_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_R3_INTERNALS_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow internals JSON sentinels missing",
            "stdout_excerpt": stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(stdout[start + len("FILTERFLOW_R3_INTERNALS_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_internals_script() -> str:
    return textwrap.dedent(
        """
        import inspect
        import json
        import os
        import sys

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import tensorflow as tf
        from filterflow.resampling.differentiable.regularized_transport.plan import (
            transport,
            transport_from_potentials,
        )
        from filterflow.resampling.differentiable.regularized_transport.sinkhorn import (
            sinkhorn_potentials,
        )
        from filterflow.resampling.differentiable.regularized_transport.utils import (
            cost,
            diameter,
            max_min,
        )

        data = json.loads(sys.stdin.read())
        rows = data["rows"]
        model = data["model"]

        def to_json(tensor):
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, tf.float64).numpy())

        def max_abs(left, right):
            left = tf.cast(left, tf.float64)
            right = tf.cast(right, tf.float64)
            return scalar(tf.reduce_max(tf.abs(left - right)))

        def internals(row):
            x = tf.constant(row["pre_particles"], dtype=tf.float32)
            logw = tf.constant(row["pre_log_weights"], dtype=tf.float32)
            eps = tf.cast(model["epsilon"], float)
            scaling = tf.cast(model["scaling"], float)
            threshold = tf.cast(model["convergence_threshold"], float)
            max_iter = tf.cast(model["max_iterations"], tf.int32)
            n = x.shape[1]
            float_n = tf.cast(n, float)
            log_n = tf.math.log(float_n)
            uniform_log_weight = -log_n * tf.ones_like(logw)
            dimension = tf.cast(x.shape[-1], tf.float32)
            centered_x = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
            diameter_value = diameter(x, x)
            scale = tf.reshape(diameter_value, [-1, 1, 1]) * tf.sqrt(dimension)
            scaled_x = centered_x / tf.stop_gradient(scale)
            cost_xy = cost(scaled_x, tf.stop_gradient(scaled_x))
            cost_yx = cost(scaled_x, tf.stop_gradient(scaled_x))
            cost_xx = cost(scaled_x, tf.stop_gradient(scaled_x))
            cost_yy = cost(scaled_x, tf.stop_gradient(scaled_x))
            sinkhorn_scale = tf.stop_gradient(max_min(scaled_x, scaled_x))
            epsilon_0 = sinkhorn_scale ** 2
            scaling_factor = scaling ** 2
            alpha, beta, _, _, iterations = sinkhorn_potentials(
                logw,
                scaled_x,
                uniform_log_weight,
                scaled_x,
                eps,
                scaling,
                threshold,
                max_iter,
            )
            transport_matrix = transport_from_potentials(
                scaled_x,
                alpha,
                beta,
                eps,
                logw,
                float_n,
            )
            transported = tf.linalg.matmul(transport_matrix, x)
            direct_transport = transport(
                x,
                logw,
                eps,
                scaling,
                threshold,
                max_iter,
                n,
            )
            return {
                "time_index": row["time_index"],
                "input_particles": to_json(x),
                "input_log_weights": to_json(logw),
                "centered_particles": to_json(centered_x),
                "diameter_value": to_json(diameter_value),
                "scale": to_json(scale),
                "scaled_particles": to_json(scaled_x),
                "cost_xy": to_json(cost_xy),
                "cost_yx": to_json(cost_yx),
                "cost_xx": to_json(cost_xx),
                "cost_yy": to_json(cost_yy),
                "sinkhorn_scale": to_json(sinkhorn_scale),
                "epsilon_0": to_json(epsilon_0),
                "scaling_factor": to_json(scaling_factor),
                "alpha": to_json(alpha),
                "beta": to_json(beta),
                "iterations": to_json(iterations),
                "transport_matrix": to_json(transport_matrix),
                "transported_particles": to_json(transported),
                "direct_transport_matrix_delta": max_abs(
                    direct_transport,
                    transport_matrix,
                ),
                "trace_transport_matrix_delta": max_abs(
                    tf.constant(row["transport_matrix"], dtype=tf.float32),
                    transport_matrix,
                ),
                "trace_post_resample_particles_delta": max_abs(
                    tf.constant(row["post_resample_particles"], dtype=tf.float32),
                    transported,
                ),
                "finite_values": bool(
                    tf.reduce_all(tf.math.is_finite(transport_matrix)).numpy()
                    and tf.reduce_all(tf.math.is_finite(transported)).numpy()
                ),
            }

        payload = {
            "status": "executed",
            "backend": "executable_filterflow_subprocess",
            "rows": [internals(row) for row in rows],
            "cpu_only_manifest": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            },
        }
        print("FILTERFLOW_R3_INTERNALS_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_R3_INTERNALS_JSON_END")
        """
    )


def _bayesfilter_internals(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = tf.float32
    try:
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "rows": [_bayesfilter_row(row, model) for row in rows],
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_row(row: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    x = tf.constant(row["pre_particles"], dtype=tf.float32)
    logw = tf.constant(row["pre_log_weights"], dtype=tf.float32)
    eps = tf.constant(model["epsilon"], dtype=tf.float32)
    scaling = tf.constant(model["scaling"], dtype=tf.float32)
    threshold = tf.constant(model["convergence_threshold"], dtype=tf.float32)
    max_iter = tf.constant(model["max_iterations"], dtype=tf.int32)
    n = tf.shape(x)[1]
    float_n = tf.cast(n, tf.float32)
    log_n = tf.math.log(float_n)
    uniform_log_weight = -log_n * tf.ones_like(logw)
    centered_x = x - tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
    diameter_value = _filterflow_diameter(x, x)
    scale = tf.reshape(diameter_value, [-1, 1, 1]) * tf.sqrt(
        tf.cast(tf.shape(x)[2], tf.float32)
    )
    scaled_x = centered_x / tf.stop_gradient(scale)
    cost_xy = annealed_transport_tf._filterflow_exact_cost(scaled_x, tf.stop_gradient(scaled_x))
    cost_yx = annealed_transport_tf._filterflow_exact_cost(scaled_x, tf.stop_gradient(scaled_x))
    cost_xx = annealed_transport_tf._filterflow_exact_cost(scaled_x, tf.stop_gradient(scaled_x))
    cost_yy = annealed_transport_tf._filterflow_exact_cost(scaled_x, tf.stop_gradient(scaled_x))
    sinkhorn_scale = tf.stop_gradient(
        annealed_transport_tf._filterflow_exact_max_min(scaled_x, scaled_x)
    )
    epsilon_0 = sinkhorn_scale ** 2
    scaling_factor = scaling ** 2
    alpha, beta, _, _, iterations = annealed_transport_tf._filterflow_exact_sinkhorn_potentials(
        logw,
        scaled_x,
        uniform_log_weight,
        scaled_x,
        eps,
        scaling,
        threshold,
        max_iter,
    )
    transport_matrix = annealed_transport_tf._filterflow_exact_transport_from_potentials(
        scaled_x,
        alpha,
        beta,
        eps,
        logw,
        float_n,
    )
    transported = tf.linalg.matmul(transport_matrix, x)
    direct_transport, direct_iterations = annealed_transport_tf._filterflow_exact_transport_matrix(
        scaled_x,
        logw,
        eps,
        scaling,
        threshold,
        max_iter,
        n,
    )
    return {
        "time_index": row["time_index"],
        "input_particles": _json(x),
        "input_log_weights": _json(logw),
        "centered_particles": _json(centered_x),
        "diameter_value": _json(diameter_value),
        "scale": _json(scale),
        "scaled_particles": _json(scaled_x),
        "cost_xy": _json(cost_xy),
        "cost_yx": _json(cost_yx),
        "cost_xx": _json(cost_xx),
        "cost_yy": _json(cost_yy),
        "sinkhorn_scale": _json(sinkhorn_scale),
        "epsilon_0": _json(epsilon_0),
        "scaling_factor": _json(scaling_factor),
        "alpha": _json(alpha),
        "beta": _json(beta),
        "iterations": _json(iterations),
        "transport_matrix": _json(transport_matrix),
        "transported_particles": _json(transported),
        "direct_transport_matrix_delta": _max_abs(direct_transport, transport_matrix),
        "direct_iterations": _json(direct_iterations),
        "trace_transport_matrix_delta": _max_abs(
            tf.constant(row["transport_matrix"], dtype=tf.float32),
            transport_matrix,
        ),
        "trace_post_resample_particles_delta": _max_abs(
            tf.constant(row["post_resample_particles"], dtype=tf.float32),
            transported,
        ),
        "finite_values": bool(
            tf.reduce_all(tf.math.is_finite(transport_matrix)).numpy()
            and tf.reduce_all(tf.math.is_finite(transported)).numpy()
        ),
    }


def _compare_internals(
    filterflow_internals: dict[str, Any],
    bayesfilter_internals: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    first_divergence = None
    max_field_deltas: dict[str, float] = {field: 0.0 for field in FIELD_ORDER}
    for ff_row, bf_row in zip(
        filterflow_internals["rows"],
        bayesfilter_internals["rows"],
        strict=True,
    ):
        row_deltas = {
            field: _max_abs(ff_row[field], bf_row[field])
            for field in FIELD_ORDER
        }
        for field, delta in row_deltas.items():
            max_field_deltas[field] = max(max_field_deltas[field], delta)
        row_first = _first_field(row_deltas)
        if first_divergence is None and row_first is not None:
            first_divergence = {
                "time_index": ff_row["time_index"],
                "field": row_first,
                "delta": row_deltas[row_first],
            }
        rows.append(
            {
                "time_index": ff_row["time_index"],
                "first_divergent_field": row_first or "none",
                "field_deltas": row_deltas,
                "filterflow_trace_transport_matrix_delta": ff_row[
                    "trace_transport_matrix_delta"
                ],
                "bayesfilter_trace_transport_matrix_delta": bf_row[
                    "trace_transport_matrix_delta"
                ],
                "filterflow_trace_post_resample_particles_delta": ff_row[
                    "trace_post_resample_particles_delta"
                ],
                "bayesfilter_trace_post_resample_particles_delta": bf_row[
                    "trace_post_resample_particles_delta"
                ],
            }
        )
    finite = all(row["finite_values"] for row in filterflow_internals["rows"]) and all(
        row["finite_values"] for row in bayesfilter_internals["rows"]
    )
    max_filterflow_trace_transport_delta = max(
        row["filterflow_trace_transport_matrix_delta"] for row in rows
    )
    max_bayesfilter_trace_transport_delta = max(
        row["bayesfilter_trace_transport_matrix_delta"] for row in rows
    )
    max_filterflow_trace_particle_delta = max(
        row["filterflow_trace_post_resample_particles_delta"] for row in rows
    )
    max_bayesfilter_trace_particle_delta = max(
        row["bayesfilter_trace_post_resample_particles_delta"] for row in rows
    )
    return {
        "status": "compared",
        "finite_values": finite,
        "first_divergence": first_divergence or {"status": "no_divergence"},
        "row_comparisons": rows,
        "max_field_deltas": max_field_deltas,
        "max_filterflow_trace_transport_matrix_delta": max_filterflow_trace_transport_delta,
        "max_bayesfilter_trace_transport_matrix_delta": max_bayesfilter_trace_transport_delta,
        "max_filterflow_trace_post_resample_particles_delta": max_filterflow_trace_particle_delta,
        "max_bayesfilter_trace_post_resample_particles_delta": max_bayesfilter_trace_particle_delta,
        "implementation_agreement": first_divergence is None and finite,
    }


def _first_field(row_deltas: dict[str, float]) -> str | None:
    for field in FIELD_ORDER:
        if row_deltas[field] > 0.0:
            return field
    return None


def _filterflow_diameter(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    diameter_x = tf.reduce_max(tf.math.reduce_std(x, axis=1), axis=-1)
    diameter_y = tf.reduce_max(tf.math.reduce_std(y, axis=1), axis=-1)
    result = tf.maximum(diameter_x, diameter_y)
    return tf.where(result == 0.0, tf.ones_like(result), result)


def _decision(comparison: dict[str, Any], comparator_drift: bool) -> str:
    if comparator_drift:
        return "filterflow_r3_transport_internals_blocked_by_comparator_drift"
    if comparison.get("status") != "compared":
        return "filterflow_r3_transport_internals_blocked"
    if not comparison.get("finite_values"):
        return "filterflow_r3_transport_internals_nonfinite_veto"
    if comparison.get("implementation_agreement"):
        return "filterflow_r3_transport_internals_no_difference_on_selected_rows"
    field = comparison["first_divergence"]["field"]
    return f"filterflow_r3_transport_internals_first_difference_{field}"


def _compact_internals(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "executed":
        return payload
    return {
        "status": payload["status"],
        "backend": payload["backend"],
        "rows": [
            {
                "time_index": row["time_index"],
                "direct_transport_matrix_delta": row["direct_transport_matrix_delta"],
                "trace_transport_matrix_delta": row["trace_transport_matrix_delta"],
                "trace_post_resample_particles_delta": row[
                    "trace_post_resample_particles_delta"
                ],
                "finite_values": row["finite_values"],
            }
            for row in payload["rows"]
        ],
        "cpu_only_manifest": payload["cpu_only_manifest"],
        "stderr_excerpt": payload.get("stderr_excerpt", ""),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "trace_validation",
        "filterflow_internals",
        "bayesfilter_internals",
        "comparison",
        "run_manifest",
        "path_boundary_manifest",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if payload["filterflow_internals"].get("status") == "executed":
        _validate_cpu(payload["filterflow_internals"]["cpu_only_manifest"], "filterflow")
    if payload["bayesfilter_internals"].get("status") == "executed":
        _validate_cpu(payload["bayesfilter_internals"]["cpu_only_manifest"], "BayesFilter")
    if payload["decision"].endswith("_blocker"):
        return
    if not payload["trace_validation"]["official_trace_match"]:
        raise ValueError("trace validation failed without blocker decision")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Filterflow R3 Transport Internals Audit",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Interpretation",
        "",
        _interpretation(payload),
        "",
        "## Comparison",
        "",
        _json_block(payload["comparison"]),
        "",
        "## Trace Validation",
        "",
        _json_block(payload["trace_validation"]),
        "",
        "## Compact Internals",
        "",
        "### Filterflow",
        "",
        _json_block(payload["filterflow_internals"]),
        "",
        "### BayesFilter",
        "",
        _json_block(payload["bayesfilter_internals"]),
        "",
        "## Non-Implications",
        "",
        *[f"- {item}" for item in payload["non_implications"]],
        "",
    ]
    return "\n".join(lines)


def _interpretation(payload: dict[str, Any]) -> str:
    comparison = payload["comparison"]
    if comparison.get("status") != "compared":
        return "The internals audit was blocked before comparison."
    first = comparison["first_divergence"]
    if first.get("status") == "no_divergence":
        return (
            "No BayesFilter-versus-filterflow difference was detected in the "
            "selected-row recomputed transport internals. Nonzero deltas remain "
            "between the stored filterflow trace and a fresh recomputation on "
            "the same rows, with max trace transport-matrix delta "
            f"{comparison['max_filterflow_trace_transport_matrix_delta']} and "
            "max trace post-resample-particle delta "
            f"{comparison['max_filterflow_trace_post_resample_particles_delta']}. "
            "The prior R3 replay discrepancy is therefore better described as "
            "float32 repeat-call/trace sensitivity around the transport output, "
            "not as a detected low-level BayesFilter formula mismatch."
        )
    return (
        "The first execution-ordered internal difference is "
        f"`{first['field']}` at row {first['time_index']} "
        f"with max absolute delta {first['delta']}. This is a "
        "BayesFilter/filterflow difference localization, not a correctness "
        "judgment."
    )


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


def _json(tensor: tf.Tensor) -> Any:
    return tf.cast(tensor, tf.float64).numpy().tolist()


def _max_abs(left: Any, right: Any) -> float:
    left_tensor = tf.cast(tf.convert_to_tensor(left), tf.float64)
    right_tensor = tf.cast(tf.convert_to_tensor(right), tf.float64)
    return float(tf.reduce_max(tf.abs(left_tensor - right_tensor)).numpy())


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


if __name__ == "__main__":
    raise SystemExit(main())
