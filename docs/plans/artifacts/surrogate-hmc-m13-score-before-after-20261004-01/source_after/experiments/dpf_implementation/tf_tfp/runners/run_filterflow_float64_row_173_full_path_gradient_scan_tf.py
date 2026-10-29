"""Full-path row-173 FilterFlow/BayesFilter cumulative gradient scan."""

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

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_smoothness_gradient_localization_tf as localizer,
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
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-row-173-full-path-gradient-scan-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-row-173-full-path-gradient-scan-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_row_173_full_path_gradient_scan_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-row-173-full-path-gradient-scan-2026-06-03.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

T = 100
BATCH_SIZE = 1
NUM_PARTICLES = 50
DATA_SEED = 123
FILTER_SEED = 1234
EPSILON = 0.25
SCALING = 0.85
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 500
RESAMPLING_NEFF = 0.9999
THETA = [0.9710526315789474, 0.9842105263157894]
MESH_INDEX = 173
SCALAR_TOLERANCE = 5e-5
GRADIENT_ABS_TOLERANCE = 2e-4
GRADIENT_REL_TOLERANCE = 2e-4


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
    reference_status = r3._filterflow_status()
    validate_filterflow_reference_status(reference_status, marker_path=FILTERFLOW_MARKER_PATH)
    initial_fingerprint = continuation._filterflow_fingerprint()
    filterflow = _filterflow_scan_subprocess()
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_full_path_scan_filterflow_blocker",
            filterflow.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow,
            reference_status,
        )
    bayesfilter = _bayesfilter_scan(filterflow)
    comparison = _compare(filterflow, bayesfilter)
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
        "question": "row_173_same_contract_full_path_cumulative_gradient_scan",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "filterflow_scan": _compact_filterflow(filterflow),
        "bayesfilter_scan": _compact_bayesfilter(bayesfilter),
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_full_path_gradient_scan_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any],
    reference_status: dict[str, Any],
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    comparison = {"status": "blocked", "blocker": blocker}
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "blocker": blocker,
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "model_contract": _model_contract(),
        "filterflow_scan": filterflow,
        "bayesfilter_scan": None,
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_full_path_gradient_scan_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _filterflow_scan_subprocess() -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_scan_script()],
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
            "blocker": "filterflow full-path scan subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_ROW_173_FULL_PATH_SCAN_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_ROW_173_FULL_PATH_SCAN_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow full-path scan JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(
        completed.stdout[start + len("FILTERFLOW_ROW_173_FULL_PATH_SCAN_JSON_BEGIN"):end].strip()
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_scan_script() -> str:
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

        np = __import__("numpy")
        import attr
        import tensorflow as tf
        from tensorflow_probability.python.internal import samplers

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_smoothness import get_data

        DTYPE = tf.float64
        NP_DTYPE = np.float64
        T = {T}
        BATCH_SIZE = {BATCH_SIZE}
        N = {NUM_PARTICLES}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}
        EPSILON = {EPSILON!r}
        SCALING = {SCALING!r}
        CONVERGENCE_THRESHOLD = {CONVERGENCE_THRESHOLD!r}
        MAX_ITERATIONS = {MAX_ITERATIONS}
        RESAMPLING_NEFF = {RESAMPLING_NEFF!r}
        THETA = np.array({THETA!r}, dtype=NP_DTYPE)

        def to_json(tensor):
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, tf.float64).numpy())

        transition_matrix_np = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=NP_DTYPE)
        transition_covariance_np = np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]], dtype=NP_DTYPE)
        observation_matrix_np = np.array([[1.0, 0.0]], dtype=NP_DTYPE)
        observation_covariance_np = np.array([[0.01]], dtype=NP_DTYPE)

        rng = np.random.RandomState(seed=DATA_SEED)
        observations_np, _kf = get_data(
            transition_matrix_np,
            observation_matrix_np,
            transition_covariance_np,
            observation_covariance_np,
            T=T,
            random_state=rng,
        )
        initial_particles_np = rng.normal(0.0, 0.01, [BATCH_SIZE, N, 2]).astype(NP_DTYPE)

        observations = tf.convert_to_tensor(observations_np, dtype=DTYPE)
        initial_state = State(tf.convert_to_tensor(initial_particles_np, dtype=DTYPE))
        modifiable_transition_matrix = tf.Variable(transition_matrix_np, trainable=False, dtype=DTYPE)
        transition_matrix = tf.linalg.diag(tf.convert_to_tensor(THETA, dtype=DTYPE)) + tf.constant(
            [[0.0, 1.0], [0.0, 0.0]],
            dtype=DTYPE,
        )
        modifiable_transition_matrix.assign(transition_matrix)
        observation_matrix = tf.convert_to_tensor(observation_matrix_np, dtype=DTYPE)
        transition_covariance_chol = tf.linalg.cholesky(
            tf.convert_to_tensor(transition_covariance_np, dtype=DTYPE)
        )
        observation_covariance_chol = tf.linalg.cholesky(
            tf.convert_to_tensor(observation_covariance_np, dtype=DTYPE)
        )
        smc = make_filter(
            observation_matrix,
            modifiable_transition_matrix,
            observation_covariance_chol,
            transition_covariance_chol,
            RegularisedTransform(
                epsilon=EPSILON,
                scaling=SCALING,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                max_iter=MAX_ITERATIONS,
            ),
            NeffCriterion(RESAMPLING_NEFF, True),
            optimal_proposal=True,
        )

        seed = tf.constant(FILTER_SEED, dtype=tf.int32)
        paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
        seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))
        state = initial_state
        cumulative_targets = []
        row_meta = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(modifiable_transition_matrix)
            for time_index in range(T):
                seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
                observation = observations[time_index]
                resampling_flag, ess = smc._resampling_criterion.apply(state)
                float_t = tf.cast(state.t, DTYPE)
                float_t_1 = float_t + tf.constant(1.0, dtype=DTYPE)
                state_for_step = attr.evolve(
                    state,
                    ess=ess / float_t_1 + state.ess * (float_t / float_t_1),
                )
                resampled_state = smc._resampling_method.apply(state_for_step, resampling_flag, seed1)
                new_state = smc.propose_and_weight(
                    resampled_state,
                    observation,
                    tf.convert_to_tensor(time_index, dtype=tf.int32),
                    seed2,
                )
                new_state = smc._resampling_correction_term(
                    resampling_flag,
                    new_state,
                    state_for_step,
                    observation,
                    tf.convert_to_tensor(time_index, dtype=tf.int32),
                )
                state = attr.evolve(new_state, t=state_for_step.t + 1)
                target = tf.reduce_mean(state.log_likelihoods)
                cumulative_targets.append(target)
                row_meta.append({{
                    "time_index": int(time_index),
                    "cumulative_mean_log_likelihood": scalar(target),
                    "resampling_flag": [bool(v) for v in tf.reshape(resampling_flag, [-1]).numpy().tolist()],
                    "ess_before_resampling": to_json(ess),
                    "finite_scalar": bool(tf.math.is_finite(target).numpy()),
                }})

        rows = []
        finite_gradient = True
        for meta, target in zip(row_meta, cumulative_targets):
            grad_matrix = tape.gradient(target, modifiable_transition_matrix)
            finite = bool(tf.reduce_all(tf.math.is_finite(grad_matrix)).numpy())
            finite_gradient = finite_gradient and finite
            rows.append({{
                **meta,
                "gradient_diag": to_json(tf.linalg.diag_part(grad_matrix)),
                "gradient_matrix": to_json(grad_matrix),
                "finite_gradient": finite,
            }})
        del tape

        payload = {{
            "status": "executed",
            "backend": "executable_filterflow_subprocess",
            "scan_contract": "single_persistent_tape_full_path_cumulative_gradients",
            "settings": {{
                "theta": THETA.astype(float).tolist(),
                "mesh_index": {MESH_INDEX},
                "T": T,
                "batch_size": BATCH_SIZE,
                "n_particles": N,
                "data_seed": DATA_SEED,
                "filter_seed": FILTER_SEED,
                "epsilon": EPSILON,
                "scaling": SCALING,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iter": MAX_ITERATIONS,
                "resampling_neff": RESAMPLING_NEFF,
                "optimal_proposal": True,
                "resampling_correction": False,
                "dtype": "float64",
            }},
            "model": {{
                "observations": observations_np.astype(float).tolist(),
                "initial_particles": initial_particles_np.astype(float).tolist(),
                "observation_matrix": observation_matrix_np.astype(float).tolist(),
                "transition_covariance_chol": to_json(transition_covariance_chol),
                "observation_covariance_chol": to_json(observation_covariance_chol),
            }},
            "rows": rows,
            "final_gradient_diag": rows[-1]["gradient_diag"],
            "final_mean_log_likelihood": rows[-1]["cumulative_mean_log_likelihood"],
            "finite_values": all(row["finite_scalar"] and row["finite_gradient"] for row in rows),
            "finite_gradients": bool(finite_gradient),
            "observation_checksum": float(np.sum(observations_np.astype(np.float64))),
            "initial_particles_checksum": float(np.sum(initial_particles_np.astype(np.float64))),
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
            "package_versions": {{
                "python": os.sys.version.split()[0],
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
            }},
        }}
        print("FILTERFLOW_ROW_173_FULL_PATH_SCAN_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_ROW_173_FULL_PATH_SCAN_JSON_END")
        """
    )


def _bayesfilter_scan(filterflow: dict[str, Any]) -> dict[str, Any]:
    mode = localizer._bayesfilter_mode_ledger("transport_upstream_clip", filterflow)
    return {
        "status": "executed",
        "backend": "tensorflow_tensorflow_probability",
        "scan_contract": "single_persistent_tape_full_path_cumulative_gradients",
        "settings": filterflow["settings"],
        "rows": mode["rows"],
        "final_gradient_diag": mode["final_gradient_diag"],
        "final_mean_log_likelihood": mode["final_mean_log_likelihood"],
        "finite_values": mode["finite_values"],
        "finite_gradients": mode["finite_gradients"],
        "cpu_only_manifest": _parent_cpu_manifest(),
    }


def _compare(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    if bayesfilter.get("status") != "executed":
        return {"status": "blocked", "blocker": "BayesFilter scan did not execute"}
    rows = []
    for ff_row, bf_row in zip(filterflow["rows"], bayesfilter["rows"], strict=True):
        scalar_delta = abs(
            float(bf_row["cumulative_mean_log_likelihood"])
            - float(ff_row["cumulative_mean_log_likelihood"])
        )
        gradient_delta = [
            float(bf_value) - float(ff_value)
            for bf_value, ff_value in zip(bf_row["gradient_diag"], ff_row["gradient_diag"], strict=True)
        ]
        max_abs_gradient_delta = max(abs(value) for value in gradient_delta)
        max_abs_ff_gradient = max(abs(float(value)) for value in ff_row["gradient_diag"])
        relative_gradient_delta = max_abs_gradient_delta / max(1.0, max_abs_ff_gradient)
        rows.append(
            {
                "time_index": ff_row["time_index"],
                "scalar_delta": scalar_delta,
                "gradient_delta": gradient_delta,
                "max_abs_gradient_delta": max_abs_gradient_delta,
                "relative_gradient_delta": relative_gradient_delta,
                "filterflow_gradient_diag": ff_row["gradient_diag"],
                "bayesfilter_gradient_diag": bf_row["gradient_diag"],
                "filterflow_gradient_max_abs": max_abs_ff_gradient,
                "bayesfilter_gradient_max_abs": bf_row["gradient_max_abs"],
                "scalar_within_tolerance": scalar_delta <= SCALAR_TOLERANCE,
                "gradient_within_tolerance": (
                    max_abs_gradient_delta <= GRADIENT_ABS_TOLERANCE
                    or relative_gradient_delta <= GRADIENT_REL_TOLERANCE
                ),
                "resampling_flag": bf_row["resampling_flag"],
                "transport_status": bf_row["transport_status"],
            }
        )
    first_scalar_failure = _first_failure(rows, "scalar_within_tolerance")
    first_gradient_failure = _first_failure(rows, "gradient_within_tolerance")
    return {
        "status": "compared",
        "first_scalar_failure": first_scalar_failure,
        "first_gradient_failure": first_gradient_failure,
        "final_row": rows[-1],
        "sample_rows": _sample_rows(rows),
        "finite_values": bool(filterflow["finite_values"] and bayesfilter["finite_values"]),
        "interpretation": _interpret(first_scalar_failure, first_gradient_failure),
    }


def _first_failure(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    for row in rows:
        if not row[field]:
            return {
                "status": "failure",
                "time_index": row["time_index"],
                "scalar_delta": row["scalar_delta"],
                "max_abs_gradient_delta": row["max_abs_gradient_delta"],
                "relative_gradient_delta": row["relative_gradient_delta"],
                "gradient_delta": row["gradient_delta"],
                "filterflow_gradient_diag": row["filterflow_gradient_diag"],
                "bayesfilter_gradient_diag": row["bayesfilter_gradient_diag"],
                "resampling_flag": row["resampling_flag"],
                "transport_status": row["transport_status"],
            }
    return {"status": "no_failure"}


def _sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {0, 1, 2, 3, 4, 9, 24, 49, 74, 99}
    selected.update(row["time_index"] for row in rows if not row["gradient_within_tolerance"])
    return [row for row in rows if row["time_index"] in selected]


def _interpret(
    first_scalar_failure: dict[str, Any],
    first_gradient_failure: dict[str, Any],
) -> str:
    if first_scalar_failure["status"] == "failure":
        return "scalar_mismatch_vetoes_gradient_interpretation"
    if first_gradient_failure["status"] == "failure":
        return f"first_true_full_path_gradient_residual_time_{first_gradient_failure['time_index']}"
    return "same_contract_full_path_gradients_match_on_row_173"


def _decision(comparison: dict[str, Any], comparator_drift: bool) -> str:
    if comparator_drift:
        return "filterflow_float64_row_173_full_path_scan_blocked_by_comparator_drift"
    if comparison.get("status") != "compared":
        return "filterflow_float64_row_173_full_path_scan_blocked"
    if not comparison.get("finite_values", False):
        return "filterflow_float64_row_173_full_path_scan_nonfinite_veto"
    if comparison["first_scalar_failure"]["status"] == "failure":
        return "filterflow_float64_row_173_full_path_scan_scalar_veto"
    if comparison["first_gradient_failure"]["status"] == "failure":
        return "filterflow_float64_row_173_full_path_scan_gradient_residual_localized"
    return "filterflow_float64_row_173_full_path_scan_gradients_match"


def _evidence_contract() -> dict[str, Any]:
    return {
        "primary_comparator": "local executable float64 filterflow checkout",
        "primary_question": "same_contract_full_path_cumulative_gradient_difference",
        "primary_pass": "identify first cumulative gradient residual time",
        "scalar_tolerance": SCALAR_TOLERANCE,
        "gradient_abs_tolerance": GRADIENT_ABS_TOLERANCE,
        "gradient_rel_tolerance": GRADIENT_REL_TOLERANCE,
        "mathematical_correctness": "not_concluded",
    }


def _model_contract() -> dict[str, Any]:
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "mesh_index": MESH_INDEX,
        "theta": THETA,
        "transition_matrix": "A(theta)=diag(theta_1, theta_2)+[[0,1],[0,0]]",
        "transition_covariance": [[1.0 / 3.0, 0.5], [0.5, 1.0]],
        "observation_matrix": [[1.0, 0.0]],
        "observation_covariance": [[0.01]],
        "T": T,
        "batch_size": BATCH_SIZE,
        "num_particles": NUM_PARTICLES,
        "data_seed": DATA_SEED,
        "filter_seed": FILTER_SEED,
        "epsilon": EPSILON,
        "scaling": SCALING,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "max_iter": MAX_ITERATIONS,
        "resampling_neff": RESAMPLING_NEFF,
        "dtype": "float64",
    }


def _compact_filterflow(filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return filterflow
    return {
        "status": filterflow["status"],
        "backend": filterflow["backend"],
        "scan_contract": filterflow["scan_contract"],
        "settings": filterflow["settings"],
        "final_gradient_diag": filterflow["final_gradient_diag"],
        "final_mean_log_likelihood": filterflow["final_mean_log_likelihood"],
        "finite_values": filterflow["finite_values"],
        "first_rows": filterflow["rows"][:5],
        "last_row": filterflow["rows"][-1],
        "cpu_only_manifest": filterflow["cpu_only_manifest"],
        "package_versions": filterflow["package_versions"],
        "stderr_excerpt": filterflow.get("stderr_excerpt", ""),
    }


def _compact_bayesfilter(bayesfilter: dict[str, Any] | None) -> dict[str, Any] | None:
    if bayesfilter is None or bayesfilter.get("status") != "executed":
        return bayesfilter
    return {
        "status": bayesfilter["status"],
        "backend": bayesfilter["backend"],
        "scan_contract": bayesfilter["scan_contract"],
        "settings": bayesfilter["settings"],
        "final_gradient_diag": bayesfilter["final_gradient_diag"],
        "final_mean_log_likelihood": bayesfilter["final_mean_log_likelihood"],
        "finite_values": bayesfilter["finite_values"],
        "first_rows": bayesfilter["rows"][:5],
        "last_row": bayesfilter["rows"][-1],
        "cpu_only_manifest": bayesfilter["cpu_only_manifest"],
    }


def _decision_table(decision: str, comparison: dict[str, Any]) -> list[dict[str, str]]:
    if decision == "filterflow_float64_row_173_full_path_scan_gradient_residual_localized":
        primary = f"first gradient residual: {comparison.get('first_gradient_failure')}"
        veto = "scalar path stayed aligned before residual"
        next_action = "run a VJP decomposition at the first true residual time"
    elif decision == "filterflow_float64_row_173_full_path_scan_gradients_match":
        primary = "no same-contract full-path residual on row 173"
        veto = "none"
        next_action = "rerun the surface with same-contract diagnostics to reconcile old row failure"
    elif decision == "filterflow_float64_row_173_full_path_scan_scalar_veto":
        primary = f"scalar mismatch: {comparison.get('first_scalar_failure')}"
        veto = "scalar mismatch"
        next_action = "repair scalar replay before interpreting gradients"
    else:
        primary = comparison.get("blocker", decision)
        veto = decision
        next_action = "repair blocker before interpreting scan"
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": veto,
            "main_uncertainty": "single mesh row only; no correctness claim",
            "next_justified_action": next_action,
            "not_concluded": "correctness of either implementation, analytic gradient correctness, production readiness",
        }
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "evidence_contract",
        "reference_policy",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "model_contract",
        "filterflow_scan",
        "bayesfilter_scan",
        "comparison",
        "path_boundary_manifest",
        "run_manifest",
        "decision_table",
        "non_implications",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    allowed = {
        "filterflow_float64_row_173_full_path_scan_filterflow_blocker",
        "filterflow_float64_row_173_full_path_scan_blocked_by_comparator_drift",
        "filterflow_float64_row_173_full_path_scan_blocked",
        "filterflow_float64_row_173_full_path_scan_nonfinite_veto",
        "filterflow_float64_row_173_full_path_scan_scalar_veto",
        "filterflow_float64_row_173_full_path_scan_gradient_residual_localized",
        "filterflow_float64_row_173_full_path_scan_gradients_match",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    for label in ("filterflow_scan", "bayesfilter_scan"):
        side = payload.get(label)
        if side is not None and side.get("status") == "executed":
            _validate_cpu(side["cpu_only_manifest"], label)


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: FilterFlow Float64 Row 173 Full-Path Gradient Scan",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            "| {decision} | {primary} | {veto} | {uncertainty} | {next_action} | {not_concluded} |".format(
                decision=row["decision"],
                primary=row["primary_criterion_status"],
                veto=row["veto_diagnostic_status"],
                uncertainty=row["main_uncertainty"],
                next_action=row["next_justified_action"],
                not_concluded=row["not_concluded"],
            )
        )
    lines.extend(
        [
            "",
            "## Model Contract",
            "",
            _key_value_table(payload["model_contract"]),
            "",
            "## Comparison",
            "",
            _json_block(payload["comparison"]),
            "",
            "## FilterFlow Scan",
            "",
            _json_block(payload["filterflow_scan"]),
            "",
            "## BayesFilter Scan",
            "",
            _json_block(payload["bayesfilter_scan"]),
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _non_implications() -> list[str]:
    return step._non_implications() + [
        "No correctness claim is made for either implementation.",
        "No analytic smoothness-gradient correctness is concluded.",
        "No full mesh_size=20 surface agreement is concluded.",
        "No production dtype default is concluded.",
        "Finite gradients alone are smoke evidence only.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
