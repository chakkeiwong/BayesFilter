"""Float64 smoothness gradient-surface difference audit."""

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
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
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
    FILTERFLOW_REFERENCE_DTYPE,
    reference_policy,
    validate_filterflow_reference_status,
)


tfd = tfp.distributions

DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-smoothness-gradient-surface-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-smoothness-gradient-surface-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_smoothness_gradient_surface_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-smoothness-gradient-surface-2026-06-03.md"
FULL_SURFACE_PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-filterflow-float64-smoothness-gradient-full-surface-plan-2026-06-03.md"
)
FULL_SURFACE_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-filterflow-float64-smoothness-gradient-full-surface-result-2026-06-03.md"
)
FULL_SURFACE_JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_smoothness_gradient_full_surface_2026-06-03.json"
)
FULL_SURFACE_REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-smoothness-gradient-full-surface-2026-06-03.md"
)
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

T = 100
BATCH_SIZE = 1
NUM_PARTICLES = 50
BOUNDED_MESH_SIZE = 4
FULL_MESH_SIZE = 20
MESH_SIZE = BOUNDED_MESH_SIZE
DATA_SEED = 123
FILTER_SEED = 1234
EPSILON = 0.25
SCALING = 0.85
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 500
RESAMPLING_NEFF = 0.9999
SCALAR_TOLERANCE = 5e-5
GRADIENT_ABS_TOLERANCE = 2e-4
GRADIENT_REL_TOLERANCE = 2e-4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--full-surface",
        action="store_true",
        help="Run the script-default mesh_size=20 surface into a separate artifact.",
    )
    parser.add_argument(
        "--row-start",
        type=int,
        default=0,
        help="First mesh row to compare within the selected surface.",
    )
    parser.add_argument(
        "--row-count",
        type=int,
        default=None,
        help="Number of mesh rows to compare; default compares the full selected surface.",
    )
    args = parser.parse_args(argv)
    config = _artifact_config(
        full_surface=args.full_surface,
        row_start=args.row_start,
        row_count=args.row_count,
    )
    if args.validate_only:
        _validate_payload(load_json(config["json_path"]))
        return 0

    start = time.perf_counter()
    payload = _run(config)
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(config["json_path"], payload)
    markdown = _markdown(payload)
    write_text(config["report_path"], markdown)
    write_text(REPO_ROOT / config["result_path"], markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _artifact_config(
    *,
    full_surface: bool,
    row_start: int = 0,
    row_count: int | None = None,
) -> dict[str, Any]:
    if row_start < 0:
        raise ValueError("row_start must be non-negative")
    if full_surface:
        total_rows = FULL_MESH_SIZE * FULL_MESH_SIZE
        row_count = total_rows - row_start if row_count is None else row_count
        if row_count <= 0 or row_start + row_count > total_rows:
            raise ValueError(
                f"invalid full-surface row window: start={row_start}, count={row_count}, total={total_rows}"
            )
        window_suffix = f"_rows_{row_start}_{row_start + row_count - 1}"
        file_suffix = f"-rows-{row_start}-{row_start + row_count - 1}"
        return {
            "full_surface": True,
            "mesh_size": FULL_MESH_SIZE,
            "row_start": row_start,
            "row_count": row_count,
            "row_stop_exclusive": row_start + row_count,
            "plan_path": FULL_SURFACE_PLAN_PATH,
            "result_path": (
                FULL_SURFACE_RESULT_PATH
                if row_start == 0 and row_count == total_rows
                else FULL_SURFACE_RESULT_PATH.replace("-result-", f"{file_suffix}-result-")
            ),
            "json_path": (
                FULL_SURFACE_JSON_PATH
                if row_start == 0 and row_count == total_rows
                else OUTPUT_DIR
                / f"dpf_filterflow_float64_smoothness_gradient_full_surface{window_suffix}_2026-06-03.json"
            ),
            "report_path": (
                FULL_SURFACE_REPORT_PATH
                if row_start == 0 and row_count == total_rows
                else REPORT_DIR
                / f"dpf-filterflow-float64-smoothness-gradient-full-surface{file_suffix}-2026-06-03.md"
            ),
            "scope_label": (
                "full script-default mesh_size=20"
                if row_start == 0 and row_count == total_rows
                else f"full script-default mesh_size=20 row window [{row_start}, {row_start + row_count})"
            ),
            "decision_prefix": "filterflow_float64_smoothness_gradient_full_surface",
            "module_suffix": (
                " --full-surface"
                if row_start == 0 and row_count == total_rows
                else f" --full-surface --row-start {row_start} --row-count {row_count}"
            ),
        }
    if row_count is not None:
        total_rows = BOUNDED_MESH_SIZE * BOUNDED_MESH_SIZE
        if row_count <= 0 or row_start + row_count > total_rows:
            raise ValueError(
                f"invalid bounded row window: start={row_start}, count={row_count}, total={total_rows}"
            )
    elif row_start != 0:
        raise ValueError("bounded run row_start requires row_count")
    return {
        "full_surface": False,
        "mesh_size": BOUNDED_MESH_SIZE,
        "row_start": row_start,
        "row_count": row_count,
        "row_stop_exclusive": None if row_count is None else row_start + row_count,
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "json_path": JSON_PATH,
        "report_path": REPORT_PATH,
        "scope_label": "bounded mesh_size=4",
        "decision_prefix": "filterflow_float64_smoothness_gradient_surface",
        "module_suffix": "",
    }


def _run(config: dict[str, Any]) -> dict[str, Any]:
    reference_status = r3._filterflow_status()
    validate_filterflow_reference_status(reference_status, marker_path=FILTERFLOW_MARKER_PATH)
    initial_fingerprint = continuation._filterflow_fingerprint()
    filterflow = _filterflow_gradient_surface_subprocess(
        mesh_size=int(config["mesh_size"]),
        row_start=int(config["row_start"]),
        row_count=config["row_count"],
    )
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            _decision_name(config, "filterflow_blocker"),
            filterflow.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow,
            reference_status,
            config,
        )

    bayesfilter = _bayesfilter_gradient_surface(filterflow)
    comparison = _compare_surfaces(filterflow, bayesfilter)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(comparison, bayesfilter, comparator_drift, config)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": config["plan_path"],
        "result_path": config["result_path"],
        "report_path": str(config["report_path"].relative_to(REPO_ROOT)),
        "json_path": str(config["json_path"].relative_to(REPO_ROOT)),
        "question": (
            f"Does BayesFilter match local float64 FilterFlow on the {config['scope_label']} "
            "simple_linear_smoothness scalar and diagonal gradient surface?"
        ),
        "evidence_contract": _evidence_contract(config),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(config),
        "scalar_contract": _scalar_contract(),
        "gradient_contract": _gradient_contract(),
        "filterflow_surface": _compact_filterflow(filterflow),
        "bayesfilter_surface": _compact_bayesfilter(bayesfilter),
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_smoothness_gradient_surface_tf"
                f"{config['module_suffix']}"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison, config),
        "non_implications": _non_implications(config),
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any],
    reference_status: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    comparison = {"status": "blocked", "blocker": blocker}
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": config["plan_path"],
        "result_path": config["result_path"],
        "blocker": blocker,
        "evidence_contract": _evidence_contract(config),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "model_contract": _model_contract(config),
        "scalar_contract": _scalar_contract(),
        "gradient_contract": _gradient_contract(),
        "filterflow_surface": filterflow,
        "bayesfilter_surface": None,
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_smoothness_gradient_surface_tf"
                f"{config['module_suffix']}"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison, config),
        "non_implications": _non_implications(config),
    }


def _filterflow_gradient_surface_subprocess(
    *,
    mesh_size: int,
    row_start: int = 0,
    row_count: int | None = None,
) -> dict[str, Any]:
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
        [
            str(FILTERFLOW_ENV_PYTHON),
            "-c",
            _filterflow_gradient_surface_script(
                mesh_size=mesh_size,
                row_start=row_start,
                row_count=row_count,
            ),
        ],
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
            "blocker": "filterflow gradient surface subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_SMOOTHNESS_GRADIENT_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_SMOOTHNESS_GRADIENT_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow gradient surface JSON sentinels missing",
            "stdout_excerpt": stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(stdout[start + len("FILTERFLOW_SMOOTHNESS_GRADIENT_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_gradient_surface_script(
    *,
    mesh_size: int | None = None,
    row_start: int = 0,
    row_count: int | None = None,
) -> str:
    mesh_size = MESH_SIZE if mesh_size is None else mesh_size
    row_count_literal = "None" if row_count is None else str(row_count)
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
        import tensorflow as tf

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_smoothness import get_data, routine

        DTYPE = tf.float64
        NP_DTYPE = np.float64
        T = {T}
        BATCH_SIZE = {BATCH_SIZE}
        N = {NUM_PARTICLES}
        MESH_SIZE = {mesh_size}
        ROW_START = {row_start}
        ROW_COUNT = {row_count_literal}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}
        EPSILON = {EPSILON!r}
        SCALING = {SCALING!r}
        CONVERGENCE_THRESHOLD = {CONVERGENCE_THRESHOLD!r}
        MAX_ITERATIONS = {MAX_ITERATIONS}
        RESAMPLING_NEFF = {RESAMPLING_NEFF!r}

        def to_json(tensor):
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, tf.float64).numpy())

        transition_matrix_np = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=NP_DTYPE)
        transition_covariance_np = np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]], dtype=NP_DTYPE)
        observation_matrix_np = np.array([[1.0, 0.0]], dtype=NP_DTYPE)
        observation_covariance_np = np.array([[0.01]], dtype=NP_DTYPE)
        x_linspace = np.linspace(0.95, 1.0, MESH_SIZE).astype(NP_DTYPE)
        y_linspace = np.linspace(0.95, 1.0, MESH_SIZE).astype(NP_DTYPE)
        mesh = np.asanyarray([(x, y) for x in x_linspace for y in y_linspace]).astype(NP_DTYPE)
        mesh_stop = mesh.shape[0] if ROW_COUNT is None else ROW_START + ROW_COUNT
        mesh_window = mesh[ROW_START:mesh_stop]

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

        rows = []
        for local_index, theta in enumerate(mesh_window):
            index = ROW_START + local_index
            matrix = tf.linalg.diag(tf.convert_to_tensor(theta, dtype=DTYPE)) + tf.constant(
                [[0.0, 1.0], [0.0, 0.0]],
                dtype=DTYPE,
            )
            modifiable_transition_matrix.assign(matrix)
            ll, grad_matrix, ess = routine(
                smc,
                initial_state,
                False,
                tf.data.Dataset.from_tensor_slices(observations),
                tf.constant(T),
                modifiable_transition_matrix,
                tf.constant(FILTER_SEED, dtype=tf.int32),
            )
            diag_grad = tf.linalg.diag_part(grad_matrix)
            rows.append({{
                "mesh_index": int(index),
                "theta": [float(theta[0]), float(theta[1])],
                "mean_log_likelihood": scalar(ll),
                "gradient_diag": to_json(diag_grad),
                "gradient_matrix": to_json(grad_matrix),
                "final_ess": to_json(ess),
                "finite_scalar": bool(tf.math.is_finite(ll).numpy()),
                "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(diag_grad)).numpy()),
            }})

        payload = {{
            "status": "executed",
            "surface_contract": "simple_linear_smoothness_scalar_and_diagonal_gradient",
            "settings": {{
                "T": T,
                "batch_size": BATCH_SIZE,
                "n_particles": N,
                "mesh_size": MESH_SIZE,
                "row_start": ROW_START,
                "row_count": ROW_COUNT,
                "row_stop_exclusive": int(mesh_stop),
                "script_default_mesh_size": 20,
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
                "transition_matrix_base": transition_matrix_np.astype(float).tolist(),
                "transition_covariance": transition_covariance_np.astype(float).tolist(),
                "transition_covariance_chol": to_json(transition_covariance_chol),
                "observation_matrix": observation_matrix_np.astype(float).tolist(),
                "observation_covariance": observation_covariance_np.astype(float).tolist(),
                "observation_covariance_chol": to_json(observation_covariance_chol),
                "observations": observations_np.astype(float).tolist(),
                "initial_particles": initial_particles_np.astype(float).tolist(),
                "mesh": mesh.astype(float).tolist(),
                "mesh_window": mesh_window.astype(float).tolist(),
            }},
            "rows": rows,
            "observation_checksum": float(np.sum(observations_np.astype(np.float64))),
            "initial_particles_checksum": float(np.sum(initial_particles_np.astype(np.float64))),
            "package_versions": {{
                "python": os.sys.version.split()[0],
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
            }},
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
        }}
        print("FILTERFLOW_SMOOTHNESS_GRADIENT_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_SMOOTHNESS_GRADIENT_JSON_END")
        """
    )


def _bayesfilter_gradient_surface(filterflow: dict[str, Any]) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        model = filterflow["model"]
        mesh = model.get("mesh_window", model["mesh"])
        row_start = int(filterflow["settings"].get("row_start") or 0)
        rows = []
        for local_index, theta in enumerate(mesh):
            rows.append(
                _bayesfilter_gradient_surface_row(
                    mesh_index=row_start + local_index,
                    theta=theta,
                    model=model,
                    settings=filterflow["settings"],
                )
            )
        finite = all(row["finite_scalar"] and row["finite_gradient"] for row in rows)
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "surface_contract": "same_scalar_and_diagonal_theta_gradient_as_filterflow_smoothness",
            "rows": rows,
            "finite_values": finite,
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_gradient_surface_row(
    *,
    mesh_index: int,
    theta: list[float],
    model: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    theta_variable = tf.Variable(theta, dtype=DTYPE)
    with tf.GradientTape() as tape:
        mean_log_likelihood, diagnostics = _bayesfilter_mean_log_likelihood(
            theta_variable,
            model=model,
            settings=settings,
        )
    gradient = tape.gradient(mean_log_likelihood, theta_variable)
    if gradient is None:
        gradient = tf.fill([2], tf.constant(float("nan"), dtype=DTYPE))
    return {
        "mesh_index": mesh_index,
        "theta": [float(theta[0]), float(theta[1])],
        "mean_log_likelihood": _float(mean_log_likelihood),
        "gradient_diag": r3._json(gradient),
        "log_likelihoods": diagnostics["log_likelihoods"],
        "final_log_neff": diagnostics["final_log_neff"],
        "resampling_count": diagnostics["resampling_count"],
        "max_row_residual": diagnostics["max_row_residual"],
        "max_column_residual": diagnostics["max_column_residual"],
        "finite_scalar": bool(tf.math.is_finite(mean_log_likelihood).numpy()),
        "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()),
    }


def _bayesfilter_mean_log_likelihood(
    theta: tf.Tensor,
    *,
    model: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[tf.Tensor, dict[str, Any]]:
    transition_matrix = _transition_matrix(theta)
    observation_matrix = tf.constant(model["observation_matrix"], dtype=DTYPE)
    transition_chol = tf.constant(model["transition_covariance_chol"], dtype=DTYPE)
    observation_chol = tf.constant(model["observation_covariance_chol"], dtype=DTYPE)
    observations = tf.constant(model["observations"], dtype=DTYPE)
    particles = tf.constant(model["initial_particles"], dtype=DTYPE)
    num_particles = int(settings["n_particles"])
    time_steps = int(settings["T"])
    log_weights = tf.fill(
        [BATCH_SIZE, num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )
    log_likelihoods = tf.zeros([BATCH_SIZE], dtype=DTYPE)
    transition_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([2], dtype=DTYPE),
        scale_tril=transition_chol,
    )
    observation_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([1], dtype=DTYPE),
        scale_tril=observation_chol,
    )
    transition_cov_inv = tf.linalg.cholesky_solve(transition_chol, tf.eye(2, dtype=DTYPE))
    observation_cov_inv = tf.linalg.cholesky_solve(observation_chol, tf.eye(1, dtype=DTYPE))
    sigma_inv = transition_cov_inv + tf.linalg.matmul(
        tf.linalg.matmul(
            observation_matrix,
            observation_cov_inv,
            transpose_a=True,
        ),
        observation_matrix,
    )
    sigma = tf.linalg.inv(sigma_inv)
    sigma_chol = tf.linalg.cholesky(sigma)
    seed = tf.constant(FILTER_SEED, dtype=tf.int32)
    paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
    seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))

    resampling_count = 0
    max_row_residual = 0.0
    max_column_residual = 0.0
    for time_index in range(time_steps):
        seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
        del seed1
        ess_log = r3._ess(log_weights)
        flags = ess_log <= tf.math.log(
            tf.cast(num_particles, DTYPE) * tf.constant(RESAMPLING_NEFF, DTYPE)
        )
        bool_flags = tf.reshape(flags, [-1])
        if bool(tf.reduce_any(bool_flags).numpy()):
            transported = annealed_transport_tf.annealed_transport_resample_tf(
                particles,
                log_weights,
                epsilon=float(settings["epsilon"]),
                scaling=float(settings["scaling"]),
                convergence_threshold=float(settings["convergence_threshold"]),
                max_iterations=int(settings["max_iter"]),
                ess_mask=bool_flags,
            )
            particles = transported.particles
            log_weights = transported.log_weights
            resampling_count += 1
            max_row_residual = max(max_row_residual, transported.diagnostics["max_row_residual"])
            max_column_residual = max(max_column_residual, transported.diagnostics["max_column_residual"])
        observation = observations[time_index]
        proposal_mean = _optimal_proposal_mean(
            particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
        )
        proposal_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
        proposed_particles = proposal_dist.sample(seed=seed2)
        observation_ll = r3._observation_log_prob(
            proposed_particles,
            observation,
            observation_matrix,
            observation_noise,
        )
        transition_ll = r3._transition_log_prob(
            particles,
            proposed_particles,
            transition_matrix,
            transition_noise,
        )
        proposal_ll = r3._optimal_proposal_log_prob(
            particles,
            proposed_particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
            sigma_chol,
        )
        unnormalized = transition_ll + observation_ll - proposal_ll + log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        log_likelihoods = log_likelihoods + increment
        log_weights = r3._filterflow_normalize(unnormalized, num_particles)
        particles = proposed_particles

    mean_log_likelihood = tf.reduce_mean(log_likelihoods)
    return mean_log_likelihood, {
        "log_likelihoods": r3._json(log_likelihoods),
        "final_log_neff": r3._json(r3._ess(log_weights)),
        "resampling_count": resampling_count,
        "max_row_residual": max_row_residual,
        "max_column_residual": max_column_residual,
    }


def _transition_matrix(theta: tf.Tensor) -> tf.Tensor:
    return tf.linalg.diag(tf.cast(theta, DTYPE)) + tf.constant(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=DTYPE,
    )


def _optimal_proposal_mean(
    prior_particles: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    observation_matrix: tf.Tensor,
    transition_cov_inv: tf.Tensor,
    observation_cov_inv: tf.Tensor,
    sigma: tf.Tensor,
) -> tf.Tensor:
    mean = tf.linalg.matvec(
        observation_matrix,
        tf.linalg.matvec(observation_cov_inv, observation),
        transpose_a=True,
    )
    mean = mean + tf.linalg.matvec(
        transition_cov_inv,
        tf.linalg.matvec(transition_matrix, prior_particles),
    )
    return tf.linalg.matvec(sigma, mean)


def _compare_surfaces(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    if bayesfilter.get("status") != "executed":
        return {
            "status": "blocked",
            "blocker": bayesfilter.get("blocker", "BayesFilter surface did not execute"),
        }
    rows = []
    first_failure = None
    for ff_row, bf_row in zip(filterflow["rows"], bayesfilter["rows"], strict=True):
        if int(ff_row["mesh_index"]) != int(bf_row["mesh_index"]):
            raise ValueError(
                f"mesh index mismatch: filterflow={ff_row['mesh_index']} "
                f"bayesfilter={bf_row['mesh_index']}"
            )
        scalar_delta = abs(
            float(bf_row["mean_log_likelihood"]) - float(ff_row["mean_log_likelihood"])
        )
        gradient_delta = [
            float(bf_value) - float(ff_value)
            for bf_value, ff_value in zip(
                bf_row["gradient_diag"],
                ff_row["gradient_diag"],
                strict=True,
            )
        ]
        max_abs_gradient_delta = max(abs(value) for value in gradient_delta)
        max_abs_filterflow_gradient = max(abs(float(value)) for value in ff_row["gradient_diag"])
        relative_gradient_delta = max_abs_gradient_delta / max(1.0, max_abs_filterflow_gradient)
        scalar_within = scalar_delta <= SCALAR_TOLERANCE
        gradient_within = (
            max_abs_gradient_delta <= GRADIENT_ABS_TOLERANCE
            or relative_gradient_delta <= GRADIENT_REL_TOLERANCE
        )
        finite = bool(
            ff_row["finite_scalar"]
            and ff_row["finite_gradient"]
            and bf_row["finite_scalar"]
            and bf_row["finite_gradient"]
        )
        row = {
            "mesh_index": ff_row["mesh_index"],
            "theta": ff_row["theta"],
            "filterflow_mean_log_likelihood": ff_row["mean_log_likelihood"],
            "bayesfilter_mean_log_likelihood": bf_row["mean_log_likelihood"],
            "scalar_delta": scalar_delta,
            "scalar_within_tolerance": scalar_within,
            "filterflow_gradient_diag": ff_row["gradient_diag"],
            "bayesfilter_gradient_diag": bf_row["gradient_diag"],
            "gradient_delta": gradient_delta,
            "max_abs_gradient_delta": max_abs_gradient_delta,
            "relative_gradient_delta": relative_gradient_delta,
            "gradient_within_tolerance": gradient_within,
            "filterflow_final_ess": ff_row.get("final_ess"),
            "bayesfilter_final_log_neff": bf_row.get("final_log_neff"),
            "bayesfilter_resampling_count": bf_row.get("resampling_count"),
            "bayesfilter_max_row_residual": bf_row.get("max_row_residual"),
            "bayesfilter_max_column_residual": bf_row.get("max_column_residual"),
            "finite": finite,
        }
        rows.append(row)
        if first_failure is None and (not scalar_within or not gradient_within or not finite):
            first_failure = {
                "mesh_index": row["mesh_index"],
                "theta": row["theta"],
                "scalar_delta": scalar_delta,
                "max_abs_gradient_delta": max_abs_gradient_delta,
                "relative_gradient_delta": relative_gradient_delta,
                "scalar_within_tolerance": scalar_within,
                "gradient_within_tolerance": gradient_within,
                "finite": finite,
            }
    scalar_deltas = tf.constant([row["scalar_delta"] for row in rows], dtype=DTYPE)
    gradient_abs_deltas = tf.constant(
        [row["max_abs_gradient_delta"] for row in rows],
        dtype=DTYPE,
    )
    gradient_rel_deltas = tf.constant(
        [row["relative_gradient_delta"] for row in rows],
        dtype=DTYPE,
    )
    summary = {
        "row_count": len(rows),
        "finite_rows": sum(1 for row in rows if row["finite"]),
        "scalar_rows_within_tolerance": sum(1 for row in rows if row["scalar_within_tolerance"]),
        "gradient_rows_within_tolerance": sum(1 for row in rows if row["gradient_within_tolerance"]),
        "max_scalar_delta": _float(tf.reduce_max(scalar_deltas)),
        "rmse_scalar_delta": _stable_rmse(scalar_deltas),
        "max_abs_gradient_delta": _float(tf.reduce_max(gradient_abs_deltas)),
        "rmse_max_abs_gradient_delta": _stable_rmse(gradient_abs_deltas),
        "max_relative_gradient_delta": _float(tf.reduce_max(gradient_rel_deltas)),
        "implementation_agreement": all(
            row["scalar_within_tolerance"] and row["gradient_within_tolerance"] and row["finite"]
            for row in rows
        ),
        "first_failure": first_failure or {"status": "no_failure"},
    }
    return {
        "status": "compared",
        "summary": summary,
        "rows": rows,
        "primary_metrics": [
            "mean_log_likelihood_scalar_delta",
            "diagonal_gradient_delta",
        ],
        "finite_gradients_alone_are_smoke_only": True,
    }


def _decision(
    comparison: dict[str, Any],
    bayesfilter: dict[str, Any],
    comparator_drift: bool,
    config: dict[str, Any],
) -> str:
    prefix = str(config["decision_prefix"])
    if comparator_drift:
        return f"{prefix}_blocked_by_comparator_drift"
    if comparison.get("status") != "compared":
        return f"{prefix}_blocked"
    if not bayesfilter.get("finite_values"):
        return f"{prefix}_nonfinite_veto"
    if comparison["summary"].get("implementation_agreement"):
        return f"{prefix}_pass"
    if comparison["summary"].get("scalar_rows_within_tolerance") == comparison["summary"].get("row_count"):
        return f"{prefix}_gradient_mismatch"
    return f"{prefix}_scalar_and_gradient_mismatch"


def _decision_name(config: dict[str, Any], suffix: str) -> str:
    return f"{config['decision_prefix']}_{suffix}"


def _evidence_contract(config: dict[str, Any] | None = None) -> dict[str, Any]:
    scope = (config or _artifact_config(full_surface=False))["scope_label"]
    return {
        "primary_comparator": "local executable float64 filterflow checkout",
        "primary_question": "cross_implementation_difference_only",
        "primary_pass": f"{scope} scalar and diagonal gradient surface values agree within tolerance",
        "scalar_tolerance": SCALAR_TOLERANCE,
        "gradient_abs_tolerance": GRADIENT_ABS_TOLERANCE,
        "gradient_rel_tolerance": GRADIENT_REL_TOLERANCE,
        "finite_gradients": "veto_gate_not_correctness_claim",
        "mathematical_correctness": "not_concluded",
    }


def _model_contract(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or _artifact_config(full_surface=False)
    mesh_size = int(config["mesh_size"])
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "transition_matrix": "A(theta)=diag(theta_1, theta_2)+[[0,1],[0,0]]",
        "transition_covariance": [[1.0 / 3.0, 0.5], [0.5, 1.0]],
        "observation_matrix": [[1.0, 0.0]],
        "observation_covariance": [[0.01]],
        "T": T,
        "batch_size": BATCH_SIZE,
        "num_particles": NUM_PARTICLES,
        "mesh_size": mesh_size,
        "row_start": config["row_start"],
        "row_count": config["row_count"],
        "row_stop_exclusive": config["row_stop_exclusive"],
        "script_default_mesh_size": 20,
        "data_seed": DATA_SEED,
        "filter_seed": FILTER_SEED,
        "epsilon": EPSILON,
        "scaling": SCALING,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "max_iter": MAX_ITERATIONS,
        "resampling_neff": RESAMPLING_NEFF,
        "optimal_proposal": True,
        "dtype": FILTERFLOW_REFERENCE_DTYPE,
    }


def _scalar_contract() -> dict[str, str]:
    return {
        "filterflow_scalar": "tf.reduce_mean(final_state.log_likelihoods)",
        "bayesfilter_scalar": "tf.reduce_mean(log_likelihoods)",
        "resampling_correction": "False",
        "batch_size_note": "batch_size=1 makes mean and single-batch total equal",
        "per_time_normalization": "not used",
        "sign": "positive log-likelihood convention as emitted by filterflow",
    }


def _gradient_contract() -> dict[str, str]:
    return {
        "filterflow_gradient": "tf.linalg.diag_part(tape.gradient(scalar, transition_matrix_variable))",
        "bayesfilter_gradient": "tape.gradient(scalar, theta_variable)",
        "theta_parameterization": "transition_matrix=diag(theta)+[[0,1],[0,0]]",
        "correction_term": "not included",
        "finite_difference_status": "not primary in this runner",
    }


def _tolerances() -> dict[str, float]:
    return {
        "scalar": SCALAR_TOLERANCE,
        "gradient_abs": GRADIENT_ABS_TOLERANCE,
        "gradient_rel": GRADIENT_REL_TOLERANCE,
    }


def _compact_filterflow(filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return filterflow
    return {
        "status": filterflow["status"],
        "surface_contract": filterflow["surface_contract"],
        "settings": filterflow["settings"],
        "observation_checksum": filterflow["observation_checksum"],
        "initial_particles_checksum": filterflow["initial_particles_checksum"],
        "first_row": filterflow["rows"][0],
        "last_row": filterflow["rows"][-1],
        "package_versions": filterflow["package_versions"],
        "cpu_only_manifest": filterflow["cpu_only_manifest"],
        "stderr_excerpt": filterflow.get("stderr_excerpt", ""),
    }


def _compact_bayesfilter(bayesfilter: dict[str, Any] | None) -> dict[str, Any] | None:
    if bayesfilter is None:
        return None
    return {
        "status": bayesfilter["status"],
        "backend": bayesfilter["backend"],
        "surface_contract": bayesfilter["surface_contract"],
        "finite_values": bayesfilter["finite_values"],
        "first_row": bayesfilter["rows"][0],
        "last_row": bayesfilter["rows"][-1],
        "cpu_only_manifest": bayesfilter["cpu_only_manifest"],
    }


def _decision_table(
    decision: str,
    comparison: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    config = config or _artifact_config(full_surface=False)
    summary = comparison.get("summary", {})
    scope = str(config["scope_label"])
    if decision == _decision_name(config, "pass"):
        primary = (
            f"{summary.get('gradient_rows_within_tolerance')} of {summary.get('row_count')} "
            f"{scope} rows within gradient tolerance"
        )
        veto = "pass"
        if config["full_surface"]:
            next_action = "add Kalman finite-difference context or multi-seed smoke only if needed"
        else:
            next_action = "rerun on the full mesh_size=20 surface or add Kalman finite-difference context"
    elif decision == _decision_name(config, "gradient_mismatch"):
        primary = f"scalar agrees, gradient mismatch: {summary.get('first_failure')}"
        veto = "gradient mismatch observed"
        next_action = "localize the first failing mesh row by per-time gradient contributions"
    elif decision == _decision_name(config, "scalar_and_gradient_mismatch"):
        primary = f"scalar and gradient mismatch: {summary.get('first_failure')}"
        veto = "scalar mismatch observed"
        next_action = "repair scalar mismatch before interpreting gradient deltas"
    else:
        primary = comparison.get("blocker", "blocked or vetoed")
        veto = decision
        next_action = "repair the blocker before using this artifact as evidence"
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": veto,
            "main_uncertainty": f"{scope} only; no analytic-gradient correctness is concluded",
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
        "scalar_contract",
        "gradient_contract",
        "filterflow_surface",
        "bayesfilter_surface",
        "comparison",
        "path_boundary_manifest",
        "run_manifest",
        "decision_table",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    allowed = set()
    for config in (_artifact_config(full_surface=False), _artifact_config(full_surface=True)):
        allowed.update(
            {
                _decision_name(config, "filterflow_blocker"),
                _decision_name(config, "blocked_by_comparator_drift"),
                _decision_name(config, "blocked"),
                _decision_name(config, "nonfinite_veto"),
                _decision_name(config, "pass"),
                _decision_name(config, "gradient_mismatch"),
                _decision_name(config, "scalar_and_gradient_mismatch"),
            }
        )
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    filterflow_surface = payload["filterflow_surface"]
    if filterflow_surface.get("status") == "executed":
        _validate_cpu(filterflow_surface["cpu_only_manifest"], "filterflow surface")
        settings = filterflow_surface["settings"]
        if settings.get("optimal_proposal") is not True:
            raise ValueError("filterflow smoothness default optimal_proposal=True not used")
        if settings.get("resampling_correction") is not False:
            raise ValueError("filterflow correction term unexpectedly enabled")
        if settings.get("dtype") != FILTERFLOW_REFERENCE_DTYPE:
            raise ValueError("filterflow dtype mismatch")
    bayesfilter_surface = payload["bayesfilter_surface"]
    if bayesfilter_surface is not None:
        _validate_cpu(bayesfilter_surface["cpu_only_manifest"], "BayesFilter surface")
    if payload["decision"].endswith("_pass"):
        if not payload["comparison"]["summary"].get("implementation_agreement"):
            raise ValueError("pass decision without implementation agreement")
    if "reduce_mean(final_state.log_likelihoods)" not in payload["scalar_contract"]["filterflow_scalar"]:
        raise ValueError("wrong scalar contract")
    if "diag_part" not in payload["gradient_contract"]["filterflow_gradient"]:
        raise ValueError("wrong gradient contract")
    if "reproducibility_digest" not in payload:
        raise ValueError("missing reproducibility digest")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# FilterFlow Float64 Smoothness Gradient Surface Comparison",
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
            "## Scalar Contract",
            "",
            _key_value_table(payload["scalar_contract"]),
            "",
            "## Gradient Contract",
            "",
            _key_value_table(payload["gradient_contract"]),
            "",
            "## Model Contract",
            "",
            _key_value_table(payload["model_contract"]),
            "",
            "## Comparison Summary",
            "",
            _json_block(payload["comparison"].get("summary")),
            "",
            "## First And Last Rows",
            "",
            "### FilterFlow",
            "",
            _json_block(
                {
                    "first_row": payload["filterflow_surface"].get("first_row"),
                    "last_row": payload["filterflow_surface"].get("last_row"),
                }
            ),
            "",
            "### BayesFilter",
            "",
            _json_block(
                {
                    "first_row": (
                        payload["bayesfilter_surface"].get("first_row")
                        if payload["bayesfilter_surface"] is not None
                        else None
                    ),
                    "last_row": (
                        payload["bayesfilter_surface"].get("last_row")
                        if payload["bayesfilter_surface"] is not None
                        else None
                    ),
                }
            ),
            "",
            "## Interpretation",
            "",
            _interpretation(payload),
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    summary = payload["comparison"].get("summary", {})
    if decision.endswith("_pass"):
        scope = payload["evidence_contract"]["primary_pass"].split(" scalar")[0]
        return (
            "BayesFilter and the local float64 FilterFlow executable agree on "
            f"the {scope} smoothness scalar and diagonal gradient surface. Max "
            f"scalar delta: `{summary['max_scalar_delta']}`. Max absolute "
            f"gradient delta: `{summary['max_abs_gradient_delta']}`."
        )
    if decision.endswith("_gradient_mismatch"):
        return (
            "The scalar surface still agrees, but a gradient mismatch remains. "
            f"First failure: `{summary['first_failure']}`."
        )
    if decision.endswith("_scalar_and_gradient_mismatch"):
        return (
            "A scalar mismatch appeared in the gradient runner; the scalar "
            f"must be localized first. First failure: `{summary['first_failure']}`."
        )
    return "The gradient-surface comparison did not produce evidence because a blocker or veto fired."


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


def _non_implications(config: dict[str, Any] | None = None) -> list[str]:
    config = config or _artifact_config(full_surface=False)
    items = step._non_implications() + [
        "No correctness claim is made for either implementation.",
        "No analytic smoothness-gradient correctness is concluded.",
        "No production dtype default is concluded.",
        "Finite gradients alone are smoke evidence only.",
    ]
    if not config["full_surface"]:
        items.insert(2, "No full mesh_size=20 surface agreement is concluded.")
    return items


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _stable_rmse(values: tf.Tensor) -> float:
    values = tf.cast(values, DTYPE)
    scale = tf.reduce_max(tf.abs(values))
    if float(scale.numpy()) == 0.0:
        return 0.0
    normalized = values / scale
    return _float(scale * tf.sqrt(tf.reduce_mean(normalized * normalized)))


if __name__ == "__main__":
    raise SystemExit(main())
