"""Float64 smoothness scalar-surface difference audit.

This runner compares BayesFilter TF/TFP against the local float64 FilterFlow
executable on the scalar used by scripts/simple_linear_smoothness.py:
tf.reduce_mean(final_state.log_likelihoods). It is a difference audit only.
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
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-smoothness-scalar-surface-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-smoothness-scalar-surface-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_smoothness_scalar_surface_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-smoothness-scalar-surface-2026-06-03.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

T = 100
BATCH_SIZE = 1
NUM_PARTICLES = 50
MESH_SIZE = 4
DATA_SEED = 123
FILTER_SEED = 1234
EPSILON = 0.25
SCALING = 0.85
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 500
RESAMPLING_NEFF = 0.9999
SCALAR_TOLERANCE = 5e-5


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
    filterflow = _filterflow_surface_subprocess()
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_smoothness_scalar_surface_filterflow_blocker",
            filterflow.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow,
            reference_status,
        )

    bayesfilter = _bayesfilter_surface(filterflow)
    comparison = _compare_surfaces(filterflow, bayesfilter)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(comparison, bayesfilter, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "Does BayesFilter match the local float64 FilterFlow executable "
            "on the bounded simple_linear_smoothness scalar surface?"
        ),
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "scalar_contract": _scalar_contract(),
        "filterflow_surface": _compact_filterflow(filterflow),
        "bayesfilter_surface": _compact_bayesfilter(bayesfilter),
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": {"scalar": SCALAR_TOLERANCE},
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_smoothness_scalar_surface_tf"
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
        "scalar_contract": _scalar_contract(),
        "filterflow_surface": filterflow,
        "bayesfilter_surface": None,
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": {"scalar": SCALAR_TOLERANCE},
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_smoothness_scalar_surface_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _filterflow_surface_subprocess() -> dict[str, Any]:
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
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_surface_script()],
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
            "blocker": "filterflow smoothness surface subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_SMOOTHNESS_SURFACE_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_SMOOTHNESS_SURFACE_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow smoothness surface JSON sentinels missing",
            "stdout_excerpt": stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(stdout[start + len("FILTERFLOW_SMOOTHNESS_SURFACE_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_surface_script() -> str:
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
        from scripts.simple_linear_smoothness import get_data

        DTYPE = tf.float64
        NP_DTYPE = np.float64
        T = {T}
        BATCH_SIZE = {BATCH_SIZE}
        N = {NUM_PARTICLES}
        MESH_SIZE = {MESH_SIZE}
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
        initial_particles = tf.convert_to_tensor(initial_particles_np, dtype=DTYPE)
        initial_state = State(initial_particles)
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
        for index, theta in enumerate(mesh):
            matrix = tf.linalg.diag(tf.convert_to_tensor(theta, dtype=DTYPE)) + tf.constant(
                [[0.0, 1.0], [0.0, 0.0]],
                dtype=DTYPE,
            )
            modifiable_transition_matrix.assign(matrix)
            final_state = smc(
                initial_state,
                tf.data.Dataset.from_tensor_slices(observations),
                n_observations=tf.constant(T),
                return_final=True,
                seed=tf.constant(FILTER_SEED, dtype=tf.int32),
            )
            mean_log_likelihood = tf.reduce_mean(final_state.log_likelihoods)
            rows.append({{
                "mesh_index": int(index),
                "theta": [float(theta[0]), float(theta[1])],
                "mean_log_likelihood": scalar(mean_log_likelihood),
                "final_ess": to_json(final_state.ess),
                "finite_scalar": bool(tf.math.is_finite(mean_log_likelihood).numpy()),
            }})

        payload = {{
            "status": "executed",
            "surface_contract": "simple_linear_smoothness_scalar_without_gradient_tape",
            "settings": {{
                "T": T,
                "batch_size": BATCH_SIZE,
                "n_particles": N,
                "mesh_size": MESH_SIZE,
                "script_default_mesh_size": 20,
                "data_seed": DATA_SEED,
                "filter_seed": FILTER_SEED,
                "epsilon": EPSILON,
                "scaling": SCALING,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iter": MAX_ITERATIONS,
                "resampling_neff": RESAMPLING_NEFF,
                "optimal_proposal": True,
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
        print("FILTERFLOW_SMOOTHNESS_SURFACE_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_SMOOTHNESS_SURFACE_JSON_END")
        """
    )


def _bayesfilter_surface(filterflow: dict[str, Any]) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        model = filterflow["model"]
        rows = []
        for index, theta in enumerate(model["mesh"]):
            row = _bayesfilter_surface_row(
                mesh_index=index,
                theta=theta,
                model=model,
                settings=filterflow["settings"],
            )
            rows.append(row)
        finite = all(row["finite_scalar"] for row in rows)
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "surface_contract": "same_scalar_as_filterflow_smoothness_reduce_mean_final_log_likelihoods",
            "rows": rows,
            "finite_values": finite,
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_surface_row(
    *,
    mesh_index: int,
    theta: list[float],
    model: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    transition_matrix = _transition_matrix(theta)
    observation_matrix = tf.constant(model["observation_matrix"], dtype=DTYPE)
    transition_chol = tf.constant(model["transition_covariance_chol"], dtype=DTYPE)
    observation_chol = tf.constant(model["observation_covariance_chol"], dtype=DTYPE)
    observations = tf.constant(model["observations"], dtype=DTYPE)
    particles = tf.constant(model["initial_particles"], dtype=DTYPE)
    num_particles = int(settings["n_particles"])
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
    finite_steps = True
    for time_index in range(T):
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
        finite_steps = finite_steps and bool(
            tf.reduce_all(tf.math.is_finite(particles)).numpy()
            and tf.reduce_all(tf.math.is_finite(log_weights)).numpy()
            and tf.reduce_all(tf.math.is_finite(log_likelihoods)).numpy()
        )

    mean_log_likelihood = tf.reduce_mean(log_likelihoods)
    return {
        "mesh_index": mesh_index,
        "theta": [float(theta[0]), float(theta[1])],
        "mean_log_likelihood": _float(mean_log_likelihood),
        "log_likelihoods": r3._json(log_likelihoods),
        "final_log_neff": r3._json(r3._ess(log_weights)),
        "resampling_count": resampling_count,
        "max_row_residual": max_row_residual,
        "max_column_residual": max_column_residual,
        "finite_scalar": bool(tf.math.is_finite(mean_log_likelihood).numpy() and finite_steps),
    }


def _transition_matrix(theta: list[float]) -> tf.Tensor:
    return tf.linalg.diag(tf.constant(theta, dtype=DTYPE)) + tf.constant(
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
        scalar_delta = abs(
            float(bf_row["mean_log_likelihood"]) - float(ff_row["mean_log_likelihood"])
        )
        within = scalar_delta <= SCALAR_TOLERANCE
        row = {
            "mesh_index": ff_row["mesh_index"],
            "theta": ff_row["theta"],
            "filterflow_mean_log_likelihood": ff_row["mean_log_likelihood"],
            "bayesfilter_mean_log_likelihood": bf_row["mean_log_likelihood"],
            "scalar_delta": scalar_delta,
            "within_tolerance": within,
            "filterflow_final_ess": ff_row.get("final_ess"),
            "bayesfilter_final_log_neff": bf_row.get("final_log_neff"),
            "bayesfilter_resampling_count": bf_row.get("resampling_count"),
            "bayesfilter_max_row_residual": bf_row.get("max_row_residual"),
            "bayesfilter_max_column_residual": bf_row.get("max_column_residual"),
            "finite": bool(ff_row["finite_scalar"] and bf_row["finite_scalar"]),
        }
        rows.append(row)
        if first_failure is None and (not within or not row["finite"]):
            first_failure = {
                "mesh_index": row["mesh_index"],
                "theta": row["theta"],
                "scalar_delta": scalar_delta,
                "within_tolerance": within,
                "finite": row["finite"],
            }
    scalar_deltas = tf.constant([row["scalar_delta"] for row in rows], dtype=DTYPE)
    summary = {
        "row_count": len(rows),
        "finite_rows": sum(1 for row in rows if row["finite"]),
        "rows_within_tolerance": sum(1 for row in rows if row["within_tolerance"]),
        "max_scalar_delta": _float(tf.reduce_max(scalar_deltas)),
        "rmse_scalar_delta": _float(tf.sqrt(tf.reduce_mean(scalar_deltas * scalar_deltas))),
        "implementation_agreement": all(row["within_tolerance"] and row["finite"] for row in rows),
        "first_failure": first_failure or {"status": "no_failure"},
    }
    return {
        "status": "compared",
        "summary": summary,
        "rows": rows,
        "primary_metric": "mean_log_likelihood_scalar_delta",
        "residuals_are_explanatory_only": True,
    }


def _decision(
    comparison: dict[str, Any],
    bayesfilter: dict[str, Any],
    comparator_drift: bool,
) -> str:
    if comparator_drift:
        return "filterflow_float64_smoothness_scalar_surface_blocked_by_comparator_drift"
    if comparison.get("status") != "compared":
        return "filterflow_float64_smoothness_scalar_surface_blocked"
    if not bayesfilter.get("finite_values"):
        return "filterflow_float64_smoothness_scalar_surface_nonfinite_veto"
    if comparison["summary"].get("implementation_agreement"):
        return "filterflow_float64_smoothness_scalar_surface_pass"
    return "filterflow_float64_smoothness_scalar_surface_mismatch"


def _evidence_contract() -> dict[str, Any]:
    return {
        "primary_comparator": "local executable float64 filterflow checkout",
        "primary_question": "cross_implementation_difference_only",
        "primary_pass": "bounded scalar surface values agree within tolerance",
        "scalar_tolerance": SCALAR_TOLERANCE,
        "residuals": "explanatory_only_not_promotion_or_veto",
        "gradient_correctness": "not_concluded",
        "mathematical_correctness": "not_concluded",
    }


def _model_contract() -> dict[str, Any]:
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "transition_matrix": "A(theta)=diag(theta_1, theta_2)+[[0,1],[0,0]]",
        "transition_covariance": [[1.0 / 3.0, 0.5], [0.5, 1.0]],
        "observation_matrix": [[1.0, 0.0]],
        "observation_covariance": [[0.01]],
        "T": T,
        "batch_size": BATCH_SIZE,
        "num_particles": NUM_PARTICLES,
        "mesh_size": MESH_SIZE,
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
        "batch_size_note": "batch_size=1 makes mean and single-batch total equal",
        "per_time_normalization": "not used",
        "sign": "positive log-likelihood convention as emitted by filterflow",
        "gradient_status": "not tested in this runner",
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


def _decision_table(decision: str, comparison: dict[str, Any]) -> list[dict[str, str]]:
    summary = comparison.get("summary", {})
    if decision == "filterflow_float64_smoothness_scalar_surface_pass":
        primary = (
            f"{summary.get('rows_within_tolerance')} of {summary.get('row_count')} "
            "bounded mesh rows within scalar tolerance"
        )
        veto = "pass"
        next_action = "repeat the scalar-surface comparison with gradient recording"
    elif decision == "filterflow_float64_smoothness_scalar_surface_mismatch":
        primary = f"first scalar mismatch: {summary.get('first_failure')}"
        veto = "implementation mismatch observed"
        next_action = "localize the first failing mesh row by per-time increments"
    else:
        primary = comparison.get("blocker", "blocked or vetoed")
        veto = decision
        next_action = "repair the blocker before using this artifact as evidence"
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": veto,
            "main_uncertainty": "bounded mesh only; gradients are not tested here",
            "next_justified_action": next_action,
            "not_concluded": "correctness of either implementation, gradient correctness, production readiness",
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
    allowed = {
        "filterflow_float64_smoothness_scalar_surface_filterflow_blocker",
        "filterflow_float64_smoothness_scalar_surface_blocked_by_comparator_drift",
        "filterflow_float64_smoothness_scalar_surface_blocked",
        "filterflow_float64_smoothness_scalar_surface_nonfinite_veto",
        "filterflow_float64_smoothness_scalar_surface_pass",
        "filterflow_float64_smoothness_scalar_surface_mismatch",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    filterflow_surface = payload["filterflow_surface"]
    if filterflow_surface.get("status") == "executed":
        _validate_cpu(filterflow_surface["cpu_only_manifest"], "filterflow surface")
        if filterflow_surface["settings"].get("optimal_proposal") is not True:
            raise ValueError("filterflow smoothness default optimal_proposal=True not used")
        if filterflow_surface["settings"].get("dtype") != FILTERFLOW_REFERENCE_DTYPE:
            raise ValueError("filterflow dtype mismatch")
    bayesfilter_surface = payload["bayesfilter_surface"]
    if bayesfilter_surface is not None:
        _validate_cpu(bayesfilter_surface["cpu_only_manifest"], "BayesFilter surface")
    if payload["decision"] == "filterflow_float64_smoothness_scalar_surface_pass":
        if not payload["comparison"]["summary"].get("implementation_agreement"):
            raise ValueError("pass decision without implementation agreement")
    if "reduce_mean(final_state.log_likelihoods)" not in payload["scalar_contract"]["filterflow_scalar"]:
        raise ValueError("wrong scalar contract")
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
        "# FilterFlow Float64 Smoothness Scalar Surface Comparison",
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
    if decision == "filterflow_float64_smoothness_scalar_surface_pass":
        summary = payload["comparison"]["summary"]
        return (
            "BayesFilter and the local float64 FilterFlow executable agree on "
            f"the bounded smoothness scalar surface. Max scalar delta: "
            f"`{summary['max_scalar_delta']}`. Transport residuals remain "
            "explanatory only in this artifact."
        )
    if decision == "filterflow_float64_smoothness_scalar_surface_mismatch":
        return (
            "A scalar-surface mismatch remains. The next step is to localize "
            f"the first row: `{payload['comparison']['summary']['first_failure']}`."
        )
    return "The scalar-surface comparison did not produce evidence because a blocker or veto fired."


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
        "No smoothness-surface gradient correctness is concluded.",
        "No full mesh_size=20 surface agreement is concluded.",
        "No production dtype default is concluded.",
        "Transport residual magnitude is explanatory only for this difference audit.",
    ]


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


if __name__ == "__main__":
    raise SystemExit(main())
