"""Same-model gradient reconciliation for filterflow smoothness."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
import textwrap
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
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


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-annealed-transport-remaining-gaps-closure-plan-2026-06-01.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_smoothness_same_model_gradient_reconciliation_2026-06-01.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-smoothness-same-model-gradient-reconciliation-2026-06-01.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
T = 100
MESH_SIZE = 4
NUM_PARTICLES = 25
BATCH_SIZE = 1
EPSILON = 0.25
SCALING = 0.85
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITER = 200
DATA_SEED = 123
FILTER_SEED = 1234
DIFF_EPSILON = 1e-2


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
    reference = _filterflow_smoothness_reference()
    if reference["status"] != "executed":
        return _blocked_payload(reference)
    observations = tf.constant(reference["observations"], dtype=DTYPE)
    initial_particles = tf.constant(reference["initial_particles"], dtype=DTYPE)
    mesh = tf.constant(reference["mesh"], dtype=DTYPE)
    base_noises = tf.random.stateless_normal(
        [T, BATCH_SIZE, NUM_PARTICLES, 2],
        seed=tf.constant([FILTER_SEED, 20260601], dtype=tf.int32),
        dtype=DTYPE,
    )
    bayesfilter_rows = []
    for theta in tf.unstack(mesh, axis=0):
        bayesfilter_rows.append(
            _bayesfilter_value_grad(
                observations=observations,
                initial_particles=initial_particles,
                base_noises=base_noises,
                theta_1=float(theta[0].numpy()),
                theta_2=float(theta[1].numpy()),
            )
        )
    filterflow_rows = _reference_rows(reference)
    comparison = _comparison(filterflow_rows, bayesfilter_rows)
    scalar_contract = _scalar_contract(reference)
    seed_contract = _seed_contract(reference)
    caveat_ledger = _caveat_ledger(comparison)
    decision_table = _decision_table(comparison)
    decision = _decision(comparison)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Same-model gradient reconciliation against executable filterflow smoothness.",
        "model_contract": _model_contract(),
        "seed_contract": seed_contract,
        "scalar_contract": scalar_contract,
        "finite_difference_contract": {
            "reference": "filterflow simple_linear_smoothness.get_surface_kf",
            "scheme": "one_sided_forward_difference",
            "diff_epsilon": DIFF_EPSILON,
            "centered_difference_status": "not_used",
        },
        "randomness_contract": {
            "filterflow_randomness_status": "extracted_observations_and_initial_particles",
            "bayesfilter_transition_noise_status": "stateless_common_random_numbers_surrogate",
            "bit_identical_filterflow_noise": False,
            "interpretation": (
                "This closes the prior wrong-model gradient harness gap, but does not prove "
                "bit-identical stochastic-gradient equality against filterflow internals."
            ),
        },
        "filterflow_reference": {
            "status": reference["status"],
            "command": reference["command"],
            "python": reference.get("python"),
            "tensorflow": reference.get("tensorflow"),
            "numpy": reference.get("numpy"),
        },
        "rows": comparison["rows"],
        "summary": comparison["summary"],
        "caveat_ledger": caveat_ledger,
        "decision_table": decision_table,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_smoothness_same_model_gradient_reconciliation_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _blocked_payload(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": "same_model_gradient_reconciliation_blocked_filterflow_reference",
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Same-model gradient reconciliation against executable filterflow smoothness.",
        "model_contract": _model_contract(),
        "seed_contract": {"status": "blocked", "reason": reference.get("status")},
        "scalar_contract": {"status": "blocked", "reason": reference.get("status")},
        "filterflow_reference": reference,
        "caveat_ledger": [{"id": "filterflow_reference_blocked", "status": reference.get("status")}],
        "decision_table": [
            {
                "decision": "blocked",
                "primary_criterion_status": "filterflow reference extraction failed",
                "veto_status": "veto",
                "next_action": "repair local filterflow reference environment",
                "not_concluded": "gradient reconciliation",
            }
        ],
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_smoothness_same_model_gradient_reconciliation_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _filterflow_smoothness_reference() -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {"status": "blocked_missing_filterflow_env", "python": str(FILTERFLOW_ENV_PYTHON)}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    command = (
        "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl "
        "PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <same-model smoothness extraction>"
    )
    if completed.returncode != 0:
        return {
            "status": "blocked_filterflow_subprocess_failed",
            "returncode": completed.returncode,
            "command": command,
            "stdout_excerpt": completed.stdout[-3000:],
            "stderr_excerpt": completed.stderr[-3000:],
        }
    start = completed.stdout.rfind("SAME_MODEL_REFERENCE_JSON_BEGIN")
    end = completed.stdout.rfind("SAME_MODEL_REFERENCE_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked_missing_json_sentinels",
            "command": command,
            "stdout_excerpt": completed.stdout[-3000:],
            "stderr_excerpt": completed.stderr[-3000:],
        }
    payload = json.loads(completed.stdout[start + len("SAME_MODEL_REFERENCE_JSON_BEGIN"):end].strip())
    payload["command"] = command
    payload["stderr_excerpt"] = completed.stderr[-1500:]
    return payload


def _filterflow_script() -> str:
    return textwrap.dedent(
        f"""
        import inspect
        import json
        import os
        import tensorflow as tf
        np = __import__("numpy")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_smoothness import get_data, get_surface, get_surface_kf

        T = {T}
        MESH_SIZE = {MESH_SIZE}
        N_PARTICLES = {NUM_PARTICLES}
        BATCH_SIZE = {BATCH_SIZE}
        EPSILON = {EPSILON}
        SCALING = {SCALING}
        CONVERGENCE_THRESHOLD = {CONVERGENCE_THRESHOLD}
        MAX_ITER = {MAX_ITER}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}
        DIFF_EPSILON = {DIFF_EPSILON}

        transition_matrix = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        transition_covariance = np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]], dtype=np.float32)
        observation_matrix = np.array([[1.0, 0.0]], dtype=np.float32)
        observation_covariance = np.array([[0.01]], dtype=np.float32)
        x_linspace = np.linspace(0.95, 1.0, MESH_SIZE).astype(np.float32)
        y_linspace = np.linspace(0.95, 1.0, MESH_SIZE).astype(np.float32)
        mesh = np.asanyarray([(x, y) for x in x_linspace for y in y_linspace])

        rng = np.random.RandomState(seed=DATA_SEED)
        data, kf = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T,
            rng,
        )
        observation_dataset = tf.data.Dataset.from_tensor_slices(data)
        initial_particles = rng.normal(0.0, 0.01, [BATCH_SIZE, N_PARTICLES, 2]).astype(np.float32)
        initial_state = State(initial_particles)
        modifiable_transition_matrix = tf.Variable(transition_matrix, trainable=False)
        smc = make_filter(
            tf.convert_to_tensor(observation_matrix),
            modifiable_transition_matrix,
            tf.linalg.cholesky(observation_covariance),
            tf.linalg.cholesky(transition_covariance),
            RegularisedTransform(
                epsilon=EPSILON,
                scaling=SCALING,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                max_iter=MAX_ITER,
            ),
            NeffCriterion(0.9999, True),
            optimal_proposal=False,
        )
        log_likelihoods, gradients = get_surface(
            mesh,
            modifiable_transition_matrix,
            smc,
            initial_state,
            False,
            observation_dataset,
            tf.constant(T),
            tf.constant(FILTER_SEED),
            False,
        )
        kalman_ll, kalman_grad = get_surface_kf(mesh, kf, data, DIFF_EPSILON, False)
        payload = {{
            "status": "executed",
            "python": os.sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "settings": {{
                "T": T,
                "mesh_size": MESH_SIZE,
                "n_particles": N_PARTICLES,
                "batch_size": BATCH_SIZE,
                "epsilon": EPSILON,
                "scaling": SCALING,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iter": MAX_ITER,
                "data_seed": DATA_SEED,
                "filter_seed": FILTER_SEED,
                "diff_epsilon": DIFF_EPSILON,
                "resampling_neff": 0.9999,
                "optimal_proposal": False,
            }},
            "model_contract": {{
                "transition_matrix_base": transition_matrix.astype(float).tolist(),
                "transition_covariance": transition_covariance.astype(float).tolist(),
                "observation_matrix": observation_matrix.astype(float).tolist(),
                "observation_covariance": observation_covariance.astype(float).tolist(),
            }},
            "observations": data.astype(float).tolist(),
            "initial_particles": initial_particles.astype(float).tolist(),
            "mesh": mesh.astype(float).tolist(),
            "observation_checksum": float(np.sum(data.astype(np.float64))),
            "initial_particles_checksum": float(np.sum(initial_particles.astype(np.float64))),
            "filterflow_log_likelihoods": log_likelihoods.numpy().astype(float).tolist(),
            "filterflow_gradients": gradients.numpy().astype(float).tolist(),
            "kalman_log_likelihoods": kalman_ll.astype(float).tolist(),
            "kalman_fd_gradients": kalman_grad.astype(float).tolist(),
        }}
        print("SAME_MODEL_REFERENCE_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("SAME_MODEL_REFERENCE_JSON_END")
        """
    )


def _bayesfilter_value_grad(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    base_noises: tf.Tensor,
    theta_1: float,
    theta_2: float,
) -> dict[str, Any]:
    theta = tf.Variable([theta_1, theta_2], dtype=DTYPE)
    with tf.GradientTape() as tape:
        mean_ll = _bayesfilter_mean_log_likelihood(
            observations=observations,
            initial_particles=initial_particles,
            base_noises=base_noises,
            theta=theta,
        )
    grad = tape.gradient(mean_ll, theta)
    total_ll = mean_ll * tf.cast(BATCH_SIZE, DTYPE)
    return {
        "theta_1": theta_1,
        "theta_2": theta_2,
        "mean_log_likelihood": _float(mean_ll),
        "total_log_likelihood": _float(total_ll),
        "per_time_mean_log_likelihood": _float(mean_ll / tf.constant(T, DTYPE)),
        "gradient_mean_log_likelihood": tf.cast(grad, DTYPE).numpy().tolist(),
        "gradient_total_log_likelihood": tf.cast(grad * tf.cast(BATCH_SIZE, DTYPE), DTYPE).numpy().tolist(),
        "finite_value": bool(tf.math.is_finite(mean_ll).numpy()),
        "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(grad)).numpy()),
    }


def _bayesfilter_mean_log_likelihood(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    base_noises: tf.Tensor,
    theta: tf.Tensor,
) -> tf.Tensor:
    particles = tf.cast(initial_particles, DTYPE)
    log_weights = tf.fill(
        [BATCH_SIZE, NUM_PARTICLES],
        -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)),
    )
    log_likelihoods = tf.zeros([BATCH_SIZE], DTYPE)
    transition_chol = tf.linalg.cholesky(_transition_covariance())
    observation_variance = tf.constant(0.01, DTYPE)
    for time_index in range(T):
        weights = tf.exp(log_weights)
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        do_resample = ess <= tf.constant(0.9999 * NUM_PARTICLES, DTYPE)
        resampled = annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=EPSILON,
            scaling=SCALING,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            max_iterations=MAX_ITER,
            ess_mask=do_resample,
        )
        particles = tf.reshape(resampled.particles, [BATCH_SIZE, NUM_PARTICLES, 2])
        log_weights = tf.reshape(resampled.log_weights, [BATCH_SIZE, NUM_PARTICLES])
        a_matrix = _transition_matrix(theta)
        mean = tf.einsum("ij,bnj->bni", a_matrix, particles)
        particles = mean + tf.einsum("ij,tbnj->tbni", transition_chol, base_noises)[time_index]
        residual = tf.cast(observations[time_index], DTYPE)[None, None, :] - particles[:, :, :1]
        obs_logp = _univariate_normal_log_prob(residual[:, :, 0], observation_variance)
        unnormalized = log_weights + obs_logp
        normalizer = tf.reduce_logsumexp(unnormalized, axis=1)
        log_likelihoods = log_likelihoods + normalizer
        log_weights = _normalize_log_weights(unnormalized, axis=1)
    return tf.reduce_mean(log_likelihoods)


def _normalize_log_weights(log_weights: tf.Tensor, axis: int) -> tf.Tensor:
    normalized = log_weights - tf.reduce_logsumexp(log_weights, axis=axis, keepdims=True)
    clipped = tf.clip_by_value(normalized, tf.constant(-1e3, DTYPE), tf.constant(0.0, DTYPE))
    threshold = tf.maximum(
        tf.constant(-13.8, DTYPE),
        tf.constant(-4.0, DTYPE) * tf.cast(NUM_PARTICLES, DTYPE),
    )
    mask = clipped < threshold
    return tf.stop_gradient(tf.where(mask, clipped, tf.zeros_like(clipped))) + tf.where(
        mask,
        tf.zeros_like(clipped),
        clipped,
    )


def _univariate_normal_log_prob(residual: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE))
        + tf.math.log(variance)
        + residual * residual / variance
    )


def _transition_matrix(theta: tf.Tensor) -> tf.Tensor:
    return tf.stack(
        [
            tf.stack([theta[0], tf.constant(1.0, DTYPE)]),
            tf.stack([tf.constant(0.0, DTYPE), theta[1]]),
        ]
    )


def _transition_covariance() -> tf.Tensor:
    return tf.constant([[1.0 / 3.0, 0.5], [0.5, 1.0]], DTYPE)


def _reference_rows(reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, mesh_row in enumerate(reference["mesh"]):
        rows.append(
            {
                "theta_1": float(mesh_row[0]),
                "theta_2": float(mesh_row[1]),
                "filterflow_mean_log_likelihood": float(reference["filterflow_log_likelihoods"][index]),
                "filterflow_gradient": [
                    float(value) for value in reference["filterflow_gradients"][index]
                ],
                "kalman_total_log_likelihood": float(reference["kalman_log_likelihoods"][index]),
                "kalman_forward_fd_gradient": [
                    float(value) for value in reference["kalman_fd_gradients"][index]
                ],
            }
        )
    return rows


def _comparison(filterflow_rows: list[dict[str, Any]], bayesfilter_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for ref, bf in zip(filterflow_rows, bayesfilter_rows):
        ff_grad = tf.constant(ref["filterflow_gradient"], DTYPE)
        kf_grad = tf.constant(ref["kalman_forward_fd_gradient"], DTYPE)
        bf_grad = tf.constant(bf["gradient_mean_log_likelihood"], DTYPE)
        rows.append(
            {
                **ref,
                **{f"bayesfilter_{key}": value for key, value in bf.items() if key not in {"theta_1", "theta_2"}},
                "bayesfilter_minus_filterflow_likelihood": (
                    bf["mean_log_likelihood"] - ref["filterflow_mean_log_likelihood"]
                ),
                "bayesfilter_minus_kalman_likelihood": (
                    bf["mean_log_likelihood"] - ref["kalman_total_log_likelihood"]
                ),
                "filterflow_minus_kalman_likelihood": (
                    ref["filterflow_mean_log_likelihood"] - ref["kalman_total_log_likelihood"]
                ),
                "gradient_delta_bayesfilter_filterflow": (bf_grad - ff_grad).numpy().tolist(),
                "gradient_delta_bayesfilter_kalman": (bf_grad - kf_grad).numpy().tolist(),
                "gradient_delta_filterflow_kalman": (ff_grad - kf_grad).numpy().tolist(),
                "finite": bool(bf["finite_value"] and bf["finite_gradient"]),
            }
        )
    summary = _summary(rows)
    return {"rows": rows, "summary": summary}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bf_ll = tf.constant([row["bayesfilter_mean_log_likelihood"] for row in rows], DTYPE)
    ff_ll = tf.constant([row["filterflow_mean_log_likelihood"] for row in rows], DTYPE)
    kf_ll = tf.constant([row["kalman_total_log_likelihood"] for row in rows], DTYPE)
    bf_grad = tf.constant([row["bayesfilter_gradient_mean_log_likelihood"] for row in rows], DTYPE)
    ff_grad = tf.constant([row["filterflow_gradient"] for row in rows], DTYPE)
    kf_grad = tf.constant([row["kalman_forward_fd_gradient"] for row in rows], DTYPE)
    return {
        "row_count": len(rows),
        "finite_rows": sum(1 for row in rows if row["finite"]),
        "bayesfilter_likelihood_rmse_vs_filterflow": _rmse(bf_ll, ff_ll),
        "bayesfilter_likelihood_rmse_vs_kalman": _rmse(bf_ll, kf_ll),
        "filterflow_likelihood_rmse_vs_kalman": _rmse(ff_ll, kf_ll),
        "bayesfilter_gradient_rmse_vs_filterflow": _rmse(bf_grad, ff_grad),
        "bayesfilter_gradient_rmse_vs_kalman": _rmse(bf_grad, kf_grad),
        "filterflow_gradient_rmse_vs_kalman": _rmse(ff_grad, kf_grad),
        "bayesfilter_gradient_cosine_vs_filterflow": _cosine(bf_grad, ff_grad),
        "bayesfilter_gradient_cosine_vs_kalman": _cosine(bf_grad, kf_grad),
        "filterflow_gradient_cosine_vs_kalman": _cosine(ff_grad, kf_grad),
        "bayesfilter_gradient_sign_agreement_vs_filterflow": _sign_agreement(bf_grad, ff_grad),
        "bayesfilter_gradient_sign_agreement_vs_kalman": _sign_agreement(bf_grad, kf_grad),
        "filterflow_gradient_sign_agreement_vs_kalman": _sign_agreement(ff_grad, kf_grad),
        "bayesfilter_gradient_norm": _float(tf.linalg.norm(tf.reshape(bf_grad, [-1]))),
        "filterflow_gradient_norm": _float(tf.linalg.norm(tf.reshape(ff_grad, [-1]))),
        "kalman_gradient_norm": _float(tf.linalg.norm(tf.reshape(kf_grad, [-1]))),
        "same_model_gap_closed": True,
        "scalar_comparability_status": "open_blocking_for_gradient_agreement",
        "gradient_agreement_concluded": False,
        "gradient_claim_status": "same_model_finite_diagnostics_not_gradient_agreement",
    }


def _model_contract() -> dict[str, Any]:
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "transition_matrix": "A(theta)=diag(theta_1, theta_2)+[[0,1],[0,0]]",
        "transition_covariance": [[1.0 / 3.0, 0.5], [0.5, 1.0]],
        "observation_matrix": [[1.0, 0.0]],
        "observation_covariance": [[0.01]],
        "horizon": T,
        "mesh_size": MESH_SIZE,
        "num_particles": NUM_PARTICLES,
        "batch_size": BATCH_SIZE,
        "resampling_threshold": "ESS <= 0.9999 N",
        "epsilon": EPSILON,
        "scaling": SCALING,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "max_iter": MAX_ITER,
    }


def _seed_contract(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "recorded",
        "data_seed": DATA_SEED,
        "filter_seed": FILTER_SEED,
        "observation_checksum": reference["observation_checksum"],
        "initial_particles_checksum": reference["initial_particles_checksum"],
        "initial_particles_shape": [BATCH_SIZE, NUM_PARTICLES, 2],
        "observations_shape": [T, 1],
    }


def _scalar_contract(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "same_model_scalar_definitions_recorded",
        "filterflow_scalar": "tf.reduce_mean(final_state.log_likelihoods), batch_size=1",
        "bayesfilter_scalar": "tf.reduce_mean(log_likelihoods), batch_size=1",
        "kalman_scalar": "total Kalman log likelihood from get_surface_kf",
        "normalization_note": "batch_size=1 makes mean and total equal; per-time values are explanatory only",
        "filterflow_first_scalar": reference["filterflow_log_likelihoods"][0],
        "kalman_first_scalar": reference["kalman_log_likelihoods"][0],
    }


def _caveat_ledger(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "same_model_wrong_comparator_gap",
            "status": "closed",
            "detail": "BayesFilter gradient harness now uses filterflow smoothness constant-velocity LGSSM.",
        },
        {
            "id": "bit_identical_randomness",
            "status": "open_controlled",
            "detail": "Observations and initial particles are extracted, but BayesFilter transition noises use stateless common random numbers rather than filterflow internal split_seed draws.",
        },
        {
            "id": "scalar_comparability",
            "status": "open_blocking_for_gradient_agreement",
            "detail": "Same-model scalar definitions are recorded, but likelihood scales remain far apart, including filterflow versus Kalman.",
        },
        {
            "id": "gradient_agreement",
            "status": comparison["summary"]["gradient_claim_status"],
            "detail": "Finite same-model diagnostics are recorded; gradient agreement is not concluded.",
        },
        {
            "id": "scientific_validity",
            "status": "bounded_diagnostic_only",
            "detail": "No production, posterior, HMC, or general nonlinear-SSM validity follows.",
        },
    ]


def _decision_table(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    summary = comparison["summary"]
    return [
        {
            "decision": "close wrong-model gradient-harness gap",
            "primary_criterion_status": "same filterflow smoothness LGSSM used",
            "veto_status": "pass" if summary["finite_rows"] == summary["row_count"] else "veto_nonfinite",
            "main_uncertainty": "bit-identical filterflow transition-noise stream not reconstructed",
            "next_action": "derive or export exact filterflow random stream if gradient equality is required",
            "not_concluded": "gradient correctness or scientific validity",
        }
    ]


def _decision(comparison: dict[str, Any]) -> str:
    summary = comparison["summary"]
    if summary["finite_rows"] != summary["row_count"]:
        return "same_model_gradient_reconciliation_nonfinite_veto"
    return "same_model_gradient_wrong_model_gap_closed_gradient_agreement_not_concluded"


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] not in {
        "same_model_gradient_wrong_model_gap_closed_gradient_agreement_not_concluded",
        "same_model_gradient_reconciliation_nonfinite_veto",
        "same_model_gradient_reconciliation_blocked_filterflow_reference",
    }:
        raise RuntimeError(payload["decision"])
    for key in ["model_contract", "seed_contract", "scalar_contract", "decision_table", "caveat_ledger"]:
        if key not in payload:
            raise RuntimeError(f"missing {key}")
    if payload["seed_contract"].get("data_seed") != DATA_SEED:
        raise RuntimeError("wrong data seed")
    if payload["seed_contract"].get("filter_seed") != FILTER_SEED:
        raise RuntimeError("wrong filter seed")
    if "initial_particles_checksum" not in payload["seed_contract"]:
        raise RuntimeError("missing initial-particle checksum")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow Smoothness Same-Model Gradient Reconciliation

## Decision

`{payload['decision']}`

## Governance

This runner closes the prior wrong-model gradient-harness gap only. It does not
claim Claude review closure by itself and does not promote production readiness.

## Same-Model Diagnostic

{_key_value_table(payload['summary'])}

## Scientific Validity Limits

{_caveat_table(payload['caveat_ledger'])}

## Model Contract

{_key_value_table(payload['model_contract'])}

## Seed Contract

{_key_value_table(payload['seed_contract'])}

## Scalar Contract

{_key_value_table(payload['scalar_contract'])}

## Decision Table

{_decision_table_markdown(payload['decision_table'])}

## Non-Implications

{_bullets(payload['non_implications'])}
"""


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _caveat_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| ID | Status | Detail |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| `{row['id']}` | `{row['status']}` | {row['detail']} |")
    return "\n".join(lines)


def _decision_table_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Decision | Primary criterion status | Veto status | Main uncertainty | Next action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {decision} | {primary} | {veto} | {uncertainty} | {next_action} | {not_concluded} |".format(
                decision=row["decision"],
                primary=row["primary_criterion_status"],
                veto=row["veto_status"],
                uncertainty=row["main_uncertainty"],
                next_action=row["next_action"],
                not_concluded=row["not_concluded"],
            )
        )
    return "\n".join(lines)


def _non_implications() -> list[str]:
    return [
        "No gradient agreement is concluded.",
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No DSGE/NAWM validation is concluded.",
        "No monograph claim is concluded.",
    ]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _rmse(a: tf.Tensor, b: tf.Tensor) -> float:
    diff = tf.cast(a, DTYPE) - tf.cast(b, DTYPE)
    return _float(tf.sqrt(tf.reduce_mean(diff * diff)))


def _cosine(a: tf.Tensor, b: tf.Tensor) -> float | None:
    flat_a = tf.reshape(tf.cast(a, DTYPE), [-1])
    flat_b = tf.reshape(tf.cast(b, DTYPE), [-1])
    denom = tf.linalg.norm(flat_a) * tf.linalg.norm(flat_b)
    if float(denom.numpy()) == 0.0:
        return None
    return _float(tf.tensordot(flat_a, flat_b, axes=1) / denom)


def _sign_agreement(a: tf.Tensor, b: tf.Tensor) -> float:
    flat_a = tf.reshape(tf.cast(a, DTYPE), [-1])
    flat_b = tf.reshape(tf.cast(b, DTYPE), [-1])
    return _float(tf.reduce_mean(tf.cast(tf.sign(flat_a) == tf.sign(flat_b), DTYPE)))


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())
