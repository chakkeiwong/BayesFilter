"""Probe historical transport-node VJP deltas for row 173."""

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
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_row_173_vjp_decomposition_tf as vjp,
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


tfd = tfp.distributions

DTYPE = tf.float64
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-historical-transport-vjp-plan-2026-06-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-historical-transport-vjp-result-2026-06-04.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_historical_transport_vjp_2026-06-04.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-historical-transport-vjp-2026-06-04.md"
)
FILTERFLOW_MARKER_PATH = vjp.FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
TARGET_TIME_INDEX = 93
TAG = "row-173-historical-transport-vjp"
VALUE_TOLERANCE = vjp.VALUE_TOLERANCE
GRADIENT_TOLERANCE = vjp.GRADIENT_TOLERANCE
MATERIAL_REDUCTION_FRACTION = 0.10


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
    config = vjp.RunConfig(
        target_time_index=TARGET_TIME_INDEX,
        tag=TAG,
        plan_path=PLAN_PATH,
        result_path=RESULT_PATH,
        json_path=JSON_PATH,
        report_path=REPORT_PATH,
    )
    filterflow = _filterflow_history_subprocess(config)
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_historical_transport_filterflow_blocker",
            filterflow.get("blocker", "unknown FilterFlow blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            None,
        )
    bayesfilter = _bayesfilter_history(filterflow, config)
    if bayesfilter.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_historical_transport_bayesfilter_blocker",
            bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            bayesfilter,
        )

    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    comparison = _compare_history(filterflow, bayesfilter)
    veto_status = _veto_status(filterflow, bayesfilter, comparison, comparator_drift)
    classification = _classify(comparison, veto_status)
    decision = _decision(classification, veto_status)
    return {
        "decision": decision,
        "hypothesis_classification": classification["classification"],
        "hypothesis_reason": classification["reason"],
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_historical_transport_vjp_residual_reconstruction",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "filterflow_history": _compact_side(filterflow),
        "bayesfilter_history": _compact_side(bayesfilter),
        "historical_transport_comparison": comparison,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_historical_transport_vjp_probe_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "data_seed": vjp.DATA_SEED,
            "filter_seed": vjp.FILTER_SEED,
            "target_time_index": TARGET_TIME_INDEX,
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        },
        "decision_table": _decision_table(decision, classification, veto_status),
        "non_implications": _non_implications(),
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    reference_status: dict[str, Any],
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any] | None,
    bayesfilter: dict[str, Any] | None,
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
        "filterflow_history": _compact_side(filterflow),
        "bayesfilter_history": _compact_side(bayesfilter),
        "historical_transport_comparison": comparison,
        "veto_status_table": {"status": "blocked", "blocker": blocker},
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_historical_transport_vjp_probe_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(
            decision,
            {"classification": "blocked_or_vetoed", "reason": blocker},
            {"status": "blocked"},
        ),
        "non_implications": _non_implications(),
    }


def _filterflow_history_subprocess(config: vjp.RunConfig) -> dict[str, Any]:
    if not vjp.FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {vjp.FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(vjp.FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(vjp.FILTERFLOW_ENV_PYTHON), "-c", _filterflow_history_script(config)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow historical transport subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_HISTORICAL_TRANSPORT_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_HISTORICAL_TRANSPORT_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow historical transport JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(
        completed.stdout[start + len("FILTERFLOW_HISTORICAL_TRANSPORT_JSON_BEGIN"):end].strip()
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_history_script(config: vjp.RunConfig) -> str:
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
        import tensorflow_probability as tfp
        from tensorflow_probability.python.internal import samplers

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from filterflow.resampling.differentiable.biased import apply_transport_matrix
        from filterflow.resampling.differentiable.regularized_transport.plan import transport
        from filterflow.utils import normalize
        from scripts.simple_linear_smoothness import get_data

        DTYPE = tf.float64
        NP_DTYPE = np.float64
        T = {vjp.T}
        BATCH_SIZE = {vjp.BATCH_SIZE}
        N = {vjp.NUM_PARTICLES}
        DATA_SEED = {vjp.DATA_SEED}
        FILTER_SEED = {vjp.FILTER_SEED}
        TARGET_TIME_INDEX = {config.target_time_index}
        EPSILON = {vjp.EPSILON!r}
        SCALING = {vjp.SCALING!r}
        CONVERGENCE_THRESHOLD = {vjp.CONVERGENCE_THRESHOLD!r}
        MAX_ITERATIONS = {vjp.MAX_ITERATIONS}
        RESAMPLING_NEFF = {vjp.RESAMPLING_NEFF!r}
        THETA = np.array({vjp.THETA!r}, dtype=NP_DTYPE)

        def to_json(tensor):
            return tf.cast(tensor, DTYPE).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, DTYPE).numpy())

        def watched_grad(tape, target, tensor):
            gradient = tape.gradient(target, tensor)
            return tf.zeros_like(tensor) if gradient is None else gradient

        def watched_grad_with_upstream(tape, target, tensor, upstream):
            gradient = tape.gradient(target, tensor, output_gradients=tf.stop_gradient(upstream))
            return tf.zeros_like(tensor) if gradient is None else gradient

        def distribution_loc(distribution):
            if hasattr(distribution, "loc"):
                return distribution.loc
            parameters = getattr(distribution, "parameters", {{}})
            if "loc" in parameters:
                return parameters["loc"]
            return distribution.mean()

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
        resampling_method = RegularisedTransform(
            epsilon=EPSILON,
            scaling=SCALING,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            max_iter=MAX_ITERATIONS,
        )
        smc = make_filter(
            observation_matrix,
            modifiable_transition_matrix,
            observation_covariance_chol,
            transition_covariance_chol,
            resampling_method,
            NeffCriterion(RESAMPLING_NEFF, True),
            optimal_proposal=True,
        )

        seed = tf.constant(FILTER_SEED, dtype=tf.int32)
        paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
        seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))
        state = initial_state
        transport_records = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(modifiable_transition_matrix)
            for time_index in range(T):
                seed, _seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
                observation = observations[time_index]
                inputs = tf.constant(time_index, dtype=tf.int32)
                t = state.t
                float_t = tf.cast(t, DTYPE)
                float_t_1 = float_t + tf.constant(1.0, dtype=DTYPE)
                flags, ess = smc._resampling_criterion.apply(state)
                state_with_ess = attr.evolve(
                    state,
                    ess=ess / float_t_1 + state.ess * (float_t / float_t_1),
                )
                transport_matrix = transport(
                    state_with_ess.particles,
                    state_with_ess.log_weights,
                    resampling_method.epsilon,
                    resampling_method.scaling,
                    resampling_method.convergence_threshold,
                    resampling_method.max_iter,
                    state_with_ess.n_particles,
                )
                transport_records.append({{
                    "time_index": time_index,
                    "transport_matrix": transport_matrix,
                    "flags": flags,
                    "log_ess": tf.math.log(ess),
                }})
                resampled_state = apply_transport_matrix(state_with_ess, transport_matrix, flags)
                proposal_dist = smc._proposal_model._get_proposal_dist(resampled_state, observation)
                proposal_loc = distribution_loc(proposal_dist)
                proposed_particles = proposal_dist.sample(seed=seed2)
                proposed_state = attr.evolve(resampled_state, particles=proposed_particles)
                observation_ll = smc._observation_model.loglikelihood(proposed_state, observation)
                transition_ll = smc._transition_model.loglikelihood(
                    resampled_state,
                    proposed_state,
                    inputs,
                )
                proposal_ll = smc._proposal_model.loglikelihood(
                    proposed_state,
                    resampled_state,
                    inputs,
                    observation,
                )
                del proposal_loc
                unnormalized = transition_ll + observation_ll - proposal_ll + resampled_state.log_weights
                increment = tf.reduce_logsumexp(unnormalized, 1)
                log_likelihoods = resampled_state.log_likelihoods + increment
                normalized = normalize(unnormalized, 1, resampled_state.n_particles, True)
                updated_state = attr.evolve(
                    proposed_state,
                    weights=tf.exp(normalized),
                    log_weights=normalized,
                    log_likelihoods=log_likelihoods,
                )
                state = attr.evolve(updated_state, t=t + 1)
                if time_index == TARGET_TIME_INDEX:
                    target = tf.reduce_mean(log_likelihoods)
                    break
            else:
                raise RuntimeError("target time not reached")

        total_gradient = watched_grad(tape, target, modifiable_transition_matrix)
        rows = []
        cumulative_vjp = tf.zeros_like(modifiable_transition_matrix)
        for record in transport_records:
            raw_upstream = watched_grad(tape, target, record["transport_matrix"])
            clipped_upstream = tf.clip_by_value(raw_upstream, -1.0, 1.0)
            vjp_tensor = watched_grad_with_upstream(
                tape,
                record["transport_matrix"],
                modifiable_transition_matrix,
                raw_upstream,
            )
            manual_clipped_vjp_tensor = watched_grad_with_upstream(
                tape,
                record["transport_matrix"],
                modifiable_transition_matrix,
                clipped_upstream,
            )
            cumulative_vjp = cumulative_vjp + vjp_tensor
            rows.append({{
                "time_index": int(record["time_index"]),
                "resampling_flag": [bool(v) for v in tf.reshape(record["flags"], [-1]).numpy().tolist()],
                "raw_upstream": to_json(raw_upstream),
                "clipped_upstream": to_json(clipped_upstream),
                "clip_mask": to_json(tf.cast(tf.abs(raw_upstream) > 1.0, DTYPE)),
                "raw_upstream_summary": {{
                    "max_abs": scalar(tf.reduce_max(tf.abs(raw_upstream))),
                    "sum": scalar(tf.reduce_sum(raw_upstream)),
                    "finite": bool(tf.reduce_all(tf.math.is_finite(raw_upstream)).numpy()),
                }},
                "clipped_upstream_summary": {{
                    "max_abs": scalar(tf.reduce_max(tf.abs(clipped_upstream))),
                    "sum": scalar(tf.reduce_sum(clipped_upstream)),
                    "finite": bool(tf.reduce_all(tf.math.is_finite(clipped_upstream)).numpy()),
                }},
                "clip_count": scalar(tf.reduce_sum(tf.cast(tf.abs(raw_upstream) > 1.0, DTYPE))),
                "transport_vjp_diag": to_json(tf.linalg.diag_part(vjp_tensor)),
                "manual_clipped_transport_vjp_diag": to_json(tf.linalg.diag_part(manual_clipped_vjp_tensor)),
                "transport_vjp_finite": bool(tf.reduce_all(tf.math.is_finite(vjp_tensor)).numpy()),
            }})
        del tape

        payload = {{
            "status": "executed",
            "backend": "executable_filterflow_subprocess",
            "settings": {{
                "mesh_index": {vjp.MESH_INDEX},
                "target_time_index": TARGET_TIME_INDEX,
                "theta": THETA.astype(float).tolist(),
                "T": T,
                "n_particles": N,
                "dtype": "float64",
                "transport_backward": "FilterFlow custom gradient clips d_transport to [-1,1]",
            }},
            "target_scalar": scalar(target),
            "total_gradient_diag": to_json(tf.linalg.diag_part(total_gradient)),
            "cumulative_transport_vjp_diag": to_json(tf.linalg.diag_part(cumulative_vjp)),
            "history_rows": rows,
            "history_length": len(rows),
            "resampling_flags": [row["resampling_flag"] for row in rows],
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
            "model": {{
                "observations": observations_np.astype(float).tolist(),
                "initial_particles": initial_particles_np.astype(float).tolist(),
                "observation_matrix": observation_matrix_np.astype(float).tolist(),
                "transition_covariance_chol": to_json(transition_covariance_chol),
                "observation_covariance_chol": to_json(observation_covariance_chol),
            }},
            "package_versions": {{
                "python": os.sys.version.split()[0],
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
            }},
        }}
        print("FILTERFLOW_HISTORICAL_TRANSPORT_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_HISTORICAL_TRANSPORT_JSON_END")
        """
    )


def _bayesfilter_history(filterflow: dict[str, Any], config: vjp.RunConfig) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        theta_variable = tf.Variable(vjp.THETA, dtype=DTYPE)
        model = _model_from_filterflow(filterflow)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta_variable)
            bundle = _bayesfilter_history_bundle(theta_variable, model, config)
            target = bundle["target"]
        total_gradient = _safe_gradient(tape, target, theta_variable)
        rows = []
        cumulative_vjp = tf.zeros_like(theta_variable)
        for record in bundle["transport_records"]:
            raw_upstream = _safe_gradient(tape, target, record["transport_matrix"])
            clipped_upstream = tf.clip_by_value(raw_upstream, -1.0, 1.0)
            vjp_tensor = _safe_gradient_with_upstream(
                tape,
                record["transport_matrix"],
                theta_variable,
                raw_upstream,
            )
            manual_clipped_vjp_tensor = _safe_gradient_with_upstream(
                tape,
                record["transport_matrix"],
                theta_variable,
                clipped_upstream,
            )
            cumulative_vjp = cumulative_vjp + vjp_tensor
            rows.append(
                {
                    "time_index": int(record["time_index"]),
                    "resampling_flag": [
                        bool(v) for v in tf.reshape(record["flags"], [-1]).numpy().tolist()
                    ],
                    "raw_upstream": r3._json(raw_upstream),
                    "clipped_upstream": r3._json(clipped_upstream),
                    "clip_mask": r3._json(tf.cast(tf.abs(raw_upstream) > 1.0, DTYPE)),
                    "raw_upstream_summary": _summary_tensor(raw_upstream),
                    "clipped_upstream_summary": _summary_tensor(clipped_upstream),
                    "clip_count": _float(
                        tf.reduce_sum(tf.cast(tf.abs(raw_upstream) > 1.0, DTYPE))
                    ),
                    "transport_vjp_diag": r3._json(vjp_tensor),
                    "manual_clipped_transport_vjp_diag": r3._json(manual_clipped_vjp_tensor),
                    "transport_vjp_finite": bool(tf.reduce_all(tf.math.is_finite(vjp_tensor)).numpy()),
                }
            )
        del tape
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "settings": filterflow["settings"],
            "target_scalar": _float(target),
            "total_gradient_diag": r3._json(total_gradient),
            "cumulative_transport_vjp_diag": r3._json(cumulative_vjp),
            "history_rows": rows,
            "history_length": len(rows),
            "resampling_flags": [row["resampling_flag"] for row in rows],
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_history_bundle(
    theta: tf.Tensor,
    model: dict[str, Any],
    config: vjp.RunConfig,
) -> dict[str, Any]:
    transition_matrix = _transition_matrix(theta)
    observation_matrix = tf.constant(model["observation_matrix"], dtype=DTYPE)
    transition_chol = tf.constant(model["transition_covariance_chol"], dtype=DTYPE)
    observation_chol = tf.constant(model["observation_covariance_chol"], dtype=DTYPE)
    observations = tf.constant(model["observations"], dtype=DTYPE)
    particles = tf.constant(model["initial_particles"], dtype=DTYPE)
    num_particles = vjp.NUM_PARTICLES
    log_weights = tf.fill(
        [vjp.BATCH_SIZE, num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )
    log_likelihoods = tf.zeros([vjp.BATCH_SIZE], dtype=DTYPE)
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
        tf.linalg.matmul(observation_matrix, observation_cov_inv, transpose_a=True),
        observation_matrix,
    )
    sigma = tf.linalg.inv(sigma_inv)
    sigma_chol = tf.linalg.cholesky(sigma)
    seed = tf.constant(vjp.FILTER_SEED, dtype=tf.int32)
    paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
    seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))
    transport_records = []

    for time_index in range(vjp.T):
        seed, _seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
        observation = observations[time_index]
        ess_log = r3._ess(log_weights)
        flags = ess_log <= tf.math.log(
            tf.cast(num_particles, DTYPE) * tf.constant(vjp.RESAMPLING_NEFF, DTYPE)
        )
        transported = annealed_transport_tf.annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=vjp.EPSILON,
            scaling=vjp.SCALING,
            convergence_threshold=vjp.CONVERGENCE_THRESHOLD,
            max_iterations=vjp.MAX_ITERATIONS,
            ess_mask=tf.reshape(flags, [-1]),
            transport_gradient_mode="filterflow_clipped",
            application_mode="filterflow_all_rows",
        )
        transport_records.append(
            {
                "time_index": time_index,
                "transport_matrix": tf.cast(transported.transport_matrix, DTYPE),
                "flags": flags,
                "log_ess": ess_log,
            }
        )
        particles = transported.particles
        log_weights = transported.log_weights
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
        observation_ll = _observation_log_prob(
            proposed_particles,
            observation,
            observation_matrix,
            observation_noise,
        )
        transition_ll = _transition_log_prob(
            particles,
            proposed_particles,
            transition_matrix,
            transition_noise,
        )
        proposal_ll = proposal_dist.log_prob(proposed_particles)
        unnormalized = transition_ll + observation_ll - proposal_ll + log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        log_likelihoods = log_likelihoods + increment
        log_weights = r3._filterflow_normalize(unnormalized, num_particles)
        particles = proposed_particles
        if time_index == config.target_time_index:
            return {
                "target": tf.reduce_mean(log_likelihoods),
                "transport_records": transport_records,
            }
    raise RuntimeError("target time not reached")


def _compare_history(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    rows = []
    if filterflow["history_length"] != bayesfilter["history_length"]:
        return {
            "status": "blocked",
            "blocker": "history length mismatch",
            "filterflow_history_length": filterflow["history_length"],
            "bayesfilter_history_length": bayesfilter["history_length"],
        }
    for ff_row, bf_row in zip(filterflow["history_rows"], bayesfilter["history_rows"], strict=True):
        transport_vjp_delta = _vector_sub(
            bf_row["transport_vjp_diag"],
            ff_row["transport_vjp_diag"],
        )
        raw_upstream_delta = _tensor_delta_summary(
            bf_row["raw_upstream"],
            ff_row["raw_upstream"],
        )
        clipped_upstream_delta = _tensor_delta_summary(
            bf_row["clipped_upstream"],
            ff_row["clipped_upstream"],
        )
        clip_mask_mismatch_count = _mask_mismatch_count(
            bf_row["clip_mask"],
            ff_row["clip_mask"],
        )
        rows.append(
            {
                "time_index": ff_row["time_index"],
                "resampling_flags_match": ff_row["resampling_flag"] == bf_row["resampling_flag"],
                "filterflow_resampling_flag": ff_row["resampling_flag"],
                "bayesfilter_resampling_flag": bf_row["resampling_flag"],
                "raw_upstream_max_abs_delta": raw_upstream_delta["max_abs_delta"],
                "raw_upstream_sum_delta": raw_upstream_delta["sum_delta"],
                "raw_upstream_finite": raw_upstream_delta["finite"],
                "clipped_upstream_max_abs_delta": clipped_upstream_delta["max_abs_delta"],
                "clipped_upstream_sum_delta": clipped_upstream_delta["sum_delta"],
                "clipped_upstream_finite": clipped_upstream_delta["finite"],
                "clip_mask_mismatch_count": clip_mask_mismatch_count,
                "clip_count_delta": float(bf_row["clip_count"]) - float(ff_row["clip_count"]),
                "filterflow_clip_count": ff_row["clip_count"],
                "bayesfilter_clip_count": bf_row["clip_count"],
                "transport_vjp_delta": transport_vjp_delta,
                "max_abs_transport_vjp_delta": _max_abs(transport_vjp_delta),
                "finite": bool(ff_row["transport_vjp_finite"] and bf_row["transport_vjp_finite"]),
            }
        )
    full_gradient_delta = _vector_sub(
        bayesfilter["total_gradient_diag"],
        filterflow["total_gradient_diag"],
    )
    cumulative_transport_vjp_delta = _vector_sub(
        bayesfilter["cumulative_transport_vjp_diag"],
        filterflow["cumulative_transport_vjp_diag"],
    )
    reconstruction_residual = _vector_sub(full_gradient_delta, cumulative_transport_vjp_delta)
    before_norm = _max_abs(full_gradient_delta)
    after_norm = _max_abs(reconstruction_residual)
    explained_fraction = 0.0 if before_norm == 0.0 else 1.0 - after_norm / before_norm
    upstream_or_mask_deltas = [
        row
        for row in rows
        if row["clip_count_delta"] != 0.0
        or row["clip_mask_mismatch_count"] != 0
        or row["raw_upstream_max_abs_delta"] > GRADIENT_TOLERANCE
        or abs(row["raw_upstream_sum_delta"]) > GRADIENT_TOLERANCE
        or row["clipped_upstream_max_abs_delta"] > GRADIENT_TOLERANCE
        or abs(row["clipped_upstream_sum_delta"]) > GRADIENT_TOLERANCE
    ]
    top_rows = sorted(rows, key=lambda row: row["max_abs_transport_vjp_delta"], reverse=True)[:10]
    return {
        "status": "compared",
        "full_gradient_delta": full_gradient_delta,
        "cumulative_transport_vjp_delta": cumulative_transport_vjp_delta,
        "reconstruction_residual": reconstruction_residual,
        "max_abs_full_gradient_delta": before_norm,
        "max_abs_reconstruction_residual": after_norm,
        "explained_fraction_by_max_abs_norm": explained_fraction,
        "history_length": len(rows),
        "rows": rows,
        "top_transport_vjp_delta_rows": top_rows,
        "upstream_or_mask_delta_rows": upstream_or_mask_deltas[:20],
        "max_abs_transport_vjp_delta": max(row["max_abs_transport_vjp_delta"] for row in rows),
        "max_abs_raw_upstream_delta": max(row["raw_upstream_max_abs_delta"] for row in rows),
        "max_abs_clipped_upstream_delta": max(
            row["clipped_upstream_max_abs_delta"] for row in rows
        ),
        "total_clip_mask_mismatch_count": sum(row["clip_mask_mismatch_count"] for row in rows),
        "all_resampling_flags_match": all(row["resampling_flags_match"] for row in rows),
        "all_transport_vjps_finite": all(row["finite"] for row in rows),
        "all_upstreams_finite": all(
            row["raw_upstream_finite"] and row["clipped_upstream_finite"] for row in rows
        ),
        "scalar_delta": abs(float(bayesfilter["target_scalar"]) - float(filterflow["target_scalar"])),
        "interpretation": _interpret_history(
            full_gradient_delta,
            cumulative_transport_vjp_delta,
            reconstruction_residual,
            rows,
            explained_fraction,
        ),
    }


def _interpret_history(
    full_gradient_delta: list[float],
    cumulative_transport_vjp_delta: list[float],
    reconstruction_residual: list[float],
    rows: list[dict[str, Any]],
    explained_fraction: float,
) -> str:
    del full_gradient_delta
    upstream_or_mask_mismatch = any(
        row["clip_count_delta"] != 0.0
        or row["clip_mask_mismatch_count"] != 0
        or row["raw_upstream_max_abs_delta"] > GRADIENT_TOLERANCE
        or abs(row["raw_upstream_sum_delta"]) > GRADIENT_TOLERANCE
        or row["clipped_upstream_max_abs_delta"] > GRADIENT_TOLERANCE
        or abs(row["clipped_upstream_sum_delta"]) > GRADIENT_TOLERANCE
        for row in rows
    )
    if upstream_or_mask_mismatch:
        return "historical_transport_upstreams_or_masks_diverge"
    if _max_abs(reconstruction_residual) <= GRADIENT_TOLERANCE:
        return "accumulated_transport_vjp_reconstructs_residual"
    if _max_abs(cumulative_transport_vjp_delta) <= GRADIENT_TOLERANCE:
        return "residual_outside_transport_nodes"
    if explained_fraction >= MATERIAL_REDUCTION_FRACTION:
        return "historical_transport_vjp_partially_explains_residual"
    return "residual_outside_transport_nodes"


def _classify(comparison: dict[str, Any], veto_status: dict[str, Any]) -> dict[str, str]:
    if not veto_status.get("all_vetoes_clear", False):
        return {"classification": "blocked_or_vetoed", "reason": str(veto_status)}
    mapping = {
        "accumulated_transport_vjp_reconstructs_residual": (
            "h1_accumulated_transport_vjp_reconstructs_residual",
            "historical transport VJP deltas reconstruct the full row residual",
        ),
        "historical_transport_vjp_partially_explains_residual": (
            "h2_historical_transport_vjp_partially_explains_residual",
            "historical transport VJP deltas materially reduce but do not close the residual",
        ),
        "historical_transport_upstreams_or_masks_diverge": (
            "h3_transport_upstreams_or_masks_diverge_historically",
            "historical upstream or clip-mask deltas exceed tolerance",
        ),
        "residual_outside_transport_nodes": (
            "h4_residual_outside_transport_nodes",
            "transport-node VJP deltas are small relative to the full residual",
        ),
    }
    classification, reason = mapping.get(
        comparison.get("interpretation"),
        ("blocked_or_vetoed", f"unclassified interpretation {comparison.get('interpretation')}"),
    )
    return {"classification": classification, "reason": reason}


def _veto_status(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
    comparison: dict[str, Any],
    comparator_drift: bool,
) -> dict[str, Any]:
    path_boundary = continuation._path_boundary_manifest()
    return {
        "all_vetoes_clear": (
            comparison.get("status") == "compared"
            and not comparator_drift
            and not any(bool(value) for value in path_boundary.values())
            and comparison["scalar_delta"] <= VALUE_TOLERANCE
            and comparison["all_resampling_flags_match"]
            and comparison["all_transport_vjps_finite"]
            and comparison["all_upstreams_finite"]
            and PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1"
        ),
        "comparator_drift": comparator_drift,
        "path_boundary_clean": not any(bool(value) for value in path_boundary.values()),
        "scalar_value_gate_pass": comparison.get("scalar_delta", float("inf")) <= VALUE_TOLERANCE,
        "all_resampling_flags_match": comparison.get("all_resampling_flags_match", False),
        "all_transport_vjps_finite": comparison.get("all_transport_vjps_finite", False),
        "all_upstreams_finite": comparison.get("all_upstreams_finite", False),
        "cpu_only_parent": PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1",
    }


def _decision(classification: dict[str, str], veto_status: dict[str, Any]) -> str:
    if not veto_status.get("all_vetoes_clear", False):
        return "filterflow_float64_row_173_historical_transport_blocked_or_vetoed"
    mapping = {
        "h1_accumulated_transport_vjp_reconstructs_residual": (
            "filterflow_float64_row_173_historical_transport_h1_reconstructs_residual"
        ),
        "h2_historical_transport_vjp_partially_explains_residual": (
            "filterflow_float64_row_173_historical_transport_h2_partially_explains_residual"
        ),
        "h3_transport_upstreams_or_masks_diverge_historically": (
            "filterflow_float64_row_173_historical_transport_h3_upstreams_or_masks_diverge"
        ),
        "h4_residual_outside_transport_nodes": (
            "filterflow_float64_row_173_historical_transport_h4_residual_outside_transport"
        ),
    }
    return mapping.get(classification["classification"], "filterflow_float64_row_173_historical_transport_blocked_or_vetoed")


def _compact_side(side: dict[str, Any] | None) -> dict[str, Any] | None:
    if side is None or side.get("status") != "executed":
        return side
    return {
        "status": side["status"],
        "backend": side["backend"],
        "settings": side["settings"],
        "target_scalar": side["target_scalar"],
        "total_gradient_diag": side["total_gradient_diag"],
        "cumulative_transport_vjp_diag": side["cumulative_transport_vjp_diag"],
        "history_length": side["history_length"],
        "resampling_flags": side["resampling_flags"],
        "cpu_only_manifest": side["cpu_only_manifest"],
        "stderr_excerpt": side.get("stderr_excerpt", ""),
    }


def _model_from_filterflow(filterflow: dict[str, Any]) -> dict[str, Any]:
    return {
        "observations": filterflow["model"]["observations"],
        "initial_particles": filterflow["model"]["initial_particles"],
        "observation_matrix": filterflow["model"]["observation_matrix"],
        "transition_covariance_chol": filterflow["model"]["transition_covariance_chol"],
        "observation_covariance_chol": filterflow["model"]["observation_covariance_chol"],
    }


def _transition_matrix(theta: tf.Tensor) -> tf.Tensor:
    return tf.linalg.diag(theta) + tf.constant([[0.0, 1.0], [0.0, 0.0]], dtype=DTYPE)


def _optimal_proposal_mean(
    particles: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    observation_matrix: tf.Tensor,
    transition_cov_inv: tf.Tensor,
    observation_cov_inv: tf.Tensor,
    sigma: tf.Tensor,
) -> tf.Tensor:
    observation_term = tf.linalg.matvec(
        observation_matrix,
        tf.linalg.matvec(observation_cov_inv, observation),
        transpose_a=True,
    )
    transition_term = tf.linalg.matvec(
        transition_cov_inv,
        tf.linalg.matvec(transition_matrix, particles),
    )
    return tf.linalg.matvec(sigma, observation_term + transition_term)


def _transition_log_prob(
    previous_particles: tf.Tensor,
    proposed_particles: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_noise: tfd.Distribution,
) -> tf.Tensor:
    transition_mean = tf.linalg.matvec(transition_matrix, previous_particles)
    return transition_noise.log_prob(proposed_particles - transition_mean)


def _observation_log_prob(
    particles: tf.Tensor,
    observation: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_noise: tfd.Distribution,
) -> tf.Tensor:
    observation_mean = tf.linalg.matvec(observation_matrix, particles)
    return observation_noise.log_prob(observation - observation_mean)


def _safe_gradient(tape: tf.GradientTape, target: tf.Tensor, tensor: tf.Tensor) -> tf.Tensor:
    gradient = tape.gradient(target, tensor)
    return tf.zeros_like(tensor) if gradient is None else gradient


def _safe_gradient_with_upstream(
    tape: tf.GradientTape,
    target: tf.Tensor,
    tensor: tf.Tensor,
    upstream: tf.Tensor,
) -> tf.Tensor:
    gradient = tape.gradient(target, tensor, output_gradients=tf.stop_gradient(upstream))
    return tf.zeros_like(tensor) if gradient is None else gradient


def _summary_tensor(tensor: tf.Tensor) -> dict[str, Any]:
    cast = tf.cast(tensor, DTYPE)
    return {
        "max_abs": _float(tf.reduce_max(tf.abs(cast))),
        "sum": _float(tf.reduce_sum(cast)),
        "finite": bool(tf.reduce_all(tf.math.is_finite(cast)).numpy()),
    }


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _model_contract() -> dict[str, Any]:
    return {
        "mesh_index": vjp.MESH_INDEX,
        "target_time_index": TARGET_TIME_INDEX,
        "theta": vjp.THETA,
        "num_particles": vjp.NUM_PARTICLES,
        "data_seed": vjp.DATA_SEED,
        "filter_seed": vjp.FILTER_SEED,
        "dtype": "float64",
        "comparison_scope": "all historical transport nodes through target time only",
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": "Do accumulated historical transport VJP deltas reconstruct row 173?",
        "baseline": "local canonical executable float64 FilterFlow reference",
        "primary_criterion": "full scalar-gradient residual minus cumulative transport VJP delta",
        "veto_diagnostics": [
            "comparator drift",
            "resampling flag mismatch",
            "scalar value mismatch",
            "non-finite transport VJPs",
            "CPU-only policy failure",
            "path boundary contamination",
        ],
        "explanatory_diagnostics": [
            "top per-time VJP deltas",
            "clip counts",
            "historical upstream tensor deltas",
            "clip-mask mismatch counts",
            "residual norm reduction",
        ],
        "artifact": str(JSON_PATH.relative_to(REPO_ROOT)),
        "not_concluded": _non_implications(),
    }


def _decision_table(
    decision: str,
    classification: dict[str, str],
    veto_status: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "decision": decision,
            "primary_criterion_status": classification["classification"],
            "veto_diagnostic_status": str(veto_status),
            "main_uncertainty": "single row and target time; no global gradient claim",
            "next_justified_action": _next_action(classification["classification"]),
            "not_concluded": "correctness, posterior correctness, production readiness, global gradient agreement",
        }
    ]


def _next_action(classification: str) -> str:
    if classification == "h1_accumulated_transport_vjp_reconstructs_residual":
        return "inspect per-time top contributors and patch transport VJP rule if executable FilterFlow contract is unambiguous"
    if classification == "h2_historical_transport_vjp_partially_explains_residual":
        return "split remaining residual between carryover topology and non-transport proposal/update paths"
    if classification == "h3_transport_upstreams_or_masks_diverge_historically":
        return "inspect first historical upstream or clip-mask divergence"
    if classification == "h4_residual_outside_transport_nodes":
        return "debug carryover/state/proposal topology outside transport nodes"
    return "repair blocker or veto"


def _non_implications() -> list[str]:
    return [
        "No correctness claim is made for either implementation.",
        "No analytic gradient correctness is concluded.",
        "No posterior correctness is concluded.",
        "No global gradient agreement is concluded.",
        "No full mesh or surface agreement is concluded.",
        "No production readiness or public API readiness is concluded.",
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "hypothesis_classification",
        "plan_path",
        "result_path",
        "evidence_contract",
        "reference_policy",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "model_contract",
        "historical_transport_comparison",
        "veto_status_table",
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
    if payload["run_manifest"].get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("pre-import CUDA_VISIBLE_DEVICES was not -1")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    allowed = {
        "filterflow_float64_row_173_historical_transport_filterflow_blocker",
        "filterflow_float64_row_173_historical_transport_bayesfilter_blocker",
        "filterflow_float64_row_173_historical_transport_blocked_or_vetoed",
        "filterflow_float64_row_173_historical_transport_h1_reconstructs_residual",
        "filterflow_float64_row_173_historical_transport_h2_partially_explains_residual",
        "filterflow_float64_row_173_historical_transport_h3_upstreams_or_masks_diverge",
        "filterflow_float64_row_173_historical_transport_h4_residual_outside_transport",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: Row 173 Historical Transport VJP Probe",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Hypothesis Classification",
        "",
        f"`{payload['hypothesis_classification']}`",
        "",
        payload.get("hypothesis_reason", ""),
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            "| {decision} | {primary_criterion_status} | {veto_diagnostic_status} | "
            "{main_uncertainty} | {next_justified_action} | {not_concluded} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Historical Transport Comparison",
            "",
            _json_block(payload["historical_transport_comparison"]),
            "",
            "## Veto Status",
            "",
            _json_block(payload["veto_status_table"]),
            "",
            "## FilterFlow History",
            "",
            _json_block(payload["filterflow_history"]),
            "",
            "## BayesFilter History",
            "",
            _json_block(payload["bayesfilter_history"]),
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _vector_sub(left: list[float], right: list[float]) -> list[float]:
    return [float(lhs) - float(rhs) for lhs, rhs in zip(left, right, strict=True)]


def _tensor_delta_summary(left: Any, right: Any) -> dict[str, Any]:
    left_tensor = tf.constant(left, dtype=DTYPE)
    right_tensor = tf.constant(right, dtype=DTYPE)
    delta = left_tensor - right_tensor
    return {
        "max_abs_delta": _float(tf.reduce_max(tf.abs(delta))),
        "sum_delta": _float(tf.reduce_sum(delta)),
        "finite": bool(tf.reduce_all(tf.math.is_finite(delta)).numpy()),
    }


def _mask_mismatch_count(left: Any, right: Any) -> int:
    left_mask = tf.not_equal(tf.constant(left, dtype=DTYPE), tf.constant(0.0, dtype=DTYPE))
    right_mask = tf.not_equal(tf.constant(right, dtype=DTYPE), tf.constant(0.0, dtype=DTYPE))
    return int(tf.reduce_sum(tf.cast(tf.not_equal(left_mask, right_mask), tf.int32)).numpy())


def _max_abs(values: list[float]) -> float:
    return max(abs(float(value)) for value in values) if values else 0.0


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
