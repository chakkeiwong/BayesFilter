"""Test direct-theta hypotheses for row-173 float64 FilterFlow mismatch."""

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
    "bayesfilter-dpf-filterflow-float64-row-173-direct-theta-hypothesis-plan-2026-06-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-direct-theta-hypothesis-result-2026-06-04.md"
)
JSON_PATH = (
    OUTPUT_DIR
    / "dpf_filterflow_float64_row_173_direct_theta_hypothesis_2026-06-04.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-direct-theta-hypothesis-2026-06-04.md"
)
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
TARGET_TIME_INDEX = 93
VALUE_TOLERANCE = 5e-8
GRADIENT_TOLERANCE = 2e-4
OBSERVED_TOTAL_GRADIENT_DELTA = [5.302734403676368, -0.1337765252068337]
SENTINEL_BEGIN = "FILTERFLOW_ROW_173_DIRECT_THETA_JSON_BEGIN"
SENTINEL_END = "FILTERFLOW_ROW_173_DIRECT_THETA_JSON_END"


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
    filterflow = _filterflow_direct_subprocess()
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_direct_theta_filterflow_blocker",
            filterflow.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow,
            reference_status,
        )

    bayesfilter = _bayesfilter_direct_probe(filterflow)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    comparison = _compare(filterflow, bayesfilter)
    decision = _decision(comparison, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_direct_current_step_theta_hypothesis_test",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "filterflow_direct_probe": _compact_side(filterflow),
        "bayesfilter_direct_probe": _compact_side(bayesfilter),
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_direct_theta_hypothesis_tf"
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
        "filterflow_direct_probe": filterflow,
        "bayesfilter_direct_probe": None,
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_direct_theta_hypothesis_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _filterflow_direct_subprocess() -> dict[str, Any]:
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
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_direct_script()],
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
            "blocker": "filterflow direct-theta subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind(SENTINEL_BEGIN)
    end = completed.stdout.rfind(SENTINEL_END)
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow direct-theta JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(completed.stdout[start + len(SENTINEL_BEGIN):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_direct_script() -> str:
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

        tfd = tfp.distributions
        DTYPE = tf.float64
        NP_DTYPE = np.float64
        T = {T}
        BATCH_SIZE = {BATCH_SIZE}
        N = {NUM_PARTICLES}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}
        TARGET_TIME_INDEX = {TARGET_TIME_INDEX}
        EPSILON = {EPSILON!r}
        SCALING = {SCALING!r}
        CONVERGENCE_THRESHOLD = {CONVERGENCE_THRESHOLD!r}
        MAX_ITERATIONS = {MAX_ITERATIONS}
        RESAMPLING_NEFF = {RESAMPLING_NEFF!r}
        THETA = np.array({THETA!r}, dtype=NP_DTYPE)

        def to_json(tensor):
            if tensor is None:
                return None
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, tf.float64).numpy())

        def field(tensor):
            if tensor is None:
                return {{"shape": None, "max_abs": None, "sum": None, "finite": False}}
            cast = tf.cast(tensor, DTYPE)
            return {{
                "shape": [int(v) for v in tensor.shape],
                "max_abs": scalar(tf.reduce_max(tf.abs(cast))),
                "sum": scalar(tf.reduce_sum(cast)),
                "finite": bool(tf.reduce_all(tf.math.is_finite(cast)).numpy()),
            }}

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
        target_bundle = None
        with tf.GradientTape(persistent=True) as full_tape:
            full_tape.watch(modifiable_transition_matrix)
            for time_index in range(T):
                seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
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
                resampled_state = apply_transport_matrix(state_with_ess, transport_matrix, flags)
                proposal_dist = smc._proposal_model._get_proposal_dist(resampled_state, observation)
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
                unnormalized_core = transition_ll + observation_ll - proposal_ll
                unnormalized = unnormalized_core + resampled_state.log_weights
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
                    target_bundle = {{
                        "target": tf.reduce_mean(log_likelihoods),
                        "post_particles": resampled_state.particles,
                        "post_log_weights": resampled_state.log_weights,
                        "observation": observation,
                        "proposed_particles": proposed_particles,
                        "transition_ll": transition_ll,
                        "observation_ll": observation_ll,
                        "proposal_ll": proposal_ll,
                        "unnormalized_core": unnormalized_core,
                        "unnormalized": unnormalized,
                        "increment": increment,
                        "flags": flags,
                    }}
                    break

        if target_bundle is None:
            raise RuntimeError("target time not reached")

        full_target = target_bundle["target"]
        full_total_gradient = watched_grad(full_tape, full_target, modifiable_transition_matrix)
        target_gradients = {{
            name: watched_grad(full_tape, full_target, tensor)
            for name, tensor in target_bundle.items()
            if name not in {{"target", "flags"}}
        }}
        del full_tape

        transition_cov_inv = tf.linalg.cholesky_solve(transition_covariance_chol, tf.eye(2, dtype=DTYPE))
        observation_cov_inv = tf.linalg.cholesky_solve(observation_covariance_chol, tf.eye(1, dtype=DTYPE))
        sigma_inv = transition_cov_inv + tf.linalg.matmul(
            tf.linalg.matmul(observation_matrix, observation_cov_inv, transpose_a=True),
            observation_matrix,
        )
        sigma = tf.linalg.inv(sigma_inv)
        sigma_chol = tf.linalg.cholesky(sigma)
        transition_noise = tfd.MultivariateNormalTriL(
            loc=tf.zeros([2], dtype=DTYPE),
            scale_tril=transition_covariance_chol,
        )
        observation_noise = tfd.MultivariateNormalTriL(
            loc=tf.zeros([1], dtype=DTYPE),
            scale_tril=observation_covariance_chol,
        )

        def direct_terms(matrix_variable, sample_mode):
            post_particles = tf.stop_gradient(target_bundle["post_particles"])
            post_log_weights = tf.stop_gradient(target_bundle["post_log_weights"])
            observation = tf.stop_gradient(target_bundle["observation"])
            proposal_mean = tf.linalg.matvec(
                observation_matrix,
                tf.linalg.matvec(observation_cov_inv, observation),
                transpose_a=True,
            )
            proposal_mean = proposal_mean + tf.linalg.matvec(
                transition_cov_inv,
                tf.linalg.matvec(matrix_variable, post_particles),
            )
            proposal_mean = tf.linalg.matvec(sigma, proposal_mean)
            if sample_mode == "frozen_sample":
                proposed_particles = tf.stop_gradient(target_bundle["proposed_particles"])
            elif sample_mode == "active_sample":
                proposed_particles = proposal_mean + tf.stop_gradient(
                    target_bundle["proposed_particles"] - proposal_mean
                )
            else:
                raise ValueError(sample_mode)
            observation_error = observation - tf.linalg.matvec(
                observation_matrix,
                proposed_particles,
            )
            observation_ll = observation_noise.log_prob(observation_error)
            transition_pushed = tf.linalg.matvec(matrix_variable, post_particles)
            transition_ll = transition_noise.log_prob(proposed_particles - transition_pushed)
            proposal_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
            proposal_ll = proposal_dist.log_prob(proposed_particles)
            unnormalized_core = transition_ll + observation_ll - proposal_ll
            unnormalized = unnormalized_core + post_log_weights
            increment = tf.reduce_logsumexp(unnormalized, axis=1)
            return {{
                "proposal_mean": proposal_mean,
                "proposed_particles": proposed_particles,
                "transition_ll": transition_ll,
                "observation_ll": observation_ll,
                "proposal_ll": proposal_ll,
                "unnormalized_core": unnormalized_core,
                "unnormalized": unnormalized,
                "increment": increment,
            }}

        direct_rows = {{}}
        for sample_mode in ("frozen_sample", "active_sample"):
            matrix_var = tf.Variable(transition_matrix, dtype=DTYPE)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(matrix_var)
                terms = direct_terms(matrix_var, sample_mode)
                objectives = {{
                    "transition_ll": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["transition_ll"]) * terms["transition_ll"]
                    ),
                    "observation_ll": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["observation_ll"]) * terms["observation_ll"]
                    ),
                    "proposal_ll": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["proposal_ll"]) * terms["proposal_ll"]
                    ),
                    "unnormalized_core": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["unnormalized_core"])
                        * terms["unnormalized_core"]
                    ),
                    "unnormalized": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["unnormalized"]) * terms["unnormalized"]
                    ),
                    "increment": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["increment"]) * terms["increment"]
                    ),
                    "proposal_mean": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["proposed_particles"])
                        * terms["proposal_mean"]
                    ),
                }}
            term_gradients = {{
                name: watched_grad(tape, objective, matrix_var)
                for name, objective in objectives.items()
            }}
            del tape
            direct_rows[sample_mode] = {{
                "term_values": {{name: field(value) for name, value in terms.items()}},
                "term_value_tensors": {{name: to_json(value) for name, value in terms.items()}},
                "term_gradient_matrix_summaries": {{
                    name: field(value) for name, value in term_gradients.items()
                }},
                "term_gradient_matrix_tensors": {{
                    name: to_json(value) for name, value in term_gradients.items()
                }},
                "term_gradient_diag": {{
                    name: to_json(tf.linalg.diag_part(value))
                    for name, value in term_gradients.items()
                }},
                "term_gradient_offdiag": {{
                    name: [scalar(value[0, 1]), scalar(value[1, 0])]
                    for name, value in term_gradients.items()
                }},
            }}

        payload = {{
            "status": "executed",
            "backend": "executable_filterflow_subprocess",
            "settings": {{
                "theta": THETA.astype(float).tolist(),
                "mesh_index": {MESH_INDEX},
                "target_time_index": TARGET_TIME_INDEX,
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
                "dtype": "float64",
            }},
            "target_scalar": scalar(full_target),
            "total_gradient_matrix": to_json(full_total_gradient),
            "total_gradient_diag": to_json(tf.linalg.diag_part(full_total_gradient)),
            "target_gradient_summaries": {{
                name: field(value) for name, value in target_gradients.items()
            }},
            "target_gradient_tensors": {{
                name: to_json(value) for name, value in target_gradients.items()
            }},
            "frozen_values": {{
                name: field(value)
                for name, value in target_bundle.items()
                if name not in {{"target", "flags"}}
            }},
            "frozen_value_tensors": {{
                name: to_json(value)
                for name, value in target_bundle.items()
                if name not in {{"target", "flags"}}
            }},
            "direct_rows": direct_rows,
            "resampling_flag": [bool(v) for v in tf.reshape(target_bundle["flags"], [-1]).numpy().tolist()],
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
            "observation_checksum": float(np.sum(observations_np.astype(np.float64))),
            "initial_particles_checksum": float(np.sum(initial_particles_np.astype(np.float64))),
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
                "tensorflow_probability": tfp.__version__,
                "numpy": np.__version__,
            }},
        }}
        print("{SENTINEL_BEGIN}")
        print(json.dumps(payload, sort_keys=True))
        print("{SENTINEL_END}")
        """
    )


def _bayesfilter_direct_probe(filterflow: dict[str, Any]) -> dict[str, Any]:
    model = filterflow["model"]
    observation_matrix = tf.constant(model["observation_matrix"], dtype=DTYPE)
    transition_chol = tf.constant(model["transition_covariance_chol"], dtype=DTYPE)
    observation_chol = tf.constant(model["observation_covariance_chol"], dtype=DTYPE)
    post_particles = tf.constant(
        filterflow["frozen_value_tensors"]["post_particles"],
        dtype=DTYPE,
    )
    post_log_weights = tf.constant(
        filterflow["frozen_value_tensors"]["post_log_weights"],
        dtype=DTYPE,
    )
    observation = tf.constant(
        filterflow["frozen_value_tensors"]["observation"],
        dtype=DTYPE,
    )
    frozen_proposed_particles = tf.constant(
        filterflow["frozen_value_tensors"]["proposed_particles"],
        dtype=DTYPE,
    )
    target_gradients = {
        name: tf.constant(value, dtype=DTYPE)
        for name, value in filterflow["target_gradient_tensors"].items()
    }

    transition_cov_inv = tf.linalg.cholesky_solve(transition_chol, tf.eye(2, dtype=DTYPE))
    observation_cov_inv = tf.linalg.cholesky_solve(observation_chol, tf.eye(1, dtype=DTYPE))
    sigma_inv = transition_cov_inv + tf.linalg.matmul(
        tf.linalg.matmul(observation_matrix, observation_cov_inv, transpose_a=True),
        observation_matrix,
    )
    sigma = tf.linalg.inv(sigma_inv)
    sigma_chol = tf.linalg.cholesky(sigma)
    transition_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([2], dtype=DTYPE),
        scale_tril=transition_chol,
    )
    observation_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([1], dtype=DTYPE),
        scale_tril=observation_chol,
    )

    def terms_from_matrix(matrix: tf.Tensor, sample_mode: str) -> dict[str, tf.Tensor]:
        proposal_mean = tf.linalg.matvec(
            observation_matrix,
            tf.linalg.matvec(observation_cov_inv, observation),
            transpose_a=True,
        )
        proposal_mean = proposal_mean + tf.linalg.matvec(
            transition_cov_inv,
            tf.linalg.matvec(matrix, post_particles),
        )
        proposal_mean = tf.linalg.matvec(sigma, proposal_mean)
        if sample_mode == "frozen_sample":
            proposed_particles = tf.stop_gradient(frozen_proposed_particles)
        elif sample_mode == "active_sample":
            proposed_particles = proposal_mean + tf.stop_gradient(
                frozen_proposed_particles - proposal_mean
            )
        else:
            raise ValueError(f"unknown sample mode {sample_mode}")
        observation_error = observation - tf.linalg.matvec(
            observation_matrix,
            proposed_particles,
        )
        observation_ll = observation_noise.log_prob(observation_error)
        pushed = tf.linalg.matvec(matrix, post_particles)
        transition_ll = transition_noise.log_prob(proposed_particles - pushed)
        proposal_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
        proposal_ll = proposal_dist.log_prob(proposed_particles)
        unnormalized_core = transition_ll + observation_ll - proposal_ll
        unnormalized = unnormalized_core + post_log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        return {
            "proposal_mean": proposal_mean,
            "proposed_particles": proposed_particles,
            "transition_ll": transition_ll,
            "observation_ll": observation_ll,
            "proposal_ll": proposal_ll,
            "unnormalized_core": unnormalized_core,
            "unnormalized": unnormalized,
            "increment": increment,
        }

    full_matrix = tf.constant([[THETA[0], 1.0], [0.0, THETA[1]]], dtype=DTYPE)
    direct_rows: dict[str, Any] = {}
    for parameterization in ("theta_vector", "full_matrix"):
        for sample_mode in ("frozen_sample", "active_sample"):
            key = f"{parameterization}:{sample_mode}"
            if parameterization == "theta_vector":
                variable = tf.Variable(THETA, dtype=DTYPE)

                def matrix_from_variable() -> tf.Tensor:
                    return tf.linalg.diag(variable) + tf.constant(
                        [[0.0, 1.0], [0.0, 0.0]],
                        dtype=DTYPE,
                    )

                def project_gradient(gradient: tf.Tensor) -> tf.Tensor:
                    return gradient

            else:
                variable = tf.Variable(full_matrix, dtype=DTYPE)

                def matrix_from_variable() -> tf.Tensor:
                    return variable

                def project_gradient(gradient: tf.Tensor) -> tf.Tensor:
                    return tf.linalg.diag_part(gradient)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(variable)
                terms = terms_from_matrix(matrix_from_variable(), sample_mode)
                objectives = {
                    "transition_ll": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["transition_ll"]) * terms["transition_ll"]
                    ),
                    "observation_ll": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["observation_ll"]) * terms["observation_ll"]
                    ),
                    "proposal_ll": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["proposal_ll"]) * terms["proposal_ll"]
                    ),
                    "unnormalized_core": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["unnormalized_core"])
                        * terms["unnormalized_core"]
                    ),
                    "unnormalized": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["unnormalized"]) * terms["unnormalized"]
                    ),
                    "increment": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["increment"]) * terms["increment"]
                    ),
                    "proposal_mean": tf.reduce_sum(
                        tf.stop_gradient(target_gradients["proposed_particles"])
                        * terms["proposal_mean"]
                    ),
                }
            raw_gradients = {
                name: _safe_gradient(tape, objective, variable)
                for name, objective in objectives.items()
            }
            del tape
            projected = {
                name: project_gradient(gradient)
                for name, gradient in raw_gradients.items()
            }
            direct_rows[key] = {
                "parameterization": parameterization,
                "sample_mode": sample_mode,
                "term_values": {name: _field(value) for name, value in terms.items()},
                "term_value_tensors": {name: r3._json(value) for name, value in terms.items()},
                "term_gradient_summaries": {
                    name: _field(value) for name, value in projected.items()
                },
                "term_gradient_tensors": {
                    name: r3._json(value) for name, value in projected.items()
                },
                "raw_gradient_summaries": {
                    name: _field(value) for name, value in raw_gradients.items()
                },
                "raw_gradient_tensors": {
                    name: r3._json(value) for name, value in raw_gradients.items()
                },
            }

    return {
        "status": "executed",
        "backend": "tensorflow_tensorflow_probability",
        "settings": filterflow["settings"],
        "direct_rows": direct_rows,
        "cpu_only_manifest": _parent_cpu_manifest(),
    }


def _compare(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    if bayesfilter.get("status") != "executed":
        return {"status": "blocked", "blocker": "BayesFilter direct probe did not execute"}
    ff_reference = filterflow["direct_rows"]["frozen_sample"]
    ff_rows = {
        "filterflow_full_matrix:frozen_sample": ff_reference,
        "filterflow_full_matrix:active_sample": filterflow["direct_rows"]["active_sample"],
    }
    rows = {}
    value_vetoes = []
    gradient_nonfinite = []
    for name, bf_row in bayesfilter["direct_rows"].items():
        ff_row = ff_reference
        row = _compare_direct_row(ff_row, bf_row)
        rows[name] = row
        if row["max_value_delta"] > VALUE_TOLERANCE:
            value_vetoes.append(name)
        if not row["all_gradient_finite"]:
            gradient_nonfinite.append(name)
    for name, ff_row in ff_rows.items():
        row = _self_compare_filterflow_row(ff_reference, ff_row)
        rows[name] = row
    term_delta_rows = _term_delta_rows(filterflow, bayesfilter)
    best_increment_row = _best_row(rows, "increment")
    best_core_row = _best_row(rows, "unnormalized_core")
    explanation = _interpret_hypotheses(rows, term_delta_rows, value_vetoes, gradient_nonfinite)
    return {
        "status": "compared",
        "rows": rows,
        "term_delta_rows": term_delta_rows,
        "value_vetoes": value_vetoes,
        "gradient_nonfinite_rows": gradient_nonfinite,
        "best_increment_row": best_increment_row,
        "best_core_row": best_core_row,
        "interpretation": explanation,
        "observed_total_gradient_delta": OBSERVED_TOTAL_GRADIENT_DELTA,
    }


def _compare_direct_row(ff_row: dict[str, Any], bf_row: dict[str, Any]) -> dict[str, Any]:
    value_deltas = {
        name: _max_abs_nested_delta(
            bf_row["term_value_tensors"][name],
            ff_row["term_value_tensors"][name],
        )
        for name in ff_row["term_value_tensors"]
    }
    gradient_deltas = {
        name: _vector_delta(
            bf_row["term_gradient_tensors"][name],
            ff_row["term_gradient_diag"][name],
        )
        for name in ff_row["term_gradient_diag"]
    }
    gradient_finite = {
        name: bool(summary["finite"])
        for name, summary in bf_row["term_gradient_summaries"].items()
    }
    return {
        "status": "compared",
        "parameterization": bf_row["parameterization"],
        "sample_mode": bf_row["sample_mode"],
        "value_deltas": value_deltas,
        "gradient_deltas": gradient_deltas,
        "max_value_delta": max(value_deltas.values()) if value_deltas else 0.0,
        "max_gradient_delta": max(
            max(abs(value) for value in delta)
            for delta in gradient_deltas.values()
        ),
        "all_gradient_finite": all(gradient_finite.values()),
        "bayesfilter_term_gradients": bf_row["term_gradient_tensors"],
        "filterflow_term_gradients": ff_row["term_gradient_diag"],
    }


def _self_compare_filterflow_row(reference: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    value_deltas = {
        name: _max_abs_nested_delta(
            row["term_value_tensors"][name],
            reference["term_value_tensors"][name],
        )
        for name in reference["term_value_tensors"]
    }
    gradient_deltas = {
        name: _vector_delta(
            row["term_gradient_diag"][name],
            reference["term_gradient_diag"][name],
        )
        for name in reference["term_gradient_diag"]
    }
    return {
        "status": "filterflow_internal_control",
        "parameterization": "full_matrix",
        "sample_mode": row.get("sample_mode", "unknown"),
        "value_deltas": value_deltas,
        "gradient_deltas": gradient_deltas,
        "max_value_delta": max(value_deltas.values()) if value_deltas else 0.0,
        "max_gradient_delta": max(
            max(abs(value) for value in delta)
            for delta in gradient_deltas.values()
        ),
        "all_gradient_finite": all(
            bool(summary["finite"])
            for summary in row["term_gradient_matrix_summaries"].values()
        ),
        "filterflow_term_gradients": row["term_gradient_diag"],
    }


def _term_delta_rows(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    observed = OBSERVED_TOTAL_GRADIENT_DELTA
    ff_reference = filterflow["direct_rows"]["frozen_sample"]
    for row_name, bf_row in bayesfilter["direct_rows"].items():
        for term_name, gradient in bf_row["term_gradient_tensors"].items():
            delta = _vector_delta(gradient, ff_reference["term_gradient_diag"][term_name])
            residual_after_term = [observed[i] - delta[i] for i in range(len(observed))]
            rows[f"{row_name}:{term_name}"] = {
                "gradient_delta": delta,
                "max_abs_gradient_delta": max(abs(value) for value in delta),
                "residual_after_subtracting_from_observed": residual_after_term,
                "max_abs_residual_after_subtracting_from_observed": max(
                    abs(value) for value in residual_after_term
                ),
            }
    return rows


def _best_row(rows: dict[str, Any], term_name: str) -> dict[str, Any] | None:
    comparable = [
        (name, row)
        for name, row in rows.items()
        if row["status"] == "compared" and term_name in row["gradient_deltas"]
    ]
    if not comparable:
        return None
    name, row = min(
        comparable,
        key=lambda item: max(abs(value) for value in item[1]["gradient_deltas"][term_name]),
    )
    return {
        "row": name,
        "term": term_name,
        "gradient_delta": row["gradient_deltas"][term_name],
        "max_abs_gradient_delta": max(abs(value) for value in row["gradient_deltas"][term_name]),
    }


def _interpret_hypotheses(
    rows: dict[str, Any],
    term_delta_rows: dict[str, Any],
    value_vetoes: list[str],
    gradient_nonfinite: list[str],
) -> str:
    if value_vetoes:
        return "value_veto_blocks_direct_theta_hypothesis_interpretation"
    if gradient_nonfinite:
        return "nonfinite_gradient_veto_blocks_direct_theta_hypothesis_interpretation"
    best_observed_match = min(
        term_delta_rows.items(),
        key=lambda item: item[1]["max_abs_residual_after_subtracting_from_observed"],
    )
    if best_observed_match[1]["max_abs_residual_after_subtracting_from_observed"] <= GRADIENT_TOLERANCE:
        return f"hypothesis_supported_by_{best_observed_match[0]}"
    if rows["theta_vector:frozen_sample"]["max_gradient_delta"] <= GRADIENT_TOLERANCE:
        return "h4_current_step_direct_derivatives_match_under_frozen_sample"
    if (
        rows["theta_vector:frozen_sample"]["max_gradient_delta"] > GRADIENT_TOLERANCE
        and rows["full_matrix:frozen_sample"]["max_gradient_delta"] <= GRADIENT_TOLERANCE
    ):
        return "h3_parameter_embedding_mismatch_supported"
    if rows["theta_vector:active_sample"]["max_gradient_delta"] < rows["theta_vector:frozen_sample"]["max_gradient_delta"]:
        return "h2_sample_topology_partially_reduces_direct_theta_delta"
    return "h1_direct_term_algebra_or_topology_mismatch_remains"


def _decision(comparison: dict[str, Any], comparator_drift: bool) -> str:
    if comparator_drift:
        return "filterflow_float64_row_173_direct_theta_blocked_by_comparator_drift"
    if comparison.get("status") != "compared":
        return "filterflow_float64_row_173_direct_theta_blocked"
    if comparison["value_vetoes"]:
        return "filterflow_float64_row_173_direct_theta_value_veto"
    if comparison["gradient_nonfinite_rows"]:
        return "filterflow_float64_row_173_direct_theta_nonfinite_veto"
    interpretation = comparison["interpretation"]
    if interpretation.startswith("hypothesis_supported_by_"):
        return "filterflow_float64_row_173_direct_theta_hypothesis_localized"
    if interpretation == "h4_current_step_direct_derivatives_match_under_frozen_sample":
        return "filterflow_float64_row_173_direct_theta_not_source"
    if interpretation == "h3_parameter_embedding_mismatch_supported":
        return "filterflow_float64_row_173_direct_theta_parameter_embedding_source"
    if interpretation == "h2_sample_topology_partially_reduces_direct_theta_delta":
        return "filterflow_float64_row_173_direct_theta_sample_topology_partial"
    return "filterflow_float64_row_173_direct_theta_unresolved"


def _evidence_contract() -> dict[str, Any]:
    return {
        "primary_comparator": "local executable float64 FilterFlow checkout",
        "primary_question": "which direct current-step theta derivative mechanism explains row 173 time 93",
        "primary_pass": "localize observed gradient delta to one tested term or parameterization, or report none",
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "finite_gradients": "veto_gate_not_correctness_claim",
        "mathematical_correctness": "not_concluded",
    }


def _model_contract() -> dict[str, Any]:
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "mesh_index": MESH_INDEX,
        "theta": THETA,
        "target_time_index": TARGET_TIME_INDEX,
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


def _compact_side(side: dict[str, Any] | None) -> dict[str, Any] | None:
    if side is None or side.get("status") != "executed":
        return side
    return {
        "status": side["status"],
        "backend": side["backend"],
        "settings": side["settings"],
        "target_scalar": side.get("target_scalar"),
        "total_gradient_diag": side.get("total_gradient_diag"),
        "resampling_flag": side.get("resampling_flag"),
        "direct_rows": _compact_direct_rows(side["direct_rows"]),
        "cpu_only_manifest": side["cpu_only_manifest"],
        "stderr_excerpt": side.get("stderr_excerpt", ""),
    }


def _compact_direct_rows(rows: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for name, row in rows.items():
        compact[name] = {
            "parameterization": row.get("parameterization", "filterflow_full_matrix"),
            "sample_mode": row.get("sample_mode", name),
            "term_values": row.get("term_values"),
            "term_gradient_summaries": row.get(
                "term_gradient_summaries",
                row.get("term_gradient_matrix_summaries"),
            ),
            "term_gradient_tensors": row.get(
                "term_gradient_tensors",
                row.get("term_gradient_diag"),
            ),
        }
    return compact


def _decision_table(decision: str, comparison: dict[str, Any]) -> list[dict[str, str]]:
    if comparison.get("status") != "compared":
        primary = comparison.get("blocker", decision)
        veto = decision
        next_action = "repair blocker before interpreting direct-theta hypotheses"
    elif comparison["value_vetoes"]:
        primary = f"value veto rows: {comparison['value_vetoes']}"
        veto = "value mismatch before gradient interpretation"
        next_action = "repair frozen-value replay before testing gradient hypotheses"
    elif comparison["gradient_nonfinite_rows"]:
        primary = f"nonfinite rows: {comparison['gradient_nonfinite_rows']}"
        veto = "nonfinite gradient"
        next_action = "repair nonfinite direct-gradient path before interpretation"
    else:
        primary = comparison["interpretation"]
        veto = "none"
        next_action = _next_action_for_interpretation(comparison["interpretation"])
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": veto,
            "main_uncertainty": "single row/time direct-gradient probe; no correctness claim",
            "next_justified_action": next_action,
            "not_concluded": "correctness of either implementation, production readiness, analytic gradient correctness",
        }
    ]


def _next_action_for_interpretation(interpretation: str) -> str:
    if interpretation.startswith("hypothesis_supported_by_"):
        return "patch only the isolated executable-FilterFlow arithmetic/topology rule if governance approves"
    if interpretation == "h4_current_step_direct_derivatives_match_under_frozen_sample":
        return "move the cut-set one boundary upstream or downstream because current direct term is not the source"
    if interpretation == "h3_parameter_embedding_mismatch_supported":
        return "test a full-matrix BayesFilter parameterization in the cumulative row-173 replay"
    if interpretation == "h2_sample_topology_partially_reduces_direct_theta_delta":
        return "build a narrower sample-topology probe before patching"
    return "inspect per-term delta rows and add a smaller arithmetic probe"


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
        "filterflow_direct_probe",
        "bayesfilter_direct_probe",
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
        "filterflow_float64_row_173_direct_theta_filterflow_blocker",
        "filterflow_float64_row_173_direct_theta_blocked_by_comparator_drift",
        "filterflow_float64_row_173_direct_theta_blocked",
        "filterflow_float64_row_173_direct_theta_value_veto",
        "filterflow_float64_row_173_direct_theta_nonfinite_veto",
        "filterflow_float64_row_173_direct_theta_hypothesis_localized",
        "filterflow_float64_row_173_direct_theta_not_source",
        "filterflow_float64_row_173_direct_theta_parameter_embedding_source",
        "filterflow_float64_row_173_direct_theta_sample_topology_partial",
        "filterflow_float64_row_173_direct_theta_unresolved",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    for label in ("filterflow_direct_probe", "bayesfilter_direct_probe"):
        side = payload.get(label)
        if side is not None and side.get("status") == "executed":
            _validate_cpu(side["cpu_only_manifest"], label)
    comparison = payload["comparison"]
    if comparison.get("status") == "compared":
        if comparison["value_vetoes"] and "value_veto" not in payload["decision"]:
            raise ValueError("value veto rows present without value-veto decision")
        if comparison["gradient_nonfinite_rows"] and "nonfinite_veto" not in payload["decision"]:
            raise ValueError("nonfinite rows present without nonfinite-veto decision")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: Row 173 Direct-Theta Hypothesis Test",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Interpretation",
        "",
        f"`{payload['comparison'].get('interpretation')}`",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            "| "
            + " | ".join(
                str(row[key]).replace("\n", " ")
                for key in (
                    "decision",
                    "primary_criterion_status",
                    "veto_diagnostic_status",
                    "main_uncertainty",
                    "next_justified_action",
                    "not_concluded",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best Rows",
            "",
            _json_block(
                {
                    "best_increment_row": payload["comparison"].get("best_increment_row"),
                    "best_core_row": payload["comparison"].get("best_core_row"),
                    "observed_total_gradient_delta": payload["comparison"].get(
                        "observed_total_gradient_delta"
                    ),
                }
            ),
            "",
            "## Term Delta Rows",
            "",
            _json_block(payload["comparison"].get("term_delta_rows", {})),
            "",
            "## Direct Row Comparison",
            "",
            _json_block(payload["comparison"].get("rows", {})),
            "",
            "## Run Manifest",
            "",
            _json_block(payload["run_manifest"]),
            "",
            "## Non-Implications",
            "",
        ]
    )
    for item in payload["non_implications"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## JSON Output",
            "",
            f"`{payload['json_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _field(tensor: tf.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"shape": None, "max_abs": None, "sum": None, "finite": False}
    cast = tf.cast(tensor, DTYPE)
    return {
        "shape": [int(v) for v in tensor.shape],
        "max_abs": _float(tf.reduce_max(tf.abs(cast))),
        "sum": _float(tf.reduce_sum(cast)),
        "finite": bool(tf.reduce_all(tf.math.is_finite(cast)).numpy()),
    }


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _safe_gradient(tape: tf.GradientTape, target: tf.Tensor, tensor: tf.Tensor) -> tf.Tensor:
    gradient = tape.gradient(target, tensor)
    return tf.zeros_like(tensor) if gradient is None else gradient


def _vector_delta(left: Any, right: Any) -> list[float]:
    return [
        float(lhs) - float(rhs)
        for lhs, rhs in zip(_flatten(left), _flatten(right), strict=True)
    ]


def _flatten(value: Any) -> list[float]:
    values: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        else:
            values.append(float(item))

    visit(value)
    return values


def _max_abs_nested_delta(left: Any, right: Any) -> float:
    deltas = [abs(lhs - rhs) for lhs, rhs in zip(_flatten(left), _flatten(right), strict=True)]
    return max(deltas) if deltas else 0.0


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


def _non_implications() -> list[str]:
    return vjp._non_implications() + [
        "This is a difference audit only; no claim is made that either side is correct.",
        "A direct-term match or mismatch does not establish global smoothness-gradient correctness.",
        "The local float64 FilterFlow checkout is the executable comparator for this lane, not pristine upstream.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
