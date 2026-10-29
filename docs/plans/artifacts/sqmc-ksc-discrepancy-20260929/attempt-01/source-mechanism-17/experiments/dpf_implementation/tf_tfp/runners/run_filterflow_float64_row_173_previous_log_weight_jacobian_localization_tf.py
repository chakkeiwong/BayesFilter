"""Localize row-173 previous log-weight carry Jacobian differences."""

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
    run_filterflow_float64_smoothness_gradient_localization_tf as localizer,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_row_173_filterflow_unit_upstream_log_weight_tf as unit_probe,
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
    reference_policy,
    validate_filterflow_reference_status,
)


tfd = tfp.distributions

PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-previous-log-weight-jacobian-localization-plan-2026-06-06.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-previous-log-weight-jacobian-localization-result-2026-06-06.md"
)
REVIEW_LOOP_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-previous-log-weight-jacobian-localization-review-loop-2026-06-06.md"
)
JSON_PATH = (
    OUTPUT_DIR
    / "dpf_filterflow_float64_row_173_previous_log_weight_jacobian_localization_2026-06-06.json"
)
REPORT_PATH = (
    REPORT_DIR
    / "dpf-filterflow-float64-row-173-previous-log-weight-jacobian-localization-2026-06-06.md"
)
PRIOR_UNIT_JSON = (
    OUTPUT_DIR
    / "dpf_filterflow_float64_row_173_filterflow_unit_upstream_log_weight_2026-06-06.json"
)

TARGET_CARRY_TIME_INDEX = 93
PREVIOUS_TIME_INDEX = TARGET_CARRY_TIME_INDEX - 1
TAG = "row-173-previous-log-weight-jacobian-localization"
DTYPE = vjp.DTYPE
VALUE_TOLERANCE = vjp.VALUE_TOLERANCE
GRADIENT_TOLERANCE = vjp.GRADIENT_TOLERANCE

COMPONENT_FIELDS = (
    "normalized",
    "unnormalized",
    "signed_sum",
    "transition_ll",
    "observation_ll",
    "proposal_ll",
    "proposal_dist_log_prob",
    "fresh_dist_log_prob",
)
PRIMARY_COMPONENT_FIELDS = tuple(
    field for field in COMPONENT_FIELDS if field != "proposal_dist_log_prob"
)
INPUT_FIELDS = ("post_particles", "proposed_particles")
UNIT_TARGET_FIELD = "post_update_log_weights_to_proposed_particles_unit"


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
    validate_filterflow_reference_status(
        reference_status,
        marker_path=vjp.FILTERFLOW_MARKER_PATH,
    )
    initial_fingerprint = continuation._filterflow_fingerprint()
    prior = _load_prior_unit_artifact()
    filterflow = _filterflow_localization_subprocess()
    if filterflow.get("status") == "executed":
        bayesfilter_modes = [
            _bayesfilter_localization_mode(filterflow, "raw"),
            _bayesfilter_localization_mode(filterflow, "fresh_proposal_log_prob"),
        ]
    else:
        bayesfilter_modes = []
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(
        initial_fingerprint,
        final_fingerprint,
    )
    comparison = _compare(filterflow, bayesfilter_modes)
    veto_status = _veto_status(
        filterflow,
        bayesfilter_modes,
        comparison,
        prior,
        reference_status,
        comparator_drift,
    )
    classification, reason = _classify(comparison, veto_status)
    decision = _decision(classification, veto_status)
    return {
        "created_at_utc": utc_now(),
        "question": "row_173_time_92_previous_log_weight_jacobian_localization_difference_audit",
        "decision": decision,
        "hypothesis_classification": classification,
        "hypothesis_reason": reason,
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "prior_unit_artifact": _compact_prior(prior),
        "filterflow_localization": _compact_side(filterflow),
        "bayesfilter_localization_modes": [
            _compact_side(mode) for mode in bayesfilter_modes
        ],
        "comparison": comparison,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "evidence_contract": _evidence_contract(),
        "decision_table": _decision_table(decision, classification, veto_status),
        "non_implications": _non_implications(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_previous_log_weight_jacobian_localization_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "target_carry_time_index": TARGET_CARRY_TIME_INDEX,
            "previous_time_index": PREVIOUS_TIME_INDEX,
        "bayesfilter_modes": ["raw", "fresh_proposal_log_prob"],
            "prior_unit_artifact": str(PRIOR_UNIT_JSON.relative_to(REPO_ROOT)),
        },
    }


def _filterflow_localization_subprocess() -> dict[str, Any]:
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
        [str(vjp.FILTERFLOW_ENV_PYTHON), "-c", _filterflow_localization_script()],
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
            "blocker": "filterflow previous-log-weight localization subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_ROW_173_PREVIOUS_LOG_WEIGHT_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_ROW_173_PREVIOUS_LOG_WEIGHT_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow previous-log-weight JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(
        completed.stdout[
            start + len("FILTERFLOW_ROW_173_PREVIOUS_LOG_WEIGHT_JSON_BEGIN") : end
        ].strip()
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_localization_script() -> str:
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
        T = {vjp.T}
        BATCH_SIZE = {vjp.BATCH_SIZE}
        N = {vjp.NUM_PARTICLES}
        DATA_SEED = {vjp.DATA_SEED}
        FILTER_SEED = {vjp.FILTER_SEED}
        TARGET_TIME_INDEX = {PREVIOUS_TIME_INDEX}
        EPSILON = {vjp.EPSILON!r}
        SCALING = {vjp.SCALING!r}
        CONVERGENCE_THRESHOLD = {vjp.CONVERGENCE_THRESHOLD!r}
        MAX_ITERATIONS = {vjp.MAX_ITERATIONS}
        RESAMPLING_NEFF = {vjp.RESAMPLING_NEFF!r}
        THETA = np.array({vjp.THETA!r}, dtype=NP_DTYPE)
        COMPONENT_FIELDS = {list(COMPONENT_FIELDS)!r}
        INPUT_FIELDS = {list(INPUT_FIELDS)!r}

        def to_json(tensor):
            if tensor is None:
                return None
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, tf.float64).numpy())

        def max_abs(tensor):
            if tensor is None:
                return None
            return scalar(tf.reduce_max(tf.abs(tf.cast(tensor, DTYPE))))

        def finite(tensor):
            return bool(tf.reduce_all(tf.math.is_finite(tf.cast(tensor, DTYPE))).numpy())

        def field(tensor):
            return {{
                "shape": None if tensor is None else [int(v) for v in tensor.shape],
                "max_abs": max_abs(tensor),
                "sum": None if tensor is None else scalar(tf.reduce_sum(tf.cast(tensor, DTYPE))),
                "finite": tensor is not None and finite(tensor),
            }}

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
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(modifiable_transition_matrix)
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
                proposal_loc = distribution_loc(proposal_dist)
                proposal_mean = proposal_dist.mean()
                proposed_particles = proposal_dist.sample(seed=seed2)
                proposal_dist_log_prob = proposal_dist.log_prob(proposed_particles)
                fresh_proposal_dist = smc._proposal_model._get_proposal_dist(
                    resampled_state,
                    observation,
                )
                fresh_proposal_loc = distribution_loc(fresh_proposal_dist)
                fresh_proposal_mean = fresh_proposal_dist.mean()
                fresh_dist_log_prob = fresh_proposal_dist.log_prob(proposed_particles)
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
                signed_sum = transition_ll + observation_ll - proposal_ll
                unnormalized = signed_sum + resampled_state.log_weights
                increment = tf.reduce_logsumexp(unnormalized, 1)
                pre_current_log_likelihoods = resampled_state.log_likelihoods
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
                        "target": tf.reduce_mean(normalized),
                        "flags": flags,
                        "log_ess": tf.math.log(ess),
                        "pre_particles": state_with_ess.particles,
                        "pre_log_weights": state_with_ess.log_weights,
                        "transport_matrix": transport_matrix,
                        "post_particles": resampled_state.particles,
                        "post_log_weights": resampled_state.log_weights,
                        "proposal_loc": proposal_loc,
                        "proposal_mean": proposal_mean,
                        "fresh_proposal_loc": fresh_proposal_loc,
                        "fresh_proposal_mean": fresh_proposal_mean,
                        "proposed_particles": proposed_particles,
                        "observation_ll": observation_ll,
                        "transition_ll": transition_ll,
                        "proposal_ll": proposal_ll,
                        "proposal_dist_log_prob": proposal_dist_log_prob,
                        "fresh_dist_log_prob": fresh_dist_log_prob,
                        "signed_sum": signed_sum,
                        "unnormalized": unnormalized,
                        "increment": increment,
                        "pre_current_log_likelihoods": pre_current_log_likelihoods,
                        "normalized": normalized,
                        "post_update_log_weights": state.log_weights,
                        "post_update_log_likelihoods": state.log_likelihoods,
                    }}
                    break

        if target_bundle is None:
            raise RuntimeError("target time not reached")

        component_vjps = {{}}
        component_vjp_tensors = {{}}
        for component in COMPONENT_FIELDS:
            component_vjps[component] = {{}}
            component_vjp_tensors[component] = {{}}
            for input_name in INPUT_FIELDS:
                tensor = watched_grad_with_upstream(
                    tape,
                    target_bundle[component],
                    target_bundle[input_name],
                    tf.ones_like(target_bundle[component]),
                )
                component_vjps[component][input_name] = field(tensor)
                component_vjp_tensors[component][input_name] = to_json(tensor)

        target_vjps = {{}}
        target_vjp_tensors = {{}}
        for input_name in INPUT_FIELDS:
            tensor = watched_grad_with_upstream(
                tape,
                target_bundle["post_update_log_weights"],
                target_bundle[input_name],
                tf.ones_like(target_bundle["post_update_log_weights"]),
            )
            target_vjps[input_name] = field(tensor)
            target_vjp_tensors[input_name] = to_json(tensor)

        signed_sum_reconstruction = (
            component_vjp_tensors["transition_ll"]["proposed_particles"],
            component_vjp_tensors["observation_ll"]["proposed_particles"],
            component_vjp_tensors["proposal_ll"]["proposed_particles"],
        )
        del tape

        payload = {{
            "status": "executed",
            "backend": "executable_filterflow_subprocess",
            "settings": {{
                "theta": THETA.astype(float).tolist(),
                "mesh_index": {vjp.MESH_INDEX},
                "target_carry_time_index": {TARGET_CARRY_TIME_INDEX},
                "previous_time_index": TARGET_TIME_INDEX,
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
            "values": {{
                name: field(tensor)
                for name, tensor in target_bundle.items()
                if name not in {{"target", "flags"}}
            }},
            "value_tensors": {{
                name: to_json(tensor)
                for name, tensor in target_bundle.items()
                if name not in {{"target", "flags"}}
            }},
            "component_unit_vjps": component_vjps,
            "component_unit_vjp_tensors": component_vjp_tensors,
            "target_unit_vjps": target_vjps,
            "target_unit_vjp_tensors": target_vjp_tensors,
            "signed_sum_reconstruction_probe": {{
                "contract": (
                    "Explanatory only: component tensors are available to verify "
                    "signed_sum = transition_ll + observation_ll - proposal_ll."
                ),
                "tensors_recorded": signed_sum_reconstruction is not None,
            }},
            "target_scalar": scalar(target_bundle["target"]),
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
                "numpy": np.__version__,
            }},
        }}
        print("FILTERFLOW_ROW_173_PREVIOUS_LOG_WEIGHT_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_ROW_173_PREVIOUS_LOG_WEIGHT_JSON_END")
        """
    )


def _bayesfilter_localization_mode(
    filterflow: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        theta_variable = tf.Variable(vjp.THETA, dtype=DTYPE)
        model = vjp._model_from_filterflow(filterflow)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta_variable)
            bundle = _bayesfilter_previous_bundle(theta_variable, model, mode)
            target = bundle["target"]
        component_vjps, component_tensors = _component_unit_vjps(tape, bundle)
        target_vjps, target_tensors = _target_unit_vjps(tape, bundle)
        del tape
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "mode": mode,
            "mode_description": _mode_description(mode),
            "settings": filterflow["settings"],
            "values": {
                name: vjp._field(tensor)
                for name, tensor in bundle.items()
                if name not in {"target", "flags"}
            },
            "value_tensors": {
                name: r3._json(tensor)
                for name, tensor in bundle.items()
                if name not in {"target", "flags"}
            },
            "component_unit_vjps": component_vjps,
            "component_unit_vjp_tensors": component_tensors,
            "target_unit_vjps": target_vjps,
            "target_unit_vjp_tensors": target_tensors,
            "target_scalar": vjp._float(target),
            "resampling_flag": [
                bool(value) for value in tf.reshape(bundle["flags"], [-1]).numpy().tolist()
            ],
            "cpu_only_manifest": vjp._parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_previous_bundle(
    theta: tf.Tensor,
    model: dict[str, Any],
    mode: str,
) -> dict[str, tf.Tensor]:
    if mode not in {"raw", "fresh_proposal_log_prob"}:
        raise ValueError(f"unknown mode: {mode}")
    transition_matrix = localizer._transition_matrix(theta)
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
    transition_cov_inv = tf.linalg.cholesky_solve(
        transition_chol,
        tf.eye(2, dtype=DTYPE),
    )
    observation_cov_inv = tf.linalg.cholesky_solve(
        observation_chol,
        tf.eye(1, dtype=DTYPE),
    )
    sigma_inv = transition_cov_inv + tf.linalg.matmul(
        tf.linalg.matmul(observation_matrix, observation_cov_inv, transpose_a=True),
        observation_matrix,
    )
    sigma = tf.linalg.inv(sigma_inv)
    sigma_chol = tf.linalg.cholesky(sigma)
    seed = tf.constant(vjp.FILTER_SEED, dtype=tf.int32)
    paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
    seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))

    for time_index in range(vjp.T):
        seed, _seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
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
        pre_particles = particles
        pre_log_weights = log_weights
        transport_matrix = tf.cast(transported.transport_matrix, DTYPE)
        particles = transported.particles
        log_weights = transported.log_weights
        observation = observations[time_index]
        proposal_mean = localizer._optimal_proposal_mean(
            particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
        )
        proposal_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
        proposal_loc = proposal_dist.loc
        proposed_particles = proposal_dist.sample(seed=seed2)
        proposal_dist_log_prob = proposal_dist.log_prob(proposed_particles)
        fresh_proposal_mean = localizer._optimal_proposal_mean(
            particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
        )
        fresh_proposal_dist = tfd.MultivariateNormalTriL(
            fresh_proposal_mean,
            sigma_chol,
        )
        fresh_proposal_loc = fresh_proposal_dist.loc
        fresh_dist_log_prob = fresh_proposal_dist.log_prob(proposed_particles)
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
        proposal_ll = fresh_dist_log_prob
        signed_sum = transition_ll + observation_ll - proposal_ll
        unnormalized = signed_sum + log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        pre_current_log_likelihoods = log_likelihoods
        log_likelihoods = log_likelihoods + increment
        normalized = r3._filterflow_normalize(unnormalized, num_particles)
        log_weights = normalized
        particles = proposed_particles
        if time_index == PREVIOUS_TIME_INDEX:
            return {
                "target": tf.reduce_mean(normalized),
                "flags": flags,
                "log_ess": ess_log,
                "pre_particles": pre_particles,
                "pre_log_weights": pre_log_weights,
                "transport_matrix": transport_matrix,
                "post_particles": transported.particles,
                "post_log_weights": transported.log_weights,
                "proposal_loc": proposal_loc,
                "proposal_mean": proposal_mean,
                "fresh_proposal_loc": fresh_proposal_loc,
                "fresh_proposal_mean": fresh_proposal_mean,
                "proposed_particles": proposed_particles,
                "observation_ll": observation_ll,
                "transition_ll": transition_ll,
                "proposal_ll": proposal_ll,
                "proposal_dist_log_prob": proposal_dist_log_prob,
                "fresh_dist_log_prob": fresh_dist_log_prob,
                "signed_sum": signed_sum,
                "unnormalized": unnormalized,
                "increment": increment,
                "pre_current_log_likelihoods": pre_current_log_likelihoods,
                "normalized": normalized,
                "post_update_log_weights": log_weights,
                "post_update_log_likelihoods": log_likelihoods,
            }
    raise RuntimeError("previous time not reached")


def _component_unit_vjps(
    tape: tf.GradientTape,
    bundle: dict[str, tf.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    for component in COMPONENT_FIELDS:
        summaries[component] = {}
        tensors[component] = {}
        for input_name in INPUT_FIELDS:
            tensor = vjp._safe_gradient_with_upstream(
                tape,
                bundle[component],
                bundle[input_name],
                tf.ones_like(bundle[component]),
            )
            summaries[component][input_name] = vjp._field(tensor)
            tensors[component][input_name] = r3._json(tensor)
    return summaries, tensors


def _target_unit_vjps(
    tape: tf.GradientTape,
    bundle: dict[str, tf.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    for input_name in INPUT_FIELDS:
        tensor = vjp._safe_gradient_with_upstream(
            tape,
            bundle["post_update_log_weights"],
            bundle[input_name],
            tf.ones_like(bundle["post_update_log_weights"]),
        )
        summaries[input_name] = vjp._field(tensor)
        tensors[input_name] = r3._json(tensor)
    return summaries, tensors


def _mode_description(mode: str) -> str:
    if mode == "raw":
        return (
            "BayesFilter default replay with proposal log_prob evaluated on a "
            "freshly recomputed proposal distribution, matching FilterFlow's "
            "proposal.loglikelihood topology. The same-distribution "
            "proposal_dist_log_prob tensor is retained as an explanatory "
            "diagnostic only."
        )
    if mode == "fresh_proposal_log_prob":
        return "BayesFilter replay with proposal log_prob evaluated on a freshly recomputed proposal distribution."
    raise ValueError(f"unknown mode: {mode}")


def _load_prior_unit_artifact() -> dict[str, Any]:
    if not PRIOR_UNIT_JSON.exists():
        return {"status": "missing", "path": str(PRIOR_UNIT_JSON)}
    payload = load_json(PRIOR_UNIT_JSON)
    return {"status": "loaded", "path": str(PRIOR_UNIT_JSON), "payload": payload}


def _compare(
    filterflow: dict[str, Any],
    bayesfilter_modes: list[dict[str, Any]],
) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return {"status": "blocked", "blocker": filterflow.get("blocker")}
    if len(bayesfilter_modes) != 2 or any(
        mode.get("status") != "executed" for mode in bayesfilter_modes
    ):
        return {
            "status": "blocked",
            "blocker": "BayesFilter localization modes did not all execute",
        }
    modes = {mode["mode"]: mode for mode in bayesfilter_modes}
    mode_rows = {
        name: _mode_row(filterflow, mode)
        for name, mode in modes.items()
    }
    return {
        "status": "compared",
        "mode_rows": mode_rows,
        "raw_primary_rows": mode_rows["raw"]["primary_rows"],
        "fresh_primary_rows": mode_rows["fresh_proposal_log_prob"]["primary_rows"],
        "raw_value_gate_pass": mode_rows["raw"]["value_gate_pass"],
        "fresh_value_gate_pass": mode_rows["fresh_proposal_log_prob"]["value_gate_pass"],
        "raw_resampling_flags_match": mode_rows["raw"]["resampling_flags_match"],
        "fresh_resampling_flags_match": mode_rows[
            "fresh_proposal_log_prob"
        ]["resampling_flags_match"],
        "first_raw_primary_delta": vjp._first_delta(
            mode_rows["raw"]["primary_rows"],
            GRADIENT_TOLERANCE,
        ),
        "first_fresh_primary_delta": vjp._first_delta(
            mode_rows["fresh_proposal_log_prob"]["primary_rows"],
            GRADIENT_TOLERANCE,
        ),
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
    }


def _mode_row(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    value_rows = _value_rows(filterflow, bayesfilter)
    component_rows = _component_rows(filterflow, bayesfilter)
    target_rows = _target_rows(filterflow, bayesfilter)
    primary_rows = _primary_rows(component_rows, target_rows)
    first_value_delta = vjp._first_delta(value_rows, VALUE_TOLERANCE)
    first_primary_delta = vjp._first_delta(primary_rows, GRADIENT_TOLERANCE)
    return {
        "mode": bayesfilter["mode"],
        "mode_description": bayesfilter["mode_description"],
        "filterflow_target_scalar": filterflow["target_scalar"],
        "bayesfilter_target_scalar": bayesfilter["target_scalar"],
        "target_scalar_delta": abs(
            float(bayesfilter["target_scalar"]) - float(filterflow["target_scalar"])
        ),
        "resampling_flags_match": (
            bayesfilter["resampling_flag"] == filterflow["resampling_flag"]
        ),
        "filterflow_resampling_flag": filterflow["resampling_flag"],
        "bayesfilter_resampling_flag": bayesfilter["resampling_flag"],
        "value_rows": value_rows,
        "component_rows": component_rows,
        "target_rows": target_rows,
        "primary_rows": primary_rows,
        "first_value_delta": first_value_delta,
        "first_primary_delta": first_primary_delta,
        "value_gate_pass": first_value_delta["status"] == "no_delta",
        "primary_vjps_within_tolerance": first_primary_delta["status"] == "no_delta",
    }


def _value_rows(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name in COMPONENT_FIELDS + INPUT_FIELDS + (
        "post_log_weights",
        "post_update_log_weights",
    ):
        rows[name] = _delta_row(
            filterflow["values"][name],
            bayesfilter["values"][name],
            filterflow["value_tensors"][name],
            bayesfilter["value_tensors"][name],
        )
    return rows


def _component_rows(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for component in COMPONENT_FIELDS:
        rows[component] = {}
        for input_name in INPUT_FIELDS:
            rows[component][input_name] = _delta_row(
                filterflow["component_unit_vjps"][component][input_name],
                bayesfilter["component_unit_vjps"][component][input_name],
                filterflow["component_unit_vjp_tensors"][component][input_name],
                bayesfilter["component_unit_vjp_tensors"][component][input_name],
            )
    return rows


def _target_rows(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    return {
        input_name: _delta_row(
            filterflow["target_unit_vjps"][input_name],
            bayesfilter["target_unit_vjps"][input_name],
            filterflow["target_unit_vjp_tensors"][input_name],
            bayesfilter["target_unit_vjp_tensors"][input_name],
        )
        for input_name in INPUT_FIELDS
    }


def _primary_rows(
    component_rows: dict[str, Any],
    target_rows: dict[str, Any],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    rows[UNIT_TARGET_FIELD] = target_rows["proposed_particles"]
    for component in PRIMARY_COMPONENT_FIELDS:
        rows[f"{component}_to_proposed_particles_unit"] = component_rows[component][
            "proposed_particles"
        ]
    for component in PRIMARY_COMPONENT_FIELDS:
        rows[f"{component}_to_post_particles_unit"] = component_rows[component][
            "post_particles"
        ]
    return rows


def _delta_row(
    filterflow_summary: dict[str, Any],
    bayesfilter_summary: dict[str, Any],
    filterflow_tensor: Any,
    bayesfilter_tensor: Any,
) -> dict[str, Any]:
    return {
        "status": "compared",
        "shape_match": filterflow_summary["shape"] == bayesfilter_summary["shape"],
        "finite": bool(filterflow_summary["finite"] and bayesfilter_summary["finite"]),
        "filterflow_max_abs": filterflow_summary["max_abs"],
        "bayesfilter_max_abs": bayesfilter_summary["max_abs"],
        "filterflow_sum": filterflow_summary["sum"],
        "bayesfilter_sum": bayesfilter_summary["sum"],
        "max_abs_delta": vjp._max_abs_nested_delta(
            bayesfilter_tensor,
            filterflow_tensor,
        ),
        "sum_delta": _sum_nested_delta(
            bayesfilter_tensor,
            filterflow_tensor,
        ),
    }


def _sum_nested_delta(left: Any, right: Any) -> float:
    deltas: list[float] = []

    def visit(lhs: Any, rhs: Any) -> None:
        if isinstance(lhs, list) and isinstance(rhs, list):
            for lhs_item, rhs_item in zip(lhs, rhs, strict=True):
                visit(lhs_item, rhs_item)
        else:
            deltas.append(float(lhs) - float(rhs))

    visit(left, right)
    return sum(deltas)


def _veto_status(
    filterflow: dict[str, Any],
    bayesfilter_modes: list[dict[str, Any]],
    comparison: dict[str, Any],
    prior: dict[str, Any],
    reference_status: dict[str, Any],
    comparator_drift: bool,
) -> dict[str, Any]:
    path_boundary = continuation._path_boundary_manifest()
    cpu_rows = {
        "parent": _cpu_status(
            environment_manifest(
                command="parent-cpu-check",
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            )
        ),
    }
    if filterflow.get("status") == "executed":
        cpu_rows["filterflow"] = _cpu_status(filterflow["cpu_only_manifest"])
    for mode in bayesfilter_modes:
        if mode.get("status") == "executed":
            cpu_rows[f"bayesfilter_{mode['mode']}"] = _cpu_status(
                mode["cpu_only_manifest"]
            )
    if prior.get("status") == "loaded":
        cpu_rows["prior_run_manifest"] = _cpu_status(prior["payload"]["run_manifest"])
    prior_artifact_valid = (
        prior.get("status") == "loaded"
        and prior["payload"].get("decision")
        == "filterflow_float64_row_173_filterflow_unit_upstream_log_weight_unit_factor_differs"
        and prior["payload"].get("hypothesis_classification")
        == "h2_unit_upstream_factor_differs"
    )
    value_gate_pass = (
        comparison.get("status") == "compared"
        and comparison["raw_value_gate_pass"]
        and comparison["fresh_value_gate_pass"]
    )
    resampling_flags_match = (
        comparison.get("status") == "compared"
        and comparison["raw_resampling_flags_match"]
        and comparison["fresh_resampling_flags_match"]
    )
    required_tensors_present = _required_tensors_present(filterflow, bayesfilter_modes)
    required_tensors_finite = _required_tensors_finite(comparison)
    status = {
        "all_vetoes_clear": False,
        "filterflow_executed": filterflow.get("status") == "executed",
        "bayesfilter_modes_executed": len(bayesfilter_modes) == 2
        and all(mode.get("status") == "executed" for mode in bayesfilter_modes),
        "prior_artifact_loaded": prior.get("status") == "loaded",
        "prior_artifact_valid": prior_artifact_valid,
        "comparator_drift": comparator_drift,
        "reference_status_validated": bool(reference_status),
        "value_gate_pass": value_gate_pass,
        "resampling_flags_match": resampling_flags_match,
        "required_tensors_present": required_tensors_present,
        "required_tensors_finite": required_tensors_finite,
        "cpu_only_pass": all(row["pass"] for row in cpu_rows.values()),
        "cpu_rows": cpu_rows,
        "path_boundary_clean": not any(bool(value) for value in path_boundary.values()),
        "path_boundary_manifest": path_boundary,
    }
    status["all_vetoes_clear"] = bool(
        status["filterflow_executed"]
        and status["bayesfilter_modes_executed"]
        and status["prior_artifact_loaded"]
        and status["prior_artifact_valid"]
        and not status["comparator_drift"]
        and status["value_gate_pass"]
        and status["resampling_flags_match"]
        and status["required_tensors_present"]
        and status["required_tensors_finite"]
        and status["cpu_only_pass"]
        and status["path_boundary_clean"]
    )
    return status


def _required_tensors_present(
    filterflow: dict[str, Any],
    bayesfilter_modes: list[dict[str, Any]],
) -> bool:
    if filterflow.get("status") != "executed":
        return False
    if len(bayesfilter_modes) != 2:
        return False
    try:
        for component in COMPONENT_FIELDS:
            for input_name in INPUT_FIELDS:
                filterflow["component_unit_vjps"][component][input_name]
                filterflow["component_unit_vjp_tensors"][component][input_name]
        for input_name in INPUT_FIELDS:
            filterflow["target_unit_vjps"][input_name]
            filterflow["target_unit_vjp_tensors"][input_name]
        for mode in bayesfilter_modes:
            for component in COMPONENT_FIELDS:
                for input_name in INPUT_FIELDS:
                    mode["component_unit_vjps"][component][input_name]
                    mode["component_unit_vjp_tensors"][component][input_name]
            for input_name in INPUT_FIELDS:
                mode["target_unit_vjps"][input_name]
                mode["target_unit_vjp_tensors"][input_name]
    except (KeyError, TypeError):
        return False
    return True


def _required_tensors_finite(comparison: dict[str, Any]) -> bool:
    if comparison.get("status") != "compared":
        return False
    for mode_row in comparison["mode_rows"].values():
        for row in mode_row["primary_rows"].values():
            if not row["shape_match"] or not row["finite"]:
                return False
    return True


def _cpu_status(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "cuda_visible_devices": manifest.get("cuda_visible_devices"),
        "pre_import_cuda_visible_devices": manifest.get(
            "pre_import_cuda_visible_devices"
        ),
        "gpu_devices_visible": manifest.get("gpu_devices_visible"),
        "pass": (
            manifest.get("cuda_visible_devices") == "-1"
            and manifest.get("pre_import_cuda_visible_devices") == "-1"
            and manifest.get("gpu_devices_visible") == []
        ),
    }


def _classify(
    comparison: dict[str, Any],
    veto_status: dict[str, Any],
) -> tuple[str, str]:
    if not veto_status.get("all_vetoes_clear", False):
        return "h1_blocked_or_vetoed", "one or more required veto gates failed"
    raw_rows = comparison["raw_primary_rows"]
    fresh_rows = comparison["fresh_primary_rows"]
    raw_proposal = raw_rows["proposal_ll_to_proposed_particles_unit"]
    raw_signed = raw_rows["signed_sum_to_proposed_particles_unit"]
    raw_transition = raw_rows["transition_ll_to_proposed_particles_unit"]
    raw_observation = raw_rows["observation_ll_to_proposed_particles_unit"]
    raw_normalized = raw_rows[UNIT_TARGET_FIELD]
    raw_unnormalized = raw_rows["unnormalized_to_proposed_particles_unit"]
    fresh_proposal = fresh_rows["proposal_ll_to_proposed_particles_unit"]
    fresh_signed = fresh_rows["signed_sum_to_proposed_particles_unit"]
    fresh_normalized = fresh_rows[UNIT_TARGET_FIELD]

    if (
        not _row_material(raw_proposal)
        and not _row_material(raw_signed)
        and not _row_material(raw_normalized)
        and not _row_material(fresh_proposal)
        and not _row_material(fresh_signed)
        and not _row_material(fresh_normalized)
    ):
        return (
            "h0_filterflow_contract_matches",
            "BayesFilter default proposal-log-prob route now matches FilterFlow at the proposal, signed-sum, and normalized carry VJPs",
        )

    if _row_material(raw_proposal) and _row_material(raw_signed):
        if not _row_material(fresh_proposal) and not _row_material(fresh_signed):
            return (
                "h2_proposal_log_prob_route_differs",
                "raw BayesFilter proposal-log-prob VJP differs, but the fresh proposal-log-prob route matches FilterFlow",
            )
        return (
            "h2_proposal_log_prob_route_differs",
            "proposal-log-prob VJP and signed component sum differ materially in the raw route",
        )
    if _row_material(raw_transition) or _row_material(raw_observation):
        return (
            "h3_transition_or_observation_route_differs",
            "transition or observation VJP differs before the proposal route explains the signed sum",
        )
    if not _row_material(raw_unnormalized) and _row_material(raw_normalized):
        return (
            "h4_normalization_route_differs",
            "signed unnormalized component VJP matches but normalized carry VJP differs materially",
        )
    if (
        float(raw_rows["signed_sum_to_proposed_particles_unit"]["filterflow_max_abs"])
        <= GRADIENT_TOLERANCE
        and float(raw_rows["signed_sum_to_proposed_particles_unit"]["bayesfilter_max_abs"])
        > GRADIENT_TOLERANCE
    ):
        return (
            "h5_filterflow_zero_bayesfilter_nonzero_localized_to_signed_sum",
            "FilterFlow signed component VJP is near zero while BayesFilter signed component VJP is materially nonzero",
        )
    if not _row_material(fresh_normalized):
        return (
            "h2_proposal_log_prob_route_differs",
            "fresh proposal-log-prob BayesFilter route collapses the normalized carry VJP difference",
        )
    return (
        "h6_unresolved_previous_log_weight_jacobian",
        "finite value-valid component VJPs did not isolate one route",
    )


def _row_material(row: dict[str, Any]) -> bool:
    return bool(
        row.get("max_abs_delta") is not None
        and float(row["max_abs_delta"]) > GRADIENT_TOLERANCE
    )


def _decision(classification: str, veto_status: dict[str, Any]) -> str:
    if not veto_status.get("all_vetoes_clear", False):
        return "filterflow_float64_row_173_previous_log_weight_jacobian_blocked_or_vetoed"
    mapping = {
        "h0_filterflow_contract_matches": (
            "filterflow_float64_row_173_previous_log_weight_jacobian_filterflow_contract_matches"
        ),
        "h2_proposal_log_prob_route_differs": (
            "filterflow_float64_row_173_previous_log_weight_jacobian_proposal_log_prob_route_differs"
        ),
        "h3_transition_or_observation_route_differs": (
            "filterflow_float64_row_173_previous_log_weight_jacobian_transition_or_observation_route_differs"
        ),
        "h4_normalization_route_differs": (
            "filterflow_float64_row_173_previous_log_weight_jacobian_normalization_route_differs"
        ),
        "h5_filterflow_zero_bayesfilter_nonzero_localized_to_signed_sum": (
            "filterflow_float64_row_173_previous_log_weight_jacobian_signed_sum_differs"
        ),
        "h6_unresolved_previous_log_weight_jacobian": (
            "filterflow_float64_row_173_previous_log_weight_jacobian_unresolved"
        ),
    }
    return mapping[classification]


def _model_contract() -> dict[str, Any]:
    config = vjp.RunConfig(
        target_time_index=PREVIOUS_TIME_INDEX,
        tag=TAG,
        plan_path=PLAN_PATH,
        result_path=RESULT_PATH,
        json_path=JSON_PATH,
        report_path=REPORT_PATH,
    )
    return {
        **vjp._model_contract(config),
        "target_carry_time_index": TARGET_CARRY_TIME_INDEX,
        "previous_time_index": PREVIOUS_TIME_INDEX,
        "primary_probe": (
            "unit VJPs of time-92 normalized/post_update_log_weights and "
            "likelihood components wrt post_particles and proposed_particles"
        ),
        "prior_unit_artifact": str(PRIOR_UNIT_JSON.relative_to(REPO_ROOT)),
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": (
            "At row 173, does the BayesFilter-vs-FilterFlow previous-log-weight "
            "carry Jacobian difference originate in the time-92 component "
            "derivative of transition_ll + observation_ll - proposal_ll, "
            "proposal log-probability route, normalization route, or an unresolved edge?"
        ),
        "comparator": "local executable float64 FilterFlow reference",
        "primary_criterion": (
            "compare unit-upstream component VJPs wrt time-92 proposed_particles "
            "after value/resampling/finiteness/fingerprint/CPU vetoes clear"
        ),
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "vetoes": [
            "FilterFlow subprocess cannot execute",
            "prior unit-upstream artifact missing or invalid",
            "comparator fingerprint changes during the run",
            "CPU-only manifest violation",
            "value-path mismatch for required tensors",
            "resampling flags differ",
            "required VJP tensors missing or non-finite",
            "path-boundary contamination",
        ],
        "explanatory_only": [
            "post-particle component VJPs",
            "fresh proposal-log-prob alternative as route localization evidence",
            "total-gradient and scalar evidence from prior artifacts",
        ],
        "not_concluded": _non_implications(),
    }


def _decision_table(
    decision: str,
    classification: str,
    veto_status: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "Decision": decision,
            "Primary criterion status": classification,
            "Veto diagnostic status": json.dumps(
                {key: value for key, value in veto_status.items() if key != "cpu_rows"},
                sort_keys=True,
            ),
            "Main uncertainty": "single row and previous time; no correctness claim",
            "Next justified action": _next_action(classification),
            "Not concluded": "correctness, posterior correctness, production readiness, global agreement",
        }
    ]


def _next_action(classification: str) -> str:
    if classification == "h0_filterflow_contract_matches":
        return "keep the fresh proposal-density topology as the BayesFilter/FilterFlow comparison contract"
    if classification == "h2_proposal_log_prob_route_differs":
        return "adopt or isolate the FilterFlow fresh proposal-log-prob route in the BayesFilter comparison harness"
    if classification == "h3_transition_or_observation_route_differs":
        return "trace the transition and observation log-prob arithmetic against executable FilterFlow"
    if classification == "h4_normalization_route_differs":
        return "trace normalize clipping and stop-gradient mask against executable FilterFlow"
    if classification == "h5_filterflow_zero_bayesfilter_nonzero_localized_to_signed_sum":
        return "inspect cancellation among transition, observation, and proposal local VJPs"
    if classification == "h6_unresolved_previous_log_weight_jacobian":
        return "add narrower tensor-entry or local algebra probes for the previous carry edge"
    return "repair blocker before interpreting previous-log-weight Jacobian evidence"


def _compact_side(side: dict[str, Any]) -> dict[str, Any]:
    if side.get("status") != "executed":
        return side
    return {
        "status": side["status"],
        "backend": side["backend"],
        "mode": side.get("mode"),
        "mode_description": side.get("mode_description"),
        "settings": side["settings"],
        "target_scalar": side["target_scalar"],
        "resampling_flag": side["resampling_flag"],
        "value_summaries": {
            name: side["values"][name]
            for name in COMPONENT_FIELDS
            + INPUT_FIELDS
            + ("post_log_weights", "post_update_log_weights")
        },
        "component_unit_vjps": side["component_unit_vjps"],
        "target_unit_vjps": side["target_unit_vjps"],
        "cpu_only_manifest": side["cpu_only_manifest"],
        "stderr_excerpt": side.get("stderr_excerpt", ""),
    }


def _compact_prior(prior: dict[str, Any]) -> dict[str, Any]:
    if prior.get("status") != "loaded":
        return prior
    payload = prior["payload"]
    return {
        "status": "loaded",
        "path": prior["path"],
        "decision": payload.get("decision"),
        "hypothesis_classification": payload.get("hypothesis_classification"),
        "reproducibility_digest": payload.get("reproducibility_digest"),
        "unit_upstream_delta": payload.get("comparison", {}).get(
            "unit_upstream_delta"
        ),
        "composed_carryover_delta": payload.get("comparison", {}).get(
            "composed_carryover_delta"
        ),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "created_at_utc",
        "question",
        "decision",
        "hypothesis_classification",
        "hypothesis_reason",
        "plan_path",
        "result_path",
        "review_loop_path",
        "reference_policy",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "model_contract",
        "prior_unit_artifact",
        "filterflow_localization",
        "bayesfilter_localization_modes",
        "comparison",
        "veto_status_table",
        "path_boundary_manifest",
        "evidence_contract",
        "decision_table",
        "non_implications",
        "run_manifest",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    validate_filterflow_reference_status(
        payload["filterflow_status"],
        marker_path=vjp.FILTERFLOW_MARKER_PATH,
    )
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    allowed = {
        "filterflow_float64_row_173_previous_log_weight_jacobian_blocked_or_vetoed",
        "filterflow_float64_row_173_previous_log_weight_jacobian_filterflow_contract_matches",
        "filterflow_float64_row_173_previous_log_weight_jacobian_proposal_log_prob_route_differs",
        "filterflow_float64_row_173_previous_log_weight_jacobian_transition_or_observation_route_differs",
        "filterflow_float64_row_173_previous_log_weight_jacobian_normalization_route_differs",
        "filterflow_float64_row_173_previous_log_weight_jacobian_signed_sum_differs",
        "filterflow_float64_row_173_previous_log_weight_jacobian_unresolved",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    if payload["decision"].endswith("blocked_or_vetoed"):
        return
    if not payload["veto_status_table"]["all_vetoes_clear"]:
        raise ValueError("non-blocked decision with uncleared vetoes")
    _validate_cpu(payload["filterflow_localization"]["cpu_only_manifest"], "filterflow")
    for mode in payload["bayesfilter_localization_modes"]:
        _validate_cpu(mode["cpu_only_manifest"], f"bayesfilter_{mode.get('mode')}")
    comparison = payload["comparison"]
    if not (comparison["raw_value_gate_pass"] and comparison["fresh_value_gate_pass"]):
        raise ValueError("non-blocked decision with failed value gate")
    if not (
        comparison["raw_resampling_flags_match"]
        and comparison["fresh_resampling_flags_match"]
    ):
        raise ValueError("non-blocked decision with resampling mismatch")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Result: Row 173 Previous Log-Weight Jacobian Localization",
            "",
            "## Decision",
            "",
            f"`{payload['decision']}`",
            "",
            "## Hypothesis Classification",
            "",
            f"`{payload['hypothesis_classification']}`",
            "",
            payload["hypothesis_reason"],
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                "| {Decision} | {Primary criterion status} | {Veto diagnostic status} | {Main uncertainty} | {Next justified action} | {Not concluded} |".format(
                    **row
                )
                for row in payload["decision_table"]
            ],
            "",
            "## Comparison",
            "",
            _json_block(payload["comparison"]),
            "",
            "## Veto Status",
            "",
            _json_block(payload["veto_status_table"]),
            "",
            "## FilterFlow Localization",
            "",
            _json_block(payload["filterflow_localization"]),
            "",
            "## BayesFilter Localization Modes",
            "",
            _json_block(payload["bayesfilter_localization_modes"]),
            "",
            "## Prior Unit Artifact",
            "",
            _json_block(payload["prior_unit_artifact"]),
            "",
            "## Model Contract",
            "",
            _json_block(payload["model_contract"]),
            "",
            "## Run Manifest",
            "",
            _json_block(payload["run_manifest"]),
            "",
            "## Non-Implications",
            "",
            "\n".join(f"- {item}" for item in payload["non_implications"]),
            "",
            "## Reproducibility Digest",
            "",
            f"`{payload.get('reproducibility_digest')}`",
            "",
        ]
    )


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("reproducibility_digest", None)
    return stable_digest(copy)


def _non_implications() -> list[str]:
    return [
        "no correctness claim for FilterFlow or BayesFilter",
        "no analytic-gradient correctness claim",
        "no posterior correctness claim",
        "no global smoothness-surface agreement claim",
        "no claim that either implementation is mathematically authoritative",
        "no claim that any boundary mode is a code fix",
        "no production readiness or public API readiness",
        "no monograph, highdim, DSGE, NAWM, or banking/model-risk claim",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
