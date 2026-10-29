"""Probe row-173 proposal/update adjoint topology at the first divergent time."""

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
PROBE_TIMES = (43,)
PROBE_TIME = 43
TARGET_TIME_INDEX = 93
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-proposal-adjoint-topology-plan-2026-06-05.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-proposal-adjoint-topology-result-2026-06-05.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_proposal_adjoint_topology_2026-06-05.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-proposal-adjoint-topology-2026-06-05.md"
)
FILTERFLOW_MARKER_PATH = vjp.FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
VALUE_TOLERANCE = vjp.VALUE_TOLERANCE
GRADIENT_TOLERANCE = vjp.GRADIENT_TOLERANCE
FORWARD_TOLERANCE = VALUE_TOLERANCE

NODE_NAMES = (
    "pre_particles",
    "pre_log_weights",
    "log_ess",
    "transport_matrix",
    "post_particles",
    "post_log_weights",
    "proposal_loc",
    "proposal_mean",
    "fresh_proposal_mean",
    "proposed_particles",
    "sample_noise",
    "observation_ll",
    "transition_ll",
    "proposal_ll",
    "proposal_dist_log_prob",
    "fresh_dist_log_prob",
    "unnormalized",
    "increment",
    "normalized_log_weights",
    "post_log_likelihoods",
)

LOCAL_GRADIENT_NAMES = (
    "target_to_proposal_mean",
    "target_to_proposed_particles",
    "proposal_ll_to_proposal_mean",
    "proposal_ll_to_proposed_particles",
    "proposal_dist_log_prob_to_proposal_mean",
    "proposal_dist_log_prob_to_proposed_particles",
    "fresh_dist_log_prob_to_fresh_proposal_mean",
    "fresh_dist_log_prob_to_proposed_particles",
    "transition_ll_to_proposed_particles",
    "observation_ll_to_proposed_particles",
    "sample_to_proposal_mean_target_upstream",
    "sample_sum_to_proposal_mean",
    "sample_noise_sum_to_proposal_mean",
)

LOG_PROB_GRADIENT_NAMES = (
    "proposal_ll_to_proposal_mean",
    "proposal_ll_to_proposed_particles",
    "proposal_dist_log_prob_to_proposal_mean",
    "proposal_dist_log_prob_to_proposed_particles",
    "fresh_dist_log_prob_to_fresh_proposal_mean",
    "fresh_dist_log_prob_to_proposed_particles",
    "transition_ll_to_proposed_particles",
    "observation_ll_to_proposed_particles",
)

SAMPLE_GRADIENT_NAMES = (
    "sample_sum_to_proposal_mean",
    "sample_noise_sum_to_proposal_mean",
)

TARGET_ADJOINT_GRADIENT_NAMES = (
    "target_to_proposal_mean",
    "target_to_proposed_particles",
    "sample_to_proposal_mean_target_upstream",
)

STOP_GRADIENT_MODES = (
    "raw",
    "stop_proposed_particles",
    "stop_proposal_mean",
    "stop_proposal_ll",
    "stop_transition_ll",
    "stop_observation_ll",
    "stop_proposal_sample_noise",
)


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
    filterflow = _filterflow_source_subprocess()
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_proposal_adjoint_topology_filterflow_blocker",
            filterflow.get("blocker", "unknown FilterFlow blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            None,
        )
    bayesfilter = _bayesfilter_source(filterflow)
    if bayesfilter.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_proposal_adjoint_topology_bayesfilter_blocker",
            bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            bayesfilter,
        )

    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    comparison = _compare_sources(filterflow, bayesfilter)
    stop_gradient_ablation = _bayesfilter_stop_gradient_ablation(filterflow)
    ablation_comparison = _compare_stop_gradient_ablation(filterflow, stop_gradient_ablation)
    veto_status = _veto_status(comparison, comparator_drift, stop_gradient_ablation)
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
        "question": "row_173_time_43_proposal_update_adjoint_topology_difference_audit",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "source_comparison": comparison,
        "stop_gradient_ablation": stop_gradient_ablation,
        "stop_gradient_ablation_comparison": ablation_comparison,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_proposal_adjoint_topology_probe_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "data_seed": vjp.DATA_SEED,
            "filter_seed": vjp.FILTER_SEED,
            "probe_times": list(PROBE_TIMES),
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
    return {
        "decision": decision,
        "hypothesis_classification": "blocked_or_vetoed",
        "hypothesis_reason": blocker,
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
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "source_comparison": {"status": "blocked", "blocker": blocker},
        "veto_status_table": {"status": "blocked", "blocker": blocker},
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_proposal_adjoint_topology_probe_tf"
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


def _filterflow_source_subprocess() -> dict[str, Any]:
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
        [str(vjp.FILTERFLOW_ENV_PYTHON), "-c", _filterflow_source_script()],
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
            "blocker": "filterflow source subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_SOURCE_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_SOURCE_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow source JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(completed.stdout[start + len("FILTERFLOW_SOURCE_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_source_script() -> str:
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
        TARGET_TIME_INDEX = {TARGET_TIME_INDEX}
        PROBE_TIMES = set({list(PROBE_TIMES)!r})
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

        def summary(tensor):
            cast = tf.cast(tensor, DTYPE)
            return {{
                "shape": [int(dim) for dim in cast.shape],
                "max_abs": scalar(tf.reduce_max(tf.abs(cast))),
                "sum": scalar(tf.reduce_sum(cast)),
                "finite": bool(tf.reduce_all(tf.math.is_finite(cast)).numpy()),
            }}

        def grad_summary(tape, target, tensor):
            gradient = tape.gradient(target, tensor)
            if gradient is None:
                gradient = tf.zeros_like(tensor)
            return {{
                "summary": summary(gradient),
                "value": to_json(gradient),
            }}

        def safe_gradient(tape, target, tensor, output_gradients=None):
            gradient = tape.gradient(target, tensor, output_gradients=output_gradients)
            if gradient is None:
                gradient = tf.zeros_like(tensor)
            return gradient

        def local_gradient_payload(tape, target, nodes):
            proposed_upstream = safe_gradient(tape, target, nodes["proposed_particles"])
            gradients = {{
                "target_to_proposal_mean": safe_gradient(tape, target, nodes["proposal_mean"]),
                "target_to_proposed_particles": proposed_upstream,
                "proposal_ll_to_proposal_mean": safe_gradient(
                    tape,
                    nodes["proposal_ll"],
                    nodes["proposal_mean"],
                ),
                "proposal_ll_to_proposed_particles": safe_gradient(
                    tape,
                    nodes["proposal_ll"],
                    nodes["proposed_particles"],
                ),
                "proposal_dist_log_prob_to_proposal_mean": safe_gradient(
                    tape,
                    nodes["proposal_dist_log_prob"],
                    nodes["proposal_mean"],
                ),
                "proposal_dist_log_prob_to_proposed_particles": safe_gradient(
                    tape,
                    nodes["proposal_dist_log_prob"],
                    nodes["proposed_particles"],
                ),
                "fresh_dist_log_prob_to_fresh_proposal_mean": safe_gradient(
                    tape,
                    nodes["fresh_dist_log_prob"],
                    nodes["fresh_proposal_mean"],
                ),
                "fresh_dist_log_prob_to_proposed_particles": safe_gradient(
                    tape,
                    nodes["fresh_dist_log_prob"],
                    nodes["proposed_particles"],
                ),
                "transition_ll_to_proposed_particles": safe_gradient(
                    tape,
                    nodes["transition_ll"],
                    nodes["proposed_particles"],
                ),
                "observation_ll_to_proposed_particles": safe_gradient(
                    tape,
                    nodes["observation_ll"],
                    nodes["proposed_particles"],
                ),
                "sample_to_proposal_mean_target_upstream": safe_gradient(
                    tape,
                    nodes["proposed_particles"],
                    nodes["proposal_mean"],
                    output_gradients=tf.stop_gradient(proposed_upstream),
                ),
                "sample_sum_to_proposal_mean": safe_gradient(
                    tape,
                    tf.reduce_sum(nodes["proposed_particles"]),
                    nodes["proposal_mean"],
                ),
                "sample_noise_sum_to_proposal_mean": safe_gradient(
                    tape,
                    tf.reduce_sum(nodes["sample_noise"]),
                    nodes["proposal_mean"],
                ),
            }}
            return {{
                name: {{"summary": summary(tensor), "value": to_json(tensor)}}
                for name, tensor in gradients.items()
            }}

        def distribution_loc(distribution):
            if hasattr(distribution, "loc"):
                return distribution.loc
            parameters = getattr(distribution, "parameters", {{}})
            if "loc" in parameters:
                return parameters["loc"]
            return distribution.mean()

        def matrix_residuals(matrix, log_weights):
            row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(matrix, axis=2) - 1.0))
            column_mass = tf.reduce_sum(matrix, axis=1)
            column_target = tf.exp(log_weights) * tf.cast(N, DTYPE)
            column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
            return {{
                "row": scalar(row_residual),
                "column": scalar(column_residual),
            }}

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
            [[0.0, 1.0], [0.0, 0.0]], dtype=DTYPE
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
        probe_records = []
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
                resampled_state = apply_transport_matrix(state_with_ess, transport_matrix, flags)
                proposal_dist = smc._proposal_model._get_proposal_dist(resampled_state, observation)
                proposal_loc = distribution_loc(proposal_dist)
                proposal_mean = proposal_dist.mean()
                proposed_particles = proposal_dist.sample(seed=seed2)
                sample_noise = proposed_particles - proposal_mean
                proposal_dist_log_prob = proposal_dist.log_prob(proposed_particles)
                proposed_state = attr.evolve(resampled_state, particles=proposed_particles)
                observation_ll = smc._observation_model.loglikelihood(proposed_state, observation)
                transition_ll = smc._transition_model.loglikelihood(
                    resampled_state,
                    proposed_state,
                    inputs,
                )
                fresh_proposal_dist = smc._proposal_model._get_proposal_dist(
                    resampled_state,
                    observation,
                )
                fresh_proposal_mean = distribution_loc(fresh_proposal_dist)
                fresh_dist_log_prob = fresh_proposal_dist.log_prob(proposed_particles)
                proposal_ll = smc._proposal_model.loglikelihood(
                    proposed_state,
                    resampled_state,
                    inputs,
                    observation,
                )
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
                if time_index in PROBE_TIMES:
                    raw_transport_upstream = None
                    probe_records.append({{
                        "time_index": int(time_index),
                        "flags": flags,
                        "nodes": {{
                            "pre_particles": state_with_ess.particles,
                            "pre_log_weights": state_with_ess.log_weights,
                            "log_ess": tf.math.log(ess),
                            "transport_matrix": transport_matrix,
                            "post_particles": resampled_state.particles,
                            "post_log_weights": resampled_state.log_weights,
                            "proposal_loc": proposal_loc,
                            "proposal_mean": proposal_mean,
                            "fresh_proposal_mean": fresh_proposal_mean,
                            "proposed_particles": proposed_particles,
                            "sample_noise": sample_noise,
                            "observation_ll": observation_ll,
                            "transition_ll": transition_ll,
                            "proposal_ll": proposal_ll,
                            "proposal_dist_log_prob": proposal_dist_log_prob,
                            "fresh_dist_log_prob": fresh_dist_log_prob,
                            "unnormalized": unnormalized,
                            "increment": increment,
                            "normalized_log_weights": normalized,
                            "post_log_likelihoods": log_likelihoods,
                        }},
                        "transport_residuals": matrix_residuals(
                            transport_matrix,
                            state_with_ess.log_weights,
                        ),
                    }})
                state = attr.evolve(updated_state, t=t + 1)
                if time_index == TARGET_TIME_INDEX:
                    target = tf.reduce_mean(log_likelihoods)
                    break
            else:
                raise RuntimeError("target time not reached")

        total_gradient = tape.gradient(target, modifiable_transition_matrix)
        if total_gradient is None:
            total_gradient = tf.zeros_like(modifiable_transition_matrix)
        rows = []
        for record in probe_records:
            nodes = record["nodes"]
            node_payload = {{}}
            for name, tensor in nodes.items():
                node_payload[name] = {{
                    "value": to_json(tensor),
                    "value_summary": summary(tensor),
                    "adjoint": grad_summary(tape, target, tensor),
                }}
            raw_upstream = tape.gradient(target, nodes["transport_matrix"])
            if raw_upstream is None:
                raw_upstream = tf.zeros_like(nodes["transport_matrix"])
            clipped_upstream = tf.clip_by_value(raw_upstream, -1.0, 1.0)
            rows.append({{
                "time_index": record["time_index"],
                "resampling_flag": [bool(v) for v in tf.reshape(record["flags"], [-1]).numpy().tolist()],
                "nodes": node_payload,
                "transport_residuals": record["transport_residuals"],
                "raw_transport_upstream": to_json(raw_upstream),
                "clipped_transport_upstream": to_json(clipped_upstream),
                "clip_mask": to_json(tf.cast(tf.abs(raw_upstream) > 1.0, DTYPE)),
                "raw_transport_upstream_summary": summary(raw_upstream),
                "clipped_transport_upstream_summary": summary(clipped_upstream),
                "clip_count": scalar(tf.reduce_sum(tf.cast(tf.abs(raw_upstream) > 1.0, DTYPE))),
                "local_gradients": local_gradient_payload(tape, target, nodes),
            }})
        del tape

        payload = {{
            "status": "executed",
            "backend": "executable_filterflow_subprocess",
            "settings": {{
                "mesh_index": {vjp.MESH_INDEX},
                "target_time_index": TARGET_TIME_INDEX,
                "probe_times": sorted(PROBE_TIMES),
                "theta": THETA.astype(float).tolist(),
                "T": T,
                "n_particles": N,
                "dtype": "float64",
            }},
            "target_scalar": scalar(target),
            "total_gradient_diag": to_json(tf.linalg.diag_part(total_gradient)),
            "probe_rows": rows,
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
        print("FILTERFLOW_SOURCE_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_SOURCE_JSON_END")
        """
    )


def _bayesfilter_source(filterflow: dict[str, Any]) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        theta_variable = tf.Variable(vjp.THETA, dtype=DTYPE)
        model = _model_from_filterflow(filterflow)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta_variable)
            bundle = _bayesfilter_source_bundle(theta_variable, model)
            target = bundle["target"]
        total_gradient = _safe_gradient(tape, target, theta_variable)
        rows = []
        for record in bundle["probe_records"]:
            nodes = record["nodes"]
            node_payload = {}
            for name, tensor in nodes.items():
                node_payload[name] = {
                    "value": r3._json(tensor),
                    "value_summary": _summary_tensor(tensor),
                    "adjoint": _adjoint_payload(tape, target, tensor),
                }
            raw_upstream = _safe_gradient(tape, target, nodes["transport_matrix"])
            clipped_upstream = tf.clip_by_value(raw_upstream, -1.0, 1.0)
            rows.append(
                {
                    "time_index": int(record["time_index"]),
                    "resampling_flag": [
                        bool(v) for v in tf.reshape(record["flags"], [-1]).numpy().tolist()
                    ],
                    "nodes": node_payload,
                    "transport_residuals": record["transport_residuals"],
                    "raw_transport_upstream": r3._json(raw_upstream),
                    "clipped_transport_upstream": r3._json(clipped_upstream),
                    "clip_mask": r3._json(tf.cast(tf.abs(raw_upstream) > 1.0, DTYPE)),
                    "raw_transport_upstream_summary": _summary_tensor(raw_upstream),
                    "clipped_transport_upstream_summary": _summary_tensor(clipped_upstream),
                    "clip_count": _float(
                        tf.reduce_sum(tf.cast(tf.abs(raw_upstream) > 1.0, DTYPE))
                    ),
                    "local_gradients": _local_gradient_payload(tape, target, nodes),
                }
            )
        del tape
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "settings": filterflow["settings"],
            "target_scalar": _float(target),
            "total_gradient_diag": r3._json(total_gradient),
            "probe_rows": rows,
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_source_bundle(
    theta: tf.Tensor,
    model: dict[str, Any],
    *,
    stop_gradient_mode: str = "raw",
) -> dict[str, Any]:
    if stop_gradient_mode not in STOP_GRADIENT_MODES:
        raise ValueError(f"unknown stop_gradient_mode: {stop_gradient_mode}")
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
    probe_records = []

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
        transport_matrix = tf.cast(transported.transport_matrix, DTYPE)
        post_particles = transported.particles
        post_log_weights = transported.log_weights
        proposal_mean = _optimal_proposal_mean(
            post_particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
        )
        if stop_gradient_mode == "stop_proposal_mean" and time_index == PROBE_TIME:
            proposal_mean = tf.stop_gradient(proposal_mean)
        proposal_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
        proposal_loc = proposal_dist.loc
        proposed_particles = proposal_dist.sample(seed=seed2)
        if stop_gradient_mode == "stop_proposal_sample_noise" and time_index == PROBE_TIME:
            proposed_particles = proposal_mean + tf.stop_gradient(proposed_particles - proposal_mean)
        if stop_gradient_mode == "stop_proposed_particles" and time_index == PROBE_TIME:
            proposed_particles = tf.stop_gradient(proposed_particles)
        sample_noise = proposed_particles - proposal_mean
        proposal_dist_log_prob = proposal_dist.log_prob(proposed_particles)
        observation_ll = _observation_log_prob(
            proposed_particles,
            observation,
            observation_matrix,
            observation_noise,
        )
        transition_ll = _transition_log_prob(
            post_particles,
            proposed_particles,
            transition_matrix,
            transition_noise,
        )
        fresh_proposal_mean = _optimal_proposal_mean(
            post_particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
        )
        fresh_proposal_dist = tfd.MultivariateNormalTriL(fresh_proposal_mean, sigma_chol)
        fresh_dist_log_prob = fresh_proposal_dist.log_prob(proposed_particles)
        proposal_ll = proposal_dist.log_prob(proposed_particles)
        if stop_gradient_mode == "stop_observation_ll" and time_index == PROBE_TIME:
            observation_ll = tf.stop_gradient(observation_ll)
        if stop_gradient_mode == "stop_transition_ll" and time_index == PROBE_TIME:
            transition_ll = tf.stop_gradient(transition_ll)
        if stop_gradient_mode == "stop_proposal_ll" and time_index == PROBE_TIME:
            proposal_ll = tf.stop_gradient(proposal_ll)
        unnormalized = transition_ll + observation_ll - proposal_ll + post_log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        next_log_likelihoods = log_likelihoods + increment
        normalized = r3._filterflow_normalize(unnormalized, num_particles)
        if time_index in PROBE_TIMES:
            probe_records.append(
                {
                    "time_index": time_index,
                    "flags": flags,
                    "nodes": {
                        "pre_particles": particles,
                        "pre_log_weights": log_weights,
                        "log_ess": ess_log,
                        "transport_matrix": transport_matrix,
                        "post_particles": post_particles,
                        "post_log_weights": post_log_weights,
                        "proposal_loc": proposal_loc,
                        "proposal_mean": proposal_mean,
                        "fresh_proposal_mean": fresh_proposal_mean,
                        "proposed_particles": proposed_particles,
                        "sample_noise": sample_noise,
                        "observation_ll": observation_ll,
                        "transition_ll": transition_ll,
                        "proposal_ll": proposal_ll,
                        "proposal_dist_log_prob": proposal_dist_log_prob,
                        "fresh_dist_log_prob": fresh_dist_log_prob,
                        "unnormalized": unnormalized,
                        "increment": increment,
                        "normalized_log_weights": normalized,
                        "post_log_likelihoods": next_log_likelihoods,
                    },
                    "transport_residuals": _matrix_residuals(transport_matrix, log_weights),
                }
            )
        log_likelihoods = next_log_likelihoods
        log_weights = normalized
        particles = proposed_particles
        if time_index == TARGET_TIME_INDEX:
            return {"target": tf.reduce_mean(log_likelihoods), "probe_records": probe_records}
    raise RuntimeError("target time not reached")


def _compare_sources(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    ff_rows = {int(row["time_index"]): row for row in filterflow["probe_rows"]}
    bf_rows = {int(row["time_index"]): row for row in bayesfilter["probe_rows"]}
    if set(ff_rows) != set(PROBE_TIMES) or set(bf_rows) != set(PROBE_TIMES):
        return {
            "status": "blocked",
            "blocker": "probe time rows missing",
            "filterflow_times": sorted(ff_rows),
            "bayesfilter_times": sorted(bf_rows),
        }
    rows = []
    for time_index in PROBE_TIMES:
        ff_row = ff_rows[time_index]
        bf_row = bf_rows[time_index]
        node_rows = []
        for name in NODE_NAMES:
            bf_node = bf_row["nodes"][name]
            ff_node = ff_row["nodes"][name]
            forward_delta = _tensor_delta_summary(
                bf_node["value"],
                ff_node["value"],
            )
            adjoint_delta = _tensor_delta_summary(
                bf_node["adjoint"]["value"],
                ff_node["adjoint"]["value"],
            )
            node_rows.append(
                {
                    "node": name,
                    "forward_max_abs_delta": forward_delta["max_abs_delta"],
                    "forward_sum_delta": forward_delta["sum_delta"],
                    "forward_finite": forward_delta["finite"],
                    "adjoint_max_abs_delta": adjoint_delta["max_abs_delta"],
                    "adjoint_sum_delta": adjoint_delta["sum_delta"],
                    "adjoint_finite": adjoint_delta["finite"],
                }
            )
        raw_upstream_delta = _tensor_delta_summary(
            bf_row["raw_transport_upstream"],
            ff_row["raw_transport_upstream"],
        )
        clipped_upstream_delta = _tensor_delta_summary(
            bf_row["clipped_transport_upstream"],
            ff_row["clipped_transport_upstream"],
        )
        local_gradient_rows = []
        for name in LOCAL_GRADIENT_NAMES:
            bf_gradient = bf_row["local_gradients"][name]
            ff_gradient = ff_row["local_gradients"][name]
            gradient_delta = _tensor_delta_summary(
                bf_gradient["value"],
                ff_gradient["value"],
            )
            local_gradient_rows.append(
                {
                    "name": name,
                    "max_abs_delta": gradient_delta["max_abs_delta"],
                    "sum_delta": gradient_delta["sum_delta"],
                    "finite": gradient_delta["finite"],
                    "filterflow_max_abs": ff_gradient["summary"]["max_abs"],
                    "bayesfilter_max_abs": bf_gradient["summary"]["max_abs"],
                }
            )
        rows.append(
            {
                "time_index": time_index,
                "resampling_flags_match": ff_row["resampling_flag"] == bf_row["resampling_flag"],
                "filterflow_resampling_flag": ff_row["resampling_flag"],
                "bayesfilter_resampling_flag": bf_row["resampling_flag"],
                "node_rows": node_rows,
                "first_forward_delta_node": _first_delta_node(
                    node_rows,
                    "forward_max_abs_delta",
                    FORWARD_TOLERANCE,
                ),
                "first_adjoint_delta_node": _first_delta_node(
                    node_rows,
                    "adjoint_max_abs_delta",
                    GRADIENT_TOLERANCE,
                ),
                "max_adjoint_delta_node": max(
                    node_rows,
                    key=lambda row: row["adjoint_max_abs_delta"],
                ),
                "max_forward_delta_node": max(
                    node_rows,
                    key=lambda row: row["forward_max_abs_delta"],
                ),
                "max_forward_delta": max(row["forward_max_abs_delta"] for row in node_rows),
                "max_adjoint_delta": max(row["adjoint_max_abs_delta"] for row in node_rows),
                "raw_transport_upstream_delta": raw_upstream_delta,
                "clipped_transport_upstream_delta": clipped_upstream_delta,
                "clip_mask_mismatch_count": _mask_mismatch_count(
                    bf_row["clip_mask"],
                    ff_row["clip_mask"],
                ),
                "clip_count_delta": float(bf_row["clip_count"]) - float(ff_row["clip_count"]),
                "filterflow_transport_residuals": ff_row["transport_residuals"],
                "bayesfilter_transport_residuals": bf_row["transport_residuals"],
                "local_gradient_rows": local_gradient_rows,
                "first_local_gradient_delta": _first_delta_named(
                    local_gradient_rows,
                    GRADIENT_TOLERANCE,
                ),
                "first_log_prob_gradient_delta": _first_delta_named(
                    [
                        row
                        for row in local_gradient_rows
                        if row["name"] in LOG_PROB_GRADIENT_NAMES
                    ],
                    GRADIENT_TOLERANCE,
                ),
                "first_sample_gradient_delta": _first_delta_named(
                    [
                        row
                        for row in local_gradient_rows
                        if row["name"] in SAMPLE_GRADIENT_NAMES
                    ],
                    GRADIENT_TOLERANCE,
                ),
                "first_target_adjoint_gradient_delta": _first_delta_named(
                    [
                        row
                        for row in local_gradient_rows
                        if row["name"] in TARGET_ADJOINT_GRADIENT_NAMES
                    ],
                    GRADIENT_TOLERANCE,
                ),
                "max_local_gradient_delta": max(
                    local_gradient_rows,
                    key=lambda row: row["max_abs_delta"],
                ),
            }
        )
    full_gradient_delta = _vector_sub(
        bayesfilter["total_gradient_diag"],
        filterflow["total_gradient_diag"],
    )
    return {
        "status": "compared",
        "target_scalar_delta": abs(float(bayesfilter["target_scalar"]) - float(filterflow["target_scalar"])),
        "total_gradient_delta": full_gradient_delta,
        "max_abs_total_gradient_delta": _max_abs(full_gradient_delta),
        "rows": rows,
        "all_resampling_flags_match": all(row["resampling_flags_match"] for row in rows),
        "all_forward_finite": all(
            node["forward_finite"] for row in rows for node in row["node_rows"]
        ),
        "all_adjoints_finite": all(
            node["adjoint_finite"] for row in rows for node in row["node_rows"]
        ),
        "all_local_gradients_finite": all(
            local["finite"] for row in rows for local in row["local_gradient_rows"]
        ),
        "interpretation": _interpret_source(rows),
    }


def _interpret_source(rows: list[dict[str, Any]]) -> str:
    first_time = rows[0]
    if first_time["max_forward_delta"] > FORWARD_TOLERANCE:
        return "forward_boundary_drift"
    if _proposal_likelihood_wiring_delta(first_time):
        return "proposal_likelihood_wiring_topology"
    if first_time["first_sample_gradient_delta"] is not None:
        return "proposal_sample_gradient_contract"
    if first_time["first_log_prob_gradient_delta"] is not None:
        return "log_prob_gradient_contract"
    if first_time["first_adjoint_delta_node"] is not None:
        node = first_time["first_adjoint_delta_node"]["node"]
        if node in {
            "proposal_mean",
            "proposed_particles",
            "observation_ll",
            "transition_ll",
            "proposal_ll",
            "unnormalized",
            "increment",
            "normalized_log_weights",
            "post_log_likelihoods",
        }:
            return "proposal_update_adjoint_source"
        return "downstream_adjoint_topology_mismatch"
    if first_time["raw_transport_upstream_delta"]["max_abs_delta"] > GRADIENT_TOLERANCE:
        return "transport_boundary_only"
    return "proposal_update_local_match_downstream_topology"


def _proposal_likelihood_wiring_delta(row: dict[str, Any]) -> bool:
    local_rows = {item["name"]: item for item in row["local_gradient_rows"]}
    official = local_rows.get("proposal_ll_to_proposed_particles")
    direct = local_rows.get("proposal_dist_log_prob_to_proposed_particles")
    fresh = local_rows.get("fresh_dist_log_prob_to_proposed_particles")
    if official is None or direct is None or fresh is None:
        return False
    return (
        official["max_abs_delta"] > GRADIENT_TOLERANCE
        and direct["max_abs_delta"] <= GRADIENT_TOLERANCE
        and fresh["max_abs_delta"] <= GRADIENT_TOLERANCE
    )


def _classify(comparison: dict[str, Any], veto_status: dict[str, Any]) -> dict[str, str]:
    if not veto_status.get("all_vetoes_clear", False):
        return {"classification": "blocked_or_vetoed", "reason": str(veto_status)}
    mapping = {
        "forward_boundary_drift": (
            "h1_forward_contract_drift",
            "forward values at or before the transport boundary differ above tolerance",
        ),
        "downstream_adjoint_topology_mismatch": (
            "h4_downstream_update_topology",
            "forward boundary values match but downstream adjoints differ",
        ),
        "proposal_sample_gradient_contract": (
            "h2_proposal_sample_gradient_contract",
            "forward tensors match but proposal sample local VJP differs",
        ),
        "log_prob_gradient_contract": (
            "h3_log_prob_gradient_contract",
            "forward tensors match but proposal/log-prob local gradients differ",
        ),
        "proposal_likelihood_wiring_topology": (
            "h4_downstream_update_topology",
            "forward tensors and direct distribution log-prob gradients match, but the official proposal-likelihood wiring into proposed particles differs",
        ),
        "local_update_gradient_contract": (
            "h3_log_prob_gradient_contract",
            "forward tensors match but transition/observation/update local gradients differ",
        ),
        "proposal_update_adjoint_source": (
            "h4_downstream_update_topology",
            "first large adjoint difference is localized to proposal/update nodes",
        ),
        "transport_boundary_only": (
            "h4_downstream_update_topology",
            "transport upstream differs without a recorded downstream-node explanation",
        ),
        "proposal_update_local_match_downstream_topology": (
            "h4_downstream_update_topology",
            "local proposal/update gradients match, but downstream target adjoints differ",
        ),
    }
    classification, reason = mapping.get(
        comparison.get("interpretation"),
        ("blocked_or_vetoed", f"unclassified interpretation {comparison.get('interpretation')}"),
    )
    return {"classification": classification, "reason": reason}


def _veto_status(
    comparison: dict[str, Any],
    comparator_drift: bool,
    stop_gradient_ablation: dict[str, Any],
) -> dict[str, Any]:
    path_boundary = continuation._path_boundary_manifest()
    return {
        "all_vetoes_clear": (
            comparison.get("status") == "compared"
            and not comparator_drift
            and not any(bool(value) for value in path_boundary.values())
            and comparison["target_scalar_delta"] <= VALUE_TOLERANCE
            and comparison["all_resampling_flags_match"]
            and comparison["all_forward_finite"]
            and comparison["all_adjoints_finite"]
            and comparison["all_local_gradients_finite"]
            and stop_gradient_ablation.get("status") == "executed"
            and PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1"
        ),
        "comparator_drift": comparator_drift,
        "path_boundary_clean": not any(bool(value) for value in path_boundary.values()),
        "scalar_value_gate_pass": comparison.get("target_scalar_delta", float("inf")) <= VALUE_TOLERANCE,
        "all_resampling_flags_match": comparison.get("all_resampling_flags_match", False),
        "all_forward_finite": comparison.get("all_forward_finite", False),
        "all_adjoints_finite": comparison.get("all_adjoints_finite", False),
        "all_local_gradients_finite": comparison.get("all_local_gradients_finite", False),
        "stop_gradient_ablation_executed": stop_gradient_ablation.get("status") == "executed",
        "cpu_only_parent": PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1",
    }


def _decision(classification: dict[str, str], veto_status: dict[str, Any]) -> str:
    if not veto_status.get("all_vetoes_clear", False):
        return "filterflow_float64_row_173_proposal_adjoint_topology_blocked_or_vetoed"
    mapping = {
        "h1_forward_contract_drift": (
            "filterflow_float64_row_173_proposal_adjoint_topology_h1_forward_contract_drift"
        ),
        "h2_proposal_sample_gradient_contract": (
            "filterflow_float64_row_173_proposal_adjoint_topology_h2_proposal_sample_gradient_contract"
        ),
        "h3_log_prob_gradient_contract": (
            "filterflow_float64_row_173_proposal_adjoint_topology_h3_log_prob_gradient_contract"
        ),
        "h4_downstream_update_topology": (
            "filterflow_float64_row_173_proposal_adjoint_topology_h4_downstream_update_topology"
        ),
    }
    return mapping.get(
        classification["classification"],
        "filterflow_float64_row_173_proposal_adjoint_topology_blocked_or_vetoed",
    )


def _compact_side(side: dict[str, Any] | None) -> dict[str, Any] | None:
    if side is None or side.get("status") != "executed":
        return side
    return {
        "status": side["status"],
        "backend": side["backend"],
        "settings": side["settings"],
        "target_scalar": side["target_scalar"],
        "total_gradient_diag": side["total_gradient_diag"],
        "probe_times": [row["time_index"] for row in side["probe_rows"]],
        "local_gradient_summary": [
            {
                "time_index": row["time_index"],
                "local_gradients": {
                    name: gradient["summary"]
                    for name, gradient in row.get("local_gradients", {}).items()
                },
            }
            for row in side["probe_rows"]
        ],
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


def _matrix_residuals(matrix: tf.Tensor, log_weights: tf.Tensor) -> dict[str, float]:
    row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(matrix, axis=2) - 1.0))
    column_mass = tf.reduce_sum(matrix, axis=1)
    column_target = tf.exp(log_weights) * tf.cast(vjp.NUM_PARTICLES, DTYPE)
    column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    return {"row": _float(row_residual), "column": _float(column_residual)}


def _safe_gradient(tape: tf.GradientTape, target: tf.Tensor, tensor: tf.Tensor) -> tf.Tensor:
    gradient = tape.gradient(target, tensor)
    return tf.zeros_like(tensor) if gradient is None else gradient


def _adjoint_payload(tape: tf.GradientTape, target: tf.Tensor, tensor: tf.Tensor) -> dict[str, Any]:
    gradient = _safe_gradient(tape, target, tensor)
    return {"summary": _summary_tensor(gradient), "value": r3._json(gradient)}


def _local_gradient_payload(
    tape: tf.GradientTape,
    target: tf.Tensor,
    nodes: dict[str, tf.Tensor],
) -> dict[str, Any]:
    proposed_upstream = _safe_gradient(tape, target, nodes["proposed_particles"])
    gradients = {
        "target_to_proposal_mean": _safe_gradient(tape, target, nodes["proposal_mean"]),
        "target_to_proposed_particles": proposed_upstream,
        "proposal_ll_to_proposal_mean": _safe_gradient(
            tape,
            nodes["proposal_ll"],
            nodes["proposal_mean"],
        ),
        "proposal_ll_to_proposed_particles": _safe_gradient(
            tape,
            nodes["proposal_ll"],
            nodes["proposed_particles"],
        ),
        "proposal_dist_log_prob_to_proposal_mean": _safe_gradient(
            tape,
            nodes["proposal_dist_log_prob"],
            nodes["proposal_mean"],
        ),
        "proposal_dist_log_prob_to_proposed_particles": _safe_gradient(
            tape,
            nodes["proposal_dist_log_prob"],
            nodes["proposed_particles"],
        ),
        "fresh_dist_log_prob_to_fresh_proposal_mean": _safe_gradient(
            tape,
            nodes["fresh_dist_log_prob"],
            nodes["fresh_proposal_mean"],
        ),
        "fresh_dist_log_prob_to_proposed_particles": _safe_gradient(
            tape,
            nodes["fresh_dist_log_prob"],
            nodes["proposed_particles"],
        ),
        "transition_ll_to_proposed_particles": _safe_gradient(
            tape,
            nodes["transition_ll"],
            nodes["proposed_particles"],
        ),
        "observation_ll_to_proposed_particles": _safe_gradient(
            tape,
            nodes["observation_ll"],
            nodes["proposed_particles"],
        ),
        "sample_to_proposal_mean_target_upstream": _safe_gradient_with_upstream(
            tape,
            nodes["proposed_particles"],
            nodes["proposal_mean"],
            tf.stop_gradient(proposed_upstream),
        ),
        "sample_sum_to_proposal_mean": _safe_gradient(
            tape,
            tf.reduce_sum(nodes["proposed_particles"]),
            nodes["proposal_mean"],
        ),
        "sample_noise_sum_to_proposal_mean": _safe_gradient(
            tape,
            tf.reduce_sum(nodes["sample_noise"]),
            nodes["proposal_mean"],
        ),
    }
    return {
        name: {"summary": _summary_tensor(tensor), "value": r3._json(tensor)}
        for name, tensor in gradients.items()
    }


def _summary_tensor(tensor: tf.Tensor) -> dict[str, Any]:
    cast = tf.cast(tensor, DTYPE)
    return {
        "shape": [int(dim) for dim in cast.shape],
        "max_abs": _float(tf.reduce_max(tf.abs(cast))),
        "sum": _float(tf.reduce_sum(cast)),
        "finite": bool(tf.reduce_all(tf.math.is_finite(cast)).numpy()),
    }


def _tensor_delta_summary(left: Any, right: Any) -> dict[str, Any]:
    left_tensor = tf.constant(left, dtype=DTYPE)
    right_tensor = tf.constant(right, dtype=DTYPE)
    delta = left_tensor - right_tensor
    return {
        "max_abs_delta": _float(tf.reduce_max(tf.abs(delta))),
        "sum_delta": _float(tf.reduce_sum(delta)),
        "finite": bool(tf.reduce_all(tf.math.is_finite(delta)).numpy()),
    }


def _safe_gradient_with_upstream(
    tape: tf.GradientTape,
    target: tf.Tensor,
    tensor: tf.Tensor,
    upstream: tf.Tensor,
) -> tf.Tensor:
    gradient = tape.gradient(target, tensor, output_gradients=upstream)
    return tf.zeros_like(tensor) if gradient is None else gradient


def _mask_mismatch_count(left: Any, right: Any) -> int:
    left_mask = tf.not_equal(tf.constant(left, dtype=DTYPE), tf.constant(0.0, dtype=DTYPE))
    right_mask = tf.not_equal(tf.constant(right, dtype=DTYPE), tf.constant(0.0, dtype=DTYPE))
    return int(tf.reduce_sum(tf.cast(tf.not_equal(left_mask, right_mask), tf.int32)).numpy())


def _first_delta_node(
    rows: list[dict[str, Any]],
    field: str,
    tolerance: float,
) -> dict[str, Any] | None:
    for row in rows:
        if abs(float(row[field])) > tolerance:
            return {"node": row["node"], field: row[field]}
    return None


def _first_delta_named(
    rows: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any] | None:
    for row in rows:
        if not row.get("finite", False) or abs(float(row["max_abs_delta"])) > tolerance:
            return {
                "name": row["name"],
                "max_abs_delta": row["max_abs_delta"],
                "tolerance": tolerance,
            }
    return None


def _bayesfilter_stop_gradient_ablation(filterflow: dict[str, Any]) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        model = _model_from_filterflow(filterflow)
        rows = []
        for mode in STOP_GRADIENT_MODES:
            theta_variable = tf.Variable(vjp.THETA, dtype=DTYPE)
            with tf.GradientTape() as tape:
                tape.watch(theta_variable)
                bundle = _bayesfilter_source_bundle(
                    theta_variable,
                    model,
                    stop_gradient_mode=mode,
                )
                target = bundle["target"]
            gradient = _safe_gradient(tape, target, theta_variable)
            del tape
            gradient_diag = r3._json(gradient)
            scalar_delta = abs(_float(target) - float(filterflow["target_scalar"]))
            gradient_delta = _vector_sub(gradient_diag, filterflow["total_gradient_diag"])
            rows.append(
                {
                    "mode": mode,
                    "description": _stop_gradient_mode_description(mode),
                    "target_scalar": _float(target),
                    "scalar_delta_to_filterflow": scalar_delta,
                    "scalar_within_tolerance": scalar_delta <= VALUE_TOLERANCE,
                    "total_gradient_diag": gradient_diag,
                    "gradient_delta_to_filterflow": gradient_delta,
                    "max_abs_gradient_delta_to_filterflow": _max_abs(gradient_delta),
                    "finite_scalar": bool(tf.math.is_finite(target).numpy()),
                    "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()),
                }
            )
        return {
            "status": "executed",
            "contract": (
                "BayesFilter-only local time-43 stop-gradient ablations. "
                "Forward values should remain scalar-aligned before gradient "
                "deltas are interpreted."
            ),
            "rows": rows,
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _stop_gradient_mode_description(mode: str) -> str:
    descriptions = {
        "raw": "No added local stop-gradient at time 43.",
        "stop_proposed_particles": "Stop gradient through proposed particles at time 43.",
        "stop_proposal_mean": "Stop gradient through proposal mean at time 43.",
        "stop_proposal_ll": "Stop gradient through proposal log probability at time 43.",
        "stop_transition_ll": "Stop gradient through transition log probability at time 43.",
        "stop_observation_ll": "Stop gradient through observation log probability at time 43.",
        "stop_proposal_sample_noise": (
            "Keep proposed-particle values but stop the sample-noise residual "
            "from feeding back through proposal mean at time 43."
        ),
    }
    if mode not in descriptions:
        raise ValueError(f"unknown stop-gradient mode: {mode}")
    return descriptions[mode]


def _compare_stop_gradient_ablation(
    filterflow: dict[str, Any],
    ablation: dict[str, Any],
) -> dict[str, Any]:
    if ablation.get("status") != "executed":
        return {"status": "blocked", "blocker": "ablation did not execute"}
    valid = [
        row
        for row in ablation["rows"]
        if row["finite_scalar"]
        and row["finite_gradient"]
        and row["scalar_within_tolerance"]
    ]
    best = (
        min(valid, key=lambda row: row["max_abs_gradient_delta_to_filterflow"])
        if valid
        else None
    )
    raw = next(row for row in ablation["rows"] if row["mode"] == "raw")
    matching = [
        row
        for row in valid
        if row["max_abs_gradient_delta_to_filterflow"] <= GRADIENT_TOLERANCE
    ]
    return {
        "status": "compared",
        "filterflow_target_scalar": filterflow["target_scalar"],
        "filterflow_gradient_diag": filterflow["total_gradient_diag"],
        "raw_max_abs_gradient_delta": raw["max_abs_gradient_delta_to_filterflow"],
        "best_value_valid_mode": best,
        "matching_modes": matching,
        "interpretation": _interpret_ablation(raw, best, matching),
    }


def _interpret_ablation(
    raw: dict[str, Any],
    best: dict[str, Any] | None,
    matching: list[dict[str, Any]],
) -> str:
    if matching:
        return f"mode_{matching[0]['mode']}_collapses_filterflow_gradient_delta"
    if best is None:
        return "all_modes_vetoed_by_scalar_or_finiteness"
    if best["max_abs_gradient_delta_to_filterflow"] < raw["max_abs_gradient_delta_to_filterflow"]:
        return f"mode_{best['mode']}_reduces_filterflow_gradient_delta"
    return "tested_modes_do_not_reduce_filterflow_gradient_delta"


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
        "probe_times": list(PROBE_TIMES),
        "theta": vjp.THETA,
        "num_particles": vjp.NUM_PARTICLES,
        "data_seed": vjp.DATA_SEED,
        "filter_seed": vjp.FILTER_SEED,
        "dtype": "float64",
        "comparison_scope": "proposal/update local forward values, target adjoints, local gradients, and time-43 BayesFilter stop-gradient ablations",
        "stop_gradient_modes": list(STOP_GRADIENT_MODES),
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": "Do BayesFilter and float64 FilterFlow differ in the local proposal/update adjoint contract at time 43?",
        "baseline": "local canonical executable float64 FilterFlow reference",
        "primary_criterion": (
            "with vetoes clear, classify the row-173 target-time-93 gradient "
            "difference into h1-h4 using time-43 proposal/update forward, "
            "adjoint, and local-gradient comparisons"
        ),
        "veto_diagnostics": [
            "comparator drift",
            "scalar value mismatch",
            "resampling flag mismatch",
            "non-finite forward tensors, adjoints, or local gradients",
            "CPU-only policy failure",
            "path boundary contamination",
            "BayesFilter stop-gradient ablation did not execute",
        ],
        "explanatory_diagnostics": [
            "first forward delta node",
            "first adjoint delta node",
            "first local-gradient delta",
            "official proposal-likelihood wiring delta versus direct/fresh distribution log-prob probes",
            "raw and clipped transport upstream deltas",
            "transport residuals",
            "clip-mask mismatch counts",
            "BayesFilter-only local stop-gradient ablation table",
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
            "main_uncertainty": "single row, target time, and one probe time; no global claim",
            "next_justified_action": _next_action(classification["classification"]),
            "not_concluded": "correctness, posterior correctness, production readiness, global gradient agreement",
        }
    ]


def _next_action(classification: str) -> str:
    if classification == "h1_forward_contract_drift":
        return "repair or further localize the first proposal/update forward tensor drift before using adjoint evidence"
    if classification == "h2_proposal_sample_gradient_contract":
        return "trace reparameterized proposal sample VJP contract and test a value-preserving sample-path boundary"
    if classification == "h3_log_prob_gradient_contract":
        return "trace proposal, transition, and observation log-prob gradients with exact FilterFlow graph arithmetic"
    if classification == "h4_downstream_update_topology":
        return "trace the official proposal-likelihood wiring path and reconcile BayesFilter proposal_ll construction with executable FilterFlow"
    return "repair blocker or veto"


def _non_implications() -> list[str]:
    return [
        "No correctness claim is made for either implementation.",
        "No analytic gradient correctness is concluded.",
        "No posterior correctness is concluded.",
        "No global gradient agreement is concluded.",
        "No full mesh or surface agreement is concluded.",
        "No production readiness or public API readiness is concluded.",
        "BayesFilter stop-gradient ablations are explanatory only and do not define a preferred algorithm.",
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
        "filterflow_probe",
        "bayesfilter_probe",
        "source_comparison",
        "stop_gradient_ablation",
        "stop_gradient_ablation_comparison",
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
    if payload["run_manifest"].get("gpu_devices_visible") != []:
        raise ValueError(
            f"GPU devices visible in parent: {payload['run_manifest'].get('gpu_devices_visible')}"
        )
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    allowed = {
        "filterflow_float64_row_173_proposal_adjoint_topology_filterflow_blocker",
        "filterflow_float64_row_173_proposal_adjoint_topology_bayesfilter_blocker",
        "filterflow_float64_row_173_proposal_adjoint_topology_blocked_or_vetoed",
        "filterflow_float64_row_173_proposal_adjoint_topology_h1_forward_contract_drift",
        "filterflow_float64_row_173_proposal_adjoint_topology_h2_proposal_sample_gradient_contract",
        "filterflow_float64_row_173_proposal_adjoint_topology_h3_log_prob_gradient_contract",
        "filterflow_float64_row_173_proposal_adjoint_topology_h4_downstream_update_topology",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    for label in ("filterflow_probe", "bayesfilter_probe"):
        side = payload.get(label)
        if side is not None and side.get("status") == "executed":
            manifest = side["cpu_only_manifest"]
            if manifest.get("pre_import_cuda_visible_devices") != "-1":
                raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES was not -1")
            if manifest.get("gpu_devices_visible") != []:
                raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")
    if payload["veto_status_table"].get("all_vetoes_clear", False):
        allowed_hypotheses = {
            "h1_forward_contract_drift",
            "h2_proposal_sample_gradient_contract",
            "h3_log_prob_gradient_contract",
            "h4_downstream_update_topology",
        }
        if payload["hypothesis_classification"] not in allowed_hypotheses:
            raise ValueError(f"unexpected hypothesis: {payload['hypothesis_classification']}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: Row 173 Proposal/Update Adjoint-Topology Probe",
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
            "## Source Comparison",
            "",
            _json_block(payload["source_comparison"]),
            "",
            "## Veto Status",
            "",
            _json_block(payload["veto_status_table"]),
            "",
            "## Stop-Gradient Ablation Comparison",
            "",
            _json_block(payload["stop_gradient_ablation_comparison"]),
            "",
            "## Stop-Gradient Ablation",
            "",
            _json_block(payload["stop_gradient_ablation"]),
            "",
            "## FilterFlow Probe",
            "",
            _json_block(payload["filterflow_probe"]),
            "",
            "## BayesFilter Probe",
            "",
            _json_block(payload["bayesfilter_probe"]),
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
