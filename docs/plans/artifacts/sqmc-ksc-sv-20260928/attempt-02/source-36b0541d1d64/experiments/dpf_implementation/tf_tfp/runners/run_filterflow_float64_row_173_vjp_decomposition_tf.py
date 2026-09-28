"""Decompose the row-173 float64 FilterFlow/BayesFilter gradient residual."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
from dataclasses import dataclass
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


tfd = tfp.distributions

DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-row-173-vjp-decomposition-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-row-173-vjp-decomposition-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_row_173_vjp_decomposition_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-row-173-vjp-decomposition-2026-06-03.md"
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
TARGET_TIME_INDEX = 1
VALUE_TOLERANCE = 5e-8
GRADIENT_TOLERANCE = 2e-4
BOUNDARY_MODES = [
    "raw",
    "filterflow_custom_transport_gradient",
    "carry_log_weights_stop_gradient",
    "carry_log_likelihoods_stop_gradient",
    "carry_both_stop_gradient",
    "proposal_mean_stop_gradient",
    "target_transport_log_weights_stop_gradient",
    "all_times_transport_log_weights_stop_gradient",
    "proposal_sample_noise_stop_gradient",
    "all_times_proposal_sample_filterflow_contract",
    "target_proposal_sample_filterflow_contract",
    "fresh_proposal_log_prob_filterflow_contract",
    "proposal_sample_stop_gradient",
    "proposal_log_prob_stop_gradient",
    "carry_both_proposal_sample_stop_gradient",
]
DEFAULT_BAYESFILTER_BOUNDARY_MODE = "fresh_proposal_log_prob_filterflow_contract"
PROBE_ONLY_FIELDS = {"manual_sample_probe_particles"}
TARGET_FIELD_EXCLUSIONS = {"target", "flags"} | PROBE_ONLY_FIELDS


@dataclass(frozen=True)
class RunConfig:
    target_time_index: int
    tag: str | None
    plan_path: str
    result_path: str
    json_path: Any
    report_path: Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--target-time-index", type=int, default=TARGET_TIME_INDEX)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    config = _run_config(args)
    if args.validate_only:
        _validate_payload(load_json(config.json_path))
        return 0

    start = time.perf_counter()
    payload = _run(config)
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(config.json_path, payload)
    markdown = _markdown(payload)
    write_text(config.report_path, markdown)
    write_text(REPO_ROOT / config.result_path, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run_config(args: argparse.Namespace) -> RunConfig:
    tag = _safe_tag(args.tag)
    if tag is None:
        return RunConfig(
            target_time_index=int(args.target_time_index),
            tag=None,
            plan_path=PLAN_PATH,
            result_path=RESULT_PATH,
            json_path=JSON_PATH,
            report_path=REPORT_PATH,
        )
    return RunConfig(
        target_time_index=int(args.target_time_index),
        tag=tag,
        plan_path=(
            "docs/plans/"
            f"bayesfilter-dpf-filterflow-float64-{tag}-vjp-decomposition-plan-2026-06-03.md"
        ),
        result_path=(
            "docs/plans/"
            f"bayesfilter-dpf-filterflow-float64-{tag}-vjp-decomposition-result-2026-06-03.md"
        ),
        json_path=OUTPUT_DIR / f"dpf_filterflow_float64_{tag.replace('-', '_')}_vjp_decomposition_2026-06-03.json",
        report_path=REPORT_DIR / f"dpf-filterflow-float64-{tag}-vjp-decomposition-2026-06-03.md",
    )


def _safe_tag(tag: str | None) -> str | None:
    if tag is None or tag.strip() == "":
        return None
    lowered = tag.strip().lower()
    chars = []
    for char in lowered:
        if char.isalnum() or char == "-":
            chars.append(char)
        elif char in {"_", " "}:
            chars.append("-")
        else:
            raise ValueError(f"unsupported tag character: {char!r}")
    safe = "".join(chars).strip("-")
    if not safe:
        raise ValueError("tag must contain at least one alphanumeric character")
    return safe


def _run(config: RunConfig) -> dict[str, Any]:
    reference_status = r3._filterflow_status()
    validate_filterflow_reference_status(reference_status, marker_path=FILTERFLOW_MARKER_PATH)
    initial_fingerprint = continuation._filterflow_fingerprint()
    filterflow = _filterflow_vjp_subprocess(config)
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            config,
            "filterflow_float64_row_173_vjp_filterflow_blocker",
            filterflow.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow,
            reference_status,
        )
    bayesfilter = _bayesfilter_vjp(filterflow, config)
    bayesfilter_boundary_modes = _bayesfilter_boundary_modes(filterflow, config)
    comparison = _compare(filterflow, bayesfilter)
    boundary_mode_comparison = _compare_boundary_modes(filterflow, bayesfilter_boundary_modes)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(comparison, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": config.plan_path,
        "result_path": config.result_path,
        "report_path": str(config.report_path.relative_to(REPO_ROOT)),
        "json_path": str(config.json_path.relative_to(REPO_ROOT)),
        "question": "row_173_time_1_clipped_default_vjp_difference_localization",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(config),
        "filterflow_vjp": _compact_side(filterflow),
        "bayesfilter_vjp": _compact_side(bayesfilter),
        "comparison": comparison,
        "bayesfilter_boundary_modes": bayesfilter_boundary_modes,
        "boundary_mode_comparison": boundary_mode_comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_vjp_decomposition_tf"
                f"{_command_suffix(config)}"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _blocked_payload(
    config: RunConfig,
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
        "plan_path": config.plan_path,
        "result_path": config.result_path,
        "blocker": blocker,
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "model_contract": _model_contract(config),
        "filterflow_vjp": filterflow,
        "bayesfilter_vjp": None,
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_vjp_decomposition_tf"
                f"{_command_suffix(config)}"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _command_suffix(config: RunConfig) -> str:
    parts = []
    if config.target_time_index != TARGET_TIME_INDEX:
        parts.append(f" --target-time-index {config.target_time_index}")
    if config.tag is not None:
        parts.append(f" --tag {config.tag}")
    return "".join(parts)


def _filterflow_vjp_subprocess(config: RunConfig) -> dict[str, Any]:
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
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_vjp_script(config)],
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
            "blocker": "filterflow VJP subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_ROW_173_VJP_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_ROW_173_VJP_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow VJP JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(
        completed.stdout[start + len("FILTERFLOW_ROW_173_VJP_JSON_BEGIN"):end].strip()
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_vjp_script(config: RunConfig) -> str:
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
        TARGET_TIME_INDEX = {config.target_time_index}
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
                manual_proposal_mean = tf.linalg.matvec(
                    smc._proposal_model._observation_matrix,
                    tf.linalg.matvec(
                        smc._proposal_model._observation_covariance_inv,
                        observation,
                    ),
                    transpose_a=True,
                )
                manual_proposal_mean += tf.linalg.matvec(
                    smc._proposal_model._transition_covariance_inv,
                    tf.linalg.matvec(
                        smc._proposal_model._transition_matrix,
                        resampled_state.particles,
                    ),
                )
                manual_proposal_mean = tf.linalg.matvec(
                    smc._proposal_model._sigma,
                    manual_proposal_mean,
                )
                proposed_particles = proposal_dist.sample(seed=seed2)
                manual_sample_probe_dist = tfd.MultivariateNormalTriL(
                    manual_proposal_mean,
                    smc._proposal_model._sigma_chol,
                )
                manual_sample_probe_particles = manual_sample_probe_dist.sample(seed=seed2)
                proposal_dist_log_prob = proposal_dist.log_prob(proposed_particles)
                manual_dist_log_prob = manual_sample_probe_dist.log_prob(proposed_particles)
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
                unnormalized = transition_ll + observation_ll - proposal_ll + resampled_state.log_weights
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
                    target = tf.reduce_mean(log_likelihoods)
                    sum_pre_current_plus_increment_mean = tf.reduce_mean(
                        pre_current_log_likelihoods + increment
                    )
                    pre_current_mean = tf.reduce_mean(pre_current_log_likelihoods)
                    increment_mean = tf.reduce_mean(increment)
                    target_bundle = {{
                        "target": target,
                        "post_update_mean": target,
                        "sum_pre_current_plus_increment_mean": sum_pre_current_plus_increment_mean,
                        "pre_current_mean": pre_current_mean,
                        "increment_mean": increment_mean,
                        "flags": flags,
                        "log_ess": tf.math.log(ess),
                        "pre_particles": state_with_ess.particles,
                        "pre_log_weights": state_with_ess.log_weights,
                        "transport_matrix": transport_matrix,
                        "post_particles": resampled_state.particles,
                        "post_log_weights": resampled_state.log_weights,
                        "proposal_loc": proposal_loc,
                        "proposal_mean": proposal_mean,
                        "manual_proposal_mean": manual_proposal_mean,
                        "fresh_proposal_loc": fresh_proposal_loc,
                        "fresh_proposal_mean": fresh_proposal_mean,
                        "proposed_particles": proposed_particles,
                        "manual_sample_probe_particles": manual_sample_probe_particles,
                        "observation_ll": observation_ll,
                        "transition_ll": transition_ll,
                        "proposal_ll": proposal_ll,
                        "proposal_dist_log_prob": proposal_dist_log_prob,
                        "manual_dist_log_prob": manual_dist_log_prob,
                        "fresh_dist_log_prob": fresh_dist_log_prob,
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

        target = target_bundle["target"]
        total_gradient = watched_grad(tape, target, modifiable_transition_matrix)
        gradients = {{
            name: watched_grad(tape, target, tensor)
            for name, tensor in target_bundle.items()
            if name not in {{"target", "flags", "manual_sample_probe_particles"}}
        }}
        clipped_transport_upstream = tf.clip_by_value(gradients["transport_matrix"], -1.0, 1.0)
        direct_pre_particle_adjoint = tf.linalg.matmul(
            target_bundle["transport_matrix"],
            gradients["post_particles"],
            transpose_a=True,
        )
        same_tape_post_particles_vjp = watched_grad_with_upstream(
            tape,
            target_bundle["post_particles"],
            target_bundle["pre_particles"],
            gradients["post_particles"],
        )
        same_tape_post_log_weights_vjp = watched_grad_with_upstream(
            tape,
            target_bundle["post_log_weights"],
            target_bundle["pre_particles"],
            gradients["post_log_weights"],
        )
        same_tape_post_state_vjp = same_tape_post_particles_vjp + same_tape_post_log_weights_vjp
        same_tape_post_state_identity_residual = gradients["pre_particles"] - same_tape_post_state_vjp
        same_tape_pre_log_weights_carryover_vjp = watched_grad_with_upstream(
            tape,
            target_bundle["pre_log_weights"],
            target_bundle["pre_particles"],
            gradients["pre_log_weights"],
        )
        same_tape_pre_current_ll_carryover_vjp = watched_grad_with_upstream(
            tape,
            target_bundle["pre_current_log_likelihoods"],
            target_bundle["pre_particles"],
            gradients["pre_current_log_likelihoods"],
        )
        same_tape_log_ess_carryover_vjp = watched_grad_with_upstream(
            tape,
            target_bundle["log_ess"],
            target_bundle["pre_particles"],
            gradients["log_ess"],
        )
        same_tape_full_recorded_state_vjp = (
            same_tape_post_state_vjp
            + same_tape_pre_log_weights_carryover_vjp
            + same_tape_pre_current_ll_carryover_vjp
            + same_tape_log_ess_carryover_vjp
        )
        same_tape_full_recorded_state_residual = (
            gradients["pre_particles"] - same_tape_full_recorded_state_vjp
        )
        same_tape_transport_matrix_vjp = watched_grad_with_upstream(
            tape,
            target_bundle["transport_matrix"],
            target_bundle["pre_particles"],
            gradients["transport_matrix"],
        )
        same_tape_reconstructed_pre_particle_adjoint = (
            direct_pre_particle_adjoint + same_tape_transport_matrix_vjp
        )
        same_tape_identity_residual = (
            gradients["pre_particles"] - same_tape_reconstructed_pre_particle_adjoint
        )
        carryover_pre_particle_adjoint = watched_grad(
            tape,
            tf.reduce_mean(target_bundle["pre_current_log_likelihoods"]),
            target_bundle["pre_particles"],
        )
        current_increment_pre_particle_adjoint = watched_grad(
            tape,
            tf.reduce_mean(target_bundle["increment"]),
            target_bundle["pre_particles"],
        )
        implicit_pre_particle_adjoint = gradients["pre_particles"] - direct_pre_particle_adjoint
        local_post_particle_adjoints = {{
            "proposal_loc_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["proposal_loc"],
                target_bundle["post_particles"],
                gradients["proposal_loc"],
            ),
            "proposal_mean_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["proposal_mean"],
                target_bundle["post_particles"],
                gradients["proposal_mean"],
            ),
            "manual_proposal_mean_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["manual_proposal_mean"],
                target_bundle["post_particles"],
                gradients["manual_proposal_mean"],
            ),
            "fresh_proposal_loc_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["fresh_proposal_loc"],
                target_bundle["post_particles"],
                gradients["fresh_proposal_loc"],
            ),
            "fresh_proposal_mean_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["fresh_proposal_mean"],
                target_bundle["post_particles"],
                gradients["fresh_proposal_mean"],
            ),
            "proposed_particles_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["proposed_particles"],
                target_bundle["post_particles"],
                gradients["proposed_particles"],
            ),
            "observation_ll_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["observation_ll"],
                target_bundle["post_particles"],
                gradients["observation_ll"],
            ),
            "transition_ll_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["transition_ll"],
                target_bundle["post_particles"],
                gradients["transition_ll"],
            ),
            "proposal_ll_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["proposal_ll"],
                target_bundle["post_particles"],
                gradients["proposal_ll"],
            ),
            "unnormalized_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["unnormalized"],
                target_bundle["post_particles"],
                gradients["unnormalized"],
            ),
            "increment_to_post_particles": watched_grad_with_upstream(
                tape,
                target_bundle["increment"],
                target_bundle["post_particles"],
                gradients["increment"],
            ),
        }}
        sample_contract_tensors = {{
            "actual_sample_to_proposal_loc": watched_grad_with_upstream(
                tape,
                target_bundle["proposed_particles"],
                target_bundle["proposal_loc"],
                gradients["proposed_particles"],
            ),
            "actual_sample_to_proposal_mean": watched_grad_with_upstream(
                tape,
                target_bundle["proposed_particles"],
                target_bundle["proposal_mean"],
                gradients["proposed_particles"],
            ),
            "actual_sample_to_manual_proposal_mean": watched_grad_with_upstream(
                tape,
                target_bundle["proposed_particles"],
                target_bundle["manual_proposal_mean"],
                gradients["proposed_particles"],
            ),
            "actual_sample_to_fresh_proposal_loc": watched_grad_with_upstream(
                tape,
                target_bundle["proposed_particles"],
                target_bundle["fresh_proposal_loc"],
                gradients["proposed_particles"],
            ),
            "actual_sample_to_fresh_proposal_mean": watched_grad_with_upstream(
                tape,
                target_bundle["proposed_particles"],
                target_bundle["fresh_proposal_mean"],
                gradients["proposed_particles"],
            ),
            "manual_probe_sample_to_manual_proposal_mean": watched_grad_with_upstream(
                tape,
                target_bundle["manual_sample_probe_particles"],
                target_bundle["manual_proposal_mean"],
                gradients["proposed_particles"],
            ),
            "manual_probe_sum_to_manual_proposal_mean": watched_grad(
                tape,
                tf.reduce_sum(target_bundle["manual_sample_probe_particles"]),
                target_bundle["manual_proposal_mean"],
            ),
            "actual_sample_to_post_particles": local_post_particle_adjoints[
                "proposed_particles_to_post_particles"
            ],
        }}
        proposal_log_prob_parameter_path_tensors = {{
            "official_proposal_ll": watched_grad_with_upstream(
                tape,
                target_bundle["proposal_ll"],
                modifiable_transition_matrix,
                gradients["proposal_ll"],
            ),
            "first_dist_log_prob": watched_grad_with_upstream(
                tape,
                target_bundle["proposal_dist_log_prob"],
                modifiable_transition_matrix,
                gradients["proposal_ll"],
            ),
            "manual_dist_log_prob": watched_grad_with_upstream(
                tape,
                target_bundle["manual_dist_log_prob"],
                modifiable_transition_matrix,
                gradients["proposal_ll"],
            ),
            "fresh_dist_log_prob": watched_grad_with_upstream(
                tape,
                target_bundle["fresh_dist_log_prob"],
                modifiable_transition_matrix,
                gradients["proposal_ll"],
            ),
        }}
        proposal_sample_gradient_contract = {{
            "contract": (
                "Probe actual FilterFlow proposal_dist.sample and a manual distribution "
                "built from the explicit proposal mean under the downstream upstream "
                "gradient for proposed_particles."
            ),
            "value_probe": {{
                "manual_probe_minus_actual_sample": field(
                    target_bundle["manual_sample_probe_particles"]
                    - target_bundle["proposed_particles"]
                ),
                "manual_probe_minus_manual_mean": field(
                    target_bundle["manual_sample_probe_particles"]
                    - target_bundle["manual_proposal_mean"]
                ),
                "actual_sample_minus_manual_mean": field(
                    target_bundle["proposed_particles"]
                    - target_bundle["manual_proposal_mean"]
                ),
            }},
            "value_probe_tensors": {{
                "manual_probe_minus_actual_sample": to_json(
                    target_bundle["manual_sample_probe_particles"]
                    - target_bundle["proposed_particles"]
                ),
                "manual_probe_minus_manual_mean": to_json(
                    target_bundle["manual_sample_probe_particles"]
                    - target_bundle["manual_proposal_mean"]
                ),
                "actual_sample_minus_manual_mean": to_json(
                    target_bundle["proposed_particles"]
                    - target_bundle["manual_proposal_mean"]
                ),
            }},
            "vjp_probe": {{
                name: field(tensor)
                for name, tensor in sample_contract_tensors.items()
            }},
            "vjp_tensors": {{
                name: to_json(tensor)
                for name, tensor in sample_contract_tensors.items()
            }},
        }}
        proposal_topology_probe = {{
            "contract": (
                "Difference-audit probe for the optimal-proposal graph topology. "
                "The executable FilterFlow scalar samples from one proposal distribution "
                "and evaluates proposal log probability through the proposal-model "
                "loglikelihood method, which constructs a fresh proposal distribution."
            ),
            "value_summaries": {{
                "proposal_loc_minus_proposal_mean": field(
                    target_bundle["proposal_loc"] - target_bundle["proposal_mean"]
                ),
                "manual_minus_proposal_loc": field(
                    target_bundle["manual_proposal_mean"] - target_bundle["proposal_loc"]
                ),
                "fresh_loc_minus_proposal_loc": field(
                    target_bundle["fresh_proposal_loc"] - target_bundle["proposal_loc"]
                ),
                "fresh_mean_minus_fresh_loc": field(
                    target_bundle["fresh_proposal_mean"] - target_bundle["fresh_proposal_loc"]
                ),
                "official_proposal_ll_minus_first_dist_log_prob": field(
                    target_bundle["proposal_ll"] - target_bundle["proposal_dist_log_prob"]
                ),
                "official_proposal_ll_minus_manual_dist_log_prob": field(
                    target_bundle["proposal_ll"] - target_bundle["manual_dist_log_prob"]
                ),
                "official_proposal_ll_minus_fresh_dist_log_prob": field(
                    target_bundle["proposal_ll"] - target_bundle["fresh_dist_log_prob"]
                ),
            }},
            "value_tensors": {{
                "proposal_loc_minus_proposal_mean": to_json(
                    target_bundle["proposal_loc"] - target_bundle["proposal_mean"]
                ),
                "manual_minus_proposal_loc": to_json(
                    target_bundle["manual_proposal_mean"] - target_bundle["proposal_loc"]
                ),
                "fresh_loc_minus_proposal_loc": to_json(
                    target_bundle["fresh_proposal_loc"] - target_bundle["proposal_loc"]
                ),
                "fresh_mean_minus_fresh_loc": to_json(
                    target_bundle["fresh_proposal_mean"] - target_bundle["fresh_proposal_loc"]
                ),
                "official_proposal_ll_minus_first_dist_log_prob": to_json(
                    target_bundle["proposal_ll"] - target_bundle["proposal_dist_log_prob"]
                ),
                "official_proposal_ll_minus_manual_dist_log_prob": to_json(
                    target_bundle["proposal_ll"] - target_bundle["manual_dist_log_prob"]
                ),
                "official_proposal_ll_minus_fresh_dist_log_prob": to_json(
                    target_bundle["proposal_ll"] - target_bundle["fresh_dist_log_prob"]
                ),
            }},
            "gradient_summaries": {{
                "target_to_proposal_loc": field(gradients["proposal_loc"]),
                "target_to_proposal_mean": field(gradients["proposal_mean"]),
                "target_to_manual_proposal_mean": field(gradients["manual_proposal_mean"]),
                "target_to_fresh_proposal_loc": field(gradients["fresh_proposal_loc"]),
                "target_to_fresh_proposal_mean": field(gradients["fresh_proposal_mean"]),
            }},
            "gradient_tensors": {{
                "target_to_proposal_loc": to_json(gradients["proposal_loc"]),
                "target_to_proposal_mean": to_json(gradients["proposal_mean"]),
                "target_to_manual_proposal_mean": to_json(gradients["manual_proposal_mean"]),
                "target_to_fresh_proposal_loc": to_json(gradients["fresh_proposal_loc"]),
                "target_to_fresh_proposal_mean": to_json(gradients["fresh_proposal_mean"]),
            }},
            "proposal_log_prob_parameter_path_summaries": {{
                name: field(tf.linalg.diag_part(tensor))
                for name, tensor in proposal_log_prob_parameter_path_tensors.items()
            }},
            "proposal_log_prob_parameter_path_tensors": {{
                name: to_json(tf.linalg.diag_part(tensor))
                for name, tensor in proposal_log_prob_parameter_path_tensors.items()
            }},
        }}
        parameter_path_adjoints = {{
            name: watched_grad_with_upstream(
                tape,
                target_bundle[name],
                modifiable_transition_matrix,
                gradients[name],
            )
            for name in gradients
        }}
        scalar_additivity_objectives = {{
            "post_update_mean": target_bundle["post_update_mean"],
            "sum_pre_current_plus_increment_mean": target_bundle[
                "sum_pre_current_plus_increment_mean"
            ],
            "pre_current_mean": target_bundle["pre_current_mean"],
            "increment_mean": target_bundle["increment_mean"],
        }}
        scalar_additivity_gradients = {{
            name: watched_grad(tape, objective, modifiable_transition_matrix)
            for name, objective in scalar_additivity_objectives.items()
        }}
        transport_clipping_objectives = scalar_additivity_objectives
        transport_clipping_raw_upstreams = {{
            name: watched_grad(tape, objective, target_bundle["transport_matrix"])
            for name, objective in transport_clipping_objectives.items()
        }}
        transport_clipping_clipped_upstreams = {{
            name: tf.clip_by_value(upstream, -1.0, 1.0)
            for name, upstream in transport_clipping_raw_upstreams.items()
        }}
        transport_clipping_masks = {{
            name: tf.cast(tf.abs(upstream) > tf.constant(1.0, DTYPE), DTYPE)
            for name, upstream in transport_clipping_raw_upstreams.items()
        }}
        transport_clipping_target_time_vjps = {{
            name: watched_grad_with_upstream(
                tape,
                target_bundle["transport_matrix"],
                modifiable_transition_matrix,
                upstream,
            )
            for name, upstream in transport_clipping_raw_upstreams.items()
        }}
        transport_clipping_manual_clipped_target_time_vjps = {{
            name: watched_grad_with_upstream(
                tape,
                target_bundle["transport_matrix"],
                modifiable_transition_matrix,
                clipped,
            )
            for name, clipped in transport_clipping_clipped_upstreams.items()
        }}
        filterflow_proposal_mean_internal_probe = {{
            "value_delta": field(target_bundle["manual_proposal_mean"] - target_bundle["proposal_mean"]),
            "proposal_mean_adjoint": field(local_post_particle_adjoints["proposal_mean_to_post_particles"]),
            "manual_proposal_mean_adjoint": field(
                local_post_particle_adjoints["manual_proposal_mean_to_post_particles"]
            ),
            "adjoint_delta": field(
                local_post_particle_adjoints["manual_proposal_mean_to_post_particles"]
                - local_post_particle_adjoints["proposal_mean_to_post_particles"]
            ),
            "value_delta_tensor": to_json(
                target_bundle["manual_proposal_mean"] - target_bundle["proposal_mean"]
            ),
            "adjoint_delta_tensor": to_json(
                local_post_particle_adjoints["manual_proposal_mean_to_post_particles"]
                - local_post_particle_adjoints["proposal_mean_to_post_particles"]
            ),
        }}
        del tape

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
                "transport_backward": "FilterFlow custom gradient clips d_transport to [-1,1]",
            }},
            "values": {{
                name: field(tensor)
                for name, tensor in target_bundle.items()
                if name not in {{"target", "flags", "manual_sample_probe_particles"}}
            }},
            "value_tensors": {{
                name: to_json(tensor)
                for name, tensor in target_bundle.items()
                if name not in {{"target", "flags", "manual_sample_probe_particles"}}
            }},
            "gradients": {{
                name: field(tensor)
                for name, tensor in gradients.items()
            }},
            "gradient_tensors": {{
                name: to_json(tensor)
                for name, tensor in gradients.items()
            }},
            "total_gradient_matrix": to_json(total_gradient),
            "total_gradient_diag": to_json(tf.linalg.diag_part(total_gradient)),
            "target_scalar": scalar(target),
            "resampling_flag": [bool(v) for v in tf.reshape(target_bundle["flags"], [-1]).numpy().tolist()],
            "transport_upstream_clipped": field(clipped_transport_upstream),
            "resampling_adjoint_decomposition": {{
                "direct_pre_particle_adjoint": field(direct_pre_particle_adjoint),
                "same_tape_post_particles_vjp": field(same_tape_post_particles_vjp),
                "same_tape_post_log_weights_vjp": field(same_tape_post_log_weights_vjp),
                "same_tape_post_state_vjp": field(same_tape_post_state_vjp),
                "same_tape_post_state_identity_residual": field(
                    same_tape_post_state_identity_residual
                ),
                "same_tape_pre_log_weights_carryover_vjp": field(
                    same_tape_pre_log_weights_carryover_vjp
                ),
                "same_tape_pre_current_ll_carryover_vjp": field(
                    same_tape_pre_current_ll_carryover_vjp
                ),
                "same_tape_log_ess_carryover_vjp": field(same_tape_log_ess_carryover_vjp),
                "same_tape_full_recorded_state_vjp": field(same_tape_full_recorded_state_vjp),
                "same_tape_full_recorded_state_residual": field(
                    same_tape_full_recorded_state_residual
                ),
                "same_tape_transport_matrix_vjp": field(same_tape_transport_matrix_vjp),
                "same_tape_reconstructed_pre_particle_adjoint": field(
                    same_tape_reconstructed_pre_particle_adjoint
                ),
                "same_tape_identity_residual": field(same_tape_identity_residual),
                "implicit_pre_particle_adjoint": field(implicit_pre_particle_adjoint),
                "carryover_pre_particle_adjoint": field(carryover_pre_particle_adjoint),
                "current_increment_pre_particle_adjoint": field(current_increment_pre_particle_adjoint),
                "direct_pre_particle_adjoint_tensor": to_json(direct_pre_particle_adjoint),
                "same_tape_post_particles_vjp_tensor": to_json(same_tape_post_particles_vjp),
                "same_tape_post_log_weights_vjp_tensor": to_json(same_tape_post_log_weights_vjp),
                "same_tape_post_state_vjp_tensor": to_json(same_tape_post_state_vjp),
                "same_tape_post_state_identity_residual_tensor": to_json(
                    same_tape_post_state_identity_residual
                ),
                "same_tape_pre_log_weights_carryover_vjp_tensor": to_json(
                    same_tape_pre_log_weights_carryover_vjp
                ),
                "same_tape_pre_current_ll_carryover_vjp_tensor": to_json(
                    same_tape_pre_current_ll_carryover_vjp
                ),
                "same_tape_log_ess_carryover_vjp_tensor": to_json(
                    same_tape_log_ess_carryover_vjp
                ),
                "same_tape_full_recorded_state_vjp_tensor": to_json(
                    same_tape_full_recorded_state_vjp
                ),
                "same_tape_full_recorded_state_residual_tensor": to_json(
                    same_tape_full_recorded_state_residual
                ),
                "same_tape_transport_matrix_vjp_tensor": to_json(same_tape_transport_matrix_vjp),
                "same_tape_reconstructed_pre_particle_adjoint_tensor": to_json(
                    same_tape_reconstructed_pre_particle_adjoint
                ),
                "same_tape_identity_residual_tensor": to_json(same_tape_identity_residual),
                "implicit_pre_particle_adjoint_tensor": to_json(implicit_pre_particle_adjoint),
                "carryover_pre_particle_adjoint_tensor": to_json(carryover_pre_particle_adjoint),
                "current_increment_pre_particle_adjoint_tensor": to_json(current_increment_pre_particle_adjoint),
            }},
            "local_post_particle_adjoint_probe": {{
                name: field(tensor)
                for name, tensor in local_post_particle_adjoints.items()
            }},
            "local_post_particle_adjoint_tensors": {{
                name: to_json(tensor)
                for name, tensor in local_post_particle_adjoints.items()
            }},
            "parameter_path_adjoint_probe": {{
                name: field(tf.linalg.diag_part(tensor))
                for name, tensor in parameter_path_adjoints.items()
            }},
            "parameter_path_adjoint_tensors": {{
                name: to_json(tf.linalg.diag_part(tensor))
                for name, tensor in parameter_path_adjoints.items()
            }},
            "scalar_additivity_probe": {{
                "contract": (
                    "Direct scalar-gradient additivity check for "
                    "post_update_log_likelihoods = pre_current_log_likelihoods "
                    "+ increment. These gradients use scalar objectives directly, "
                    "not target-upstream VJPs through intermediate tensors."
                ),
                "scalar_values": {{
                    name: scalar(value)
                    for name, value in scalar_additivity_objectives.items()
                }},
                "gradient_matrix_summaries": {{
                    name: field(value)
                    for name, value in scalar_additivity_gradients.items()
                }},
                "gradient_matrix_tensors": {{
                    name: to_json(value)
                    for name, value in scalar_additivity_gradients.items()
                }},
                "gradient_diag_tensors": {{
                    name: to_json(tf.linalg.diag_part(value))
                    for name, value in scalar_additivity_gradients.items()
                }},
            }},
            "transport_clipping_probe": {{
                "contract": (
                    "Target-time transport custom-gradient clipping probe. "
                    "Raw upstreams are d scalar / d transport_matrix; clipped "
                    "upstreams apply clip_by_value(upstream, -1, 1); VJPs are "
                    "d transport_matrix / d transition parameter under the "
                    "implementation's recorded target-time transport graph."
                ),
                "raw_upstream_summaries": {{
                    name: field(value)
                    for name, value in transport_clipping_raw_upstreams.items()
                }},
                "raw_upstream_tensors": {{
                    name: to_json(value)
                    for name, value in transport_clipping_raw_upstreams.items()
                }},
                "clipped_upstream_summaries": {{
                    name: field(value)
                    for name, value in transport_clipping_clipped_upstreams.items()
                }},
                "clipped_upstream_tensors": {{
                    name: to_json(value)
                    for name, value in transport_clipping_clipped_upstreams.items()
                }},
                "clip_mask_summaries": {{
                    name: field(value)
                    for name, value in transport_clipping_masks.items()
                }},
                "clip_mask_tensors": {{
                    name: to_json(value)
                    for name, value in transport_clipping_masks.items()
                }},
                "target_time_vjp_summaries": {{
                    name: field(tf.linalg.diag_part(value))
                    for name, value in transport_clipping_target_time_vjps.items()
                }},
                "target_time_vjp_diag_tensors": {{
                    name: to_json(tf.linalg.diag_part(value))
                    for name, value in transport_clipping_target_time_vjps.items()
                }},
                "manual_clipped_target_time_vjp_summaries": {{
                    name: field(tf.linalg.diag_part(value))
                    for name, value in transport_clipping_manual_clipped_target_time_vjps.items()
                }},
                "manual_clipped_target_time_vjp_diag_tensors": {{
                    name: to_json(tf.linalg.diag_part(value))
                    for name, value in transport_clipping_manual_clipped_target_time_vjps.items()
                }},
            }},
            "filterflow_proposal_mean_internal_probe": filterflow_proposal_mean_internal_probe,
            "proposal_sample_gradient_contract": proposal_sample_gradient_contract,
            "proposal_topology_probe": proposal_topology_probe,
            "transport_upstream_clip_fraction": scalar(
                tf.reduce_mean(
                    tf.cast(tf.abs(gradients["transport_matrix"]) > tf.constant(1.0, DTYPE), DTYPE)
                )
            ),
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
        print("FILTERFLOW_ROW_173_VJP_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_ROW_173_VJP_JSON_END")
        """
    )


def _bayesfilter_vjp(filterflow: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        theta_variable = tf.Variable(THETA, dtype=DTYPE)
        model = _model_from_filterflow(filterflow)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta_variable)
            bundle = _bayesfilter_target_bundle(
                theta_variable,
                model,
                config,
                boundary_mode=DEFAULT_BAYESFILTER_BOUNDARY_MODE,
            )
            target = bundle["target"]
        total_gradient = _safe_gradient(tape, target, theta_variable)
        gradients = {
            name: _safe_gradient(tape, target, tensor)
            for name, tensor in bundle.items()
            if name not in TARGET_FIELD_EXCLUSIONS
        }
        clipped_transport_upstream = tf.clip_by_value(gradients["transport_matrix"], -1.0, 1.0)
        direct_pre_particle_adjoint = tf.linalg.matmul(
            bundle["transport_matrix"],
            gradients["post_particles"],
            transpose_a=True,
        )
        same_tape_post_particles_vjp = _safe_gradient_with_upstream(
            tape,
            bundle["post_particles"],
            bundle["pre_particles"],
            gradients["post_particles"],
        )
        same_tape_post_log_weights_vjp = _safe_gradient_with_upstream(
            tape,
            bundle["post_log_weights"],
            bundle["pre_particles"],
            gradients["post_log_weights"],
        )
        same_tape_post_state_vjp = same_tape_post_particles_vjp + same_tape_post_log_weights_vjp
        same_tape_post_state_identity_residual = (
            gradients["pre_particles"] - same_tape_post_state_vjp
        )
        same_tape_pre_log_weights_carryover_vjp = _safe_gradient_with_upstream(
            tape,
            bundle["pre_log_weights"],
            bundle["pre_particles"],
            gradients["pre_log_weights"],
        )
        same_tape_pre_current_ll_carryover_vjp = _safe_gradient_with_upstream(
            tape,
            bundle["pre_current_log_likelihoods"],
            bundle["pre_particles"],
            gradients["pre_current_log_likelihoods"],
        )
        same_tape_log_ess_carryover_vjp = _safe_gradient_with_upstream(
            tape,
            bundle["log_ess"],
            bundle["pre_particles"],
            gradients["log_ess"],
        )
        same_tape_full_recorded_state_vjp = (
            same_tape_post_state_vjp
            + same_tape_pre_log_weights_carryover_vjp
            + same_tape_pre_current_ll_carryover_vjp
            + same_tape_log_ess_carryover_vjp
        )
        same_tape_full_recorded_state_residual = (
            gradients["pre_particles"] - same_tape_full_recorded_state_vjp
        )
        same_tape_transport_matrix_vjp = _safe_gradient_with_upstream(
            tape,
            bundle["transport_matrix"],
            bundle["pre_particles"],
            gradients["transport_matrix"],
        )
        same_tape_reconstructed_pre_particle_adjoint = (
            direct_pre_particle_adjoint + same_tape_transport_matrix_vjp
        )
        same_tape_identity_residual = (
            gradients["pre_particles"] - same_tape_reconstructed_pre_particle_adjoint
        )
        carryover_pre_particle_adjoint = _safe_gradient(
            tape,
            tf.reduce_mean(bundle["pre_current_log_likelihoods"]),
            bundle["pre_particles"],
        )
        current_increment_pre_particle_adjoint = _safe_gradient(
            tape,
            tf.reduce_mean(bundle["increment"]),
            bundle["pre_particles"],
        )
        implicit_pre_particle_adjoint = gradients["pre_particles"] - direct_pre_particle_adjoint
        local_post_particle_adjoints = {
            "proposal_loc_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["proposal_loc"],
                bundle["post_particles"],
                gradients["proposal_loc"],
            ),
            "proposal_mean_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["proposal_mean"],
                bundle["post_particles"],
                gradients["proposal_mean"],
            ),
            "manual_proposal_mean_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["manual_proposal_mean"],
                bundle["post_particles"],
                gradients["manual_proposal_mean"],
            ),
            "fresh_proposal_loc_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["fresh_proposal_loc"],
                bundle["post_particles"],
                gradients["fresh_proposal_loc"],
            ),
            "fresh_proposal_mean_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["fresh_proposal_mean"],
                bundle["post_particles"],
                gradients["fresh_proposal_mean"],
            ),
            "proposed_particles_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["proposed_particles"],
                bundle["post_particles"],
                gradients["proposed_particles"],
            ),
            "observation_ll_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["observation_ll"],
                bundle["post_particles"],
                gradients["observation_ll"],
            ),
            "transition_ll_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["transition_ll"],
                bundle["post_particles"],
                gradients["transition_ll"],
            ),
            "proposal_ll_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["proposal_ll"],
                bundle["post_particles"],
                gradients["proposal_ll"],
            ),
            "unnormalized_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["unnormalized"],
                bundle["post_particles"],
                gradients["unnormalized"],
            ),
            "increment_to_post_particles": _safe_gradient_with_upstream(
                tape,
                bundle["increment"],
                bundle["post_particles"],
                gradients["increment"],
            ),
        }
        sample_contract_tensors = {
            "actual_sample_to_proposal_loc": _safe_gradient_with_upstream(
                tape,
                bundle["proposed_particles"],
                bundle["proposal_loc"],
                gradients["proposed_particles"],
            ),
            "actual_sample_to_proposal_mean": _safe_gradient_with_upstream(
                tape,
                bundle["proposed_particles"],
                bundle["proposal_mean"],
                gradients["proposed_particles"],
            ),
            "actual_sample_to_manual_proposal_mean": _safe_gradient_with_upstream(
                tape,
                bundle["proposed_particles"],
                bundle["manual_proposal_mean"],
                gradients["proposed_particles"],
            ),
            "actual_sample_to_fresh_proposal_loc": _safe_gradient_with_upstream(
                tape,
                bundle["proposed_particles"],
                bundle["fresh_proposal_loc"],
                gradients["proposed_particles"],
            ),
            "actual_sample_to_fresh_proposal_mean": _safe_gradient_with_upstream(
                tape,
                bundle["proposed_particles"],
                bundle["fresh_proposal_mean"],
                gradients["proposed_particles"],
            ),
            "manual_probe_sample_to_manual_proposal_mean": _safe_gradient_with_upstream(
                tape,
                bundle["manual_sample_probe_particles"],
                bundle["manual_proposal_mean"],
                gradients["proposed_particles"],
            ),
            "manual_probe_sum_to_manual_proposal_mean": _safe_gradient(
                tape,
                tf.reduce_sum(bundle["manual_sample_probe_particles"]),
                bundle["manual_proposal_mean"],
            ),
            "actual_sample_to_post_particles": local_post_particle_adjoints[
                "proposed_particles_to_post_particles"
            ],
        }
        proposal_log_prob_parameter_path_tensors = {
            "official_proposal_ll": _safe_gradient_with_upstream(
                tape,
                bundle["proposal_ll"],
                theta_variable,
                gradients["proposal_ll"],
            ),
            "first_dist_log_prob": _safe_gradient_with_upstream(
                tape,
                bundle["proposal_dist_log_prob"],
                theta_variable,
                gradients["proposal_ll"],
            ),
            "manual_dist_log_prob": _safe_gradient_with_upstream(
                tape,
                bundle["manual_dist_log_prob"],
                theta_variable,
                gradients["proposal_ll"],
            ),
            "fresh_dist_log_prob": _safe_gradient_with_upstream(
                tape,
                bundle["fresh_dist_log_prob"],
                theta_variable,
                gradients["proposal_ll"],
            ),
        }
        graph_embedding_probe = _bayesfilter_transport_graph_embedding_probe(
            tape,
            bundle,
            gradients,
        )
        proposal_sample_gradient_contract = {
            "contract": (
                "Probe BayesFilter proposal_dist.sample and a manual distribution "
                "built from the explicit proposal mean under the downstream upstream "
                "gradient for proposed_particles."
            ),
            "value_probe": {
                "manual_probe_minus_actual_sample": _field(
                    bundle["manual_sample_probe_particles"]
                    - bundle["proposed_particles"]
                ),
                "manual_probe_minus_manual_mean": _field(
                    bundle["manual_sample_probe_particles"]
                    - bundle["manual_proposal_mean"]
                ),
                "actual_sample_minus_manual_mean": _field(
                    bundle["proposed_particles"]
                    - bundle["manual_proposal_mean"]
                ),
            },
            "value_probe_tensors": {
                "manual_probe_minus_actual_sample": r3._json(
                    bundle["manual_sample_probe_particles"]
                    - bundle["proposed_particles"]
                ),
                "manual_probe_minus_manual_mean": r3._json(
                    bundle["manual_sample_probe_particles"]
                    - bundle["manual_proposal_mean"]
                ),
                "actual_sample_minus_manual_mean": r3._json(
                    bundle["proposed_particles"]
                    - bundle["manual_proposal_mean"]
                ),
            },
            "vjp_probe": {
                name: _field(tensor)
                for name, tensor in sample_contract_tensors.items()
            },
            "vjp_tensors": {
                name: r3._json(tensor)
                for name, tensor in sample_contract_tensors.items()
            },
        }
        proposal_topology_probe = {
            "contract": (
                "Difference-audit probe for the optimal-proposal graph topology. "
                "BayesFilter constructs the proposal distribution in the replay "
                "loop and exposes explicit loc/mean/log-prob paths for comparison "
                "against executable FilterFlow."
            ),
            "value_summaries": {
                "proposal_loc_minus_proposal_mean": _field(
                    bundle["proposal_loc"] - bundle["proposal_mean"]
                ),
                "manual_minus_proposal_loc": _field(
                    bundle["manual_proposal_mean"] - bundle["proposal_loc"]
                ),
                "fresh_loc_minus_proposal_loc": _field(
                    bundle["fresh_proposal_loc"] - bundle["proposal_loc"]
                ),
                "fresh_mean_minus_fresh_loc": _field(
                    bundle["fresh_proposal_mean"] - bundle["fresh_proposal_loc"]
                ),
                "official_proposal_ll_minus_first_dist_log_prob": _field(
                    bundle["proposal_ll"] - bundle["proposal_dist_log_prob"]
                ),
                "official_proposal_ll_minus_manual_dist_log_prob": _field(
                    bundle["proposal_ll"] - bundle["manual_dist_log_prob"]
                ),
                "official_proposal_ll_minus_fresh_dist_log_prob": _field(
                    bundle["proposal_ll"] - bundle["fresh_dist_log_prob"]
                ),
            },
            "value_tensors": {
                "proposal_loc_minus_proposal_mean": r3._json(
                    bundle["proposal_loc"] - bundle["proposal_mean"]
                ),
                "manual_minus_proposal_loc": r3._json(
                    bundle["manual_proposal_mean"] - bundle["proposal_loc"]
                ),
                "fresh_loc_minus_proposal_loc": r3._json(
                    bundle["fresh_proposal_loc"] - bundle["proposal_loc"]
                ),
                "fresh_mean_minus_fresh_loc": r3._json(
                    bundle["fresh_proposal_mean"] - bundle["fresh_proposal_loc"]
                ),
                "official_proposal_ll_minus_first_dist_log_prob": r3._json(
                    bundle["proposal_ll"] - bundle["proposal_dist_log_prob"]
                ),
                "official_proposal_ll_minus_manual_dist_log_prob": r3._json(
                    bundle["proposal_ll"] - bundle["manual_dist_log_prob"]
                ),
                "official_proposal_ll_minus_fresh_dist_log_prob": r3._json(
                    bundle["proposal_ll"] - bundle["fresh_dist_log_prob"]
                ),
            },
            "gradient_summaries": {
                "target_to_proposal_loc": _field(gradients["proposal_loc"]),
                "target_to_proposal_mean": _field(gradients["proposal_mean"]),
                "target_to_manual_proposal_mean": _field(gradients["manual_proposal_mean"]),
                "target_to_fresh_proposal_loc": _field(gradients["fresh_proposal_loc"]),
                "target_to_fresh_proposal_mean": _field(gradients["fresh_proposal_mean"]),
            },
            "gradient_tensors": {
                "target_to_proposal_loc": r3._json(gradients["proposal_loc"]),
                "target_to_proposal_mean": r3._json(gradients["proposal_mean"]),
                "target_to_manual_proposal_mean": r3._json(gradients["manual_proposal_mean"]),
                "target_to_fresh_proposal_loc": r3._json(gradients["fresh_proposal_loc"]),
                "target_to_fresh_proposal_mean": r3._json(gradients["fresh_proposal_mean"]),
            },
            "proposal_log_prob_parameter_path_summaries": {
                name: _field(tensor)
                for name, tensor in proposal_log_prob_parameter_path_tensors.items()
            },
            "proposal_log_prob_parameter_path_tensors": {
                name: r3._json(tensor)
                for name, tensor in proposal_log_prob_parameter_path_tensors.items()
            },
        }
        parameter_path_adjoints = {
            name: _safe_gradient_with_upstream(
                tape,
                bundle[name],
                theta_variable,
                gradients[name],
            )
            for name in gradients
        }
        scalar_additivity_objectives = {
            "post_update_mean": bundle["post_update_mean"],
            "sum_pre_current_plus_increment_mean": bundle[
                "sum_pre_current_plus_increment_mean"
            ],
            "pre_current_mean": bundle["pre_current_mean"],
            "increment_mean": bundle["increment_mean"],
        }
        scalar_additivity_gradients = {
            name: _safe_gradient(tape, objective, theta_variable)
            for name, objective in scalar_additivity_objectives.items()
        }
        transport_clipping_objectives = scalar_additivity_objectives
        transport_clipping_raw_upstreams = {
            name: _safe_gradient(tape, objective, bundle["transport_matrix"])
            for name, objective in transport_clipping_objectives.items()
        }
        transport_clipping_clipped_upstreams = {
            name: tf.clip_by_value(upstream, -1.0, 1.0)
            for name, upstream in transport_clipping_raw_upstreams.items()
        }
        transport_clipping_masks = {
            name: tf.cast(tf.abs(upstream) > tf.constant(1.0, DTYPE), DTYPE)
            for name, upstream in transport_clipping_raw_upstreams.items()
        }
        transport_clipping_target_time_vjps = {
            name: _safe_gradient_with_upstream(
                tape,
                bundle["transport_matrix"],
                theta_variable,
                upstream,
            )
            for name, upstream in transport_clipping_raw_upstreams.items()
        }
        transport_clipping_manual_clipped_target_time_vjps = {
            name: _safe_gradient_with_upstream(
                tape,
                bundle["transport_matrix"],
                theta_variable,
                clipped,
            )
            for name, clipped in transport_clipping_clipped_upstreams.items()
        }
        del tape
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "boundary_mode": DEFAULT_BAYESFILTER_BOUNDARY_MODE,
            "boundary_mode_description": _boundary_mode_description(
                DEFAULT_BAYESFILTER_BOUNDARY_MODE
            ),
            "settings": filterflow["settings"],
            "values": {
                name: _field(tensor)
                for name, tensor in bundle.items()
                if name not in TARGET_FIELD_EXCLUSIONS
            },
            "value_tensors": {
                name: r3._json(tensor)
                for name, tensor in bundle.items()
                if name not in TARGET_FIELD_EXCLUSIONS
            },
            "gradients": {name: _field(tensor) for name, tensor in gradients.items()},
            "gradient_tensors": {name: r3._json(tensor) for name, tensor in gradients.items()},
            "total_gradient_diag": r3._json(total_gradient),
            "target_scalar": _float(target),
            "resampling_flag": [bool(v) for v in tf.reshape(bundle["flags"], [-1]).numpy().tolist()],
            "transport_upstream_clipped": _field(clipped_transport_upstream),
            "resampling_adjoint_decomposition": {
                "direct_pre_particle_adjoint": _field(direct_pre_particle_adjoint),
                "same_tape_post_particles_vjp": _field(same_tape_post_particles_vjp),
                "same_tape_post_log_weights_vjp": _field(same_tape_post_log_weights_vjp),
                "same_tape_post_state_vjp": _field(same_tape_post_state_vjp),
                "same_tape_post_state_identity_residual": _field(
                    same_tape_post_state_identity_residual
                ),
                "same_tape_pre_log_weights_carryover_vjp": _field(
                    same_tape_pre_log_weights_carryover_vjp
                ),
                "same_tape_pre_current_ll_carryover_vjp": _field(
                    same_tape_pre_current_ll_carryover_vjp
                ),
                "same_tape_log_ess_carryover_vjp": _field(same_tape_log_ess_carryover_vjp),
                "same_tape_full_recorded_state_vjp": _field(same_tape_full_recorded_state_vjp),
                "same_tape_full_recorded_state_residual": _field(
                    same_tape_full_recorded_state_residual
                ),
                "same_tape_transport_matrix_vjp": _field(same_tape_transport_matrix_vjp),
                "same_tape_reconstructed_pre_particle_adjoint": _field(
                    same_tape_reconstructed_pre_particle_adjoint
                ),
                "same_tape_identity_residual": _field(same_tape_identity_residual),
                "implicit_pre_particle_adjoint": _field(implicit_pre_particle_adjoint),
                "carryover_pre_particle_adjoint": _field(carryover_pre_particle_adjoint),
                "current_increment_pre_particle_adjoint": _field(current_increment_pre_particle_adjoint),
                "direct_pre_particle_adjoint_tensor": r3._json(direct_pre_particle_adjoint),
                "same_tape_post_particles_vjp_tensor": r3._json(same_tape_post_particles_vjp),
                "same_tape_post_log_weights_vjp_tensor": r3._json(same_tape_post_log_weights_vjp),
                "same_tape_post_state_vjp_tensor": r3._json(same_tape_post_state_vjp),
                "same_tape_post_state_identity_residual_tensor": r3._json(
                    same_tape_post_state_identity_residual
                ),
                "same_tape_pre_log_weights_carryover_vjp_tensor": r3._json(
                    same_tape_pre_log_weights_carryover_vjp
                ),
                "same_tape_pre_current_ll_carryover_vjp_tensor": r3._json(
                    same_tape_pre_current_ll_carryover_vjp
                ),
                "same_tape_log_ess_carryover_vjp_tensor": r3._json(
                    same_tape_log_ess_carryover_vjp
                ),
                "same_tape_full_recorded_state_vjp_tensor": r3._json(
                    same_tape_full_recorded_state_vjp
                ),
                "same_tape_full_recorded_state_residual_tensor": r3._json(
                    same_tape_full_recorded_state_residual
                ),
                "same_tape_transport_matrix_vjp_tensor": r3._json(same_tape_transport_matrix_vjp),
                "same_tape_reconstructed_pre_particle_adjoint_tensor": r3._json(
                    same_tape_reconstructed_pre_particle_adjoint
                ),
                "same_tape_identity_residual_tensor": r3._json(same_tape_identity_residual),
                "implicit_pre_particle_adjoint_tensor": r3._json(implicit_pre_particle_adjoint),
                "carryover_pre_particle_adjoint_tensor": r3._json(carryover_pre_particle_adjoint),
                "current_increment_pre_particle_adjoint_tensor": r3._json(current_increment_pre_particle_adjoint),
            },
            "local_post_particle_adjoint_probe": {
                name: _field(tensor)
                for name, tensor in local_post_particle_adjoints.items()
            },
            "local_post_particle_adjoint_tensors": {
                name: r3._json(tensor)
                for name, tensor in local_post_particle_adjoints.items()
            },
            "graph_embedding_probe": graph_embedding_probe,
            "parameter_path_adjoint_probe": {
                name: _field(tensor)
                for name, tensor in parameter_path_adjoints.items()
            },
            "parameter_path_adjoint_tensors": {
                name: r3._json(tensor)
                for name, tensor in parameter_path_adjoints.items()
            },
            "scalar_additivity_probe": {
                "contract": (
                    "Direct scalar-gradient additivity check for "
                    "post_update_log_likelihoods = pre_current_log_likelihoods "
                    "+ increment. These gradients use scalar objectives directly, "
                    "not target-upstream VJPs through intermediate tensors."
                ),
                "scalar_values": {
                    name: _float(value)
                    for name, value in scalar_additivity_objectives.items()
                },
                "gradient_summaries": {
                    name: _field(value)
                    for name, value in scalar_additivity_gradients.items()
                },
                "gradient_tensors": {
                    name: r3._json(value)
                    for name, value in scalar_additivity_gradients.items()
                },
            },
            "transport_clipping_probe": {
                "contract": (
                    "Target-time transport custom-gradient clipping probe. "
                    "Raw upstreams are d scalar / d transport_matrix; clipped "
                    "upstreams apply clip_by_value(upstream, -1, 1); VJPs are "
                    "d transport_matrix / d theta under the recorded target-time "
                    "transport graph."
                ),
                "raw_upstream_summaries": {
                    name: _field(value)
                    for name, value in transport_clipping_raw_upstreams.items()
                },
                "raw_upstream_tensors": {
                    name: r3._json(value)
                    for name, value in transport_clipping_raw_upstreams.items()
                },
                "clipped_upstream_summaries": {
                    name: _field(value)
                    for name, value in transport_clipping_clipped_upstreams.items()
                },
                "clipped_upstream_tensors": {
                    name: r3._json(value)
                    for name, value in transport_clipping_clipped_upstreams.items()
                },
                "clip_mask_summaries": {
                    name: _field(value)
                    for name, value in transport_clipping_masks.items()
                },
                "clip_mask_tensors": {
                    name: r3._json(value)
                    for name, value in transport_clipping_masks.items()
                },
                "target_time_vjp_summaries": {
                    name: _field(value)
                    for name, value in transport_clipping_target_time_vjps.items()
                },
                "target_time_vjp_tensors": {
                    name: r3._json(value)
                    for name, value in transport_clipping_target_time_vjps.items()
                },
                "manual_clipped_target_time_vjp_summaries": {
                    name: _field(value)
                    for name, value in transport_clipping_manual_clipped_target_time_vjps.items()
                },
                "manual_clipped_target_time_vjp_tensors": {
                    name: r3._json(value)
                    for name, value in transport_clipping_manual_clipped_target_time_vjps.items()
                },
            },
            "proposal_sample_gradient_contract": proposal_sample_gradient_contract,
            "proposal_topology_probe": proposal_topology_probe,
            "transport_upstream_clip_fraction": _float(
                tf.reduce_mean(
                    tf.cast(tf.abs(gradients["transport_matrix"]) > tf.constant(1.0, DTYPE), DTYPE)
                )
            ),
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_transport_graph_embedding_probe(
    tape: tf.GradientTape,
    bundle: dict[str, tf.Tensor],
    gradients: dict[str, tf.Tensor],
) -> dict[str, Any]:
    upstream = tf.stop_gradient(gradients["transport_matrix"])
    clipped_upstream = tf.clip_by_value(upstream, -1.0, 1.0)

    recorded_raw = _safe_gradient_with_upstream(
        tape,
        bundle["transport_matrix"],
        bundle["pre_particles"],
        upstream,
    )
    recorded_clipped = _safe_gradient_with_upstream(
        tape,
        bundle["transport_matrix"],
        bundle["pre_particles"],
        clipped_upstream,
    )
    local_probe = _local_transport_derivative_probe(
        bundle["pre_particles"],
        bundle["pre_log_weights"],
        upstream,
        clipped_upstream,
    )
    pre_log_weights_vjp = _safe_gradient_with_upstream(
        tape,
        bundle["pre_log_weights"],
        bundle["pre_particles"],
        gradients["pre_log_weights"],
    )
    post_particles_vjp = _safe_gradient_with_upstream(
        tape,
        bundle["post_particles"],
        bundle["pre_particles"],
        gradients["post_particles"],
    )
    tensors = {
        "recorded_transport_vjp_raw_upstream": recorded_raw,
        "recorded_transport_vjp_clipped_upstream": recorded_clipped,
        "local_transport_particles_vjp_raw_upstream": local_probe[
            "particles_vjp_raw_upstream"
        ],
        "local_transport_particles_vjp_clipped_upstream": local_probe[
            "particles_vjp_clipped_upstream"
        ],
        "local_transport_particles_vjp_manual_clipped_upstream": local_probe[
            "particles_vjp_manual_clipped_upstream"
        ],
        "local_transport_log_weights_vjp_raw_upstream": local_probe[
            "log_weights_vjp_raw_upstream"
        ],
        "local_transport_log_weights_vjp_clipped_upstream": local_probe[
            "log_weights_vjp_clipped_upstream"
        ],
        "local_transport_log_weights_vjp_manual_clipped_upstream": local_probe[
            "log_weights_vjp_manual_clipped_upstream"
        ],
        "recorded_minus_local_clipped_particles_vjp": (
            recorded_clipped - local_probe["particles_vjp_clipped_upstream"]
        ),
        "pre_log_weights_to_pre_particles_vjp": pre_log_weights_vjp,
        "post_particles_to_pre_particles_vjp": post_particles_vjp,
    }
    return {
        "contract": (
            "BayesFilter-only same-tape graph embedding probe. Compare the "
            "recorded transport matrix VJP in the full persistent tape with a "
            "fresh local transport derivative that treats pre-particles and "
            "pre-log-weights as independent inputs."
        ),
        "summaries": {name: _field(tensor) for name, tensor in tensors.items()},
        "tensors": {name: r3._json(tensor) for name, tensor in tensors.items()},
    }


def _local_transport_derivative_probe(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    upstream: tf.Tensor,
    clipped_upstream: tf.Tensor,
) -> dict[str, tf.Tensor]:
    local_particles = tf.identity(particles)
    local_log_weights = tf.identity(log_weights)
    with tf.GradientTape(persistent=True) as local_tape:
        local_tape.watch([local_particles, local_log_weights])
        raw_result = annealed_transport_tf.annealed_transport_resample_tf(
            local_particles,
            local_log_weights,
            epsilon=EPSILON,
            scaling=SCALING,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            max_iterations=MAX_ITERATIONS,
            ess_mask=tf.ones([tf.shape(local_particles)[0]], dtype=tf.bool),
            transport_gradient_mode="raw",
            application_mode="filterflow_all_rows",
        )
        clipped_result = annealed_transport_tf.annealed_transport_resample_tf(
            local_particles,
            local_log_weights,
            epsilon=EPSILON,
            scaling=SCALING,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            max_iterations=MAX_ITERATIONS,
            ess_mask=tf.ones([tf.shape(local_particles)[0]], dtype=tf.bool),
            transport_gradient_mode="filterflow_clipped",
            application_mode="filterflow_all_rows",
        )
        raw_transport_matrix = tf.cast(raw_result.transport_matrix, DTYPE)
        clipped_transport_matrix = tf.cast(clipped_result.transport_matrix, DTYPE)
    particles_raw = _safe_gradient_with_upstream(
        local_tape,
        raw_transport_matrix,
        local_particles,
        upstream,
    )
    particles_clipped = _safe_gradient_with_upstream(
        local_tape,
        clipped_transport_matrix,
        local_particles,
        upstream,
    )
    particles_manual_clipped = _safe_gradient_with_upstream(
        local_tape,
        raw_transport_matrix,
        local_particles,
        clipped_upstream,
    )
    log_weights_raw = _safe_gradient_with_upstream(
        local_tape,
        raw_transport_matrix,
        local_log_weights,
        upstream,
    )
    log_weights_clipped = _safe_gradient_with_upstream(
        local_tape,
        clipped_transport_matrix,
        local_log_weights,
        upstream,
    )
    log_weights_manual_clipped = _safe_gradient_with_upstream(
        local_tape,
        raw_transport_matrix,
        local_log_weights,
        clipped_upstream,
    )
    del local_tape
    return {
        "particles_vjp_raw_upstream": particles_raw,
        "particles_vjp_clipped_upstream": particles_clipped,
        "particles_vjp_manual_clipped_upstream": particles_manual_clipped,
        "log_weights_vjp_raw_upstream": log_weights_raw,
        "log_weights_vjp_clipped_upstream": log_weights_clipped,
        "log_weights_vjp_manual_clipped_upstream": log_weights_manual_clipped,
    }

def _model_from_filterflow(filterflow: dict[str, Any]) -> dict[str, Any]:
    return {
        "observations": filterflow["model"]["observations"],
        "initial_particles": filterflow["model"]["initial_particles"],
        "observation_matrix": filterflow["model"]["observation_matrix"],
        "transition_covariance_chol": filterflow["model"]["transition_covariance_chol"],
        "observation_covariance_chol": filterflow["model"]["observation_covariance_chol"],
    }


def _bayesfilter_boundary_modes(filterflow: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    rows = []
    for mode in BOUNDARY_MODES:
        theta_variable = tf.Variable(THETA, dtype=DTYPE)
        model = _model_from_filterflow(filterflow)
        with tf.GradientTape() as tape:
            tape.watch(theta_variable)
            bundle = _bayesfilter_target_bundle(
                theta_variable,
                model,
                config,
                boundary_mode=mode,
            )
            target = bundle["target"]
        total_gradient = _safe_gradient(tape, target, theta_variable)
        rows.append(
            {
                "mode": mode,
                "mode_description": _boundary_mode_description(mode),
                "target_scalar": _float(target),
                "total_gradient_diag": r3._json(total_gradient),
                "finite_scalar": bool(tf.math.is_finite(target).numpy()),
                "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(total_gradient)).numpy()),
                "resampling_flag": [bool(v) for v in tf.reshape(bundle["flags"], [-1]).numpy().tolist()],
            }
        )
    return {
        "status": "executed",
        "boundary_contract": (
            "BayesFilter-only diagnostic stop-gradient modes at the carried-state "
            "and optimal-proposal graph boundaries; values must remain aligned "
            "before any gradient interpretation."
        ),
        "modes": rows,
    }


def _boundary_mode_description(mode: str) -> str:
    descriptions = {
        "raw": (
            "Legacy same-distribution proposal log_prob diagnostic; value-correct "
            "for proposal log probability but not the FilterFlow proposal-density "
            "AD topology."
        ),
        "filterflow_custom_transport_gradient": (
            "Use a whole-transport-call custom gradient matching FilterFlow's "
            "transport(x, logw, ...) backward signature."
        ),
        "carry_log_weights_stop_gradient": "Stop gradient through carried log weights after each update.",
        "carry_log_likelihoods_stop_gradient": "Stop gradient through carried cumulative log likelihoods after each update.",
        "carry_both_stop_gradient": "Stop gradient through both carried log weights and cumulative log likelihoods.",
        "proposal_mean_stop_gradient": "Stop gradient through the optimal-proposal mean at each step.",
        "target_transport_log_weights_stop_gradient": (
            "At the target time only, stop gradient through log weights as an "
            "input to the transport solve."
        ),
        "all_times_transport_log_weights_stop_gradient": (
            "At every time, stop gradient through log weights as an input to "
            "the transport solve."
        ),
        "proposal_sample_noise_stop_gradient": (
            "Keep proposal sample values, but stop the reparameterized sample-noise path "
            "from sampled particles back through proposal mean."
        ),
        "all_times_proposal_sample_filterflow_contract": (
            "At every time, keep proposal sample values but stop the sampled particle "
            "path back through proposal mean while preserving proposal log-probability "
            "dependence on the proposal distribution."
        ),
        "target_proposal_sample_filterflow_contract": (
            "At the target time only, keep proposal sample values but stop the sampled "
            "particle path back through the proposal mean while preserving proposal "
            "log-probability dependence on the proposal distribution."
        ),
        "fresh_proposal_log_prob_filterflow_contract": (
            "Evaluate proposal log probability with a freshly recomputed proposal "
            "distribution, matching executable FilterFlow OptimalProposalModel.loglikelihood."
        ),
        "proposal_sample_stop_gradient": "Stop gradient through sampled proposal particles at each step.",
        "proposal_log_prob_stop_gradient": "Stop gradient through proposal log probability at each step.",
        "carry_both_proposal_sample_stop_gradient": (
            "Stop carried log weights/log likelihoods and sampled proposal particles."
        ),
    }
    if mode not in descriptions:
        raise ValueError(f"unknown boundary mode: {mode}")
    return descriptions[mode]


def _bayesfilter_target_bundle(
    theta: tf.Tensor,
    model: dict[str, Any],
    config: RunConfig,
    *,
    boundary_mode: str = DEFAULT_BAYESFILTER_BOUNDARY_MODE,
) -> dict[str, tf.Tensor]:
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"unknown boundary_mode: {boundary_mode}")
    transition_matrix = localizer._transition_matrix(theta)
    observation_matrix = tf.constant(model["observation_matrix"], dtype=DTYPE)
    transition_chol = tf.constant(model["transition_covariance_chol"], dtype=DTYPE)
    observation_chol = tf.constant(model["observation_covariance_chol"], dtype=DTYPE)
    observations = tf.constant(model["observations"], dtype=DTYPE)
    particles = tf.constant(model["initial_particles"], dtype=DTYPE)
    num_particles = NUM_PARTICLES
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
        tf.linalg.matmul(observation_matrix, observation_cov_inv, transpose_a=True),
        observation_matrix,
    )
    sigma = tf.linalg.inv(sigma_inv)
    sigma_chol = tf.linalg.cholesky(sigma)
    seed = tf.constant(FILTER_SEED, dtype=tf.int32)
    paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
    seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))

    for time_index in range(T):
        seed, _seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
        ess_log = r3._ess(log_weights)
        flags = ess_log <= tf.math.log(
            tf.cast(num_particles, DTYPE) * tf.constant(RESAMPLING_NEFF, DTYPE)
        )
        bool_flags = tf.reshape(flags, [-1])
        transport_log_weights = log_weights
        if (
            boundary_mode == "all_times_transport_log_weights_stop_gradient"
            or (
                boundary_mode == "target_transport_log_weights_stop_gradient"
                and time_index == config.target_time_index
            )
        ):
            transport_log_weights = tf.stop_gradient(transport_log_weights)
        transport_gradient_mode = (
            "filterflow_custom_op"
            if boundary_mode == "filterflow_custom_transport_gradient"
            else "filterflow_clipped"
        )
        transported = annealed_transport_tf.annealed_transport_resample_tf(
            particles,
            transport_log_weights,
            epsilon=EPSILON,
            scaling=SCALING,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            max_iterations=MAX_ITERATIONS,
            ess_mask=bool_flags,
            transport_gradient_mode=transport_gradient_mode,
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
        manual_proposal_mean = proposal_mean
        if boundary_mode == "proposal_mean_stop_gradient":
            proposal_mean = tf.stop_gradient(proposal_mean)
            manual_proposal_mean = proposal_mean
        proposal_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
        proposal_loc = proposal_dist.loc
        proposed_particles = proposal_dist.sample(seed=seed2)
        manual_sample_probe_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
        manual_sample_probe_particles = manual_sample_probe_dist.sample(seed=seed2)
        proposal_dist_log_prob = proposal_dist.log_prob(proposed_particles)
        manual_dist_log_prob = manual_sample_probe_dist.log_prob(proposed_particles)
        fresh_proposal_mean = localizer._optimal_proposal_mean(
            particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
        )
        fresh_proposal_dist = tfd.MultivariateNormalTriL(fresh_proposal_mean, sigma_chol)
        fresh_proposal_loc = fresh_proposal_dist.loc
        fresh_dist_log_prob = fresh_proposal_dist.log_prob(proposed_particles)
        if boundary_mode == "proposal_sample_noise_stop_gradient":
            proposed_particles = proposal_mean + tf.stop_gradient(proposed_particles - proposal_mean)
        if (
            boundary_mode == "all_times_proposal_sample_filterflow_contract"
            or (
                boundary_mode == "target_proposal_sample_filterflow_contract"
                and time_index == config.target_time_index
            )
        ):
            proposed_particles = tf.stop_gradient(proposed_particles)
        if boundary_mode in {
            "proposal_sample_stop_gradient",
            "carry_both_proposal_sample_stop_gradient",
        }:
            proposed_particles = tf.stop_gradient(proposed_particles)
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
        proposal_ll = proposal_dist.log_prob(proposed_particles)
        if boundary_mode == "fresh_proposal_log_prob_filterflow_contract":
            proposal_ll = fresh_dist_log_prob
        if boundary_mode == "proposal_log_prob_stop_gradient":
            proposal_ll = tf.stop_gradient(proposal_ll)
        unnormalized = transition_ll + observation_ll - proposal_ll + log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        pre_current_log_likelihoods = log_likelihoods
        log_likelihoods = log_likelihoods + increment
        normalized = r3._filterflow_normalize(unnormalized, num_particles)
        log_weights = normalized
        if boundary_mode in {
            "carry_log_weights_stop_gradient",
            "carry_both_stop_gradient",
            "carry_both_proposal_sample_stop_gradient",
        }:
            log_weights = tf.stop_gradient(log_weights)
        if boundary_mode in {
            "carry_log_likelihoods_stop_gradient",
            "carry_both_stop_gradient",
            "carry_both_proposal_sample_stop_gradient",
        }:
            log_likelihoods = tf.stop_gradient(log_likelihoods)
        particles = proposed_particles
        if time_index == config.target_time_index:
            target = tf.reduce_mean(log_likelihoods)
            sum_pre_current_plus_increment_mean = tf.reduce_mean(
                pre_current_log_likelihoods + increment
            )
            pre_current_mean = tf.reduce_mean(pre_current_log_likelihoods)
            increment_mean = tf.reduce_mean(increment)
            return {
                "target": target,
                "post_update_mean": target,
                "sum_pre_current_plus_increment_mean": sum_pre_current_plus_increment_mean,
                "pre_current_mean": pre_current_mean,
                "increment_mean": increment_mean,
                "flags": flags,
                "log_ess": ess_log,
                "pre_particles": pre_particles,
                "pre_log_weights": pre_log_weights,
                "transport_matrix": transport_matrix,
                "post_particles": transported.particles,
                "post_log_weights": transported.log_weights,
                "proposal_loc": proposal_loc,
                "proposal_mean": proposal_mean,
                "manual_proposal_mean": manual_proposal_mean,
                "fresh_proposal_loc": fresh_proposal_loc,
                "fresh_proposal_mean": fresh_proposal_mean,
                "proposed_particles": proposed_particles,
                "manual_sample_probe_particles": manual_sample_probe_particles,
                "observation_ll": observation_ll,
                "transition_ll": transition_ll,
                "proposal_ll": proposal_ll,
                "proposal_dist_log_prob": proposal_dist_log_prob,
                "manual_dist_log_prob": manual_dist_log_prob,
                "fresh_dist_log_prob": fresh_dist_log_prob,
                "unnormalized": unnormalized,
                "increment": increment,
                "pre_current_log_likelihoods": pre_current_log_likelihoods,
                "normalized": normalized,
                "post_update_log_weights": log_weights,
                "post_update_log_likelihoods": log_likelihoods,
            }
    raise RuntimeError("target time not reached")


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


def _compare(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    if bayesfilter.get("status") != "executed":
        return {"status": "blocked", "blocker": "BayesFilter VJP did not execute"}
    value_deltas = _field_deltas(
        filterflow["values"],
        bayesfilter["values"],
        filterflow.get("value_tensors", {}),
        bayesfilter.get("value_tensors", {}),
    )
    gradient_deltas = _field_deltas(
        filterflow["gradients"],
        bayesfilter["gradients"],
        filterflow.get("gradient_tensors", {}),
        bayesfilter.get("gradient_tensors", {}),
    )
    total_gradient_delta = [
        float(bf) - float(ff)
        for bf, ff in zip(bayesfilter["total_gradient_diag"], filterflow["total_gradient_diag"], strict=True)
    ]
    scalar_delta = abs(float(bayesfilter["target_scalar"]) - float(filterflow["target_scalar"]))
    first_value_delta = _first_delta(value_deltas, VALUE_TOLERANCE)
    first_gradient_delta = _first_delta(gradient_deltas, GRADIENT_TOLERANCE)
    adjoint_decomposition = _compare_adjoint_decomposition(filterflow, bayesfilter)
    local_post_particle_adjoint = _compare_local_post_particle_adjoints(filterflow, bayesfilter)
    parameter_path_adjoint = _compare_parameter_path_adjoints(filterflow, bayesfilter)
    proposal_sample_gradient_contract = _compare_proposal_sample_gradient_contract(
        filterflow,
        bayesfilter,
    )
    proposal_topology_probe = _compare_proposal_topology_probe(filterflow, bayesfilter)
    filterflow_proposal_mean_internal = filterflow.get("filterflow_proposal_mean_internal_probe")
    return {
        "status": "compared",
        "scalar_delta": scalar_delta,
        "total_gradient_delta": total_gradient_delta,
        "max_abs_total_gradient_delta": max(abs(value) for value in total_gradient_delta),
        "value_deltas": value_deltas,
        "gradient_deltas": gradient_deltas,
        "first_value_delta_over_tolerance": first_value_delta,
        "first_gradient_delta_over_tolerance": first_gradient_delta,
        "transport_upstream_clip_fraction_delta": abs(
            float(bayesfilter["transport_upstream_clip_fraction"])
            - float(filterflow["transport_upstream_clip_fraction"])
        ),
        "adjoint_decomposition": adjoint_decomposition,
        "local_post_particle_adjoint": local_post_particle_adjoint,
        "parameter_path_adjoint": parameter_path_adjoint,
        "proposal_sample_gradient_contract": proposal_sample_gradient_contract,
        "proposal_topology_probe": proposal_topology_probe,
        "filterflow_proposal_mean_internal_probe": filterflow_proposal_mean_internal,
        "resampling_flags_match": bayesfilter["resampling_flag"] == filterflow["resampling_flag"],
        "interpretation": _interpret_delta(first_value_delta, first_gradient_delta),
    }


def _compare_parameter_path_adjoints(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_fields = filterflow.get("parameter_path_adjoint_probe", {})
    bf_fields = bayesfilter.get("parameter_path_adjoint_probe", {})
    ff_tensors = filterflow.get("parameter_path_adjoint_tensors", {})
    bf_tensors = bayesfilter.get("parameter_path_adjoint_tensors", {})
    rows = {}
    for name in ff_fields:
        bf = bf_fields.get(name)
        if bf is None:
            rows[name] = {
                "shape_match": False,
                "finite": False,
                "max_abs_delta": None,
                "sum_delta": None,
                "status": "missing_bayesfilter_field",
            }
            continue
        rows[name] = {
            "shape_match": ff_fields[name]["shape"] == bf["shape"],
            "finite": bool(ff_fields[name]["finite"] and bf["finite"]),
            "filterflow_max_abs": ff_fields[name]["max_abs"],
            "bayesfilter_max_abs": bf["max_abs"],
            "max_abs_delta": _max_abs_nested_delta(
                bf_tensors[name],
                ff_tensors[name],
            ),
            "sum_delta": _sum_nested_delta(
                bf_tensors[name],
                ff_tensors[name],
            ),
            "status": "compared",
        }
    first_delta = _first_delta(rows, GRADIENT_TOLERANCE)
    return {
        "status": "compared",
        "rows": rows,
        "first_delta_over_tolerance": first_delta,
        "interpretation": (
            "parameter_path_adjoints_match"
            if first_delta["status"] == "no_delta"
            else f"first_parameter_path_delta_{first_delta['field']}"
        ),
    }


def _compare_local_post_particle_adjoints(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_tensors = filterflow.get("local_post_particle_adjoint_tensors", {})
    bf_tensors = bayesfilter.get("local_post_particle_adjoint_tensors", {})
    rows = {}
    for name in ff_tensors:
        if name not in bf_tensors:
            rows[name] = {"status": "missing_bayesfilter_tensor"}
            continue
        rows[name] = {
            "status": "compared",
            "max_abs_delta": _max_abs_nested_delta(bf_tensors[name], ff_tensors[name]),
            "filterflow_max_abs": filterflow["local_post_particle_adjoint_probe"][name]["max_abs"],
            "bayesfilter_max_abs": bayesfilter["local_post_particle_adjoint_probe"][name]["max_abs"],
            "sum_delta": _sum_nested_delta(bf_tensors[name], ff_tensors[name]),
        }
    first_delta = _first_delta(
        {
            name: {
                "shape_match": row.get("status") == "compared",
                "finite": row.get("status") == "compared",
                "max_abs_delta": row.get("max_abs_delta"),
                "sum_delta": row.get("sum_delta"),
            }
            for name, row in rows.items()
        },
        GRADIENT_TOLERANCE,
    )
    return {
        "status": "compared",
        "rows": rows,
        "first_delta_over_tolerance": first_delta,
        "interpretation": (
            "local_post_particle_adjoints_match"
            if first_delta["status"] == "no_delta"
            else f"first_local_post_particle_adjoint_delta_{first_delta['field']}"
        ),
    }


def _compare_proposal_sample_gradient_contract(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_contract = filterflow.get("proposal_sample_gradient_contract")
    bf_contract = bayesfilter.get("proposal_sample_gradient_contract")
    if ff_contract is None or bf_contract is None:
        return {
            "status": "missing",
            "filterflow_present": ff_contract is not None,
            "bayesfilter_present": bf_contract is not None,
        }

    value_rows = _nested_delta_rows(
        ff_contract.get("value_probe", {}),
        bf_contract.get("value_probe", {}),
        ff_contract.get("value_probe_tensors", {}),
        bf_contract.get("value_probe_tensors", {}),
    )
    vjp_rows = _nested_delta_rows(
        ff_contract.get("vjp_probe", {}),
        bf_contract.get("vjp_probe", {}),
        ff_contract.get("vjp_tensors", {}),
        bf_contract.get("vjp_tensors", {}),
    )
    first_value_delta = _first_delta(value_rows, VALUE_TOLERANCE)
    first_vjp_delta = _first_delta(vjp_rows, GRADIENT_TOLERANCE)
    return {
        "status": "compared",
        "filterflow_contract": ff_contract.get("contract"),
        "bayesfilter_contract": bf_contract.get("contract"),
        "value_rows": value_rows,
        "vjp_rows": vjp_rows,
        "first_value_delta_over_tolerance": first_value_delta,
        "first_vjp_delta_over_tolerance": first_vjp_delta,
        "interpretation": (
            "proposal_sample_gradient_contract_matches"
            if first_value_delta["status"] == "no_delta"
            and first_vjp_delta["status"] == "no_delta"
            else "proposal_sample_gradient_contract_differs"
        ),
    }


def _compare_proposal_topology_probe(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_probe = filterflow.get("proposal_topology_probe")
    bf_probe = bayesfilter.get("proposal_topology_probe")
    if ff_probe is None or bf_probe is None:
        return {
            "status": "missing",
            "filterflow_present": ff_probe is not None,
            "bayesfilter_present": bf_probe is not None,
        }
    value_rows = _nested_delta_rows(
        ff_probe.get("value_summaries", {}),
        bf_probe.get("value_summaries", {}),
        ff_probe.get("value_tensors", {}),
        bf_probe.get("value_tensors", {}),
    )
    gradient_rows = _nested_delta_rows(
        ff_probe.get("gradient_summaries", {}),
        bf_probe.get("gradient_summaries", {}),
        ff_probe.get("gradient_tensors", {}),
        bf_probe.get("gradient_tensors", {}),
    )
    log_prob_parameter_rows = _nested_delta_rows(
        ff_probe.get("proposal_log_prob_parameter_path_summaries", {}),
        bf_probe.get("proposal_log_prob_parameter_path_summaries", {}),
        ff_probe.get("proposal_log_prob_parameter_path_tensors", {}),
        bf_probe.get("proposal_log_prob_parameter_path_tensors", {}),
    )
    first_value_delta = _first_delta(value_rows, VALUE_TOLERANCE)
    first_gradient_delta = _first_delta(gradient_rows, GRADIENT_TOLERANCE)
    first_log_prob_parameter_delta = _first_delta(log_prob_parameter_rows, GRADIENT_TOLERANCE)
    return {
        "status": "compared",
        "filterflow_contract": ff_probe.get("contract"),
        "bayesfilter_contract": bf_probe.get("contract"),
        "value_rows": value_rows,
        "gradient_rows": gradient_rows,
        "proposal_log_prob_parameter_rows": log_prob_parameter_rows,
        "first_value_delta_over_tolerance": first_value_delta,
        "first_gradient_delta_over_tolerance": first_gradient_delta,
        "first_log_prob_parameter_delta_over_tolerance": first_log_prob_parameter_delta,
        "interpretation": (
            "proposal_topology_probe_matches"
            if first_value_delta["status"] == "no_delta"
            and first_gradient_delta["status"] == "no_delta"
            and first_log_prob_parameter_delta["status"] == "no_delta"
            else "proposal_topology_probe_differs"
        ),
    }


def _nested_delta_rows(
    filterflow_fields: dict[str, dict[str, Any]],
    bayesfilter_fields: dict[str, dict[str, Any]],
    filterflow_tensors: dict[str, Any],
    bayesfilter_tensors: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = {}
    for name, ff in filterflow_fields.items():
        bf = bayesfilter_fields.get(name)
        if bf is None:
            rows[name] = {
                "shape_match": False,
                "finite": False,
                "max_abs_delta": None,
                "sum_delta": None,
                "status": "missing_bayesfilter_field",
            }
            continue
        rows[name] = {
            "shape_match": ff["shape"] == bf["shape"],
            "finite": bool(ff["finite"] and bf["finite"]),
            "filterflow_max_abs": ff["max_abs"],
            "bayesfilter_max_abs": bf["max_abs"],
            "max_abs_delta": _max_abs_nested_delta(
                bayesfilter_tensors[name],
                filterflow_tensors[name],
            ),
            "sum_delta": _sum_nested_delta(
                bayesfilter_tensors[name],
                filterflow_tensors[name],
            ),
            "status": "compared",
        }
    return rows


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


def _compare_boundary_modes(
    filterflow: dict[str, Any],
    bayesfilter_boundary_modes: dict[str, Any],
) -> dict[str, Any]:
    if bayesfilter_boundary_modes.get("status") != "executed":
        return {
            "status": "blocked",
            "blocker": "BayesFilter boundary modes did not execute",
        }
    rows = []
    for row in bayesfilter_boundary_modes["modes"]:
        gradient_delta = [
            float(bf) - float(ff)
            for bf, ff in zip(row["total_gradient_diag"], filterflow["total_gradient_diag"], strict=True)
        ]
        max_abs_gradient_delta = max(abs(value) for value in gradient_delta)
        scalar_delta = abs(float(row["target_scalar"]) - float(filterflow["target_scalar"]))
        rows.append(
            {
                "mode": row["mode"],
                "mode_description": row["mode_description"],
                "scalar_delta": scalar_delta,
                "gradient_delta": gradient_delta,
                "max_abs_gradient_delta": max_abs_gradient_delta,
                "scalar_within_tolerance": scalar_delta <= VALUE_TOLERANCE,
                "gradient_within_tolerance": max_abs_gradient_delta <= GRADIENT_TOLERANCE,
                "finite_scalar": row["finite_scalar"],
                "finite_gradient": row["finite_gradient"],
                "resampling_flag": row["resampling_flag"],
                "bayesfilter_gradient_diag": row["total_gradient_diag"],
            }
        )
    value_valid_rows = [
        row for row in rows if row["scalar_within_tolerance"] and row["finite_scalar"] and row["finite_gradient"]
    ]
    best_value_valid = (
        min(value_valid_rows, key=lambda row: row["max_abs_gradient_delta"])
        if value_valid_rows
        else None
    )
    matching_modes = [row for row in value_valid_rows if row["gradient_within_tolerance"]]
    return {
        "status": "compared",
        "filterflow_target_scalar": filterflow["target_scalar"],
        "filterflow_gradient_diag": filterflow["total_gradient_diag"],
        "rows": rows,
        "best_value_valid_mode": best_value_valid,
        "matching_modes": matching_modes,
        "interpretation": _interpret_boundary_modes(best_value_valid, matching_modes),
    }


def _interpret_boundary_modes(
    best_value_valid: dict[str, Any] | None,
    matching_modes: list[dict[str, Any]],
) -> str:
    if matching_modes:
        return f"boundary_mode_{matching_modes[0]['mode']}_collapses_row_gradient_delta"
    if best_value_valid is None:
        return "all_boundary_modes_vetoed_by_value_or_finiteness"
    if best_value_valid["max_abs_gradient_delta"] < 1.0:
        return f"boundary_mode_{best_value_valid['mode']}_materially_reduces_but_does_not_close_delta"
    return "tested_boundary_modes_do_not_explain_row_gradient_delta"


def _compare_adjoint_decomposition(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_decomp = filterflow.get("resampling_adjoint_decomposition", {})
    bf_decomp = bayesfilter.get("resampling_adjoint_decomposition", {})
    tensor_keys = [
        "direct_pre_particle_adjoint",
        "same_tape_post_particles_vjp",
        "same_tape_post_log_weights_vjp",
        "same_tape_post_state_vjp",
        "same_tape_post_state_identity_residual",
        "same_tape_pre_log_weights_carryover_vjp",
        "same_tape_pre_current_ll_carryover_vjp",
        "same_tape_log_ess_carryover_vjp",
        "same_tape_full_recorded_state_vjp",
        "same_tape_full_recorded_state_residual",
        "same_tape_transport_matrix_vjp",
        "same_tape_reconstructed_pre_particle_adjoint",
        "same_tape_identity_residual",
        "implicit_pre_particle_adjoint",
        "carryover_pre_particle_adjoint",
        "current_increment_pre_particle_adjoint",
    ]
    rows = {}
    for key in tensor_keys:
        ff_tensor = ff_decomp.get(f"{key}_tensor")
        bf_tensor = bf_decomp.get(f"{key}_tensor")
        if ff_tensor is None or bf_tensor is None:
            rows[key] = {
                "status": "missing",
                "filterflow_present": ff_tensor is not None,
                "bayesfilter_present": bf_tensor is not None,
            }
        else:
            rows[key] = {
                "status": "compared",
                "max_abs_delta": _max_abs_nested_delta(bf_tensor, ff_tensor),
                "filterflow_max_abs": ff_decomp[key]["max_abs"],
                "bayesfilter_max_abs": bf_decomp[key]["max_abs"],
            }
    ff_identity = ff_decomp.get("same_tape_identity_residual", {})
    bf_identity = bf_decomp.get("same_tape_identity_residual", {})
    ff_post_state_identity = ff_decomp.get("same_tape_post_state_identity_residual", {})
    bf_post_state_identity = bf_decomp.get("same_tape_post_state_identity_residual", {})
    ff_full_recorded_identity = ff_decomp.get("same_tape_full_recorded_state_residual", {})
    bf_full_recorded_identity = bf_decomp.get("same_tape_full_recorded_state_residual", {})
    ff_identity_max = ff_identity.get("max_abs")
    bf_identity_max = bf_identity.get("max_abs")
    ff_post_state_identity_max = ff_post_state_identity.get("max_abs")
    bf_post_state_identity_max = bf_post_state_identity.get("max_abs")
    ff_full_recorded_identity_max = ff_full_recorded_identity.get("max_abs")
    bf_full_recorded_identity_max = bf_full_recorded_identity.get("max_abs")
    return {
        "rows": rows,
        "filterflow_same_tape_identity_max_abs": ff_identity_max,
        "bayesfilter_same_tape_identity_max_abs": bf_identity_max,
        "filterflow_post_state_identity_max_abs": ff_post_state_identity_max,
        "bayesfilter_post_state_identity_max_abs": bf_post_state_identity_max,
        "filterflow_full_recorded_state_identity_max_abs": ff_full_recorded_identity_max,
        "bayesfilter_full_recorded_state_identity_max_abs": bf_full_recorded_identity_max,
        "filterflow_same_tape_identity_holds": (
            ff_identity_max is not None and float(ff_identity_max) <= GRADIENT_TOLERANCE
        ),
        "bayesfilter_same_tape_identity_holds": (
            bf_identity_max is not None and float(bf_identity_max) <= GRADIENT_TOLERANCE
        ),
        "filterflow_post_state_identity_holds": (
            ff_post_state_identity_max is not None
            and float(ff_post_state_identity_max) <= GRADIENT_TOLERANCE
        ),
        "bayesfilter_post_state_identity_holds": (
            bf_post_state_identity_max is not None
            and float(bf_post_state_identity_max) <= GRADIENT_TOLERANCE
        ),
        "filterflow_full_recorded_state_identity_holds": (
            ff_full_recorded_identity_max is not None
            and float(ff_full_recorded_identity_max) <= GRADIENT_TOLERANCE
        ),
        "bayesfilter_full_recorded_state_identity_holds": (
            bf_full_recorded_identity_max is not None
            and float(bf_full_recorded_identity_max) <= GRADIENT_TOLERANCE
        ),
        "same_tape_identity_contract": (
            "pre_particles_adjoint == T^T post_particles_adjoint "
            "+ VJP(transport_matrix wrt pre_particles)"
        ),
        "post_state_identity_contract": (
            "pre_particles_adjoint == VJP(post_particles wrt pre_particles) "
            "+ VJP(post_log_weights wrt pre_particles)"
        ),
        "full_recorded_state_identity_contract": (
            "pre_particles_adjoint == post-state VJP plus recorded carryover VJPs "
            "through pre_log_weights, pre_current_log_likelihoods, and log_ess"
        ),
    }


def _field_deltas(
    filterflow_fields: dict[str, dict[str, Any]],
    bayesfilter_fields: dict[str, dict[str, Any]],
    filterflow_tensors: dict[str, Any],
    bayesfilter_tensors: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = {}
    for name in filterflow_fields:
        ff = filterflow_fields[name]
        bf = bayesfilter_fields[name]
        tensor_delta = None
        if name in filterflow_tensors and name in bayesfilter_tensors:
            tensor_delta = _max_abs_nested_delta(bayesfilter_tensors[name], filterflow_tensors[name])
        rows[name] = {
            "filterflow_max_abs": ff["max_abs"],
            "bayesfilter_max_abs": bf["max_abs"],
            "max_abs_delta": tensor_delta
            if tensor_delta is not None
            else None
            if ff["max_abs"] is None or bf["max_abs"] is None
            else abs(float(bf["max_abs"]) - float(ff["max_abs"])),
            "sum_delta": None
            if ff["sum"] is None or bf["sum"] is None
            else abs(float(bf["sum"]) - float(ff["sum"])),
            "shape_match": ff["shape"] == bf["shape"],
            "finite": bool(ff["finite"] and bf["finite"]),
        }
    return rows


def _max_abs_nested_delta(left: Any, right: Any) -> float:
    deltas: list[float] = []

    def visit(lhs: Any, rhs: Any) -> None:
        if isinstance(lhs, list) and isinstance(rhs, list):
            for lhs_item, rhs_item in zip(lhs, rhs, strict=True):
                visit(lhs_item, rhs_item)
        else:
            deltas.append(abs(float(lhs) - float(rhs)))

    visit(left, right)
    return max(deltas) if deltas else 0.0


def _first_delta(rows: dict[str, dict[str, Any]], tolerance: float) -> dict[str, Any]:
    for name, row in rows.items():
        max_abs_delta = row["max_abs_delta"]
        sum_delta = row["sum_delta"]
        if (
            not row["shape_match"]
            or not row["finite"]
            or (max_abs_delta is not None and max_abs_delta > tolerance)
            or (sum_delta is not None and sum_delta > tolerance)
        ):
            return {"status": "delta", "field": name, "row": row, "tolerance": tolerance}
    return {"status": "no_delta", "tolerance": tolerance}


def _interpret_delta(
    first_value_delta: dict[str, Any],
    first_gradient_delta: dict[str, Any],
) -> str:
    if first_value_delta["status"] == "delta":
        return "value_path_difference_before_vjp_interpretation"
    if first_gradient_delta["status"] == "delta":
        return f"first_vjp_difference_field_{first_gradient_delta['field']}"
    return "no_vjp_difference_detected_at_recorded_fields"


def _decision(comparison: dict[str, Any], comparator_drift: bool) -> str:
    if comparator_drift:
        return "filterflow_float64_row_173_vjp_blocked_by_comparator_drift"
    if comparison.get("status") != "compared":
        return "filterflow_float64_row_173_vjp_blocked"
    if not comparison.get("resampling_flags_match", False):
        return "filterflow_float64_row_173_vjp_resampling_flag_mismatch"
    if comparison["scalar_delta"] > VALUE_TOLERANCE:
        return "filterflow_float64_row_173_vjp_scalar_veto"
    if comparison["first_value_delta_over_tolerance"]["status"] == "delta":
        return "filterflow_float64_row_173_vjp_value_difference_localized"
    if comparison["max_abs_total_gradient_delta"] <= GRADIENT_TOLERANCE:
        return "filterflow_float64_row_173_vjp_total_gradient_match"
    if comparison["first_gradient_delta_over_tolerance"]["status"] == "delta":
        return "filterflow_float64_row_173_vjp_gradient_difference_localized"
    return "filterflow_float64_row_173_vjp_no_difference_detected"


def _evidence_contract() -> dict[str, Any]:
    return {
        "primary_comparator": "local executable float64 filterflow checkout",
        "primary_question": "row_173_time_1_clipped_default_vjp_difference_localization",
        "primary_pass": "identify first value/VJP field that differs across implementations",
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "finite_vjps": "veto_gate_not_correctness_claim",
        "mathematical_correctness": "not_concluded",
    }


def _model_contract(config: RunConfig) -> dict[str, Any]:
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "mesh_index": MESH_INDEX,
        "theta": THETA,
        "target_time_index": config.target_time_index,
        "artifact_tag": config.tag,
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
        "target_scalar": side["target_scalar"],
        "total_gradient_diag": side["total_gradient_diag"],
        "resampling_flag": side["resampling_flag"],
        "transport_upstream_clip_fraction": side["transport_upstream_clip_fraction"],
        "resampling_adjoint_decomposition": side.get("resampling_adjoint_decomposition"),
        "local_post_particle_adjoint_probe": side.get("local_post_particle_adjoint_probe"),
        "graph_embedding_probe": side.get("graph_embedding_probe"),
        "parameter_path_adjoint_probe": side.get("parameter_path_adjoint_probe"),
        "filterflow_proposal_mean_internal_probe": side.get("filterflow_proposal_mean_internal_probe"),
        "proposal_sample_gradient_contract": side.get("proposal_sample_gradient_contract"),
        "proposal_topology_probe": side.get("proposal_topology_probe"),
        "value_summaries": side["values"],
        "gradient_summaries": side["gradients"],
        "cpu_only_manifest": side["cpu_only_manifest"],
        "stderr_excerpt": side.get("stderr_excerpt", ""),
    }


def _decision_table(decision: str, comparison: dict[str, Any]) -> list[dict[str, str]]:
    if decision == "filterflow_float64_row_173_vjp_gradient_difference_localized":
        primary = f"first VJP delta: {comparison.get('first_gradient_delta_over_tolerance')}"
        next_action = "inspect exact arithmetic for the first VJP-delta field and patch only if the executable FilterFlow rule is unambiguous"
        veto = "scalar/value path stayed within tolerance"
    elif decision == "filterflow_float64_row_173_vjp_total_gradient_match":
        primary = (
            "time-1 scalar and total gradient match; intermediate VJP deltas "
            "are explanatory graph-structure diagnostics"
        )
        next_action = "run a same-contract full-path cumulative gradient scan to find the first true residual time"
        veto = "none"
    elif decision == "filterflow_float64_row_173_vjp_value_difference_localized":
        primary = f"first value delta: {comparison.get('first_value_delta_over_tolerance')}"
        next_action = "repair value-path replay before using VJP evidence"
        veto = "value mismatch"
    elif decision == "filterflow_float64_row_173_vjp_no_difference_detected":
        primary = "no recorded field exceeded tolerance"
        next_action = "add narrower tensor-level VJP fields or rerun cumulative gradient localizer"
        veto = "none"
    else:
        primary = comparison.get("blocker", decision)
        next_action = "repair blocker before interpreting VJP evidence"
        veto = decision
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": veto,
            "main_uncertainty": "single row and single time index; no correctness claim",
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
        "filterflow_vjp",
        "bayesfilter_vjp",
        "comparison",
        "bayesfilter_boundary_modes",
        "boundary_mode_comparison",
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
        "filterflow_float64_row_173_vjp_filterflow_blocker",
        "filterflow_float64_row_173_vjp_blocked_by_comparator_drift",
        "filterflow_float64_row_173_vjp_blocked",
        "filterflow_float64_row_173_vjp_resampling_flag_mismatch",
        "filterflow_float64_row_173_vjp_scalar_veto",
        "filterflow_float64_row_173_vjp_value_difference_localized",
        "filterflow_float64_row_173_vjp_gradient_difference_localized",
        "filterflow_float64_row_173_vjp_total_gradient_match",
        "filterflow_float64_row_173_vjp_no_difference_detected",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    for label in ("filterflow_vjp", "bayesfilter_vjp"):
        side = payload.get(label)
        if side is not None and side.get("status") == "executed":
            _validate_cpu(side["cpu_only_manifest"], label)
    if payload["comparison"].get("status") == "compared":
        if payload["comparison"]["scalar_delta"] > VALUE_TOLERANCE and "scalar_veto" not in payload["decision"]:
            raise ValueError("scalar delta exceeded tolerance without scalar veto")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: FilterFlow Float64 Row 173 VJP Decomposition",
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
            "## Boundary Mode Comparison",
            "",
            _json_block(payload["boundary_mode_comparison"]),
            "",
            "## BayesFilter Boundary Modes",
            "",
            _json_block(payload["bayesfilter_boundary_modes"]),
            "",
            "## FilterFlow VJP",
            "",
            _json_block(payload["filterflow_vjp"]),
            "",
            "## BayesFilter VJP",
            "",
            _json_block(payload["bayesfilter_vjp"]),
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


def _non_implications() -> list[str]:
    return step._non_implications() + [
        "No correctness claim is made for either implementation.",
        "No analytic smoothness-gradient correctness is concluded.",
        "No full mesh_size=20 surface agreement is concluded.",
        "No production dtype default is concluded.",
        "Finite VJPs alone are smoke evidence only.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
