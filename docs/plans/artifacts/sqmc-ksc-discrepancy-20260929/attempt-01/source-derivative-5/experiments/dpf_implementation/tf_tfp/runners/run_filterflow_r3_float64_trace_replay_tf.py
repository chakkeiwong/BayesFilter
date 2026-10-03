"""R3 float64 proposal-stream trace/replay against executable filterflow.

This runner uses a non-mutating external filterflow trace loop. The trace is
evidence only if it first reproduces filterflow's official state-series output.
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

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_agreement_ladder_tf as agreement,
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
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_BRANCH_MARKER,
    FILTERFLOW_REFERENCE_COMMIT,
    FILTERFLOW_REFERENCE_DTYPE,
    reference_policy,
    validate_filterflow_reference_status,
)


tfd = tfp.distributions

DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-r3-float64-trace-replay-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-r3-float64-trace-replay-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_r3_float64_trace_replay_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-r3-float64-trace-replay-2026-06-03.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

T = 100
NUM_PARTICLES = step.NUM_PARTICLES
DATA_SEED = 123
FILTER_SEED = 1234
EPSILON = step.EPSILON
SCALING = 0.85
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 500
RESAMPLING_NEFF = 0.9999
TRACE_TOLERANCE = 5e-5


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
    reference_status = _filterflow_status()
    validate_filterflow_reference_status(reference_status, marker_path=FILTERFLOW_MARKER_PATH)
    initial_fingerprint = continuation._filterflow_fingerprint()
    filterflow_trace = _filterflow_trace_subprocess()
    if filterflow_trace.get("status") != "executed":
        return _blocked_payload(
            "filterflow_r3_float64_trace_replay_filterflow_blocker",
            filterflow_trace.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow_trace,
            reference_status,
        )
    if not filterflow_trace["trace_validation"]["official_trace_match"]:
        return _blocked_payload(
            "filterflow_r3_float64_trace_replay_trace_validation_failed",
            "external trace loop did not reproduce official filterflow output",
            initial_fingerprint,
            filterflow_trace,
            reference_status,
        )

    computed_replay = _bayesfilter_replay(
        filterflow_trace,
        replay_mode="computed_resampling_state",
    )
    computed_comparison = _compare_replay(computed_replay, filterflow_trace)
    traced_replay = _bayesfilter_replay(
        filterflow_trace,
        replay_mode="traced_resampling_state",
    )
    traced_comparison = _compare_replay(traced_replay, filterflow_trace)
    traced_input_computed_replay = _bayesfilter_replay(
        filterflow_trace,
        replay_mode="traced_input_computed_resampling_state",
    )
    traced_input_computed_comparison = _compare_replay(
        traced_input_computed_replay,
        filterflow_trace,
    )
    traced_transport_replay = _bayesfilter_replay(
        filterflow_trace,
        replay_mode="traced_input_traced_transport_matrix",
    )
    traced_transport_comparison = _compare_replay(
        traced_transport_replay,
        filterflow_trace,
    )
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(
        computed_comparison,
        traced_comparison,
        traced_input_computed_comparison,
        traced_transport_comparison,
        comparator_drift,
    )
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": (
            "Can BayesFilter match filterflow through the R3 proposal stream "
            "when proposal particles are traced from executable filterflow?"
        ),
        "evidence_contract": {
            "primary_comparator": "local executable float64 filterflow checkout",
            "primary_question": "cross_implementation_difference_only",
            "trace_must_match_official_filterflow_first": True,
            "mathematical_correctness": "not_concluded",
        },
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "model_contract": _model_contract(),
        "filterflow_trace": _compact_trace(filterflow_trace),
        "bayesfilter_replay": _compact_replay(computed_replay),
        "comparison": computed_comparison,
        "computed_resampling_replay": _compact_replay(computed_replay),
        "computed_resampling_comparison": computed_comparison,
        "traced_resampling_replay": _compact_replay(traced_replay),
        "traced_resampling_comparison": traced_comparison,
        "traced_input_computed_resampling_replay": _compact_replay(
            traced_input_computed_replay,
        ),
        "traced_input_computed_resampling_comparison": traced_input_computed_comparison,
        "traced_transport_matrix_replay": _compact_replay(traced_transport_replay),
        "traced_transport_matrix_comparison": traced_transport_comparison,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r3_float64_trace_replay_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications()
        + [
            "No correctness claim is made for either implementation.",
            "No smoothness-surface gradient correctness is concluded.",
            "No production dtype default is concluded.",
        ],
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
    filterflow_trace: dict[str, Any],
    reference_status: dict[str, Any],
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "blocker": blocker,
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "model_contract": _model_contract(),
        "filterflow_trace": _compact_trace(filterflow_trace),
        "bayesfilter_replay": None,
        "comparison": {"status": "blocked", "blocker": blocker},
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_r3_float64_trace_replay_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _filterflow_trace_subprocess() -> dict[str, Any]:
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
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_trace_script()],
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
            "blocker": "filterflow trace subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_R3_TRACE_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_R3_TRACE_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow R3 trace JSON sentinels missing",
            "stdout_excerpt": stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(stdout[start + len("FILTERFLOW_R3_TRACE_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_trace_script() -> str:
    return textwrap.dedent(
        f"""
        import attr
        import inspect
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import numpy as np
        import tensorflow as tf
        from tensorflow_probability.python.internal import samplers

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from filterflow.resampling.differentiable.regularized_transport.plan import transport
        from filterflow.utils import normalize
        from scripts.simple_linear_smoothness import get_data

        T = {T}
        N = {NUM_PARTICLES}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}
        EPSILON = {EPSILON!r}
        SCALING = {SCALING!r}
        CONVERGENCE_THRESHOLD = {CONVERGENCE_THRESHOLD!r}
        MAX_ITERATIONS = {MAX_ITERATIONS}
        RESAMPLING_NEFF = {RESAMPLING_NEFF!r}
        DTYPE = tf.float64
        NP_DTYPE = np.float64

        def to_json(tensor):
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, tf.float64).numpy())

        def max_abs(left, right):
            return scalar(tf.reduce_max(tf.abs(tf.cast(left, tf.float64) - tf.cast(right, tf.float64))))

        transition_matrix_np = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=NP_DTYPE)
        observation_matrix_np = np.array([[1.0, 0.0]], dtype=NP_DTYPE)
        transition_covariance_np = np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]], dtype=NP_DTYPE)
        observation_covariance_np = np.array([[0.01]], dtype=NP_DTYPE)

        np_random_state = np.random.RandomState(seed=DATA_SEED)
        observations_np, _kf = get_data(
            transition_matrix_np,
            observation_matrix_np,
            transition_covariance_np,
            observation_covariance_np,
            T=T,
            random_state=np_random_state,
        )
        initial_particles_np = np_random_state.normal(0.0, 0.01, [1, N, 2]).astype(NP_DTYPE)

        transition_matrix = tf.Variable(transition_matrix_np, trainable=False)
        observation_matrix = tf.convert_to_tensor(observation_matrix_np)
        transition_covariance_chol = tf.linalg.cholesky(tf.convert_to_tensor(transition_covariance_np))
        observation_covariance_chol = tf.linalg.cholesky(tf.convert_to_tensor(observation_covariance_np))
        resampling_method = RegularisedTransform(
            epsilon=EPSILON,
            scaling=SCALING,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            max_iter=MAX_ITERATIONS,
        )
        resampling_criterion = NeffCriterion(RESAMPLING_NEFF, True)
        pf = make_filter(
            observation_matrix,
            transition_matrix,
            observation_covariance_chol,
            transition_covariance_chol,
            resampling_method,
            resampling_criterion,
            optimal_proposal=True,
        )

        initial_state = State(initial_particles_np)
        observations = tf.convert_to_tensor(observations_np, dtype=DTYPE)
        observation_dataset = tf.data.Dataset.from_tensor_slices(observations)
        official = pf(
            initial_state,
            observation_dataset,
            n_observations=tf.constant(T),
            return_final=False,
            seed=tf.constant(FILTER_SEED, dtype=tf.int32),
        )

        def propose_from_particles(state, observation, inputs, proposed_particles):
            proposed_state = attr.evolve(state, particles=proposed_particles)
            observation_ll = pf._observation_model.loglikelihood(proposed_state, observation)
            transition_ll = pf._transition_model.loglikelihood(state, proposed_state, inputs)
            proposal_ll = pf._proposal_model.loglikelihood(proposed_state, state, inputs, observation)
            log_weights = transition_ll + observation_ll - proposal_ll + state.log_weights
            increment = tf.reduce_logsumexp(log_weights, 1)
            log_likelihoods = state.log_likelihoods + increment
            normalized = normalize(log_weights, 1, state.n_particles, True)
            return attr.evolve(
                proposed_state,
                weights=tf.exp(normalized),
                log_weights=normalized,
                log_likelihoods=log_likelihoods,
            ), observation_ll, transition_ll, proposal_ll, increment

        seed = tf.constant(FILTER_SEED, dtype=tf.int32)
        paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
        seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))
        state = initial_state
        trace_rows = []
        particles_rows = []
        log_weights_rows = []
        log_likelihood_rows = []
        for time_index in range(T):
            observation = observations[time_index]
            inputs = tf.constant(time_index, dtype=tf.int32)
            seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
            t = state.t
            float_t = tf.cast(t, DTYPE)
            float_t_1 = float_t + 1.0
            flags, ess = pf._resampling_criterion.apply(state)
            state_with_ess = attr.evolve(state, ess=ess / float_t_1 + state.ess * (float_t / float_t_1))
            resampled_state = pf._resampling_method.apply(state_with_ess, flags, seed1)
            transport_matrix = transport(
                state_with_ess.particles,
                state_with_ess.log_weights,
                resampling_method.epsilon,
                resampling_method.scaling,
                resampling_method.convergence_threshold,
                resampling_method.max_iter,
                state_with_ess.n_particles,
            )
            proposal_dist = pf._proposal_model._get_proposal_dist(resampled_state, observation)
            proposed_particles = proposal_dist.sample(seed=seed2)
            updated_state, observation_ll, transition_ll, proposal_ll, increment = propose_from_particles(
                resampled_state,
                observation,
                inputs,
                proposed_particles,
            )
            state = attr.evolve(updated_state, t=t + 1)
            particles_rows.append(state.particles)
            log_weights_rows.append(state.log_weights)
            log_likelihood_rows.append(state.log_likelihoods)
            trace_rows.append({{
                "time_index": time_index,
                "seed1": to_json(seed1),
                "seed2": to_json(seed2),
                "resampling_flags": [bool(v) for v in flags.numpy().tolist()],
                "ess": to_json(ess),
                "observation": to_json(observation),
                "pre_particles": to_json(state_with_ess.particles),
                "pre_log_weights": to_json(state_with_ess.log_weights),
                "post_resample_particles": to_json(resampled_state.particles),
                "post_resample_log_weights": to_json(resampled_state.log_weights),
                "transport_matrix": to_json(transport_matrix),
                "proposal_particles": to_json(proposed_particles),
                "observation_log_likelihoods": to_json(observation_ll),
                "transition_log_likelihoods": to_json(transition_ll),
                "proposal_log_likelihoods": to_json(proposal_ll),
                "log_likelihood_increment": to_json(increment),
                "post_update_particles": to_json(state.particles),
                "post_update_log_weights": to_json(state.log_weights),
                "post_update_log_likelihoods": to_json(state.log_likelihoods),
            }})

        traced_particles = tf.stack(particles_rows)
        traced_log_weights = tf.stack(log_weights_rows)
        traced_log_likelihoods = tf.stack(log_likelihood_rows)
        official_trace_deltas = {{
            "particles": max_abs(traced_particles, official.particles),
            "log_weights": max_abs(traced_log_weights, official.log_weights),
            "log_likelihoods": max_abs(traced_log_likelihoods, official.log_likelihoods),
        }}
        official_trace_match = all(value <= {TRACE_TOLERANCE!r} for value in official_trace_deltas.values())
        payload = {{
            "status": "executed",
            "trace_contract": "external_eager_loop_reproduces_official_filterflow_before_replay_without_runtime_shims",
            "non_mutating_filterflow_runtime_shims": [],
            "trace_validation": {{
                "official_trace_match": official_trace_match,
                "official_trace_deltas": official_trace_deltas,
                "tolerance": {TRACE_TOLERANCE!r},
            }},
            "model": {{
                "T": T,
                "num_particles": N,
                "data_seed": DATA_SEED,
                "filter_seed": FILTER_SEED,
                "epsilon": EPSILON,
                "scaling": SCALING,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iterations": MAX_ITERATIONS,
                "resampling_neff": RESAMPLING_NEFF,
                "dtype": "float64",
                "transition_matrix": transition_matrix_np.astype(float).tolist(),
                "observation_matrix": observation_matrix_np.astype(float).tolist(),
                "transition_covariance_chol": to_json(transition_covariance_chol),
                "observation_covariance_chol": to_json(observation_covariance_chol),
                "initial_particles": initial_particles_np.astype(float).tolist(),
                "observations": observations_np.astype(float).tolist(),
                "proposal": "filterflow_optimal_proposal",
            }},
            "official_series": {{
                "particles": to_json(official.particles),
                "log_weights": to_json(official.log_weights),
                "log_likelihoods": to_json(official.log_likelihoods),
            }},
            "trace_series": {{
                "particles": to_json(traced_particles),
                "log_weights": to_json(traced_log_weights),
                "log_likelihoods": to_json(traced_log_likelihoods),
            }},
            "trace_rows": trace_rows,
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
        }}
        print("FILTERFLOW_R3_TRACE_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_R3_TRACE_JSON_END")
        """
    )


def _bayesfilter_replay(
    filterflow_trace: dict[str, Any],
    *,
    replay_mode: str,
) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        model = filterflow_trace["model"]
        transition_matrix = tf.constant(model["transition_matrix"], dtype=DTYPE)
        observation_matrix = tf.constant(model["observation_matrix"], dtype=DTYPE)
        transition_chol = tf.constant(model["transition_covariance_chol"], dtype=DTYPE)
        observation_chol = tf.constant(model["observation_covariance_chol"], dtype=DTYPE)
        observations = tf.constant(model["observations"], dtype=DTYPE)
        particles = tf.constant(model["initial_particles"], dtype=DTYPE)
        num_particles = int(model["num_particles"])
        log_weights = tf.fill(
            [1, num_particles],
            -tf.math.log(tf.cast(num_particles, DTYPE)),
        )
        log_likelihoods = tf.zeros([1], dtype=DTYPE)
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
        particles_rows = []
        log_weights_rows = []
        log_likelihood_rows = []
        ledger = []
        for time_index, row in enumerate(filterflow_trace["trace_rows"]):
            if replay_mode in {
                "traced_resampling_state",
                "traced_input_computed_resampling_state",
                "traced_input_traced_transport_matrix",
            }:
                particles = tf.constant(row["pre_particles"], dtype=DTYPE)
                log_weights = tf.constant(row["pre_log_weights"], dtype=DTYPE)
            ess = _ess(log_weights)
            flags = ess <= tf.math.log(
                tf.cast(num_particles, DTYPE) * tf.constant(RESAMPLING_NEFF, DTYPE)
            )
            bool_flags = tf.reshape(flags, [-1])
            if replay_mode == "traced_resampling_state":
                particles = tf.constant(row["post_resample_particles"], dtype=DTYPE)
                log_weights = tf.constant(row["post_resample_log_weights"], dtype=DTYPE)
                transport_matrix = tf.zeros(
                    [1, num_particles, num_particles],
                    dtype=DTYPE,
                )
                transport_matrix_status = "bypassed_traced_filterflow_resampling_state"
            elif replay_mode == "traced_input_traced_transport_matrix":
                if bool(tf.reduce_any(bool_flags).numpy()):
                    transport_matrix = tf.constant(row["transport_matrix"], dtype=DTYPE)
                    particles = tf.linalg.matmul(transport_matrix, particles)
                    log_weights = tf.fill(
                        [1, num_particles],
                        -tf.math.log(tf.cast(num_particles, DTYPE)),
                    )
                    transport_matrix_status = "traced_filterflow_transport_matrix"
                else:
                    transport_matrix = tf.zeros(
                        [1, num_particles, num_particles],
                        dtype=DTYPE,
                    )
                    transport_matrix_status = "not_triggered"
            elif bool(tf.reduce_any(bool_flags).numpy()):
                transported = annealed_transport_tf.annealed_transport_resample_tf(
                    particles,
                    log_weights,
                    epsilon=float(model["epsilon"]),
                    scaling=float(model["scaling"]),
                    convergence_threshold=float(model["convergence_threshold"]),
                    max_iterations=int(model["max_iterations"]),
                    ess_mask=bool_flags,
                )
                particles = transported.particles
                log_weights = transported.log_weights
                transport_matrix = tf.cast(transported.transport_matrix, DTYPE)
                transport_matrix_status = "computed"
            else:
                transport_matrix = tf.zeros(
                    [1, num_particles, num_particles],
                    dtype=DTYPE,
                )
                transport_matrix_status = "not_triggered"
            post_resample_particles = particles
            post_resample_log_weights = log_weights
            proposed_particles = tf.constant(row["proposal_particles"], dtype=DTYPE)
            observation = observations[time_index]
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
            proposal_ll = _optimal_proposal_log_prob(
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
            log_weights = _filterflow_normalize(unnormalized, num_particles)
            particles = proposed_particles
            particles_rows.append(particles)
            log_weights_rows.append(log_weights)
            log_likelihood_rows.append(log_likelihoods)
            ledger.append(
                {
                    "time_index": time_index,
                    "resampling_flags": [bool(v) for v in bool_flags.numpy().tolist()],
                    "ess": _json(ess),
                    "post_resample_particles": _json(post_resample_particles),
                    "post_resample_log_weights": _json(post_resample_log_weights),
                    "transport_matrix": _json(transport_matrix),
                    "transport_matrix_status": transport_matrix_status,
                    "post_update_particles": _json(particles),
                    "post_update_log_weights": _json(log_weights),
                    "post_update_log_likelihoods": _json(log_likelihoods),
                    "observation_log_likelihoods": _json(observation_ll),
                    "transition_log_likelihoods": _json(transition_ll),
                    "proposal_log_likelihoods": _json(proposal_ll),
                    "log_likelihood_increment": _json(increment),
                    "finite_values": bool(
                        tf.reduce_all(tf.math.is_finite(particles)).numpy()
                        and tf.reduce_all(tf.math.is_finite(log_weights)).numpy()
                        and tf.reduce_all(tf.math.is_finite(log_likelihoods)).numpy()
                    ),
                }
            )
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "replay_contract": "proposal_particles_replayed_from_valid_filterflow_trace",
            "replay_mode": replay_mode,
            "series": {
                "particles": _json(tf.stack(particles_rows)),
                "log_weights": _json(tf.stack(log_weights_rows)),
                "log_likelihoods": _json(tf.stack(log_likelihood_rows)),
            },
            "ledger": ledger,
            "finite_values": all(row["finite_values"] for row in ledger),
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _ess(log_weights: tf.Tensor) -> tf.Tensor:
    log_neff = -tf.reduce_logsumexp(2.0 * log_weights, axis=1)
    return log_neff


def _observation_log_prob(
    particles: tf.Tensor,
    observation: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_noise: tfd.Distribution,
) -> tf.Tensor:
    error = observation - tf.linalg.matvec(observation_matrix, particles)
    return observation_noise.log_prob(error)


def _transition_log_prob(
    prior_particles: tf.Tensor,
    proposed_particles: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_noise: tfd.Distribution,
) -> tf.Tensor:
    pushed = tf.linalg.matvec(transition_matrix, prior_particles)
    return transition_noise.log_prob(proposed_particles - pushed)


def _optimal_proposal_log_prob(
    prior_particles: tf.Tensor,
    proposed_particles: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    observation_matrix: tf.Tensor,
    transition_cov_inv: tf.Tensor,
    observation_cov_inv: tf.Tensor,
    sigma: tf.Tensor,
    sigma_chol: tf.Tensor,
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
    mean = tf.linalg.matvec(sigma, mean)
    proposal_dist = tfd.MultivariateNormalTriL(mean, sigma_chol)
    return proposal_dist.log_prob(proposed_particles)


def _filterflow_normalize(log_weights: tf.Tensor, num_particles: int) -> tf.Tensor:
    normalized = log_weights - tf.reduce_logsumexp(log_weights, axis=1, keepdims=True)
    normalized = tf.clip_by_value(
        normalized,
        tf.constant(-1e3, dtype=DTYPE),
        tf.constant(0.0, dtype=DTYPE),
    )
    float_n = tf.cast(num_particles, DTYPE)
    stop_gradient_mask = normalized < tf.maximum(
        tf.constant(-13.8, dtype=DTYPE),
        tf.constant(-4.0, dtype=DTYPE) * float_n,
    )
    mask = tf.cast(stop_gradient_mask, DTYPE)
    return tf.stop_gradient(mask * normalized) + (1.0 - mask) * normalized


def _compare_replay(
    bayesfilter_replay: dict[str, Any],
    filterflow_trace: dict[str, Any],
) -> dict[str, Any]:
    if bayesfilter_replay.get("status") != "executed":
        return {
            "status": "blocked",
            "blocker": bayesfilter_replay.get("blocker", "BayesFilter replay did not execute"),
        }
    deltas = {
        key: _max_abs(
            bayesfilter_replay["series"][key],
            filterflow_trace["trace_series"][key],
        )
        for key in ("particles", "log_weights", "log_likelihoods")
    }
    flag_match = True
    first_failure = None
    per_time_deltas = []
    for bf_row, ff_row in zip(bayesfilter_replay["ledger"], filterflow_trace["trace_rows"], strict=True):
        row_deltas = {
            "post_resample_particles": _max_abs(
                bf_row["post_resample_particles"],
                ff_row["post_resample_particles"],
            ),
            "post_resample_log_weights": _max_abs(
                bf_row["post_resample_log_weights"],
                ff_row["post_resample_log_weights"],
            ),
            "observation_log_likelihoods": _max_abs(
                bf_row["observation_log_likelihoods"],
                ff_row["observation_log_likelihoods"],
            ),
            "transition_log_likelihoods": _max_abs(
                bf_row["transition_log_likelihoods"],
                ff_row["transition_log_likelihoods"],
            ),
            "proposal_log_likelihoods": _max_abs(
                bf_row["proposal_log_likelihoods"],
                ff_row["proposal_log_likelihoods"],
            ),
            "log_likelihood_increment": _max_abs(
                bf_row["log_likelihood_increment"],
                ff_row["log_likelihood_increment"],
            ),
            "post_update_log_weights": _max_abs(
                bf_row["post_update_log_weights"],
                ff_row["post_update_log_weights"],
            ),
            "post_update_log_likelihoods": _max_abs(
                bf_row["post_update_log_likelihoods"],
                ff_row["post_update_log_likelihoods"],
            ),
        }
        if bf_row.get("transport_matrix_status") == "computed":
            row_deltas["transport_matrix"] = _max_abs(
                bf_row["transport_matrix"],
                ff_row["transport_matrix"],
            )
        failing_fields = sorted(
            field for field, delta in row_deltas.items() if delta > TRACE_TOLERANCE
        )
        per_time_deltas.append(
            {
                "time_index": bf_row["time_index"],
                "deltas": row_deltas,
                "failing_fields": failing_fields,
            }
        )
        if first_failure is None and failing_fields:
            first_failure = {
                "time_index": bf_row["time_index"],
                "field": failing_fields[0],
                "delta": row_deltas[failing_fields[0]],
            }
        if bf_row["resampling_flags"] != ff_row["resampling_flags"]:
            flag_match = False
            if first_failure is None:
                first_failure = {
                    "time_index": bf_row["time_index"],
                    "field": "resampling_flags",
                    "bayesfilter": bf_row["resampling_flags"],
                    "filterflow": ff_row["resampling_flags"],
                }
    within = all(value <= TRACE_TOLERANCE for value in deltas.values()) and flag_match
    if not within and first_failure is None:
        key = max(deltas, key=lambda item: deltas[item])
        first_failure = {
            "time_index": None,
            "field": key,
            "delta": deltas[key],
        }
    return {
        "status": "compared",
        "implementation_agreement": within,
        "series_deltas": deltas,
        "per_time_deltas": per_time_deltas,
        "resampling_flags_match": flag_match,
        "finite_bayesfilter_replay": bayesfilter_replay["finite_values"],
        "first_failure": first_failure or {"status": "no_failure"},
    }


def _decision(
    computed_comparison: dict[str, Any],
    traced_comparison: dict[str, Any],
    traced_input_computed_comparison: dict[str, Any],
    traced_transport_comparison: dict[str, Any],
    comparator_drift: bool,
) -> str:
    if comparator_drift:
        return "filterflow_r3_float64_trace_replay_blocked_by_comparator_drift"
    if (
        computed_comparison.get("status") != "compared"
        or traced_comparison.get("status") != "compared"
        or traced_input_computed_comparison.get("status") != "compared"
        or traced_transport_comparison.get("status") != "compared"
    ):
        return "filterflow_r3_float64_trace_replay_blocked"
    if (
        not computed_comparison.get("finite_bayesfilter_replay")
        or not traced_comparison.get("finite_bayesfilter_replay")
        or not traced_input_computed_comparison.get("finite_bayesfilter_replay")
        or not traced_transport_comparison.get("finite_bayesfilter_replay")
    ):
        return "filterflow_r3_float64_trace_replay_nonfinite_veto"
    if computed_comparison.get("implementation_agreement"):
        return "filterflow_r3_float64_trace_replay_pass"
    if traced_input_computed_comparison.get("implementation_agreement"):
        return "filterflow_r3_float64_trace_replay_accumulated_roundoff_localized"
    if traced_transport_comparison.get("implementation_agreement"):
        return "filterflow_r3_float64_trace_replay_transport_matrix_delta_localized"
    if traced_comparison.get("implementation_agreement"):
        return "filterflow_r3_float64_trace_replay_resampling_state_delta_localized"
    return "filterflow_r3_float64_trace_replay_mismatch"


def _model_contract() -> dict[str, Any]:
    return {
        "T": T,
        "num_particles": NUM_PARTICLES,
        "data_seed": DATA_SEED,
        "filter_seed": FILTER_SEED,
        "epsilon": EPSILON,
        "scaling": SCALING,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "max_iterations": MAX_ITERATIONS,
        "resampling_neff": RESAMPLING_NEFF,
        "dtype": "float64",
        "proposal": "filterflow_locally_optimal_lgssm_proposal",
        "transition_covariance": [[1.0 / 3.0, 0.5], [0.5, 1.0]],
        "observation_covariance": [[0.01]],
    }


def _compact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    if trace.get("status") != "executed":
        return trace
    return {
        "status": trace["status"],
        "trace_contract": trace["trace_contract"],
        "non_mutating_filterflow_runtime_shims": trace.get(
            "non_mutating_filterflow_runtime_shims",
            [],
        ),
        "trace_validation": trace["trace_validation"],
        "model": {
            key: trace["model"][key]
            for key in (
                "T",
                "num_particles",
                "data_seed",
                "filter_seed",
                "epsilon",
                "scaling",
                "convergence_threshold",
                "max_iterations",
                "resampling_neff",
                "dtype",
                "proposal",
            )
        },
        "first_trace_row": trace["trace_rows"][0],
        "last_trace_row": trace["trace_rows"][-1],
        "cpu_only_manifest": trace["cpu_only_manifest"],
        "stderr_excerpt": trace.get("stderr_excerpt", ""),
    }


def _compact_replay(replay: dict[str, Any] | None) -> dict[str, Any] | None:
    if replay is None:
        return None
    return {
        "status": replay.get("status"),
        "backend": replay.get("backend"),
        "replay_contract": replay.get("replay_contract"),
        "finite_values": replay.get("finite_values"),
        "first_ledger_row": replay["ledger"][0],
        "last_ledger_row": replay["ledger"][-1],
        "cpu_only_manifest": replay["cpu_only_manifest"],
    }


def _tolerances() -> dict[str, float]:
    return {"trace_replay": TRACE_TOLERANCE, **agreement._agreement_tolerances()}


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "model_contract",
        "filterflow_trace",
        "comparison",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "reference_policy",
        "filterflow_status",
        "path_boundary_manifest",
        "run_manifest",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    if payload["model_contract"].get("dtype") != FILTERFLOW_REFERENCE_DTYPE:
        raise ValueError(f"model dtype mismatch: {payload['model_contract'].get('dtype')}")
    allowed = {
        "filterflow_r3_float64_trace_replay_filterflow_blocker",
        "filterflow_r3_float64_trace_replay_trace_validation_failed",
        "filterflow_r3_float64_trace_replay_blocked_by_comparator_drift",
        "filterflow_r3_float64_trace_replay_blocked",
        "filterflow_r3_float64_trace_replay_nonfinite_veto",
        "filterflow_r3_float64_trace_replay_pass",
        "filterflow_r3_float64_trace_replay_accumulated_roundoff_localized",
        "filterflow_r3_float64_trace_replay_transport_matrix_delta_localized",
        "filterflow_r3_float64_trace_replay_resampling_state_delta_localized",
        "filterflow_r3_float64_trace_replay_mismatch",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    trace = payload["filterflow_trace"]
    if trace.get("status") == "executed":
        _validate_cpu(trace["cpu_only_manifest"], "filterflow trace")
        if not trace["trace_validation"]["official_trace_match"]:
            if payload["decision"] != "filterflow_r3_float64_trace_replay_trace_validation_failed":
                raise ValueError("trace mismatch without trace-validation decision")
            return
    if payload["decision"] == "filterflow_r3_float64_trace_replay_pass":
        if not payload["comparison"].get("implementation_agreement"):
            raise ValueError("pass decision without implementation agreement")
    if payload["decision"] == "filterflow_r3_float64_trace_replay_resampling_state_delta_localized":
        if payload["computed_resampling_comparison"].get("implementation_agreement"):
            raise ValueError("localized decision but computed replay passed")
        if not payload["traced_resampling_comparison"].get("implementation_agreement"):
            raise ValueError("localized decision but traced replay failed")
    if payload["decision"] == "filterflow_r3_float64_trace_replay_accumulated_roundoff_localized":
        if payload["computed_resampling_comparison"].get("implementation_agreement"):
            raise ValueError("roundoff decision but computed replay passed")
        if not payload["traced_input_computed_resampling_comparison"].get("implementation_agreement"):
            raise ValueError("roundoff decision but traced-input computed replay failed")
    if payload["decision"] == "filterflow_r3_float64_trace_replay_transport_matrix_delta_localized":
        if payload["traced_input_computed_resampling_comparison"].get("implementation_agreement"):
            raise ValueError("transport-matrix decision but traced-input computed replay passed")
        if not payload["traced_transport_matrix_comparison"].get("implementation_agreement"):
            raise ValueError("transport-matrix decision but traced transport replay failed")
    if payload["bayesfilter_replay"] is not None:
        _validate_cpu(payload["bayesfilter_replay"]["cpu_only_manifest"], "BayesFilter replay")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Filterflow R3 Float64 Proposal Trace Replay",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Reference",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in payload["reference_policy"].items():
        lines.append(f"| `{key}` | `{value}` |")
    shims = payload["filterflow_trace"].get("non_mutating_filterflow_runtime_shims", [])
    lines.extend(
        [
        "",
        "## Runtime Shims",
        "",
        *([f"- {shim}" for shim in shims] if shims else ["- None recorded."]),
        "",
        "## Trace Validation",
        "",
        _json_block(payload["filterflow_trace"].get("trace_validation")),
        "",
        "## Replay Comparison",
        "",
        "### Computed Resampling State",
        "",
        _json_block(payload.get("computed_resampling_comparison")),
        "",
        "### Traced Resampling State",
        "",
        _json_block(payload.get("traced_resampling_comparison")),
        "",
        "### Traced Input, Computed Resampling State",
        "",
        _json_block(payload.get("traced_input_computed_resampling_comparison")),
        "",
        "### Traced Input, Traced Transport Matrix",
        "",
        _json_block(payload.get("traced_transport_matrix_comparison")),
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
    if decision == "filterflow_r3_float64_trace_replay_pass":
        return "R3 is cleared for this bounded trace/replay scenario."
    if decision == "filterflow_r3_float64_trace_replay_resampling_state_delta_localized":
        return (
            "Proposal particles and proposal-density accounting match when "
            "the filterflow post-resample state is replayed. The remaining "
            "computed replay mismatch is localized to small 2D resampling-state "
            "deltas that later affect log weights."
        )
    if decision == "filterflow_r3_float64_trace_replay_accumulated_roundoff_localized":
        return (
            "BayesFilter matches filterflow when each step starts from "
            "filterflow's exact pre-resampling state and BayesFilter computes "
            "the transport. The remaining computed replay mismatch is therefore "
            "localized to accumulated small roundoff/state deltas, not to the "
            "single-step transport formula or proposal-density accounting."
        )
    if decision == "filterflow_r3_float64_trace_replay_transport_matrix_delta_localized":
        return (
            "BayesFilter matches filterflow when filterflow's traced transport "
            "matrix is applied from the exact pre-resampling state, but not "
            "when BayesFilter computes that matrix. The remaining discrepancy "
            "is localized to tiny transport-matrix deltas in the 2D "
            "RegularisedTransform mirror; those deltas are amplified by the "
            "proposal-density terms."
        )
    if decision == "filterflow_r3_float64_trace_replay_trace_validation_failed":
        return "The non-mutating trace loop did not reproduce official filterflow, so replay evidence is blocked."
    if decision == "filterflow_r3_float64_trace_replay_mismatch":
        return "A direct BayesFilter/filterflow replay mismatch remains at R3."
    return "R3 trace/replay did not produce evidence because a blocker or veto fired."


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
    left_tensor = tf.constant(left, dtype=tf.float64)
    right_tensor = tf.constant(right, dtype=tf.float64)
    return float(tf.reduce_max(tf.abs(left_tensor - right_tensor)).numpy())


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _filterflow_status() -> dict[str, Any]:
    return {
        "path": str(FILTERFLOW_PATH),
        "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git_filterflow(["rev-parse", "HEAD"]),
        "status_short": _git_filterflow(["status", "--short"]),
        "status_branch": _git_filterflow(["status", "--short", "--branch"]),
        "dtype": FILTERFLOW_REFERENCE_DTYPE,
        "expected_commit": FILTERFLOW_REFERENCE_COMMIT,
        "marker_path": str(FILTERFLOW_MARKER_PATH),
    }


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
