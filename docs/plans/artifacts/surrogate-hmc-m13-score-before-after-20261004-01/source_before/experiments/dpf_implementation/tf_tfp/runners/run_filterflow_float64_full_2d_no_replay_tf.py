"""Full 2D float64 BayesFilter/filterflow comparison without proposal replay."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
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
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-full-2d-no-replay-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-full-2d-no-replay-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_full_2d_no_replay_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-full-2d-no-replay-2026-06-03.md"
FILTERFLOW_MARKER_PATH = r3.FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
TRACE_TOLERANCE = r3.TRACE_TOLERANCE


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
    filterflow_trace = r3._filterflow_trace_subprocess()
    if filterflow_trace.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_full_2d_no_replay_filterflow_blocker",
            filterflow_trace.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow_trace,
            reference_status,
        )
    if not filterflow_trace["trace_validation"]["official_trace_match"]:
        return _blocked_payload(
            "filterflow_float64_full_2d_no_replay_trace_validation_failed",
            "external trace loop did not reproduce official filterflow output",
            initial_fingerprint,
            filterflow_trace,
            reference_status,
        )

    bayesfilter = _bayesfilter_no_replay(filterflow_trace)
    comparison = _compare_no_replay(bayesfilter, filterflow_trace)
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
            "Does BayesFilter match executable filterflow on the full 2D "
            "constant-velocity run without replaying filterflow proposal particles?"
        ),
        "evidence_contract": {
            "primary_comparator": "local executable float64 filterflow pf output",
            "primary_question": "cross_implementation_difference_only",
            "trace_must_match_official_filterflow_first": True,
            "proposal_particles_replayed": False,
            "mathematical_correctness": "not_concluded",
        },
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "model_contract": _model_contract(filterflow_trace),
        "filterflow_trace": _compact_trace(filterflow_trace),
        "bayesfilter_no_replay": _compact_no_replay(bayesfilter),
        "comparison": comparison,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": {"trace_replay": TRACE_TOLERANCE},
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_full_2d_no_replay_tf"
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
        "model_contract": {},
        "filterflow_trace": filterflow_trace,
        "bayesfilter_no_replay": None,
        "comparison": {"status": "blocked", "blocker": blocker},
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": {"trace_replay": TRACE_TOLERANCE},
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_full_2d_no_replay_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _bayesfilter_no_replay(filterflow_trace: dict[str, Any]) -> dict[str, Any]:
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
        seed = tf.constant(r3.FILTER_SEED, dtype=tf.int32)
        paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
        seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))

        particles_rows = []
        log_weights_rows = []
        log_likelihood_rows = []
        ledger = []
        for time_index, row in enumerate(filterflow_trace["trace_rows"]):
            seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
            ess = r3._ess(log_weights)
            flags = ess <= tf.math.log(
                tf.cast(num_particles, DTYPE) * tf.constant(r3.RESAMPLING_NEFF, DTYPE)
            )
            bool_flags = tf.reshape(flags, [-1])
            if bool(tf.reduce_any(bool_flags).numpy()):
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
                transport_matrix = tf.zeros([1, num_particles, num_particles], dtype=DTYPE)
                transport_matrix_status = "not_triggered"
            post_resample_particles = particles
            post_resample_log_weights = log_weights
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
            particles_rows.append(particles)
            log_weights_rows.append(log_weights)
            log_likelihood_rows.append(log_likelihoods)
            ledger.append(
                {
                    "time_index": time_index,
                    "seed1": r3._json(seed1),
                    "seed2": r3._json(seed2),
                    "resampling_flags": [bool(v) for v in bool_flags.numpy().tolist()],
                    "ess": r3._json(ess),
                    "post_resample_particles": r3._json(post_resample_particles),
                    "post_resample_log_weights": r3._json(post_resample_log_weights),
                    "transport_matrix": r3._json(transport_matrix),
                    "transport_matrix_status": transport_matrix_status,
                    "proposal_mean": r3._json(proposal_mean),
                    "proposal_particles": r3._json(proposed_particles),
                    "post_update_particles": r3._json(particles),
                    "post_update_log_weights": r3._json(log_weights),
                    "post_update_log_likelihoods": r3._json(log_likelihoods),
                    "observation_log_likelihoods": r3._json(observation_ll),
                    "transition_log_likelihoods": r3._json(transition_ll),
                    "proposal_log_likelihoods": r3._json(proposal_ll),
                    "log_likelihood_increment": r3._json(increment),
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
            "comparison_contract": "BayesFilter samples proposals from matching split seeds; no proposal replay",
            "series": {
                "particles": r3._json(tf.stack(particles_rows)),
                "log_weights": r3._json(tf.stack(log_weights_rows)),
                "log_likelihoods": r3._json(tf.stack(log_likelihood_rows)),
            },
            "ledger": ledger,
            "finite_values": all(row["finite_values"] for row in ledger),
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


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


def _compare_no_replay(
    bayesfilter: dict[str, Any],
    filterflow_trace: dict[str, Any],
) -> dict[str, Any]:
    if bayesfilter.get("status") != "executed":
        return {
            "status": "blocked",
            "blocker": bayesfilter.get("blocker", "BayesFilter no-replay run did not execute"),
        }
    deltas = {
        key: r3._max_abs(bayesfilter["series"][key], filterflow_trace["trace_series"][key])
        for key in ("particles", "log_weights", "log_likelihoods")
    }
    flag_match = True
    first_failure = None
    per_time_deltas = []
    for bf_row, ff_row in zip(bayesfilter["ledger"], filterflow_trace["trace_rows"], strict=True):
        row_deltas = {
            "seed2": r3._max_abs(bf_row["seed2"], ff_row["seed2"]),
            "post_resample_particles": r3._max_abs(
                bf_row["post_resample_particles"],
                ff_row["post_resample_particles"],
            ),
            "post_resample_log_weights": r3._max_abs(
                bf_row["post_resample_log_weights"],
                ff_row["post_resample_log_weights"],
            ),
            "proposal_particles": r3._max_abs(
                bf_row["proposal_particles"],
                ff_row["proposal_particles"],
            ),
            "observation_log_likelihoods": r3._max_abs(
                bf_row["observation_log_likelihoods"],
                ff_row["observation_log_likelihoods"],
            ),
            "transition_log_likelihoods": r3._max_abs(
                bf_row["transition_log_likelihoods"],
                ff_row["transition_log_likelihoods"],
            ),
            "proposal_log_likelihoods": r3._max_abs(
                bf_row["proposal_log_likelihoods"],
                ff_row["proposal_log_likelihoods"],
            ),
            "log_likelihood_increment": r3._max_abs(
                bf_row["log_likelihood_increment"],
                ff_row["log_likelihood_increment"],
            ),
            "post_update_particles": r3._max_abs(
                bf_row["post_update_particles"],
                ff_row["post_update_particles"],
            ),
            "post_update_log_weights": r3._max_abs(
                bf_row["post_update_log_weights"],
                ff_row["post_update_log_weights"],
            ),
            "post_update_log_likelihoods": r3._max_abs(
                bf_row["post_update_log_likelihoods"],
                ff_row["post_update_log_likelihoods"],
            ),
        }
        if bf_row.get("transport_matrix_status") == "computed":
            row_deltas["transport_matrix"] = r3._max_abs(
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
                "all_failing_fields": failing_fields,
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
        first_failure = {"time_index": None, "field": key, "delta": deltas[key]}
    return {
        "status": "compared",
        "implementation_agreement": within,
        "series_deltas": deltas,
        "per_time_deltas": per_time_deltas,
        "resampling_flags_match": flag_match,
        "finite_bayesfilter_no_replay": bayesfilter["finite_values"],
        "first_failure": first_failure or {"status": "no_failure"},
    }


def _decision(
    comparison: dict[str, Any],
    bayesfilter: dict[str, Any],
    comparator_drift: bool,
) -> str:
    if comparator_drift:
        return "filterflow_float64_full_2d_no_replay_blocked_by_comparator_drift"
    if comparison.get("status") != "compared":
        return "filterflow_float64_full_2d_no_replay_blocked"
    if not bayesfilter.get("finite_values"):
        return "filterflow_float64_full_2d_no_replay_nonfinite_veto"
    if comparison.get("implementation_agreement"):
        return "filterflow_float64_full_2d_no_replay_pass"
    first = comparison.get("first_failure", {})
    if first.get("field") == "proposal_particles" and first.get("time_index") == 0:
        return "filterflow_float64_full_2d_no_replay_random_stream_mismatch"
    return "filterflow_float64_full_2d_no_replay_mismatch"


def _model_contract(filterflow_trace: dict[str, Any]) -> dict[str, Any]:
    if filterflow_trace.get("status") != "executed":
        return {}
    model = filterflow_trace["model"]
    return {
        "T": model["T"],
        "num_particles": model["num_particles"],
        "data_seed": model["data_seed"],
        "filter_seed": model["filter_seed"],
        "epsilon": model["epsilon"],
        "scaling": model["scaling"],
        "convergence_threshold": model["convergence_threshold"],
        "max_iterations": model["max_iterations"],
        "resampling_neff": model["resampling_neff"],
        "dtype": model.get("dtype", FILTERFLOW_REFERENCE_DTYPE),
        "proposal": model["proposal"],
        "proposal_particles_replayed": False,
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


def _compact_no_replay(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "status": run.get("status"),
        "backend": run.get("backend"),
        "comparison_contract": run.get("comparison_contract"),
        "finite_values": run.get("finite_values"),
        "first_ledger_row": run["ledger"][0],
        "last_ledger_row": run["ledger"][-1],
        "cpu_only_manifest": run["cpu_only_manifest"],
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "model_contract",
        "filterflow_trace",
        "bayesfilter_no_replay",
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
    allowed = {
        "filterflow_float64_full_2d_no_replay_filterflow_blocker",
        "filterflow_float64_full_2d_no_replay_trace_validation_failed",
        "filterflow_float64_full_2d_no_replay_blocked_by_comparator_drift",
        "filterflow_float64_full_2d_no_replay_blocked",
        "filterflow_float64_full_2d_no_replay_nonfinite_veto",
        "filterflow_float64_full_2d_no_replay_pass",
        "filterflow_float64_full_2d_no_replay_random_stream_mismatch",
        "filterflow_float64_full_2d_no_replay_mismatch",
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
            if payload["decision"] != "filterflow_float64_full_2d_no_replay_trace_validation_failed":
                raise ValueError("trace mismatch without trace-validation decision")
            return
        if trace.get("non_mutating_filterflow_runtime_shims") != []:
            raise ValueError("runtime shims unexpectedly present")
    if payload["bayesfilter_no_replay"] is not None:
        _validate_cpu(payload["bayesfilter_no_replay"]["cpu_only_manifest"], "BayesFilter no-replay")
    if payload["decision"] == "filterflow_float64_full_2d_no_replay_pass":
        if not payload["comparison"].get("implementation_agreement"):
            raise ValueError("pass decision without implementation agreement")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Filterflow Float64 Full 2D No-Replay Comparison",
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
    lines.extend(
        [
            "",
            "## Trace Gate",
            "",
            _json_block(payload["filterflow_trace"].get("trace_validation")),
            "",
            "## Comparison",
            "",
            _json_block(payload["comparison"]),
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
    if decision == "filterflow_float64_full_2d_no_replay_pass":
        return (
            "BayesFilter and the local float64 FilterFlow reference agree in "
            "this bounded full 2D run without proposal replay."
        )
    if decision == "filterflow_float64_full_2d_no_replay_random_stream_mismatch":
        return (
            "The first mismatch occurs in proposal particles at the first time "
            "step. R4 trace replay passed, so the next issue is localized to "
            "cross-environment random sampling/seed semantics rather than the "
            "transport or likelihood arithmetic under replay."
        )
    if decision == "filterflow_float64_full_2d_no_replay_mismatch":
        first = payload["comparison"].get("first_failure")
        return f"A no-replay mismatch remains. First failure: `{first}`."
    return "The no-replay comparison did not produce evidence because a blocker or veto fired."


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


if __name__ == "__main__":
    raise SystemExit(main())
