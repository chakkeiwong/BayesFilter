"""Compare theta-vector and full-matrix BayesFilter gradients for row 173."""

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
    run_filterflow_float64_row_173_vjp_decomposition_tf as vjp,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r3_float64_trace_replay_tf as r3,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_smoothness_gradient_localization_tf as localizer,
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
    "bayesfilter-dpf-filterflow-float64-row-173-full-matrix-gradient-parameterization-plan-2026-06-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-full-matrix-gradient-parameterization-result-2026-06-04.md"
)
JSON_PATH = (
    OUTPUT_DIR
    / "dpf_filterflow_float64_row_173_full_matrix_gradient_parameterization_2026-06-04.json"
)
REPORT_PATH = (
    REPORT_DIR
    / "dpf-filterflow-float64-row-173-full-matrix-gradient-parameterization-2026-06-04.md"
)
FILTERFLOW_MARKER_PATH = vjp.FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

TAG = "row-173-full-matrix-gradient-parameterization"
TARGET_TIME_INDEX = 93
VALUE_TOLERANCE = 5e-8
GRADIENT_TOLERANCE = 2e-4
MATERIAL_REDUCTION_RATIO = 0.5


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
    config = vjp.RunConfig(
        target_time_index=TARGET_TIME_INDEX,
        tag=TAG,
        plan_path=PLAN_PATH,
        result_path=RESULT_PATH,
        json_path=JSON_PATH,
        report_path=REPORT_PATH,
    )
    initial_fingerprint = vjp.continuation._filterflow_fingerprint()
    filterflow = vjp._filterflow_vjp_subprocess(config)
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_full_matrix_parameterization_filterflow_blocker",
            filterflow.get("blocker", "unknown FilterFlow blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            None,
        )

    model = vjp._model_from_filterflow(filterflow)
    bayesfilter = {
        "theta_vector": _bayesfilter_parameterization_probe(
            "theta_vector",
            model,
            config,
        ),
        "full_matrix": _bayesfilter_parameterization_probe(
            "full_matrix",
            model,
            config,
        ),
    }
    final_fingerprint = vjp.continuation._filterflow_fingerprint()
    comparator_drift = vjp.continuation._fingerprints_drifted(
        initial_fingerprint,
        final_fingerprint,
    )
    comparison = _compare(filterflow, bayesfilter, comparator_drift)
    decision = _decision(comparison, comparator_drift)
    payload = {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_full_matrix_gradient_parameterization_probe",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "input_manifest": _input_manifest(filterflow),
        "filterflow_probe": _compact_filterflow(filterflow),
        "bayesfilter_parameterization_probes": bayesfilter,
        "comparison": comparison,
        "path_boundary_manifest": vjp.continuation._path_boundary_manifest(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_full_matrix_gradient_parameterization_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "data_seed": vjp.DATA_SEED,
            "filter_seed": vjp.FILTER_SEED,
            "gpu_visibility_note": (
                "CPU-only was intentionally forced before TensorFlow import; "
                "visible GPU list is informational only."
            ),
        },
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }
    return payload


def _blocked_payload(
    decision: str,
    blocker: str,
    reference_status: dict[str, Any],
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    final_fingerprint = vjp.continuation._filterflow_fingerprint()
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
        "comparator_drift": vjp.continuation._fingerprints_drifted(
            initial_fingerprint,
            final_fingerprint,
        ),
        "filterflow_probe": filterflow,
        "comparison": comparison or {"status": "blocked", "blocker": blocker},
        "path_boundary_manifest": vjp.continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_full_matrix_gradient_parameterization_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": [
            {
                "decision": decision,
                "primary_criterion_status": "blocked",
                "veto_diagnostic_status": blocker,
                "main_uncertainty": "execution blocker",
                "next_justified_action": "repair blocker before gradient interpretation",
                "not_concluded": "parameterization source, correctness, production readiness",
            }
        ],
        "non_implications": _non_implications(),
    }


def _bayesfilter_parameterization_probe(
    parameterization: str,
    model: dict[str, Any],
    config: vjp.RunConfig,
) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        if parameterization == "theta_vector":
            variable = tf.Variable(vjp.THETA, dtype=DTYPE)
            variable_contract = "watch theta vector and build A(theta)=diag(theta)+shift"
        elif parameterization == "full_matrix":
            variable = tf.Variable(_transition_matrix_value(), dtype=DTYPE)
            variable_contract = "watch full transition matrix and compare diag_part(d target / d A)"
        else:
            raise ValueError(f"unknown parameterization: {parameterization}")

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(variable)
            bundle = _bayesfilter_target_bundle_from_variable(
                variable,
                parameterization,
                model,
                config,
            )
            target = bundle["target"]
        raw_gradient = vjp._safe_gradient(tape, target, variable)
        total_gradient_diag = (
            raw_gradient
            if parameterization == "theta_vector"
            else tf.linalg.diag_part(raw_gradient)
        )
        gradients = {
            name: vjp._safe_gradient(tape, target, tensor)
            for name, tensor in bundle.items()
            if name not in vjp.TARGET_FIELD_EXCLUSIONS
        }
        parameter_path_adjoints = {
            name: _project_parameter_gradient(
                vjp._safe_gradient_with_upstream(tape, bundle[name], variable, gradients[name]),
                parameterization,
            )
            for name in gradients
        }
        del tape
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "parameterization": parameterization,
            "variable_contract": variable_contract,
            "target_scalar": vjp._float(target),
            "total_gradient_diag": r3._json(total_gradient_diag),
            "raw_total_gradient": r3._json(raw_gradient),
            "raw_total_gradient_summary": vjp._field(raw_gradient),
            "off_diagonal_gradient": _off_diagonal(raw_gradient, parameterization),
            "resampling_flag": [bool(v) for v in tf.reshape(bundle["flags"], [-1]).numpy().tolist()],
            "values": {
                name: vjp._field(tensor)
                for name, tensor in bundle.items()
                if name not in vjp.TARGET_FIELD_EXCLUSIONS
            },
            "value_tensors": {
                name: r3._json(tensor)
                for name, tensor in bundle.items()
                if name not in vjp.TARGET_FIELD_EXCLUSIONS
            },
            "gradients": {name: vjp._field(tensor) for name, tensor in gradients.items()},
            "gradient_tensors": {name: r3._json(tensor) for name, tensor in gradients.items()},
            "parameter_path_adjoint_probe": {
                name: vjp._field(tensor)
                for name, tensor in parameter_path_adjoints.items()
            },
            "parameter_path_adjoint_tensors": {
                name: r3._json(tensor)
                for name, tensor in parameter_path_adjoints.items()
            },
            "cpu_only_manifest": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            },
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_target_bundle_from_variable(
    variable: tf.Tensor,
    parameterization: str,
    model: dict[str, Any],
    config: vjp.RunConfig,
) -> dict[str, tf.Tensor]:
    transition_matrix = _matrix_from_variable(variable, parameterization)
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

    for time_index in range(vjp.T):
        seed, _seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
        ess_log = r3._ess(log_weights)
        flags = ess_log <= tf.math.log(
            tf.cast(num_particles, DTYPE) * tf.constant(vjp.RESAMPLING_NEFF, DTYPE)
        )
        bool_flags = tf.reshape(flags, [-1])
        transported = annealed_transport_tf.annealed_transport_resample_tf(
            particles,
            log_weights,
            epsilon=vjp.EPSILON,
            scaling=vjp.SCALING,
            convergence_threshold=vjp.CONVERGENCE_THRESHOLD,
            max_iterations=vjp.MAX_ITERATIONS,
            ess_mask=bool_flags,
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
        fresh_proposal_dist = tfd.MultivariateNormalTriL(fresh_proposal_mean, sigma_chol)
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
        unnormalized = transition_ll + observation_ll - proposal_ll + log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        pre_current_log_likelihoods = log_likelihoods
        log_likelihoods = log_likelihoods + increment
        normalized = r3._filterflow_normalize(unnormalized, num_particles)
        log_weights = normalized
        particles = proposed_particles
        if time_index == config.target_time_index:
            return {
                "target": tf.reduce_mean(log_likelihoods),
                "flags": flags,
                "log_ess": ess_log,
                "pre_particles": pre_particles,
                "pre_log_weights": pre_log_weights,
                "transport_matrix": transport_matrix,
                "post_particles": transported.particles,
                "post_log_weights": transported.log_weights,
                "proposal_loc": proposal_loc,
                "proposal_mean": proposal_mean,
                "manual_proposal_mean": proposal_mean,
                "fresh_proposal_loc": fresh_proposal_loc,
                "fresh_proposal_mean": fresh_proposal_mean,
                "proposed_particles": proposed_particles,
                "manual_sample_probe_particles": proposed_particles,
                "observation_ll": observation_ll,
                "transition_ll": transition_ll,
                "proposal_ll": proposal_ll,
                "proposal_dist_log_prob": proposal_dist_log_prob,
                "manual_dist_log_prob": proposal_dist_log_prob,
                "fresh_dist_log_prob": fresh_dist_log_prob,
                "unnormalized": unnormalized,
                "increment": increment,
                "pre_current_log_likelihoods": pre_current_log_likelihoods,
                "normalized": normalized,
                "post_update_log_weights": log_weights,
                "post_update_log_likelihoods": log_likelihoods,
            }
    raise RuntimeError("target time not reached")


def _matrix_from_variable(variable: tf.Tensor, parameterization: str) -> tf.Tensor:
    if parameterization == "theta_vector":
        return localizer._transition_matrix(variable)
    if parameterization == "full_matrix":
        return tf.cast(variable, DTYPE)
    raise ValueError(f"unknown parameterization: {parameterization}")


def _transition_matrix_value() -> tf.Tensor:
    return tf.constant(
        [[vjp.THETA[0], 1.0], [0.0, vjp.THETA[1]]],
        dtype=DTYPE,
    )


def _project_parameter_gradient(gradient: tf.Tensor, parameterization: str) -> tf.Tensor:
    if parameterization == "theta_vector":
        return gradient
    if parameterization == "full_matrix":
        return tf.linalg.diag_part(gradient)
    raise ValueError(f"unknown parameterization: {parameterization}")


def _off_diagonal(raw_gradient: tf.Tensor, parameterization: str) -> list[float] | None:
    if parameterization != "full_matrix":
        return None
    return [vjp._float(raw_gradient[0, 1]), vjp._float(raw_gradient[1, 0])]


def _compare(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, dict[str, Any]],
    comparator_drift: bool,
) -> dict[str, Any]:
    if comparator_drift:
        return {"status": "blocked", "blocker": "FilterFlow comparator fingerprint drifted"}
    ff_diag = filterflow["total_gradient_diag"]
    ff_scalar = float(filterflow["target_scalar"])
    rows = {}
    for name, probe in bayesfilter.items():
        scalar_delta = abs(float(probe["target_scalar"]) - ff_scalar)
        diag_delta = [
            float(bf) - float(ff)
            for bf, ff in zip(probe["total_gradient_diag"], ff_diag, strict=True)
        ]
        value_deltas = vjp._field_deltas(
            filterflow["values"],
            probe["values"],
            filterflow.get("value_tensors", {}),
            probe.get("value_tensors", {}),
        )
        parameter_path = vjp._compare_parameter_path_adjoints(filterflow, probe)
        rows[name] = {
            "status": "compared",
            "parameterization": probe["parameterization"],
            "scalar_delta": scalar_delta,
            "resampling_flags_match": probe["resampling_flag"] == filterflow["resampling_flag"],
            "filterflow_total_gradient_diag": ff_diag,
            "bayesfilter_total_gradient_diag": probe["total_gradient_diag"],
            "total_gradient_diag_delta": diag_delta,
            "max_abs_total_gradient_diag_delta": max(abs(value) for value in diag_delta),
            "finite_gradient": bool(
                tf.reduce_all(
                    tf.math.is_finite(tf.constant(probe["total_gradient_diag"], dtype=DTYPE))
                ).numpy()
            ),
            "value_deltas": value_deltas,
            "first_value_delta_over_tolerance": vjp._first_delta(value_deltas, VALUE_TOLERANCE),
            "raw_total_gradient": probe["raw_total_gradient"],
            "off_diagonal_gradient": probe["off_diagonal_gradient"],
            "parameter_path_adjoint": parameter_path,
        }
    theta_delta = rows["theta_vector"]["max_abs_total_gradient_diag_delta"]
    full_delta = rows["full_matrix"]["max_abs_total_gradient_diag_delta"]
    internal_delta = [
        float(full) - float(theta)
        for full, theta in zip(
            bayesfilter["full_matrix"]["total_gradient_diag"],
            bayesfilter["theta_vector"]["total_gradient_diag"],
            strict=True,
        )
    ]
    return {
        "status": "compared",
        "rows": rows,
        "theta_vector_max_abs_delta": theta_delta,
        "full_matrix_max_abs_delta": full_delta,
        "full_matrix_reduction_ratio": (
            full_delta / theta_delta if theta_delta > 0.0 else 0.0
        ),
        "bayesfilter_full_matrix_minus_theta_vector_diag": internal_delta,
        "scalar_gate_pass": all(row["scalar_delta"] <= VALUE_TOLERANCE for row in rows.values()),
        "resampling_gate_pass": all(row["resampling_flags_match"] for row in rows.values()),
        "gradient_finite_gate_pass": all(row["finite_gradient"] for row in rows.values()),
        "value_gate_pass": all(
            row["first_value_delta_over_tolerance"]["status"] == "no_delta"
            for row in rows.values()
        ),
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "value_tolerance": VALUE_TOLERANCE,
    }


def _decision(comparison: dict[str, Any], comparator_drift: bool) -> str:
    if comparator_drift or comparison.get("status") == "blocked":
        return "filterflow_float64_row_173_full_matrix_parameterization_blocked"
    gates_pass = (
        comparison["scalar_gate_pass"]
        and comparison["resampling_gate_pass"]
        and comparison["gradient_finite_gate_pass"]
        and comparison["value_gate_pass"]
    )
    if not gates_pass:
        return "filterflow_float64_row_173_full_matrix_parameterization_scalar_or_resampling_veto"

    theta_match = comparison["theta_vector_max_abs_delta"] <= GRADIENT_TOLERANCE
    full_match = comparison["full_matrix_max_abs_delta"] <= GRADIENT_TOLERANCE
    reduction = comparison["full_matrix_reduction_ratio"]
    if full_match and not theta_match:
        return "filterflow_float64_row_173_full_matrix_parameterization_explains_delta"
    if full_match and theta_match:
        return "filterflow_float64_row_173_full_matrix_parameterization_both_match"
    if reduction <= MATERIAL_REDUCTION_RATIO:
        return "filterflow_float64_row_173_full_matrix_parameterization_partial_effect_only"
    return "filterflow_float64_row_173_full_matrix_parameterization_not_source"


def _input_manifest(filterflow: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": vjp.MESH_INDEX,
        "target_time_index": TARGET_TIME_INDEX,
        "theta": vjp.THETA,
        "T": vjp.T,
        "batch_size": vjp.BATCH_SIZE,
        "num_particles": vjp.NUM_PARTICLES,
        "data_seed": vjp.DATA_SEED,
        "filter_seed": vjp.FILTER_SEED,
        "epsilon": vjp.EPSILON,
        "scaling": vjp.SCALING,
        "convergence_threshold": vjp.CONVERGENCE_THRESHOLD,
        "max_iterations": vjp.MAX_ITERATIONS,
        "resampling_neff": vjp.RESAMPLING_NEFF,
        "dtype": "float64",
        "observation_checksum": filterflow.get("observation_checksum"),
        "initial_particles_checksum": filterflow.get("initial_particles_checksum"),
        "filterflow_settings": filterflow.get("settings"),
    }


def _compact_filterflow(filterflow: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": filterflow.get("status"),
        "backend": filterflow.get("backend"),
        "settings": filterflow.get("settings"),
        "target_scalar": filterflow.get("target_scalar"),
        "total_gradient_diag": filterflow.get("total_gradient_diag"),
        "total_gradient_matrix": filterflow.get("total_gradient_matrix"),
        "resampling_flag": filterflow.get("resampling_flag"),
        "cpu_only_manifest": filterflow.get("cpu_only_manifest"),
        "package_versions": filterflow.get("package_versions"),
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": (
            "Does BayesFilter's theta-vector parameterization explain the "
            "row-173/time-93 gradient delta, as tested by switching to a "
            "FilterFlow-style watched full transition matrix?"
        ),
        "comparator": "local executable float64 FilterFlow checkout",
        "primary_criterion": "full-matrix diagonal gradient delta against FilterFlow",
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "value_tolerance": VALUE_TOLERANCE,
        "vetoes": [
            "FilterFlow subprocess failure",
            "BayesFilter replay failure",
            "scalar/value mismatch before gradient interpretation",
            "resampling flag mismatch",
            "non-finite gradient",
            "comparator drift",
        ],
        "explanatory_only": [
            "off-diagonal full-matrix gradients",
            "BayesFilter full-matrix minus theta-vector internal delta",
            "parameter-path adjoint rows",
        ],
    }


def _decision_table(decision: str, comparison: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "decision": decision,
            "primary_criterion_status": _primary_status(comparison),
            "veto_diagnostic_status": _veto_status(comparison),
            "main_uncertainty": (
                "single row/time diagnostic; no correctness or global "
                "smoothness-gradient claim"
            ),
            "next_justified_action": _next_action(decision),
            "not_concluded": (
                "correctness of either implementation, production readiness, "
                "global gradient agreement"
            ),
        }
    ]


def _primary_status(comparison: dict[str, Any]) -> str:
    if comparison.get("status") == "blocked":
        return str(comparison.get("blocker", "blocked"))
    return (
        "theta_delta={theta:.6g}; full_matrix_delta={full:.6g}; ratio={ratio:.6g}"
    ).format(
        theta=comparison["theta_vector_max_abs_delta"],
        full=comparison["full_matrix_max_abs_delta"],
        ratio=comparison["full_matrix_reduction_ratio"],
    )


def _veto_status(comparison: dict[str, Any]) -> str:
    if comparison.get("status") == "blocked":
        return "blocked"
    vetoes = []
    for key in (
        "scalar_gate_pass",
        "resampling_gate_pass",
        "gradient_finite_gate_pass",
        "value_gate_pass",
    ):
        if not comparison[key]:
            vetoes.append(key)
    return "none" if not vetoes else ", ".join(vetoes)


def _next_action(decision: str) -> str:
    if decision.endswith("_explains_delta"):
        return "audit whether BayesFilter should expose this full-matrix diagnostic path"
    if decision.endswith("_partial_effect_only"):
        return "separate parameterization effect from post-update tape topology"
    if decision.endswith("_not_source"):
        return "continue with post-update/resampling tape-topology probe"
    if decision.endswith("_both_match"):
        return "rerun broader row window with both parameterizations"
    return "repair veto or blocker before interpreting gradients"


def _non_implications() -> list[str]:
    return vjp._non_implications() + [
        "No permanent BayesFilter parameterization policy is concluded.",
        "No claim is made that full-matrix gradients are the desired production API.",
        "This is a row-173/time-93 parameterization difference audit only.",
    ]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: Row 173 Full-Matrix Gradient Parameterization Probe",
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
            "| {decision} | {primary_criterion_status} | {veto_diagnostic_status} | "
            "{main_uncertainty} | {next_justified_action} | {not_concluded} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            _json_block(payload["comparison"]),
            "",
            "## FilterFlow Probe",
            "",
            _json_block(payload["filterflow_probe"]),
            "",
            "## BayesFilter Probes",
            "",
            _json_block(payload["bayesfilter_parameterization_probes"]),
            "",
            "## Input Manifest",
            "",
            _json_block(payload["input_manifest"]),
            "",
            "## Run Manifest",
            "",
            _json_block(payload["run_manifest"]),
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
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


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("created_at_utc", None)
    clone.pop("run_manifest", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


def _validate_payload(payload: dict[str, Any]) -> None:
    required = [
        "decision",
        "comparison",
        "filterflow_probe",
        "bayesfilter_parameterization_probes",
        "input_manifest",
        "run_manifest",
        "decision_table",
        "non_implications",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing payload fields: {missing}")
    if payload["decision"].endswith("_blocked"):
        return
    comparison = payload["comparison"]
    if comparison.get("status") != "compared":
        raise ValueError("non-blocked payload did not compare parameterizations")
    if "theta_vector" not in comparison["rows"] or "full_matrix" not in comparison["rows"]:
        raise ValueError("missing parameterization comparison rows")
    if not isinstance(comparison["full_matrix_max_abs_delta"], float):
        raise ValueError("full matrix delta is not a float")


if __name__ == "__main__":
    raise SystemExit(main())
