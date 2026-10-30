"""Gated LGSSM paper-table comparator against executable FilterFlow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import subprocess
import textwrap
import time
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_lgssm_matched_cross_audit_tf as legacy,
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
    FILTERFLOW_REFERENCE_BRANCH,
    FILTERFLOW_REFERENCE_DTYPE,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
THETAS = (0.25, 0.5, 0.75)
EPSILONS = (0.25, 0.5, 0.75)
SMOKE_THETAS = (0.75,)
SMOKE_EPSILONS = (0.25,)
HORIZON = 150
FULL_NUM_REALIZATIONS = 100
SMOKE_NUM_REALIZATIONS = 20
NUM_PARTICLES = 25
DATA_SEED = 111
FILTER_SEED = 555
SCALING = 0.9
CONVERGENCE_THRESHOLD = 1e-8
MAX_ITERATIONS = 500
RESIDUAL_TOLERANCE = 1e-4
RESIDUAL_POLICY = (
    "diagnostic-only after same-state localization showed executable "
    "FilterFlow shares the smoke residual breach"
)
LOCALIZATION_RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-lgssm-paper-table-smoke-residual-localization-result-2026-06-06.md"
)
PLAN_PATH = "docs/plans/bayesfilter-dpf-lgssm-paper-table-gated-comparator-plan-2026-06-06.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-lgssm-paper-table-gated-comparator-result-2026-06-06.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_lgssm_paper_table_gated_comparator_2026-06-06.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-lgssm-paper-table-gated-comparator-2026-06-06.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"


@dataclass(frozen=True)
class TableConfig:
    mode: str
    thetas: tuple[float, ...]
    epsilons: tuple[float, ...]
    num_realizations: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    config = (
        TableConfig("smoke", SMOKE_THETAS, SMOKE_EPSILONS, SMOKE_NUM_REALIZATIONS)
        if args.smoke
        else TableConfig("full", THETAS, EPSILONS, FULL_NUM_REALIZATIONS)
    )
    start = time.perf_counter()
    payload = _run(config)
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run(config: TableConfig) -> dict[str, Any]:
    filterflow_status = legacy._filterflow_status()
    filterflow = _run_filterflow_subprocess(config)
    spec = legacy.MatchedSpec()
    observations = tf.constant(filterflow["observations"], dtype=DTYPE)
    initial_particles = tf.constant(filterflow["initial_particles"], dtype=DTYPE)
    kalman_tf_rows = _kalman_rows_tf(observations, spec, config.thetas)
    kalman_check = _kalman_alignment(filterflow["kalman"], kalman_tf_rows)
    bayesfilter_rows = _run_bayesfilter_rows(observations, initial_particles, spec, config)
    comparison = _compare_rows(filterflow["runs"], bayesfilter_rows, config)
    decision = _decision(filterflow, kalman_check, comparison, config)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "BayesFilter-vs-FilterFlow LGSSM paper-table gated comparator",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "mode": config.mode,
        "filterflow_status": filterflow_status,
        "filterflow_command": filterflow["command"],
        "filterflow_runs": filterflow["runs"],
        "filterflow_kalman": filterflow["kalman"],
        "bayesfilter_kalman": kalman_tf_rows,
        "kalman_alignment": kalman_check,
        "bayesfilter_runs": bayesfilter_rows,
        "comparison": comparison,
        "settings": {
            "horizon": HORIZON,
            "num_particles": NUM_PARTICLES,
            "num_realizations": config.num_realizations,
            "theta_grid": list(config.thetas),
            "epsilon_grid": list(config.epsilons),
            "data_seed": DATA_SEED,
            "filter_seed": FILTER_SEED,
            "transition_covariance": "I_2",
            "observation_covariance": "0.1 I_2",
            "resampling_threshold": "relative Neff/ESS 0.5",
            "same_observation_path": True,
            "same_initial_particle_cloud": True,
            "algorithmic_randomness": "fixed but not bitwise matched across implementations",
            "filterflow_reference_branch": FILTERFLOW_REFERENCE_BRANCH,
            "filterflow_dtype": FILTERFLOW_REFERENCE_DTYPE,
            "regularized_kwargs": {
                "scaling": SCALING,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iter": MAX_ITERATIONS,
            },
            "residual_diagnostic_tolerance": RESIDUAL_TOLERANCE,
            "residual_policy": RESIDUAL_POLICY,
            "shared_residual_localization_result": LOCALIZATION_RESULT_PATH,
        },
        "summary": _summary(comparison),
        "run_manifest": environment_manifest(
            command=_command_for_mode(config.mode),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": legacy._non_implications(),
    }


def _command_for_mode(mode: str) -> str:
    suffix = " --smoke" if mode == "smoke" else ""
    return (
        "CUDA_VISIBLE_DEVICES=-1 python -m "
        "experiments.dpf_implementation.tf_tfp.runners."
        f"run_filterflow_lgssm_paper_table_gated_comparator_tf{suffix}"
    )


def _run_filterflow_subprocess(config: TableConfig) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        raise RuntimeError(f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_subprocess_script(config)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "filterflow subprocess failed\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_TABLE_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_TABLE_JSON_END")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"filterflow JSON sentinels missing from stdout:\n{stdout[-4000:]}")
    payload = json.loads(stdout[start + len("FILTERFLOW_TABLE_JSON_BEGIN"):end].strip())
    payload["command"] = (
        f"CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <structured LGSSM table script>"
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_subprocess_script(config: TableConfig) -> str:
    return textwrap.dedent(
        f"""
        import json
        import os
        import tensorflow as tf
        np = __import__("numpy")

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_common import get_data, kf_loglikelihood

        THETAS = {list(config.thetas)!r}
        EPSILONS = {list(config.epsilons)!r}
        T = {HORIZON}
        BATCH_SIZE = {config.num_realizations}
        N = {NUM_PARTICLES}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}

        transition_matrix = 0.5 * np.eye(2, dtype=np.float64)
        transition_covariance = np.eye(2, dtype=np.float64)
        observation_matrix = np.eye(2, dtype=np.float64)
        observation_covariance = 0.1 * np.eye(2, dtype=np.float64)
        rng = np.random.RandomState(seed=DATA_SEED)
        data, kf = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T,
            rng,
            dtype=np.float64,
        )
        initial_particles = rng.normal(0., 1., [BATCH_SIZE, N, 2]).astype(np.float64)

        def kalman_rows():
            rows = []
            for theta in THETAS:
                kf.transition_matrices = np.diag([theta, theta]).astype(np.float64)
                ll = float(kf_loglikelihood(kf, data))
                rows.append({{"theta": float(theta), "log_likelihood": ll, "per_time": ll / T}})
            return rows

        def run_regularized(epsilon):
            kwargs = {{
                "epsilon": float(epsilon),
                "scaling": {SCALING!r},
                "convergence_threshold": {CONVERGENCE_THRESHOLD!r},
                "max_iter": {MAX_ITERATIONS},
            }}
            criterion = NeffCriterion(0.5, True)
            observation_dataset = tf.data.Dataset.from_tensor_slices(data)
            transition_covariance_chol = tf.linalg.cholesky(transition_covariance)
            observation_covariance_chol = tf.linalg.cholesky(observation_covariance)
            rows = []
            for theta in THETAS:
                transition_var = tf.Variable(np.diag([theta, theta]).astype(np.float64), trainable=True, dtype=tf.float64)
                smc = make_filter(
                    tf.convert_to_tensor(observation_matrix, dtype=tf.float64),
                    transition_var,
                    observation_covariance_chol,
                    transition_covariance_chol,
                    RegularisedTransform(**kwargs),
                    criterion,
                )
                final_state = smc(
                    State(tf.constant(initial_particles)),
                    observation_dataset,
                    n_observations=tf.constant(T),
                    return_final=True,
                    seed=tf.constant(FILTER_SEED),
                )
                values = final_state.log_likelihoods.numpy().astype(float)
                rows.append({{
                    "method_id": "filterflow_regularized",
                    "theta": float(theta),
                    "epsilon": float(epsilon),
                    "mean_log_likelihood": float(np.mean(values)),
                    "std_log_likelihood": float(np.std(values, ddof=1)),
                    "mean_per_time": float(np.mean(values) / T),
                    "std_per_time": float(np.std(values, ddof=1) / T),
                    "finite": bool(np.all(np.isfinite(values))),
                    "raw_log_likelihoods": values.tolist(),
                    "kwargs": kwargs,
                }})
            return rows

        kalman = kalman_rows()
        runs = []
        for eps in EPSILONS:
            for row in run_regularized(eps):
                kalman_value = next(k["log_likelihood"] for k in kalman if k["theta"] == row["theta"])
                row["mean_error_per_time"] = (row["mean_log_likelihood"] - kalman_value) / T
                row["std_error_per_time"] = row["std_log_likelihood"] / T
                runs.append(row)

        payload = {{
            "status": "executed",
            "python": os.sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "dtype": "float64",
            "kalman": kalman,
            "runs": runs,
            "observations": data.astype(float).tolist(),
            "initial_particles": initial_particles.astype(float).tolist(),
            "settings": {{
                "transition_covariance": "I_2",
                "observation_covariance": "0.1 I_2",
                "resampling": "NeffCriterion(0.5, True)",
                "regularized_kwargs": {{
                    "scaling": {SCALING!r},
                    "convergence_threshold": {CONVERGENCE_THRESHOLD!r},
                    "max_iter": {MAX_ITERATIONS},
                }},
            }},
        }}
        print("FILTERFLOW_TABLE_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_TABLE_JSON_END")
        """
    )


def _run_bayesfilter_rows(
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    spec: legacy.MatchedSpec,
    config: TableConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kalman_by_theta = {theta: legacy._kalman_log_likelihood(observations, theta, spec) for theta in config.thetas}
    for epsilon in config.epsilons:
        for theta in config.thetas:
            try:
                result = _run_bayesfilter_filterflow_style(
                    observations,
                    initial_particles,
                    spec,
                    theta,
                    epsilon,
                )
                errors = (result["log_likelihoods"] - kalman_by_theta[theta]) / tf.cast(HORIZON, DTYPE)
                row = {
                    "method_id": "bayesfilter_exact_filterflow_style_transport_ess",
                    "theta": theta,
                    "epsilon": epsilon,
                    "row_status": "executed",
                    "mean_error_per_time": legacy._float(tf.reduce_mean(errors)),
                    "std_error_per_time": legacy._sample_std(errors),
                    "mean_log_likelihood": legacy._float(tf.reduce_mean(result["log_likelihoods"])),
                    "std_log_likelihood": legacy._sample_std(result["log_likelihoods"]),
                    "finite": result["finite"],
                    "median_resampling_count": legacy._float(legacy._median_tensor(result["resampling_counts"])),
                    "median_min_ess": legacy._float(legacy._median_tensor(result["min_ess"])),
                    "max_residual": result["max_residual"],
                    "max_row_residual": result["max_row_residual"],
                    "max_column_residual": result["max_column_residual"],
                    "residual_exceeds_diagnostic_tolerance": result[
                        "residual_exceeds_diagnostic_tolerance"
                    ],
                    "max_iterations_used": result["max_iterations_used"],
                    "diagnostic": result["diagnostic"],
                }
            except FloatingPointError as exc:
                row = {
                    "method_id": "bayesfilter_exact_filterflow_style_transport_ess",
                    "theta": theta,
                    "epsilon": epsilon,
                    "row_status": "veto",
                    "mean_error_per_time": None,
                    "std_error_per_time": None,
                    "mean_log_likelihood": None,
                    "std_log_likelihood": None,
                    "finite": False,
                    "median_resampling_count": None,
                    "median_min_ess": None,
                    "max_residual": legacy._extract_residual(str(exc)),
                    "max_row_residual": legacy._extract_residual(str(exc)),
                    "max_column_residual": None,
                    "residual_exceeds_diagnostic_tolerance": True,
                    "max_iterations_used": None,
                    "diagnostic": str(exc),
                }
            rows.append(row)
    return rows


def _run_bayesfilter_filterflow_style(
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    spec: legacy.MatchedSpec,
    theta: float,
    epsilon: float,
) -> dict[str, Any]:
    particles = tf.cast(initial_particles, DTYPE)
    batch_size = tf.shape(particles)[0]
    log_weights = tf.fill([batch_size, NUM_PARTICLES], -tf.math.log(tf.cast(NUM_PARTICLES, DTYPE)))
    log_likelihoods = tf.zeros([batch_size], DTYPE)
    resampling_counts = tf.zeros([batch_size], DTYPE)
    min_ess = tf.ones([batch_size], DTYPE) * tf.cast(NUM_PARTICLES, DTYPE)
    max_row_residual = tf.constant(0.0, DTYPE)
    max_column_residual = tf.constant(0.0, DTYPE)
    max_iterations_used = tf.constant(0.0, DTYPE)
    finite = True
    for t in range(HORIZON):
        weights = tf.exp(log_weights)
        ess_before = legacy._ess_batched(weights)
        do_resample = ess_before <= tf.constant(0.5 * NUM_PARTICLES, DTYPE)
        if bool(tf.reduce_any(do_resample).numpy()):
            transported = annealed_transport_resample_tf(
                particles,
                log_weights,
                epsilon=float(epsilon),
                scaling=SCALING,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                max_iterations=MAX_ITERATIONS,
                ess_mask=do_resample,
                transport_gradient_mode="filterflow_clipped",
                application_mode="active_rows_only",
            )
            particles = transported.particles
            log_weights = transported.log_weights
            diag = transported.diagnostics
            max_row_residual = tf.maximum(max_row_residual, tf.constant(diag["max_row_residual"], DTYPE))
            max_column_residual = tf.maximum(max_column_residual, tf.constant(diag["max_column_residual"], DTYPE))
            max_iterations_used = tf.maximum(max_iterations_used, tf.constant(diag["max_iterations_used"], DTYPE))
        resampling_counts += tf.cast(do_resample, DTYPE)
        particles = legacy._transition_particles(
            particles,
            theta,
            spec,
            "bayesfilter_filterflow_style_transport_ess",
            t,
        )
        obs_logp = legacy._observation_log_prob(particles, observations[t], spec)
        unnormalized = log_weights + obs_logp
        normalizer = tf.reduce_logsumexp(unnormalized, axis=1)
        log_likelihoods += normalizer
        log_weights = unnormalized - normalizer[:, None]
        weights = tf.exp(log_weights)
        ess_after = legacy._ess_batched(weights)
        min_ess = tf.minimum(min_ess, ess_after)
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(particles)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_weights)).numpy())
        finite = finite and bool(tf.reduce_all(tf.math.is_finite(log_likelihoods)).numpy())
    max_residual = tf.maximum(max_row_residual, max_column_residual)
    residual_exceeds_diagnostic_tolerance = legacy._float(max_residual) > RESIDUAL_TOLERANCE
    return {
        "log_likelihoods": log_likelihoods,
        "resampling_counts": resampling_counts,
        "min_ess": min_ess,
        "finite": finite,
        "max_residual": legacy._float(max_residual),
        "max_row_residual": legacy._float(max_row_residual),
        "max_column_residual": legacy._float(max_column_residual),
        "residual_exceeds_diagnostic_tolerance": residual_exceeds_diagnostic_tolerance,
        "max_iterations_used": legacy._float(max_iterations_used),
        "diagnostic": (
            "executed_exact_filterflow_style_transport;"
            f" residual_policy={RESIDUAL_POLICY};"
            f" residual_exceeds_diagnostic_tolerance={residual_exceeds_diagnostic_tolerance}"
        ),
    }


def _kalman_rows_tf(
    observations: tf.Tensor,
    spec: legacy.MatchedSpec,
    thetas: tuple[float, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "theta": theta,
            "log_likelihood": legacy._float(legacy._kalman_log_likelihood(observations, theta, spec)),
            "per_time": legacy._float(legacy._kalman_log_likelihood(observations, theta, spec) / tf.cast(HORIZON, DTYPE)),
        }
        for theta in thetas
    ]


def _kalman_alignment(
    filterflow_rows: list[dict[str, Any]],
    tf_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for frow in filterflow_rows:
        trow = next(row for row in tf_rows if row["theta"] == frow["theta"])
        delta = trow["log_likelihood"] - frow["log_likelihood"]
        rows.append({"theta": frow["theta"], "delta": delta, "abs_delta": abs(delta)})
    return {
        "rows": rows,
        "max_abs_delta": max(row["abs_delta"] for row in rows),
        "all_within_tolerance": all(row["abs_delta"] < 1e-6 for row in rows),
    }


def _compare_rows(
    filterflow_rows: list[dict[str, Any]],
    bayesfilter_rows: list[dict[str, Any]],
    config: TableConfig,
) -> list[dict[str, Any]]:
    comparisons = []
    for epsilon in config.epsilons:
        for theta in config.thetas:
            frow = _find_row(filterflow_rows, "filterflow_regularized", theta, epsilon)
            brow = _find_row(
                bayesfilter_rows,
                "bayesfilter_exact_filterflow_style_transport_ess",
                theta,
                epsilon,
            )
            delta = None
            within = None
            within_two_se = None
            if frow and brow and brow["mean_error_per_time"] is not None:
                delta = brow["mean_error_per_time"] - frow["mean_error_per_time"]
                within = abs(delta) <= frow["std_error_per_time"]
                standard_error = frow["std_error_per_time"] / math.sqrt(config.num_realizations)
                within_two_se = abs(delta) <= 2.0 * standard_error
            comparisons.append(
                {
                    "external_method": "filterflow_regularized",
                    "bayesfilter_method": "bayesfilter_exact_filterflow_style_transport_ess",
                    "theta": theta,
                    "epsilon": epsilon,
                    "filterflow_mean_error": None if not frow else frow["mean_error_per_time"],
                    "filterflow_std_error": None if not frow else frow["std_error_per_time"],
                    "bayesfilter_mean_error": None if not brow else brow["mean_error_per_time"],
                    "bayesfilter_std_error": None if not brow else brow["std_error_per_time"],
                    "delta": delta,
                    "within_one_filterflow_sd": within,
                    "within_two_filterflow_se": within_two_se,
                    "bayesfilter_status": None if not brow else brow["row_status"],
                    "bayesfilter_max_residual": None if not brow else brow["max_residual"],
                    "bayesfilter_max_iterations_used": None if not brow else brow["max_iterations_used"],
                    "bayesfilter_residual_exceeds_diagnostic_tolerance": (
                        None if not brow else brow["residual_exceeds_diagnostic_tolerance"]
                    ),
                    "bayesfilter_finite": None if not brow else brow["finite"],
                }
            )
    return comparisons


def _find_row(
    rows: list[dict[str, Any]],
    method_id: str,
    theta: float,
    epsilon: float,
) -> dict[str, Any] | None:
    for row in rows:
        if row["method_id"] == method_id and row["theta"] == theta and row.get("epsilon") == epsilon:
            return row
    return None


def _decision(
    filterflow: dict[str, Any],
    kalman_check: dict[str, Any],
    comparison: list[dict[str, Any]],
    config: TableConfig,
) -> str:
    if filterflow["status"] != "executed":
        return "lgssm_table_filterflow_blocked"
    if not kalman_check["all_within_tolerance"]:
        return "lgssm_table_kalman_mismatch"
    if any(row["bayesfilter_status"] != "executed" or not row["bayesfilter_finite"] for row in comparison):
        return f"lgssm_table_{config.mode}_bayesfilter_veto"
    if all(row["within_one_filterflow_sd"] for row in comparison):
        return f"lgssm_table_{config.mode}_within_filterflow_mc_band"
    return f"lgssm_table_{config.mode}_outside_filterflow_mc_band"


def _summary(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    executed = [row for row in comparison if row["bayesfilter_status"] == "executed"]
    deltas = [abs(row["delta"]) for row in executed if row["delta"] is not None]
    residuals = [row["bayesfilter_max_residual"] for row in executed if row["bayesfilter_max_residual"] is not None]
    residual_flags = [
        row["bayesfilter_residual_exceeds_diagnostic_tolerance"]
        for row in executed
        if row["bayesfilter_residual_exceeds_diagnostic_tolerance"] is not None
    ]
    return {
        "rows": len(comparison),
        "executed_rows": len(executed),
        "all_within_one_filterflow_sd": all(row["within_one_filterflow_sd"] for row in comparison),
        "all_within_two_filterflow_se": all(row["within_two_filterflow_se"] for row in comparison),
        "max_abs_delta": max(deltas) if deltas else None,
        "max_bayesfilter_residual": max(residuals) if residuals else None,
        "rows_exceeding_residual_diagnostic_tolerance": sum(bool(flag) for flag in residual_flags),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    validate_filterflow_reference_status(payload["filterflow_status"])
    if payload["filterflow_runs"][0]["method_id"] != "filterflow_regularized":
        raise RuntimeError("filterflow rows missing")
    if not payload["kalman_alignment"]["all_within_tolerance"]:
        raise RuntimeError("Kalman alignment failed")
    manifest = payload["run_manifest"]
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if manifest["gpu_devices_visible"] != []:
        raise RuntimeError("GPU devices visible in CPU-only run")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")
    expected_rows = len(payload["settings"]["theta_grid"]) * len(payload["settings"]["epsilon_grid"])
    if len(payload["comparison"]) != expected_rows:
        raise RuntimeError("comparison row count mismatch")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# LGSSM Paper-Table Gated Comparator

## Decision

`{payload['decision']}`

## Contract

- Mode: `{payload['mode']}`
- T: `{payload['settings']['horizon']}`
- Particles: `{payload['settings']['num_particles']}`
- Replications: `{payload['settings']['num_realizations']}`
- Theta grid: `{payload['settings']['theta_grid']}`
- Epsilon grid: `{payload['settings']['epsilon_grid']}`
- Solver: `{payload['settings']['regularized_kwargs']}`

## Kalman Alignment

- Max absolute log-likelihood delta: `{payload['kalman_alignment']['max_abs_delta']:.3e}`
- Within tolerance: `{payload['kalman_alignment']['all_within_tolerance']}`

## Summary

- Executed rows: `{payload['summary']['executed_rows']}/{payload['summary']['rows']}`
- All within one FilterFlow SD: `{payload['summary']['all_within_one_filterflow_sd']}`
- All within two FilterFlow SE: `{payload['summary']['all_within_two_filterflow_se']}`
- Max absolute per-time error delta: `{payload['summary']['max_abs_delta']}`
- Max BayesFilter residual: `{payload['summary']['max_bayesfilter_residual']}`
- Rows exceeding residual diagnostic tolerance: `{payload['summary']['rows_exceeding_residual_diagnostic_tolerance']}`
- Residual policy: `{payload['settings']['residual_policy']}`

## Comparison

{_comparison_table(payload['comparison'])}

## Non-Implications

{legacy._non_implications_markdown()}
"""


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| eps | theta | FilterFlow mean | BayesFilter mean | delta | FF SD | within 1 SD | BF residual | residual flag | BF iter | status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {eps} | {theta} | {ff} | {bf} | {delta} | {sd} | {within} | {resid} | {resid_flag} | {iters} | `{status}` |".format(
                eps=legacy._fmt(row["epsilon"]),
                theta=legacy._fmt(row["theta"]),
                ff=legacy._fmt(row["filterflow_mean_error"]),
                bf=legacy._fmt(row["bayesfilter_mean_error"]),
                delta=legacy._fmt(row["delta"]),
                sd=legacy._fmt(row["filterflow_std_error"]),
                within=row["within_one_filterflow_sd"],
                resid=legacy._fmt(row["bayesfilter_max_residual"]),
                resid_flag=row["bayesfilter_residual_exceeds_diagnostic_tolerance"],
                iters=legacy._fmt(row["bayesfilter_max_iterations_used"]),
                status=row["bayesfilter_status"],
            )
        )
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())
