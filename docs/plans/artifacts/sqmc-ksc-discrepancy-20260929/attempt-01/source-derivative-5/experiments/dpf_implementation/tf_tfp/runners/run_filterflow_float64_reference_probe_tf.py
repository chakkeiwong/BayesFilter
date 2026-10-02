"""Probe an experimental float64 filterflow LGSSM table variant."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-reference-probe-plan-2026-06-03.md"
RESULT_PATH = Path("docs/plans/bayesfilter-dpf-filterflow-float64-reference-probe-result-2026-06-03.md")
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_reference_probe_2026-06-03.json"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_UPSTREAM_COMMIT = "5d8300ba247c4c17e1a301a22560c24fd0670bfe"
FILTERFLOW_FLOAT64_BRANCH = "bayesfilter-py311-float64-reference"

THETAS = (0.25, 0.5, 0.75)
EPSILONS = (0.25, 0.5, 0.75)
HORIZON = 150
NUM_REALIZATIONS = 100
NUM_PARTICLES = 25
DATA_SEED = 111
FILTER_SEED = 555
KALMAN_PER_TIME_TOLERANCE = 1e-6


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
    write_text(REPO_ROOT / RESULT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    filterflow_status = _filterflow_status()
    canonical = _run_filterflow_subprocess("float32")
    float64 = _run_filterflow_subprocess("float64_branch")
    comparison = _compare_payloads(canonical, float64)
    decision = _decision(canonical, float64, comparison)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": str(RESULT_PATH),
        "question": (
            "Does an experimental in-memory float64 filterflow variant preserve "
            "the Section-5.1-style LGSSM table scale?"
        ),
        "filterflow_status": filterflow_status,
        "reference_policy": {
            "canonical_reference": "local executable filterflow float32 checkout",
            "float64_status": "local_filterflow_float64_reference_branch",
            "float64_branch": FILTERFLOW_FLOAT64_BRANCH,
            "source_mutation": "forbidden_and_not_performed",
            "use_decision": (
                "Future BayesFilter/filterflow difference audits should compare "
                "against the float64 reference branch unless a paper-reproduction "
                "audit specifically requires the native float32 executable path."
            ),
        },
        "settings": {
            "horizon": HORIZON,
            "num_particles": NUM_PARTICLES,
            "num_realizations": NUM_REALIZATIONS,
            "theta_grid": list(THETAS),
            "epsilon_grid": list(EPSILONS),
            "data_seed": DATA_SEED,
            "filter_seed": FILTER_SEED,
            "transition_matrix_for_data": "0.5 I_2",
            "transition_covariance": "I_2 executable filterflow reproduction setting",
            "observation_matrix": "I_2",
            "observation_covariance": "0.1 I_2",
            "resampling_threshold": "NeffCriterion(0.5, True)",
            "regularized_kwargs": {
                "scaling": 0.9,
                "convergence_threshold": 1e-3,
                "max_iter": 100,
            },
            "comparison_policy": (
                "Kalman per-time deltas must be below tolerance; PF and "
                "RegularisedTransform rows must remain within one canonical "
                "float32 Monte Carlo standard deviation."
            ),
        },
        "canonical_float32": canonical,
        "experimental_float64": float64,
        "comparison": comparison,
        "summary": _summary(comparison),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_float64_reference_probe_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _run_filterflow_subprocess(dtype_name: str) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        raise RuntimeError(f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}")
    script = _filterflow_subprocess_script(dtype_name)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"filterflow {dtype_name} subprocess failed\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_JSON_END")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"filterflow {dtype_name} JSON sentinels missing:\n{stdout[-4000:]}")
    payload = json.loads(stdout[start + len("FILTERFLOW_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    payload["command"] = (
        f"CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl "
        f"PYTHONPATH=.localsource/filterflow {FILTERFLOW_ENV_PYTHON} "
        f"-c <filterflow {dtype_name} table script>"
    )
    return payload


def _filterflow_subprocess_script(dtype_name: str) -> str:
    if dtype_name not in {"float32", "float64_branch"}:
        raise ValueError(dtype_name)
    run_dtype = "float64" if dtype_name == "float64_branch" else "float32"
    return textwrap.dedent(
        f"""
        import attr
        import inspect
        import json
        import math
        import os

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import numpy as np
        import tensorflow as tf

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import MultinomialResampler, RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_common import get_data, kf_loglikelihood

        DTYPE_NAME = {run_dtype!r}
        DTYPE = tf.float64 if DTYPE_NAME == "float64" else tf.float32
        NP_DTYPE = np.float64 if DTYPE_NAME == "float64" else np.float32
        THETAS = {list(THETAS)!r}
        EPSILONS = {list(EPSILONS)!r}
        T = {HORIZON}
        BATCH_SIZE = {NUM_REALIZATIONS}
        N = {NUM_PARTICLES}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}

        transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
        transition_covariance = np.eye(2, dtype=np.float32)
        observation_matrix = np.eye(2, dtype=np.float32)
        observation_covariance = 0.1 * np.eye(2, dtype=np.float32)
        rng = np.random.RandomState(seed=DATA_SEED)
        data_float32, kf = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T,
            rng,
        )
        initial_particles_float32 = rng.normal(0.0, 1.0, [BATCH_SIZE, N, 2]).astype(np.float32)
        data = data_float32.astype(NP_DTYPE)
        initial_particles = initial_particles_float32.astype(NP_DTYPE)

        transition_covariance = transition_covariance.astype(NP_DTYPE)
        observation_matrix = observation_matrix.astype(NP_DTYPE)
        observation_covariance = observation_covariance.astype(NP_DTYPE)

        def kalman_rows():
            rows = []
            for theta in THETAS:
                kf.transition_matrices = np.diag([theta, theta]).astype(NP_DTYPE)
                ll = float(kf_loglikelihood(kf, data))
                rows.append({{
                    "theta": float(theta),
                    "log_likelihood": ll,
                    "per_time": ll / T,
                    "finite": bool(np.isfinite(ll)),
                }})
            return rows

        def run_method(method, epsilon=None):
            if method == "pf":
                resampling_method = MultinomialResampler()
                kwargs = {{}}
            elif method == "regularized":
                kwargs = {{"epsilon": float(epsilon), "scaling": 0.9, "convergence_threshold": 1e-3}}
                resampling_method = RegularisedTransform(**kwargs)
                if DTYPE_NAME == "float64":
                    resampling_method.epsilon = tf.constant(float(epsilon), dtype=DTYPE)
                    resampling_method.scaling = tf.constant(0.9, dtype=DTYPE)
                    resampling_method.convergence_threshold = tf.constant(1e-3, dtype=DTYPE)
            else:
                raise ValueError(method)
            criterion = NeffCriterion(0.5, True)
            observation_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(data, dtype=DTYPE))
            transition_covariance_chol = tf.linalg.cholesky(tf.convert_to_tensor(transition_covariance, dtype=DTYPE))
            observation_covariance_chol = tf.linalg.cholesky(tf.convert_to_tensor(observation_covariance, dtype=DTYPE))
            rows = []
            for theta in THETAS:
                transition_var = tf.Variable(np.diag([theta, theta]).astype(NP_DTYPE), trainable=True, dtype=DTYPE)
                smc = make_filter(
                    tf.convert_to_tensor(observation_matrix, dtype=DTYPE),
                    transition_var,
                    observation_covariance_chol,
                    transition_covariance_chol,
                    resampling_method,
                    criterion,
                    observation_error_bias=tf.zeros([2], dtype=DTYPE),
                    transition_noise_bias=tf.zeros([2], dtype=DTYPE),
                )
                final_state = smc(
                    State(tf.constant(initial_particles, dtype=DTYPE)),
                    observation_dataset,
                    n_observations=tf.constant(T),
                    return_final=True,
                    seed=tf.constant(FILTER_SEED),
                )
                values = final_state.log_likelihoods.numpy().astype(float)
                rows.append({{
                    "method_id": "filterflow_" + method,
                    "theta": float(theta),
                    "epsilon": None if epsilon is None else float(epsilon),
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
        for row in run_method("pf"):
            kalman_value = next(k["log_likelihood"] for k in kalman if k["theta"] == row["theta"])
            row["mean_error_per_time"] = (row["mean_log_likelihood"] - kalman_value) / T
            row["std_error_per_time"] = row["std_log_likelihood"] / T
            runs.append(row)
        for eps in EPSILONS:
            for row in run_method("regularized", eps):
                kalman_value = next(k["log_likelihood"] for k in kalman if k["theta"] == row["theta"])
                row["mean_error_per_time"] = (row["mean_log_likelihood"] - kalman_value) / T
                row["std_error_per_time"] = row["std_log_likelihood"] / T
                runs.append(row)

        payload = {{
            "status": "executed",
            "dtype": DTYPE_NAME,
            "python": os.sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "float64_branch_native": DTYPE_NAME == "float64",
            "source_mutation": "branch_source_checked_out_before_run",
            "kalman": kalman,
            "runs": runs,
            "settings": {{
                "observation_path": "canonical float32 get_data values cast to run dtype",
                "initial_particles": "canonical float32 random values cast to run dtype",
                "transition_covariance": "I_2",
                "observation_covariance": "0.1 I_2",
                "resampling": "NeffCriterion(0.5, True)",
                "regularized_kwargs": {{"scaling": 0.9, "convergence_threshold": 1e-3}},
            }},
        }}
        print("FILTERFLOW_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_JSON_END")
        """
    )


def _compare_payloads(canonical: dict[str, Any], float64: dict[str, Any]) -> dict[str, Any]:
    kalman_rows = []
    for canonical_row in canonical["kalman"]:
        float64_row = _find_kalman(float64["kalman"], canonical_row["theta"])
        delta = float64_row["per_time"] - canonical_row["per_time"]
        kalman_rows.append(
            {
                "theta": canonical_row["theta"],
                "canonical_per_time": canonical_row["per_time"],
                "float64_per_time": float64_row["per_time"],
                "delta_per_time": delta,
                "abs_delta_per_time": abs(delta),
                "within_tolerance": abs(delta) <= KALMAN_PER_TIME_TOLERANCE,
            }
        )
    stochastic_rows = []
    for canonical_row in canonical["runs"]:
        float64_row = _find_run(
            float64["runs"],
            canonical_row["method_id"],
            canonical_row["theta"],
            canonical_row.get("epsilon"),
        )
        delta = float64_row["mean_error_per_time"] - canonical_row["mean_error_per_time"]
        canonical_sd = canonical_row["std_error_per_time"]
        within_band = abs(delta) <= canonical_sd
        stochastic_rows.append(
            {
                "method_id": canonical_row["method_id"],
                "theta": canonical_row["theta"],
                "epsilon": canonical_row.get("epsilon"),
                "canonical_mean_error_per_time": canonical_row["mean_error_per_time"],
                "float64_mean_error_per_time": float64_row["mean_error_per_time"],
                "delta_mean_error_per_time": delta,
                "abs_delta_mean_error_per_time": abs(delta),
                "canonical_std_error_per_time": canonical_sd,
                "float64_std_error_per_time": float64_row["std_error_per_time"],
                "within_one_canonical_sd": within_band,
                "finite": bool(canonical_row["finite"] and float64_row["finite"]),
            }
        )
    return {
        "kalman_rows": kalman_rows,
        "stochastic_rows": stochastic_rows,
        "kalman_all_within_tolerance": all(row["within_tolerance"] for row in kalman_rows),
        "stochastic_all_within_band": all(row["within_one_canonical_sd"] for row in stochastic_rows),
        "finite": all(row["finite"] for row in stochastic_rows)
        and all(row["finite"] for row in canonical["kalman"])
        and all(row["finite"] for row in float64["kalman"]),
    }


def _find_kalman(rows: list[dict[str, Any]], theta: float) -> dict[str, Any]:
    for row in rows:
        if row["theta"] == theta:
            return row
    raise KeyError(theta)


def _find_run(
    rows: list[dict[str, Any]],
    method_id: str,
    theta: float,
    epsilon: float | None,
) -> dict[str, Any]:
    for row in rows:
        if row["method_id"] == method_id and row["theta"] == theta and row.get("epsilon") == epsilon:
            return row
    raise KeyError((method_id, theta, epsilon))


def _summary(comparison: dict[str, Any]) -> dict[str, Any]:
    stochastic = comparison["stochastic_rows"]
    kalman = comparison["kalman_rows"]
    return {
        "kalman_row_count": len(kalman),
        "stochastic_row_count": len(stochastic),
        "kalman_all_within_tolerance": comparison["kalman_all_within_tolerance"],
        "stochastic_all_within_band": comparison["stochastic_all_within_band"],
        "finite": comparison["finite"],
        "max_kalman_abs_delta_per_time": max(row["abs_delta_per_time"] for row in kalman),
        "max_stochastic_abs_delta_mean_error_per_time": max(
            row["abs_delta_mean_error_per_time"] for row in stochastic
        ),
        "outside_band_rows": sum(1 for row in stochastic if not row["within_one_canonical_sd"]),
    }


def _decision(canonical: dict[str, Any], float64: dict[str, Any], comparison: dict[str, Any]) -> str:
    if canonical["status"] != "executed" or float64["status"] != "executed":
        return "blocked_filterflow_variant_not_executed"
    if not comparison["finite"]:
        return "blocked_nonfinite_float64_reference_probe"
    if not comparison["kalman_all_within_tolerance"]:
        return "float64_reference_probe_kalman_scale_gap"
    if not comparison["stochastic_all_within_band"]:
        return "float64_reference_probe_table_scale_gap"
    return "float64_filterflow_variant_preserves_table_scale"


def _filterflow_status() -> dict[str, str]:
    return {
        "branch": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "HEAD"]),
        "status": _git(["git", "-C", str(FILTERFLOW_PATH), "status", "--short", "--branch"]),
        "diff_summary": _git(["git", "-C", str(FILTERFLOW_PATH), "diff", "--stat"]) or "clean",
        "upstream_base": FILTERFLOW_UPSTREAM_COMMIT,
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != "float64_filterflow_variant_preserves_table_scale":
        raise RuntimeError(payload["decision"])
    if payload["filterflow_status"]["commit"] != FILTERFLOW_UPSTREAM_COMMIT:
        raise RuntimeError("filterflow commit mismatch")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if payload["filterflow_status"]["branch"] != FILTERFLOW_FLOAT64_BRANCH:
        raise RuntimeError("filterflow float64 branch mismatch")
    if not payload["experimental_float64"]["float64_branch_native"]:
        raise RuntimeError("float64 branch run was not branch-native")
    if payload["summary"]["outside_band_rows"] != 0:
        raise RuntimeError("float64 rows outside canonical Monte Carlo band")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["canonical_float32"] = dict(comparable["canonical_float32"])
    comparable["experimental_float64"] = dict(comparable["experimental_float64"])
    comparable["canonical_float32"]["stderr_excerpt"] = "STDERR"
    comparable["experimental_float64"]["stderr_excerpt"] = "STDERR"
    return stable_digest(comparable)


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow Float64 Reference Probe Result

## Decision

`{payload['decision']}`

## Interpretation

The canonical executable filterflow float32 table remains the reproduction
reference.  The float64 path tested here is an experimental in-memory execution
variant with no `.localsource/filterflow` source mutation.

## Summary

{_key_value_table(payload['summary'])}

## Kalman Rows

{_kalman_table(payload['comparison']['kalman_rows'])}

## PF And RegularisedTransform Rows

{_stochastic_table(payload['comparison']['stochastic_rows'])}

## Reference Policy

{_key_value_table(payload['reference_policy'])}

## Non-Implications

{_bullet_list(payload['non_implications'])}
"""


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _kalman_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| theta | canonical per-time | float64 per-time | delta | within tol |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {theta} | {canon:.9g} | {flt:.9g} | {delta:.3e} | {within} |".format(
                theta=row["theta"],
                canon=row["canonical_per_time"],
                flt=row["float64_per_time"],
                delta=row["delta_per_time"],
                within=row["within_tolerance"],
            )
        )
    return "\n".join(lines)


def _stochastic_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| method | eps | theta | canonical mean error | float64 mean error | delta | canonical sd | within band |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        epsilon = "N/A" if row["epsilon"] is None else row["epsilon"]
        lines.append(
            "| {method} | {eps} | {theta} | {canon:.9g} | {flt:.9g} | {delta:.3e} | {sd:.9g} | {within} |".format(
                method=row["method_id"],
                eps=epsilon,
                theta=row["theta"],
                canon=row["canonical_mean_error_per_time"],
                flt=row["float64_mean_error_per_time"],
                delta=row["delta_mean_error_per_time"],
                sd=row["canonical_std_error_per_time"],
                within=row["within_one_canonical_sd"],
            )
        )
    return "\n".join(lines)


def _bullet_list(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _non_implications() -> list[str]:
    return [
        "No production readiness claim.",
        "No paper correctness claim.",
        "No posterior correctness claim.",
        "No gradient correctness claim.",
        "No public API readiness claim.",
        "No monograph claim.",
        "No DSGE/NAWM validation claim.",
        "No claim that the float64 variant is pristine upstream filterflow.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
