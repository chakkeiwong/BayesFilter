"""Verify the local filterflow float64 reference branch."""

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
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_BRANCH_MARKER,
    FILTERFLOW_REFERENCE_BRANCH,
    FILTERFLOW_REFERENCE_DTYPE,
    reference_policy,
    validate_filterflow_reference_status,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-branch-reference-plan-2026-06-03.md"
RESULT_PATH = Path("docs/plans/bayesfilter-dpf-filterflow-float64-branch-reference-result-2026-06-03.md")
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_branch_reference_2026-06-03.json"
PRIOR_PROBE_JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_reference_probe_2026-06-03.json"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
BRANCH_MARKER = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

THETAS = (0.25, 0.5, 0.75)
EPSILONS = (0.25, 0.5, 0.75)
HORIZON = 150
NUM_REALIZATIONS = 100
NUM_PARTICLES = 25
DATA_SEED = 111
FILTER_SEED = 555
DETERMINISTIC_TOLERANCE = 1e-7
STOCHASTIC_TOLERANCE = 1e-7


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
    validate_filterflow_reference_status(filterflow_status, marker_path=BRANCH_MARKER)
    branch_payload = _run_filterflow_branch_subprocess()
    prior_probe = load_json(PRIOR_PROBE_JSON_PATH)
    comparison = _compare_to_prior_probe(branch_payload, prior_probe)
    decision = _decision(branch_payload, prior_probe, comparison)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": str(RESULT_PATH),
        "question": "Verify local filterflow float64 reference branch for future BayesFilter comparison audits.",
        "reference_policy": reference_policy(),
        "filterflow_status": filterflow_status,
        "prior_probe": {
            "path": str(PRIOR_PROBE_JSON_PATH),
            "decision": prior_probe["decision"],
            "summary": prior_probe["summary"],
        },
        "settings": {
            "horizon": HORIZON,
            "num_particles": NUM_PARTICLES,
            "num_realizations": NUM_REALIZATIONS,
            "theta_grid": list(THETAS),
            "epsilon_grid": list(EPSILONS),
            "data_seed": DATA_SEED,
            "filter_seed": FILTER_SEED,
            "dtype": FILTERFLOW_REFERENCE_DTYPE,
            "resampling_threshold": "NeffCriterion(0.5, True)",
        },
        "branch_run": branch_payload,
        "comparison_to_prior_probe": comparison,
        "summary": _summary(comparison, branch_payload),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_float64_branch_reference_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _run_filterflow_branch_subprocess() -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        raise RuntimeError(f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "filterflow float64 branch subprocess failed\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_JSON_END")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"filterflow JSON sentinels missing:\n{stdout[-4000:]}")
    payload = json.loads(stdout[start + len("FILTERFLOW_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    payload["command"] = (
        f"CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl "
        f"PYTHONPATH=.localsource/filterflow {FILTERFLOW_ENV_PYTHON} "
        "-c <filterflow float64 branch table script>"
    )
    return payload


def _filterflow_script() -> str:
    return textwrap.dedent(
        f"""
        import json
        import os
        import numpy as np
        import tensorflow as tf

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import MultinomialResampler, RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_common import get_data, kf_loglikelihood

        THETAS = {list(THETAS)!r}
        EPSILONS = {list(EPSILONS)!r}
        T = {HORIZON}
        BATCH_SIZE = {NUM_REALIZATIONS}
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
        initial_particles = rng.normal(0.0, 1.0, [BATCH_SIZE, N, 2]).astype(np.float64)

        def kalman_rows():
            rows = []
            for theta in THETAS:
                kf.transition_matrices = np.diag([theta, theta]).astype(np.float64)
                ll = float(kf_loglikelihood(kf, data))
                rows.append({{"theta": float(theta), "log_likelihood": ll, "per_time": ll / T, "finite": bool(np.isfinite(ll))}})
            return rows

        def run_method(method, epsilon=None):
            if method == "pf":
                resampling_method = MultinomialResampler()
                kwargs = {{}}
            elif method == "regularized":
                kwargs = {{"epsilon": float(epsilon), "scaling": 0.9, "convergence_threshold": 1e-3}}
                resampling_method = RegularisedTransform(**kwargs)
            else:
                raise ValueError(method)
            criterion = NeffCriterion(0.5, True)
            observation_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(data, dtype=tf.float64))
            transition_covariance_chol = tf.linalg.cholesky(tf.convert_to_tensor(transition_covariance, dtype=tf.float64))
            observation_covariance_chol = tf.linalg.cholesky(tf.convert_to_tensor(observation_covariance, dtype=tf.float64))
            rows = []
            for theta in THETAS:
                transition_var = tf.Variable(np.diag([theta, theta]).astype(np.float64), trainable=True, dtype=tf.float64)
                smc = make_filter(
                    tf.convert_to_tensor(observation_matrix, dtype=tf.float64),
                    transition_var,
                    observation_covariance_chol,
                    transition_covariance_chol,
                    resampling_method,
                    criterion,
                )
                final_state = smc(
                    State(tf.constant(initial_particles, dtype=tf.float64)),
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
            "dtype": "float64",
            "python": os.sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "kalman": kalman,
            "runs": runs,
            "settings": {{
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


def _compare_to_prior_probe(branch_payload: dict[str, Any], prior_probe: dict[str, Any]) -> dict[str, Any]:
    prior = prior_probe["experimental_float64"]
    kalman_rows = []
    for row in branch_payload["kalman"]:
        prior_row = _find_kalman(prior["kalman"], row["theta"])
        delta = row["per_time"] - prior_row["per_time"]
        kalman_rows.append(
            {
                "theta": row["theta"],
                "branch_per_time": row["per_time"],
                "prior_per_time": prior_row["per_time"],
                "delta": delta,
                "abs_delta": abs(delta),
                "within_tolerance": abs(delta) <= DETERMINISTIC_TOLERANCE,
            }
        )
    stochastic_rows = []
    for row in branch_payload["runs"]:
        prior_row = _find_run(prior["runs"], row["method_id"], row["theta"], row.get("epsilon"))
        delta = row["mean_error_per_time"] - prior_row["mean_error_per_time"]
        stochastic_rows.append(
            {
                "method_id": row["method_id"],
                "theta": row["theta"],
                "epsilon": row.get("epsilon"),
                "branch_mean_error_per_time": row["mean_error_per_time"],
                "prior_mean_error_per_time": prior_row["mean_error_per_time"],
                "delta": delta,
                "abs_delta": abs(delta),
                "within_tolerance": abs(delta) <= STOCHASTIC_TOLERANCE,
                "finite": bool(row["finite"]),
            }
        )
    return {
        "kalman_rows": kalman_rows,
        "stochastic_rows": stochastic_rows,
        "kalman_all_within_tolerance": all(row["within_tolerance"] for row in kalman_rows),
        "stochastic_all_within_tolerance": all(row["within_tolerance"] for row in stochastic_rows),
        "finite": all(row["finite"] for row in stochastic_rows) and all(row["finite"] for row in branch_payload["kalman"]),
    }


def _find_kalman(rows: list[dict[str, Any]], theta: float) -> dict[str, Any]:
    for row in rows:
        if row["theta"] == theta:
            return row
    raise KeyError(theta)


def _find_run(rows: list[dict[str, Any]], method_id: str, theta: float, epsilon: float | None) -> dict[str, Any]:
    for row in rows:
        if row["method_id"] == method_id and row["theta"] == theta and row.get("epsilon") == epsilon:
            return row
    raise KeyError((method_id, theta, epsilon))


def _summary(comparison: dict[str, Any], branch_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch_dtype": branch_payload["dtype"],
        "kalman_row_count": len(comparison["kalman_rows"]),
        "stochastic_row_count": len(comparison["stochastic_rows"]),
        "finite": comparison["finite"],
        "kalman_all_within_tolerance": comparison["kalman_all_within_tolerance"],
        "stochastic_all_within_tolerance": comparison["stochastic_all_within_tolerance"],
        "max_kalman_abs_delta": max(row["abs_delta"] for row in comparison["kalman_rows"]),
        "max_stochastic_abs_delta": max(row["abs_delta"] for row in comparison["stochastic_rows"]),
    }


def _decision(branch_payload: dict[str, Any], prior_probe: dict[str, Any], comparison: dict[str, Any]) -> str:
    if prior_probe["decision"] != "float64_filterflow_variant_preserves_table_scale":
        return "blocked_prior_float64_probe_not_accepted"
    if branch_payload["status"] != "executed":
        return "blocked_filterflow_float64_branch_not_executed"
    if not comparison["finite"]:
        return "blocked_filterflow_float64_branch_nonfinite"
    if not comparison["kalman_all_within_tolerance"]:
        return "blocked_filterflow_float64_branch_kalman_mismatch"
    if not comparison["stochastic_all_within_tolerance"]:
        return "blocked_filterflow_float64_branch_table_mismatch"
    return "filterflow_float64_branch_reference_ready"


def _filterflow_status() -> dict[str, str]:
    return {
        "branch": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "HEAD"]),
        "status": _git(["git", "-C", str(FILTERFLOW_PATH), "status", "--short", "--branch"]),
        "diff_summary": _git(["git", "-C", str(FILTERFLOW_PATH), "diff", "--stat"]) or "clean",
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != "filterflow_float64_branch_reference_ready":
        raise RuntimeError(payload["decision"])
    validate_filterflow_reference_status(payload["filterflow_status"])
    if payload["branch_run"]["dtype"] != FILTERFLOW_REFERENCE_DTYPE:
        raise RuntimeError("branch run did not use float64")
    if not payload["summary"]["finite"]:
        raise RuntimeError("nonfinite branch output")
    if not payload["summary"]["kalman_all_within_tolerance"]:
        raise RuntimeError("Kalman mismatch")
    if not payload["summary"]["stochastic_all_within_tolerance"]:
        raise RuntimeError("stochastic table mismatch")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["branch_run"] = dict(comparable["branch_run"])
    comparable["branch_run"]["stderr_excerpt"] = "STDERR"
    return stable_digest(comparable)


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow Float64 Branch Reference Result

## Decision

`{payload['decision']}`

## Summary

{_key_value_table(payload['summary'])}

## Reference Policy

{_key_value_table(payload['reference_policy'])}

## Filterflow Branch

{_key_value_table(payload['filterflow_status'])}

## Kalman Comparison To Prior Probe

{_kalman_table(payload['comparison_to_prior_probe']['kalman_rows'])}

## Table Comparison To Prior Probe

{_stochastic_table(payload['comparison_to_prior_probe']['stochastic_rows'])}

## Non-Implications

{_bullet_list(payload['non_implications'])}
"""


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _kalman_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| theta | branch | prior | delta | within tol |", "| ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(
            f"| {row['theta']} | {row['branch_per_time']:.9g} | {row['prior_per_time']:.9g} | "
            f"{row['delta']:.3e} | {row['within_tolerance']} |"
        )
    return "\n".join(lines)


def _stochastic_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| method | eps | theta | branch mean error | prior mean error | delta | within tol |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        epsilon = "N/A" if row["epsilon"] is None else row["epsilon"]
        lines.append(
            f"| {row['method_id']} | {epsilon} | {row['theta']} | "
            f"{row['branch_mean_error_per_time']:.9g} | {row['prior_mean_error_per_time']:.9g} | "
            f"{row['delta']:.3e} | {row['within_tolerance']} |"
        )
    return "\n".join(lines)


def _bullet_list(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _non_implications() -> list[str]:
    return [
        "No production readiness claim.",
        "No pristine upstream filterflow claim.",
        "No paper correctness claim.",
        "No posterior correctness claim.",
        "No gradient correctness claim.",
        "No public API readiness claim.",
        "No monograph claim.",
        "No DSGE/NAWM validation claim.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
