#!/usr/bin/env python3
"""Compare current-scope GenUT LGSSM value/score to the Kalman oracle."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from docs.benchmarks import run_genut_b098_radial2_four_model as runner
from docs.benchmarks.genut_fd_regression import (
    FD_REGRESSION_STEPS,
    evaluate_regression_derivative,
    fit_quadratic_step_regression,
)
from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score


STEPS = FD_REGRESSION_STEPS
SEEDS = (98201, 98202, 98203, 98204)


def _evaluate(evaluator, target, theta, observations, seed, design):
    row = runner._row(evaluator, target, observations, seed)
    value, score, _ = evaluator(
        theta,
        observations,
        runner.base._noise(seed, int(observations.shape[0]), target["state_dim"])[0],
        runner.base._noise(seed, int(observations.shape[0]), target["state_dim"])[1],
        design,
    )
    del value, score
    return row


def _fd(evaluator, target, seed, step):
    initial, process = runner.base._noise(seed, int(target["observations"].shape[0]), target["state_dim"])
    theta = target["theta"]
    _, score, _ = evaluator(theta, target["observations"], initial, process, target["design"])
    rows = []
    for i in range(target["parameter_dim"]):
        direction = tf.one_hot(i, target["parameter_dim"], dtype=theta.dtype)
        plus = evaluator(theta + step * direction, target["observations"], initial, process, target["design"])[0]
        minus = evaluator(theta - step * direction, target["observations"], initial, process, target["design"])[0]
        fd = (plus - minus) / (2.0 * step)
        error = abs(float(fd.numpy()) - float(score[i].numpy()))
        rows.append({"parameter": i, "score": float(score[i].numpy()), "finite_difference": float(fd.numpy()), "absolute_error": error, "normalized_error": error / max(abs(float(score[i].numpy())), 1.0)})
    return rows


def _fd_regression(evaluator, target, seed):
    initial, process = runner.base._noise(seed, int(target["observations"].shape[0]), target["state_dim"])
    theta = target["theta"]
    _, score, _ = evaluator(theta, target["observations"], initial, process, target["design"])
    rows = []
    for index in range(target["parameter_dim"]):
        direction = tf.one_hot(index, target["parameter_dim"], dtype=theta.dtype)
        finite_differences = []
        endpoint_valid = True
        for step in STEPS:
            plus, _, plus_diagnostics = evaluator(
                theta + step * direction,
                target["observations"],
                initial,
                process,
                target["design"],
            )
            minus, _, minus_diagnostics = evaluator(
                theta - step * direction,
                target["observations"],
                initial,
                process,
                target["design"],
            )
            endpoint_valid = endpoint_valid and bool(
                plus_diagnostics["program_valid"].numpy()
            ) and bool(minus_diagnostics["program_valid"].numpy())
            finite_differences.append(float(((plus - minus) / (2.0 * step)).numpy()))
        if endpoint_valid:
            regression = fit_quadratic_step_regression(STEPS, finite_differences)
            rows.append(
                {
                    "parameter": index,
                    **evaluate_regression_derivative(
                        float(score[index].numpy()), regression
                    ),
                }
            )
        else:
            rows.append(
                {
                    "parameter": index,
                    "score": float(score[index].numpy()),
                    "steps": list(STEPS),
                    "endpoint_valid": False,
                    "diagnostic_pass": False,
                    "reason": "invalid_fd_endpoint",
                }
            )
    return rows


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    with tf.device("/CPU:0"):
        target = runner.base._build_targets()["lgssm_T50"]
    kalman_value, kalman_score = _kalman_value_score(target["theta"], target["observations"])
    arms = {}
    for arm in ("diagonal", "dual_cap"):
        controls = runner._arm_controls(runner._baseline("lgssm_T50", target), arm, "lgssm_T50")
        evaluator = runner._evaluator(target, controls)
        rows = [runner._row(evaluator, target, target["observations"], seed) for seed in SEEDS]
        finite = all(row["finite"] for row in rows)
        value_errors = [abs(row["value"] - float(kalman_value.numpy())) for row in rows]
        score_errors = [[row["score"][i] - float(kalman_score[i].numpy()) for i in range(target["parameter_dim"])] for row in rows]
        arms[arm] = {"controls": controls, "finite": finite, "value_error_mean": statistics.mean(value_errors), "value_error_max": max(value_errors), "score_error_mean": [statistics.mean([x[i] for x in score_errors]) for i in range(target["parameter_dim"])], "score_error_max_abs": [max(abs(x[i]) for x in score_errors) for i in range(target["parameter_dim"])], "rows": rows, "finite_difference": {str(step): _fd(evaluator, target, SEEDS[0], step) for step in STEPS}, "finite_difference_regression": _fd_regression(evaluator, target, SEEDS[0])}
    payload = {"schema": "bayesfilter.genut.lgssm_oracle_validation.v1", "status": "COMPLETE", "model": "lgssm_T50", "particle_count": runner.N, "seeds": SEEDS, "steps": STEPS, "source_observation_sha256": target["source_observation_sha256"], "kalman_oracle": {"value": float(kalman_value.numpy()), "score": [float(x) for x in kalman_score.numpy()]}, "arms": arms, "memory_policy": dict(memory), "tf32": False, "jit_compile": True, "nonclaims": ["no nonlinear score theorem", "no default or HMC promotion"], "command": sys.argv}
    (output / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
