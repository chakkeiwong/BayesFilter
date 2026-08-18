"""Trace the exact V7 SIR path that first became non-finite."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py"
OUT = ROOT / "docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/sir_failure_trace_attempt01.json"
PATH_INDEX = 9864
COORDINATE = 1
DELTA_INDEX = 1
SIGN_INDEX = 1
DELTA = 0.01


def load_runner():
    spec = importlib.util.spec_from_file_location("v7_runner_trace", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def vector(value: tf.Tensor) -> list[float | None]:
    result = []
    for item in tf.reshape(value, [-1]).numpy().tolist():
        scalar = float(item)
        result.append(scalar if math.isfinite(scalar) else None)
    return result


def finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def main() -> None:
    runner = load_runner()
    sir = runner.sir
    noise = runner.make_noise(
        "sir", 16384, (0, 10, COORDINATE, DELTA_INDEX, SIGN_INDEX)
    )
    theta = runner.THETA + tf.constant([0.0, DELTA, 0.0], tf.float64)
    kappa = sir.BASE_KAPPA * tf.exp(theta[0])
    nu = sir.BASE_NU * tf.exp(theta[1])
    state = sir.INITIAL_MEAN + noise[0][PATH_INDEX]
    rows = []
    for time_index in range(11):
        previous = tf.identity(state)
        current = state[None, :]
        stages = []
        for substep in range(sir.SUBSTEPS):
            k1 = sir._rhs(current, kappa, nu)
            k2 = sir._rhs(current + 0.5 * sir.STEP * k1, kappa, nu)
            k3 = sir._rhs(current + 0.5 * sir.STEP * k2, kappa, nu)
            k4 = sir._rhs(current + 0.5 * sir.STEP * k3, kappa, nu)
            stages.append(
                {
                    "substep": substep + 1,
                    "state": vector(current[0]),
                    "rhs": vector(k1[0]),
                    "finite": finite(current),
                }
            )
            current = current + (sir.STEP / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
        deterministic_mean = current[0]
        process_noise = noise[1][PATH_INDEX, time_index]
        latent = deterministic_mean + process_noise
        state = tf.reshape(
            tf.stack([tf.maximum(latent[0::2], 0.0), latent[1::2]], axis=1),
            [sir.STATE_DIMENSION],
        )
        observation_noise = noise[2][PATH_INDEX, time_index]
        observation = state[1::2] + 10.0 * tf.exp(theta[2]) * observation_noise
        rows.append(
            {
                "time": time_index + 1,
                "previous_state": vector(previous),
                "deterministic_mean": vector(deterministic_mean),
                "process_noise": vector(process_noise),
                "latent_before_clip": vector(latent),
                "state_after_clip": vector(state),
                "observation_noise": vector(observation_noise),
                "observation": vector(observation),
                "finite": finite(state) and finite(observation),
                "stages": stages,
            }
        )
        if not rows[-1]["finite"]:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
