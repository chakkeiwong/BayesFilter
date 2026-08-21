"""Test whether the exact V7 SIR failure is an RK discretization artifact."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py"
OUT = ROOT / "docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/sir_integrator_sensitivity_attempt01.json"
PATH_INDEX = 9864


def load_runner():
    spec = importlib.util.spec_from_file_location("v7_runner_integrator", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_scalar(value: tf.Tensor) -> float | None:
    scalar = float(value.numpy())
    return scalar if math.isfinite(scalar) else None


def transition(
    sir,
    state: tf.Tensor,
    kappa: tf.Tensor,
    nu: tf.Tensor,
    *,
    internal_step: float,
    classical: bool,
) -> tf.Tensor:
    step = tf.constant(internal_step, tf.float64)
    substeps = int(round(0.02 / internal_step))
    current = state[None, :]
    for _ in range(substeps):
        k1 = sir._rhs(current, kappa, nu)
        k2 = sir._rhs(current + 0.5 * step * k1, kappa, nu)
        k3 = sir._rhs(current + 0.5 * step * k2, kappa, nu)
        k4_scale = 1.0 if classical else 0.5
        k4 = sir._rhs(current + k4_scale * step * k3, kappa, nu)
        current = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current[0]


def replay(runner, noise, *, internal_step: float, classical: bool) -> dict[str, object]:
    sir = runner.sir
    theta = tf.constant([0.0, 0.01, 0.0], tf.float64)
    kappa = sir.BASE_KAPPA * tf.exp(theta[0])
    nu = sir.BASE_NU * tf.exp(theta[1])
    state = sir.INITIAL_MEAN + noise[0][PATH_INDEX]
    rows = []
    first_nonfinite_time = None
    for time_index in range(50):
        mean = transition(
            sir,
            state,
            kappa,
            nu,
            internal_step=internal_step,
            classical=classical,
        )
        latent = mean + noise[1][PATH_INDEX, time_index]
        state = tf.reshape(
            tf.stack([tf.maximum(latent[0::2], 0.0), latent[1::2]], axis=1),
            [18],
        )
        is_finite = bool(tf.reduce_all(tf.math.is_finite(state)).numpy())
        rows.append(
            {
                "time": time_index + 1,
                "finite": is_finite,
                "s9": finite_scalar(state[16]),
                "i9": finite_scalar(state[17]),
                "max_abs_state": finite_scalar(tf.reduce_max(tf.abs(state))),
            }
        )
        if not is_finite:
            first_nonfinite_time = time_index + 1
            break
    return {
        "variant": "classical_rk4" if classical else "author_half_k4",
        "internal_step": internal_step,
        "substeps_per_observation": int(round(0.02 / internal_step)),
        "first_nonfinite_time": first_nonfinite_time,
        "rows": rows,
    }


def main() -> None:
    runner = load_runner()
    noise = runner.make_noise("sir", 16384, (0, 10, 1, 1, 1))
    configurations = []
    for internal_step in (0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001):
        configurations.append(
            replay(runner, noise, internal_step=internal_step, classical=False)
        )
        configurations.append(
            replay(runner, noise, internal_step=internal_step, classical=True)
        )
    initial_noise_i9 = float(noise[0][PATH_INDEX, 17].numpy())
    payload = {
        "schema": "bayesfilter.classifier_score_sir_integrator_sensitivity.v1",
        "path_index": PATH_INDEX,
        "theta": [0.0, 0.01, 0.0],
        "noise_key": [0, 10, 1, 1, 1],
        "initial_i9_mean": 5.0,
        "initial_i9_noise": initial_noise_i9,
        "initial_i9": 5.0 + initial_noise_i9,
        "first_process_noise_i9": float(noise[1][PATH_INDEX, 0, 17].numpy()),
        "configurations": configurations,
        "role": "explanatory diagnostic only; no target-law or campaign change",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
