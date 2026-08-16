"""Compare the V7 standalone failing path with the canonical SIR model."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py"
OUT = ROOT / "docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/sir_canonical_path_comparison_attempt01.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("v7_runner_compare", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_runner()
    from bayesfilter.highdim.models import parameterized_zhao_cui_sir_austria_model

    path_index = 9864
    theta = tf.constant([0.0, 0.01, 0.0], tf.float64)
    noise = runner.make_noise("sir", 16384, (0, 10, 1, 1, 1))
    standalone_state = runner.sir.INITIAL_MEAN + noise[0][path_index]
    canonical = parameterized_zhao_cui_sir_austria_model()
    canonical_model = canonical.scaled_model(theta)
    canonical_state = canonical_model.initial_mean + noise[0][path_index]
    rows = []
    for time_index in range(11):
        standalone_mean = runner.sir._transition_mean(
            standalone_state[None, :],
            runner.sir.BASE_KAPPA * tf.exp(theta[0]),
            runner.sir.BASE_NU * tf.exp(theta[1]),
        )[0]
        canonical_error = None
        try:
            canonical_mean = canonical_model.transition_mean(canonical_state)[0]
        except ValueError as exc:
            canonical_error = str(exc)
            canonical_mean = tf.fill([18], tf.constant(float("nan"), tf.float64))
        mean_difference = tf.reduce_max(tf.abs(standalone_mean - canonical_mean))
        standalone_latent = standalone_mean + noise[1][path_index, time_index]
        canonical_latent = canonical_mean + noise[1][path_index, time_index]
        standalone_state = tf.reshape(
            tf.stack(
                [tf.maximum(standalone_latent[0::2], 0.0), standalone_latent[1::2]],
                axis=1,
            ),
            [18],
        )
        if canonical_error is None:
            canonical_state = canonical_model._apply_process_noise_policy(
                canonical_latent[None, :]
            )[0]
            state_difference = tf.reduce_max(
                tf.abs(standalone_state - canonical_state)
            )
        else:
            canonical_state = tf.fill([18], tf.constant(float("nan"), tf.float64))
            state_difference = tf.constant(float("nan"), tf.float64)
        rows.append(
            {
                "time": time_index + 1,
                "max_transition_mean_abs_difference": float(mean_difference.numpy()),
                "max_state_abs_difference": float(state_difference.numpy()),
                "canonical_error": canonical_error,
                "standalone_state": standalone_state.numpy().tolist(),
                "canonical_state": canonical_state.numpy().tolist(),
            }
        )
        if not bool(tf.reduce_all(tf.math.is_finite(standalone_state)).numpy()):
            break
    payload = {
        "schema": "bayesfilter.classifier_score_sir_canonical_path_comparison.v1",
        "path_index": path_index,
        "theta": theta,
        "noise_key": [0, 10, 1, 1, 1],
        "rows": rows,
        "exact_match_through_finite_steps": all(
            row["max_transition_mean_abs_difference"] == 0.0
            and row["max_state_abs_difference"] == 0.0
            for row in rows
            if row["max_transition_mean_abs_difference"] < float("inf")
        ),
        "standalone_source": str(runner.SIR_PATH.relative_to(ROOT)),
        "canonical_source": "bayesfilter/highdim/models.py::ParameterizedZhaoCuiSIRSSM",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(runner.safe(payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runner.safe(payload), indent=2))


if __name__ == "__main__":
    main()
