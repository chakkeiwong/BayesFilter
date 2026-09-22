"""Full canonical pipeline (WITH Contract-E reset) on frozen Austria scope.

The reset-less probe showed flow collapse at chaos-takeoff steps driven by
UKF covariance explosion from unbounded cloud spread. This run includes
the OT/Contract-E reset — the production design's spread stabilizer —
via the actual `canonical_value_and_diagnostics` pipeline. Diagnostic,
descriptive; frozen tensors upcast to float64; runner-seeded noise
(recorded: not the frozen process-noise draws).
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _ROOT)

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def main() -> None:
    started = time.time()
    from bayesfilter.highdim.cubature_genut_neutra_targets import (
        make_genut_neutra_target,
    )
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_filter_tf import (
        CanonicalModelCallbacks,
        canonical_value_and_diagnostics,
    )
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    with tf.device("/CPU:0"):
        target = make_genut_neutra_target(
            "austria_sir", particle_count=1008
        )
    theta0 = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model, _sd = austria_sir_canonical_model(theta0)
    observations = tf.cast(target.observations, DTYPE)
    initial_mean = tf.cast(zhao_cui_sir_austria_model().initial_mean, DTYPE)
    variance = 100.0

    def transition_log_density_fn(points, ancestors, _t):
        mean = model.transition_mean_fn(theta0, ancestors)
        residual = points - mean
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1)
            + 18.0 * tf.constant(np.log(2.0 * np.pi), DTYPE)
        )

    def observation_log_density_fn(points, observation, _t):
        observed = model.observation_fn(points)
        residual = observation[None, :] - observed
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1) / variance
            + 9.0
            * tf.constant(np.log(2.0 * np.pi * variance), DTYPE)
        )

    callbacks = CanonicalModelCallbacks(
        model_id="austria_sir_frozen_scope_probe",
        state_dim=18,
        observation_dim=9,
        transition_mean_fn=lambda p, t: model.transition_mean_fn(theta0, p),
        transition_log_density_fn=transition_log_density_fn,
        process_noise_covariance=model.process_covariance,
        process_noise_covariance_provenance="model_exact",
        observation_fn=lambda p, t: model.observation_fn(p),
        observation_jacobian_fn=lambda p, t: model.observation_jacobian_fn(p),
        observation_covariance=model.observation_covariance,
        observation_log_density_fn=observation_log_density_fn,
        initial_mean=initial_mean,
        initial_covariance=tf.eye(18, dtype=DTYPE),
        initial_covariance_provenance="model_exact",
    )

    cells = {}
    for temper in (1, 4):
        result = canonical_value_and_diagnostics(
            callbacks,
            observations,
            particle_count=1008,
            seed=0,
            flow_substeps=16,
            temper_stages=temper,
        )
        ess = result["per_step_ess"].numpy()
        cells[f"canonical_reset_temper{temper}"] = {
            "value": float(result["value"].numpy()),
            "program_valid": bool(result["program_valid"].numpy()),
            "min_ess": float(ess.min()),
            "ess_at_hard_steps": [round(float(ess[2]), 1), round(float(ess[4]), 1)],
            "ess": [round(float(e), 1) for e in ess],
        }

    payload = {
        "schema": "bayesfilter.ledh_canonical_frozen_austria_full.v1",
        "scope": "frozen Austria tensors T=20 N=1008; full pipeline with Contract-E reset",
        "note": "diagnostic; runner-seeded noise; float64 with float32 reset islands",
        "cells": cells,
        "wall_seconds": time.time() - started,
    }
    out = os.path.join(
        _ROOT, "docs", "benchmarks", "artifacts",
        "ledh_canonical_leaderboard_2026-08", "frozen_austria_full.json",
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
