"""Bounded CPU-hidden support screen for the opt-in repaired GenUT route.

This is a mechanics screen only.  It uses each model's current frozen adapter
and full configured horizon, but it is not scope tuning, NeuTra admission, or a
posterior/HMC claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_neutra_targets import (
    GenUTControls,
    make_genut_neutra_target,
)


OUTPUT = ROOT / "docs/plans/artifacts/genut-feasible-trust-region-repair-20260815/cpu_screen_result.json"


def _hash_tensor(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def main() -> int:
    started = time.perf_counter()
    controls = GenUTControls(
        epsilon=2.0,
        sinkhorn_steps=2,
        balance_steps=2,
        ridge=1.0e-5,
        higher_moment_correction_steps=2,
        higher_moment_strength=0.1,
        higher_moment_floor=1.0e-5,
        higher_moment_lm_damping=1.0e-2,
        higher_moment_lm_scale_floor=1.0e-4,
        higher_moment_trust_radius=0.5,
        tuning_scope="support_screen_not_scope_tuning",
        tuning_artifact="diagnostic_only",
    )
    rows = {}
    particle_counts = {
        "lgssm": 12,
        "ksc_sv": 12,
        "austria_sir": 36,
        "predator_prey": 12,
    }
    for model in particle_counts:
        particle_count = particle_counts[model]
        target = make_genut_neutra_target(
            model, particle_count=particle_count, controls=controls
        )
        theta = tf.zeros([2, target.parameter_dim], tf.float64)
        theta = tf.tensor_scatter_nd_update(
            theta, [[1, 0]], [tf.constant(0.05, tf.float64)]
        )
        value, score, status = target.neutra_batch_log_prob_and_grad_status(theta)
        finite = bool(
            tf.reduce_all(tf.math.is_finite(value)).numpy()
            and tf.reduce_all(tf.math.is_finite(score)).numpy()
            and tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
        )
        rows[model] = {
            "status": "PASS_FINITE" if finite else "FAIL_NONFINITE",
            "particle_count": particle_count,
            "horizon": int(target.observations.shape[0]),
            "parameter_dim": target.parameter_dim,
            "target_signature": target.target_signature,
            "observations_sha256": _hash_tensor(target.observations),
            "value": tf.cast(value, tf.float64).numpy().tolist(),
            "score_max_abs": tf.reduce_max(tf.abs(score), axis=1).numpy().tolist(),
            "minimum_pearson_feasibility_margin": status[
                "minimum_pearson_feasibility_margin"
            ].numpy().tolist(),
            "minimum_finite_particle_upper_margin": status[
                "minimum_finite_particle_upper_margin"
            ].numpy().tolist(),
            "maximum_diagonal_scaled_system_condition": status[
                "maximum_diagonal_scaled_system_condition"
            ].numpy().tolist(),
            "maximum_diagonal_pre_cap_particle_rms": status[
                "maximum_diagonal_pre_cap_particle_rms"
            ].numpy().tolist(),
            "maximum_diagonal_post_cap_particle_rms": status[
                "maximum_diagonal_post_cap_particle_rms"
            ].numpy().tolist(),
        }
    passed = all(row["status"] == "PASS_FINITE" for row in rows.values())
    payload = {
        "schema": "bayesfilter.genut_feasible_trust_region_cpu_screen.v1",
        "status": "PASS_FINITE" if passed else "FAIL_NONFINITE",
        "scientific_claim": "none",
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_version": tf.__version__,
        "python": sys.executable,
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "controls": dict(controls.payload()),
        "models": rows,
        "wall_time_seconds": time.perf_counter() - started,
        "nonclaims": [
            "no exact moment matching claim",
            "no statistical ranking",
            "no scope tuning or admission",
            "no NeuTra or HMC claim",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    OUTPUT.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
