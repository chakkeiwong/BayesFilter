"""Bounded GPU/XLA replay of the repaired GenUT higher-moment route."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import tensorflow as tf


def _configure_memory_growth() -> None:
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)


_configure_memory_growth()

from bayesfilter.highdim.cubature_genut_batch_tf import batch_finite_value_score
from bayesfilter.highdim.cubature_genut_batch_adapters import (
    diagonal_lgssm_batch_adapter,
)
from bayesfilter.highdim.cubature_genut_neutra_targets import (
    GenUTControls,
    make_genut_neutra_target,
)


OUTPUT_ROOT = ROOT / "docs/plans/artifacts/genut-feasible-trust-region-repair-20260815"
PARTICLE_COUNT = 1008
HORIZON = 10
NOISE_SEED = 140000


def _hash_tensor(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _memory_growth() -> list[dict[str, object]]:
    physical = tf.config.list_physical_devices("GPU")
    return [
        {
            "device": device.name,
            "memory_growth": bool(tf.config.experimental.get_memory_growth(device)),
        }
        for device in physical
    ]


def main() -> int:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    controls = GenUTControls(
        epsilon=2.0,
        sinkhorn_steps=2,
        balance_steps=2,
        ridge=1.0e-5,
        higher_moment_correction_steps=4,
        higher_moment_strength=0.2,
        higher_moment_floor=1.0e-5,
        higher_moment_lm_damping=1.0e-2,
        higher_moment_lm_scale_floor=1.0e-4,
        higher_moment_trust_radius=0.5,
        tuning_scope="lgssm_T10_N1008_repair_replay",
        tuning_artifact="local_repair_diagnostic_not_admission",
    )
    base = make_genut_neutra_target(
        "lgssm",
        particle_count=PARTICLE_COUNT,
        noise_seed=NOISE_SEED,
        controls=controls,
    )
    observations = tf.ensure_shape(base.observations[:HORIZON], [HORIZON, 3])
    process_noise = tf.ensure_shape(base.process_noise[:HORIZON], [HORIZON, PARTICLE_COUNT, 3])
    adapter = diagonal_lgssm_batch_adapter(
        observation_matrix=tf.constant(
            ((1.0, 0.25, -0.15), (0.2, 1.1, 0.3), (-0.1, 0.35, 0.9)),
            tf.float32,
        )
    )
    initial_noise = base.initial_noise
    design = base.design
    chart_rows = tf.constant(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05, -0.05, 0.1, 0.0, -0.1],
            [-0.1, 0.1, -0.05, 0.15, 0.05],
            [0.2, -0.15, 0.15, -0.1, 0.1],
        ],
        tf.float64,
    )
    probability = 0.5 * (
        1.0 + tf.math.erf(chart_rows / tf.sqrt(tf.constant(2.0, tf.float64)))
    )
    lower = tf.constant((-0.95, -0.95, -0.95, 0.05, 0.05), tf.float64)
    upper = tf.constant((0.95, 0.95, 0.95, 2.0, 2.0), tf.float64)
    physical = lower[None, :] + (upper - lower)[None, :] * probability
    theta = tf.cast(physical, tf.float32)

    @tf.function(jit_compile=True)
    def compiled(values: tf.Tensor):
        return batch_finite_value_score(
            adapter,
            values,
            observations,
            initial_noise,
            process_noise,
            design,
            epsilon=controls.epsilon,
            sinkhorn_steps=controls.sinkhorn_steps,
            balance_steps=controls.balance_steps,
            ridge=controls.ridge,
            transition_before_first_observation=False,
            higher_moment_correction_steps=controls.higher_moment_correction_steps,
            higher_moment_strength=controls.higher_moment_strength,
            higher_moment_floor=controls.higher_moment_floor,
            higher_moment_lm_damping=controls.higher_moment_lm_damping,
            higher_moment_lm_scale_floor=controls.higher_moment_lm_scale_floor,
            higher_moment_trust_radius=controls.higher_moment_trust_radius,
        )

    value, score, diagnostics = compiled(theta)
    finite = bool(
        tf.reduce_all(
            tf.math.is_finite(value)
            & tf.reduce_all(tf.math.is_finite(score), axis=1)
            & diagnostics["program_valid"]
        ).numpy()
    )
    payload = {
        "schema": "bayesfilter.genut_feasible_trust_region_gpu_replay.v1",
        "status": "PASS_FINITE" if finite else "FAIL_NONFINITE",
        "scientific_claim": "none",
        "target": "LGSSM T=10 finite value/JVP replay",
        "plan": (
            "docs/plans/"
            "bayesfilter-genut-feasible-trust-region-repair-plan-2026-08-15.md"
        ),
        "command": " ".join(sys.argv),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "wall_time_seconds": time.time() - started,
        "controls": dict(controls.payload()),
        "particle_count": PARTICLE_COUNT,
        "horizon": HORIZON,
        "noise_seed": NOISE_SEED,
        "tensorflow_version": tf.__version__,
        "python": sys.executable,
        "platform": platform.platform(),
        "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "memory_growth": _memory_growth(),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "xla": True,
        "input_hashes": {
            "observations": _hash_tensor(observations),
            "initial_noise": _hash_tensor(initial_noise),
            "process_noise": _hash_tensor(process_noise),
            "design": _hash_tensor(design),
            "theta": _hash_tensor(theta),
        },
        "value": tf.cast(value, tf.float64).numpy().tolist(),
        "score_max_abs": tf.reduce_max(tf.abs(score), axis=1).numpy().tolist(),
        "diagnostics": {
            key: tf.cast(value, tf.float64).numpy().tolist()
            for key, value in diagnostics.items()
        },
    }
    output = OUTPUT_ROOT / "gpu_replay_result.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
