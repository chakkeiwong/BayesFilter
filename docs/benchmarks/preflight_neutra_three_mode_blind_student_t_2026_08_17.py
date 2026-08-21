#!/usr/bin/env python3
"""Preflight a centered Student-t proposal without using known mode locations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-neutra-three-mode-provenance-and-evidence-closure-plan-2026-08-17.md"
)
SCALES = (1.0, 2.0, 4.0, 8.0)
DF = 3.0
BATCH_SIZE = 4096
ESS_FRACTION_MIN = 1.0 / 16.0
ROOT_SEED = (20260817, 82001)


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=65_536)
    parser.add_argument("--device", default="1")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not PLAN.is_file():
        raise FileNotFoundError(PLAN)
    if int(args.rows) < BATCH_SIZE or int(args.rows) % BATCH_SIZE != 0:
        raise ValueError("rows must be a multiple of 4096 and at least 4096")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()

    import tensorflow as tf
    import tensorflow_probability as tfp

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob,
        gaussian_mixture_log_prob_responsibilities_score,
    )
    from bayesfilter.testing.weighted_neutra_gaussian_mixture_hmc_tf import (
        analytic_three_mode_target,
    )

    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible GPU, found {logical}")
    target = analytic_three_mode_target()
    tfd = tfp.distributions

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(scale: Any, seed: Any) -> tuple[Any, ...]:
        proposal = tfd.Independent(
            tfd.StudentT(
                df=tf.constant(DF, tf.float64),
                loc=tf.zeros((4,), tf.float64),
                scale=tf.fill((4,), tf.cast(scale, tf.float64)),
            ),
            reinterpreted_batch_ndims=1,
        )
        rows = proposal.sample(int(args.rows), seed=seed)
        target_log_prob = gaussian_mixture_log_prob(
            rows,
            target["probabilities"],
            target["means"],
            target["covariances"],
        )
        log_weights = target_log_prob - proposal.log_prob(rows)
        weights = tf.nn.softmax(log_weights)
        ess_fraction = tf.math.reciprocal(tf.reduce_sum(tf.square(weights))) / tf.cast(
            int(args.rows), tf.float64
        )
        batched = tf.reshape(log_weights, (-1, BATCH_SIZE))
        batch_weights = tf.nn.softmax(batched, axis=-1)
        batch_ess_fraction = tf.math.reciprocal(
            tf.reduce_sum(tf.square(batch_weights), axis=-1)
        ) / tf.constant(float(BATCH_SIZE), tf.float64)
        _value, responsibilities, _score = (
            gaussian_mixture_log_prob_responsibilities_score(
                rows,
                target["probabilities"],
                target["means"],
                target["covariances"],
            )
        )
        component_mass = tf.reduce_sum(weights[:, tf.newaxis] * responsibilities, axis=0)
        return (
            ess_fraction,
            tfp.stats.percentile(batch_ess_fraction, 50.0, interpolation="midpoint"),
            tf.reduce_min(batch_ess_fraction),
            tf.reduce_max(weights),
            component_mass,
            tf.reduce_all(tf.math.is_finite(rows)),
            tf.reduce_all(tf.math.is_finite(log_weights)),
        )

    arms = []
    for index, scale in enumerate(SCALES):
        seed = tf.random.experimental.stateless_fold_in(
            tf.constant(ROOT_SEED, tf.int32), index
        )
        values = evaluate(tf.constant(scale, tf.float64), seed)
        global_ess = float(values[0].numpy())
        median_ess = float(values[1].numpy())
        arms.append(
            {
                "scale": scale,
                "global_ess_fraction": global_ess,
                "median_batch_ess_fraction": median_ess,
                "minimum_batch_ess_fraction": float(values[2].numpy()),
                "maximum_normalized_weight": float(values[3].numpy()),
                "weighted_component_mass": values[4],
                "rows_all_finite": bool(values[5].numpy()),
                "log_weights_all_finite": bool(values[6].numpy()),
                "passed_support": bool(
                    global_ess >= ESS_FRACTION_MIN and median_ess >= ESS_FRACTION_MIN
                ),
            }
        )
    passing = [arm for arm in arms if arm["passed_support"]]
    manifest = {
        "schema": "bayesfilter.neutra.mode_blind_student_t_preflight_manifest.v1",
        "plan": PLAN.as_posix(),
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "device": tuple(str(device) for device in logical),
        "memory_policy": memory,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_enabled": False,
        "rows_per_arm": int(args.rows),
        "batch_size": BATCH_SIZE,
        "proposal_center_source": "fixed_coordinate_origin_without_target_queries",
        "proposal_uses_component_parameters": False,
        "evaluation_uses_exact_component_parameters": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }
    result = {
        "schema": "bayesfilter.neutra.mode_blind_student_t_preflight_result.v1",
        "manifest": manifest,
        "proposal_family": "iid_centered_student_t",
        "degrees_of_freedom": DF,
        "scale_ladder": SCALES,
        "support_threshold": {
            "ess_fraction_minimum": ESS_FRACTION_MIN,
            "derivation": "at_least_256_effective_rows_per_4096_row_batch",
        },
        "arms": arms,
        "passed": bool(passing),
        "decision": (
            "proposal_support_passed_nominate_training"
            if passing
            else "proposal_support_failed_stop_before_training_and_hmc"
        ),
        "nonclaims": (
            "one naive mode-blind proposal family only",
            "proposal failure does not reject mode discovery in general",
            "component diagnostics are evaluation-only and do not enter proposal construction",
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output / "run_manifest.json", manifest)
    _write(output / "result.json", result)
    _write(
        output / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.mode_blind_student_t_preflight_hashes.v1",
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(json.dumps({"passed": bool(passing), "output_root": output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
