#!/usr/bin/env python3
"""Diagnose three-mode chain initialization and fixed-kernel mixing."""

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

CHECKPOINT = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "three-mode/component-aware-budget1000-r1/trainer_states.json"
)
TUNING = ROOT / (
    "docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/three-mode-full/"
    "tuning/tuning_result.json"
)
PLAN = ROOT / "docs/plans/bayesfilter-neutra-gap-closure-plan-2026-08-17.md"


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(v) for v in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial_state(tf: Any, transport: Any, target: Mapping[str, Any], labels: tuple[int, ...]) -> Any:
    means = target["means"]
    dimension = int(means.shape[1])
    direction = tf.cast(tf.range(dimension), tf.float64)
    unit = tf.math.divide_no_nan(direction + 1.0, tf.linalg.norm(direction + 1.0))
    signs = tf.where(
        tf.math.floormod(tf.range(len(labels)), 2) == 0,
        tf.constant(-1.0, tf.float64), tf.constant(1.0, tf.float64)
    )
    physical = tf.gather(means, tf.constant(labels, tf.int32)) + 0.05 * signs[:, tf.newaxis] * unit
    latent, _ = transport.inverse_and_forward_logdet(physical)
    tf.debugging.assert_all_finite(latent, "diagnostic initial state")
    return latent


def _labels(tf: Any, physical: Any, means: Any) -> Any:
    distances = tf.reduce_sum(tf.square(physical[:, :, tf.newaxis, :] - means[tf.newaxis, tf.newaxis, :, :]), axis=-1)
    return tf.argmin(distances, axis=-1, output_type=tf.int32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--results", type=int, default=1000)
    parser.add_argument("--burnin", type=int, default=64)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    if not CHECKPOINT.is_file() or not TUNING.is_file() or not PLAN.is_file():
        raise FileNotFoundError("mixing diagnostic inputs are missing")
    if int(args.results) < 128 or int(args.burnin) < 1:
        raise ValueError("results/burnin are too small")
    root.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    import tensorflow as tf

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        FixedTransportFullChainConfig,
        FixedTransportHMCPolicy,
        run_fixed_transport_full_chain_tfp_hmc,
    )
    from bayesfilter.testing.weighted_neutra_gaussian_mixture_hmc_tf import (
        AnalyticGaussianMixtureValueScoreAdapter,
        analytic_three_mode_target,
        load_weighted_neutra_transport,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible GPU, found {logical}")
    loaded = load_weighted_neutra_transport(CHECKPOINT, required_dimension=4)
    target = analytic_three_mode_target()
    base = AnalyticGaussianMixtureValueScoreAdapter(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=loaded.transport,
        target_scope="weighted_neutra_three_mode_mixing_diagnostic_v1",
        runtime_backend="tensorflow_exact_three_mode_mixing_diagnostic",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    tuning = json.loads(TUNING.read_text())
    candidates = {int(c["num_leapfrog_steps"]): c for c in tuning.get("candidates", [])}
    kernels = []
    for leapfrog in (20, 25):
        candidate = candidates.get(leapfrog)
        if not candidate or not candidate.get("selected_step_size"):
            raise RuntimeError(f"missing recorded step size for L={leapfrog}")
        kernels.append((leapfrog, float(candidate["selected_step_size"])))
    initializations = {
        "component_aware_0120": (0, 1, 2, 0),
        "balanced_0121": (0, 1, 2, 1),
        "local_0000": (0, 0, 0, 0),
    }
    rows = []
    for init_name, labels in initializations.items():
        initial = _initial_state(tf, loaded.transport, target, labels)
        for leapfrog, step_size in kernels:
            for seed_offset in (0, 1):
                seed = (20260817, 73000 + 100 * leapfrog + 10 * seed_offset + len(rows))
                result = run_fixed_transport_full_chain_tfp_hmc(
                    adapter,
                    initial,
                    FixedTransportFullChainConfig(
                        num_results=int(args.results),
                        num_burnin_steps=int(args.burnin),
                        step_size=step_size,
                        num_leapfrog_steps=leapfrog,
                        seed=seed,
                        use_xla=True,
                        trace_policy="full",
                        target_status_trace_policy="per_chain_step",
                        tuning_policy=FixedTransportHMCPolicy.fixed(source=PLAN.as_posix()),
                        target_scope=f"weighted_neutra_three_mode_mixing:{init_name}:L{leapfrog}",
                        chain_execution_mode="tf_function",
                    ),
                )
                physical = loaded.transport.forward_batch(tf.reshape(result.samples, [-1, 4]))
                physical = tf.reshape(physical, tf.shape(result.samples))
                labels_tensor = _labels(tf, physical, target["means"])
                transition_count = tf.reduce_sum(tf.cast(tf.not_equal(labels_tensor[1:], labels_tensor[:-1]), tf.int32), axis=0)
                rows.append({
                    "initialization": init_name,
                    "initial_labels": labels,
                    "num_leapfrog_steps": leapfrog,
                    "step_size": step_size,
                    "seed": seed,
                    "diagnostics": result.diagnostics,
                    "mode_counts_by_chain": tf.math.bincount(tf.reshape(labels_tensor, [-1]), minlength=3, maxlength=3),
                    "mode_counts_by_chain_rows": tf.transpose(tf.map_fn(lambda x: tf.math.bincount(x, minlength=3, maxlength=3), tf.transpose(labels_tensor), fn_output_signature=tf.int32)),
                    "transition_count_by_chain": transition_count,
                    "chain_moved": tf.reduce_any(tf.not_equal(result.samples, initial[tf.newaxis, ...]), axis=(0, 2)),
                })
    manifest = {
        "schema": "bayesfilter.neutra.three_mode_mixing_manifest.v1",
        "plan": PLAN.as_posix(),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "tuning_sha256": _sha256(TUNING),
        "memory_policy": memory,
        "device": str(logical[0]),
        "dtype": "float64",
        "jit_compile": True,
        "results_per_chain": int(args.results),
        "burnin": int(args.burnin),
        "nonclaims": ["initialization and mixing diagnostic only", "no posterior claim", "no repair promotion"],
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
    }
    _write(root / "run_manifest.json", manifest)
    _write(root / "result.json", {"schema": "bayesfilter.neutra.three_mode_mixing_result.v1", "manifest": manifest, "rows": rows, "wall_seconds": time.perf_counter() - started})
    _write(root / "artifact_hashes.json", {"schema": "bayesfilter.neutra.three_mode_mixing_hashes.v1", "artifacts": {p.name: _sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"output_root": root.as_posix(), "rows": len(rows), "wall_seconds": time.perf_counter() - started}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
