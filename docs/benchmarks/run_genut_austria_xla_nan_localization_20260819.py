#!/usr/bin/env python3
"""XLA nonfiniteness localization for the Austria GenUT endpoint.

Diagnostic lane under
docs/plans/bayesfilter-austria-genut-xla-nan-localization-plan-2026-08-19.md.
Same frozen target/guards/manifest as the root-cause runner (by import), but
serializes the FULL diagnostics dict returned by batch_finite_value and
batch_finite_value_score, which the frozen endpoint runner discards. Also
supports a TF32-off reference arm. Promotes nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import (  # noqa: E402
    run_genut_austria_endpoint_root_cause_20260817 as base,
)

import tensorflow as tf  # noqa: E402

batch = base.batch  # the production module, imported once by the base runner


def _parse_case(text: str) -> tuple[int, int, str]:
    horizon_text, steps_text, mode = text.split(":")
    horizon = int(horizon_text)
    steps = int(steps_text)
    if mode not in ("eager", "graph", "xla"):
        raise ValueError(f"unknown mode in case {text!r}")
    if not 1 <= horizon <= 20:
        raise ValueError(f"horizon out of frozen range in case {text!r}")
    if steps < 0:
        raise ValueError(f"negative correction steps in case {text!r}")
    return horizon, steps, mode


def _diag_summary(diagnostics: dict[str, tf.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, tensor in diagnostics.items():
        tensor = tf.convert_to_tensor(tensor)
        if tensor.dtype == tf.bool:
            out[key] = base._json_value(tensor)  # noqa: SLF001
        else:
            out[key] = {
                "values": base._json_value(tensor),  # noqa: SLF001
                "all_finite": bool(
                    tf.reduce_all(tf.math.is_finite(tensor)).numpy()
                ),
            }
    return out


def _diagnostic_endpoint(
    target: Any,
    *,
    horizon: int,
    correction_steps: int,
    mode: str,
    control_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    theta = tf.zeros([1, target.parameter_dim], tf.float32)
    observations = target.observations[:horizon]
    process_noise = target.process_noise[:horizon]
    kwargs = base._controls(target, correction_steps=correction_steps)  # noqa: SLF001
    if control_overrides:
        for key, value in control_overrides.items():
            if key not in kwargs:
                raise ValueError(f"unknown control override {key!r}")
            kwargs[key] = value

    def value_call(values):
        return batch.batch_finite_value(
            target.filter_adapter,
            values,
            observations,
            target.initial_noise,
            process_noise,
            target.design,
            **kwargs,
        )

    def score_call(values):
        return batch.batch_finite_value_score(
            target.filter_adapter,
            values,
            observations,
            target.initial_noise,
            process_noise,
            target.design,
            **kwargs,
        )

    started = time.time()
    if mode == "eager":
        value, value_diag = value_call(theta)
        score_value, score, score_diag = score_call(theta)
    else:
        jit_compile = mode == "xla"
        value_graph = tf.function(
            value_call, jit_compile=jit_compile, autograph=False
        )
        score_graph = tf.function(
            score_call, jit_compile=jit_compile, autograph=False
        )
        value, value_diag = value_graph(theta)
        score_value, score, score_diag = score_graph(theta)
    return {
        "mode": mode,
        "horizon": horizon,
        "correction_steps": correction_steps,
        "value_only": base._summary(value),  # noqa: SLF001
        "value_score_value": base._summary(score_value),  # noqa: SLF001
        "score": base._summary(score),  # noqa: SLF001
        "value_comparison": base._comparison(value, score_value),  # noqa: SLF001
        "value_only_diagnostics": _diag_summary(value_diag),
        "value_score_diagnostics": _diag_summary(score_diag),
        "case_wall_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-index", choices=("0", "1"), default="0")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument(
        "--tf32",
        choices=("on", "off"),
        default="on",
        help="off disables TF32 execution before any case runs (reference arm)",
    )
    parser.add_argument(
        "--lm-damping",
        type=float,
        default=None,
        help="override higher_moment_lm_damping (Class C evaluation arm)",
    )
    parser.add_argument(
        "--lm-scale-floor",
        type=float,
        default=None,
        help="override higher_moment_lm_scale_floor",
    )
    parser.add_argument(
        "--trust-radius",
        type=float,
        default=None,
        help="override higher_moment_trust_radius (Class C evaluation arm)",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.time()
    if args.device != base._EARLY_DEVICE:  # noqa: SLF001
        raise RuntimeError("parsed device does not match pre-import device selection")
    if args.tf32 == "off":
        tf.config.experimental.enable_tensor_float_32_execution(False)
    control_overrides: dict[str, float] = {}
    if args.lm_damping is not None:
        control_overrides["higher_moment_lm_damping"] = float(args.lm_damping)
    if args.lm_scale_floor is not None:
        control_overrides["higher_moment_lm_scale_floor"] = float(
            args.lm_scale_floor
        )
    if args.trust_radius is not None:
        control_overrides["higher_moment_trust_radius"] = float(
            args.trust_radius
        )

    with tf.device("/CPU:0"):
        target = base.make_genut_neutra_target("austria_sir", particle_count=1008)
    target_hashes = {
        "observations": base._tensor_hash(target.observations),  # noqa: SLF001
        "initial_noise": base._tensor_hash(target.initial_noise),  # noqa: SLF001
        "process_noise": base._tensor_hash(target.process_noise),  # noqa: SLF001
        "design": base._tensor_hash(target.design),  # noqa: SLF001
    }
    if target.target_signature != base.EXPECTED_TARGET_SIGNATURE:
        raise RuntimeError("frozen target signature mismatch")
    if target.adapter_signature() != base.EXPECTED_ADAPTER_SIGNATURE:
        raise RuntimeError("frozen adapter signature mismatch")
    if target_hashes != base.EXPECTED_TARGET_HASHES:
        raise RuntimeError("frozen target tensor hash mismatch")

    source_paths = (
        ROOT / "bayesfilter/highdim/cubature_genut_batch_tf.py",
        ROOT / "bayesfilter/highdim/cubature_genut_neutra_targets.py",
        ROOT / "bayesfilter/highdim/cubature_genut_batch_adapters.py",
        ROOT / "bayesfilter/highdim/higher_moment_contract_e.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        ROOT / "bayesfilter/highdim/cubature_genut_filter.py",
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
    )
    payload: dict[str, Any] = base._manifest(  # noqa: SLF001
        output=output,
        device=args.device,
        memory_policy=base._MEMORY_POLICY,  # noqa: SLF001
        started=started,
    )
    payload.update(
        {
            "schema": "bayesfilter.genut_austria_xla_nan_localization_result.v1",
            "plan": (
                "docs/plans/"
                "bayesfilter-austria-genut-xla-nan-localization-plan-2026-08-19.md"
            ),
            "source_sha256": {
                base._display_path(path): base._sha256_file(path)  # noqa: SLF001
                for path in source_paths
            },
            "target_signature": target.target_signature,
            "adapter_signature": target.adapter_signature(),
            "target_construction_device": "/CPU:0",
            "execution_device": "/GPU:0" if args.device == "gpu" else "/CPU:0",
            "target_hashes": target_hashes,
            "frozen_identity_guard": "PASS",
            "route_classification": "batch_diagonal_candidate",
            "tf32_requested": args.tf32,
            "tf32_enabled_at_run": (
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "control_overrides": control_overrides,
            "results": {"cases": []},
        }
    )
    base._write_json(output, payload | {"status": "RUNNING"})  # noqa: SLF001
    try:
        execution_device = "/GPU:0" if args.device == "gpu" else "/CPU:0"
        with tf.device(execution_device):
            for text in args.cases:
                horizon, steps, mode = _parse_case(text)
                payload["results"]["cases"].append(
                    _diagnostic_endpoint(
                        target,
                        horizon=horizon,
                        correction_steps=steps,
                        mode=mode,
                        control_overrides=control_overrides or None,
                    )
                )
                base._write_json(output, payload | {"status": "RUNNING"})  # noqa: SLF001
        payload["status"] = "COMPLETE"
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        payload["wall_seconds"] = time.time() - started
        base._write_json(output, payload)  # noqa: SLF001


if __name__ == "__main__":
    main()
