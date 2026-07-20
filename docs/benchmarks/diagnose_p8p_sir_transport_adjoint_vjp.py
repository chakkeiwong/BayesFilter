"""P8p transport-adjoint VJP diagnostic.

This Phase 4 diagnostic compares ``p8p._manual_transport_vjp_tf`` against
TensorFlow autodiff of the same stopped-scale-key forward transport, using the
non-custom-gradient value helper
``annealed_transport_tf._filterflow_manual_streaming_finite_transport_value_stopped_scale_keys``.

It is a CPU-only local transport algebra check.  It does not exercise the full
SIR filter, GPU/TF32 execution, or HMC suitability.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu",), default="cpu")
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DEFAULT_MAX_ABS = 1.0e-8
DEFAULT_REL_L2 = 1.0e-7
FORBIDDEN_COMPARATOR_ROUTES = (
    "_filterflow_manual_streaming_finite_transport_stopped_scale_keys",
    "_filterflow_manual_streaming_blockwise_vjp_finite_transport_stopped_scale_keys",
)
COMPARATOR_SYMBOL = (
    "experiments/dpf_implementation/tf_tfp/resampling/"
    "annealed_transport_tf.py::_filterflow_manual_streaming_finite_transport_value_stopped_scale_keys"
)
NONCLAIMS = (
    "transport wrapper algebra check only",
    "not full SIR score correctness",
    "not GPU/TF32 runtime evidence",
    "not HMC readiness",
    "not non-centered process-noise evidence",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.7)
    parser.add_argument("--annealed-scaling", type=float, default=0.8)
    parser.add_argument("--row-chunk-size", type=int, default=2)
    parser.add_argument("--col-chunk-size", type=int, default=2)
    parser.add_argument("--dtype", choices=("float64",), default="float64")
    parser.add_argument("--tf32-mode", choices=("disabled",), default="disabled")
    parser.add_argument("--device-scope", choices=("cpu",), default=_PRE_ARGS.device_scope)
    parser.add_argument("--max-abs-tolerance", type=float, default=DEFAULT_MAX_ABS)
    parser.add_argument("--rel-l2-tolerance", type=float, default=DEFAULT_REL_L2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    if args.state_dim <= 0:
        raise ValueError("state-dim must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn-iterations must be positive")
    if args.sinkhorn_epsilon <= 0.0:
        raise ValueError("sinkhorn-epsilon must be positive")
    if not 0.0 < args.annealed_scaling <= 1.0:
        raise ValueError("annealed-scaling must be in (0, 1]")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.max_abs_tolerance < 0.0 or args.rel_l2_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")
    return args


def _configure_float64_cpu() -> dict[str, Any]:
    args = argparse.Namespace(dtype="float64", tf32_mode="disabled")
    return p8p._configure_precision(args)


def _transport_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        transport_plan_mode="streaming",
        transport_gradient_mode=p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_ad_mode="stabilized",
        transport_policy="active-all",
        sinkhorn_iterations=int(args.sinkhorn_iterations),
        sinkhorn_epsilon=float(args.sinkhorn_epsilon),
        annealed_scaling=float(args.annealed_scaling),
        row_chunk_size=int(args.row_chunk_size),
        col_chunk_size=int(args.col_chunk_size),
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def _fixed_post_flow(batch_size: int, num_particles: int, state_dim: int) -> tf.Tensor:
    total = batch_size * num_particles * state_dim
    values = tf.reshape(
        tf.linspace(
            tf.constant(-0.45, p8p.DTYPE),
            tf.constant(0.55, p8p.DTYPE),
            total,
        ),
        [batch_size, num_particles, state_dim],
    )
    batch_shift = tf.reshape(
        tf.linspace(
            tf.constant(0.0, p8p.DTYPE),
            tf.constant(0.08, p8p.DTYPE),
            batch_size,
        ),
        [batch_size, 1, 1],
    )
    return values + batch_shift


def _fixed_log_weights(batch_size: int, num_particles: int) -> tf.Tensor:
    raw = tf.reshape(
        tf.linspace(
            tf.constant(-0.35, p8p.DTYPE),
            tf.constant(0.25, p8p.DTYPE),
            batch_size * num_particles,
        ),
        [batch_size, num_particles],
    )
    return raw - tf.reduce_logsumexp(raw, axis=1, keepdims=True)


def _fixed_particle_upstream(batch_size: int, num_particles: int, state_dim: int) -> tf.Tensor:
    total = batch_size * num_particles * state_dim
    raw = tf.cast(tf.range(total), p8p.DTYPE)
    values = tf.sin(raw * tf.constant(0.13, p8p.DTYPE)) - tf.cos(
        raw * tf.constant(0.19, p8p.DTYPE)
    )
    return tf.reshape(values, [batch_size, num_particles, state_dim])


def _fixed_log_weight_upstream(batch_size: int, num_particles: int) -> tf.Tensor:
    total = batch_size * num_particles
    raw = tf.cast(tf.range(total), p8p.DTYPE)
    values = tf.cos(raw * tf.constant(0.23, p8p.DTYPE))
    return tf.reshape(values, [batch_size, num_particles])


def _mask_cases(batch_size: int) -> dict[str, tf.Tensor]:
    if batch_size == 1:
        mixed = tf.constant([True], dtype=tf.bool)
    else:
        mixed_values = [index % 2 == 0 for index in range(batch_size)]
        mixed = tf.constant(mixed_values, dtype=tf.bool)
    return {
        "active_all": tf.ones([batch_size], dtype=tf.bool),
        "inactive_all": tf.zeros([batch_size], dtype=tf.bool),
        "mixed": mixed,
    }


@contextlib.contextmanager
def _forbidden_comparator_route_guard() -> Iterator[dict[str, int]]:
    calls = {name: 0 for name in FORBIDDEN_COMPARATOR_ROUTES}
    originals = {
        name: getattr(annealed_transport_tf, name)
        for name in FORBIDDEN_COMPARATOR_ROUTES
    }

    def make_forbidden(name: str):
        def forbidden(*_args: Any, **_kwargs: Any):
            calls[name] += 1
            raise RuntimeError(f"forbidden comparator route called: {name}")

        return forbidden

    try:
        for name in FORBIDDEN_COMPARATOR_ROUTES:
            setattr(annealed_transport_tf, name, make_forbidden(name))
        yield calls
    finally:
        for name, original in originals.items():
            setattr(annealed_transport_tf, name, original)


def _noncustom_stopped_scale_transport(
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    args: argparse.Namespace,
) -> tuple[tf.Tensor, tf.Tensor]:
    center = tf.stop_gradient(tf.reduce_mean(post_flow, axis=1, keepdims=True))
    scale = tf.stop_gradient(annealed_transport_tf._filterflow_scale(post_flow))  # noqa: SLF001
    scaled_x = (post_flow - center) / scale[:, None, None]
    epsilon = tf.convert_to_tensor(args.sinkhorn_epsilon, dtype=p8p.DTYPE)
    epsilon0 = tf.stop_gradient(annealed_transport_tf._filterflow_epsilon_start(scaled_x))  # noqa: SLF001
    scaling = tf.convert_to_tensor(args.annealed_scaling, dtype=p8p.DTYPE)
    steps = p8p.core_tf._manual_dense_finite_steps(args.sinkhorn_iterations)  # noqa: SLF001
    return annealed_transport_tf._filterflow_manual_streaming_finite_transport_value_stopped_scale_keys(  # noqa: SLF001
        scaled_x,
        post_flow,
        normalized_log_weights,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )


def _autodiff_transport_contribution_vjp(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
    upstream_particles: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, int]]:
    with _forbidden_comparator_route_guard() as forbidden_calls:
        with tf.GradientTape() as tape:
            tape.watch([post_flow, normalized_log_weights])
            transported, _row_residual = _noncustom_stopped_scale_transport(
                post_flow,
                normalized_log_weights,
                args,
            )
            active_transport = tf.where(
                mask[:, None, None],
                transported,
                tf.zeros_like(transported),
            )
            scalar = tf.reduce_sum(active_transport * upstream_particles)
        bar_post, bar_logw = tape.gradient(
            scalar,
            [post_flow, normalized_log_weights],
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
    return (
        tf.convert_to_tensor(bar_post, dtype=p8p.DTYPE),
        tf.convert_to_tensor(bar_logw, dtype=p8p.DTYPE),
        dict(forbidden_calls),
    )


def _autodiff_full_masked_step_vjp(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
    upstream_particles: tf.Tensor,
    upstream_log_weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, int]]:
    with _forbidden_comparator_route_guard() as forbidden_calls:
        with tf.GradientTape() as tape:
            tape.watch([post_flow, normalized_log_weights])
            transported, _row_residual = _noncustom_stopped_scale_transport(
                post_flow,
                normalized_log_weights,
                args,
            )
            batch_size = int(post_flow.shape[0])
            num_particles = int(post_flow.shape[1])
            uniform_log = p8p.core_tf.uniform_log_weights(batch_size, num_particles)
            next_particles = tf.where(mask[:, None, None], transported, post_flow)
            next_log_weights = tf.where(mask[:, None], uniform_log, normalized_log_weights)
            scalar = (
                tf.reduce_sum(next_particles * upstream_particles)
                + tf.reduce_sum(next_log_weights * upstream_log_weights)
            )
        bar_post, bar_logw = tape.gradient(
            scalar,
            [post_flow, normalized_log_weights],
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
    return (
        tf.convert_to_tensor(bar_post, dtype=p8p.DTYPE),
        tf.convert_to_tensor(bar_logw, dtype=p8p.DTYPE),
        dict(forbidden_calls),
    )


def _manual_transport_contribution_vjp(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
    upstream_particles: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    return p8p._manual_transport_vjp_tf(
        post_flow=post_flow,
        normalized_log_weights=normalized_log_weights,
        mask=mask,
        args=args,
        upstream_particles=upstream_particles,
    )


def _manual_full_masked_step_vjp(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
    upstream_particles: tf.Tensor,
    upstream_log_weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    bar_post, bar_logw = _manual_transport_contribution_vjp(
        post_flow=post_flow,
        normalized_log_weights=normalized_log_weights,
        mask=mask,
        args=args,
        upstream_particles=upstream_particles,
    )
    inactive = tf.logical_not(mask)
    return (
        bar_post
        + tf.where(
            inactive[:, None, None],
            upstream_particles,
            tf.zeros_like(upstream_particles),
        ),
        bar_logw
        + tf.where(
            inactive[:, None],
            upstream_log_weights,
            tf.zeros_like(upstream_log_weights),
        ),
    )


def _as_float(value: tf.Tensor) -> float:
    return float(tf.convert_to_tensor(value, dtype=p8p.DTYPE).numpy())


def _comparison(
    *,
    name: str,
    mask_name: str,
    tensor_family: str,
    manual: tf.Tensor,
    reference: tf.Tensor,
    max_abs_tolerance: float,
    rel_l2_tolerance: float,
) -> dict[str, Any]:
    manual = tf.convert_to_tensor(manual, dtype=p8p.DTYPE)
    reference = tf.convert_to_tensor(reference, dtype=p8p.DTYPE)
    diff = manual - reference
    max_abs = tf.reduce_max(tf.abs(diff))
    ref_norm = tf.linalg.norm(tf.reshape(reference, [-1]))
    denominator = tf.maximum(ref_norm, tf.constant(1.0, dtype=p8p.DTYPE))
    rel_l2 = tf.linalg.norm(tf.reshape(diff, [-1])) / denominator
    finite = bool(
        (
            tf.reduce_all(tf.math.is_finite(manual))
            & tf.reduce_all(tf.math.is_finite(reference))
            & tf.reduce_all(tf.math.is_finite(diff))
        ).numpy()
    )
    passed = (
        finite
        and _as_float(max_abs) <= max_abs_tolerance
        and _as_float(rel_l2) <= rel_l2_tolerance
    )
    return {
        "name": name,
        "mask_case": mask_name,
        "tensor_family": tensor_family,
        "shape": [int(dim) for dim in manual.shape],
        "reference_shape": [int(dim) for dim in reference.shape],
        "max_abs_residual": _as_float(max_abs),
        "relative_l2_residual": _as_float(rel_l2),
        "reference_l2_norm": _as_float(ref_norm),
        "finite": finite,
        "passed": bool(passed),
    }


def _localize_failure(comparisons: list[dict[str, Any]]) -> str:
    failed = [item for item in comparisons if not item["passed"]]
    if not failed:
        return "pass"
    failed_masks = {item["mask_case"] for item in failed}
    failed_families = {item["tensor_family"] for item in failed}
    if failed_masks == {"active_all"}:
        return "active_mask_handling"
    if failed_masks == {"inactive_all"}:
        return "inactive_mask_handling"
    if "bar_normalized_log_weights" in failed_families and "bar_post_flow" not in failed_families:
        return "log_weight_cotangent"
    if "bar_post_flow" in failed_families and "bar_normalized_log_weights" not in failed_families:
        return "particle_or_scaled_particle_cotangent"
    if len(failed_masks) == 1 and "mixed" in failed_masks:
        return "mixed_mask_handling"
    return "diffuse_transport_mismatch"


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    precision = _configure_float64_cpu()
    started = time.time()
    transport_args = _transport_args(args)
    post_flow = _fixed_post_flow(args.batch_size, args.num_particles, args.state_dim)
    normalized_log_weights = _fixed_log_weights(args.batch_size, args.num_particles)
    upstream_particles = _fixed_particle_upstream(
        args.batch_size,
        args.num_particles,
        args.state_dim,
    )
    upstream_log_weights = _fixed_log_weight_upstream(args.batch_size, args.num_particles)

    comparisons: list[dict[str, Any]] = []
    guard_calls_total = {name: 0 for name in FORBIDDEN_COMPARATOR_ROUTES}
    for mask_name, mask in _mask_cases(args.batch_size).items():
        manual_transport = _manual_transport_contribution_vjp(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=transport_args,
            upstream_particles=upstream_particles,
        )
        reference_transport = _autodiff_transport_contribution_vjp(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=transport_args,
            upstream_particles=upstream_particles,
        )
        for key, value in reference_transport[2].items():
            guard_calls_total[key] += int(value)
        manual_full = _manual_full_masked_step_vjp(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=transport_args,
            upstream_particles=upstream_particles,
            upstream_log_weights=upstream_log_weights,
        )
        reference_full = _autodiff_full_masked_step_vjp(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=transport_args,
            upstream_particles=upstream_particles,
            upstream_log_weights=upstream_log_weights,
        )
        for key, value in reference_full[2].items():
            guard_calls_total[key] += int(value)
        for prefix, manual_pair, reference_pair in (
            ("transport_contribution", manual_transport, reference_transport[:2]),
            ("caller_masked_step", manual_full, reference_full[:2]),
        ):
            comparisons.append(
                _comparison(
                    name=f"{prefix}.{mask_name}.bar_post_flow",
                    mask_name=mask_name,
                    tensor_family="bar_post_flow",
                    manual=manual_pair[0],
                    reference=reference_pair[0],
                    max_abs_tolerance=args.max_abs_tolerance,
                    rel_l2_tolerance=args.rel_l2_tolerance,
                )
            )
            comparisons.append(
                _comparison(
                    name=f"{prefix}.{mask_name}.bar_normalized_log_weights",
                    mask_name=mask_name,
                    tensor_family="bar_normalized_log_weights",
                    manual=manual_pair[1],
                    reference=reference_pair[1],
                    max_abs_tolerance=args.max_abs_tolerance,
                    rel_l2_tolerance=args.rel_l2_tolerance,
                )
            )

    guard_clean = all(count == 0 for count in guard_calls_total.values())
    passed = guard_clean and all(item["passed"] for item in comparisons)
    localization = _localize_failure(comparisons)
    if not guard_clean:
        localization = "comparator_used_forbidden_custom_gradient_route"

    devices = sorted(
        {
            post_flow.device,
            normalized_log_weights.device,
            upstream_particles.device,
            upstream_log_weights.device,
        }
    )
    return {
        "phase": "sir_gradient_reparam_rootcause_phase4_transport_adjoint",
        "status": "PASS" if passed else "FAIL",
        "passed": bool(passed),
        "failure_localization": localization,
        "created_at": _dt.datetime.now(tz=_dt.UTC).isoformat(),
        "wall_time_seconds": time.time() - started,
        "git_commit": _git_commit(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "devices": devices,
        },
        "precision": precision,
        "contract": {
            "question": "manual P8p transport VJP vs independent stopped-scale-key autodiff comparator",
            "comparator_symbol": COMPARATOR_SYMBOL,
            "forbidden_comparator_routes": list(FORBIDDEN_COMPARATOR_ROUTES),
            "max_abs_tolerance": args.max_abs_tolerance,
            "rel_l2_tolerance": args.rel_l2_tolerance,
            "relative_l2_denominator": "max(norm(comparator), 1.0)",
            "nonclaims": list(NONCLAIMS),
        },
        "configuration": {
            "batch_size": args.batch_size,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "mask_cases": {
                name: [bool(value) for value in mask.numpy().tolist()]
                for name, mask in _mask_cases(args.batch_size).items()
            },
        },
        "comparator_guard": {
            "forbidden_route_call_counts": guard_calls_total,
            "passed": bool(guard_clean),
        },
        "comparisons": comparisons,
    }


def main() -> int:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    payload = run_diagnostic(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
