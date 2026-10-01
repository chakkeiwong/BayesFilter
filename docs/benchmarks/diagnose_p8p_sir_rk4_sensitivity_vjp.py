"""Transition-only P8p SIR RK4 VJP diagnostic.

This Phase 3 diagnostic compares the hand-coded SIR RHS/RK4 reverse pass in
``benchmark_p8p_parameterized_sir_gradient`` against TensorFlow autodiff on
identical fixed tensors.  It is a CPU-only algebra check and does not exercise
LEDH transport, resampling, GPU/TF32 behavior, or HMC suitability.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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


DEFAULT_MAX_ABS = 1.0e-8
DEFAULT_REL_L2 = 1.0e-7
NONCLAIMS = (
    "transition-only RK4/RHS VJP algebra check",
    "not LEDH transport adjoint correctness",
    "not stopped-scale-key correctness",
    "not GPU/TF32 runtime evidence",
    "not full SIR gradient correctness",
    "not HMC readiness",
)


def _parse_float_csv(value: str, *, expected: int) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parsed) != expected:
        raise ValueError(f"expected {expected} comma-separated floats")
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError("theta entries must be finite")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-particles", type=int, default=3)
    parser.add_argument("--theta", default="0.02,-0.01,0.01")
    parser.add_argument("--dtype", choices=("float64",), default="float64")
    parser.add_argument("--tf32-mode", choices=("disabled",), default="disabled")
    parser.add_argument("--device-scope", choices=("cpu",), default=_PRE_ARGS.device_scope)
    parser.add_argument("--max-abs-tolerance", type=float, default=DEFAULT_MAX_ABS)
    parser.add_argument("--rel-l2-tolerance", type=float, default=DEFAULT_REL_L2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.theta_values = _parse_float_csv(args.theta, expected=3)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_particles <= 0:
        raise ValueError("num-particles must be positive")
    if args.max_abs_tolerance < 0.0 or args.rel_l2_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")
    return args


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


def _configure_float64_cpu() -> dict[str, Any]:
    args = argparse.Namespace(dtype="float64", tf32_mode="disabled")
    return p8p._configure_precision(args)


def _fixed_points(batch_size: int, num_particles: int) -> tf.Tensor:
    base = tf.cast(p8p._SIR_MODEL.initial_mean, p8p.DTYPE)  # noqa: SLF001
    total = batch_size * num_particles * int(base.shape[0])
    offsets = tf.reshape(
        tf.linspace(
            tf.constant(-0.035, dtype=p8p.DTYPE),
            tf.constant(0.035, dtype=p8p.DTYPE),
            total,
        ),
        [batch_size, num_particles, int(base.shape[0])],
    )
    batch_offsets = tf.reshape(
        tf.linspace(
            tf.constant(-0.01, dtype=p8p.DTYPE),
            tf.constant(0.01, dtype=p8p.DTYPE),
            batch_size,
        ),
        [batch_size, 1, 1],
    )
    return base[tf.newaxis, tf.newaxis, :] + offsets + batch_offsets


def _fixed_upstream(batch_size: int, num_particles: int) -> tf.Tensor:
    state_dim = 18
    total = batch_size * num_particles * state_dim
    raw = tf.cast(tf.range(total), p8p.DTYPE)
    values = tf.sin(raw * tf.constant(0.17, dtype=p8p.DTYPE)) + tf.cos(
        raw * tf.constant(0.07, dtype=p8p.DTYPE)
    )
    return tf.reshape(values, [batch_size, num_particles, state_dim])


def _parameters(theta_values: list[float]) -> dict[str, tf.Tensor]:
    scaled = p8p._scaled_parameters(p8p._theta_components(theta_values))
    return {
        "kappa": tf.convert_to_tensor(scaled["kappa"], dtype=p8p.DTYPE),
        "nu": tf.convert_to_tensor(scaled["nu"], dtype=p8p.DTYPE),
        "adjacency": tf.cast(p8p._SIR_ADJACENCY_MATRIX, p8p.DTYPE),  # noqa: SLF001
        "neighbor_degree": tf.cast(p8p._SIR_NEIGHBOR_DEGREE, p8p.DTYPE),  # noqa: SLF001
        "step_size": tf.cast(p8p._SIR_DELTA, p8p.DTYPE)  # noqa: SLF001
        / tf.cast(int(p8p._SIR_RK4_SUBSTEPS), p8p.DTYPE),  # noqa: SLF001
    }


def _as_float(value: tf.Tensor) -> float:
    return float(tf.convert_to_tensor(value, dtype=p8p.DTYPE).numpy())


def _comparison(
    name: str,
    manual: tf.Tensor,
    reference: tf.Tensor,
    *,
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
        "shape": [int(dim) for dim in manual.shape],
        "reference_shape": [int(dim) for dim in reference.shape],
        "max_abs_residual": _as_float(max_abs),
        "relative_l2_residual": _as_float(rel_l2),
        "reference_l2_norm": _as_float(ref_norm),
        "finite": finite,
        "passed": bool(passed),
    }


def _loss_per_batch(output: tf.Tensor, upstream: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(output * upstream, axis=[1, 2])


def _rhs_autodiff(
    points: tf.Tensor,
    upstream: tf.Tensor,
    params: dict[str, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    points = tf.identity(points)
    kappa = tf.identity(params["kappa"])
    nu = tf.identity(params["nu"])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch((points, kappa, nu))
        rhs = p8p._sir_rhs_tf(
            points,
            kappa=kappa,
            nu=nu,
            adjacency=params["adjacency"],
            neighbor_degree=params["neighbor_degree"],
        )
        per_batch_loss = _loss_per_batch(rhs, upstream)
        total_loss = tf.reduce_sum(per_batch_loss)
    bar_state = tape.gradient(total_loss, points)
    bar_kappa = tape.jacobian(per_batch_loss, kappa, experimental_use_pfor=False)
    bar_nu = tape.jacobian(per_batch_loss, nu, experimental_use_pfor=False)
    del tape
    if bar_state is None or bar_kappa is None or bar_nu is None:
        raise RuntimeError("RHS autodiff comparator produced a disconnected gradient")
    return (
        tf.convert_to_tensor(bar_state, dtype=p8p.DTYPE),
        tf.convert_to_tensor(bar_kappa, dtype=p8p.DTYPE),
        tf.convert_to_tensor(bar_nu, dtype=p8p.DTYPE),
    )


def _transition_autodiff(
    points: tf.Tensor,
    upstream: tf.Tensor,
    params: dict[str, tf.Tensor],
    *,
    substeps: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    points = tf.identity(points)
    kappa = tf.identity(params["kappa"])
    nu = tf.identity(params["nu"])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch((points, kappa, nu))
        transition_mean, _aux = p8p._sir_transition_mean_with_aux_tf(
            points,
            kappa=kappa,
            nu=nu,
            adjacency=params["adjacency"],
            neighbor_degree=params["neighbor_degree"],
            substeps=substeps,
            step_size=params["step_size"],
        )
        per_batch_loss = _loss_per_batch(transition_mean, upstream)
        total_loss = tf.reduce_sum(per_batch_loss)
    bar_state = tape.gradient(total_loss, points)
    bar_kappa = tape.jacobian(per_batch_loss, kappa, experimental_use_pfor=False)
    bar_nu = tape.jacobian(per_batch_loss, nu, experimental_use_pfor=False)
    del tape
    if bar_state is None or bar_kappa is None or bar_nu is None:
        raise RuntimeError("transition autodiff comparator produced a disconnected gradient")
    return (
        tf.convert_to_tensor(bar_state, dtype=p8p.DTYPE),
        tf.convert_to_tensor(bar_kappa, dtype=p8p.DTYPE),
        tf.convert_to_tensor(bar_nu, dtype=p8p.DTYPE),
    )


def _regional_log_autodiff(
    points: tf.Tensor,
    upstream: tf.Tensor,
    theta_values: list[float],
    *,
    parameter: str,
    substeps: int,
) -> tf.Tensor:
    log_values = tf.fill([9], tf.constant(theta_values[0 if parameter == "kappa" else 1], dtype=p8p.DTYPE))
    other_theta = tf.constant(theta_values[1 if parameter == "kappa" else 0], dtype=p8p.DTYPE)
    base_kappa = tf.cast(p8p._SIR_BASE_KAPPA, p8p.DTYPE)  # noqa: SLF001
    base_nu = tf.cast(p8p._SIR_BASE_NU, p8p.DTYPE)  # noqa: SLF001
    adjacency = tf.cast(p8p._SIR_ADJACENCY_MATRIX, p8p.DTYPE)  # noqa: SLF001
    neighbor_degree = tf.cast(p8p._SIR_NEIGHBOR_DEGREE, p8p.DTYPE)  # noqa: SLF001
    step_size = tf.cast(p8p._SIR_DELTA, p8p.DTYPE) / tf.cast(  # noqa: SLF001
        int(p8p._SIR_RK4_SUBSTEPS),  # noqa: SLF001
        p8p.DTYPE,
    )
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(log_values)
        if parameter == "kappa":
            kappa = base_kappa * tf.exp(log_values)
            nu = base_nu * tf.exp(other_theta)
        elif parameter == "nu":
            kappa = base_kappa * tf.exp(other_theta)
            nu = base_nu * tf.exp(log_values)
        else:
            raise ValueError("parameter must be kappa or nu")
        transition_mean, _aux = p8p._sir_transition_mean_with_aux_tf(
            points,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
            substeps=substeps,
            step_size=step_size,
        )
        per_batch_loss = _loss_per_batch(transition_mean, upstream)
    jacobian = tape.jacobian(per_batch_loss, log_values, experimental_use_pfor=False)
    del tape
    if jacobian is None:
        raise RuntimeError(f"regional log-{parameter} comparator is disconnected")
    return tf.convert_to_tensor(jacobian, dtype=p8p.DTYPE)


def _manual_rhs(
    points: tf.Tensor,
    upstream: tf.Tensor,
    params: dict[str, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return p8p._sir_rhs_vjp_tf(
        points,
        upstream,
        kappa=params["kappa"],
        nu=params["nu"],
        adjacency=params["adjacency"],
        neighbor_degree=params["neighbor_degree"],
    )


def _manual_transition(
    points: tf.Tensor,
    upstream: tf.Tensor,
    params: dict[str, tf.Tensor],
    *,
    substeps: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    _transition_mean, aux = p8p._sir_transition_mean_with_aux_tf(
        points,
        kappa=params["kappa"],
        nu=params["nu"],
        adjacency=params["adjacency"],
        neighbor_degree=params["neighbor_degree"],
        substeps=substeps,
        step_size=params["step_size"],
    )
    return p8p._sir_transition_mean_vjp_tf(
        aux,
        upstream,
        kappa=params["kappa"],
        nu=params["nu"],
        adjacency=params["adjacency"],
        neighbor_degree=params["neighbor_degree"],
        step_size=params["step_size"],
    )


def _compare_vjp_family(
    *,
    prefix: str,
    manual: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    reference: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    params: dict[str, tf.Tensor],
    max_abs_tolerance: float,
    rel_l2_tolerance: float,
) -> list[dict[str, Any]]:
    bar_state, bar_kappa, bar_nu = manual
    ref_state, ref_kappa, ref_nu = reference
    return [
        _comparison(
            f"{prefix}.bar_state",
            bar_state,
            ref_state,
            max_abs_tolerance=max_abs_tolerance,
            rel_l2_tolerance=rel_l2_tolerance,
        ),
        _comparison(
            f"{prefix}.bar_kappa",
            bar_kappa,
            ref_kappa,
            max_abs_tolerance=max_abs_tolerance,
            rel_l2_tolerance=rel_l2_tolerance,
        ),
        _comparison(
            f"{prefix}.bar_nu",
            bar_nu,
            ref_nu,
            max_abs_tolerance=max_abs_tolerance,
            rel_l2_tolerance=rel_l2_tolerance,
        ),
        _comparison(
            f"{prefix}.regional_log_kappa_contraction",
            p8p._regional_kappa_score_from_cotangent(
                kappa=params["kappa"],
                bar_kappa=bar_kappa,
            ),
            p8p._regional_kappa_score_from_cotangent(
                kappa=params["kappa"],
                bar_kappa=ref_kappa,
            ),
            max_abs_tolerance=max_abs_tolerance,
            rel_l2_tolerance=rel_l2_tolerance,
        ),
        _comparison(
            f"{prefix}.regional_log_nu_contraction",
            p8p._regional_nu_score_from_cotangent(
                nu=params["nu"],
                bar_nu=bar_nu,
            ),
            p8p._regional_nu_score_from_cotangent(
                nu=params["nu"],
                bar_nu=ref_nu,
            ),
            max_abs_tolerance=max_abs_tolerance,
            rel_l2_tolerance=rel_l2_tolerance,
        ),
    ]


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    precision = _configure_float64_cpu()
    started = time.time()
    points = _fixed_points(args.batch_size, args.num_particles)
    upstream = _fixed_upstream(args.batch_size, args.num_particles)
    params = _parameters(args.theta_values)
    full_substeps = int(p8p._SIR_RK4_SUBSTEPS)  # noqa: SLF001

    rhs_manual = _manual_rhs(points, upstream, params)
    rhs_reference = _rhs_autodiff(points, upstream, params)
    one_step_manual = _manual_transition(points, upstream, params, substeps=1)
    one_step_reference = _transition_autodiff(points, upstream, params, substeps=1)
    full_manual = _manual_transition(points, upstream, params, substeps=full_substeps)
    full_reference = _transition_autodiff(points, upstream, params, substeps=full_substeps)

    comparisons = []
    comparisons.extend(
        _compare_vjp_family(
            prefix="rhs",
            manual=rhs_manual,
            reference=rhs_reference,
            params=params,
            max_abs_tolerance=args.max_abs_tolerance,
            rel_l2_tolerance=args.rel_l2_tolerance,
        )
    )
    comparisons.extend(
        _compare_vjp_family(
            prefix="rk4_one_step",
            manual=one_step_manual,
            reference=one_step_reference,
            params=params,
            max_abs_tolerance=args.max_abs_tolerance,
            rel_l2_tolerance=args.rel_l2_tolerance,
        )
    )
    comparisons.extend(
        _compare_vjp_family(
            prefix="rk4_full_scan",
            manual=full_manual,
            reference=full_reference,
            params=params,
            max_abs_tolerance=args.max_abs_tolerance,
            rel_l2_tolerance=args.rel_l2_tolerance,
        )
    )

    regional_kappa_reference = _regional_log_autodiff(
        points,
        upstream,
        args.theta_values,
        parameter="kappa",
        substeps=full_substeps,
    )
    regional_nu_reference = _regional_log_autodiff(
        points,
        upstream,
        args.theta_values,
        parameter="nu",
        substeps=full_substeps,
    )
    comparisons.append(
        _comparison(
            "rk4_full_scan.regional_log_kappa_chain_rule_vs_regional_autodiff",
            p8p._regional_kappa_score_from_cotangent(
                kappa=params["kappa"],
                bar_kappa=full_manual[1],
            ),
            regional_kappa_reference,
            max_abs_tolerance=args.max_abs_tolerance,
            rel_l2_tolerance=args.rel_l2_tolerance,
        )
    )
    comparisons.append(
        _comparison(
            "rk4_full_scan.regional_log_nu_chain_rule_vs_regional_autodiff",
            p8p._regional_nu_score_from_cotangent(
                nu=params["nu"],
                bar_nu=full_manual[2],
            ),
            regional_nu_reference,
            max_abs_tolerance=args.max_abs_tolerance,
            rel_l2_tolerance=args.rel_l2_tolerance,
        )
    )

    localization = "pass"
    if not all(item["passed"] for item in comparisons):
        failed = [item["name"] for item in comparisons if not item["passed"]]
        if any(name.startswith("rhs.") for name in failed):
            localization = "rhs"
        elif any(name.startswith("rk4_one_step.") for name in failed):
            localization = "rk4_step"
        elif any(name.startswith("rk4_full_scan.") for name in failed):
            localization = "multi_substep_scan"
        else:
            localization = "regional_chain_rule"

    devices = sorted({points.device, upstream.device, *[tensor.device for tensor in params.values()]})
    passed = bool(all(item["passed"] for item in comparisons))
    return {
        "phase": "sir_gradient_reparam_rootcause_phase3_rk4_sensitivity",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
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
            "question": "manual SIR RHS/RK4 transition VJP vs TensorFlow autodiff before LEDH transport",
            "comparator": "TensorFlow autodiff on identical fixed tensors, float64 CPU",
            "max_abs_tolerance": args.max_abs_tolerance,
            "rel_l2_tolerance": args.rel_l2_tolerance,
            "relative_l2_denominator": "max(norm(comparator), 1.0)",
            "nonclaims": list(NONCLAIMS),
        },
        "configuration": {
            "batch_size": args.batch_size,
            "num_particles": args.num_particles,
            "theta": list(args.theta_values),
            "full_substeps": full_substeps,
            "state_dim": int(points.shape[-1]),
            "region_count": int(params["kappa"].shape[0]),
        },
        "comparisons": comparisons,
    }


def main() -> int:
    args = _parse_args()
    payload = run_diagnostic(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
