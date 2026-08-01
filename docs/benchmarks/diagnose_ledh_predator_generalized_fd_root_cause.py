"""CPU/FP64 reference diagnostic for the remaining LEDH FD failures."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


# This harness is reference-only. Hide GPUs before importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.ledh_fd_policy import coordinate_relative_error  # noqa: E402
from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)


SCHEMA_VERSION = "bayesfilter.ledh.predator_generalized_fd_root_cause.v2"
PLAN_PATH = (
    "docs/plans/bayesfilter-ledh-predator-generalized-fd-root-cause-repair-subplan-2026-07-11.md"
)
RELATIVE_STEP_COEFFICIENTS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 5.0e-3, 1.0e-2)
REFERENCE_ATOL = 1.0e-8
REFERENCE_RTOL = 1.0e-6

ROW_MODULES = {
    "predator-prey": "docs.benchmarks.benchmark_ledh_same_target_predator_prey_score",
    "generalized-sv": "docs.benchmarks.benchmark_ledh_same_target_generalized_sv_score",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}:{exc}"


def _prepared_input_fingerprint(prepared: Mapping[str, Any]) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []

    def visit(path: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(f"{path}.{key}" if path else str(key), value[key])
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(f"{path}[{index}]", item)
            return
        if not tf.is_tensor(value):
            raise ValueError(f"prepared input leaf {path} must be a TensorFlow tensor")
        tensor = tf.convert_to_tensor(value)
        serialized = bytes(tf.io.serialize_tensor(tensor).numpy())
        leaves.append(
            {
                "path": path,
                "dtype": tensor.dtype.name,
                "shape": tensor.shape.as_list(),
                "sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )

    visit("", {key: value for key, value in prepared.items() if key != "semantics"})
    canonical = json.dumps(leaves, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "algorithm": "sha256_tf_serialize_tensor_tree_v1",
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "tensor_leaf_count": len(leaves),
        "tensor_leaves": leaves,
    }


def _args_for_row(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        batch_seeds=[int(cli.seed)],
        time_steps=int(cli.time_steps),
        num_particles=int(cli.num_particles),
        transport_policy="active-all",
        sinkhorn_iterations=int(cli.sinkhorn_iterations),
        sinkhorn_epsilon=1.0,
        annealed_scaling=0.9,
        annealed_convergence_threshold=1.0e-3,
        flow_observation_variance=(2.0 if cli.row == "generalized-sv" else None),
        transport_plan_mode="streaming",
        transport_gradient_mode=core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_ad_mode="full",
        row_chunk_size=int(cli.row_chunk_size),
        col_chunk_size=int(cli.col_chunk_size),
        particle_chunk_size=int(cli.particle_chunk_size),
        dtype="float64",
        tf32_mode="disabled",
    )


def _call_manual(module: Any, args: argparse.Namespace, theta: tf.Tensor, prepared: Mapping[str, Any]):
    if module.__name__.endswith("predator_prey_score"):
        return module._compact_value_and_score_from_components(  # noqa: SLF001
            args,
            theta,
            prepared_tensors=prepared["tensors"],
            prepared_transition_noise=prepared["transition_noise"],
        )
    return module._compact_value_and_score_from_components(  # noqa: SLF001
        args,
        theta,
        prepared_tensors=prepared["tensors"],
        prepared_initial_noise=prepared["initial_noise"],
        prepared_proposal_noise=prepared["proposal_noise"],
    )


def _call_manual_with_step_offset(
    module: Any,
    args: argparse.Namespace,
    theta: tf.Tensor,
    prepared: Mapping[str, Any],
    step_offset: int,
):
    if step_offset == 0:
        return _call_manual(module, args, theta, prepared)
    original_steps = core_tf._manual_dense_finite_steps  # noqa: SLF001

    def offset_steps(max_iterations: int | tf.Tensor) -> int:
        return max(0, original_steps(max_iterations) + step_offset)

    core_tf._manual_dense_finite_steps = offset_steps  # noqa: SLF001
    try:
        return _call_manual(module, args, theta, prepared)
    finally:
        core_tf._manual_dense_finite_steps = original_steps  # noqa: SLF001


def _call_value(module: Any, args: argparse.Namespace, theta: tf.Tensor, prepared: Mapping[str, Any]):
    if module.__name__.endswith("predator_prey_score"):
        return module._manual_value_only_from_components(  # noqa: SLF001
            args,
            theta,
            prepared_tensors=prepared["tensors"],
            prepared_transition_noise=prepared["transition_noise"],
        )
    return module._manual_value_only_from_components(  # noqa: SLF001
        args,
        theta,
        prepared_tensors=prepared["tensors"],
        prepared_initial_noise=prepared["initial_noise"],
        prepared_proposal_noise=prepared["proposal_noise"],
    )


def _full_transport_autodiff(
    module: Any,
    args: argparse.Namespace,
    theta: tf.Tensor,
    prepared: Mapping[str, Any],
) -> tf.Tensor:
    """Differentiate the unchanged forward scalar without stabilized transport stops."""

    if not module.__name__.endswith("generalized_sv_score"):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(theta)
            objective = _call_value(module, args, theta, prepared)["objective"]
        gradient = tape.gradient(objective, theta)
        if gradient is None:
            raise ValueError("full-transport reference autodiff returned no gradient")
        return gradient

    original_forward_transport = module._forward_transport_tf  # noqa: SLF001

    def full_forward_transport_tf(
        *,
        post_flow: tf.Tensor,
        normalized_log_weights: tf.Tensor,
        mask: tf.Tensor,
        args: argparse.Namespace,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        transported = core_tf.batched_annealed_transport_core_tf(
            post_flow,
            normalized_log_weights,
            mask,
            epsilon=args.sinkhorn_epsilon,
            scaling=args.annealed_scaling,
            convergence_threshold=args.annealed_convergence_threshold,
            max_iterations=args.sinkhorn_iterations,
            transport_gradient_mode="raw",
            transport_plan_mode="streaming",
            transport_ad_mode="full",
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
        return transported.particles, transported.log_weights

    module._forward_transport_tf = full_forward_transport_tf  # noqa: SLF001
    try:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(theta)
            objective = _call_value(module, args, theta, prepared)["objective"]
        gradient = tape.gradient(objective, theta)
    finally:
        module._forward_transport_tf = original_forward_transport  # noqa: SLF001
    if gradient is None:
        raise ValueError("full-transport reference autodiff returned no gradient")
    return gradient


def _float(value: tf.Tensor | float) -> float:
    output = float(tf.convert_to_tensor(value).numpy())
    if not math.isfinite(output):
        raise ValueError("diagnostic produced a nonfinite scalar")
    return output


def _comparison(reference: float, observed: float) -> dict[str, Any]:
    absolute_error, relative_scale, relative_error = coordinate_relative_error(reference, observed)
    tolerance = REFERENCE_ATOL + REFERENCE_RTOL * relative_scale
    return {
        "reference": reference,
        "observed": observed,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "mixed_tolerance": tolerance,
        "status": "pass" if absolute_error <= tolerance else "fail",
    }


def _objective_ulp(value: tf.Tensor) -> float:
    value = tf.convert_to_tensor(value)
    next_value = tf.math.nextafter(value, tf.constant(math.inf, dtype=value.dtype))
    return abs(_float(next_value - value))


def _fd_entry(
    module: Any,
    args: argparse.Namespace,
    prepared: Mapping[str, Any],
    theta: tf.Tensor,
    index: int,
    nominal_step: float,
    *,
    strategy: str,
    coefficient: float | None,
    base_objective_ulp: float,
    manual_score: float,
    autodiff_score: float,
    full_transport_autodiff_score: float,
) -> dict[str, Any]:
    basis = tf.one_hot(index, int(theta.shape[0]), dtype=theta.dtype)
    nominal = tf.constant(float(nominal_step), dtype=theta.dtype)
    plus_theta = theta + nominal * basis
    minus_theta = theta - nominal * basis
    effective_separation = plus_theta[index] - minus_theta[index]
    effective_step = effective_separation / tf.constant(2.0, dtype=theta.dtype)
    if _float(effective_separation) == 0.0:
        raise ValueError("FD endpoints collapse to the same parameter value")
    plus = _call_value(module, args, plus_theta, prepared)["objective"]
    minus = _call_value(module, args, minus_theta, prepared)["objective"]
    numerator = plus - minus
    fd = numerator / effective_separation
    numerator_value = _float(numerator)
    return {
        "strategy": strategy,
        "coefficient": coefficient,
        "nominal_step": float(nominal_step),
        "effective_step": _float(effective_step),
        "minus_parameter": _float(minus_theta[index]),
        "plus_parameter": _float(plus_theta[index]),
        "minus_objective": _float(minus),
        "plus_objective": _float(plus),
        "objective_numerator": numerator_value,
        "objective_numerator_ulps": (
            None if base_objective_ulp == 0.0 else numerator_value / base_objective_ulp
        ),
        "endpoint_objectives_equal": bool(_float(plus) == _float(minus)),
        "finite_difference": _float(fd),
        "versus_manual": _comparison(manual_score, _float(fd)),
        "versus_autodiff": _comparison(autodiff_score, _float(fd)),
        "versus_full_transport_autodiff": _comparison(
            full_transport_autodiff_score,
            _float(fd),
        ),
    }


def diagnose(cli: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    module = importlib.import_module(ROW_MODULES[cli.row])
    args = _args_for_row(cli)
    module._configure_precision(args)  # noqa: SLF001
    prepared = module._prepare_compact_xla_inputs(args)  # noqa: SLF001
    theta = tf.constant(module.TRUTH_THETA, dtype=module.DTYPE)
    manual_step_offset = int(getattr(cli, "manual_step_offset", 0))
    manual = _call_manual_with_step_offset(
        module,
        args,
        theta,
        prepared,
        manual_step_offset,
    )
    replay = _call_value(module, args, theta, prepared)
    manual_objective = _float(manual["objective"])
    replay_objective = _float(replay["objective"])
    if manual_objective != replay_objective:
        raise ValueError("manual score and value-only objectives differ at the base point")

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(theta)
        autodiff_objective = _call_value(module, args, theta, prepared)["objective"]
    autodiff = tape.gradient(autodiff_objective, theta)
    if autodiff is None:
        raise ValueError("reference autodiff returned no gradient")
    full_transport_autodiff = _full_transport_autodiff(module, args, theta, prepared)

    manual_values = [_float(value) for value in tf.reshape(manual["gradient_tensor"], [-1])]
    autodiff_values = [_float(value) for value in tf.reshape(autodiff, [-1])]
    full_transport_autodiff_values = [
        _float(value) for value in tf.reshape(full_transport_autodiff, [-1])
    ]
    parameter_names = [str(value) for value in module.PARAMETER_NAMES]
    if len(parameter_names) != len(manual_values) or len(manual_values) != len(autodiff_values):
        raise ValueError("parameter and score dimensions differ")

    base_ulp = _objective_ulp(replay["objective"])
    parameters = []
    for index, name in enumerate(parameter_names):
        theta_value = _float(theta[index])
        entries = [
            _fd_entry(
                module,
                args,
                prepared,
                theta,
                index,
                1.0e-4,
                strategy="legacy_absolute",
                coefficient=None,
                base_objective_ulp=base_ulp,
                manual_score=manual_values[index],
                autodiff_score=autodiff_values[index],
                full_transport_autodiff_score=full_transport_autodiff_values[index],
            )
        ]
        for coefficient in RELATIVE_STEP_COEFFICIENTS:
            entries.append(
                _fd_entry(
                    module,
                    args,
                    prepared,
                    theta,
                    index,
                    coefficient * max(1.0, abs(theta_value)),
                    strategy="relative_scale",
                    coefficient=coefficient,
                    base_objective_ulp=base_ulp,
                    manual_score=manual_values[index],
                    autodiff_score=autodiff_values[index],
                    full_transport_autodiff_score=full_transport_autodiff_values[index],
                )
            )
        parameters.append(
            {
                "parameter": name,
                "theta": theta_value,
                "manual_score": manual_values[index],
                "autodiff_score": autodiff_values[index],
                "full_transport_autodiff_score": full_transport_autodiff_values[index],
                "manual_vs_autodiff": _comparison(
                    autodiff_values[index],
                    manual_values[index],
                ),
                "manual_vs_full_transport_autodiff": _comparison(
                    full_transport_autodiff_values[index],
                    manual_values[index],
                ),
                "finite_difference_ladder": entries,
            }
        )

    manual_status = (
        "pass"
        if all(item["manual_vs_autodiff"]["status"] == "pass" for item in parameters)
        else "fail"
    )
    full_transport_manual_status = (
        "pass"
        if all(
            item["manual_vs_full_transport_autodiff"]["status"] == "pass"
            for item in parameters
        )
        else "fail"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": "completed",
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "question": (
            "Does the manual JVP equal the forward scalar's full transport derivative, "
            "how does stabilized autodiff differ, and where is FD numerically resolved?"
        ),
        "row": cli.row,
        "parameter_names": parameter_names,
        "theta": [_float(value) for value in theta],
        "shape": {
            "time_steps": int(args.time_steps),
            "num_particles": int(args.num_particles),
            "batch_seeds": list(args.batch_seeds),
        },
        "transport": {
            "policy": args.transport_policy,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "plan_mode": args.transport_plan_mode,
            "gradient_mode": args.transport_gradient_mode,
            "ad_mode": args.transport_ad_mode,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
            "manual_step_offset": manual_step_offset,
        },
        "precision": {
            "dtype": "float64",
            "tf32_mode": "disabled",
            "execution_target": "cpu_only_reference",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": False,
        },
        "prepared_input_fingerprint": _prepared_input_fingerprint(prepared),
        "objective": replay_objective,
        "objective_ulp": base_ulp,
        "objective_route_equality": {
            "manual_score_objective": manual_objective,
            "value_only_objective": replay_objective,
            "status": "pass",
        },
        "manual_jvp_vs_autodiff_status": manual_status,
        "manual_jvp_vs_full_transport_autodiff_status": full_transport_manual_status,
        "parameters": parameters,
        "nonclaims": [
            "reference-only autodiff is not the production score route",
            "tiny CPU/FP64 evidence is not production GPU/XLA evidence",
            "a single FD step is explanatory only",
            "no HMC, posterior, admission, default-readiness, or superiority claim",
        ],
        "run_manifest": {
            "git_commit": _git_output(("git", "rev-parse", "HEAD")),
            "git_status_short": _git_output(("git", "status", "--short")),
            "command": shlex.join(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
            "host": platform.node(),
            "platform": platform.platform(),
            "plan_path": PLAN_PATH,
            "plan_sha256": _sha256(ROOT / PLAN_PATH),
            "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": _sha256(Path(__file__)),
            "wall_time_seconds": time.perf_counter() - started,
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--row", choices=tuple(ROW_MODULES), required=True)
    parser.add_argument("--time-steps", type=int, required=True)
    parser.add_argument("--num-particles", type=int, required=True)
    parser.add_argument("--seed", type=int, default=81120)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--row-chunk-size", type=int, default=512)
    parser.add_argument("--col-chunk-size", type=int, default=512)
    parser.add_argument("--particle-chunk-size", type=int, default=512)
    parser.add_argument("--manual-step-offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args(argv)
    if cli.time_steps <= 0 or cli.num_particles <= 1:
        raise ValueError("time_steps must be positive and num_particles must exceed one")
    if min(cli.row_chunk_size, cli.col_chunk_size, cli.particle_chunk_size) <= 0:
        raise ValueError("chunk sizes must be positive")
    payload = diagnose(cli)
    _write_json_atomic(Path(cli.output), payload)
    print(
        json.dumps(
            {
                "row": payload["row"],
                "manual_jvp_vs_autodiff_status": payload["manual_jvp_vs_autodiff_status"],
                "objective": payload["objective"],
                "objective_ulp": payload["objective_ulp"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
