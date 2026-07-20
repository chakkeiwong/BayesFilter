"""Tiny HMC mechanics smoke for experimental streaming LEDH-PFPF-OT.

This is a Phase 5 hard-veto diagnostic. It records whether a bounded TFP HMC
run can evaluate finite values, gradients, log-accept ratios, and MH decisions
on a tiny fixed fixture. It is not a posterior convergence or HMC readiness
claim.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from bayesfilter.inference import FullChainHMCConfig, ValueScoreCapability
from bayesfilter.inference.hmc import run_full_chain_tfp_hmc
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from docs.benchmarks import (
    compare_experimental_batched_ledh_pfpf_ot_gradient_structure as gradient_structure,
)


NONCLAIMS = (
    "tiny HMC mechanics smoke only",
    "mixed precision means FP64 HMC state with target-internal DPF dtype",
    "no HMC readiness claim",
    "no posterior convergence claim",
    "no posterior validity claim",
    "no production/default/public API readiness claim",
    "no TF32 superiority claim",
    "no full FP32 HMC mechanics claim",
)


class _Namespace(argparse.Namespace):
    pass


class StreamingDPFAdapter:
    parameter_dim = 3

    def __init__(self, args: argparse.Namespace, tensors: dict[str, tf.Tensor]) -> None:
        self.args = args
        self.tensors = tensors

    def adapter_signature(self) -> str:
        shape = self.args
        return (
            "experimental_streaming_ledh_pfpf_ot_phase5_hmc_mechanics_"
            f"b{shape.batch_size}_t{shape.time_steps}_n{shape.num_particles}_"
            f"d{shape.state_dim}_m{shape.obs_dim}_{shape.transport_policy}_"
            f"{shape.dtype}_{shape.tf32_mode}"
        )

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow_streaming_ledh_pfpf_ot_phase5_mechanics",
            evidence_path=(
                "docs/plans/"
                "bayesfilter-dpf-tf32-batched-dpf-p5-hmc-facing-diagnostics-subplan-2026-06-17.md"
            ),
            target_scope="experimental_streaming_ledh_pfpf_ot_phase5_mechanics",
            nonclaims=NONCLAIMS,
            full_chain_xla_diagnostic_ready=False,
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        input_values = tf.convert_to_tensor(theta)
        values = tf.cast(input_values, core_tf.DTYPE)
        if values.shape.rank != 2:
            raise ValueError("phase5 DPF adapter requires theta rank 2 [batch, parameter]")
        with tf.GradientTape() as tape:
            tape.watch(values)
            log_likelihood = gradient_structure._value_for_arm(
                values,
                self.tensors,
                self.args,
                "streaming_streaming_tensor",
            )
            objective = tf.reduce_sum(log_likelihood)
        score = tape.gradient(
            objective,
            values,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        if score is None:
            score = tf.zeros_like(values)
        return log_likelihood, score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=2)
    parser.add_argument("--obs-dim", type=int, default=2)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-odd",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=3)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=16)
    parser.add_argument("--col-chunk-size", type=int, default=16)
    parser.add_argument("--particle-chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--hmc-seed", type=int, nargs=2, default=(20260617, 17))
    parser.add_argument("--num-results", type=int, default=6)
    parser.add_argument("--num-burnin-steps", type=int, default=2)
    parser.add_argument("--num-leapfrog-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.002)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument(
        "--tf32-mode",
        choices=("default", "enabled", "disabled"),
        default="disabled",
    )
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_results <= 0 or args.num_burnin_steps <= 0:
        raise ValueError("num-results and num-burnin-steps must be positive")
    if args.num_leapfrog_steps <= 0:
        raise ValueError("num-leapfrog-steps must be positive")
    if args.step_size <= 0.0:
        raise ValueError("step-size must be positive")
    return args


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    dtype = tf.float64 if args.dtype == "float64" else tf.float32
    core_tf.DTYPE = dtype
    streaming_tf.DTYPE = dtype
    annealed_transport_tf.DTYPE = dtype
    gradient_structure.DTYPE = dtype
    if args.tf32_mode != "default":
        tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    metadata = core_tf.precision_policy_metadata()
    metadata.update(
        {
            "dtype": args.dtype,
            "tf_dtype": dtype.name,
            "tf32_mode": args.tf32_mode,
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
        }
    )
    return metadata


def _make_gradient_args(args: argparse.Namespace) -> argparse.Namespace:
    return _Namespace(
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        num_particles=args.num_particles,
        state_dim=args.state_dim,
        obs_dim=args.obs_dim,
        transport_policy=args.transport_policy,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=args.annealed_scaling,
        annealed_convergence_threshold=args.annealed_convergence_threshold,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        particle_chunk_size=args.particle_chunk_size,
        seed=args.seed,
        no_jit_compile=False,
        dtype=args.dtype,
        tf32_mode=args.tf32_mode,
        include_dynamic_callback=False,
        structure_value_atol=1.0e-6,
        structure_value_rtol=1.0e-6,
        structure_score_atol=1.0e-5,
        structure_score_rtol=1.0e-5,
        device=args.device,
        device_scope=args.device_scope,
        cuda_visible_devices=args.cuda_visible_devices,
        expect_device_kind=args.expect_device_kind,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, tf.Tensor):
        return _json_safe(value.numpy())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _device_check(tensors: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in tensors]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU outputs, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU outputs, got {devices}")
    return devices


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    diagnostics = result["hmc_diagnostics"]
    lines = [
        "# LEDH-PFPF-OT HMC Mechanics Smoke",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Hard veto status: `{result['hard_veto_status']}`",
        f"- Acceptance rate: `{diagnostics.get('acceptance_rate')}`",
        f"- Nonfinite samples: `{diagnostics.get('nonfinite_sample_count')}`",
        f"- Nonfinite log accept ratios: `{result['nonfinite_log_accept_count']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    precision = _configure_precision(args)
    physical_gpus = [str(device) for device in tf.config.list_physical_devices("GPU")]
    logical_gpus = [str(device) for device in tf.config.list_logical_devices("GPU")]
    gradient_args = _make_gradient_args(args)
    tensors = gradient_structure._to_tensors(gradient_structure._fixture(gradient_args))
    adapter = StreamingDPFAdapter(gradient_args, tensors)
    # HMC state stays FP64; the adapter casts internally to the configured DPF
    # dtype and the reviewed value/score target casts back to the HMC dtype.
    initial_state = tf.cast(tensors["theta"], tf.float64)
    hmc_config = FullChainHMCConfig(
        num_results=args.num_results,
        num_burnin_steps=args.num_burnin_steps,
        step_size=args.step_size,
        num_leapfrog_steps=args.num_leapfrog_steps,
        seed=tuple(args.hmc_seed),
        use_xla=False,
        trace_policy="standard",
        target_scope="experimental_streaming_ledh_pfpf_ot_phase5_mechanics",
        chain_execution_mode="tf_function",
    )
    with tf.device(args.device):
        initial_value, initial_score = adapter.log_prob_and_grad(initial_state)
        result = run_full_chain_tfp_hmc(adapter, initial_state, hmc_config)
    samples = tf.convert_to_tensor(result.samples)
    log_accept = tf.convert_to_tensor(result.trace["log_accept_ratio"])
    accepted = tf.convert_to_tensor(result.trace["is_accepted"])
    target_log_prob = tf.convert_to_tensor(result.trace["target_log_prob"])
    _device_check(
        (initial_value, initial_score, samples, log_accept, target_log_prob),
        args.expect_device_kind,
    )
    finite_initial = bool(
        tf.reduce_all(tf.math.is_finite(initial_value)).numpy()
        and tf.reduce_all(tf.math.is_finite(initial_score)).numpy()
    )
    finite_samples = bool(tf.reduce_all(tf.math.is_finite(samples)).numpy())
    finite_log_accept = bool(tf.reduce_all(tf.math.is_finite(log_accept)).numpy())
    finite_target = bool(tf.reduce_all(tf.math.is_finite(target_log_prob)).numpy())
    nonfinite_log_accept_count = int(
        tf.size(log_accept).numpy()
        - tf.reduce_sum(tf.cast(tf.math.is_finite(log_accept), tf.int32)).numpy()
    )
    hard_veto_passed = (
        finite_initial
        and finite_samples
        and finite_log_accept
        and finite_target
        and "log_accept_ratio" in result.trace
        and "is_accepted" in result.trace
    )
    payload = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "precision": precision,
        "mixed_precision_contract": {
            "hmc_state_dtype": initial_state.dtype.name,
            "target_computation_dtype": core_tf.DTYPE.name,
            "target_return_dtype_seen_by_hmc": result.metadata["initial_state_dtype"],
            "tf32_execution_enabled": precision["tf32_execution_enabled"],
            "boundary": (
                "FP64 HMC state -> adapter casts to DPF dtype -> value/score "
                "cast back to HMC state dtype by reviewed target wrapper"
            ),
            "nonclaim": "not a full FP32 HMC mechanics diagnostic",
        },
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
            "parameter_dim": 3,
        },
        "transport": {
            "policy": args.transport_policy,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
        },
        "hmc_config": hmc_config.signature_payload(),
        "initial_value": _json_safe(initial_value),
        "initial_score": _json_safe(initial_score),
        "initial_finite": finite_initial,
        "hmc_diagnostics": _json_safe(result.diagnostics),
        "hmc_metadata": _json_safe(result.metadata),
        "trace_keys": sorted(result.trace.keys()),
        "acceptance_trace": _json_safe(accepted),
        "log_accept_ratio": _json_safe(log_accept),
        "target_log_prob_trace": _json_safe(target_log_prob),
        "finite_samples": finite_samples,
        "finite_log_accept_ratio": finite_log_accept,
        "finite_target_log_prob": finite_target,
        "nonfinite_log_accept_count": nonfinite_log_accept_count,
        "hard_veto_status": "passed" if hard_veto_passed else "failed",
        "overall_passed": bool(hard_veto_passed),
        "diagnostic_roles": {
            "finite_initial_value_score": "hard_veto",
            "finite_samples": "hard_veto",
            "finite_log_accept_ratio": "hard_veto",
            "finite_target_log_prob": "hard_veto",
            "acceptance_rate": "explanatory_only_for_short_chain",
            "sample_chain_timing": "explanatory_only",
        },
        "nonclaims": list(NONCLAIMS),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown, payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
