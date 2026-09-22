#!/usr/bin/env python3
"""Run the bounded q=20 tempered transport Phase 7 mechanics smoke.

The default route is a trusted GPU/XLA smoke.  ``--cpu-debug`` is an explicit
diagnostic exception used to localize implementation failures when no GPU is
available; its result is never GPU or scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md"
)

# A benchmark launched by pathname does not inherit the repository root on
# ``sys.path``; install it before importing any BayesFilter-owned module.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(command: tuple[str, ...]) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any, tf: Any) -> Any:
    """Materialize TensorFlow values only at the artifact boundary."""

    if tf.is_tensor(value):
        return _json_safe(value.numpy(), tf)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, tf) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, tf) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), tf)
    if hasattr(value, "item"):
        return _json_safe(value.item(), tf)
    return str(value)


def _device(value: Any) -> str:
    return str(getattr(value, "device", "unknown"))


def _static_scan(paths: tuple[Path, ...]) -> Mapping[str, Any]:
    forbidden = (
        "tf.map_fn",
        "tf.vectorized_map",
        "GradientTape.jacobian",
        "GradientTape.batch_jacobian",
        "pfor",
    )
    hits: dict[str, list[str]] = {token: [] for token in forbidden}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits[token].append(str(path.relative_to(ROOT)))
    return {
        "paths": [str(path.relative_to(ROOT)) for path in paths],
        "forbidden_tokens": list(forbidden),
        "hits": hits,
        "new_route_tokens_absent": not any(hits.values()),
    }


def _memory_info(tf: Any, logical_devices: tuple[Any, ...]) -> Mapping[str, Any]:
    rows: dict[str, Any] = {}
    for device in logical_devices:
        try:
            rows[str(device.name)] = dict(
                tf.config.experimental.get_memory_info(device.name)
            )
        except (AttributeError, RuntimeError, ValueError) as exc:
            rows[str(device.name)] = {"unavailable": type(exc).__name__}
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--cpu-debug",
        action="store_true",
        help="run an explicit CPU non-XLA diagnostic exception",
    )
    parser.add_argument("--q", type=int, default=20)
    parser.add_argument(
        "--principal-sqrt-backend",
        default="compiled_custom_op",
        choices=("compiled_custom_op", "tensorflow_eigh_strict"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = time.monotonic()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    gpu_requested = not bool(args.cpu_debug)
    if int(args.q) != 20:
        raise RuntimeError("Phase 7 smoke is frozen to q=20")
    if gpu_requested:
        if not _truthy(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")):
            raise RuntimeError(
                "GPU smoke requires TF_FORCE_GPU_ALLOW_GROWTH=true before import"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1":
            raise RuntimeError("GPU smoke cannot run with CUDA_VISIBLE_DEVICES=-1")
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != "-1":
        raise RuntimeError(
            "--cpu-debug requires CUDA_VISIBLE_DEVICES=-1 to make the exception explicit"
        )

    # Import TensorFlow only after launch-policy checks.  The memory helper is
    # called before querying logical devices or creating any TensorFlow value.
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=gpu_requested
    )
    # CPU debug is an explicit localization exception.  The default GPU route
    # remains XLA-enabled and uses the production target backend.
    runtime_jit = not bool(args.cpu_debug)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical_devices = tuple(tf.config.list_logical_devices("GPU"))
    if gpu_requested and not logical_devices:
        raise RuntimeError("GPU smoke requires at least one logical GPU")

    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        build_fixed_transport_value_score_adapter,
    )
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.tempered_transitions_tf import (
        BoundWithinTemperatureKernel,
        FixedChartKernelMixture,
        ProperBridgeReplicaExchange,
        ProperReplicaExchangeTransitionProgram,
        build_fixed_transport_hmc_kernel,
        screen_transport_reliability,
    )
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        AffineDiagonalTransport,
        IndependentTemperedReverseKLTrainer,
        TransportBank,
        prepare_transport_initialization,
        transport_preflight_state_hash,
    )

    route_paths = (
        ROOT / "bayesfilter/inference/tempered_target_tf.py",
        ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py",
        ROOT / "bayesfilter/inference/tempered_lineage_tf.py",
        ROOT / "bayesfilter/inference/tempered_transitions_tf.py",
        ROOT / "bayesfilter/inference/fixed_transport_hmc_mechanics_tf.py",
    )
    route_scan = _static_scan(route_paths)
    if not route_scan["new_route_tokens_absent"]:
        raise RuntimeError(f"forbidden row-mapping/pfor token in route: {route_scan}")

    bridge = make_q20_tempered_bridge(
        args.q,
        jit_compile=runtime_jit,
        principal_sqrt_backend=args.principal_sqrt_backend,
    )
    dimension = int(bridge.parameter_dim)
    batch_size = 4
    chain_count = 2
    component_ids = ("chart-0", "chart-1")
    betas = (0.0, 0.5, 1.0)
    seeds = {
        "preflight": ((20260829, 101), (20260829, 102)),
        "training": ((20260829, 201), (20260829, 202)),
        "latent_bank": (20260829, 301),
        "direct_transition": (20260829, 401),
        "replica_transition": (20260829, 501),
    }

    center = tf.convert_to_tensor(bridge.prior_center, tf.float64)
    endpoint_points = tf.stack(
        (center, center + tf.constant([0.1, -0.1, 0.1, -0.1], tf.float64)),
        axis=0,
    )
    endpoint_records: dict[str, Any] = {}
    for beta in betas:
        value, score, status = bridge.value_score_status(
            endpoint_points, tf.constant(beta, tf.float64)
        )
        endpoint_records[str(beta)] = {
            "value_finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
            "score_finite": bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
            "valid_rows": int(
                tf.reduce_sum(
                    tf.cast(status["bridge_valid"], tf.int32)
                ).numpy()
            ),
            "batch_size": int(value.shape[0]),
            "value_device": _device(value),
            "score_device": _device(score),
        }
        if endpoint_records[str(beta)]["valid_rows"] != 2:
            raise RuntimeError(f"bridge endpoint/interior status failed at beta={beta}")

    charts = []
    preflight_records = []
    training_records = []
    # One learned chart is enough to exercise the optimizer boundary.  An
    # exact affine second chart keeps the ensemble/swap portion bounded while
    # still testing cross-chart density evaluation and categorical selection.
    component_id = component_ids[0]
    config = WeightedNeuTraConfig(
        dimension=dimension,
        hidden_layers=(4,),
        stages=1,
        initialization_scale=0.01,
        initialization_seed=(20260829, 601),
        learning_rate=1.0e-3,
        jit_compile=runtime_jit,
    )
    raw = WeightedDenseIAFTransport(config)
    prepared = prepare_transport_initialization(
        raw,
        bridge,
        component_id=component_id,
        seed=seeds["preflight"][0],
        batch_size=batch_size,
        repair_scales=(1.0, 0.5, 0.25),
        beta=0.5,
        reference_center=center,
        reference_scale=4.0,
    )
    receipt = prepared.receipt
    if not receipt.valid:
        raise RuntimeError(f"preflight failed for {component_id}: {receipt.payload()}")
    preflight_records.append(receipt.payload())
    trainer = IndependentTemperedReverseKLTrainer(
        config,
        bridge,
        beta=0.5,
        component_id=component_id,
        batch_size=batch_size,
        prepared_initialization=prepared,
    )
    update = trainer.train_step(seeds["training"][0])
    if not bool(update.valid.numpy()) or int(update.step.numpy()) != 1:
        raise RuntimeError(f"optimizer update failed for {component_id}")
    state_hash = transport_preflight_state_hash(prepared.transport)
    prepared.transport.bind_frozen_identity(
        {
            "checkpoint_sha256": hashlib.sha256(
                f"phase7:{component_id}:checkpoint".encode("ascii")
            ).hexdigest(),
            "training_state_hash": state_hash,
            "transport_tensor_hash": state_hash,
        }
    )
    charts.append(prepared.transport)
    training_records.append(
        {
            "component_id": component_id,
            "step": int(update.step.numpy()),
            "valid": bool(update.valid.numpy()),
            "target_call_count": int(update.target_call_count.numpy()),
            "loss": float(update.loss.numpy()),
            "gradient_norm": float(update.gradient_norm.numpy()),
            "state_hash": state_hash,
            "manifest_hash": hashlib.sha256(
                json.dumps(
                    prepared.transport.manifest_payload(),
                    sort_keys=True,
                    ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest(),
        }
    )
    affine_chart = AffineDiagonalTransport(
        center + tf.constant([0.75, -0.5, 0.25, -0.25], tf.float64),
        tf.constant([1.5, 1.25, 1.1, 0.9], tf.float64),
        component_id=component_ids[1],
    )
    charts.append(affine_chart)
    preflight_records.append(
        {
            "component_id": component_ids[1],
            "valid": True,
            "optimizer_state_absent": True,
            "actual_map_repaired": False,
            "reason": "exact_affine_mechanics_chart",
            "transport_state_hash": transport_preflight_state_hash(affine_chart),
        }
    )

    latent = tf.random.stateless_normal(
        [len(charts), 1, dimension],
        tf.constant(seeds["latent_bank"], tf.int32),
        dtype=tf.float64,
    )
    bank = TransportBank(charts, component_ids=component_ids)
    physical_bank, logdet_bank = bank.forward_bank(latent)
    cross_density = bank.cross_component_log_prob(physical_bank)
    latent_bank_batch_size = 1
    if tuple(cross_density.shape) != (len(charts), len(charts), latent_bank_batch_size):
        raise RuntimeError(f"unexpected cross-density shape: {cross_density.shape}")

    mid_adapter = bridge.fixed_beta_adapter(0.5)
    reference_points = tf.stack((center,), axis=0)
    declared_points = tf.stack(
        (center + tf.constant([1.0, 0.0, 0.0, 0.0], tf.float64),), axis=0
    )
    reliability = screen_transport_reliability(
        charts,
        component_ids=component_ids,
        self_latent_bank=latent,
        cross_physical_bank=physical_bank,
        reference_points=reference_points,
        declared_points=declared_points,
        physical_score_fn=lambda values: mid_adapter.log_prob_and_grad(values)[1],
        maximum_condition_number=1.0e8,
        tolerance=1.0e-8,
    )
    if not reliability.passed:
        raise RuntimeError(f"learned-map reliability screen failed: {reliability.payload()}")

    # Exercise one real transformed HMC kernel at the interior temperature.
    # Replica exchange below uses exact identity kernels for the remaining
    # slots, isolating the bridge/swap graph from repeated target compilation.
    mid_adapter = bridge.fixed_beta_adapter(0.5)
    chart0_adapter = build_fixed_transport_value_score_adapter(
        base_adapter=mid_adapter,
        fixed_transport=charts[0],
        target_scope=f"{mid_adapter.target_scope}:chart={component_ids[0]}",
        evidence_path=str(PLAN_PATH.relative_to(ROOT)),
        xla_hmc_ready=runtime_jit,
        full_chain_xla_diagnostic_ready=False,
    )
    chart0_kernel = build_fixed_transport_hmc_kernel(
        chart0_adapter,
        state_shape=(chain_count, dimension),
        step_size=0.005,
        num_leapfrog_steps=1,
        jit_compile=runtime_jit,
    )

    direct_state = tf.broadcast_to(center, [chain_count, dimension])
    direct_state = direct_state + tf.constant(
        [[0.00, 0.00, 0.00, 0.00], [0.01, 0.00, 0.00, 0.00]],
        tf.float64,
    )
    direct_transition = chart0_kernel(
        direct_state, tf.constant(seeds["direct_transition"], tf.int32)
    )
    direct_transition_finite = bool(
        tf.reduce_all(tf.math.is_finite(direct_transition)).numpy()
    )
    if not direct_transition_finite:
        raise RuntimeError("direct fixed-chart transition returned nonfinite state")

    exchange = ProperBridgeReplicaExchange(bridge, betas)
    bindings = []
    def identity_kernel(state: Any, seed: Any) -> Any:
        del seed
        return tf.identity(state)

    for beta in betas:
        mixture = FixedChartKernelMixture(
            (identity_kernel, identity_kernel),
            gamma=(0.5, 0.5),
            chart_ids=component_ids,
        )
        bindings.append(
            BoundWithinTemperatureKernel(
                beta=beta,
                bridge_signature=bridge.signature,
                kernel_signature=mixture.selection.signature,
                kernel=mixture.transition,
                mechanics_role="phase7_q20_mechanics_smoke",
            )
        )
    program = ProperReplicaExchangeTransitionProgram(
        exchange, tuple(bindings), jit_compile=runtime_jit
    )
    initial_state = tf.broadcast_to(center, [len(betas), chain_count, dimension])
    initial_state = initial_state + tf.constant(
        [
            [[0.00, 0.00, 0.00, 0.00], [0.01, 0.00, 0.00, 0.00]],
            [[0.02, 0.00, 0.00, 0.00], [0.00, -0.02, 0.00, 0.00]],
            [[0.03, 0.00, 0.00, 0.00], [0.00, -0.03, 0.00, 0.00]],
        ],
        tf.float64,
    )
    transition_result = program(
        program.initial_state(initial_state),
        num_results=1,
        seed=tf.constant(seeds["replica_transition"], tf.int32),
        stage="retained",
    )
    health = transition_result["health"]
    if not bool(health["passed"]):
        raise RuntimeError(f"replica-exchange mechanics failed: {health}")
    posterior_samples = transition_result["posterior_samples"]
    if posterior_samples.shape != (1, chain_count, dimension):
        raise RuntimeError(f"unexpected beta-one sample shape: {posterior_samples.shape}")
    if not bool(tf.reduce_all(tf.math.is_finite(posterior_samples)).numpy()):
        raise RuntimeError("beta-one posterior boundary stream is nonfinite")

    source_paths = route_paths + (Path(__file__), PLAN_PATH)
    source_hashes = {
        str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
    }
    manifest = {
        "schema": "bayesfilter.ssl_lstm_q20_tempered_rkl_transport_ensemble_phase7_smoke.v1",
        "status": "PASS_CPU_DEBUG_ONLY" if args.cpu_debug else "PASS_PHASE7_GPU_MECHANICS_SMOKE",
        "phase": "7",
        "role": "non_claim_bearing_mechanics_smoke",
        "nonclaims": [
            "CPU debug does not establish GPU placement or GPU readiness",
            "smoke loss, whitening, swap rate, and movement are not research evidence",
            "no posterior convergence, mode-discovery, scaling, or default-readiness claim",
        ],
        "command": list(sys.argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": __import__("tensorflow_probability").__version__,
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "q": int(args.q),
        "parameter_dim": dimension,
        "internal_filter_state_dim": 60,
        "bridge_signature": bridge.signature,
        "target_signature": bridge.target_signature,
        "properness_receipt": _json_safe(bridge.properness_receipt.payload(), tf),
        "betas": list(betas),
        "positive_temperature_count": sum(beta > 0.0 for beta in betas),
        "component_ids": list(component_ids),
        "component_count": len(charts),
        "static_batch_size": batch_size,
        "reliability_bank_batch_size": latent_bank_batch_size,
        "chain_count": chain_count,
        "endpoint_records": endpoint_records,
        "preflight_records": preflight_records,
        "training_records": training_records,
        "cross_density_shape": list(cross_density.shape),
        "cross_density_work": len(charts) * len(charts) * batch_size,
        "target_work_for_joint_formula": len(charts) * batch_size,
        "reliability": _json_safe(reliability.payload(), tf),
        "direct_fixed_chart_transition": {
            "finite": direct_transition_finite,
            "device": _device(direct_transition),
            "step_size": 0.005,
            "num_leapfrog_steps": 1,
        },
        "replica_exchange_transition": {
            "transition_signature": program.transition_signature,
            "health": _json_safe(health, tf),
            "posterior_samples_shape": list(posterior_samples.shape),
            "posterior_temperature": transition_result["posterior_temperature"],
            "posterior_stream_only": transition_result["posterior_stream_only"],
            "posterior_sample_device": _device(posterior_samples),
        },
        "memory_policy": _json_safe(memory_policy, tf),
        "gpu_requested": gpu_requested,
        "logical_gpu_devices": [str(device.name) for device in logical_devices],
        "cpu_debug_cuda_hidden": bool(
            args.cpu_debug and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        ),
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "jit_compile": runtime_jit,
        "principal_sqrt_backend": args.principal_sqrt_backend,
        "route_scan": route_scan,
        "memory_info_after_run": _json_safe(_memory_info(tf, logical_devices), tf),
        "source_hashes": source_hashes,
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": _git(("git", "status", "--porcelain")),
        "seeds": _json_safe(seeds, tf),
        "wall_time_seconds": time.monotonic() - started,
        "output_dir": str(output_dir),
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(
        output_dir / "result.json",
        {
            "schema": "bayesfilter.ssl_lstm_q20_tempered_rkl_transport_ensemble_phase7_result.v1",
            "status": manifest["status"],
            "hard_vetoes": [],
            "mechanics_passed": True,
            "nonclaims": manifest["nonclaims"],
            "run_manifest": "run_manifest.json",
        },
    )
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # preserve a structured failure artifact
        # The output directory is known only after argument parsing; make a
        # best-effort failure record without masking the original exception.
        try:
            parsed = _parse_args()
            failure_dir = parsed.output_dir.expanduser().resolve()
            failure_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                failure_dir / "failure.json",
                {
                    "schema": "bayesfilter.ssl_lstm_q20_phase7_failure.v1",
                    "status": "FAIL_PHASE7_SMOKE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "command": list(sys.argv),
                    "cpu_debug": bool(parsed.cpu_debug),
                },
            )
        except Exception:
            pass
        print(f"PHASE7_SMOKE_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
