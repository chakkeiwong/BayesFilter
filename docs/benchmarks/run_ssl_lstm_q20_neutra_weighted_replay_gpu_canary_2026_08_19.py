#!/usr/bin/env python3
"""GPU/XLA weighted NeuTra replay canary for SSL-LSTM q=20.

The replay rows come from the terminal populations of the previously completed
two-region annealed-SMC runs.  They are optimization evidence only.  This
canary deliberately runs a short exact transformed-target HMC screen after the
update so that mode-locked conditional chains cannot be mistaken for a result.
"""

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
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-plan-2026-08-19.md"
)
SMC_ROOT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2"
)
RECOVERY = SMC_ROOT / "receipt-recovery-v1.json"
RECOVERY_SCHEMA = "bayesfilter.ssl_lstm.q20_physical_annealed_smc.receipt_recovery.v1"
RECOVERY_STATUS = "SMC_RECEIPT_RECOVERY_PASSED"
RECOVERY_SHA256 = "3aea988e7b27381a6b62e7a2d452db8251b9bd7d8b9f5e68ad08fcbe711b6d97"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
GEOMETRY = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
)
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-global-mixing-2026-08-19/gpu-canary-r1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite canary artifact: {path}")
    encoded = (
        json.dumps(_json_ready(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(path)


def _reserve_output_root(path: Path) -> Path:
    absolute = path if path.is_absolute() else ROOT / path
    if absolute.exists():
        raise RuntimeError(f"refusing to reuse canary output root: {absolute}")
    absolute.mkdir(parents=True, exist_ok=False)
    return absolute


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "dirty": bool(
            subprocess.run(
                ("git", "status", "--short"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
    }


def _verified_replay_sources(
    bank_indices: tuple[int, ...],
    *,
    recovery_path: Path = RECOVERY,
    allow_reserved_audit: bool = False,
) -> tuple[tuple[tuple[Path, Path, Mapping[str, Any]], ...], Mapping[str, Any]]:
    if not bank_indices or len(set(bank_indices)) != len(bank_indices):
        raise RuntimeError("replay bank indices must be nonempty and unique")
    if allow_reserved_audit:
        if bank_indices != (7,):
            raise RuntimeError("the audit loader may load only central-07")
    elif any(index < 0 or index > 6 for index in bank_indices):
        raise RuntimeError("the canary may load only central-00 through central-06")
    if _sha256(recovery_path) != RECOVERY_SHA256:
        raise RuntimeError("SMC recovery receipt SHA-256 mismatch")
    recovery = json.loads(recovery_path.read_text(encoding="ascii"))
    if recovery.get("schema") != RECOVERY_SCHEMA:
        raise RuntimeError("SMC recovery receipt schema mismatch")
    if recovery.get("status") != RECOVERY_STATUS:
        raise RuntimeError("SMC recovery receipt did not pass")
    gates = recovery.get("gates")
    if not isinstance(gates, Mapping) or not gates or any(
        value is not True for value in gates.values()
    ):
        raise RuntimeError("SMC recovery receipt gates did not all pass")
    children_raw = recovery.get("children")
    inventory_raw = recovery.get("tensor_inventory")
    if not isinstance(children_raw, list) or not isinstance(inventory_raw, list):
        raise RuntimeError("SMC recovery child or tensor inventory is missing")
    children: dict[str, Mapping[str, Any]] = {}
    for child in children_raw:
        if not isinstance(child, Mapping):
            raise RuntimeError("SMC recovery child record is malformed")
        name = str(child.get("name", ""))
        if not name or name in children:
            raise RuntimeError("SMC recovery child names must be unique")
        children[name] = child
    inventory: dict[str, Mapping[str, Any]] = {}
    for record in inventory_raw:
        if not isinstance(record, Mapping):
            raise RuntimeError("SMC recovery tensor record is malformed")
        path = str(record.get("path", ""))
        if not path or path in inventory:
            raise RuntimeError("SMC recovery tensor paths must be unique")
        inventory[path] = record

    verified = []
    for index in bank_indices:
        name = f"central-{index:02d}"
        child = children.get(name)
        if child is None:
            raise RuntimeError(f"SMC recovery child is missing: {name}")
        stage_count = int(child.get("stage_count", 0))
        if stage_count < 1:
            raise RuntimeError(f"SMC recovery stage count is invalid: {name}")
        stage = stage_count - 1
        child_root = SMC_ROOT / name
        theta_path = child_root / f"stage-{stage:02d}-pre-theta.tftensor"
        weight_path = (
            child_root / f"stage-{stage:02d}-pre-normalized_weights.tftensor"
        )
        selected_records = []
        for path, dtype, shape in (
            (theta_path, "float64", [100, 4]),
            (weight_path, "float64", [100]),
        ):
            relative = path.relative_to(ROOT).as_posix()
            record = inventory.get(relative)
            if record is None:
                raise RuntimeError(f"SMC recovery tensor is unlisted: {relative}")
            if record.get("dtype") != dtype or record.get("shape") != shape:
                raise RuntimeError(f"SMC recovery tensor contract mismatch: {relative}")
            if not path.is_file():
                raise RuntimeError(f"SMC replay tensor is missing: {relative}")
            if int(record.get("bytes", -1)) != path.stat().st_size:
                raise RuntimeError(f"SMC replay tensor byte count mismatch: {relative}")
            if _sha256(path) != record.get("sha256"):
                raise RuntimeError(f"SMC replay tensor SHA-256 mismatch: {relative}")
            selected_records.append(dict(record))
        verified.append(
            (
                theta_path,
                weight_path,
                {
                    "child": name,
                    "terminal_stage": stage,
                    "theta": selected_records[0],
                    "normalized_weights": selected_records[1],
                },
            )
        )

    reserved = children.get("central-07")
    if reserved is None or int(reserved.get("stage_count", 0)) < 1:
        raise RuntimeError("reserved audit child is absent from the recovery receipt")
    metadata = {
        "recovery_receipt": {
            "path": recovery_path.relative_to(ROOT).as_posix(),
            "sha256": RECOVERY_SHA256,
            "schema": RECOVERY_SCHEMA,
            "status": RECOVERY_STATUS,
            "gates": dict(gates),
        },
        "loaded_bank_indices": list(bank_indices),
        "loaded_bank_count": len(bank_indices),
        "reserved_audit_bank": {
            "child": "central-07",
            "stage_count": int(reserved["stage_count"]),
            "tensor_loaded": bool(allow_reserved_audit),
            "target_evaluated": False,
            "used_for_training_selection_or_nomination": False,
        },
    }
    return tuple(verified), metadata


def _load_replay(tf: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    """Load seven receipt-bound banks; central-07 remains untouched."""

    verified, receipt_metadata = _verified_replay_sources(tuple(range(7)))
    rows = []
    log_weights = []
    sources = []
    for theta_path, weight_path, source in verified:
        theta = tf.io.parse_tensor(tf.io.read_file(str(theta_path)), tf.float64)
        normalized = tf.io.parse_tensor(
            tf.io.read_file(str(weight_path)), tf.float64
        )
        theta = tf.ensure_shape(theta, (100, 4))
        normalized = tf.ensure_shape(normalized, (100,))
        tf.debugging.assert_all_finite(theta, f"replay theta {source['child']}")
        tf.debugging.assert_all_finite(
            normalized, f"replay weights {source['child']}"
        )
        tf.debugging.assert_positive(
            normalized, f"replay weights must be strictly positive: {source['child']}"
        )
        tf.debugging.assert_near(
            tf.reduce_sum(normalized), tf.constant(1.0, tf.float64), atol=1.0e-10
        )
        rows.append(theta)
        # Each loaded SMC population contributes one seventh of this canary
        # measure.  Adding log(1/7) is constant within the concatenated bank,
        # but records the intended equal-bank aggregation explicitly.
        log_weights.append(
            tf.math.log(normalized) - tf.math.log(tf.constant(7.0, tf.float64))
        )
        sources.append(source)
    return (
        tf.ensure_shape(tf.concat(rows, axis=0), (700, 4)),
        tf.ensure_shape(tf.concat(log_weights, axis=0), (700,)),
        {**receipt_metadata, "sources": sources},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="1")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.updates < 1 or args.hidden_width < 1 or args.stages < 1:
        raise SystemExit("updates, hidden-width, and stages must be positive")
    if not str(args.device).isdigit():
        raise SystemExit("device must be one nonnegative physical GPU index")
    output = _reserve_output_root(args.output_root)

    # Set the allocator and visible device before any TensorFlow import.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    import tensorflow as tf

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible logical GPU, found {logical}")

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        FixedTransportFullChainConfig,
        FixedTransportHMCPolicy,
        run_fixed_transport_full_chain_tfp_hmc,
    )
    from bayesfilter.inference.neutra_end_to_end import BatchNativeBoundAdapter
    from bayesfilter.inference.neutra_global_mixing import assess_retained_mode_mixing
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedForwardKLNeuTraTrainer,
        WeightedNeuTraConfig,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )
    started = time.perf_counter()
    physical, replay_log_weights, replay_meta = _load_replay(tf)
    # Six banks train and one selects. central-07 is not loaded by this canary.
    train_rows = physical[:600]
    train_weights = replay_log_weights[:600]
    selection_rows = physical[600:700]
    selection_weights = replay_log_weights[600:700]

    config = WeightedNeuTraConfig(
        dimension=4,
        hidden_layers=(int(args.hidden_width), int(args.hidden_width)),
        stages=int(args.stages),
        activation="tanh",
        initialization_scale=0.02,
        initialization_seed=(20260819, 1000 + int(args.seed)),
        learning_rate=1.0e-3,
        gradient_clip_norm=10.0,
        jit_compile=True,
    )
    trainer = WeightedForwardKLNeuTraTrainer(config)
    checkpoints = []
    for update in range(1, int(args.updates) + 1):
        step = trainer.train_step(train_rows, train_weights)
        if update in {1, int(args.updates)} or update % max(1, int(args.updates) // 5) == 0:
            selection = trainer.validation_batch(selection_rows, selection_weights)
            checkpoints.append(
                {
                    "update": update,
                    "loss": step.loss,
                    "gradient_norm": step.gradient_norm,
                    "selection_loss": selection.loss,
                    "selection_ess_fraction": selection.effective_sample_size_fraction,
                }
            )
    state = trainer.state_payload()
    state_bytes = json.dumps(_json_ready(state), sort_keys=True).encode("utf-8")
    state_hash = hashlib.sha256(state_bytes).hexdigest()
    trainer.transport.bind_frozen_identity(
        {
            "checkpoint_sha256": state_hash,
            "training_state_hash": state.get("state_hash", state_hash),
            "transport_tensor_hash": state_hash,
        }
    )

    # Construct the exact pullback target.  The initial states are diagnostic
    # only: they include both mapped representatives, but no equal pooling is
    # performed if the one common kernel fails to cross.
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    if target.target_signature() != TARGET_SIGNATURE:
        raise RuntimeError("SSL-LSTM q=20 target signature drift")
    if target.adapter_signature() != ADAPTER_SIGNATURE:
        raise RuntimeError("SSL-LSTM q=20 adapter signature drift")
    base = BatchNativeBoundAdapter(target, target_signature=target.target_signature())
    transformed = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=trainer.transport,
        target_scope="ssl_lstm_q20_neutra_global_mixing_canary",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False,
        require_batch_native=True,
        nonclaims=(
            "canary transport only",
            "replay rows are optimization evidence, not posterior samples",
            "no conditional-chain pooling",
            "no posterior or predictive claim",
        ),
    )
    if _sha256(GEOMETRY) != GEOMETRY_SHA256:
        raise RuntimeError("SSL-LSTM q=20 geometry receipt drift")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    representatives = tf.constant(
        [
            geometry["representatives"][label]["position"]
            for label in ("plus", "minus")
        ],
        tf.float64,
    )
    latent_representatives, _ = trainer.transport.inverse_and_forward_logdet(
        representatives
    )
    recovered_representatives = trainer.transport.forward_batch(latent_representatives)
    inverse_forward_logdet = trainer.transport.inverse_and_forward_logdet(
        representatives
    )[1]
    direct_forward_logdet = trainer.transport.log_abs_det_jacobian_batch(
        latent_representatives
    )
    transformed_value, transformed_score, transformed_status = (
        transformed.log_prob_and_grad_status(latent_representatives)
    )
    physical_value, physical_score, physical_status = (
        target.neutra_batch_log_prob_and_grad_status(representatives)
    )
    expected_score = trainer.transport.pullback_score_batch(
        latent_representatives, tf.stop_gradient(physical_score)
    ) + trainer.transport.log_abs_det_jacobian_score_batch(latent_representatives)
    inverse_forward_residual = tf.reduce_max(
        tf.abs(recovered_representatives - representatives)
    )
    logdet_residual = tf.reduce_max(
        tf.abs(inverse_forward_logdet - direct_forward_logdet)
    )
    value_residual = tf.reduce_max(
        tf.abs(transformed_value - physical_value - direct_forward_logdet)
    )
    score_residual = tf.reduce_max(tf.abs(transformed_score - expected_score))
    parity_tolerance = tf.constant(1.0e-10, tf.float64)
    tf.debugging.assert_near(
        recovered_representatives,
        representatives,
        atol=parity_tolerance,
        rtol=parity_tolerance,
        message="frozen transport inverse/forward parity",
    )
    tf.debugging.assert_near(
        inverse_forward_logdet,
        direct_forward_logdet,
        atol=parity_tolerance,
        rtol=parity_tolerance,
        message="frozen transport log-Jacobian parity",
    )
    tf.debugging.assert_near(
        transformed_value,
        physical_value + direct_forward_logdet,
        atol=parity_tolerance,
        rtol=parity_tolerance,
        message="transformed-target value identity",
    )
    tf.debugging.assert_near(
        transformed_score,
        expected_score,
        atol=parity_tolerance,
        rtol=parity_tolerance,
        message="transformed-target explicit score composition",
    )
    tf.debugging.assert_all_finite(transformed_score, "transformed score")
    status_valid = tf.reduce_all(
        (tf.cast(physical_status["status_code"], tf.int32) == 0)
        & tf.cast(physical_status["valid_pre_regularized_score"], tf.bool)
        & (tf.cast(transformed_status["status_code"], tf.int32) == 0)
        & tf.cast(transformed_status["valid_pre_regularized_score"], tf.bool)
    )
    tf.debugging.assert_equal(status_valid, True, message="target status parity")
    initial = tf.concat(
        (
            latent_representatives,
            latent_representatives + tf.constant(
                ((0.05, 0.0, 0.0, 0.0), (-0.05, 0.0, 0.0, 0.0)), tf.float64
            ),
        ),
        axis=0,
    )
    hmc = run_fixed_transport_full_chain_tfp_hmc(
        transformed,
        initial,
        FixedTransportFullChainConfig(
            num_results=64,
            num_burnin_steps=64,
            step_size=0.10,
            num_leapfrog_steps=5,
            seed=(20260819, 2000 + int(args.seed)),
            use_xla=True,
            trace_policy="full",
            target_status_trace_policy="per_chain_step",
            tuning_policy=FixedTransportHMCPolicy.fixed(source=PLAN.as_posix()),
            target_scope="ssl_lstm_q20_neutra_global_mixing_canary_hmc",
            chain_execution_mode="tf_function",
        ),
    )
    samples_draw_chain = tf.ensure_shape(hmc.samples, (64, 4, 4))
    tf.debugging.assert_all_finite(samples_draw_chain, "canary HMC samples")
    physical_samples = trainer.transport.forward_batch(
        tf.reshape(samples_draw_chain, (-1, 4))
    )
    # sample_chain stores [draw, chain, parameter]. Preserve that ordering
    # through the flattened transport call, then transpose to the diagnostic's
    # required [chain, retained_draw] contract. A direct reshape to [chain,
    # draw] would silently interleave chains.
    labels_draw_chain = tf.reshape(physical_samples[:, 2] < 0.0, (64, 4))
    labels = tf.cast(tf.transpose(labels_draw_chain, (1, 0)), tf.int32)
    mixing = assess_retained_mode_mixing(labels, region_count=2)
    result = {
        "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_gpu_canary.v1",
        "status": "GPU_REPLAY_TRAINING_AND_PULLBACK_HMC_COMPLETED",
        "plan": PLAN.as_posix(),
        "plan_sha256": _sha256(PLAN),
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
        "geometry": {
            "path": GEOMETRY.as_posix(),
            "sha256": GEOMETRY_SHA256,
        },
        "transformed_adapter_signature": transformed.adapter_signature(),
        "memory_policy": memory_policy,
        "visible_logical_gpus": [str(device) for device in logical],
        "requested_physical_device_selector": str(args.device),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "managed_session_trust_basis": (
            "owner_designated_managed_session_visible_gpu_trusted"
        ),
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "jit_compile": True,
        "dtype": "float64",
        "replay": replay_meta,
        "replay_rows": int(physical.shape[0]),
        "train_rows": int(train_rows.shape[0]),
        "selection_rows": int(selection_rows.shape[0]),
        "audit_rows_loaded_or_evaluated": 0,
        "training_batch_size": int(train_rows.shape[0]),
        "batch_native_target_backend": (
            "ssl_lstm_q20_batch_native_tensorflow_xla"
        ),
        "sample_wise_loop_used": False,
        "scalar_target_fallback_used": False,
        "training_config": config.manifest_payload(),
        "training_checkpoints": checkpoints,
        "exact_pullback_parity": {
            "inverse_forward_maximum_absolute_residual": inverse_forward_residual,
            "logdet_maximum_absolute_residual": logdet_residual,
            "value_identity_maximum_absolute_residual": value_residual,
            "score_composition_maximum_absolute_residual": score_residual,
            "tolerance": parity_tolerance,
            "target_status_valid": status_valid,
            "passed": (
                (inverse_forward_residual <= parity_tolerance)
                & (logdet_residual <= parity_tolerance)
                & (value_residual <= parity_tolerance)
                & (score_residual <= parity_tolerance)
                & status_valid
            ),
        },
        "hmc_diagnostics": hmc.diagnostics,
        "hmc_metadata": hmc.metadata,
        "mode_labels": labels,
        "global_mixing_report": mixing.payload(),
        "wall_seconds": time.perf_counter() - started,
        "git": _git_manifest(),
        "nonclaims": [
            "replay rows are not a posterior archive",
            "SMC weights are not imposed as HMC mode weights",
            "short HMC canary is not convergence evidence",
            "no pooling of mode-locked conditional chains",
            "no posterior-predictive or default-readiness claim",
        ],
    }
    _write(output / "result.json", result)
    _write(
        output / "trainer_state.json",
        {"schema": "bayesfilter.ssl_lstm.q20_neutra_canary_state.v1", "state": state},
    )
    _write(
        output / "manifest.json",
        {
            "schema": "bayesfilter.ssl_lstm.q20_neutra_global_mixing_gpu_manifest.v1",
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "result": (output / "result.json").as_posix(),
            "gpu_memory_growth_required": True,
            "requested_physical_device_selector": str(args.device),
            "visible_logical_gpu_count": len(logical),
            "managed_session_trust_basis": (
                "owner_designated_managed_session_visible_gpu_trusted"
            ),
            "training_batch_size": int(train_rows.shape[0]),
            "sample_wise_loop_used": False,
            "scalar_target_fallback_used": False,
            "jit_compile": True,
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "global_mixing_passed": result["global_mixing_report"]["passed"],
                "output": output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
