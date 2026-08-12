"""Load the clean q=20 seed-B terminal NeuTra transport for HMC campaigns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-budgeted-continuation-2026-08-06/"
    "r1/seed-b/checkpoint-4000.json"
)
TRAINING_RESULT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-budgeted-continuation-2026-08-06/"
    "r1/seed-b/result.json"
)
EXPECTED_CHECKPOINT_SCHEMA = (
    "bayesfilter.ssl_lstm.q20_neutra_budgeted_continuation.v1.checkpoint"
)
EXPECTED_RESULT_STATUS = "GPU_CONTINUATION_COMPLETED_CANDIDATE_NOMINATED"
EXPECTED_OPTIMIZER_STEP = 6250
EXPECTED_CONTINUATION_UPDATE = 4000


class SeedBTerminalError(RuntimeError):
    """Raised when the clean seed-B training evidence does not bind exactly."""


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload).rstrip(b"\n")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise SeedBTerminalError(f"expected JSON object: {path}")
    return payload


def _verified_checkpoint_and_state() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    checkpoint = read_json(CHECKPOINT)
    if checkpoint.get("schema") != EXPECTED_CHECKPOINT_SCHEMA:
        raise SeedBTerminalError("seed-B terminal checkpoint schema mismatch")
    raw = dict(checkpoint)
    supplied = str(raw.pop("checkpoint_hash", ""))
    if supplied != hashlib.sha256(canonical_bytes(raw)).hexdigest():
        raise SeedBTerminalError("seed-B terminal checkpoint payload hash mismatch")
    if checkpoint.get("stream") != "seed-b":
        raise SeedBTerminalError("terminal checkpoint is not seed B")
    if int(checkpoint.get("continuation_update", -1)) != EXPECTED_CONTINUATION_UPDATE:
        raise SeedBTerminalError("terminal checkpoint continuation update mismatch")
    if int(checkpoint.get("optimizer_step", -1)) != EXPECTED_OPTIMIZER_STEP:
        raise SeedBTerminalError("terminal checkpoint optimizer step mismatch")
    state = checkpoint.get("trainer_state")
    if not isinstance(state, Mapping) or int(state.get("step", -1)) != EXPECTED_OPTIMIZER_STEP:
        raise SeedBTerminalError("terminal checkpoint trainer state is missing or stale")
    state_raw = dict(state)
    state_hash = str(state_raw.pop("state_hash", ""))
    if state_hash != stable_hash(state_raw):
        raise SeedBTerminalError("terminal trainer state hash mismatch")
    return checkpoint, state


def _verify_training_selection() -> Mapping[str, Any]:
    result = read_json(TRAINING_RESULT)
    if result.get("status") != EXPECTED_RESULT_STATUS or result.get("vetoes") != []:
        raise SeedBTerminalError("seed-B training result is not a clean nomination")
    target_validity_failures = result.get("target_validity_failures")
    if target_validity_failures not in (None, []):
        raise SeedBTerminalError("seed-B training result contains target-validity failures")
    selection = result.get("selection")
    if not isinstance(selection, Mapping):
        raise SeedBTerminalError("seed-B training result lacks checkpoint selection")
    if int(selection.get("selected_continuation_update", -1)) != EXPECTED_CONTINUATION_UPDATE:
        raise SeedBTerminalError("seed-B selected checkpoint is not terminal update 4000")
    if int(selection.get("selected_optimizer_step", -1)) != EXPECTED_OPTIMIZER_STEP:
        raise SeedBTerminalError("seed-B selected optimizer step mismatch")
    selected_rows = [
        row
        for row in selection.get("rows", ())
        if int(row.get("continuation_update", -1)) == EXPECTED_CONTINUATION_UPDATE
    ]
    if len(selected_rows) != 1:
        raise SeedBTerminalError("seed-B selection does not uniquely bind checkpoint 4000")
    receipt = selected_rows[0].get("checkpoint")
    if not isinstance(receipt, Mapping):
        raise SeedBTerminalError("seed-B selection checkpoint receipt is missing")
    if (ROOT / str(receipt.get("path"))).resolve() != CHECKPOINT.resolve():
        raise SeedBTerminalError("seed-B selected checkpoint path mismatch")
    if str(receipt.get("sha256")) != sha256(CHECKPOINT):
        raise SeedBTerminalError("seed-B selected checkpoint SHA-256 mismatch")
    support = result.get("support_probe")
    if not isinstance(support, Mapping) or support.get("all_finite") is not True:
        raise SeedBTerminalError("seed-B selected transport failed its support probe")
    return result


def _migrated_state(
    state: Mapping[str, Any], current_config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    archived_config = state.get("config")
    if not isinstance(archived_config, Mapping):
        raise SeedBTerminalError("terminal trainer state lacks its config")
    if archived_config == current_config:
        return state, None
    expected = dict(current_config)
    empty_fields = ("fixed_output_scale", "fixed_output_factor")
    if any(expected.pop(field, None) != [] for field in empty_fields) or dict(archived_config) != expected:
        # Older checkpoints also predate the optional null chart-signature key.
        if expected.get("chart_signature") is None:
            expected.pop("chart_signature", None)
        if dict(archived_config) != expected:
            raise SeedBTerminalError("terminal trainer config mismatch is not empty-scale drift")
    raw = dict(state)
    source_hash = str(raw.pop("state_hash"))
    migrated_config = dict(archived_config)
    migrated_config["fixed_output_scale"] = []
    migrated_config["fixed_output_factor"] = []
    migrated_config["chart_signature"] = None
    migrated = {**raw, "config": migrated_config}
    migrated_hash = stable_hash(migrated)
    return {**migrated, "state_hash": migrated_hash}, {
        "schema": "bayesfilter.neutra.trainer_state_compatibility_migration.v1",
        "source_state_hash": source_hash,
        "migrated_state_hash": migrated_hash,
        "added_fields": [
            "config.fixed_output_scale",
            "config.fixed_output_factor",
            "config.chart_signature",
        ],
        "added_value": "empty_or_null",
        "numerical_transform_changed": False,
        "historical_checkpoint_modified": False,
    }


def configure_cpu_tensorflow(threads: int) -> Any:
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(int(threads))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise SeedBTerminalError("CPU/XLA HMC worker found a visible GPU")
    return tf


def build_seed_b_terminal(
    *, threads: int, evidence_path: str, target_scope_suffix: str
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Return the exact seed-B target bridge, frozen transport, and provenance."""

    tf = configure_cpu_tensorflow(threads)
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_training import (
        NeuTraReverseKLTrainer,
        ssl_lstm_tuned_capacity_neutra_config,
    )
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import FREE_NAMES

    checkpoint, state = _verified_checkpoint_and_state()
    training_result = _verify_training_selection()
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise SeedBTerminalError("terminal state config is missing")
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    trainer_config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=int(config["dimension"]),
        fixed_translation=tuple(float(value) for value in config["fixed_translation"]),
        fixed_output_scale=tuple(float(value) for value in config.get("fixed_output_scale", ())),
        target_parameter_names=tuple(str(value) for value in config["target_parameter_names"]),
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=float(config["learning_rate"]),
        initialization_scale=float(config["initialization_scale"]),
        gradient_clip_norm=float(config["gradient_clip_norm"]),
        target_chart=str(config["target_chart"]),
        initialization_seed=tuple(int(value) for value in config["initialization_seed"]),
        jit_compile=True,
    )
    trainer = NeuTraReverseKLTrainer(target, trainer_config)
    restored_state, migration = _migrated_state(state, trainer.config.manifest_payload())
    trainer.restore_state(restored_state)
    if int(trainer.step.numpy()) != EXPECTED_OPTIMIZER_STEP:
        raise SeedBTerminalError("restored seed-B optimizer step mismatch")
    frozen = trainer.frozen_transport_payload(
        transport_id="ssl-lstm-q20-seed-b-terminal-step-6250",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )

    class Bridge:
        parameter_dim = 4
        parameter_names = tuple(FREE_NAMES)
        supports_retained_draw_batch = False
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True
        target_status_invalid_rows_become_nonfinite = True

        def __init__(self) -> None:
            self.target_scope = f"{target.target_scope}:{target_scope_suffix}"

        def adapter_signature(self) -> str:
            return target.adapter_signature()

        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,
                runtime_backend="ssl_lstm_q20_seed_b_terminal_cpu_xla_bridge",
                evidence_path=str(evidence_path),
                target_scope=self.target_scope,
                nonclaims=(
                    "CPU/XLA HMC validation exception to GPU training default",
                    "no posterior oracle",
                ),
            )

        def log_prob_and_grad(self, values: Any) -> tuple[Any, Any]:
            tensor = tf.convert_to_tensor(values, tf.float64)
            if tensor.shape.rank != 2 or tensor.shape[-1] != 4:
                raise ValueError("q=20 HMC target requires static shape [batch,4]")
            return target.batch_value_and_score(tensor)

        def log_prob_and_grad_status(
            self, values: Any
        ) -> tuple[Any, Any, Mapping[str, Any]]:
            return target.neutra_batch_log_prob_and_grad_status(values)

        def target_status_telemetry(self, values: Any) -> Mapping[str, Any]:
            return self.log_prob_and_grad_status(values)[2]

    provenance = {
        "chart": "seed-b-terminal",
        "checkpoint_path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "continuation_update": EXPECTED_CONTINUATION_UPDATE,
        "optimizer_step": EXPECTED_OPTIMIZER_STEP,
        "trainer_state_hash": state["state_hash"],
        "restored_state_hash": restored_state["state_hash"],
        "compatibility_migration": migration,
        "training_result_path": TRAINING_RESULT.relative_to(ROOT).as_posix(),
        "training_result_sha256": sha256(TRAINING_RESULT),
        "training_result_status": training_result["status"],
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "transport_hash": loaded.manifest.transport_hash,
        "transport_artifact_signature": loaded.artifact_signature,
    }
    return Bridge(), loaded.transport, provenance


def binding_payload() -> Mapping[str, Any]:
    checkpoint, state = _verified_checkpoint_and_state()
    result = _verify_training_selection()
    return {
        "checkpoint_path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "trainer_state_hash": state["state_hash"],
        "optimizer_step": EXPECTED_OPTIMIZER_STEP,
        "continuation_update": EXPECTED_CONTINUATION_UPDATE,
        "training_result_path": TRAINING_RESULT.relative_to(ROOT).as_posix(),
        "training_result_sha256": sha256(TRAINING_RESULT),
        "training_result_status": result["status"],
        "vetoes": result["vetoes"],
        "target_validity_failure_count": (
            None
            if result.get("target_validity_failures") is None
            else len(result["target_validity_failures"])
        ),
        "target_validity_event_telemetry": (
            "not_explicitly_recorded_by_v1_result; completion under the legacy "
            "finite-only route implies no invalid status reached that route"
            if result.get("target_validity_failures") is None
            else "explicit"
        ),
        "selected_mean_loss": result["selection"]["selected_mean_loss"],
        "audit_mean_loss": result["audit"]["mean_loss"],
        "support_all_finite": result["support_probe"]["all_finite"],
        "roundtrip_max_abs": result["support_probe"]["roundtrip_max_abs"],
    }
