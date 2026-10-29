"""Retired CPU-only LGSSM NeuTra-style affine training fixture.

The historical implementation lacked a repository-bound batch-native target
and is non-executable migration evidence. The public entry point rejects before
optimizer or artifact side effects. Use ``bayesfilter.inference.neutra_training``
with a repository-issued batch-native target binding for live training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.inference import (
    HMCTuningPolicy,
    bind_fixed_transport_hmc_mechanics,
    load_frozen_neutra_artifact,
    stable_frozen_neutra_artifact_signature,
)
from bayesfilter.ssm import stable_ssm_target_signature
from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
    make_lgssm_generic_target_fixture,
)


LGSSM_NEUTRA_TRAINING_NONCLAIMS = (
    "tiny CPU-only affine LGSSM NeuTra-style training fixture",
    "no dense IAF training claim",
    "no HMC convergence claim",
    "no posterior correctness claim",
    "no sampler superiority claim",
    "no production readiness claim",
    "no scientific validity claim",
)

DEFAULT_SEED = 20260707
DEFAULT_STEPS = 80
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_RAW_SCALE = -1.3862943611198906
DEFAULT_ARTIFACT_DIR = Path(
    "docs/plans/artifacts/lgssm-neutra-training-2026-07-07"
)
DEFAULT_VALIDATION_PATH = Path(
    "docs/plans/"
    "bayesfilter-lgssm-first-neutra-hmc-phase6-lgssm-neutra-training-validation-"
    "2026-07-07.json"
)


class LGSSMNeuTraTrainingError(ValueError):
    """Raised when the tiny LGSSM NeuTra-style training gate fails."""


@dataclass(frozen=True)
class LGSSMAffineNeuTraTrainingConfig:
    """Configuration for the bounded Phase 6 affine training fixture."""

    seed: int = DEFAULT_SEED
    steps: int = DEFAULT_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    initial_raw_scale: float = DEFAULT_RAW_SCALE
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    validation_path: Path = DEFAULT_VALIDATION_PATH

    def normalized(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.lgssm_affine_neutra_training_config.v1",
            "seed": int(self.seed),
            "steps": int(self.steps),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "initial_raw_scale": float(self.initial_raw_scale),
            "artifact_dir": str(self.artifact_dir),
            "validation_path": str(self.validation_path),
            "device": "cpu",
            "nonclaims": LGSSM_NEUTRA_TRAINING_NONCLAIMS,
        }


@dataclass(frozen=True)
class LGSSMAffineNeuTraTrainingResult:
    """Materialized training, frozen payload, and validation metadata."""

    config: LGSSMAffineNeuTraTrainingConfig
    target_signature: str
    training_state_path: Path
    payload_path: Path
    validation_path: Path
    training_state: Mapping[str, Any]
    frozen_payload: Mapping[str, Any]
    validation: Mapping[str, Any]

    @property
    def final_loss(self) -> float:
        return float(self.training_state["final_loss"])


def train_and_validate_lgssm_affine_neutra(
    config: LGSSMAffineNeuTraTrainingConfig | None = None,
) -> LGSSMAffineNeuTraTrainingResult:
    """Reject the retired scalar/row-mapped affine optimizer fixture."""

    cfg = LGSSMAffineNeuTraTrainingConfig() if config is None else config
    _validate_config(cfg)
    raise LGSSMNeuTraTrainingError(
        "retired migration debt: this affine fixture lacks a repository-bound "
        "batch-native target and cannot perform NeuTra optimizer updates"
    )


def _historical_train_and_validate_lgssm_affine_neutra(
    config: LGSSMAffineNeuTraTrainingConfig,
) -> LGSSMAffineNeuTraTrainingResult:
    """Preserve the old source as unreachable migration evidence."""

    raise LGSSMNeuTraTrainingError(
        "historical implementation is non-executable migration evidence"
    )
    cfg = config
    start = time.monotonic()
    tf.random.set_seed(int(cfg.seed))
    fixture = make_lgssm_generic_target_fixture()
    target_signature = stable_ssm_target_signature(fixture.contract)
    initial_shift = tf.cast(fixture.initial_batch[0], tf.float64)
    shift = tf.Variable(initial_shift)
    raw_scale = tf.Variable(
        tf.fill(tf.shape(initial_shift), tf.cast(cfg.initial_raw_scale, tf.float64))
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg.learning_rate))

    losses: list[float] = []
    for step in range(int(cfg.steps)):
        z = tf.random.stateless_normal(
            shape=(int(cfg.batch_size), int(initial_shift.shape[0])),
            seed=tf.constant([int(cfg.seed), step], dtype=tf.int32),
            dtype=tf.float64,
        )
        with tf.GradientTape() as tape:
            loss = _reverse_kl_style_loss(
                fixture.adapter,
                z,
                shift=shift,
                raw_scale=raw_scale,
            )
        gradients = tape.gradient(loss, [shift, raw_scale])
        if any(gradient is None for gradient in gradients):
            raise LGSSMNeuTraTrainingError("training gradient must be available")
        optimizer.apply_gradients(zip(gradients, [shift, raw_scale]))
        loss_value = float(loss.numpy())
        if not _is_finite_number(loss_value):
            raise LGSSMNeuTraTrainingError("training loss became nonfinite")
        losses.append(loss_value)

    learned_shift = tuple(float(item) for item in shift.numpy().tolist())
    learned_raw_scale = tuple(float(item) for item in raw_scale.numpy().tolist())
    state_payload = _training_state_payload(
        config=cfg,
        target_signature=target_signature,
        initial_shift=tuple(float(item) for item in initial_shift.numpy().tolist()),
        learned_shift=learned_shift,
        learned_raw_scale=learned_raw_scale,
        losses=losses,
        elapsed_seconds=time.monotonic() - start,
    )
    training_state_hash = _stable_json_hash(state_payload)
    frozen_payload = _frozen_affine_payload(
        seed=int(cfg.seed),
        target_signature=target_signature,
        shift=learned_shift,
        raw_scale=learned_raw_scale,
        training_state_hash=f"sha256:{training_state_hash}",
    )
    loaded = load_frozen_neutra_artifact(
        frozen_payload,
        expected_target_signature=target_signature,
    )
    validation = _validation_payload(
        fixture=fixture,
        loaded_artifact=loaded,
        config=cfg,
        target_signature=target_signature,
        training_state_hash=training_state_hash,
        training_state=state_payload,
    )
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.validation_path.parent.mkdir(parents=True, exist_ok=True)
    training_state_path = cfg.artifact_dir / (
        f"lgssm_affine_neutra_training_state_seed{int(cfg.seed)}.json"
    )
    payload_path = cfg.artifact_dir / (
        f"lgssm_affine_neutra_payload_seed{int(cfg.seed)}.json"
    )
    _write_json(training_state_path, state_payload)
    _write_json(payload_path, frozen_payload)
    _write_json(cfg.validation_path, validation)
    return LGSSMAffineNeuTraTrainingResult(
        config=cfg,
        target_signature=target_signature,
        training_state_path=training_state_path,
        payload_path=payload_path,
        validation_path=cfg.validation_path,
        training_state=state_payload,
        frozen_payload=frozen_payload,
        validation=validation,
    )


def load_lgssm_affine_neutra_payload(path: str | Path) -> Mapping[str, Any]:
    """Read a frozen affine payload JSON file for validation/tests."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise LGSSMNeuTraTrainingError("frozen payload JSON must be a mapping")
    return payload


def _reverse_kl_style_loss(
    adapter: Any,
    z: tf.Tensor,
    *,
    shift: tf.Variable,
    raw_scale: tf.Variable,
) -> tf.Tensor:
    scale = tf.exp(raw_scale)
    theta = shift[tf.newaxis, :] + scale[tf.newaxis, :] * z
    value, _score = adapter.log_prob_and_grad(theta)
    logdet = tf.reduce_sum(raw_scale)
    loss = -tf.reduce_mean(value + logdet)
    tf.debugging.assert_all_finite(loss, "LGSSM affine NeuTra loss must be finite")
    return loss


def _training_state_payload(
    *,
    config: LGSSMAffineNeuTraTrainingConfig,
    target_signature: str,
    initial_shift: Sequence[float],
    learned_shift: Sequence[float],
    learned_raw_scale: Sequence[float],
    losses: Sequence[float],
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    if not losses:
        raise LGSSMNeuTraTrainingError("loss history must be nonempty")
    return {
        "schema": "bayesfilter.lgssm_affine_neutra_training_state.v1",
        "target_signature": target_signature,
        "training_config": config.normalized(),
        "initial_shift": list(initial_shift),
        "initial_raw_scale": [float(config.initial_raw_scale)] * len(initial_shift),
        "final_shift": list(learned_shift),
        "final_raw_scale": list(learned_raw_scale),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_history": [float(item) for item in losses],
        "elapsed_seconds": float(elapsed_seconds),
        "training_objective": (
            "reverse_kl_style_mean_negative_log_p_forward_transport_plus_logdet"
        ),
        "device": "cpu",
        "nonclaims": LGSSM_NEUTRA_TRAINING_NONCLAIMS,
    }


def _frozen_affine_payload(
    *,
    seed: int,
    target_signature: str,
    shift: Sequence[float],
    raw_scale: Sequence[float],
    training_state_hash: str,
) -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.neutra.frozen_affine_diag.v1",
        "transport_id": f"lgssm-learned-affine-diag-neutra-seed{int(seed)}",
        "dimension": len(tuple(shift)),
        "target_signature": target_signature,
        "log_jacobian_available": True,
        "shift": [float(item) for item in shift],
        "raw_scale": [float(item) for item in raw_scale],
        "training_state_hash": training_state_hash,
    }


def _validation_payload(
    *,
    fixture: Any,
    loaded_artifact: Any,
    config: LGSSMAffineNeuTraTrainingConfig,
    target_signature: str,
    training_state_hash: str,
    training_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    z = tf.constant([[0.0, 0.0], [0.15, -0.20]], dtype=tf.float64)
    theta = loaded_artifact.transport.forward_batch(z)
    base_value, base_score = fixture.adapter.log_prob_and_grad(theta)
    source_value, source_score = fixture.source_target.target_log_prob_and_grad(
        fixture.initial_batch[0]
    )
    adapter_value, adapter_score = fixture.adapter.log_prob_and_grad(fixture.initial_batch)
    mechanics = bind_fixed_transport_hmc_mechanics(
        base_adapter=fixture.adapter,
        loaded_artifact=loaded_artifact,
        initial_position=tf.constant([[0.0, 0.0]], dtype=tf.float64),
        target_scope="lgssm-learned-affine-neutra-phase6-mechanics",
        tuning_policy=HMCTuningPolicy.fixed_kernel_screen(),
        use_xla=False,
        seed=int(config.seed),
        execution_device="cpu",
    )
    value_residual = float(tf.reduce_max(tf.abs(adapter_value[0] - source_value)).numpy())
    score_residual = float(
        tf.reduce_max(tf.abs(adapter_score[0] - source_score)).numpy()
    )
    mechanics_value_finite = bool(
        tf.reduce_all(tf.math.is_finite(mechanics.value)).numpy()
    )
    mechanics_score_finite = bool(
        tf.reduce_all(tf.math.is_finite(mechanics.score)).numpy()
    )
    base_values_finite = bool(tf.reduce_all(tf.math.is_finite(base_value)).numpy())
    base_scores_finite = bool(tf.reduce_all(tf.math.is_finite(base_score)).numpy())
    passed = (
        loaded_artifact.manifest.target_signature == target_signature
        and mechanics_value_finite
        and mechanics_score_finite
        and base_values_finite
        and base_scores_finite
        and value_residual <= 1.0e-10
        and score_residual <= 1.0e-8
    )
    return {
        "schema": "bayesfilter.lgssm_affine_neutra_training_validation.v1",
        "passed": bool(passed),
        "target_signature": target_signature,
        "artifact_signature": stable_frozen_neutra_artifact_signature(loaded_artifact),
        "training_state_hash": f"sha256:{training_state_hash}",
        "transport_hash": loaded_artifact.manifest.transport_hash,
        "mechanics_manifest": mechanics.manifest.manifest_payload(),
        "mechanics_diagnostics": _json_safe(mechanics.diagnostics),
        "mechanics_value": _json_safe(mechanics.value),
        "mechanics_score": _json_safe(mechanics.score),
        "reference_residuals": {
            "initial_batch_value_residual": value_residual,
            "initial_batch_score_residual": score_residual,
        },
        "finite_checks": {
            "base_values_finite": base_values_finite,
            "base_scores_finite": base_scores_finite,
            "mechanics_value_finite": mechanics_value_finite,
            "mechanics_score_finite": mechanics_score_finite,
        },
        "training_summary": {
            "initial_loss": float(training_state["initial_loss"]),
            "final_loss": float(training_state["final_loss"]),
            "steps": int(config.steps),
            "batch_size": int(config.batch_size),
            "learning_rate": float(config.learning_rate),
            "loss_is_explanatory_only": True,
        },
        "device": "cpu",
        "nonclaims": LGSSM_NEUTRA_TRAINING_NONCLAIMS,
    }


def _validate_config(config: LGSSMAffineNeuTraTrainingConfig) -> None:
    if int(config.seed) < 0:
        raise LGSSMNeuTraTrainingError("seed must be nonnegative")
    if int(config.steps) <= 0 or int(config.steps) > 1000:
        raise LGSSMNeuTraTrainingError("steps must be in 1..1000")
    if int(config.batch_size) <= 0 or int(config.batch_size) > 2048:
        raise LGSSMNeuTraTrainingError("batch_size must be in 1..2048")
    if float(config.learning_rate) <= 0.0:
        raise LGSSMNeuTraTrainingError("learning_rate must be positive")
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        raise LGSSMNeuTraTrainingError("Phase 6 fixture must run with CPU-only device hiding")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_json_hash(payload: Any) -> str:
    blob = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _is_finite_number(value: float) -> bool:
    return value == value and abs(value) < float("inf")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the bounded Phase 6 validation run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--validation-path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args(argv)
    result = train_and_validate_lgssm_affine_neutra(
        LGSSMAffineNeuTraTrainingConfig(
            seed=args.seed,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            artifact_dir=args.artifact_dir,
            validation_path=args.validation_path,
        )
    )
    print(
        json.dumps(
            {
                "passed": bool(result.validation["passed"]),
                "training_state_path": str(result.training_state_path),
                "payload_path": str(result.payload_path),
                "validation_path": str(result.validation_path),
                "target_signature": result.target_signature,
                "final_loss": result.final_loss,
                "nonclaims": LGSSM_NEUTRA_TRAINING_NONCLAIMS,
            },
            sort_keys=True,
        )
    )
    if not bool(result.validation["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
