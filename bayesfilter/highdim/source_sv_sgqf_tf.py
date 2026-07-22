"""Source-order scalar-SV dataset shared by actual and KSC leaderboard rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.models import StochasticVolatilitySSM


SOURCE_SV_SEED = 81101
SOURCE_SV_HORIZON = 1000
SOURCE_SV_STATE_SHA256 = (
    "0c650cc40dee423556117f62ed599c6c206674eb6c8708423873ec754ca3c52c"
)
SOURCE_SV_OBSERVATION_SHA256 = (
    "f52d6a0221b21bf4d3bdf667d3b64b97a7ef14d0b18ab1f98a471820cc491c79"
)
SOURCE_SV_TARGET_ID = "zhao_cui_sv_tf_seed81101_x0_then_y1_y1000_v1"


def _tensor_hash(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def generate_source_order_sv_dataset_tf() -> tuple[tf.Tensor, tf.Tensor]:
    """Replay ``x0:x1000`` and ``y1:y1000`` in author-program event order."""

    with tf.device("/CPU:0"):
        model = StochasticVolatilitySSM(sigma=1.0)
        theta = model.unconstrained_from_physical(gamma=0.6, beta=0.4)
        parameters = model.physical_parameters(theta)
        gamma = parameters["gamma"]
        beta = parameters["beta"]
        generator = tf.random.Generator.from_seed(SOURCE_SV_SEED)
        state = (
            model.sigma
            / tf.sqrt(1.0 - tf.square(gamma))
            * generator.normal([], dtype=tf.float64)
        )
        states = [state]
        observations = []
        for _time_index in range(SOURCE_SV_HORIZON):
            state = gamma * state + model.sigma * generator.normal([], dtype=tf.float64)
            observations.append(
                beta
                * tf.exp(0.5 * state)
                * generator.normal([], dtype=tf.float64)
            )
            states.append(state)
        state_path = tf.reshape(tf.stack(states), [-1, 1])
        observation_path = tf.reshape(tf.stack(observations), [-1, 1])
        if _tensor_hash(state_path) != SOURCE_SV_STATE_SHA256:
            raise ValueError("source-order SV state hash mismatch")
        if _tensor_hash(observation_path) != SOURCE_SV_OBSERVATION_SHA256:
            raise ValueError("source-order SV observation hash mismatch")
        return state_path, observation_path


@dataclass(frozen=True)
class SourceOrderSVDataset:
    states: tf.Tensor
    observations: tf.Tensor
    identity: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        states = tf.convert_to_tensor(self.states, dtype=tf.float64)
        observations = tf.convert_to_tensor(self.observations, dtype=tf.float64)
        if states.shape != (SOURCE_SV_HORIZON + 1, 1):
            raise ValueError("canonical source-order SV states require shape [1001, 1]")
        if observations.shape != (SOURCE_SV_HORIZON, 1):
            raise ValueError("canonical source-order SV observations require shape [1000, 1]")
        if _tensor_hash(states) != SOURCE_SV_STATE_SHA256:
            raise ValueError("canonical source-order SV state identity rejected")
        if _tensor_hash(observations) != SOURCE_SV_OBSERVATION_SHA256:
            raise ValueError("canonical source-order SV observation identity rejected")
        manifest = dict(self.manifest)
        if self.identity != _semantic_hash(manifest):
            raise ValueError("source-order SV dataset identity rejected")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))


def make_source_order_sv_dataset() -> SourceOrderSVDataset:
    states, observations = generate_source_order_sv_dataset_tf()
    manifest: dict[str, object] = {
        "schema": "bayesfilter.source_order_sv_dataset.v1",
        "target_id": SOURCE_SV_TARGET_ID,
        "seed": SOURCE_SV_SEED,
        "horizon": SOURCE_SV_HORIZON,
        "time_order": "x0_then_1000_transition_then_observe_steps_y1_y1000",
        "state_sha256": SOURCE_SV_STATE_SHA256,
        "observation_sha256": SOURCE_SV_OBSERVATION_SHA256,
        "truth_physical": {"gamma": 0.6, "beta": 0.4, "sigma": 1.0},
        "dtype": "float64",
        "source_anchor": (
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m:34"
        ),
        "rng_classification": "tensorflow_source_model_synthetic_replication",
        "nonclaims": [
            "not MATLAB rng(1) bitwise replay",
            "not likelihood or filtering correctness evidence by itself",
        ],
    }
    return SourceOrderSVDataset(
        states=states,
        observations=observations,
        identity=_semantic_hash(manifest),
        manifest=manifest,
    )


__all__ = [
    "SOURCE_SV_HORIZON",
    "SOURCE_SV_OBSERVATION_SHA256",
    "SOURCE_SV_SEED",
    "SOURCE_SV_STATE_SHA256",
    "SOURCE_SV_TARGET_ID",
    "SourceOrderSVDataset",
    "generate_source_order_sv_dataset_tf",
    "make_source_order_sv_dataset",
]
