"""Contracts for experimental TF structural state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import tensorflow as tf


DTYPE = tf.float64


@dataclass(frozen=True)
class StructuralFlowProposalTF:
    post_z: tf.Tensor
    pre_flow_log_density: tf.Tensor
    forward_log_det: tf.Tensor
    diagnostics: dict


class StructuralSSMModelTF(Protocol):
    model_id: str

    def initial_z_sample(self, num_particles: int, seed: int) -> tf.Tensor:
        ...

    def initial_z_log_prob(self, z: tf.Tensor) -> tf.Tensor:
        ...

    def initial_s_from_z(self, z: tf.Tensor) -> tf.Tensor:
        ...

    def transition_z_sample(self, previous_z: tf.Tensor, seed: int, time_index: int) -> tf.Tensor:
        ...

    def transition_z_log_prob(self, current_z: tf.Tensor, previous_z: tf.Tensor, time_index: int) -> tf.Tensor:
        ...

    def complete_s(self, previous_s: tf.Tensor, previous_z: tf.Tensor, current_z: tf.Tensor) -> tf.Tensor:
        ...

    def observation_log_prob(
        self,
        current_z: tf.Tensor,
        current_s: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        ...

    def completion_residual(
        self,
        previous_s: tf.Tensor,
        previous_z: tf.Tensor,
        current_z: tf.Tensor,
        current_s: tf.Tensor,
    ) -> tf.Tensor:
        ...

    def local_flow_proposal(
        self,
        pre_z: tf.Tensor,
        previous_z: tf.Tensor,
        previous_s: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> StructuralFlowProposalTF:
        ...
