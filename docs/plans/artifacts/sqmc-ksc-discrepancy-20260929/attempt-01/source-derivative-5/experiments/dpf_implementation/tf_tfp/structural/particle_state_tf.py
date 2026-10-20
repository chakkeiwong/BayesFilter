"""Particle state containers for structural TF filters."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class StructuralParticleStateTF:
    previous_z: tf.Tensor
    previous_s: tf.Tensor
    current_z: tf.Tensor
    current_s: tf.Tensor
    ancestor_indices: tf.Tensor | None = None

    def completed_state(self) -> tf.Tensor:
        return tf.concat([self.current_z, self.current_s], axis=1)

    def previous_completed_state(self) -> tf.Tensor:
        return tf.concat([self.previous_z, self.previous_s], axis=1)


@dataclass(frozen=True)
class StructuralResamplingResultTF:
    next_z: tf.Tensor
    next_s: tf.Tensor
    next_log_weights: tf.Tensor
    diagnostics: dict
