"""TensorFlow/XLA execution primitives for searched NeuTra curricula."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import tensorflow as tf

from bayesfilter.inference.neutra_staged_training import (
    NeuTraVariableGroup,
    neutra_full_variable_masks,
)


class NeuTraCurriculumTrainingError(RuntimeError):
    """Raised when curriculum training violates its finite or budget contract."""


@dataclass(frozen=True)
class NeuTraCurriculumPhaseResult:
    name: str
    active_groups: tuple[str, ...]
    first_global_update: int
    last_global_update: int
    updates: int
    clipped_updates: int
    terminal_gradient_norm: float


@dataclass(frozen=True)
class NeuTraCurriculumProtocolResult:
    sequence: tuple[str, ...]
    learning_rate: float
    terminal_loss: float
    executed_updates: int
    phases: tuple[NeuTraCurriculumPhaseResult, ...]
    final_state: tuple[tf.Tensor, ...]


@dataclass(frozen=True)
class NeuTraCurriculumProbeRateResult:
    learning_rate: float
    terminal_loss: float
    clipped_updates: int
    terminal_gradient_norm: float
    terminal_state: tuple[tf.Tensor, ...]


@dataclass(frozen=True)
class NeuTraCurriculumProbeResult:
    active_groups: tuple[str, ...]
    incoming_loss: float
    selected_learning_rate: float
    selected_loss: float
    selected_state: tuple[tf.Tensor, ...]
    candidates: tuple[NeuTraCurriculumProbeRateResult, ...]
    tuning_optimizer_updates: int


def _state(variables: Sequence[tf.Variable]) -> tuple[tf.Tensor, ...]:
    return tuple(tf.identity(variable) for variable in variables)


def _restore(variables: Sequence[tf.Variable], state: Sequence[tf.Tensor]) -> None:
    if len(variables) != len(state):
        raise NeuTraCurriculumTrainingError("state variable count mismatch")
    for variable, value in zip(variables, state, strict=True):
        tensor = tf.convert_to_tensor(value, variable.dtype)
        if tensor.shape != variable.shape:
            raise NeuTraCurriculumTrainingError("state variable shape mismatch")
        variable.assign(tensor)


def _selection_loss(
    transport: Any, selection_loss_fn: Callable[[Any], tf.Tensor]
) -> float:
    value = tf.convert_to_tensor(selection_loss_fn(transport), tf.float64)
    if value.shape.rank != 0:
        raise NeuTraCurriculumTrainingError("selection_loss_fn must return a scalar")
    tf.debugging.assert_all_finite(value, "curriculum selection loss")
    return float(value.numpy())


def _scheduled_learning_rate(peak: float, update: int, total_updates: int) -> float:
    fraction = float(update) / float(total_updates)
    multiplier = 1.0 if fraction < 0.60 else 0.1 if fraction < 0.85 else 0.01
    return float(peak) * multiplier


def _train_phase(
    *,
    transport: Any,
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    masks: Sequence[tf.Tensor],
    latent_batch_fn: Callable[[int], tf.Tensor],
    first_global_update: int,
    updates: int,
    total_updates: int,
    peak_learning_rate: float,
    gradient_clip_norm: float,
    jit_compile: bool,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[int, float]:
    variables = tuple(transport.trainable_variables)
    masks = tuple(tf.cast(mask, variable.dtype) for mask, variable in zip(masks, variables, strict=True))
    if not any(bool(tf.reduce_any(mask > 0.0).numpy()) for mask in masks):
        raise ValueError("curriculum phase resolved to no active parameters")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(peak_learning_rate),
        beta_1=float(beta1),
        beta_2=float(beta2),
        epsilon=float(epsilon),
    )
    optimizer.build(variables)

    @tf.function(jit_compile=bool(jit_compile), reduce_retracing=True)
    def train_step(latent: tf.Tensor) -> tuple[tf.Tensor, ...]:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)
            physical, logdet = transport.forward_and_logdet(latent)
            target = tf.convert_to_tensor(target_log_prob_fn(physical), tf.float64)
            loss = tf.reduce_mean(-target - logdet)
        gradients = tuple(tape.gradient(loss, variables))
        if any(gradient is None for gradient in gradients):
            raise NeuTraCurriculumTrainingError("curriculum gradient is missing")
        masked = tuple(
            tf.convert_to_tensor(gradient) * mask
            for gradient, mask in zip(gradients, masks, strict=True)
        )
        gradient_norm = tf.linalg.global_norm(masked)
        clipped, _ = tf.clip_by_global_norm(
            masked,
            tf.constant(float(gradient_clip_norm), tf.float64),
            use_norm=gradient_norm,
        )
        finite = tf.reduce_all(
            tf.stack(
                (
                    tf.reduce_all(tf.math.is_finite(loss)),
                    tf.reduce_all(tf.math.is_finite(target)),
                    tf.reduce_all(tf.math.is_finite(logdet)),
                    tf.reduce_all(tf.math.is_finite(gradient_norm)),
                    *(tf.reduce_all(tf.math.is_finite(value)) for value in clipped),
                )
            )
        )

        def update() -> tf.Tensor:
            optimizer.apply_gradients(zip(clipped, variables))
            return tf.cast(optimizer.iterations, tf.int64)

        step = tf.cond(finite, update, lambda: tf.cast(-1, tf.int64))
        return loss, gradient_norm, finite, step

    clipped_updates = 0
    terminal_gradient_norm = 0.0
    for local_update in range(1, int(updates) + 1):
        global_update = int(first_global_update) + local_update - 1
        optimizer.learning_rate.assign(
            _scheduled_learning_rate(
                float(peak_learning_rate), global_update, int(total_updates)
            )
        )
        latent = tf.convert_to_tensor(latent_batch_fn(global_update), tf.float64)
        if latent.shape.rank != 2 or int(latent.shape[0]) <= 1:
            raise NeuTraCurriculumTrainingError(
                "latent batches must be rank two with batch size above one"
            )
        _loss, gradient_norm, finite, _step = train_step(latent)
        if not bool(finite.numpy()):
            raise NeuTraCurriculumTrainingError(
                f"nonfinite curriculum update at global step {global_update}"
            )
        terminal_gradient_norm = float(gradient_norm.numpy())
        clipped_updates += int(terminal_gradient_norm > float(gradient_clip_norm))
    return clipped_updates, terminal_gradient_norm


def tune_neutra_curriculum_probe(
    *,
    transport: Any,
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    variable_groups: Sequence[NeuTraVariableGroup],
    active_groups: Sequence[str],
    learning_rates: Sequence[float],
    updates: int,
    latent_batch_fn: Callable[[int], tf.Tensor],
    selection_loss_fn: Callable[[Any], tf.Tensor],
    gradient_clip_norm: float = 10.0,
    jit_compile: bool = True,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-7,
) -> NeuTraCurriculumProbeResult:
    """Tune one equal-budget cumulative group-addition probe."""

    if int(updates) <= 0:
        raise ValueError("probe updates must be positive")
    rates = tuple(float(value) for value in learning_rates)
    if not rates or any(not math.isfinite(value) or value <= 0.0 for value in rates):
        raise ValueError("probe learning rates must be finite and positive")
    variables = tuple(transport.trainable_variables)
    incoming_state = _state(variables)
    incoming_loss = _selection_loss(transport, selection_loss_fn)
    masks = neutra_full_variable_masks(
        transport=transport,
        variable_groups=variable_groups,
        active_groups=tuple(active_groups),
    )
    candidates = []
    for rate in rates:
        _restore(variables, incoming_state)
        clipped_updates, gradient_norm = _train_phase(
            transport=transport,
            target_log_prob_fn=target_log_prob_fn,
            masks=masks,
            latent_batch_fn=latent_batch_fn,
            first_global_update=1,
            updates=int(updates),
            total_updates=int(updates),
            peak_learning_rate=rate,
            gradient_clip_norm=float(gradient_clip_norm),
            jit_compile=bool(jit_compile),
            beta1=float(beta1),
            beta2=float(beta2),
            epsilon=float(epsilon),
        )
        candidates.append(
            NeuTraCurriculumProbeRateResult(
                learning_rate=rate,
                terminal_loss=_selection_loss(transport, selection_loss_fn),
                clipped_updates=clipped_updates,
                terminal_gradient_norm=gradient_norm,
                terminal_state=_state(variables),
            )
        )
    selected = min(candidates, key=lambda item: (item.terminal_loss, item.learning_rate))
    _restore(variables, selected.terminal_state)
    return NeuTraCurriculumProbeResult(
        active_groups=tuple(active_groups),
        incoming_loss=incoming_loss,
        selected_learning_rate=selected.learning_rate,
        selected_loss=selected.terminal_loss,
        selected_state=_state(variables),
        candidates=tuple(candidates),
        tuning_optimizer_updates=len(rates) * int(updates),
    )


def train_neutra_curriculum_protocol(
    *,
    transport: Any,
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    variable_groups: Sequence[NeuTraVariableGroup],
    sequence: Sequence[str],
    learning_rate: float,
    total_updates: int,
    warmup_updates_per_group: int,
    latent_batch_fn: Callable[[int], tf.Tensor],
    selection_loss_fn: Callable[[Any], tf.Tensor],
    gradient_clip_norm: float = 10.0,
    jit_compile: bool = True,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-7,
) -> NeuTraCurriculumProtocolResult:
    """Train one fixed sequence/LR protocol under an exact update budget."""

    sequence = tuple(str(item) for item in sequence)
    if len(set(sequence)) != len(sequence):
        raise ValueError("curriculum sequence must not contain duplicate groups")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if int(total_updates) <= 0 or int(warmup_updates_per_group) <= 0:
        raise ValueError("protocol update budgets must be positive")
    warmup_total = len(sequence) * int(warmup_updates_per_group)
    if warmup_total >= int(total_updates):
        raise ValueError("protocol must retain at least one joint update")
    all_groups = tuple(group.name for group in variable_groups)
    phases = []
    global_update = 1
    for index in range(len(sequence)):
        active = sequence[: index + 1]
        masks = neutra_full_variable_masks(
            transport=transport,
            variable_groups=variable_groups,
            active_groups=active,
        )
        clipped, gradient_norm = _train_phase(
            transport=transport,
            target_log_prob_fn=target_log_prob_fn,
            masks=masks,
            latent_batch_fn=latent_batch_fn,
            first_global_update=global_update,
            updates=int(warmup_updates_per_group),
            total_updates=int(total_updates),
            peak_learning_rate=float(learning_rate),
            gradient_clip_norm=float(gradient_clip_norm),
            jit_compile=bool(jit_compile),
            beta1=float(beta1),
            beta2=float(beta2),
            epsilon=float(epsilon),
        )
        last = global_update + int(warmup_updates_per_group) - 1
        phases.append(
            NeuTraCurriculumPhaseResult(
                name=f"warmup_{index}_{sequence[index]}",
                active_groups=active,
                first_global_update=global_update,
                last_global_update=last,
                updates=int(warmup_updates_per_group),
                clipped_updates=clipped,
                terminal_gradient_norm=gradient_norm,
            )
        )
        global_update = last + 1
    joint_updates = int(total_updates) - warmup_total
    joint_masks = neutra_full_variable_masks(
        transport=transport,
        variable_groups=variable_groups,
        active_groups=all_groups,
    )
    clipped, gradient_norm = _train_phase(
        transport=transport,
        target_log_prob_fn=target_log_prob_fn,
        masks=joint_masks,
        latent_batch_fn=latent_batch_fn,
        first_global_update=global_update,
        updates=joint_updates,
        total_updates=int(total_updates),
        peak_learning_rate=float(learning_rate),
        gradient_clip_norm=float(gradient_clip_norm),
        jit_compile=bool(jit_compile),
        beta1=float(beta1),
        beta2=float(beta2),
        epsilon=float(epsilon),
    )
    phases.append(
        NeuTraCurriculumPhaseResult(
            name="joint",
            active_groups=all_groups,
            first_global_update=global_update,
            last_global_update=int(total_updates),
            updates=joint_updates,
            clipped_updates=clipped,
            terminal_gradient_norm=gradient_norm,
        )
    )
    return NeuTraCurriculumProtocolResult(
        sequence=sequence,
        learning_rate=float(learning_rate),
        terminal_loss=_selection_loss(transport, selection_loss_fn),
        executed_updates=sum(phase.updates for phase in phases),
        phases=tuple(phases),
        final_state=_state(tuple(transport.trainable_variables)),
    )
