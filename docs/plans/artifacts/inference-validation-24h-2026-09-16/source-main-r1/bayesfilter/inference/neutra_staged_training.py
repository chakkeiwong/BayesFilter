"""Generic five-stage continuation training for TensorFlow NeuTra transports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import tensorflow as tf


class NeuTraStagedTrainingError(RuntimeError):
    """Raised when a staged NeuTra run violates its finite or phase contract."""


@dataclass(frozen=True)
class NeuTraVariablePart:
    """One trainable tensor or a binary-masked portion of that tensor."""

    variable: tf.Variable
    mask: tf.Tensor | None = None


@dataclass(frozen=True)
class NeuTraVariableGroup:
    """Named collection of trainable tensor parts activated as one unit."""

    name: str
    parts: tuple[NeuTraVariablePart, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.parts:
            raise ValueError("variable groups require a name and at least one part")


@dataclass(frozen=True)
class NeuTraAdaptiveStagePolicy:
    """Bounded held-out plateau scheduler for one optimizer phase."""

    minimum_updates: int
    patience_checkpoints: int
    minimum_improvement: float
    learning_rate_reduction_factor: float
    maximum_learning_rate_reductions: int

    def __post_init__(self) -> None:
        if int(self.minimum_updates) < 0:
            raise ValueError("minimum_updates must be nonnegative")
        if int(self.patience_checkpoints) <= 0:
            raise ValueError("patience_checkpoints must be positive")
        if (
            not math.isfinite(float(self.minimum_improvement))
            or float(self.minimum_improvement) < 0.0
        ):
            raise ValueError("minimum_improvement must be finite and nonnegative")
        factor = float(self.learning_rate_reduction_factor)
        if not math.isfinite(factor) or not 0.0 < factor < 1.0:
            raise ValueError("learning_rate_reduction_factor must lie between zero and one")
        if int(self.maximum_learning_rate_reductions) < 0:
            raise ValueError("maximum_learning_rate_reductions must be nonnegative")


@dataclass(frozen=True)
class NeuTraStageSpec:
    """One independently tuned optimizer phase inside stages one through four."""

    name: str
    stage: int
    active_groups: tuple[str, ...]
    updates: int
    learning_rates: tuple[float, ...]
    checkpoint_every: int
    learning_rate_schedule: str = "piecewise_60_85"
    adaptive_policy: NeuTraAdaptiveStagePolicy | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must be nonempty")
        if int(self.stage) not in {1, 2, 3, 4}:
            raise ValueError("optimizer stages must be numbered one through four")
        if not self.active_groups or len(set(self.active_groups)) != len(self.active_groups):
            raise ValueError("active_groups must be nonempty and unique")
        if int(self.updates) <= 0:
            raise ValueError("stage updates must be positive")
        if int(self.checkpoint_every) <= 0 or int(self.checkpoint_every) > int(
            self.updates
        ):
            raise ValueError("checkpoint_every must lie within the stage budget")
        if not self.learning_rates or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.learning_rates
        ):
            raise ValueError("learning_rates must be finite and positive")
        if self.learning_rate_schedule not in {"constant", "piecewise_60_85"}:
            raise ValueError("unsupported learning_rate_schedule")
        if self.adaptive_policy is not None:
            if self.learning_rate_schedule != "constant":
                raise ValueError("adaptive phases require a constant base schedule")
            if int(self.adaptive_policy.minimum_updates) > int(self.updates):
                raise ValueError("adaptive minimum_updates cannot exceed the phase cap")


@dataclass(frozen=True)
class NeuTraFiveStageSpec:
    """Four optimizer stages followed by one untouched validation stage."""

    affine: NeuTraStageSpec
    simple: NeuTraStageSpec
    progressive: tuple[NeuTraStageSpec, ...]
    joint: NeuTraStageSpec

    def __post_init__(self) -> None:
        if int(self.affine.stage) != 1 or int(self.simple.stage) != 2:
            raise ValueError("affine and simple phases must be stages one and two")
        if not self.progressive or any(
            int(phase.stage) != 3 for phase in self.progressive
        ):
            raise ValueError("progressive phases must be a nonempty stage-three sequence")
        if int(self.joint.stage) != 4:
            raise ValueError("joint phase must be stage four")
        names = tuple(phase.name for phase in self.optimizer_phases())
        if len(set(names)) != len(names):
            raise ValueError("optimizer phase names must be unique")

    def optimizer_phases(self) -> tuple[NeuTraStageSpec, ...]:
        return (self.affine, self.simple, *self.progressive, self.joint)


@dataclass(frozen=True)
class NeuTraLearningRateResult:
    learning_rate: float
    selected_update: int
    selected_loss: float
    terminal_loss: float
    clipped_updates: int
    gradient_norm: float
    executed_updates: int
    learning_rate_reductions: int
    stop_reason: str
    checkpoint_history: tuple[tuple[int, float, float, int], ...]
    selected_state: tuple[tf.Tensor, ...]
    terminal_state: tuple[tf.Tensor, ...]
    selected_optimizer_state: tuple[tf.Tensor, ...] = ()
    terminal_optimizer_state: tuple[tf.Tensor, ...] = ()


@dataclass(frozen=True)
class NeuTraStageResult:
    name: str
    stage: int
    active_groups: tuple[str, ...]
    trainable_variables: tuple[str, ...]
    incoming_loss: float
    selected_learning_rate: float
    selected_update: int
    selected_loss: float
    optimizer_state_policy: str
    incoming_optimizer_iterations: int
    selected_optimizer_iterations: int
    candidates: tuple[NeuTraLearningRateResult, ...]


@dataclass(frozen=True)
class NeuTraFiveStageResult:
    stages: tuple[NeuTraStageResult, ...]
    validation: Mapping[str, Any]
    selected_path_updates: int
    tuning_optimizer_updates: int
    final_state: tuple[tf.Tensor, ...]
    optimizer_state_policy: str
    nonclaims: tuple[str, ...]


def _scheduled_learning_rate(
    peak: float, schedule: str, update: int, total_updates: int
) -> float:
    if schedule == "constant":
        return float(peak)
    fraction = float(update) / float(total_updates)
    if fraction < 0.60:
        multiplier = 1.0
    elif fraction < 0.85:
        multiplier = 0.1
    else:
        multiplier = 0.01
    return float(peak) * multiplier


def _state(variables: Sequence[tf.Variable]) -> tuple[tf.Tensor, ...]:
    return tuple(tf.identity(variable) for variable in variables)


def _restore(variables: Sequence[tf.Variable], state: Sequence[tf.Tensor]) -> None:
    if len(variables) != len(state):
        raise NeuTraStagedTrainingError("state variable count mismatch")
    for variable, value in zip(variables, state, strict=True):
        tensor = tf.convert_to_tensor(value, variable.dtype)
        if tensor.shape != variable.shape:
            raise NeuTraStagedTrainingError("state variable shape mismatch")
        variable.assign(tensor)


def _finite_scalar(value: Any, name: str) -> float:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 0:
        raise NeuTraStagedTrainingError(f"{name} must return a scalar")
    tf.debugging.assert_all_finite(tensor, name)
    return float(tensor.numpy())


def _normalize_groups(
    transport_variables: tuple[tf.Variable, ...],
    groups: Sequence[NeuTraVariableGroup],
) -> Mapping[str, Mapping[int, tf.Tensor]]:
    variable_by_id = {id(variable): variable for variable in transport_variables}
    if len(variable_by_id) != len(transport_variables):
        raise ValueError("transport trainable variables must be unique")
    output: dict[str, dict[int, tf.Tensor]] = {}
    occupancy: dict[int, tf.Tensor] = {
        id(variable): tf.zeros_like(variable, tf.float64)
        for variable in transport_variables
    }
    for group in groups:
        if group.name in output:
            raise ValueError(f"duplicate variable group: {group.name}")
        normalized: dict[int, tf.Tensor] = {}
        for part in group.parts:
            key = id(part.variable)
            if key not in variable_by_id:
                raise ValueError("variable group contains a foreign variable")
            variable = variable_by_id[key]
            mask = (
                tf.ones_like(variable, tf.float64)
                if part.mask is None
                else tf.convert_to_tensor(part.mask, tf.float64)
            )
            if mask.shape != variable.shape:
                raise ValueError("variable-part mask shape mismatch")
            valid = tf.reduce_all(
                tf.logical_or(tf.equal(mask, 0.0), tf.equal(mask, 1.0))
            )
            if not bool(valid.numpy()):
                raise ValueError("variable-part masks must be binary")
            existing = normalized.get(key, tf.zeros_like(mask))
            normalized[key] = tf.maximum(existing, mask)
        output[group.name] = normalized
        for key, mask in normalized.items():
            occupancy[key] = occupancy[key] + mask
    for value in occupancy.values():
        if not bool(tf.reduce_all(value <= 1.0).numpy()):
            raise ValueError("variable groups must not overlap elementwise")
    return output


def _active_variables_and_masks(
    transport_variables: tuple[tf.Variable, ...],
    groups: Mapping[str, Mapping[int, tf.Tensor]],
    names: Sequence[str],
) -> tuple[tuple[tf.Variable, ...], tuple[tf.Tensor, ...]]:
    missing = tuple(name for name in names if name not in groups)
    if missing:
        raise ValueError("unknown variable groups: " + ", ".join(missing))
    active_masks: dict[int, tf.Tensor] = {}
    for name in names:
        for key, mask in groups[name].items():
            active_masks[key] = tf.maximum(
                active_masks.get(key, tf.zeros_like(mask)), mask
            )
    variables = []
    masks = []
    for variable in transport_variables:
        key = id(variable)
        if key in active_masks and bool(tf.reduce_any(active_masks[key] > 0.0).numpy()):
            variables.append(variable)
            masks.append(tf.cast(active_masks[key], variable.dtype))
    if not variables:
        raise ValueError("optimizer phase resolved to no trainable variables")
    return tuple(variables), tuple(masks)


def _validate_joint_coverage(
    transport_variables: tuple[tf.Variable, ...],
    active_variables: tuple[tf.Variable, ...],
    active_masks: tuple[tf.Tensor, ...],
) -> None:
    coverage = {id(variable): mask for variable, mask in zip(active_variables, active_masks)}
    for variable in transport_variables:
        mask = coverage.get(id(variable))
        if mask is None or not bool(tf.reduce_all(tf.equal(mask, 1.0)).numpy()):
            raise ValueError("joint stage must cover every transport parameter")


def neutra_full_variable_masks(
    *,
    transport: Any,
    variable_groups: Sequence[NeuTraVariableGroup],
    active_groups: Sequence[str],
) -> tuple[tf.Tensor, ...]:
    """Return one validated mask for every transport trainable variable."""

    transport_variables = tuple(transport.trainable_variables)
    if not transport_variables:
        raise ValueError("transport must expose trainable_variables")
    groups = _normalize_groups(transport_variables, variable_groups)
    active_variables, active_masks = _active_variables_and_masks(
        transport_variables, groups, tuple(active_groups)
    )
    by_id = {
        id(variable): tf.cast(mask, variable.dtype)
        for variable, mask in zip(active_variables, active_masks, strict=True)
    }
    return tuple(
        by_id.get(id(variable), tf.zeros_like(variable))
        for variable in transport_variables
    )


def _validate_cumulative_groups(
    transport_variables: tuple[tf.Variable, ...],
    groups: Mapping[str, Mapping[int, tf.Tensor]],
    phases: Sequence[NeuTraStageSpec],
) -> None:
    previous = {
        id(variable): tf.zeros_like(variable, tf.float64)
        for variable in transport_variables
    }
    for phase in phases:
        active_variables, active_masks = _active_variables_and_masks(
            transport_variables, groups, phase.active_groups
        )
        current = {id(variable): tf.cast(mask, tf.float64) for variable, mask in zip(
            active_variables, active_masks, strict=True
        )}
        for variable in transport_variables:
            key = id(variable)
            mask = current.get(key, tf.zeros_like(variable, tf.float64))
            if not bool(tf.reduce_all(mask >= previous[key]).numpy()):
                raise ValueError(
                    "carry_selected requires cumulative active variable masks"
                )
            previous[key] = mask


def _optimizer(
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    variables: Sequence[tf.Variable],
) -> tf.keras.optimizers.Adam:
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(learning_rate),
        beta_1=float(beta1),
        beta_2=float(beta2),
        epsilon=float(epsilon),
    )
    optimizer.build(tuple(variables))
    return optimizer


def train_neutra_five_stage(
    *,
    transport: Any,
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    variable_groups: Sequence[NeuTraVariableGroup],
    spec: NeuTraFiveStageSpec,
    latent_batch_fn: Callable[[str, int, int], tf.Tensor],
    selection_loss_fn: Callable[[Any], tf.Tensor],
    validation_fn: Callable[[Any], Mapping[str, Any]],
    gradient_clip_norm: float = 10.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-7,
    jit_compile: bool = True,
    optimizer_state_policy: str = "phase_reset",
) -> NeuTraFiveStageResult:
    """Tune and execute four continuation stages, then validate without updates.

    The controller is target-agnostic. Model code supplies parameter groups,
    latent batches, held-out selection loss, and the untouched validation gate.
    Every learning-rate candidate starts from the identical incoming state.
    """

    if not callable(target_log_prob_fn):
        raise ValueError("target_log_prob_fn must be callable")
    for callback, name in (
        (latent_batch_fn, "latent_batch_fn"),
        (selection_loss_fn, "selection_loss_fn"),
        (validation_fn, "validation_fn"),
    ):
        if not callable(callback):
            raise ValueError(f"{name} must be callable")
    if not math.isfinite(float(gradient_clip_norm)) or gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    if optimizer_state_policy not in {"phase_reset", "carry_selected"}:
        raise ValueError("unsupported optimizer_state_policy")
    transport_variables = tuple(transport.trainable_variables)
    if not transport_variables:
        raise ValueError("transport must expose trainable_variables")
    groups = _normalize_groups(transport_variables, variable_groups)
    joint_variables, joint_masks = _active_variables_and_masks(
        transport_variables, groups, spec.joint.active_groups
    )
    _validate_joint_coverage(transport_variables, joint_variables, joint_masks)
    if optimizer_state_policy == "carry_selected":
        _validate_cumulative_groups(
            transport_variables, groups, spec.optimizer_phases()
        )
        initial_optimizer = _optimizer(
            learning_rate=float(spec.affine.learning_rates[0]),
            beta1=float(beta1),
            beta2=float(beta2),
            epsilon=float(epsilon),
            variables=transport_variables,
        )
        carried_optimizer_state = _state(initial_optimizer.variables)
    else:
        carried_optimizer_state = ()

    stage_results = []
    selected_path_updates = 0
    tuning_optimizer_updates = 0
    for phase_index, phase in enumerate(spec.optimizer_phases()):
        active_variables, active_masks = _active_variables_and_masks(
            transport_variables, groups, phase.active_groups
        )
        incoming_state = _state(transport_variables)
        incoming_optimizer_state = carried_optimizer_state
        incoming_optimizer_iterations = (
            int(tf.convert_to_tensor(incoming_optimizer_state[0]).numpy())
            if incoming_optimizer_state
            else 0
        )
        incoming_loss = _finite_scalar(selection_loss_fn(transport), "selection loss")
        candidates = []
        for learning_rate_index, peak_learning_rate in enumerate(phase.learning_rates):
            _restore(transport_variables, incoming_state)
            optimizer_variables = (
                transport_variables
                if optimizer_state_policy == "carry_selected"
                else active_variables
            )
            optimizer = _optimizer(
                learning_rate=float(peak_learning_rate),
                beta1=float(beta1),
                beta2=float(beta2),
                epsilon=float(epsilon),
                variables=optimizer_variables,
            )
            if incoming_optimizer_state:
                _restore(optimizer.variables, incoming_optimizer_state)
                optimizer.learning_rate.assign(float(peak_learning_rate))

            @tf.function(jit_compile=bool(jit_compile), reduce_retracing=True)
            def train_step(latent: tf.Tensor) -> tuple[tf.Tensor, ...]:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(active_variables)
                    physical, logdet = transport.forward_and_logdet(latent)
                    target = tf.convert_to_tensor(
                        target_log_prob_fn(physical), tf.float64
                    )
                    loss = tf.reduce_mean(-target - logdet)
                gradients = tuple(tape.gradient(loss, active_variables))
                if any(gradient is None for gradient in gradients):
                    raise NeuTraStagedTrainingError("staged gradient is missing")
                masked = tuple(
                    tf.convert_to_tensor(gradient) * mask
                    for gradient, mask in zip(gradients, active_masks, strict=True)
                )
                gradient_norm = tf.linalg.global_norm(masked)
                clipped, _ = tf.clip_by_global_norm(
                    masked,
                    tf.constant(float(gradient_clip_norm), tf.float64),
                    use_norm=gradient_norm,
                )
                clipped_norm = tf.linalg.global_norm(clipped)
                finite = tf.reduce_all(
                    tf.stack(
                        (
                            tf.reduce_all(tf.math.is_finite(loss)),
                            tf.reduce_all(tf.math.is_finite(target)),
                            tf.reduce_all(tf.math.is_finite(logdet)),
                            tf.reduce_all(tf.math.is_finite(gradient_norm)),
                            tf.reduce_all(tf.math.is_finite(clipped_norm)),
                            *(
                                tf.reduce_all(tf.math.is_finite(value))
                                for value in clipped
                            ),
                        )
                    )
                )

                def update() -> tf.Tensor:
                    optimizer.apply_gradients(zip(clipped, active_variables))
                    return tf.cast(optimizer.iterations, tf.int64)

                step = tf.cond(finite, update, lambda: tf.cast(-1, tf.int64))
                return loss, gradient_norm, clipped_norm, finite, step

            selected_state = incoming_state
            selected_optimizer_state = _state(optimizer.variables)
            selected_update = 0
            selected_loss = incoming_loss
            clipped_updates = 0
            terminal_loss = incoming_loss
            last_gradient_norm = 0.0
            executed_updates = 0
            learning_rate_reductions = 0
            stop_reason = "update_cap"
            current_peak_learning_rate = float(peak_learning_rate)
            scheduler_best_loss = incoming_loss
            plateau_checkpoints = 0
            checkpoint_history = []
            for update in range(1, int(phase.updates) + 1):
                current_learning_rate = _scheduled_learning_rate(
                    current_peak_learning_rate,
                    phase.learning_rate_schedule,
                    update,
                    int(phase.updates),
                )
                optimizer.learning_rate.assign(current_learning_rate)
                latent = tf.convert_to_tensor(
                    latent_batch_fn(phase.name, update, learning_rate_index),
                    tf.float64,
                )
                if latent.shape.rank != 2 or int(latent.shape[0]) <= 1:
                    raise NeuTraStagedTrainingError(
                        "latent batches must be rank two with batch size above one"
                    )
                loss, gradient_norm, _clipped_norm, finite, _step = train_step(latent)
                if not bool(finite.numpy()):
                    raise NeuTraStagedTrainingError(
                        f"nonfinite update in phase {phase.name}"
                    )
                last_gradient_norm = float(gradient_norm.numpy())
                clipped_updates += int(last_gradient_norm > float(gradient_clip_norm))
                executed_updates = update
                if update % int(phase.checkpoint_every) == 0 or update == int(
                    phase.updates
                ):
                    terminal_loss = _finite_scalar(
                        selection_loss_fn(transport), "selection loss"
                    )
                    if terminal_loss < selected_loss:
                        selected_loss = terminal_loss
                        selected_update = update
                        selected_state = _state(transport_variables)
                        selected_optimizer_state = _state(optimizer.variables)
                    policy = phase.adaptive_policy
                    if policy is not None:
                        if terminal_loss < (
                            scheduler_best_loss - float(policy.minimum_improvement)
                        ):
                            scheduler_best_loss = terminal_loss
                            plateau_checkpoints = 0
                        else:
                            plateau_checkpoints += 1
                        if (
                            update >= int(policy.minimum_updates)
                            and plateau_checkpoints >= int(policy.patience_checkpoints)
                        ):
                            if learning_rate_reductions < int(
                                policy.maximum_learning_rate_reductions
                            ):
                                current_peak_learning_rate *= float(
                                    policy.learning_rate_reduction_factor
                                )
                                learning_rate_reductions += 1
                                plateau_checkpoints = 0
                            else:
                                stop_reason = "plateau_after_maximum_lr_reductions"
                    checkpoint_history.append(
                        (
                            int(update),
                            float(terminal_loss),
                            float(current_learning_rate),
                            int(learning_rate_reductions),
                        )
                    )
                    if stop_reason != "update_cap":
                        break
            candidates.append(
                NeuTraLearningRateResult(
                    learning_rate=float(peak_learning_rate),
                    selected_update=selected_update,
                    selected_loss=selected_loss,
                    terminal_loss=terminal_loss,
                    clipped_updates=clipped_updates,
                    gradient_norm=last_gradient_norm,
                    executed_updates=executed_updates,
                    learning_rate_reductions=learning_rate_reductions,
                    stop_reason=stop_reason,
                    checkpoint_history=tuple(checkpoint_history),
                    selected_state=selected_state,
                    terminal_state=_state(transport_variables),
                    selected_optimizer_state=selected_optimizer_state,
                    terminal_optimizer_state=_state(optimizer.variables),
                )
            )
            tuning_optimizer_updates += int(executed_updates)
        selected = min(
            candidates,
            key=lambda result: (result.selected_loss, result.learning_rate),
        )
        _restore(transport_variables, selected.selected_state)
        if optimizer_state_policy == "carry_selected":
            carried_optimizer_state = selected.selected_optimizer_state
        selected_path_updates += int(selected.selected_update)
        stage_results.append(
            NeuTraStageResult(
                name=phase.name,
                stage=int(phase.stage),
                active_groups=tuple(phase.active_groups),
                trainable_variables=tuple(variable.name for variable in active_variables),
                incoming_loss=incoming_loss,
                selected_learning_rate=float(selected.learning_rate),
                selected_update=int(selected.selected_update),
                selected_loss=float(selected.selected_loss),
                optimizer_state_policy=str(optimizer_state_policy),
                incoming_optimizer_iterations=int(incoming_optimizer_iterations),
                selected_optimizer_iterations=(
                    int(tf.convert_to_tensor(selected.selected_optimizer_state[0]).numpy())
                    if selected.selected_optimizer_state
                    else int(selected.selected_update)
                ),
                candidates=tuple(candidates),
            )
        )

    before_validation = _state(transport_variables)
    validation = validation_fn(transport)
    if not isinstance(validation, Mapping):
        raise NeuTraStagedTrainingError("validation_fn must return a mapping")
    after_validation = _state(transport_variables)
    for before, after in zip(before_validation, after_validation, strict=True):
        tf.debugging.assert_equal(before, after, "validation must not mutate transport")
    return NeuTraFiveStageResult(
        stages=tuple(stage_results),
        validation=dict(validation),
        selected_path_updates=selected_path_updates,
        tuning_optimizer_updates=tuning_optimizer_updates,
        final_state=_state(transport_variables),
        optimizer_state_policy=str(optimizer_state_policy),
        nonclaims=(
            "held-out loss selects tuning candidates but does not establish correctness",
            "stage-five validation performs no optimizer update",
            "model-specific validation is required for any scientific claim",
            "no HMC, posterior-correctness, or default-readiness claim",
        ),
    )


def dense_iaf_five_stage_variable_groups(transport: Any) -> tuple[NeuTraVariableGroup, ...]:
    """Partition a composed dense IAF into generic continuation groups."""

    stages = tuple(getattr(transport, "stages", ()))
    if not stages:
        raise ValueError("dense IAF transport must expose at least one stage")
    first = stages[0]
    dimension = int(getattr(first, "dimension", 0))
    if dimension <= 0 or first.unbounded_scale_linear_weight is None:
        raise ValueError("dense IAF requires an unbounded first-stage scale path")
    output_bias = first.biases[-1]
    if output_bias.shape != (2 * dimension,):
        raise ValueError("dense IAF output bias shape mismatch")
    location_mask = tf.concat(
        (tf.zeros((dimension,), tf.float64), tf.ones((dimension,), tf.float64)),
        axis=0,
    )
    scale_bias_mask = tf.concat(
        (tf.ones((dimension,), tf.float64), tf.zeros((dimension,), tf.float64)),
        axis=0,
    )
    groups = [
        NeuTraVariableGroup(
            "affine_location",
            (NeuTraVariablePart(output_bias, location_mask),),
        ),
        NeuTraVariableGroup(
            "simple_linear_scale",
            (NeuTraVariablePart(first.unbounded_scale_linear_weight),),
        ),
    ]
    first_residual_parts = [
        *(NeuTraVariablePart(variable) for variable in first.weights),
        *(NeuTraVariablePart(variable) for variable in first.biases[:-1]),
        NeuTraVariablePart(output_bias, scale_bias_mask),
    ]
    if first.scale_linear_skip_weight is not None:
        first_residual_parts.append(NeuTraVariablePart(first.scale_linear_skip_weight))
    groups.append(NeuTraVariableGroup("stage_0_residual", tuple(first_residual_parts)))
    for index, stage in enumerate(stages[1:], start=1):
        groups.append(
            NeuTraVariableGroup(
                f"stage_{index}",
                tuple(NeuTraVariablePart(variable) for variable in stage.trainable_variables),
            )
        )
    return tuple(groups)


def dense_iaf_five_stage_spec(
    *,
    stages: int,
    learning_rates: Sequence[float] = (2.0e-4, 5.0e-4, 1.0e-3),
    affine_updates: int = 250,
    simple_updates: int = 2000,
    progressive_updates: int = 500,
    joint_updates: int = 1000,
    checkpoint_every: int = 250,
    joint_adaptive_policy: NeuTraAdaptiveStagePolicy | None = None,
) -> NeuTraFiveStageSpec:
    """Return the generic dense-IAF continuation recipe for a stage count."""

    if int(stages) <= 0:
        raise ValueError("stages must be positive")
    rates = tuple(float(value) for value in learning_rates)
    affine = NeuTraStageSpec(
        "affine_location",
        1,
        ("affine_location",),
        int(affine_updates),
        rates,
        min(int(checkpoint_every), int(affine_updates)),
    )
    simple = NeuTraStageSpec(
        "simple_linear_scale",
        2,
        ("affine_location", "simple_linear_scale"),
        int(simple_updates),
        rates,
        min(int(checkpoint_every), int(simple_updates)),
    )
    active = ["affine_location", "simple_linear_scale"]
    progressive = []
    for index in range(int(stages)):
        active.append("stage_0_residual" if index == 0 else f"stage_{index}")
        progressive.append(
            NeuTraStageSpec(
                f"progressive_stage_{index}",
                3,
                tuple(active),
                int(progressive_updates),
                rates,
                min(int(checkpoint_every), int(progressive_updates)),
            )
        )
    joint = NeuTraStageSpec(
        "joint_fine_tune",
        4,
        tuple(active),
        int(joint_updates),
        rates,
        min(int(checkpoint_every), int(joint_updates)),
        "constant" if joint_adaptive_policy is not None else "piecewise_60_85",
        joint_adaptive_policy,
    )
    return NeuTraFiveStageSpec(
        affine=affine,
        simple=simple,
        progressive=tuple(progressive),
        joint=joint,
    )
