"""Deterministic plateau control for bounded NeuTra training runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


class NeuTraPlateauError(RuntimeError):
    """Raised when plateau state or observations violate the contract."""


@dataclass(frozen=True)
class NeuTraPlateauConfig:
    """Frozen validation and learning-rate repair policy."""

    validation_check_every: int = 100
    patience_steps: int = 500
    max_steps: int = 5000
    initial_learning_rate: float = 1.0e-3
    learning_rate_factor: float = 0.5
    post_repair_no_improvement_cycles: int = 1
    minimum_learning_rate_fraction: float = 1.0 / 16.0
    absolute_min_delta: float = 0.0
    one_sided_critical_value: float = 1.6694022215079607
    saturation_max: float = 0.05
    saturation_repair_enabled: bool = True
    roundtrip_max_abs: float = 1.0e-9
    moderate_shell_max_inverse_radius: float = 4.30
    inverse_radius_policy: str = "hard_veto"

    def __post_init__(self) -> None:
        if int(self.validation_check_every) <= 0:
            raise ValueError("validation_check_every must be positive")
        if int(self.patience_steps) <= 0:
            raise ValueError("patience_steps must be positive")
        if int(self.patience_steps) % int(self.validation_check_every) != 0:
            raise ValueError("patience_steps must be a validation interval multiple")
        if int(self.post_repair_no_improvement_cycles) <= 0:
            raise ValueError("post_repair_no_improvement_cycles must be positive")
        if int(self.max_steps) < self.plateau_stop_steps:
            raise ValueError("max_steps must allow the complete plateau repair window")
        for name in (
            "initial_learning_rate",
            "learning_rate_factor",
            "minimum_learning_rate_fraction",
            "one_sided_critical_value",
            "saturation_max",
            "roundtrip_max_abs",
            "moderate_shell_max_inverse_radius",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < float(self.learning_rate_factor) < 1.0:
            raise ValueError("learning_rate_factor must lie strictly between zero and one")
        if not 0.0 < float(self.minimum_learning_rate_fraction) <= 1.0:
            raise ValueError(
                "minimum_learning_rate_fraction must lie in (0, 1]"
            )
        if not math.isfinite(float(self.absolute_min_delta)) or float(
            self.absolute_min_delta
        ) < 0.0:
            raise ValueError("absolute_min_delta must be finite and nonnegative")
        if float(self.saturation_max) > 1.0:
            raise ValueError("saturation_max cannot exceed one")
        if self.inverse_radius_policy not in {"hard_veto", "explanatory_only"}:
            raise ValueError(
                "inverse_radius_policy must be 'hard_veto' or 'explanatory_only'"
            )

    @property
    def minimum_learning_rate(self) -> float:
        return float(self.initial_learning_rate) * float(
            self.minimum_learning_rate_fraction
        )

    @property
    def plateau_stop_steps(self) -> int:
        return int(self.patience_steps) * (
            1 + int(self.post_repair_no_improvement_cycles)
        )

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.plateau_config.v2",
            **asdict(self),
            "minimum_learning_rate": self.minimum_learning_rate,
        }


@dataclass(frozen=True)
class NeuTraPlateauAction:
    """One deterministic controller decision at a validation boundary."""

    kind: str
    step: int
    meaningful_improvement: bool
    checkpoint_eligible: bool
    checkpoint_eligibility_vetoes: tuple[str, ...]
    checkpoint_diagnostics: Mapping[str, Any]
    paired_mean_delta: float | None
    paired_one_sided_upper: float | None
    current_learning_rate: float
    best_step: int | None
    steps_since_best: int | None
    stop_reason: str | None
    repair_trigger: str | None

    @property
    def should_reduce_learning_rate(self) -> bool:
        return self.kind in {
            "reduce_learning_rate",
            "reduce_learning_rate_for_saturation",
            "improved_and_reduce_learning_rate_for_saturation",
        }

    @property
    def should_stop(self) -> bool:
        return self.stop_reason is not None

    def payload(self) -> Mapping[str, Any]:
        return {"schema": "bayesfilter.neutra.plateau_action.v2", **asdict(self)}


class NeuTraPlateauController:
    """Track paired validation improvements and one repair per plateau."""

    def __init__(self, config: NeuTraPlateauConfig) -> None:
        self.config = config
        self.current_learning_rate = float(config.initial_learning_rate)
        self.best_step: int | None = None
        self.best_per_sample_loss: tuple[float, ...] = ()
        self.best_trainer_state_hash: str | None = None
        self.best_checkpoint_diagnostics: Mapping[str, Any] | None = None
        self.last_checkpoint_diagnostics: Mapping[str, Any] | None = None
        self.steps_since_best: int | None = None
        self.learning_rate_reductions = 0
        self.last_reduction_step: int | None = None
        self.reduction_for_current_plateau = False
        self.minimum_learning_rate_reached = False
        self.last_observation_step: int | None = None
        self.status = "running"
        self.stop_reason: str | None = None

    def observe(
        self,
        *,
        step: int,
        per_sample_loss: Sequence[float],
        saturation_fraction: float,
        all_finite: bool = True,
        roundtrip_max_abs: float = 0.0,
        moderate_shell_max_inverse_radius: float = 0.0,
        trainer_state_hash: str,
    ) -> NeuTraPlateauAction:
        """Consume one immutable validation result and return the next action."""

        if self.status != "running":
            raise NeuTraPlateauError("cannot observe after controller termination")
        step = int(step)
        self._validate_step(step)
        losses = _finite_tuple(per_sample_loss, "per_sample_loss")
        if len(losses) < 2:
            raise ValueError("per_sample_loss must contain at least two values")
        saturation = float(saturation_fraction)
        if not math.isfinite(saturation) or not 0.0 <= saturation <= 1.0:
            raise ValueError("saturation_fraction must be finite and in [0, 1]")
        state_hash = str(trainer_state_hash)
        if len(state_hash) != 64 or any(
            character not in "0123456789abcdef" for character in state_hash
        ):
            raise ValueError("trainer_state_hash must be lowercase SHA-256 hex")
        self.last_observation_step = step

        diagnostics, eligibility_vetoes = self._checkpoint_eligibility(
            saturation_fraction=saturation,
            all_finite=all_finite,
            roundtrip_max_abs=roundtrip_max_abs,
            moderate_shell_max_inverse_radius=moderate_shell_max_inverse_radius,
        )
        self.last_checkpoint_diagnostics = diagnostics
        saturation_trigger = saturation > float(self.config.saturation_max)
        saturation_repair_trigger = (
            saturation_trigger and bool(self.config.saturation_repair_enabled)
        )
        # Saturation is a repair signal, not a mathematical-validity veto. A
        # finite/support-valid saturated row remains eligible for loss-based
        # best-state selection while the repair is recorded separately.
        eligible = not eligibility_vetoes
        repair_trigger = (
            "scale_saturation_above_cap" if saturation_repair_trigger else None
        )

        mean_delta = None
        upper = None
        improved = eligible and self.best_step is None
        if eligible and self.best_step is not None:
            if len(losses) != len(self.best_per_sample_loss):
                raise NeuTraPlateauError("validation batch size changed")
            mean_delta, upper = paired_one_sided_upper_bound(
                self.best_per_sample_loss,
                losses,
                critical_value=self.config.one_sided_critical_value,
            )
            improved = upper < -float(self.config.absolute_min_delta)

        if improved:
            self.best_step = step
            self.best_per_sample_loss = losses
            self.best_trainer_state_hash = state_hash
            self.best_checkpoint_diagnostics = diagnostics
            self.steps_since_best = 0
            saturation_repair = (
                saturation_repair_trigger and not self.reduction_for_current_plateau
            )
            if saturation_repair:
                proposed = self.current_learning_rate * float(
                    self.config.learning_rate_factor
                )
                self.reduction_for_current_plateau = True
                self.last_reduction_step = step
                if proposed < self.config.minimum_learning_rate:
                    self.minimum_learning_rate_reached = True
                    kind = "improved_saturation_repair_unavailable"
                else:
                    self.current_learning_rate = proposed
                    self.learning_rate_reductions += 1
                    kind = "improved_and_reduce_learning_rate_for_saturation"
            else:
                self.reduction_for_current_plateau = False
                kind = "initialize_best" if mean_delta is None else "improved"
            if step >= int(self.config.max_steps):
                self.status = "stopped"
                self.stop_reason = "maximum_steps_reached"
                kind = "stop"
            return self._action(
                kind=kind,
                step=step,
                meaningful_improvement=True,
                checkpoint_eligible=True,
                checkpoint_diagnostics=diagnostics,
                repair_trigger=repair_trigger,
                paired_mean_delta=mean_delta,
                paired_one_sided_upper=upper,
            )

        if saturation_repair_trigger and not self.reduction_for_current_plateau:
            proposed = self.current_learning_rate * float(
                self.config.learning_rate_factor
            )
            self.reduction_for_current_plateau = True
            self.last_reduction_step = step
            if proposed < self.config.minimum_learning_rate:
                self.minimum_learning_rate_reached = True
                kind = "saturation_repair_unavailable"
            else:
                self.current_learning_rate = proposed
                self.learning_rate_reductions += 1
                kind = "reduce_learning_rate_for_saturation"
            self.steps_since_best = (
                None if self.best_step is None else step - self.best_step
            )
            return self._action(
                kind=kind,
                step=step,
                checkpoint_eligible=False,
                checkpoint_eligibility_vetoes=eligibility_vetoes,
                checkpoint_diagnostics=diagnostics,
                repair_trigger=repair_trigger,
            )

        if self.best_step is None:
            if step >= int(self.config.max_steps):
                self.status = "stopped"
                self.stop_reason = "maximum_steps_reached_without_eligible_checkpoint"
                return self._action(
                    kind="stop",
                    step=step,
                    checkpoint_eligible=eligible,
                    checkpoint_eligibility_vetoes=eligibility_vetoes,
                    checkpoint_diagnostics=diagnostics,
                )
            return self._action(
                kind="checkpoint_ineligible",
                step=step,
                checkpoint_eligible=False,
                checkpoint_eligibility_vetoes=eligibility_vetoes,
                checkpoint_diagnostics=diagnostics,
            )
        self.steps_since_best = step - self.best_step
        if step >= int(self.config.max_steps):
            self.status = "stopped"
            self.stop_reason = "maximum_steps_reached"
            return self._action(
                kind="stop",
                step=step,
                checkpoint_eligible=eligible,
                checkpoint_eligibility_vetoes=eligibility_vetoes,
                checkpoint_diagnostics=diagnostics,
                paired_mean_delta=mean_delta,
                paired_one_sided_upper=upper,
            )
        if (
            self.reduction_for_current_plateau
            and self.steps_since_best >= self.config.plateau_stop_steps
        ):
            self.status = "stopped"
            self.stop_reason = "plateau_after_lr_repair"
            return self._action(
                kind="stop",
                step=step,
                checkpoint_eligible=eligible,
                checkpoint_eligibility_vetoes=eligibility_vetoes,
                checkpoint_diagnostics=diagnostics,
                paired_mean_delta=mean_delta,
                paired_one_sided_upper=upper,
            )

        if (
            not self.reduction_for_current_plateau
            and self.steps_since_best >= int(self.config.patience_steps)
        ):
            proposed = self.current_learning_rate * float(
                self.config.learning_rate_factor
            )
            self.reduction_for_current_plateau = True
            self.last_reduction_step = step
            if proposed < self.config.minimum_learning_rate:
                self.minimum_learning_rate_reached = True
                kind = "minimum_learning_rate_reached"
            else:
                self.current_learning_rate = proposed
                self.learning_rate_reductions += 1
                kind = "reduce_learning_rate"
            return self._action(
                kind=kind,
                step=step,
                checkpoint_eligible=eligible,
                checkpoint_eligibility_vetoes=eligibility_vetoes,
                checkpoint_diagnostics=diagnostics,
                paired_mean_delta=mean_delta,
                paired_one_sided_upper=upper,
            )

        return self._action(
            kind="continue",
            step=step,
            checkpoint_eligible=eligible,
            checkpoint_eligibility_vetoes=eligibility_vetoes,
            checkpoint_diagnostics=diagnostics,
            paired_mean_delta=mean_delta,
            paired_one_sided_upper=upper,
        )

    def state_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": "bayesfilter.neutra.plateau_state.v2",
            "config": self.config.manifest_payload(),
            "current_learning_rate": self.current_learning_rate,
            "best_step": self.best_step,
            "best_per_sample_loss": list(self.best_per_sample_loss),
            "best_trainer_state_hash": self.best_trainer_state_hash,
            "best_checkpoint_diagnostics": self.best_checkpoint_diagnostics,
            "last_checkpoint_diagnostics": self.last_checkpoint_diagnostics,
            "steps_since_best": self.steps_since_best,
            "learning_rate_reductions": self.learning_rate_reductions,
            "last_reduction_step": self.last_reduction_step,
            "reduction_for_current_plateau": self.reduction_for_current_plateau,
            "minimum_learning_rate_reached": self.minimum_learning_rate_reached,
            "last_observation_step": self.last_observation_step,
            "repair_trigger_policy": (
                "saturation_is_repair_trigger_not_veto"
                if self.config.saturation_repair_enabled
                else "loss_plateau_only_saturation_telemetry"
            ),
            "status": self.status,
            "stop_reason": self.stop_reason,
        }
        return {**payload, "state_hash": _stable_hash(payload)}

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        state = dict(payload)
        supplied_hash = str(state.pop("state_hash", ""))
        if supplied_hash != _stable_hash(state):
            raise NeuTraPlateauError("plateau state_hash mismatch")
        if state.get("schema") != "bayesfilter.neutra.plateau_state.v2":
            raise NeuTraPlateauError("unsupported plateau state schema")
        if not self._config_payload_matches(state.get("config")):
            raise NeuTraPlateauError("plateau config mismatch")

        restored = NeuTraPlateauController(self.config)
        restored.current_learning_rate = _finite_positive(
            state.get("current_learning_rate"), "current_learning_rate"
        )
        if restored.current_learning_rate > float(self.config.initial_learning_rate):
            raise NeuTraPlateauError("restored learning rate exceeds initial rate")
        restored.best_step = _optional_nonnegative_int(state.get("best_step"), "best_step")
        restored.best_per_sample_loss = (
            ()
            if not state.get("best_per_sample_loss")
            else _finite_tuple(state.get("best_per_sample_loss"), "best_per_sample_loss")
        )
        restored.best_trainer_state_hash = state.get("best_trainer_state_hash")
        if restored.best_trainer_state_hash is not None:
            restored.best_trainer_state_hash = str(restored.best_trainer_state_hash)
            if len(restored.best_trainer_state_hash) != 64:
                raise NeuTraPlateauError("best trainer state hash is invalid")
        restored.best_checkpoint_diagnostics = _optional_checkpoint_diagnostics(
            state.get("best_checkpoint_diagnostics"), "best_checkpoint_diagnostics"
        )
        restored.last_checkpoint_diagnostics = _optional_checkpoint_diagnostics(
            state.get("last_checkpoint_diagnostics"), "last_checkpoint_diagnostics"
        )
        restored.steps_since_best = _optional_nonnegative_int(
            state.get("steps_since_best"), "steps_since_best"
        )
        restored.learning_rate_reductions = _nonnegative_int(
            state.get("learning_rate_reductions"), "learning_rate_reductions"
        )
        restored.last_reduction_step = _optional_nonnegative_int(
            state.get("last_reduction_step"), "last_reduction_step"
        )
        restored.reduction_for_current_plateau = bool(
            state.get("reduction_for_current_plateau")
        )
        restored.minimum_learning_rate_reached = bool(
            state.get("minimum_learning_rate_reached")
        )
        restored.last_observation_step = _optional_nonnegative_int(
            state.get("last_observation_step"), "last_observation_step"
        )
        expected_policy = (
            "saturation_is_repair_trigger_not_veto"
            if self.config.saturation_repair_enabled
            else "loss_plateau_only_saturation_telemetry"
        )
        if state.get("repair_trigger_policy", expected_policy) != expected_policy:
            raise NeuTraPlateauError("unsupported repair-trigger policy")
        restored.status = str(state.get("status"))
        restored.stop_reason = state.get("stop_reason")
        if restored.status not in {"running", "stopped"}:
            raise NeuTraPlateauError("invalid plateau status")
        if (restored.status == "stopped") != (restored.stop_reason is not None):
            raise NeuTraPlateauError("plateau stop status/reason mismatch")
        restored._validate_consistency()
        self.__dict__.update(restored.__dict__)

    def _config_payload_matches(self, payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            return False
        supplied = dict(payload)
        if (
            "post_repair_no_improvement_cycles" not in supplied
            and int(self.config.post_repair_no_improvement_cycles) == 1
        ):
            supplied["post_repair_no_improvement_cycles"] = 1
        if (
            "saturation_repair_enabled" not in supplied
            and bool(self.config.saturation_repair_enabled)
        ):
            supplied["saturation_repair_enabled"] = True
        if (
            "inverse_radius_policy" not in supplied
            and self.config.inverse_radius_policy == "hard_veto"
        ):
            supplied["inverse_radius_policy"] = "hard_veto"
        return supplied == self.config.manifest_payload()

    def _validate_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("step must be nonnegative")
        if step != 0 and step % int(self.config.validation_check_every) != 0:
            raise ValueError("step is not a validation boundary")
        if self.last_observation_step is not None and step <= self.last_observation_step:
            raise NeuTraPlateauError("validation steps must increase strictly")

    def _validate_consistency(self) -> None:
        best_fields = (
            self.best_step,
            self.best_trainer_state_hash,
            self.steps_since_best,
        )
        if self.best_step is None:
            if any(value is not None for value in best_fields[1:]) or self.best_per_sample_loss:
                raise NeuTraPlateauError("partial best-checkpoint state")
        elif (
            not self.best_per_sample_loss
            or self.best_trainer_state_hash is None
            or self.steps_since_best is None
        ):
            raise NeuTraPlateauError("incomplete best-checkpoint state")
        if self.best_step is None and self.best_checkpoint_diagnostics is not None:
            raise NeuTraPlateauError("best diagnostics exist without a best checkpoint")
        if self.best_step is not None and self.best_checkpoint_diagnostics is None:
            raise NeuTraPlateauError("best checkpoint diagnostics are missing")
        if self.best_checkpoint_diagnostics is not None:
            diagnostic_policy = self.best_checkpoint_diagnostics.get(
                "inverse_radius_policy", "hard_veto"
            )
            if diagnostic_policy != self.config.inverse_radius_policy:
                raise NeuTraPlateauError(
                    "best checkpoint inverse-radius policy mismatch"
                )
            diagnostic_threshold = self.best_checkpoint_diagnostics.get(
                "inverse_radius_threshold",
                self.config.moderate_shell_max_inverse_radius,
            )
            if float(diagnostic_threshold) != float(
                self.config.moderate_shell_max_inverse_radius
            ):
                raise NeuTraPlateauError(
                    "best checkpoint inverse-radius threshold mismatch"
                )
            _, vetoes = self._checkpoint_eligibility(
                saturation_fraction=float(
                    self.best_checkpoint_diagnostics["saturation_fraction"]
                ),
                all_finite=bool(self.best_checkpoint_diagnostics["all_finite"]),
                roundtrip_max_abs=float(
                    self.best_checkpoint_diagnostics["roundtrip_max_abs"]
                ),
                moderate_shell_max_inverse_radius=float(
                    self.best_checkpoint_diagnostics[
                        "moderate_shell_max_inverse_radius"
                    ]
                ),
            )
            if vetoes:
                raise NeuTraPlateauError("best checkpoint is not support-admissible")
        if self.last_observation_step is None and self.last_checkpoint_diagnostics is not None:
            raise NeuTraPlateauError("last diagnostics exist without an observation")
        if self.last_observation_step is not None and self.last_checkpoint_diagnostics is None:
            raise NeuTraPlateauError("last checkpoint diagnostics are missing")
        if self.last_observation_step is not None and self.best_step is not None:
            if self.best_step > self.last_observation_step:
                raise NeuTraPlateauError("best step exceeds last observation")
            if self.steps_since_best != self.last_observation_step - self.best_step:
                raise NeuTraPlateauError("steps_since_best mismatch")
        if self.reduction_for_current_plateau and self.last_reduction_step is None:
            raise NeuTraPlateauError("plateau reduction step is missing")

    def _action(
        self,
        *,
        kind: str,
        step: int,
        meaningful_improvement: bool = False,
        checkpoint_eligible: bool = False,
        checkpoint_eligibility_vetoes: Sequence[str] = (),
        checkpoint_diagnostics: Mapping[str, Any] | None = None,
        paired_mean_delta: float | None = None,
        paired_one_sided_upper: float | None = None,
        repair_trigger: str | None = None,
    ) -> NeuTraPlateauAction:
        if (
            repair_trigger is None
            and self.config.saturation_repair_enabled
            and checkpoint_diagnostics is not None
            and float(checkpoint_diagnostics.get("saturation_fraction", 0.0))
            > float(self.config.saturation_max)
        ):
            repair_trigger = "scale_saturation_above_cap"
        return NeuTraPlateauAction(
            kind=kind,
            step=int(step),
            meaningful_improvement=bool(meaningful_improvement),
            checkpoint_eligible=bool(checkpoint_eligible),
            checkpoint_eligibility_vetoes=tuple(checkpoint_eligibility_vetoes),
            checkpoint_diagnostics=dict(checkpoint_diagnostics or {}),
            paired_mean_delta=paired_mean_delta,
            paired_one_sided_upper=paired_one_sided_upper,
            current_learning_rate=self.current_learning_rate,
            best_step=self.best_step,
            steps_since_best=self.steps_since_best,
            stop_reason=self.stop_reason,
            repair_trigger=repair_trigger,
        )

    def _checkpoint_eligibility(
        self,
        *,
        saturation_fraction: float,
        all_finite: bool,
        roundtrip_max_abs: float,
        moderate_shell_max_inverse_radius: float,
    ) -> tuple[Mapping[str, Any], tuple[str, ...]]:
        roundtrip = float(roundtrip_max_abs)
        radius = float(moderate_shell_max_inverse_radius)
        finite = bool(all_finite) and math.isfinite(roundtrip) and math.isfinite(radius)
        diagnostics = {
            "all_finite": finite,
            "saturation_fraction": float(saturation_fraction),
            "roundtrip_max_abs": roundtrip,
            "moderate_shell_max_inverse_radius": radius,
            "inverse_radius_policy": self.config.inverse_radius_policy,
            "inverse_radius_threshold": float(
                self.config.moderate_shell_max_inverse_radius
            ),
            "inverse_radius_threshold_exceeded": bool(
                math.isfinite(radius)
                and radius > float(self.config.moderate_shell_max_inverse_radius)
            ),
        }
        vetoes = []
        if not finite:
            vetoes.append("checkpoint_nonfinite")
        if not math.isfinite(roundtrip) or roundtrip > float(self.config.roundtrip_max_abs):
            vetoes.append("roundtrip_residual_above_threshold")
        if self.config.inverse_radius_policy == "hard_veto" and (
            not math.isfinite(radius)
            or radius > float(self.config.moderate_shell_max_inverse_radius)
        ):
            vetoes.append("moderate_shell_missing_support")
        return diagnostics, tuple(vetoes)


def paired_one_sided_upper_bound(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    critical_value: float,
) -> tuple[float, float]:
    """Return mean(candidate-baseline) and its paired one-sided upper bound."""

    left = _finite_tuple(baseline, "baseline")
    right = _finite_tuple(candidate, "candidate")
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired samples must have equal length of at least two")
    critical = _finite_positive(critical_value, "critical_value")
    differences = tuple(b - a for a, b in zip(left, right))
    mean = math.fsum(differences) / len(differences)
    variance = math.fsum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    standard_error = math.sqrt(max(variance, 0.0) / len(differences))
    return mean, mean + critical * standard_error


def joint_training_checkpoint_payload(
    *,
    trainer_state: Mapping[str, Any],
    controller_state: Mapping[str, Any],
    best_trainer_state: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Bind trainer, controller, and best state into one replayable payload."""

    current = dict(trainer_state)
    controller = dict(controller_state)
    best = None if best_trainer_state is None else dict(best_trainer_state)
    expected_best_hash = controller.get("best_trainer_state_hash")
    actual_best_hash = None if best is None else best.get("state_hash")
    if expected_best_hash != actual_best_hash:
        raise NeuTraPlateauError("best trainer/controller state hash mismatch")
    if current.get("config") != (best or current).get("config"):
        raise NeuTraPlateauError("current and best trainer configs differ")
    payload = {
        "schema": "bayesfilter.neutra.joint_training_checkpoint.v1",
        "trainer_state": current,
        "controller_state": controller,
        "best_trainer_state": best,
    }
    return {**payload, "checkpoint_hash": _stable_hash(payload)}


def validate_joint_training_checkpoint(payload: Mapping[str, Any]) -> None:
    """Validate a joint checkpoint without mutating a trainer or controller."""

    state = dict(payload)
    supplied_hash = str(state.pop("checkpoint_hash", ""))
    if supplied_hash != _stable_hash(state):
        raise NeuTraPlateauError("joint checkpoint hash mismatch")
    if state.get("schema") != "bayesfilter.neutra.joint_training_checkpoint.v1":
        raise NeuTraPlateauError("unsupported joint checkpoint schema")
    joint_training_checkpoint_payload(
        trainer_state=state.get("trainer_state", {}),
        controller_state=state.get("controller_state", {}),
        best_trainer_state=state.get("best_trainer_state"),
    )


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _finite_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise NeuTraPlateauError(f"{name} must be finite and positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise NeuTraPlateauError(f"{name} must be nonnegative")
    return result


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    return None if value is None else _nonnegative_int(value, name)


def _optional_checkpoint_diagnostics(
    value: Any, name: str
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise NeuTraPlateauError(f"{name} must be a mapping")
    legacy_fields = {
        "all_finite",
        "saturation_fraction",
        "roundtrip_max_abs",
        "moderate_shell_max_inverse_radius",
    }
    policy_fields = {
        "inverse_radius_policy",
        "inverse_radius_threshold",
        "inverse_radius_threshold_exceeded",
    }
    actual_fields = frozenset(value)
    if actual_fields not in {
        frozenset(legacy_fields),
        frozenset(legacy_fields | policy_fields),
    }:
        raise NeuTraPlateauError(f"{name} fields are invalid")
    result = {
        "all_finite": bool(value["all_finite"]),
        "saturation_fraction": float(value["saturation_fraction"]),
        "roundtrip_max_abs": float(value["roundtrip_max_abs"]),
        "moderate_shell_max_inverse_radius": float(
            value["moderate_shell_max_inverse_radius"]
        ),
    }
    if policy_fields <= set(value):
        policy = str(value["inverse_radius_policy"])
        if policy not in {"hard_veto", "explanatory_only"}:
            raise NeuTraPlateauError(f"{name} inverse-radius policy is invalid")
        threshold = float(value["inverse_radius_threshold"])
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise NeuTraPlateauError(f"{name} inverse-radius threshold is invalid")
        expected_exceeded = bool(
            math.isfinite(result["moderate_shell_max_inverse_radius"])
            and result["moderate_shell_max_inverse_radius"] > threshold
        )
        if bool(value["inverse_radius_threshold_exceeded"]) != expected_exceeded:
            raise NeuTraPlateauError(f"{name} inverse-radius exceedance is inconsistent")
        result.update(
            {
                "inverse_radius_policy": policy,
                "inverse_radius_threshold": threshold,
                "inverse_radius_threshold_exceeded": expected_exceeded,
            }
        )
    saturation = result["saturation_fraction"]
    if not math.isfinite(saturation) or not 0.0 <= saturation <= 1.0:
        raise NeuTraPlateauError(f"{name} saturation is invalid")
    return result
