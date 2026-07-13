"""Role-separated diagnostics for the Phase 30 serious HMC trial.

This module evaluates preserved chain-major transitions with the operational
criteria declared by the Phase 30 plan.  It keeps four decisions independent:
engineering validity, warm-up qualification, posterior sampler health, and
posterior Monte Carlo precision.  Exact ``Delta H`` tails are retained as
explanatory diagnostics; no finite energy magnitude can veto continuation,
promotion, or a precision-only extension.

The statistical calculations reuse the rank-normalized R-hat, ESS, MCSE,
E-BFMI, initialization-memory, and epoch-drift implementations from
``hmc_posterior_diagnostics``.  The legacy Phase 29 evaluators and their
absolute-energy threshold are intentionally not called here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bayesfilter.inference.hmc_posterior_diagnostics import (
    compute_coordinate_diagnostics,
    epoch_drift_statistics,
    per_chain_ebfmi,
)
from bayesfilter.inference.hmc_transition_archive import (
    summarize_hmc_exact_mechanics_identity,
)


REQUIRED_COORDINATE_SYSTEMS = frozenset(("final_latent", "named_model"))
ENERGY_TAIL_CUTOFFS = (10.0, 100.0, 1000.0, 1.0e6)


def _finite_positive(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _probability(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True)
class Phase30WarmupThresholds:
    """Prospectively fixed qualification thresholds for each 512-draw epoch."""

    rhat_max: float = 1.05
    bulk_ess_min: float = 100.0
    tail_ess_min: float = 100.0
    mcse_sd_ratio_max: float = 0.10
    acceptance_min: float = 0.20
    acceptance_max: float = 0.95
    ebfmi_min: float = 0.30
    initialization_memory_max: float = 3.70
    epoch_drift_z_max: float = 3.50
    epoch_sd_ratio_min: float = 0.80
    epoch_sd_ratio_max: float = 1.25

    def __post_init__(self) -> None:
        for name in (
            "rhat_max",
            "bulk_ess_min",
            "tail_ess_min",
            "mcse_sd_ratio_max",
            "ebfmi_min",
            "initialization_memory_max",
            "epoch_drift_z_max",
            "epoch_sd_ratio_min",
            "epoch_sd_ratio_max",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
        for name in ("acceptance_min", "acceptance_max"):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))
        if self.rhat_max <= 1.0:
            raise ValueError("rhat_max must be greater than one")
        if self.acceptance_min >= self.acceptance_max:
            raise ValueError("acceptance bounds are reversed")
        if self.epoch_sd_ratio_min >= self.epoch_sd_ratio_max:
            raise ValueError("epoch SD ratio bounds are reversed")

    def payload(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Phase30PosteriorThresholds:
    """Final sampler-health and precision thresholds for posterior checkpoints."""

    rhat_max_exclusive: float = 1.01
    bulk_ess_min_exclusive: float = 400.0
    tail_ess_min_exclusive: float = 400.0
    mcse_sd_ratio_max: float = 0.05
    acceptance_min: float = 0.20
    acceptance_max: float = 0.95
    ebfmi_min: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "rhat_max_exclusive",
            "bulk_ess_min_exclusive",
            "tail_ess_min_exclusive",
            "mcse_sd_ratio_max",
            "ebfmi_min",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
        for name in ("acceptance_min", "acceptance_max"):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))
        if self.rhat_max_exclusive <= 1.0:
            raise ValueError("rhat_max_exclusive must be greater than one")
        if self.acceptance_min >= self.acceptance_max:
            raise ValueError("acceptance bounds are reversed")

    def payload(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def _to_python(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_python(item) for item in value]
    if hasattr(value, "numpy"):
        return _to_python(value.numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _chain_major_tensor(value: Any, *, name: str, rank: int) -> Any:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value)
    if tensor.shape.rank != rank or any(dim is None for dim in tensor.shape):
        raise ValueError(f"{name} must have fully static rank {rank}")
    return tensor


def _prepare_inputs(
    coordinate_samples: Mapping[str, Any],
    *,
    pre_state: Any,
    accepted_target_log_prob: Any,
    is_accepted: Any,
    log_accept_ratio: Any,
    initial_energy: Any,
    delta_h: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    import tensorflow as tf

    if set(coordinate_samples) != REQUIRED_COORDINATE_SYSTEMS:
        raise ValueError(
            "coordinate_samples must contain exactly final_latent and named_model"
        )
    coordinates = {
        str(name): tf.cast(
            _chain_major_tensor(samples, name=f"{name} samples", rank=3),
            tf.float64,
        )
        for name, samples in coordinate_samples.items()
    }
    reference_shape = tuple(coordinates["final_latent"].shape)
    if any(tuple(tensor.shape) != reference_shape for tensor in coordinates.values()):
        raise ValueError("coordinate systems must have identical sample shapes")
    chains, draws, parameters = (int(item) for item in reference_shape)
    if chains < 2 or draws < 4 or draws % 2 or parameters < 1:
        raise ValueError("coordinate samples require [chain>=2, even draw>=4, parameter>=1]")

    mechanics = {
        "pre_state": tf.cast(
            _chain_major_tensor(pre_state, name="pre_state", rank=3), tf.float64
        ),
        "accepted_target_log_prob": tf.cast(
            _chain_major_tensor(
                accepted_target_log_prob,
                name="accepted_target_log_prob",
                rank=2,
            ),
            tf.float64,
        ),
        "is_accepted": tf.cast(
            _chain_major_tensor(is_accepted, name="is_accepted", rank=2), tf.bool
        ),
        "log_accept_ratio": tf.cast(
            _chain_major_tensor(log_accept_ratio, name="log_accept_ratio", rank=2),
            tf.float64,
        ),
        "initial_energy": tf.cast(
            _chain_major_tensor(initial_energy, name="initial_energy", rank=2),
            tf.float64,
        ),
        "delta_h": tf.cast(
            _chain_major_tensor(delta_h, name="delta_h", rank=2), tf.float64
        ),
    }
    if tuple(mechanics["pre_state"].shape) != reference_shape:
        raise ValueError("pre_state must match the coordinate sample shape")
    for name in (
        "accepted_target_log_prob",
        "is_accepted",
        "log_accept_ratio",
        "initial_energy",
        "delta_h",
    ):
        if tuple(mechanics[name].shape) != (chains, draws):
            raise ValueError(f"{name} must match [chain, draw]")

    engineering: list[str] = []
    for name, tensor in coordinates.items():
        if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
            engineering.append(f"{name}_accepted_samples_nonfinite")
    for name in (
        "pre_state",
        "accepted_target_log_prob",
        "log_accept_ratio",
        "initial_energy",
        "delta_h",
    ):
        if not bool(tf.reduce_all(tf.math.is_finite(mechanics[name])).numpy()):
            engineering.append(f"{name}_nonfinite")

    identity = summarize_hmc_exact_mechanics_identity(
        np.asarray(mechanics["delta_h"].numpy(), dtype=np.float64),
        -np.asarray(mechanics["log_accept_ratio"].numpy(), dtype=np.float64),
        identity_name="delta_h_equals_negative_log_accept_ratio",
    )
    if identity["passed"] is not True:
        engineering.append("hamiltonian_log_accept_identity_failure")
    mechanics["hamiltonian_identity"] = identity
    return coordinates, mechanics, list(dict.fromkeys(engineering))


def _energy_diagnostics(delta_h: Any) -> dict[str, Any]:
    import tensorflow as tf
    import tensorflow_probability as tfp

    values = tf.cast(tf.convert_to_tensor(delta_h), tf.float64)
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
        return {
            "schema": "bayesfilter.phase30_energy_explanatory_diagnostics.v1",
            "status": "unavailable_nonfinite_delta_h_engineering_veto",
            "evidence_role": "explanatory_only",
            "finite_energy_magnitude_control_effect": "none",
            "finite_energy_magnitude_can_veto": False,
            "element_count": int(tf.size(values).numpy()),
        }
    absolute = tf.abs(values)
    flat = tf.reshape(absolute, (-1,))
    worst_flat = int(tf.argmax(flat, output_type=tf.int64).numpy())
    worst = tuple(
        int(item)
        for item in np.unravel_index(worst_flat, tuple(int(dim) for dim in values.shape))
    )
    counts = {
        f"abs_delta_h_gt_{cutoff:g}": int(
            tf.reduce_sum(tf.cast(absolute > cutoff, tf.int64)).numpy()
        )
        for cutoff in ENERGY_TAIL_CUTOFFS
    }
    per_chain_counts = {
        f"abs_delta_h_gt_{cutoff:g}": _to_python(
            tf.reduce_sum(tf.cast(absolute > cutoff, tf.int64), axis=1)
        )
        for cutoff in ENERGY_TAIL_CUTOFFS
    }
    quantiles = tfp.stats.percentile(
        flat,
        (50.0, 90.0, 95.0, 99.0),
        interpolation="linear",
    )
    return {
        "schema": "bayesfilter.phase30_energy_explanatory_diagnostics.v1",
        "evidence_role": "explanatory_only",
        "finite_energy_magnitude_control_effect": "none",
        "finite_energy_magnitude_can_veto": False,
        "element_count": int(tf.size(values).numpy()),
        "minimum_signed_delta_h": float(tf.reduce_min(values).numpy()),
        "maximum_signed_delta_h": float(tf.reduce_max(values).numpy()),
        "maximum_abs_delta_h": float(tf.reduce_max(absolute).numpy()),
        "maximum_abs_delta_h_index_chain_draw": worst,
        "abs_delta_h_quantiles": {
            name: float(value)
            for name, value in zip(("q50", "q90", "q95", "q99"), quantiles.numpy())
        },
        "global_tail_counts": counts,
        "per_chain_tail_counts": per_chain_counts,
    }


def _movement(final_latent: Any, pre_state: Any) -> Any:
    import tensorflow as tf

    return tf.reduce_any(tf.not_equal(final_latent, pre_state), axis=(1, 2))


def _acceptance_and_ebfmi(
    mechanics: Mapping[str, Any],
) -> tuple[Any, Any]:
    import tensorflow as tf

    acceptance = tf.reduce_mean(tf.cast(mechanics["is_accepted"], tf.float64), axis=1)
    return acceptance, per_chain_ebfmi(mechanics["initial_energy"])


def _coordinate_bundle(samples: Any) -> Mapping[str, Any]:
    return compute_coordinate_diagnostics(samples)


def _all_finite(value: Any) -> bool:
    import tensorflow as tf

    return bool(
        tf.reduce_all(tf.math.is_finite(tf.cast(tf.convert_to_tensor(value), tf.float64))).numpy()
    )


def evaluate_phase30_warmup_epoch(
    coordinate_samples: Mapping[str, Any],
    *,
    pre_state: Any,
    accepted_target_log_prob: Any,
    is_accepted: Any,
    log_accept_ratio: Any,
    initial_energy: Any,
    delta_h: Any,
    previous_coordinate_samples: Mapping[str, Any] | None = None,
    thresholds: Phase30WarmupThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate one non-overlapping Phase 30 warm-up diagnostic epoch."""

    import tensorflow as tf

    active = Phase30WarmupThresholds() if thresholds is None else thresholds
    coordinates, mechanics, engineering = _prepare_inputs(
        coordinate_samples,
        pre_state=pre_state,
        accepted_target_log_prob=accepted_target_log_prob,
        is_accepted=is_accepted,
        log_accept_ratio=log_accept_ratio,
        initial_energy=initial_energy,
        delta_h=delta_h,
    )
    previous: dict[str, Any] | None = None
    if previous_coordinate_samples is not None:
        if set(previous_coordinate_samples) != REQUIRED_COORDINATE_SYSTEMS:
            raise ValueError("previous coordinate systems do not match")
        previous = {
            str(name): tf.cast(
                _chain_major_tensor(value, name=f"previous {name}", rank=3),
                tf.float64,
            )
            for name, value in previous_coordinate_samples.items()
        }
        for name in REQUIRED_COORDINATE_SYSTEMS:
            if tuple(previous[name].shape) != tuple(coordinates[name].shape):
                raise ValueError("previous epoch shape does not match current epoch")

    promotion: list[str] = []
    coordinate_reports: dict[str, Any] = {}
    if not engineering:
        for name, samples in coordinates.items():
            report = dict(_coordinate_bundle(samples))
            rhat = report["rank_normalized_split_rhat"]["maximum"]
            ess = report["rank_normalized_ess"]
            ratio = report["mean"]["mcse_sd_ratio"]
            memory = report["initialization_memory"]["max_abs_standardized_difference"]
            if not _all_finite(rhat) or not bool(tf.reduce_all(rhat <= active.rhat_max).numpy()):
                promotion.append(f"{name}:rank_normalized_split_rhat_above_or_nonfinite")
            if not _all_finite(ess["bulk"]) or not bool(
                tf.reduce_all(ess["bulk"] >= active.bulk_ess_min).numpy()
            ):
                promotion.append(f"{name}:bulk_ess_below_or_nonfinite")
            if not _all_finite(ess["tail"]) or not bool(
                tf.reduce_all(ess["tail"] >= active.tail_ess_min).numpy()
            ):
                promotion.append(f"{name}:tail_ess_below_or_nonfinite")
            if not _all_finite(ratio) or not bool(
                tf.reduce_all(ratio <= active.mcse_sd_ratio_max).numpy()
            ):
                promotion.append(f"{name}:mcse_sd_ratio_above_or_nonfinite")
            if not _all_finite(memory) or not bool(
                tf.reduce_all(memory <= active.initialization_memory_max).numpy()
            ):
                promotion.append(f"{name}:initialization_memory_above_or_nonfinite")
            if previous is None:
                report["epoch_drift"] = {"status": "not_applicable_first_epoch"}
            else:
                drift = epoch_drift_statistics(samples, previous[name])
                drift_z = drift["abs_standardized_mean_difference"]
                sd_ratio = drift["sd_ratio_current_over_previous"]
                report["epoch_drift"] = drift
                if not _all_finite(drift_z) or not _all_finite(sd_ratio):
                    promotion.append(f"{name}:epoch_drift_nonfinite")
                else:
                    if not bool(tf.reduce_all(drift_z <= active.epoch_drift_z_max).numpy()):
                        promotion.append(f"{name}:epoch_mean_drift_above_threshold")
                    if not bool(
                        tf.reduce_all(
                            (sd_ratio >= active.epoch_sd_ratio_min)
                            & (sd_ratio <= active.epoch_sd_ratio_max)
                        ).numpy()
                    ):
                        promotion.append(f"{name}:epoch_sd_ratio_outside_threshold")
            coordinate_reports[name] = _to_python(report)

        acceptance, ebfmi = _acceptance_and_ebfmi(mechanics)
        moved = _movement(coordinates["final_latent"], mechanics["pre_state"])
        if not _all_finite(acceptance) or not bool(
            tf.reduce_all(
                (acceptance >= active.acceptance_min)
                & (acceptance <= active.acceptance_max)
            ).numpy()
        ):
            promotion.append("per_chain_acceptance_outside_or_nonfinite")
        if not _all_finite(ebfmi) or not bool(tf.reduce_all(ebfmi > active.ebfmi_min).numpy()):
            promotion.append("per_chain_ebfmi_below_or_nonfinite")
        if not bool(tf.reduce_all(moved).numpy()):
            promotion.append("unmoved_chain")
    else:
        acceptance = ebfmi = moved = None

    engineering = list(dict.fromkeys(engineering))
    promotion = list(dict.fromkeys(promotion))
    return {
        "artifact_schema": "bayesfilter.phase30_warmup_diagnostics.v1",
        "role": "warmup_diagnostic",
        "engineering_valid": not engineering,
        "engineering_continuation_vetoes": engineering,
        "warmup_qualified": not engineering and not promotion,
        "candidate_promotion_vetoes": promotion,
        "passed": not engineering and not promotion,
        "thresholds": active.payload(),
        "coordinate_diagnostics": coordinate_reports,
        "mechanics_diagnostics": {
            "acceptance_rate_by_chain": _to_python(acceptance),
            "ebfmi_by_chain": _to_python(ebfmi),
            "chain_moved": _to_python(moved),
            "hamiltonian_identity": mechanics["hamiltonian_identity"],
        },
        "energy_diagnostics": _energy_diagnostics(mechanics["delta_h"]),
        "diagnostic_roles": {
            "engineering_continuation_vetoes": "continuation_veto",
            "candidate_promotion_vetoes": "warmup_qualification_veto",
            "energy_diagnostics": "explanatory_only_no_control_effect",
        },
        "nonclaims": (
            "finite-sample operational warm-up screen only",
            "no stationarity or convergence proof",
            "no sampler ranking, identification, or scientific-validity claim",
        ),
    }


def classify_phase30_posterior_checkpoint(
    *,
    engineering_continuation_vetoes: Sequence[str],
    posterior_health_vetoes: Sequence[str],
    posterior_precision_vetoes: Sequence[str],
    draws: int,
    maximum_draws: int,
) -> Mapping[str, Any]:
    """Apply the prospective precision-only extension state machine."""

    draw_count = int(draws)
    cap = int(maximum_draws)
    if draw_count <= 0 or cap <= 0 or draw_count > cap:
        raise ValueError("posterior draw count must be positive and no larger than the cap")
    engineering = tuple(dict.fromkeys(str(item) for item in engineering_continuation_vetoes))
    health = tuple(dict.fromkeys(str(item) for item in posterior_health_vetoes))
    precision = tuple(dict.fromkeys(str(item) for item in posterior_precision_vetoes))
    if engineering:
        decision = "engineering_continuation_veto"
    elif health:
        decision = "posterior_health_nonpromotion"
    elif not precision:
        decision = "posterior_checkpoint_pass"
    elif draw_count < cap:
        decision = "extend_precision_only"
    else:
        decision = "posterior_precision_nonpromotion_at_cap"
    return {
        "decision": decision,
        "passed": decision == "posterior_checkpoint_pass",
        "precision_extension_eligible": decision == "extend_precision_only",
        "at_maximum_draws": draw_count == cap,
        "draws": draw_count,
        "maximum_draws": cap,
    }


def evaluate_phase30_posterior_checkpoint(
    coordinate_samples: Mapping[str, Any],
    *,
    pre_state: Any,
    accepted_target_log_prob: Any,
    is_accepted: Any,
    log_accept_ratio: Any,
    initial_energy: Any,
    delta_h: Any,
    maximum_draws: int = 4096,
    thresholds: Phase30PosteriorThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate one cumulative Phase 30 held-out posterior checkpoint."""

    import tensorflow as tf

    active = Phase30PosteriorThresholds() if thresholds is None else thresholds
    coordinates, mechanics, engineering = _prepare_inputs(
        coordinate_samples,
        pre_state=pre_state,
        accepted_target_log_prob=accepted_target_log_prob,
        is_accepted=is_accepted,
        log_accept_ratio=log_accept_ratio,
        initial_energy=initial_energy,
        delta_h=delta_h,
    )
    draws = int(coordinates["final_latent"].shape[1])
    if draws > int(maximum_draws):
        raise ValueError("posterior checkpoint exceeds maximum_draws")

    health: list[str] = []
    precision: list[str] = []
    coordinate_reports: dict[str, Any] = {}
    if not engineering:
        for name, samples in coordinates.items():
            report = dict(_coordinate_bundle(samples))
            rhat = report["rank_normalized_split_rhat"]["maximum"]
            ess = report["rank_normalized_ess"]
            mean = report["mean"]
            posterior_sd = mean["posterior_sd"]
            ratio = mean["mcse_sd_ratio"]
            if not _all_finite(rhat) or not bool(
                tf.reduce_all(rhat < active.rhat_max_exclusive).numpy()
            ):
                health.append(f"{name}:rank_normalized_split_rhat_not_below_threshold")
            if not _all_finite(posterior_sd) or not bool(tf.reduce_all(posterior_sd > 0.0).numpy()):
                health.append(f"{name}:posterior_sd_nonpositive_or_nonfinite")
            if not _all_finite(ess["bulk"]):
                health.append(f"{name}:bulk_ess_nonfinite")
            elif not bool(tf.reduce_all(ess["bulk"] > active.bulk_ess_min_exclusive).numpy()):
                precision.append(f"{name}:bulk_ess_not_above_threshold")
            if not _all_finite(ess["tail"]):
                health.append(f"{name}:tail_ess_nonfinite")
            elif not bool(tf.reduce_all(ess["tail"] > active.tail_ess_min_exclusive).numpy()):
                precision.append(f"{name}:tail_ess_not_above_threshold")
            if not _all_finite(ratio):
                health.append(f"{name}:mcse_sd_ratio_nonfinite")
            elif not bool(tf.reduce_all(ratio <= active.mcse_sd_ratio_max).numpy()):
                precision.append(f"{name}:mcse_sd_ratio_above_threshold")
            report["initialization_memory_role"] = "explanatory_only_for_posterior"
            coordinate_reports[name] = _to_python(report)

        acceptance, ebfmi = _acceptance_and_ebfmi(mechanics)
        moved = _movement(coordinates["final_latent"], mechanics["pre_state"])
        if not _all_finite(acceptance) or not bool(
            tf.reduce_all(
                (acceptance >= active.acceptance_min)
                & (acceptance <= active.acceptance_max)
            ).numpy()
        ):
            health.append("per_chain_acceptance_outside_or_nonfinite")
        if not _all_finite(ebfmi) or not bool(tf.reduce_all(ebfmi > active.ebfmi_min).numpy()):
            health.append("per_chain_ebfmi_below_or_nonfinite")
        if not bool(tf.reduce_all(moved).numpy()):
            health.append("unmoved_chain")
    else:
        acceptance = ebfmi = moved = None

    engineering = list(dict.fromkeys(engineering))
    health = list(dict.fromkeys(health))
    precision = list(dict.fromkeys(precision))
    classification = classify_phase30_posterior_checkpoint(
        engineering_continuation_vetoes=engineering,
        posterior_health_vetoes=health,
        posterior_precision_vetoes=precision,
        draws=draws,
        maximum_draws=int(maximum_draws),
    )
    return {
        "artifact_schema": "bayesfilter.phase30_posterior_diagnostics.v1",
        "role": "posterior",
        **classification,
        "engineering_valid": not engineering,
        "engineering_continuation_vetoes": engineering,
        "posterior_health_passed": not engineering and not health,
        "posterior_health_vetoes": health,
        "posterior_precision_passed": not engineering and not health and not precision,
        "posterior_precision_vetoes": precision,
        "thresholds": active.payload(),
        "coordinate_diagnostics": coordinate_reports,
        "mechanics_diagnostics": {
            "acceptance_rate_by_chain": _to_python(acceptance),
            "ebfmi_by_chain": _to_python(ebfmi),
            "chain_moved": _to_python(moved),
            "hamiltonian_identity": mechanics["hamiltonian_identity"],
        },
        "energy_diagnostics": _energy_diagnostics(mechanics["delta_h"]),
        "diagnostic_roles": {
            "engineering_continuation_vetoes": "continuation_veto",
            "posterior_health_vetoes": "promotion_and_extension_veto",
            "posterior_precision_vetoes": "precision_extension_trigger_below_cap",
            "energy_diagnostics": "explanatory_only_no_control_effect",
        },
        "nonclaims": (
            "finite-sample operational posterior screen only",
            "no stationarity, global-mode, or convergence proof",
            "no sampler ranking, identification, or scientific-validity claim",
        ),
    }


__all__ = [
    "ENERGY_TAIL_CUTOFFS",
    "Phase30PosteriorThresholds",
    "Phase30WarmupThresholds",
    "classify_phase30_posterior_checkpoint",
    "evaluate_phase30_posterior_checkpoint",
    "evaluate_phase30_warmup_epoch",
]
