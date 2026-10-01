"""Public validation targets backed by BayesFilter state-space adapters.

The ordinary validation target used to contain a dense Gaussian fixture for
``lgssm_location``.  The profiles in this module deliberately cross the public
target boundary and call the admitted QR Kalman or deterministic sigma-point
adapter.  They are validation fixtures, not production model definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
    make_lgssm_generic_target_fixture,
)
from bayesfilter.testing.simple_nonlinear_generic_target_adapter_tf import (
    SIMPLE_NONLINEAR_DEFAULT_FILTER_ID,
    make_simple_nonlinear_generic_target_fixture,
)


@dataclass(frozen=True)
class SSMProfile:
    target_id: str
    family: str
    horizon: int
    persistence_cap: float | None
    approximate: bool
    parameter_names: tuple[str, ...]
    default_observations: tuple[float, ...]
    filter_id: str | None = None


def _series(horizon: int, *, phase: float, scale: float) -> tuple[float, ...]:
    """A deterministic, predeclared observation panel for a profile."""

    return tuple(
        float(0.12 + scale * math.sin(0.37 * index + phase)
              + 0.015 * math.cos(0.11 * index - phase))
        for index in range(horizon)
    )


_PROFILES: tuple[SSMProfile, ...] = (
    SSMProfile(
        "ssm_lgssm_qr", "lgssm", 4, 0.75, False,
        ("rho_unconstrained", "log_measurement_noise"),
        (0.18, 0.05, 0.16, 0.11),
    ),
    SSMProfile(
        "ssm_lgssm_near_unit", "lgssm", 32, 0.97, False,
        ("rho_unconstrained", "log_measurement_noise"),
        _series(32, phase=0.13, scale=0.045),
    ),
    SSMProfile(
        "ssm_lgssm_small_noise", "lgssm", 16, 0.75, False,
        ("rho_unconstrained", "log_measurement_noise"),
        _series(16, phase=0.41, scale=0.025),
    ),
    SSMProfile(
        "ssm_lgssm_long_horizon", "lgssm", 128, 0.75, False,
        ("rho_unconstrained", "log_measurement_noise"),
        _series(128, phase=0.73, scale=0.06),
    ),
    SSMProfile(
        "ssm_nonlinear", "nonlinear_ssm", 3, None, True,
        ("rho", "sigma", "beta"),
        (0.10, 0.04, 0.16),
        SIMPLE_NONLINEAR_DEFAULT_FILTER_ID,
    ),
    SSMProfile(
        "ssm_nonlinear_long_horizon", "nonlinear_ssm", 16, None, True,
        ("rho", "sigma", "beta"),
        _series(16, phase=0.29, scale=0.035),
        SIMPLE_NONLINEAR_DEFAULT_FILTER_ID,
    ),
)
PROFILES: Mapping[str, SSMProfile] = {profile.target_id: profile for profile in _PROFILES}


def get_ssm_profile(target_id: str) -> SSMProfile:
    try:
        return PROFILES[str(target_id)]
    except KeyError as exc:
        raise ValueError(f"unknown state-space validation profile: {target_id}") from exc


def available_ssm_profiles() -> tuple[str, ...]:
    return tuple(profile.target_id for profile in _PROFILES)


def _sha256(prefix: str, payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"sha256:{hashlib.sha256((prefix + ":" + blob).encode()).hexdigest()}"


def _observations(profile: SSMProfile, data: Any) -> tf.Tensor:
    values = profile.default_observations if data is None else data
    tensor = tf.convert_to_tensor(values, dtype=tf.float64)
    if tensor.shape.rank == 1:
        tensor = tensor[:, tf.newaxis]
    if tensor.shape.rank != 2 or tensor.shape[-1] != 1:
        raise ValueError(f"{profile.target_id} observations must have shape [time, 1]")
    if tensor.shape[0] != profile.horizon:
        raise ValueError(
            f"{profile.target_id} expects {profile.horizon} observations; "
            f"received {tensor.shape[0]}"
        )
    tf.debugging.assert_all_finite(tensor, "state-space observations must be finite")
    return tensor


class SSMValidationTarget:
    """ValidationTarget-compatible wrapper around an admitted SSM adapter."""

    batch_rank_policy = "rank2_required"

    def __init__(self, profile: SSMProfile, *, parameters: Mapping[str, Any] | None,
                 data: Any = None, control: str = "baseline", jit_compile: bool = True) -> None:
        if control not in {"baseline", "noop"}:
            raise ValueError(f"state-space profiles do not support control={control!r}")
        self.target_id = profile.target_id
        self.profile = profile
        self.control = control
        self.data_tensor = _observations(profile, data)
        self.data = [float(item) for item in tf.reshape(self.data_tensor, [-1]).numpy().tolist()]
        supplied = dict(parameters or {})
        if profile.family == "lgssm":
            cap = float(supplied.pop("persistence_cap", profile.persistence_cap))
            if not 0.0 < cap < 1.0 or not math.isfinite(cap):
                raise ValueError("LGSSM persistence_cap must be finite and strictly between zero and one")
            supplied["persistence_cap"] = cap
            self.parameters = supplied
            suffix = profile.target_id
            fixture = make_lgssm_generic_target_fixture(
                observations=self.data_tensor,
                persistence_cap=cap,
                data_hash=_sha256("ssm-data", self.data),
                model_hash=_sha256("ssm-model", {"profile": suffix, "cap": cap}),
                transform_hash=_sha256("ssm-transform", {"profile": suffix, "cap": cap}),
                filter_hash=_sha256("ssm-filter", {"profile": suffix}),
            )
            self.adapter = fixture.adapter
        else:
            filter_id = str(supplied.pop("filter_id", profile.filter_id))
            supplied["filter_id"] = filter_id
            self.parameters = supplied
            fixture = make_simple_nonlinear_generic_target_fixture(
                filter_id=filter_id,
                observations=self.data_tensor,
                data_hash=_sha256("ssm-data", self.data),
                model_hash=_sha256("ssm-model", {"profile": profile.target_id}),
                transform_hash=_sha256("ssm-transform", {"profile": profile.target_id}),
                filter_hash=_sha256("ssm-filter", {"profile": profile.target_id, "filter": filter_id}),
                jit_compile=jit_compile,
            )
            self.adapter = fixture.adapter
        self.parameter_dim = int(self.adapter.parameter_dim)
        self.spec = None  # Filled by ValidationTarget after catalog lookup.

    def adapter_signature(self) -> str:
        return self.adapter.adapter_signature()

    def value_score_capability(self):
        from dataclasses import replace
        # The public validation route owns the experiment scope.  The generic
        # adapter keeps its fixture-local scope in its signature, while the
        # tuner binding must see the same public scope used by the design.
        return replace(
            self.adapter.value_score_capability(),
            target_scope="inference_validation",
        )

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.adapter.parameter_names)

    def to_model(self, q: Any) -> tf.Tensor:
        tensor = tf.convert_to_tensor(q, dtype=tf.float64)
        if self.profile.family == "lgssm":
            cap = tf.constant(self.parameters["persistence_cap"], tf.float64)
            return tf.stack((cap * tf.math.tanh(tensor[..., 0]),
                             tf.exp(tensor[..., 1])), axis=-1)
        return tensor

    def _value_score(self, position: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(position, dtype=tf.float64)
        scalar = tensor.shape.rank == 1
        batch = tensor[tf.newaxis, :] if scalar else tensor
        value, score = self.adapter.log_prob_and_grad(batch)
        return (value[0], score[0]) if scalar else (value, score)

    def log_density(self, q: Any) -> tf.Tensor:
        value, _score = self._value_score(q)
        return value

    def log_prob_and_grad(self, position: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self._value_score(position)

    def _batch_score(self, q: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.adapter.log_prob_and_grad(q)

    def target_status_telemetry(self, position: Any):
        q = tf.convert_to_tensor(position, dtype=tf.float64)
        finite = tf.reduce_all(tf.math.is_finite(q), axis=-1)
        return {
            "status_code": tf.where(finite, tf.zeros_like(tf.cast(finite, tf.int32)),
                                    tf.ones_like(tf.cast(finite, tf.int32))),
            "valid_pre_regularized_score": finite,
            "floor_count_value": tf.zeros(tf.shape(q)[:-1], tf.int32),
        }


def make_ssm_validation_target(target_id: str, parameters: Mapping[str, Any] | None = None,
                               data: Any = None, *, control: str = "baseline",
                               jit_compile: bool = True) -> SSMValidationTarget:
    return SSMValidationTarget(get_ssm_profile(target_id), parameters=parameters,
                               data=data, control=control, jit_compile=jit_compile)
