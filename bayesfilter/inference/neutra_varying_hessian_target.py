"""TensorFlow target and explicit score for smooth varying-Hessian NeuTra tests.

The mathematical source is ``dsge_hmc.benchmarks.nk_like_mild``. This module
implements its smooth batched value formula and its analytic score using only
TensorFlow and Python standard-library provenance handling. It does not import
the source repository at candidate runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability


SMOOTH_RADIUS_TAU = 0.05
SMOOTH_WEAK_TAU = 0.002
_TARGET_PARAMETERS = {
    "nk_like_mild_smooth": (0.35, 0.6, 0.25),
    "nk_like_strong_smooth": (0.70, 0.9, 0.45),
}


class VaryingHessianTargetError(RuntimeError):
    """Raised when source-bound smooth target inputs are invalid."""


@dataclass(frozen=True)
class VaryingHessianTargetSpec:
    """Frozen affine lift and smooth ridge parameters in physical coordinates."""

    name: str
    dimension: int
    mu: tuple[float, ...]
    lchol: tuple[tuple[float, ...], ...]
    rot_alpha: float
    weak_collapse: float
    stiff_growth: float
    constants_path: str
    constants_sha256: str
    constants_hash: str | None

    def __post_init__(self) -> None:
        if self.name not in _TARGET_PARAMETERS:
            raise ValueError("unsupported varying-Hessian target")
        if isinstance(self.dimension, bool) or int(self.dimension) < 2:
            raise ValueError("dimension must be at least two")
        dimension = int(self.dimension)
        if len(self.mu) != dimension or len(self.lchol) != dimension:
            raise ValueError("affine-lift dimensions are inconsistent")
        if any(len(row) != dimension for row in self.lchol):
            raise ValueError("lchol must be square")
        values = (*self.mu, *(item for row in self.lchol for item in row))
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("affine-lift values must be finite")
        if any(float(self.lchol[index][index]) <= 0.0 for index in range(dimension)):
            raise ValueError("lchol diagonal must be positive")
        expected = _TARGET_PARAMETERS[self.name]
        actual = (self.rot_alpha, self.weak_collapse, self.stiff_growth)
        if any(
            not math.isclose(float(value), float(reference), rel_tol=0.0, abs_tol=1.0e-12)
            for value, reference in zip(actual, expected)
        ):
            raise ValueError("smooth target parameters do not match the source target")
        if len(self.constants_sha256) != 64:
            raise ValueError("constants_sha256 must be a SHA-256 hex digest")

    def manifest_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["mu"] = list(self.mu)
        payload["lchol"] = [list(row) for row in self.lchol]
        payload["schema"] = "bayesfilter.neutra.varying_hessian_target.v1"
        payload["source_formula"] = (
            "dsge_hmc.benchmarks.nk_like_mild.log_prob_batch_tf"
        )
        payload["smooth"] = True
        return payload


def load_varying_hessian_target_spec(
    path: str | Path,
    *,
    expected_name: str,
) -> VaryingHessianTargetSpec:
    """Load and bind a source frozen-constants file without importing its code."""

    constants_path = Path(path).resolve()
    raw = constants_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VaryingHessianTargetError("frozen constants JSON is unreadable") from error
    if not isinstance(payload, Mapping):
        raise VaryingHessianTargetError("frozen constants JSON object is required")
    values = payload.get("target_constants", payload)
    if not isinstance(values, Mapping):
        raise VaryingHessianTargetError("target_constants object is required")
    name = str(values.get("name", ""))
    if name != str(expected_name):
        raise VaryingHessianTargetError(
            f"target constants name mismatch: {name!r} != {str(expected_name)!r}"
        )
    try:
        spec = VaryingHessianTargetSpec(
            name=name,
            dimension=int(values["dim"]),
            mu=tuple(float(value) for value in values["mu"]),
            lchol=tuple(tuple(float(value) for value in row) for row in values["lchol"]),
            rot_alpha=float(values["rot_alpha"]),
            weak_collapse=float(values["weak_collapse"]),
            stiff_growth=float(values["stiff_growth"]),
            constants_path=constants_path.as_posix(),
            constants_sha256=hashlib.sha256(raw).hexdigest(),
            constants_hash=(
                None
                if payload.get("target_constants_hash") is None
                else str(payload["target_constants_hash"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise VaryingHessianTargetError("frozen constants payload is invalid") from error
    return spec


def varying_hessian_log_prob_and_score_batch(
    spec: VaryingHessianTargetSpec,
    physical: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate the source-equivalent smooth ridge target and exact score."""

    rows = _rank2(physical, spec.dimension, "physical")
    lchol = tf.constant(spec.lchol, tf.float64)
    mu = tf.constant(spec.mu, tf.float64)
    x = tf.transpose(
        tf.linalg.triangular_solve(lchol, tf.transpose(rows - mu), lower=True)
    )
    radius = tf.sqrt(
        tf.reduce_sum(tf.square(x), axis=1) + tf.constant(1.0e-24, tf.float64)
    )
    scale = radius / tf.constant(3.0, tf.float64)
    raw_weak = tf.constant(0.15, tf.float64) * (
        tf.constant(1.0, tf.float64)
        - tf.constant(spec.weak_collapse, tf.float64) * tf.square(scale)
    )
    radius_tau = tf.constant(SMOOTH_RADIUS_TAU, tf.float64)
    weak_tau = tf.constant(SMOOTH_WEAK_TAU, tf.float64)
    angle_scale = tf.constant(1.0, tf.float64) - radius_tau * tf.nn.softplus(
        (tf.constant(1.0, tf.float64) - scale) / radius_tau
    )
    weak = tf.constant(1.0e-3, tf.float64) + weak_tau * tf.nn.softplus(
        (raw_weak - tf.constant(1.0e-3, tf.float64)) / weak_tau
    )
    tanh_x0 = tf.math.tanh(x[:, 0])
    angle = tf.constant(spec.rot_alpha, tf.float64) * tanh_x0 * angle_scale
    cosine = tf.math.cos(angle)
    sine = tf.math.sin(angle)
    stiff = tf.constant(1.0, tf.float64) + tf.constant(
        spec.stiff_growth, tf.float64
    ) * tf.square(scale)
    y0 = cosine * x[:, 0] + sine * x[:, 1]
    y1 = -sine * x[:, 0] + cosine * x[:, 1]
    quadratic = weak * tf.square(y0) + stiff * tf.square(y1)
    if spec.dimension > 2:
        quadratic = quadratic + tf.reduce_sum(tf.square(x[:, 2:]), axis=1)
    value = -0.5 * quadratic

    a00 = tf.square(cosine) * weak + tf.square(sine) * stiff
    a11 = tf.square(sine) * weak + tf.square(cosine) * stiff
    a01 = cosine * sine * (weak - stiff)
    direct_first = tf.stack(
        (
            2.0 * (a00 * x[:, 0] + a01 * x[:, 1]),
            2.0 * (a01 * x[:, 0] + a11 * x[:, 1]),
        ),
        axis=1,
    )
    direct = (
        direct_first
        if spec.dimension == 2
        else tf.concat((direct_first, 2.0 * x[:, 2:]), axis=1)
    )
    scale_score = x / (tf.constant(3.0, tf.float64) * radius[:, tf.newaxis])
    angle_scale_score = tf.math.sigmoid(
        (tf.constant(1.0, tf.float64) - scale) / radius_tau
    )[:, tf.newaxis] * scale_score
    x0_direction = tf.one_hot(0, spec.dimension, dtype=tf.float64)[tf.newaxis, :]
    angle_score = tf.constant(spec.rot_alpha, tf.float64) * (
        (tf.constant(1.0, tf.float64) - tf.square(tanh_x0))[:, tf.newaxis]
        * angle_scale[:, tf.newaxis]
        * x0_direction
        + tanh_x0[:, tf.newaxis] * angle_scale_score
    )
    raw_weak_score = -tf.constant(0.3 * spec.weak_collapse, tf.float64) * scale[
        :, tf.newaxis
    ] * scale_score
    weak_score = tf.math.sigmoid(
        (raw_weak - tf.constant(1.0e-3, tf.float64)) / weak_tau
    )[:, tf.newaxis] * raw_weak_score
    stiff_score = tf.constant(2.0 * spec.stiff_growth, tf.float64) * scale[
        :, tf.newaxis
    ] * scale_score
    quadratic_score = (
        direct
        + (2.0 * (weak - stiff) * y0 * y1)[:, tf.newaxis] * angle_score
        + tf.square(y0)[:, tf.newaxis] * weak_score
        + tf.square(y1)[:, tf.newaxis] * stiff_score
    )
    score_x = -0.5 * quadratic_score
    score_u = tf.transpose(
        tf.linalg.triangular_solve(lchol, tf.transpose(score_x), lower=True, adjoint=True)
    )
    tf.debugging.assert_all_finite(value, "varying-Hessian target value")
    tf.debugging.assert_all_finite(score_u, "varying-Hessian target score")
    return value, score_u


class VaryingHessianValueScoreAdapter:
    """Graph-native batch value/score adapter for a frozen smooth ridge target."""

    supports_retained_draw_batch = False
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True
    target_status_invalid_rows_become_nonfinite = False

    def __init__(self, spec: VaryingHessianTargetSpec) -> None:
        self.spec = spec
        self.parameter_dim = int(spec.dimension)
        self.target_scope = f"weighted_neutra_varying_hessian:{spec.name}"

    def log_prob_and_grad(self, physical: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return varying_hessian_log_prob_and_score_batch(self.spec, physical)

    def log_prob_and_grad_status(
        self, physical: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        value, score = self.log_prob_and_grad(physical)
        finite = tf.logical_and(
            tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=-1)
        )
        zeros = tf.zeros_like(value, tf.int32)
        ones = tf.ones_like(value, tf.float64)
        return value, score, {
            "status_code": tf.where(finite, zeros, tf.ones_like(zeros)),
            "valid_pre_regularized_score": finite,
            "floor_count_value": zeros,
            "min_innovation_eigenvalue": ones,
            "innovation_condition_estimate": ones,
        }

    def target_status_telemetry(self, physical: Any) -> Mapping[str, tf.Tensor]:
        return self.log_prob_and_grad_status(physical)[2]

    def adapter_signature(self) -> str:
        return _stable_hash(
            {
                "schema": "bayesfilter.neutra.varying_hessian_value_score.v1",
                "target_scope": self.target_scope,
                "target": self.spec.manifest_payload(),
                "value_score_authority": "graph_native_explicit_smooth_ridge_score",
            }
        )

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_explicit_smooth_varying_hessian_score",
            evidence_path=(
                "docs/plans/"
                "bayesfilter-weighted-forward-kl-positive-control-regression-plan-2026-08-12.md"
            ),
            target_scope=self.target_scope,
            nonclaims=(
                "source-bound smooth surrogate only",
                "no posterior-reference or HMC validity claim",
            ),
        )


def affine_local_to_physical(
    spec: VaryingHessianTargetSpec,
    local: Any,
) -> tf.Tensor:
    """Map rank-two affine-lift coordinates to physical target coordinates."""

    rows = _rank2(local, spec.dimension, "local")
    return tf.constant(spec.mu, tf.float64)[tf.newaxis, :] + tf.linalg.matvec(
        tf.constant(spec.lchol, tf.float64)[tf.newaxis, :, :], rows
    )


def physical_to_affine_local(
    spec: VaryingHessianTargetSpec,
    physical: Any,
) -> tf.Tensor:
    """Map rank-two physical coordinates to the frozen affine-lift chart."""

    rows = _rank2(physical, spec.dimension, "physical")
    return tf.transpose(
        tf.linalg.triangular_solve(
            tf.constant(spec.lchol, tf.float64),
            tf.transpose(rows - tf.constant(spec.mu, tf.float64)),
            lower=True,
        )
    )


class FrozenAffineLiftWeightedTransport:
    """Compose a frozen weighted IAF with the source-bound affine lift.

    The learned IAF maps HMC coordinates to affine-local coordinates.  The
    final fixed map is ``theta = mu + L @ local``.  It supplies the same
    explicit batch-native methods required by ``FixedTransportValueScoreAdapter``.
    """

    def __init__(self, spec: VaryingHessianTargetSpec, local_transport: Any) -> None:
        parameter_dim = int(getattr(local_transport, "parameter_dim", 0))
        if parameter_dim != int(spec.dimension):
            raise ValueError("affine lift and local transport dimensions differ")
        for name in (
            "forward",
            "forward_batch",
            "log_abs_det_jacobian",
            "log_abs_det_jacobian_batch",
            "pullback_score",
            "pullback_score_batch",
            "log_abs_det_jacobian_score",
            "log_abs_det_jacobian_score_batch",
            "manifest_payload",
        ):
            if not callable(getattr(local_transport, name, None)):
                raise TypeError(f"local transport must expose {name}")
        self.spec = spec
        self.local_transport = local_transport
        self.parameter_dim = int(spec.dimension)
        self._lchol = tf.constant(spec.lchol, tf.float64)
        self._mu = tf.constant(spec.mu, tf.float64)
        self._affine_logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self._lchol)))

    def manifest_payload(self) -> Mapping[str, Any]:
        local_manifest = self.local_transport.manifest_payload()
        payload = {
            "schema": "bayesfilter.neutra.varying_hessian_affine_lift_weighted_iaf.v1",
            "transport_id": "frozen_affine_lift_plus_weighted_dense_iaf",
            "parameter_dim": self.parameter_dim,
            "affine_lift": self.spec.manifest_payload(),
            "local_transport": local_manifest,
        }
        return {**payload, "transport_hash": _stable_hash(payload)}

    def forward_and_logdet(self, latent: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _rank2(latent, self.parameter_dim, "latent")
        local, local_logdet = self.local_transport.forward_and_logdet(values)
        physical = affine_local_to_physical(self.spec, local)
        return physical, local_logdet + self._affine_logdet

    def forward(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.forward_and_logdet(values[tf.newaxis, :])[0][0]
        return self.forward_and_logdet(values)[0]

    def forward_batch(self, latent: Any) -> tf.Tensor:
        return self.forward_and_logdet(latent)[0]

    def log_abs_det_jacobian(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.forward_and_logdet(values[tf.newaxis, :])[1][0]
        return self.forward_and_logdet(values)[1]

    def log_abs_det_jacobian_batch(self, latent: Any) -> tf.Tensor:
        return self.forward_and_logdet(latent)[1]

    def pullback_score(self, latent: Any, physical_score: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        score = tf.convert_to_tensor(physical_score, tf.float64)
        if values.shape.rank == 1 and score.shape.rank == 1:
            return self.pullback_score_batch(
                values[tf.newaxis, :], score[tf.newaxis, :]
            )[0]
        return self.pullback_score_batch(values, score)

    def pullback_score_batch(self, latent: Any, physical_score: Any) -> tf.Tensor:
        values = _rank2(latent, self.parameter_dim, "latent")
        score = _rank2(physical_score, self.parameter_dim, "physical_score")
        local_score = tf.matmul(score, self._lchol)
        return self.local_transport.pullback_score_batch(values, local_score)

    def log_abs_det_jacobian_score(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.log_abs_det_jacobian_score_batch(values[tf.newaxis, :])[0]
        return self.log_abs_det_jacobian_score_batch(values)

    def log_abs_det_jacobian_score_batch(self, latent: Any) -> tf.Tensor:
        values = _rank2(latent, self.parameter_dim, "latent")
        return self.local_transport.log_abs_det_jacobian_score_batch(values)

    def inverse_physical_to_latent_batch(self, physical: Any) -> tf.Tensor:
        local = physical_to_affine_local(self.spec, physical)
        latent, _local_logdet = self.local_transport.inverse_and_forward_logdet(local)
        return latent


def affine_scale_mixture_proposal(
    spec: VaryingHessianTargetSpec,
    *,
    x0_scales: tuple[float, ...] = (1.5, 4.0, 12.0, 32.0),
    probabilities: tuple[float, ...] = (0.40, 0.30, 0.20, 0.10),
) -> Mapping[str, tf.Tensor]:
    """Return a full-support affine-lift Gaussian scale-mixture proposal."""

    if len(x0_scales) != len(probabilities) or len(probabilities) < 2:
        raise ValueError("proposal scales and probabilities must have one shared count")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in x0_scales):
        raise ValueError("proposal scales must be finite and positive")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in probabilities):
        raise ValueError("proposal probabilities must be finite and positive")
    if not math.isclose(sum(float(value) for value in probabilities), 1.0, abs_tol=1.0e-12):
        raise ValueError("proposal probabilities must sum to one")
    dimension = int(spec.dimension)
    lchol = tf.constant(spec.lchol, tf.float64)
    scales = tf.constant(x0_scales, tf.float64)
    rest = tf.ones((len(x0_scales), dimension - 1), tf.float64)
    diagonal = tf.concat((scales[:, tf.newaxis], rest), axis=1)
    x_covariance = tf.linalg.diag(tf.square(diagonal))
    covariances = tf.matmul(
        lchol[tf.newaxis, :, :],
        tf.matmul(x_covariance, lchol[tf.newaxis, :, :], transpose_b=True),
    )
    means = tf.broadcast_to(
        tf.constant(spec.mu, tf.float64)[tf.newaxis, :],
        (len(x0_scales), dimension),
    )
    return {
        "identity": "affine_lift_x0_scale_mixture_v1",
        "probabilities": tf.constant(probabilities, tf.float64),
        "means": means,
        "covariances": covariances,
        "x0_scales": scales,
    }


def affine_ridge_tangent_mixture_proposal(
    spec: VaryingHessianTargetSpec,
    *,
    radii: tuple[float, ...] = (0.0, 6.0, 6.0, 18.0, 18.0),
    signs: tuple[float, ...] = (0.0, 1.0, -1.0, 1.0, -1.0),
    weak_scales: tuple[float, ...] = (3.0, 5.0, 5.0, 20.0, 20.0),
    stiff_scales: tuple[float, ...] = (1.5, 1.5, 1.5, 2.5, 2.5),
    probabilities: tuple[float, ...] = (0.10, 0.15, 0.15, 0.30, 0.30),
) -> Mapping[str, tf.Tensor]:
    """Return a full-support mixture following the source target's two tails.

    The smooth target's weak precision direction approaches angle ``+alpha``
    on the positive ``x0`` branch and ``-alpha`` on the negative branch. The
    two noncentral component pairs therefore cover the curved ridge locally;
    this construction is a geometry proposal, not a posterior approximation.
    """

    count = len(radii)
    if count < 3 or not (
        len(signs) == len(weak_scales) == len(stiff_scales) == len(probabilities) == count
    ):
        raise ValueError("ridge proposal profiles must share a component count of at least three")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (*weak_scales, *stiff_scales, *probabilities)
    ):
        raise ValueError("ridge proposal scales and probabilities must be finite and positive")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in radii):
        raise ValueError("ridge proposal radii must be finite and nonnegative")
    if not math.isclose(sum(float(value) for value in probabilities), 1.0, abs_tol=1.0e-12):
        raise ValueError("ridge proposal probabilities must sum to one")
    dimension = int(spec.dimension)
    lchol = tf.constant(spec.lchol, tf.float64)
    alpha = tf.constant(spec.rot_alpha, tf.float64)
    # Core, two intermediate branch components, and two broad tail components.
    weights = tf.constant(probabilities, tf.float64)
    radius_values = tf.constant(radii, tf.float64)
    sign_values = tf.constant(signs, tf.float64)
    angles = sign_values * alpha
    cosine = tf.math.cos(angles)
    sine = tf.math.sin(angles)
    # The negative branch has x0 < 0 and x1 > 0 because its weak direction is
    # (cos(alpha), -sin(alpha)) and the low-energy direction is negative.
    x0 = sign_values * radius_values * cosine
    x1 = tf.abs(sign_values) * radius_values * sine
    x_means = tf.concat(
        (x0[:, tf.newaxis], x1[:, tf.newaxis], tf.zeros((count, dimension - 2), tf.float64)),
        axis=1,
    )
    weak_scale = tf.constant(weak_scales, tf.float64)
    stiff_scale = tf.constant(stiff_scales, tf.float64)
    a00 = tf.square(cosine) * tf.square(weak_scale) + tf.square(sine) * tf.square(stiff_scale)
    a11 = tf.square(sine) * tf.square(weak_scale) + tf.square(cosine) * tf.square(stiff_scale)
    a01 = cosine * sine * (tf.square(weak_scale) - tf.square(stiff_scale))
    x_covariances = tf.linalg.diag(
        tf.concat(
            (
                tf.ones((count, 2), tf.float64),
                tf.ones((count, dimension - 2), tf.float64),
            ),
            axis=1,
        )
    )
    first_two = tf.stack(
        (
            tf.stack((a00, a01), axis=1),
            tf.stack((a01, a11), axis=1),
        ),
        axis=1,
    )
    indices = tf.constant(((0, 0), (0, 1), (1, 0), (1, 1)), tf.int32)
    x_covariances = tf.stack(
        [
            tf.tensor_scatter_nd_update(x_covariances[index], indices, tf.reshape(first_two[index], (-1,)))
            for index in range(count)
        ]
    )
    means = tf.constant(spec.mu, tf.float64)[tf.newaxis, :] + tf.linalg.matvec(
        lchol[tf.newaxis, :, :], x_means
    )
    covariances = tf.matmul(
        lchol[tf.newaxis, :, :],
        tf.matmul(x_covariances, lchol[tf.newaxis, :, :], transpose_b=True),
    )
    return {
        "identity": "affine_lift_smooth_ridge_tangent_mixture_v1",
        "probabilities": weights,
        "means": means,
        "covariances": covariances,
        "x_means": x_means,
        "x_covariances": x_covariances,
    }


def fit_defensive_branch_mixture_proposal(
    spec: VaryingHessianTargetSpec,
    pilot_physical: Any,
    pilot_log_weights: Any,
    *,
    defensive_proposal: Mapping[str, Any],
    defensive_weight: float = 0.05,
) -> Mapping[str, tf.Tensor]:
    """Fit two local branch Gaussians from a disjoint weighted pilot cloud.

    The pilot is used only to construct a replay proposal. Callers must assess
    the resulting mixture on independent rows before using it for training.
    """

    rows = _rank2(pilot_physical, spec.dimension, "pilot_physical")
    log_weights = tf.convert_to_tensor(pilot_log_weights, tf.float64)
    if log_weights.shape != (rows.shape[0],):
        raise ValueError("pilot_log_weights must match the static pilot row count")
    tf.debugging.assert_all_finite(log_weights, "pilot_log_weights")
    defensive_probability = float(defensive_weight)
    if not math.isfinite(defensive_probability) or not 0.0 < defensive_probability < 0.5:
        raise ValueError("defensive_weight must lie strictly between zero and 0.5")
    base_probabilities = tf.convert_to_tensor(defensive_proposal["probabilities"], tf.float64)
    base_means = tf.convert_to_tensor(defensive_proposal["means"], tf.float64)
    base_covariances = tf.convert_to_tensor(defensive_proposal["covariances"], tf.float64)
    if base_means.shape.rank != 2 or base_means.shape[1] != int(spec.dimension):
        raise ValueError("defensive proposal dimension mismatch")
    tf.debugging.assert_near(
        tf.reduce_sum(base_probabilities), tf.constant(1.0, tf.float64), atol=1.0e-12
    )
    lchol = tf.constant(spec.lchol, tf.float64)
    local = tf.transpose(
        tf.linalg.triangular_solve(lchol, tf.transpose(rows - tf.constant(spec.mu, tf.float64)), lower=True)
    )
    weights = tf.nn.softmax(log_weights)
    labels = tf.cast(local[:, 0] >= 0.0, tf.int32)
    branch_mass = tf.stack(
        [tf.reduce_sum(weights * tf.cast(labels == branch, tf.float64)) for branch in range(2)]
    )
    branch_means = []
    branch_covariances = []
    branch_ess = []
    for branch in range(2):
        branch_weights = weights * tf.cast(labels == branch, tf.float64)
        normalized = branch_weights / branch_mass[branch]
        effective = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized)))
        if float(effective.numpy()) < 20.0:
            raise VaryingHessianTargetError(
                f"pilot branch {branch} effective sample size is below 20"
            )
        mean = tf.reduce_sum(normalized[:, tf.newaxis] * local, axis=0)
        centered = local - mean
        covariance = tf.matmul(
            centered, normalized[:, tf.newaxis] * centered, transpose_a=True
        ) + tf.eye(spec.dimension, dtype=tf.float64) * tf.constant(1.0e-6, tf.float64)
        tf.linalg.cholesky(covariance)
        branch_means.append(mean)
        branch_covariances.append(covariance)
        branch_ess.append(effective)
    fitted_local_means = tf.stack(branch_means)
    fitted_local_covariances = tf.stack(branch_covariances)
    fitted_means = tf.constant(spec.mu, tf.float64)[tf.newaxis, :] + tf.linalg.matvec(
        lchol[tf.newaxis, :, :], fitted_local_means
    )
    fitted_covariances = tf.matmul(
        lchol[tf.newaxis, :, :],
        tf.matmul(fitted_local_covariances, lchol[tf.newaxis, :, :], transpose_b=True),
    )
    probabilities = tf.concat(
        (
            tf.constant(defensive_probability, tf.float64) * base_probabilities,
            tf.constant(1.0 - defensive_probability, tf.float64) * branch_mass,
        ),
        axis=0,
    )
    means = tf.concat((base_means, fitted_means), axis=0)
    covariances = tf.concat((base_covariances, fitted_covariances), axis=0)
    return {
        "identity": "affine_lift_weighted_pilot_branch_defensive_mixture_v1",
        "probabilities": probabilities,
        "means": means,
        "covariances": covariances,
        "pilot_branch_mass": branch_mass,
        "pilot_branch_effective_sample_size": tf.stack(branch_ess),
        "pilot_local_branch_means": fitted_local_means,
        "pilot_local_branch_covariances": fitted_local_covariances,
        "defensive_weight": tf.constant(defensive_probability, tf.float64),
    }


def reflect_first_local_coordinate(
    spec: VaryingHessianTargetSpec,
    physical: Any,
) -> tf.Tensor:
    """Reflect physical rows through the exact local ``x[0]`` symmetry plane."""

    rows = _rank2(physical, spec.dimension, "physical")
    local = physical_to_affine_local(spec, rows)
    reflected_local = tf.concat((-local[:, :1], local[:, 1:]), axis=1)
    return affine_local_to_physical(spec, reflected_local)


def fit_reflected_positive_branch_mixture_proposal(
    spec: VaryingHessianTargetSpec,
    pilot_physical: Any,
    pilot_log_weights: Any,
    *,
    defensive_proposal: Mapping[str, Any],
    defensive_weight: float = 0.05,
) -> Mapping[str, tf.Tensor]:
    """Fit a positive local branch and reflect it through a verified symmetry.

    This is valid only after an independent source-parity check establishes
    that the target is invariant under ``x[0] -> -x[0]``.  The fitted and
    reflected branches deliberately receive equal mass; that mass comes from
    the target symmetry, not from an imbalanced pilot importance estimate.
    """

    rows = _rank2(pilot_physical, spec.dimension, "pilot_physical")
    log_weights = tf.convert_to_tensor(pilot_log_weights, tf.float64)
    if log_weights.shape != (rows.shape[0],):
        raise ValueError("pilot_log_weights must match the static pilot row count")
    tf.debugging.assert_all_finite(log_weights, "pilot_log_weights")
    defensive_probability = float(defensive_weight)
    if not math.isfinite(defensive_probability) or not 0.0 < defensive_probability < 0.5:
        raise ValueError("defensive_weight must lie strictly between zero and 0.5")
    base_probabilities = tf.convert_to_tensor(defensive_proposal["probabilities"], tf.float64)
    base_means = tf.convert_to_tensor(defensive_proposal["means"], tf.float64)
    base_covariances = tf.convert_to_tensor(defensive_proposal["covariances"], tf.float64)
    if base_means.shape.rank != 2 or base_means.shape[1] != int(spec.dimension):
        raise ValueError("defensive proposal dimension mismatch")
    tf.debugging.assert_near(
        tf.reduce_sum(base_probabilities), tf.constant(1.0, tf.float64), atol=1.0e-12
    )
    lchol = tf.constant(spec.lchol, tf.float64)
    mu = tf.constant(spec.mu, tf.float64)
    local = tf.transpose(
        tf.linalg.triangular_solve(lchol, tf.transpose(rows - mu), lower=True)
    )
    all_weights = tf.nn.softmax(log_weights)
    positive_weights = all_weights * tf.cast(local[:, 0] >= 0.0, tf.float64)
    positive_mass = tf.reduce_sum(positive_weights)
    normalized = positive_weights / positive_mass
    positive_ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized)))
    if float(positive_ess.numpy()) < 20.0:
        raise VaryingHessianTargetError("positive pilot branch effective sample size is below 20")
    positive_mean = tf.reduce_sum(normalized[:, tf.newaxis] * local, axis=0)
    centered = local - positive_mean
    positive_covariance = tf.matmul(
        centered, normalized[:, tf.newaxis] * centered, transpose_a=True
    ) + tf.eye(spec.dimension, dtype=tf.float64) * tf.constant(1.0e-6, tf.float64)
    tf.linalg.cholesky(positive_covariance)
    reflection = tf.linalg.diag(
        tf.concat((tf.constant((-1.0,), tf.float64), tf.ones((spec.dimension - 1,), tf.float64)), axis=0)
    )
    fitted_local_means = tf.stack(
        (positive_mean, tf.linalg.matvec(reflection, positive_mean))
    )
    fitted_local_covariances = tf.stack(
        (
            positive_covariance,
            tf.matmul(
                reflection,
                tf.matmul(positive_covariance, reflection, transpose_b=True),
            ),
        )
    )
    fitted_means = mu[tf.newaxis, :] + tf.linalg.matvec(
        lchol[tf.newaxis, :, :], fitted_local_means
    )
    fitted_covariances = tf.matmul(
        lchol[tf.newaxis, :, :],
        tf.matmul(fitted_local_covariances, lchol[tf.newaxis, :, :], transpose_b=True),
    )
    learned_probability = tf.constant((1.0 - defensive_probability) / 2.0, tf.float64)
    probabilities = tf.concat(
        (defensive_probability * base_probabilities, tf.fill((2,), learned_probability)),
        axis=0,
    )
    return {
        "identity": "affine_lift_reflected_positive_pilot_defensive_mixture_v1",
        "probabilities": probabilities,
        "means": tf.concat((base_means, fitted_means), axis=0),
        "covariances": tf.concat((base_covariances, fitted_covariances), axis=0),
        "pilot_positive_branch_mass": positive_mass,
        "pilot_positive_branch_effective_sample_size": positive_ess,
        "pilot_positive_local_mean": positive_mean,
        "pilot_positive_local_covariance": positive_covariance,
        "reflection_matrix_local": reflection,
        "symmetry_assumption": "x0_local_reflection_exact_source_checked",
        "defensive_weight": tf.constant(defensive_probability, tf.float64),
    }


def _rank2(value: Any, dimension: int, name: str) -> tf.Tensor:
    rows = tf.convert_to_tensor(value, tf.float64)
    if rows.shape.rank != 2 or rows.shape[-1] != int(dimension):
        raise ValueError(f"{name} must have static shape [row, {int(dimension)}]")
    tf.debugging.assert_all_finite(rows, name)
    return rows


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "VaryingHessianTargetError",
    "VaryingHessianTargetSpec",
    "VaryingHessianValueScoreAdapter",
    "FrozenAffineLiftWeightedTransport",
    "affine_scale_mixture_proposal",
    "affine_ridge_tangent_mixture_proposal",
    "affine_local_to_physical",
    "fit_reflected_positive_branch_mixture_proposal",
    "load_varying_hessian_target_spec",
    "fit_defensive_branch_mixture_proposal",
    "physical_to_affine_local",
    "reflect_first_local_coordinate",
    "varying_hessian_log_prob_and_score_batch",
]
