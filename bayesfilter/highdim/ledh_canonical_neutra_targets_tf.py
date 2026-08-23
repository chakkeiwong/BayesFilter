"""Canonical NeuTra target factory (P7 rebind).

Replaces the bootstrap-lane `make_genut_neutra_target` binding: NeuTra
targets are now built on the CANONICAL LEDH-PF-PF stack (UKF lifecycle,
per-particle flow, analytical score, fused batch lane). The frozen
observation datasets and stateless noise streams are reused so the DATA
scope is unchanged; target signatures are FRESH by design — comparability
with pre-2026-08-21 artifacts is severed per the invalidation notice.

Bridge semantics: each model binds a `PerPointScoreModel` (fused lane) and
a `NonlinearScoreModel` (single-cloud authority) built from the same
model functions; the fused lane is the NeuTra-eligible batch backend, the
single-cloud lane is the parity oracle's subject.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    PerPointScoreModel,
    canonical_batch_fused_value_score,
)

Tensor = tf.Tensor
DTYPE = tf.float64


@dataclass(frozen=True)
class CanonicalNeuTraTarget:
    model_id: str
    parameter_names: tuple[str, ...]
    fused_model: PerPointScoreModel
    observations: Tensor
    initial_states: Tensor
    initial_covariances: Tensor
    noises: Tensor
    substeps: int
    data_id: str
    algorithm_id: str

    def target_signature(self) -> str:
        payload = {
            "schema": "bayesfilter.canonical_neutra_target.v1",
            "model_id": self.model_id,
            "algorithm_id": self.algorithm_id,
            "data_id": self.data_id,
            "observations_sha256": _tensor_hash(self.observations),
            "initial_sha256": _tensor_hash(self.initial_states),
            "noises_sha256": _tensor_hash(self.noises),
            "substeps": self.substeps,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def batch_value_score(
        self, theta: Tensor, directions: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return canonical_batch_fused_value_score(
            self.fused_model,
            theta,
            directions,
            self.initial_states,
            self.initial_covariances,
            self.noises,
            self.observations,
            substeps=self.substeps,
        )


def _tensor_hash(value: Tensor) -> str:
    encoded = tf.io.serialize_tensor(
        tf.convert_to_tensor(value)
    ).numpy()
    return hashlib.sha256(encoded).hexdigest()


def make_canonical_neutra_target(
    model: str,
    *,
    particle_count: int = 1008,
    noise_seed: int = 140000,
    substeps: int = 12,
) -> CanonicalNeuTraTarget:
    """Build a canonical-lane NeuTra target on the frozen datasets.

    The per-point fused model callbacks receive theta rows [M, P] aligned
    with points [M, d] and must be elementwise in the row dimension.
    """

    name = str(model).lower().replace("-", "_")
    if name in ("austria", "austria_sir", "sir"):
        from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
            SIR_OBSERVATION_SHA256,
            generate_frozen_sir_dataset_tf,
        )
        from bayesfilter.highdim.models import zhao_cui_sir_austria_model

        _states, observations64, _all = generate_frozen_sir_dataset_tf()
        observations = tf.cast(observations64, DTYPE)
        dimension, horizon = 18, 20
        parameter_names = (
            "log_kappa_scale",
            "log_nu_scale",
            "log_observation_noise_scale",
        )
        data_id = f"austria_sir_y1_y20_sha256_{SIR_OBSERVATION_SHA256}"
        initial_mean = tf.cast(
            zhao_cui_sir_austria_model().initial_mean, DTYPE
        )
        fused_model = _austria_fused_model()
        initial_covariance_scale = 1.0
    else:
        raise ValueError(
            f"canonical NeuTra factory: model {model!r} not yet bridged "
            "(austria_sir is the claim-critical first binding; remaining "
            "models follow the same template)"
        )

    initial_noise = tf.cast(
        tf.random.stateless_normal(
            [particle_count, dimension], [noise_seed, 101], dtype=tf.float32
        ),
        DTYPE,
    )
    process_noise = tf.cast(
        tf.random.stateless_normal(
            [horizon, particle_count, dimension],
            [noise_seed, 102],
            dtype=tf.float32,
        ),
        DTYPE,
    )
    initial_states = initial_mean[None, :] + initial_noise
    initial_covariances = initial_covariance_scale * tf.eye(
        dimension, batch_shape=[particle_count], dtype=DTYPE
    )

    return CanonicalNeuTraTarget(
        model_id=name,
        parameter_names=parameter_names,
        fused_model=fused_model,
        observations=observations,
        initial_states=initial_states,
        initial_covariances=initial_covariances,
        noises=process_noise,
        substeps=substeps,
        data_id=data_id,
        algorithm_id="ledh_canonical_pfpf_ot_ukf_analytical_v1",
    )


def _austria_fused_model() -> PerPointScoreModel:
    """Austria SIR with PER-POINT theta rows (fused-lane contract).

    Same dynamics/derivations as `austria_sir_canonical_model`, with theta
    broadcast per point; observation covariance uses the reference
    theta_2=0 scale (100*I_9) — theta_2 dependence of R enters the weight
    via the observation log-density, and its score contribution is the
    R-derivative term, handled at the density level (linear-H flow input
    R is a proposal-design choice corrected by the PF-PF identity).
    """

    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    base = zhao_cui_sir_austria_model()
    adjacency = tf.cast(base._adjacency_matrix, DTYPE)  # noqa: SLF001
    degree = tf.reduce_sum(adjacency, axis=1)
    step = tf.constant(0.005, DTYPE)
    infectious_matrix = tf.stack(
        [tf.one_hot(2 * index + 1, 18, dtype=DTYPE) for index in range(9)],
        axis=0,
    )

    def rhs(theta_rows, state):
        kappa = 0.1 * tf.exp(theta_rows[:, 0])
        nu = 18.0 * tf.exp(theta_rows[:, 1])
        susceptible = state[:, 0::2]
        infectious = state[:, 1::2]
        neighbor_s = (
            tf.einsum("nj,kj->nk", susceptible, adjacency)
            - susceptible * degree
        )
        neighbor_i = (
            tf.einsum("nj,kj->nk", infectious, adjacency)
            - infectious * degree
        )
        infection = kappa[:, None] * susceptible * infectious
        rhs_s = -infection + 0.5 * neighbor_s
        rhs_i = infection - nu[:, None] * infectious + 0.5 * neighbor_i
        return tf.reshape(
            tf.stack([rhs_s, rhs_i], axis=2), [tf.shape(state)[0], 18]
        )

    def rhs_tangent(theta_rows, state, d_state, d_theta_rows):
        kappa = 0.1 * tf.exp(theta_rows[:, 0])
        nu = 18.0 * tf.exp(theta_rows[:, 1])
        d_kappa = kappa * d_theta_rows[:, 0]
        d_nu = nu * d_theta_rows[:, 1]
        susceptible = state[:, 0::2]
        infectious = state[:, 1::2]
        d_susceptible = d_state[:, 0::2]
        d_infectious = d_state[:, 1::2]
        d_neighbor_s = (
            tf.einsum("nj,kj->nk", d_susceptible, adjacency)
            - d_susceptible * degree
        )
        d_neighbor_i = (
            tf.einsum("nj,kj->nk", d_infectious, adjacency)
            - d_infectious * degree
        )
        infection = kappa[:, None] * susceptible * infectious
        d_infection = (
            d_kappa[:, None] * susceptible * infectious
            + kappa[:, None] * d_susceptible * infectious
            + kappa[:, None] * susceptible * d_infectious
        )
        d_rhs_s = -d_infection + 0.5 * d_neighbor_s
        d_rhs_i = (
            d_infection
            - d_nu[:, None] * infectious
            - nu[:, None] * d_infectious
            + 0.5 * d_neighbor_i
        )
        return tf.reshape(
            tf.stack([d_rhs_s, d_rhs_i], axis=2), [tf.shape(state)[0], 18]
        )

    def transition_mean_fn(theta_rows, points):
        current = points
        for _ in range(4):
            k1 = rhs(theta_rows, current)
            k2 = rhs(theta_rows, current + 0.5 * step * k1)
            k3 = rhs(theta_rows, current + 0.5 * step * k2)
            # SOURCE HALF-STEP k4 (reference adapter quirk)
            k4 = rhs(theta_rows, current + 0.5 * step * k3)
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return current

    def transition_mean_tangent_fn(theta_rows, points, d_points, d_theta_rows):
        current, d_current = points, d_points
        for _ in range(4):
            k1 = rhs(theta_rows, current)
            d1 = rhs_tangent(theta_rows, current, d_current, d_theta_rows)
            k2 = rhs(theta_rows, current + 0.5 * step * k1)
            d2 = rhs_tangent(
                theta_rows, current + 0.5 * step * k1,
                d_current + 0.5 * step * d1, d_theta_rows,
            )
            k3 = rhs(theta_rows, current + 0.5 * step * k2)
            d3 = rhs_tangent(
                theta_rows, current + 0.5 * step * k2,
                d_current + 0.5 * step * d2, d_theta_rows,
            )
            # SOURCE HALF-STEP k4 (reference adapter quirk)
            k4 = rhs(theta_rows, current + 0.5 * step * k3)
            d4 = rhs_tangent(
                theta_rows, current + 0.5 * step * k3,
                d_current + 0.5 * step * d3, d_theta_rows,
            )
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            d_current = d_current + step / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        return d_current

    return PerPointScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: tf.einsum(
            "oi,ni->no", infectious_matrix, points
        ),
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            infectious_matrix, [tf.shape(points)[0], 9, 18]
        ),
        observation_tangent_fn=lambda points, d_points: tf.einsum(
            "oi,ni->no", infectious_matrix, d_points
        ),
        process_covariance=tf.eye(18, dtype=DTYPE),
        observation_covariance=100.0 * tf.eye(9, dtype=DTYPE),
    )


__all__ = [
    "CanonicalNeuTraTarget",
    "make_canonical_neutra_target",
]
