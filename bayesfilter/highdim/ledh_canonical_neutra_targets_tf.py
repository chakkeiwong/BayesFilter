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
from typing import Any, Callable, Mapping

import tensorflow as tf

from bayesfilter.highdim.ledh_alg1_contract import LEDH_PRODUCTION_PROGRAM_V1
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
    reset_design: Tensor
    score_kwargs: Mapping[str, Any]

    def target_signature(self) -> str:
        payload = {
            "schema": "bayesfilter.canonical_neutra_target.v2",
            "model_id": self.model_id,
            "algorithm_id": self.algorithm_id,
            "data_id": self.data_id,
            "observations_sha256": _tensor_hash(self.observations),
            "initial_sha256": _tensor_hash(self.initial_states),
            "noises_sha256": _tensor_hash(self.noises),
            "reset_design_sha256": _tensor_hash(self.reset_design),
            "score_kwargs": dict(self.score_kwargs),
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
            reset_design=self.reset_design,
            **self.score_kwargs,
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
    elif name in ("predator_prey", "pp"):
        from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
            PP_OBSERVATION_SHA256,
            generate_frozen_predator_prey_dataset_tf,
        )

        _states, observations64 = generate_frozen_predator_prey_dataset_tf()
        observations = tf.cast(observations64, DTYPE)
        dimension, horizon = 2, int(observations.shape[0])
        parameter_names = tuple(
            f"{letter}_source" for letter in "rKasuv"
        )
        data_id = f"predator_prey_T20_sha256_{PP_OBSERVATION_SHA256}"
        initial_mean = tf.constant([50.0, 5.0], DTYPE)
        fused_model = _predator_prey_fused_model()
        initial_covariance_scale = 1.0
    elif name == "lgssm":
        observations = _lgssm_frozen_observations()
        dimension, horizon = 3, 50
        parameter_names = (
            "phi1", "phi2", "phi3", "q_scale", "r_scale",
        )
        data_id = "benchmark_lgssm_m3_T50_seed81100"
        initial_mean = tf.zeros([3], DTYPE)
        fused_model = _diagonal_lgssm_fused_model()
        initial_covariance_scale = 1.0
    elif name in ("ksc", "ksc_sv"):
        from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
            generate_frozen_exact_sv_dataset_tf,
        )
        from bayesfilter.testing.ksc_ukf_neutra_target_tf import (
            transformed_ksc_observations,
        )

        _states, raw = generate_frozen_exact_sv_dataset_tf()
        observations = tf.cast(
            transformed_ksc_observations(raw), DTYPE
        )[:, None] if tf.cast(
            transformed_ksc_observations(raw), DTYPE
        ).shape.rank == 1 else tf.cast(
            transformed_ksc_observations(raw), DTYPE
        )
        dimension, horizon = 1, int(observations.shape[0])
        parameter_names = ("gamma_probit", "log_beta")
        data_id = "zhao_cui_sv_ksc_T1000_seed81101"
        initial_mean = tf.zeros([1], DTYPE)
        fused_model = _ksc_fused_model()
        initial_covariance_scale = 1.0
    else:
        raise ValueError(
            f"canonical NeuTra factory: model {model!r} not bridged"
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
    reset_basis = tf.concat(
        [tf.eye(dimension, dtype=DTYPE), -tf.eye(dimension, dtype=DTYPE)],
        axis=0,
    )
    reset_repeats = (particle_count + 2 * dimension - 1) // (2 * dimension)
    reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:particle_count]
    score_kwargs = dict(LEDH_PRODUCTION_PROGRAM_V1["score"])

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
        algorithm_id=(
            "ledh_canonical_pfpf_ot_contract_e_dual_cap_"
            "trust_region_analytical_v2"
        ),
        reset_design=reset_design,
        score_kwargs=score_kwargs,
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


def _lgssm_frozen_observations() -> Tensor:
    """Frozen LGSSM T=50 dataset (verbatim generator semantics from the
    historical factory, seed 81100, float64)."""

    generator = tf.random.Generator.from_seed(81100)
    phi = tf.constant([0.72, 0.55, 0.35], tf.float64)
    q_scale = tf.constant(0.35, tf.float64)
    r_scale = tf.constant(0.45, tf.float64)
    matrix = tf.constant(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]],
        tf.float64,
    )
    state = (
        q_scale
        / tf.sqrt(1.0 - tf.square(phi))
        * generator.normal([3], dtype=tf.float64)
    )
    rows = []
    for _ in range(50):
        state = phi * state + q_scale * generator.normal(
            [3], dtype=tf.float64
        )
        rows.append(
            tf.linalg.matvec(matrix, state)
            + r_scale * generator.normal([3], dtype=tf.float64)
        )
    return tf.stack(rows)


def _predator_prey_fused_model() -> PerPointScoreModel:
    """Predator-prey with per-point theta rows [M, 6]."""

    step = tf.constant(0.1, DTYPE)

    def rhs(theta_rows, state):
        r = theta_rows[:, 0]
        cap = theta_rows[:, 1]
        half = theta_rows[:, 2]
        s_r = theta_rows[:, 3]
        u_r = theta_rows[:, 4]
        v_r = theta_rows[:, 5]
        prey, predator = state[:, 0], state[:, 1]
        inter = prey * predator / (half + prey)
        return tf.stack(
            [
                r * prey * (1.0 - prey / cap) - s_r * inter,
                u_r * inter - v_r * predator,
            ],
            axis=1,
        )

    def rhs_tangent(theta_rows, state, d_state, d_theta_rows):
        r, cap, half = theta_rows[:, 0], theta_rows[:, 1], theta_rows[:, 2]
        s_r, u_r, v_r = theta_rows[:, 3], theta_rows[:, 4], theta_rows[:, 5]
        dr, dcap, dhalf = d_theta_rows[:, 0], d_theta_rows[:, 1], d_theta_rows[:, 2]
        ds_r, du_r, dv_r = d_theta_rows[:, 3], d_theta_rows[:, 4], d_theta_rows[:, 5]
        prey, predator = state[:, 0], state[:, 1]
        d_prey, d_predator = d_state[:, 0], d_state[:, 1]
        denom = half + prey
        inter = prey * predator / denom
        d_inter = (
            predator * half / tf.square(denom) * d_prey
            + prey / denom * d_predator
            - prey * predator / tf.square(denom) * dhalf
        )
        logistic = prey * (1.0 - prey / cap)
        d_logistic = (
            (1.0 - 2.0 * prey / cap) * d_prey
            + tf.square(prey) / tf.square(cap) * dcap
        )
        return tf.stack(
            [
                dr * logistic + r * d_logistic - ds_r * inter - s_r * d_inter,
                du_r * inter + u_r * d_inter - dv_r * predator - v_r * d_predator,
            ],
            axis=1,
        )

    def transition_mean_fn(theta_rows, points):
        current = points
        for _ in range(20):
            k1 = rhs(theta_rows, current)
            k2 = rhs(theta_rows, current + 0.5 * step * k1)
            k3 = rhs(theta_rows, current + 0.5 * step * k2)
            k4 = rhs(theta_rows, current + step * k3)
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return current

    def transition_mean_tangent_fn(theta_rows, points, d_points, d_theta_rows):
        current, d_current = points, d_points
        for _ in range(20):
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
            k4 = rhs(theta_rows, current + step * k3)
            d4 = rhs_tangent(
                theta_rows, current + step * k3,
                d_current + step * d3, d_theta_rows,
            )
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            d_current = d_current + step / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        return d_current

    return PerPointScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda p: p,
        observation_jacobian_fn=lambda p: tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(p)[0], 2, 2]
        ),
        observation_tangent_fn=lambda p, d: d,
        process_covariance=4.0 * tf.eye(2, dtype=DTYPE),
        observation_covariance=4.0 * tf.eye(2, dtype=DTYPE),
    )


def _diagonal_lgssm_fused_model() -> PerPointScoreModel:
    """Diagonal LGSSM with per-point theta rows [M, 5] and the reference
    observation matrix. Frozen-scope noise scales (q=0.35, r=0.45) pinned
    as flow/weight inputs; phi directions carry the score."""

    matrix = tf.constant(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]], DTYPE
    )

    def transition_mean_fn(theta_rows, points):
        return points * theta_rows[:, :3]

    def transition_mean_tangent_fn(theta_rows, points, d_points, d_theta_rows):
        return d_points * theta_rows[:, :3] + points * d_theta_rows[:, :3]

    return PerPointScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda p: tf.einsum("od,nd->no", matrix, p),
        observation_jacobian_fn=lambda p: tf.broadcast_to(
            matrix, [tf.shape(p)[0], 3, 3]
        ),
        observation_tangent_fn=lambda p, d: tf.einsum(
            "od,nd->no", matrix, d
        ),
        process_covariance=tf.square(tf.constant(0.35, DTYPE))
        * tf.eye(3, dtype=DTYPE),
        observation_covariance=tf.square(tf.constant(0.45, DTYPE))
        * tf.eye(3, dtype=DTYPE),
    )


def _ksc_fused_model() -> PerPointScoreModel:
    """KSC with per-point theta rows [M, 2] and the reference 7-component
    mixture observation density (per-point mirror of the corrected
    single-cloud model)."""

    import numpy as _np

    weights = tf.constant(
        [0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750],
        DTYPE,
    )
    means = tf.constant(
        [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819],
        DTYPE,
    ) - tf.constant(1.2704, DTYPE)
    variances = tf.constant(
        [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261],
        DTYPE,
    )
    mixture_mean = tf.reduce_sum(weights * means)
    mixture_var = tf.reduce_sum(
        weights * (variances + tf.square(means))
    ) - tf.square(mixture_mean)
    log_two_pi = tf.constant(float(_np.log(2.0 * _np.pi)), DTYPE)

    def gamma_of(theta_rows):
        return 0.5 * (
            1.0
            + tf.math.erf(
                theta_rows[:, 0] / tf.sqrt(tf.constant(2.0, DTYPE))
            )
        )

    def transition_mean_fn(theta_rows, points):
        return gamma_of(theta_rows)[:, None] * points

    def transition_mean_tangent_fn(theta_rows, points, d_points, d_theta_rows):
        gamma = gamma_of(theta_rows)
        normalizer = tf.constant(
            float(1.0 / _np.sqrt(2.0 * _np.pi)), DTYPE
        )
        dgamma = (
            normalizer
            * tf.exp(-0.5 * tf.square(theta_rows[:, 0]))
            * d_theta_rows[:, 0]
        )
        return dgamma[:, None] * points + gamma[:, None] * d_points

    def mixture_terms(theta_rows, points, observation):
        w = observation[0] - 2.0 * theta_rows[:, 1] - points[:, 0]
        terms = (
            tf.math.log(weights)[None, :]
            - 0.5
            * (
                tf.square(w[:, None] - means[None, :]) / variances[None, :]
                + tf.math.log(variances)[None, :]
                + log_two_pi
            )
        )
        return w, terms

    def observation_log_density_fn(theta_rows, points, observation):
        _w, terms = mixture_terms(theta_rows, points, observation)
        return tf.reduce_logsumexp(terms, axis=1)

    def observation_log_density_tangent_fn(
        theta_rows, points, observation, d_points, d_theta_rows
    ):
        w, terms = mixture_terms(theta_rows, points, observation)
        responsibilities = tf.nn.softmax(terms, axis=1)
        location_score = tf.reduce_sum(
            responsibilities
            * (w[:, None] - means[None, :])
            / variances[None, :],
            axis=1,
        )
        d_w = -d_points[:, 0] - 2.0 * d_theta_rows[:, 1]
        return -location_score * d_w

    def observation_fn(points):
        # flow proposal input; frozen-scope log_beta=0.1 offset
        # (proposal-design choice, corrected by the PF-PF identity)
        return points + mixture_mean + 2.0 * tf.constant(0.1, DTYPE)

    return PerPointScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=lambda p: tf.ones(
            [tf.shape(p)[0], 1, 1], DTYPE
        ),
        observation_tangent_fn=lambda p, d: d,
        process_covariance=tf.ones([1, 1], DTYPE),
        observation_covariance=mixture_var[None, None],
        observation_log_density_fn=observation_log_density_fn,
        observation_log_density_tangent_fn=observation_log_density_tangent_fn,
    )
