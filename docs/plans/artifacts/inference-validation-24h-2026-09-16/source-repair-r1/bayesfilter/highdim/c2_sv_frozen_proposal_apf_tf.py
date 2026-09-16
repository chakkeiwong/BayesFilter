"""C2 stochastic-volatility model and frozen-proposal APF compilers.

The compilers create parameter-independent branches.  Claim-bearing value and
score evaluation is delegated to ``FrozenProposalAPFProgram``; this module does
not provide a second APF evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Mapping, Protocol, Sequence

import tensorflow as tf

from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    GaussianHermiteRetainedProposal,
    stateless_proposal_random_inputs,
)
from bayesfilter.highdim.c2_transformed_observation_student_proposal_tf import (
    C2TransformedObservationStudentProposal,
    build_c2_transformed_observation_student_proposal,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    MEASURE_ID,
    SCORE_BACKEND_ID,
    FrozenProposalAPFProgram,
    PreparedFrozenProposalBranch,
    prepare_frozen_proposal_apf_program,
    prepare_frozen_proposal_branch,
)


DTYPE = tf.float64
MODEL_ID = "c2_sv_gamma_log_beta_stationary_v1"
COMPILER_ID = "c2_frozen_proposal_branch_compiler_v1"
COMPILER_CLASSIFICATION = "extension_or_invention"
INITIAL_PROPOSAL_ID = "frozen_stationary_prior_at_theta_reference_v1"
_GAUSSIAN_TRANSFORM_CACHE: dict[tuple[str, int, bool], object] = {}


def _kron_same(matrix: tf.Tensor) -> tf.Tensor:
    dimension = int(matrix.shape[0])
    return tf.reshape(
        tf.einsum("ik,jl->ijkl", matrix, matrix),
        [dimension * dimension, dimension * dimension],
    )


@dataclass(frozen=True)
class C2StochasticVolatilityFrozenAPFModel:
    """Manual value/parameter-score model for theta=(gamma, log(beta))."""

    coupling_matrix: tf.Tensor
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if not tf.executing_eagerly():
            raise RuntimeError("construct the C2 model before tracing")
        coupling = tf.convert_to_tensor(self.coupling_matrix, DTYPE)
        if (
            coupling.shape.rank != 2
            or not coupling.shape.is_fully_defined()
            or coupling.shape[0] != coupling.shape[1]
            or int(coupling.shape[0]) < 1
        ):
            raise ValueError("coupling_matrix must be a static nonempty square matrix")
        if not bool(tf.reduce_all(tf.math.is_finite(coupling)).numpy()):
            raise ValueError("coupling_matrix must be finite")
        if not math.isfinite(float(self.sigma)) or float(self.sigma) <= 0.0:
            raise ValueError("sigma must be finite and positive")
        object.__setattr__(self, "coupling_matrix", coupling)

    def parameter_dim(self) -> int:
        return 2

    def state_dim(self) -> int:
        return int(self.coupling_matrix.shape[0])

    def observation_dim(self) -> int:
        return self.state_dim()

    def frozen_apf_measure_id(self) -> str:
        return MEASURE_ID

    def frozen_apf_score_backend_id(self) -> str:
        return SCORE_BACKEND_ID

    def transition_matrix(self, theta: tf.Tensor) -> tf.Tensor:
        parameters = tf.ensure_shape(tf.convert_to_tensor(theta, DTYPE), [2])
        return self.coupling_matrix + parameters[0] * tf.eye(self.state_dim(), dtype=DTYPE)

    def stationary_covariance_and_derivative(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        transition = self.transition_matrix(theta)
        dimension = self.state_dim()
        operator = tf.eye(dimension * dimension, dtype=DTYPE) - _kron_same(transition)
        process_covariance = tf.eye(dimension, dtype=DTYPE) * (float(self.sigma) ** 2)
        covariance = tf.reshape(
            tf.linalg.solve(operator, tf.reshape(process_covariance, [-1, 1])),
            [dimension, dimension],
        )
        covariance = 0.5 * (covariance + tf.transpose(covariance))
        derivative_rhs = (
            tf.linalg.matmul(covariance, transition, transpose_b=True)
            + tf.linalg.matmul(transition, covariance)
        )
        derivative = tf.reshape(
            tf.linalg.solve(operator, tf.reshape(derivative_rhs, [-1, 1])),
            [dimension, dimension],
        )
        derivative = 0.5 * (derivative + tf.transpose(derivative))
        return covariance, derivative

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        states = tf.ensure_shape(
            tf.convert_to_tensor(x0, DTYPE), [None, self.state_dim()]
        )
        covariance, _ = self.stationary_covariance_and_derivative(theta)
        chol = tf.linalg.cholesky(covariance)
        whitened = tf.transpose(
            tf.linalg.triangular_solve(chol, tf.transpose(states), lower=True)
        )
        log_det_chol = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        return -0.5 * (
            tf.cast(self.state_dim(), DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
            + tf.reduce_sum(tf.square(whitened), axis=1)
        ) - log_det_chol

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_previous: tf.Tensor,
        x_current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        previous = tf.ensure_shape(
            tf.convert_to_tensor(x_previous, DTYPE), [None, self.state_dim()]
        )
        current = tf.ensure_shape(
            tf.convert_to_tensor(x_current, DTYPE), [None, self.state_dim()]
        )
        transition = self.transition_matrix(theta)
        residual = current - tf.linalg.matmul(previous, transition, transpose_b=True)
        variance = tf.constant(float(self.sigma) ** 2, DTYPE)
        return -0.5 * (
            tf.cast(self.state_dim(), DTYPE)
            * tf.math.log(tf.constant(2.0 * math.pi, DTYPE) * variance)
            + tf.reduce_sum(tf.square(residual), axis=1) / variance
        )

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        parameters = tf.ensure_shape(tf.convert_to_tensor(theta, DTYPE), [2])
        states = tf.ensure_shape(
            tf.convert_to_tensor(state, DTYPE), [None, self.state_dim()]
        )
        observed = tf.ensure_shape(
            tf.convert_to_tensor(observation, DTYPE), [self.observation_dim()]
        )
        xi = parameters[1]
        return tf.reduce_sum(
            -0.5 * tf.constant(math.log(2.0 * math.pi), DTYPE)
            - xi
            - 0.5 * states
            - 0.5
            * tf.square(observed)[None, :]
            * tf.exp(-states - 2.0 * xi),
            axis=1,
        )

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, x0: tf.Tensor
    ) -> tf.Tensor:
        states = tf.ensure_shape(
            tf.convert_to_tensor(x0, DTYPE), [None, self.state_dim()]
        )
        covariance, derivative = self.stationary_covariance_and_derivative(theta)
        precision_derivative = tf.linalg.solve(covariance, derivative)
        trace_term = tf.linalg.trace(precision_derivative)
        precision_states = tf.transpose(
            tf.linalg.solve(covariance, tf.transpose(states))
        )
        quadratic_term = tf.einsum(
            "ni,ij,nj->n", precision_states, derivative, precision_states
        )
        gamma_score = -0.5 * trace_term + 0.5 * quadratic_term
        return tf.stack([gamma_score, tf.zeros_like(gamma_score)], axis=1)

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        x_previous: tf.Tensor,
        x_current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        previous = tf.ensure_shape(
            tf.convert_to_tensor(x_previous, DTYPE), [None, self.state_dim()]
        )
        current = tf.ensure_shape(
            tf.convert_to_tensor(x_current, DTYPE), [None, self.state_dim()]
        )
        transition = self.transition_matrix(theta)
        residual = current - tf.linalg.matmul(previous, transition, transpose_b=True)
        gamma_score = tf.reduce_sum(residual * previous, axis=1) / (
            float(self.sigma) ** 2
        )
        return tf.stack([gamma_score, tf.zeros_like(gamma_score)], axis=1)

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        parameters = tf.ensure_shape(tf.convert_to_tensor(theta, DTYPE), [2])
        states = tf.ensure_shape(
            tf.convert_to_tensor(state, DTYPE), [None, self.state_dim()]
        )
        observed = tf.ensure_shape(
            tf.convert_to_tensor(observation, DTYPE), [self.observation_dim()]
        )
        xi_score = tf.reduce_sum(
            -tf.ones_like(states)
            + tf.square(observed)[None, :] * tf.exp(-states - 2.0 * parameters[1]),
            axis=1,
        )
        return tf.stack([tf.zeros_like(xi_score), xi_score], axis=1)

    def stability_diagnostics(self, theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
        transition = self.transition_matrix(theta)
        eigenvalues = tf.linalg.eigvals(tf.cast(transition, tf.complex128))
        covariance, derivative = self.stationary_covariance_and_derivative(theta)
        residual = covariance - tf.linalg.matmul(
            transition,
            tf.linalg.matmul(covariance, transition, transpose_b=True),
        ) - tf.eye(self.state_dim(), dtype=DTYPE) * (float(self.sigma) ** 2)
        derivative_residual = (
            derivative
            - tf.linalg.matmul(
                transition,
                tf.linalg.matmul(derivative, transition, transpose_b=True),
            )
            - covariance @ tf.transpose(transition)
            - transition @ covariance
        )
        return {
            "spectral_radius": tf.reduce_max(tf.abs(eigenvalues)),
            "minimum_covariance_eigenvalue": tf.reduce_min(tf.linalg.eigvalsh(covariance)),
            "lyapunov_residual_max": tf.reduce_max(tf.abs(residual)),
            "derivative_lyapunov_residual_max": tf.reduce_max(
                tf.abs(derivative_residual)
            ),
        }

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "model_id": MODEL_ID,
            "parameter_names": ("gamma", "log_beta"),
            "state_dimension": self.state_dim(),
            "observation_dimension": self.observation_dim(),
            "coupling_matrix": self.coupling_matrix,
            "sigma": float(self.sigma),
            "initial_covariance": "stationary_discrete_lyapunov",
            "initial_covariance_derivative": "differentiated_discrete_lyapunov",
            "score_backend_id": SCORE_BACKEND_ID,
            "uses_autodiff_for_score": False,
        }


class _IndependentProposal(Protocol):
    time_index: int

    def sample_with_seed(
        self, particle_count: int, seed: tuple[int, int], *, jit_compile: bool
    ) -> Mapping[str, tf.Tensor]: ...

    def manifest_payload(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class FrozenGaussianStateProposal:
    """A parameter-independent multivariate Gaussian state proposal."""

    mean: tf.Tensor
    chol: tf.Tensor
    time_index: int
    family: str
    proposal_id: str = field(init=False)

    def __post_init__(self) -> None:
        mean = tf.reshape(tf.convert_to_tensor(self.mean, DTYPE), [-1])
        dimension = int(mean.shape[0])
        chol = tf.convert_to_tensor(self.chol, DTYPE)
        if chol.shape != (dimension, dimension):
            raise ValueError("Gaussian proposal chol shape mismatch")
        if not bool(
            tf.reduce_all(tf.math.is_finite(mean)).numpy()
            and tf.reduce_all(tf.math.is_finite(chol)).numpy()
        ):
            raise ValueError("Gaussian proposal parameters must be finite")
        if bool(tf.reduce_any(tf.linalg.diag_part(chol) <= 0.0).numpy()):
            raise ValueError("Gaussian proposal chol diagonal must be positive")
        upper = tf.linalg.band_part(chol, 0, -1) - tf.linalg.band_part(chol, 0, 0)
        if float(tf.reduce_max(tf.abs(upper)).numpy()) > 1e-12:
            raise ValueError("Gaussian proposal chol must be lower triangular")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "chol", chol)
        digest = hashlib.sha256()
        digest.update(str(self.family).encode("utf-8"))
        digest.update(str(int(self.time_index)).encode("ascii"))
        digest.update(bytes(tf.io.serialize_tensor(mean).numpy()))
        digest.update(bytes(tf.io.serialize_tensor(chol).numpy()))
        object.__setattr__(self, "proposal_id", digest.hexdigest())

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    def log_density(self, states: tf.Tensor) -> tf.Tensor:
        states = tf.ensure_shape(
            tf.convert_to_tensor(states, DTYPE), [None, self.dimension]
        )
        whitened = tf.transpose(
            tf.linalg.triangular_solve(
                self.chol, tf.transpose(states - self.mean[None, :]), lower=True
            )
        )
        return -0.5 * (
            tf.cast(self.dimension, DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
            + tf.reduce_sum(tf.square(whitened), axis=1)
        ) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.chol)))

    def sample_with_seed(
        self, particle_count: int, seed: tuple[int, int], *, jit_compile: bool
    ) -> Mapping[str, tf.Tensor]:
        count = int(particle_count)
        normal = tf.random.stateless_normal(
            [count, self.dimension], [int(seed[0]), int(seed[1])], dtype=DTYPE
        )

        return self.compiled_transform(count, jit_compile=jit_compile)(normal)

    def compiled_transform(self, particle_count: int, *, jit_compile: bool):
        count = int(particle_count)
        cache_key = (self.proposal_id, count, bool(jit_compile))
        cached = _GAUSSIAN_TRANSFORM_CACHE.get(cache_key)
        if cached is not None:
            return cached

        @tf.function(
            input_signature=[tf.TensorSpec([count, self.dimension], DTYPE)],
            jit_compile=bool(jit_compile),
            autograph=False,
        )
        def transform(standard_normal):
            states = self.mean[None, :] + tf.einsum(
                "ij,nj->ni", self.chol, standard_normal
            )
            return {
                "physical_points": states,
                "physical_log_density": self.log_density(states),
                "finite": tf.reduce_all(tf.math.is_finite(states)),
            }

        _GAUSSIAN_TRANSFORM_CACHE[cache_key] = transform
        return transform

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "family": self.family,
            "time_index": int(self.time_index),
            "dimension": self.dimension,
            "parameter_dependence": "none_after_compilation",
        }


@dataclass(frozen=True)
class FrozenC2ProposalCompilation:
    branch: PreparedFrozenProposalBranch
    compiler_id: str
    manifest: Mapping[str, object]
    proposal_diagnostics: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.branch, PreparedFrozenProposalBranch):
            raise TypeError("branch must be a PreparedFrozenProposalBranch")
        if len(str(self.compiler_id)) != 64:
            raise ValueError("compiler_id must be a SHA-256 digest")
        object.__setattr__(self, "manifest", dict(self.manifest))
        object.__setattr__(
            self, "proposal_diagnostics", tuple(dict(row) for row in self.proposal_diagnostics)
        )


def compile_c2_independent_proposal_branch(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    transition_proposals: Sequence[
        GaussianHermiteRetainedProposal | FrozenGaussianStateProposal
    ],
    particle_count: int,
    seed: int,
    family: str,
    jit_compile_sampler: bool = True,
) -> FrozenC2ProposalCompilation:
    proposals = tuple(transition_proposals)

    def sample_transition(time_index, _parents, transition_seed):
        proposal = proposals[time_index - 1]
        if int(proposal.time_index) != time_index:
            raise ValueError("transition proposal time index mismatch")
        if isinstance(proposal, GaussianHermiteRetainedProposal):
            random_inputs = stateless_proposal_random_inputs(
                proposal, particle_count, transition_seed
            )
            sampled = proposal.compiled_sampler(
                particle_count, jit_compile=bool(jit_compile_sampler)
            )(*random_inputs)
            diagnostics = {
                "time_index": time_index,
                "proposal_id": proposal.proposal_id,
                "family": family,
                "selected_polynomial_count": tf.reduce_sum(
                    tf.cast(sampled["selected_polynomial"], tf.int32)
                ),
                "polynomial_probability": sampled["polynomial_probability"],
                "maximum_inverse_cdf_residual": sampled[
                    "maximum_inverse_cdf_residual"
                ],
                "minimum_conditional_mass": sampled["minimum_conditional_mass"],
                "minimum_endpoint_margin": sampled["minimum_endpoint_margin"],
                "cdf_bracket_valid": sampled["cdf_bracket_valid"],
                "finite": sampled["finite"],
            }
        elif isinstance(proposal, FrozenGaussianStateProposal):
            sampled = proposal.sample_with_seed(
                particle_count, transition_seed, jit_compile=bool(jit_compile_sampler)
            )
            diagnostics = {
                "time_index": time_index,
                "proposal_id": proposal.proposal_id,
                "family": family,
                "finite": sampled["finite"],
            }
        else:
            raise TypeError("unsupported independent transition proposal")
        return sampled["physical_points"], sampled["physical_log_density"], diagnostics

    return _compile_c2_branch(
        model=model,
        observations=observations,
        theta_reference=theta_reference,
        particle_count=particle_count,
        seed=seed,
        family=family,
        sample_transition=sample_transition,
        transition_manifests=tuple(proposal.manifest_payload() for proposal in proposals),
    )


def transformed_student_proposals(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    nu: float,
) -> tuple[C2TransformedObservationStudentProposal, ...]:
    """Build one frozen log-square Student guide for each transition."""

    observed = tf.convert_to_tensor(observations, DTYPE)
    if observed.shape.rank != 2 or not observed.shape.is_fully_defined():
        raise ValueError("observations require static shape [time, observation]")
    transition = model.transition_matrix(theta_reference)
    process = tf.eye(model.state_dim(), dtype=DTYPE) * float(model.sigma) ** 2
    return tuple(
        build_c2_transformed_observation_student_proposal(
            transition_matrix=transition,
            process_covariance=process,
            observation=observed[time_index],
            theta_reference=theta_reference,
            nu=float(nu),
            time_index=time_index,
        )
        for time_index in range(1, int(observed.shape[0]))
    )


def compile_c2_transformed_student_proposal_branch(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    nu: float = 8.0,
    particle_count: int,
    seed: int,
    jit_compile_sampler: bool = True,
) -> FrozenC2ProposalCompilation:
    """Compile the defensive transformed Student guide as a single proposal."""

    proposals = transformed_student_proposals(
        model=model,
        observations=observations,
        theta_reference=theta_reference,
        nu=float(nu),
    )
    count = int(particle_count)

    def sample_transition(time_index, parents, transition_seed):
        proposal = proposals[time_index - 1]
        sampled = proposal.sample_with_seed(
            parents,
            count,
            (int(transition_seed[0]), int(transition_seed[1])),
            jit_compile=bool(jit_compile_sampler),
        )
        return sampled["physical_points"], sampled["physical_log_density"], {
            "time_index": time_index,
            "family": "transformed_observation_student",
            "proposal_id": proposal.proposal_id,
            "finite": sampled["finite"],
        }

    return _compile_c2_branch(
        model=model,
        observations=observations,
        theta_reference=theta_reference,
        particle_count=count,
        seed=seed,
        family="transformed_observation_student",
        sample_transition=sample_transition,
        transition_manifests=tuple(proposal.manifest_payload() for proposal in proposals),
    )


def compile_c2_dmis_proposal_branch(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    transition_proposals: Sequence[GaussianHermiteRetainedProposal],
    defensive_proposals: Sequence[C2TransformedObservationStudentProposal] | None = None,
    particle_count: int,
    seed: int,
    alpha: float = 0.5,
    nu: float = 8.0,
    jit_compile_sampler: bool = True,
) -> FrozenC2ProposalCompilation:
    """Compile an equal-bank deterministic-mixture C2 proposal branch.

    The returned branch is consumed by the shared ``FrozenProposalAPFProgram``.
    The TT proposal is the complete retained density (including its internal
    floor); the defensive density is the per-ancestor transformed Student law.
    """

    count = int(particle_count)
    if count < 4 or count % 2:
        raise ValueError("DMIS particle_count must be an even integer at least four")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("DMIS alpha must lie strictly between zero and one")
    proposals = tuple(transition_proposals)
    observed = tf.convert_to_tensor(observations, DTYPE)
    if observed.shape.rank != 2 or not observed.shape.is_fully_defined():
        raise ValueError("observations require static shape [time, observation]")
    horizon = int(observed.shape[0])
    if len(proposals) != horizon - 1:
        raise ValueError("retained TT proposal count must equal horizon - 1")
    for time_index, proposal in enumerate(proposals, start=1):
        if not isinstance(proposal, GaussianHermiteRetainedProposal):
            raise TypeError("DMIS retained proposals must be GaussianHermiteRetainedProposal")
        if int(proposal.time_index) != time_index:
            raise ValueError("retained TT proposal time index mismatch")
    defensive = (
        tuple(defensive_proposals)
        if defensive_proposals is not None
        else transformed_student_proposals(
            model=model,
            observations=observed,
            theta_reference=theta_reference,
            nu=float(nu),
        )
    )
    if len(defensive) != horizon - 1:
        raise ValueError("defensive proposal count must equal horizon - 1")
    for time_index, proposal in enumerate(defensive, start=1):
        if int(proposal.time_index) != time_index:
            raise ValueError("defensive proposal time index mismatch")

    if not tf.executing_eagerly():
        raise RuntimeError("compile C2 DMIS branches before tracing")
    theta_reference = tf.ensure_shape(
        tf.convert_to_tensor(theta_reference, DTYPE), [model.parameter_dim()]
    )
    stability = model.stability_diagnostics(theta_reference)
    if float(stability["spectral_radius"].numpy()) >= 1.0:
        raise ValueError("theta_reference is outside the stationary stability domain")
    covariance, _ = model.stationary_covariance_and_derivative(theta_reference)
    initial_chol = tf.linalg.cholesky(covariance)
    bank_count = count // 2
    initial_normal = tf.random.stateless_normal(
        [count, model.state_dim()], [int(seed), 1001], dtype=DTYPE
    )
    initial_states = tf.einsum("ij,nj->ni", initial_chol, initial_normal)
    initial_log_q = model.initial_log_density(theta_reference, initial_states)
    states = [initial_states]
    ancestors = []
    auxiliary_log_probabilities = []
    transition_log_q = []
    transition_base_mass = []
    proposal_diagnostics = []

    partial_branch = prepare_frozen_proposal_branch(
        observations=observed[:1],
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.zeros([0, count], tf.int32),
        auxiliary_log_probabilities=tf.zeros([0, count], DTYPE),
        transition_log_proposal_density=tf.zeros([0, count], DTYPE),
    )
    reference_log_weights = prepare_frozen_proposal_apf_program(
        model, partial_branch
    ).evaluate(theta_reference)["final_log_weights"]
    log_alpha = tf.math.log(tf.constant(float(alpha), DTYPE))
    log_one_minus_alpha = tf.math.log(tf.constant(1.0 - float(alpha), DTYPE))
    log_bank_count = tf.math.log(tf.cast(bank_count, DTYPE))

    for time_index in range(1, horizon):
        auxiliary_log = tf.identity(reference_log_weights)
        cdf = tf.math.cumsum(tf.exp(auxiliary_log))
        cdf = tf.concat([cdf[:-1], tf.ones([1], DTYPE)], axis=0)
        ancestor_tt = tf.searchsorted(
            cdf,
            tf.random.stateless_uniform(
                [bank_count], [int(seed), 2000 + 37 * time_index], dtype=DTYPE
            ),
            side="right",
            out_type=tf.int32,
        )
        ancestor_defensive = tf.searchsorted(
            cdf,
            tf.random.stateless_uniform(
                [bank_count], [int(seed), 2100 + 37 * time_index], dtype=DTYPE
            ),
            side="right",
            out_type=tf.int32,
        )
        parents_tt = tf.gather(states[-1], ancestor_tt)
        parents_defensive = tf.gather(states[-1], ancestor_defensive)

        tt_random_inputs = stateless_proposal_random_inputs(
            proposals[time_index - 1], bank_count, (int(seed), 3000 + 41 * time_index)
        )
        tt_sampled = proposals[time_index - 1].compiled_sampler(
            bank_count, jit_compile=bool(jit_compile_sampler)
        )(*tt_random_inputs)
        defensive_sampled = defensive[time_index - 1].sample_with_seed(
            parents_defensive,
            bank_count,
            (int(seed), 4000 + 43 * time_index),
            jit_compile=bool(jit_compile_sampler),
        )
        tt_states = tf.ensure_shape(
            tt_sampled["physical_points"], [bank_count, model.state_dim()]
        )
        defensive_states = tf.ensure_shape(
            defensive_sampled["physical_points"], [bank_count, model.state_dim()]
        )
        tt_log_q = tf.ensure_shape(
            tt_sampled["physical_log_density"], [bank_count]
        )
        defensive_log_q = tf.ensure_shape(
            defensive_sampled["physical_log_density"], [bank_count]
        )
        tt_log_defensive = defensive[time_index - 1].log_density(
            tt_states, parents_tt
        )
        defensive_log_tt = proposals[time_index - 1].physical_log_density(
            defensive_states
        )
        tt_mixture_log_q = tf.reduce_logsumexp(
            tf.stack(
                [log_one_minus_alpha + tt_log_q, log_alpha + tt_log_defensive], axis=1
            ),
            axis=1,
        )
        defensive_mixture_log_q = tf.reduce_logsumexp(
            tf.stack(
                [log_one_minus_alpha + defensive_log_tt, log_alpha + defensive_log_q],
                axis=1,
            ),
            axis=1,
        )
        states.append(tf.concat([tt_states, defensive_states], axis=0))
        ancestors.append(tf.concat([ancestor_tt, ancestor_defensive], axis=0))
        auxiliary_log_probabilities.append(auxiliary_log)
        transition_log_q.append(tf.concat([tt_mixture_log_q, defensive_mixture_log_q], axis=0))
        transition_base_mass.append(
            tf.concat(
                [
                    tf.fill([bank_count], log_one_minus_alpha - log_bank_count),
                    tf.fill([bank_count], log_alpha - log_bank_count),
                ],
                axis=0,
            )
        )
        proposal_diagnostics.append(
            {
                "time_index": time_index,
                "family": "tt_transformed_student_dmis",
                "alpha": tf.constant(float(alpha), DTYPE),
                "nu": tf.constant(float(nu), DTYPE),
                "tt_bank_count": tf.constant(bank_count, tf.int32),
                "defensive_bank_count": tf.constant(bank_count, tf.int32),
                "tt_finite": tt_sampled["finite"],
                "defensive_finite": defensive_sampled["finite"],
                "mixture_finite": tf.reduce_all(
                    tf.math.is_finite(tf.concat([tt_mixture_log_q, defensive_mixture_log_q], 0))
                ),
                "finite": tt_sampled["finite"]
                & defensive_sampled["finite"]
                & tf.reduce_all(
                    tf.math.is_finite(
                        tf.concat([tt_mixture_log_q, defensive_mixture_log_q], 0)
                    )
                ),
            }
        )

        partial_branch = prepare_frozen_proposal_branch(
            observations=observed[: time_index + 1],
            states=tf.stack(states),
            initial_log_proposal_density=initial_log_q,
            ancestors=tf.stack(ancestors),
            auxiliary_log_probabilities=tf.stack(auxiliary_log_probabilities),
            transition_log_proposal_density=tf.stack(transition_log_q),
            transition_log_base_mass=tf.stack(transition_base_mass),
        )
        reference_log_weights = prepare_frozen_proposal_apf_program(
            model, partial_branch
        ).evaluate(theta_reference)["final_log_weights"]

    manifest = {
        "compiler_route_id": "c2_deterministic_equal_bank_tt_student_dmis_v1",
        "compiler_classification": "extension_or_invention",
        "family": "tt_transformed_student_dmis",
        "alpha": float(alpha),
        "nu": float(nu),
        "particle_count": count,
        "tt_bank_count": bank_count,
        "defensive_bank_count": bank_count,
        "seed": int(seed),
        "branch_id": partial_branch.branch_id,
        "model": model.manifest_payload(),
        "retained_proposals": tuple(proposal.manifest_payload() for proposal in proposals),
        "defensive_proposals": tuple(proposal.manifest_payload() for proposal in defensive),
        "auxiliary_law": "generic_apf_prefix_weights_at_theta_reference_shared_for_banks",
        "base_mass_policy": "component_weight_over_bank_count",
        "complete_mixture_density": True,
        "runtime_branch_parameter_dependence": "none",
        "exact_pseudo_marginal_claimed": False,
        "status": "scout_not_truth",
        "stability": stability,
    }
    return FrozenC2ProposalCompilation(
        branch=partial_branch,
        compiler_id=_compilation_fingerprint(manifest),
        manifest=manifest,
        proposal_diagnostics=tuple(proposal_diagnostics),
    )


def compile_c2_bootstrap_proposal_branch(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    particle_count: int,
    seed: int,
    jit_compile_sampler: bool = True,
) -> FrozenC2ProposalCompilation:
    transition = model.transition_matrix(theta_reference)
    sigma = tf.constant(float(model.sigma), DTYPE)
    state_dimension = model.state_dim()
    count = int(particle_count)

    @tf.function(
        input_signature=[
            tf.TensorSpec([count, state_dimension], DTYPE),
            tf.TensorSpec([count, state_dimension], DTYPE),
        ],
        jit_compile=bool(jit_compile_sampler),
        autograph=False,
    )
    def transform(parent_states, standard_normal):
        states = tf.linalg.matmul(parent_states, transition, transpose_b=True) + sigma * standard_normal
        log_q = model.transition_log_density(
            theta_reference, parent_states, states, 1
        )
        return states, log_q

    def sample_transition(time_index, parents, transition_seed):
        noise = tf.random.stateless_normal(
            [count, state_dimension],
            [int(transition_seed[0]), int(transition_seed[1])],
            dtype=DTYPE,
        )

        states, log_q = transform(parents, noise)
        return states, log_q, {
            "time_index": time_index,
            "family": "bootstrap_conditional",
            "finite": tf.reduce_all(tf.math.is_finite(states)),
        }

    horizon = int(tf.convert_to_tensor(observations).shape[0])
    return _compile_c2_branch(
        model=model,
        observations=observations,
        theta_reference=theta_reference,
        particle_count=particle_count,
        seed=seed,
        family="bootstrap_conditional",
        sample_transition=sample_transition,
        transition_manifests=tuple(
            {
                "family": "bootstrap_conditional",
                "time_index": time_index,
                "theta_reference": tf.convert_to_tensor(theta_reference, DTYPE),
            }
            for time_index in range(1, horizon)
        ),
    )


def stationary_gaussian_proposals(
    model: C2StochasticVolatilityFrozenAPFModel,
    theta_reference: tf.Tensor,
    horizon: int,
) -> tuple[FrozenGaussianStateProposal, ...]:
    covariance, _ = model.stationary_covariance_and_derivative(theta_reference)
    chol = tf.linalg.cholesky(covariance)
    mean = tf.zeros([model.state_dim()], DTYPE)
    return tuple(
        FrozenGaussianStateProposal(
            mean=mean,
            chol=chol,
            time_index=time_index,
            family="stationary_independence",
        )
        for time_index in range(1, int(horizon))
    )


def _compile_c2_branch(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    particle_count: int,
    seed: int,
    family: str,
    sample_transition,
    transition_manifests: Sequence[Mapping[str, object]],
) -> FrozenC2ProposalCompilation:
    if not tf.executing_eagerly():
        raise RuntimeError("compile C2 proposal branches before tracing")
    observations = tf.convert_to_tensor(observations, DTYPE)
    theta_reference = tf.ensure_shape(
        tf.convert_to_tensor(theta_reference, DTYPE), [model.parameter_dim()]
    )
    if observations.shape.rank != 2 or not observations.shape.is_fully_defined():
        raise ValueError("observations require static shape [time, observation]")
    horizon = int(observations.shape[0])
    if horizon < 1 or observations.shape[1] != model.observation_dim():
        raise ValueError("observation shape does not match the C2 model")
    if len(tuple(transition_manifests)) != horizon - 1:
        raise ValueError("transition proposal count must equal horizon - 1")
    count = int(particle_count)
    if count < 2:
        raise ValueError("particle_count must be at least two")
    stability = model.stability_diagnostics(theta_reference)
    if float(stability["spectral_radius"].numpy()) >= 1.0:
        raise ValueError("theta_reference is outside the stationary stability domain")
    if float(stability["minimum_covariance_eigenvalue"].numpy()) <= 0.0:
        raise ValueError("theta_reference stationary covariance is not positive definite")

    covariance, _ = model.stationary_covariance_and_derivative(theta_reference)
    initial_chol = tf.linalg.cholesky(covariance)
    initial_normal = tf.random.stateless_normal(
        [count, model.state_dim()], [int(seed), 1001], dtype=DTYPE
    )
    initial_states = tf.einsum("ij,nj->ni", initial_chol, initial_normal)
    initial_log_q = model.initial_log_density(theta_reference, initial_states)
    states = [initial_states]
    ancestors = []
    auxiliary_log_probabilities = []
    transition_log_q = []
    proposal_diagnostics = []

    partial_branch = prepare_frozen_proposal_branch(
        observations=observations[:1],
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.zeros([0, count], tf.int32),
        auxiliary_log_probabilities=tf.zeros([0, count], DTYPE),
        transition_log_proposal_density=tf.zeros([0, count], DTYPE),
    )
    reference_log_weights = prepare_frozen_proposal_apf_program(
        model, partial_branch
    ).evaluate(theta_reference)["final_log_weights"]

    for time_index in range(1, horizon):
        auxiliary_log = tf.identity(reference_log_weights)
        categorical_uniforms = tf.random.stateless_uniform(
            [count], [int(seed), 2000 + 17 * time_index], dtype=DTYPE
        )
        cdf = tf.math.cumsum(tf.exp(auxiliary_log))
        cdf = tf.concat([cdf[:-1], tf.ones([1], DTYPE)], axis=0)
        ancestor = tf.searchsorted(
            cdf, categorical_uniforms, side="right", out_type=tf.int32
        )
        parents = tf.gather(states[-1], ancestor)
        sampled_states, sampled_log_q, sampled_diagnostics = sample_transition(
            time_index,
            parents,
            (int(seed), 3000 + 31 * time_index),
        )
        sampled_states = tf.ensure_shape(
            tf.convert_to_tensor(sampled_states, DTYPE),
            [count, model.state_dim()],
        )
        sampled_log_q = tf.ensure_shape(
            tf.convert_to_tensor(sampled_log_q, DTYPE), [count]
        )
        if not bool(
            tf.reduce_all(tf.math.is_finite(sampled_states)).numpy()
            and tf.reduce_all(tf.math.is_finite(sampled_log_q)).numpy()
        ):
            raise ValueError(f"non-finite proposal output at time {time_index}")
        if "cdf_bracket_valid" in sampled_diagnostics and not bool(
            sampled_diagnostics["cdf_bracket_valid"].numpy()
        ):
            raise ValueError(f"invalid Hermite CDF bracket at time {time_index}")
        if "finite" in sampled_diagnostics and not bool(
            sampled_diagnostics["finite"].numpy()
        ):
            raise ValueError(f"invalid proposal diagnostic at time {time_index}")

        states.append(sampled_states)
        ancestors.append(ancestor)
        auxiliary_log_probabilities.append(auxiliary_log)
        transition_log_q.append(sampled_log_q)
        proposal_diagnostics.append(dict(sampled_diagnostics))

        partial_branch = prepare_frozen_proposal_branch(
            observations=observations[: time_index + 1],
            states=tf.stack(states),
            initial_log_proposal_density=initial_log_q,
            ancestors=tf.stack(ancestors),
            auxiliary_log_probabilities=tf.stack(auxiliary_log_probabilities),
            transition_log_proposal_density=tf.stack(transition_log_q),
        )
        reference_log_weights = prepare_frozen_proposal_apf_program(
            model, partial_branch
        ).evaluate(theta_reference)["final_log_weights"]

    branch = partial_branch
    manifest = {
        "compiler_route_id": COMPILER_ID,
        "compiler_classification": COMPILER_CLASSIFICATION,
        "family": family,
        "initial_proposal_id": INITIAL_PROPOSAL_ID,
        "theta_reference": theta_reference,
        "seed": int(seed),
        "particle_count": count,
        "horizon": horizon,
        "branch_id": branch.branch_id,
        "model": model.manifest_payload(),
        "transition_proposals": tuple(transition_manifests),
        "auxiliary_law": "generic_apf_prefix_weights_at_theta_reference",
        "runtime_branch_parameter_dependence": "none",
        "jit_compile_default": True,
        "exact_pseudo_marginal_claimed": False,
        "stability": stability,
    }
    compiler_id = _compilation_fingerprint(manifest)
    return FrozenC2ProposalCompilation(
        branch=branch,
        compiler_id=compiler_id,
        manifest=manifest,
        proposal_diagnostics=tuple(proposal_diagnostics),
    )


def _compilation_fingerprint(manifest: Mapping[str, object]) -> str:
    digest = hashlib.sha256()

    def update(value: object) -> None:
        if isinstance(value, tf.Tensor):
            digest.update(value.dtype.name.encode("ascii"))
            digest.update(bytes(tf.io.serialize_tensor(value).numpy()))
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                update(str(key))
                update(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                update(item)
        elif value is None:
            digest.update(b"none")
        elif isinstance(value, (bool, int, float, str)):
            digest.update(
                json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
            )
        else:
            raise TypeError(f"unsupported compiler identity type: {type(value).__name__}")

    update(manifest)
    return digest.hexdigest()


__all__ = [
    "C2StochasticVolatilityFrozenAPFModel",
    "COMPILER_CLASSIFICATION",
    "COMPILER_ID",
    "FrozenC2ProposalCompilation",
    "FrozenGaussianStateProposal",
    "MODEL_ID",
    "compile_c2_bootstrap_proposal_branch",
    "compile_c2_dmis_proposal_branch",
    "compile_c2_independent_proposal_branch",
    "compile_c2_transformed_student_proposal_branch",
    "stationary_gaussian_proposals",
    "transformed_student_proposals",
]
