"""C2 transformed-observation Student proposal mechanics.

This module is a frozen, data-guided proposal component.  It is an
``extension_or_invention`` diagnostic route: the raw C2 observation density
remains the target used by the APF evaluator, while the log-square Gaussian
closure is used only to construct a full-support proposal geometry.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.fixed_signature_tf import fixed_signature_function
from bayesfilter.ops.stateless_gamma_tf import philox_gamma_float64
from bayesfilter.ops.stateless_random_tf import philox_normal_float64

DTYPE = tf.float64
ROUTE_ID = "c2_transformed_log_square_student_defense_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
GUIDE_CONVENTION_ID = "transformed_log_square_moment_ukf_v1"
RAW_ZERO_GAIN_ID = "c2_raw_observation_zero_cross_covariance_v1"
EULER_GAMMA = 0.5772156649015328606
LOG_CHI_SQUARE_MEAN = -EULER_GAMMA - math.log(2.0)
LOG_CHI_SQUARE_VARIANCE = math.pi**2 / 2.0
_TRANSFORM_CACHE = OrderedDict()


def _cache_program(key, program):
    _TRANSFORM_CACHE[key] = program
    _TRANSFORM_CACHE.move_to_end(key)
    if len(_TRANSFORM_CACHE) > 16:
        _TRANSFORM_CACHE.popitem(last=False)
    return program


@fixed_signature_function(floating_dtype=DTYPE)
def _transformed_observation_core(observed, parameters):
    return tf.math.log(tf.square(observed)) - 2. * parameters[1] - tf.constant(LOG_CHI_SQUARE_MEAN, DTYPE)


@lru_cache(maxsize=16)
def _geometry_program(dimension, nu):
    @tf.function(input_signature=[tf.TensorSpec([dimension, dimension], DTYPE)],
                 jit_compile=True, autograph=False)
    def evaluate(process):
        innovation = process + tf.eye(dimension, dtype=DTYPE) * tf.constant(LOG_CHI_SQUARE_VARIANCE, DTYPE)
        gain = tf.transpose(tf.linalg.solve(innovation, tf.transpose(process)))
        posterior = process - tf.linalg.matmul(gain, process)
        posterior = .5 * (posterior + tf.transpose(posterior))
        # Preserve the existing tf.cast(Python float) scale conversion.
        scale = tf.cast((nu - 2.) / nu, DTYPE) * posterior
        return innovation, gain, posterior, scale, tf.linalg.cholesky(scale), tf.reduce_max(tf.abs(gain @ innovation - process))

    return evaluate


def transformed_log_square_observation(
    observations: tf.Tensor, theta_reference: tf.Tensor
) -> tf.Tensor:
    """Return ``log(y**2) - 2*log(beta) - E[log(E**2)]`` at setup time."""

    observed = tf.convert_to_tensor(observations, DTYPE)
    if observed.shape.rank != 1 or not observed.shape.is_fully_defined():
        raise ValueError("observations must have static shape [observation]")
    parameters = tf.ensure_shape(tf.convert_to_tensor(theta_reference, DTYPE), [2])
    if bool(tf.reduce_any(tf.equal(observed, 0.0)).numpy()):
        raise ValueError("the transformed C2 guide requires nonzero observations")
    transformed = _transformed_observation_core(observed, parameters)
    if not bool(tf.reduce_all(tf.math.is_finite(transformed)).numpy()):
        raise ValueError("transformed observations must be finite")
    return transformed


def raw_observation_zero_cross_covariance(
    state_dimension: int, *, beta: float = 1.0
) -> tf.Tensor:
    """Analytic negative control for the signed raw C2 observation.

    Conditional on the state, the observation has zero mean.  Therefore the
    population state/raw-observation cross-covariance is the zero matrix,
    independently of the finite positive value of ``beta``.
    """

    dimension = int(state_dimension)
    if dimension < 1 or not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("state_dimension and beta must be finite and positive")
    return tf.zeros([dimension, dimension], DTYPE)


def student_scale_from_covariance(
    covariance: tf.Tensor, nu: float
) -> tf.Tensor:
    """Convert a desired covariance to the multivariate Student scale."""

    covariance = tf.convert_to_tensor(covariance, DTYPE)
    if covariance.shape.rank != 2 or not covariance.shape.is_fully_defined():
        raise ValueError("covariance must have static shape [state, state]")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not math.isfinite(float(nu)) or float(nu) <= 2.0:
        raise ValueError("finite Student degrees of freedom must be greater than two")
    _check_spd(covariance, "covariance")
    return tf.cast((float(nu) - 2.0) / float(nu), DTYPE) * covariance


@dataclass(frozen=True)
class C2TransformedObservationStudentProposal:
    """Per-transition full-support Student proposal conditioned on parents."""

    transition_matrix: tf.Tensor
    process_covariance: tf.Tensor
    transformed_observation: tf.Tensor
    nu: float
    time_index: int
    theta_reference: tf.Tensor
    proposal_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not tf.executing_eagerly():
            raise RuntimeError("construct transformed proposals before tracing")
        transition = tf.convert_to_tensor(self.transition_matrix, DTYPE)
        process = tf.convert_to_tensor(self.process_covariance, DTYPE)
        observed = tf.convert_to_tensor(self.transformed_observation, DTYPE)
        theta = tf.ensure_shape(tf.convert_to_tensor(self.theta_reference, DTYPE), [2])
        if (
            transition.shape.rank != 2
            or not transition.shape.is_fully_defined()
            or transition.shape[0] != transition.shape[1]
            or transition.shape[0] < 1
        ):
            raise ValueError("transition_matrix must be a static square matrix")
        dimension = int(transition.shape[0])
        if process.shape != (dimension, dimension):
            raise ValueError("process_covariance shape mismatch")
        if observed.shape != (dimension,):
            raise ValueError("transformed_observation shape mismatch")
        _check_spd(process, "process_covariance")
        if not bool(tf.reduce_all(tf.math.is_finite(transition)).numpy()):
            raise ValueError("transition_matrix must be finite")
        if not bool(tf.reduce_all(tf.math.is_finite(observed)).numpy()):
            raise ValueError("transformed_observation must be finite")
        if not math.isfinite(float(self.nu)) or float(self.nu) <= 2.0:
            raise ValueError("proposal nu must be finite and greater than two")
        if int(self.time_index) < 1:
            raise ValueError("transformed proposals require time_index >= 1")

        innovation_covariance, gain, posterior_covariance, scale, chol, residual = _geometry_program(
            dimension, float(self.nu))(process)
        _check_spd(innovation_covariance, "innovation_covariance")
        _check_spd(posterior_covariance, "posterior_covariance")
        residual_max = float(residual.numpy())
        if residual_max > 2e-11:
            raise ValueError("transformed-guide gain solve residual is too large")

        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "process_covariance", process)
        object.__setattr__(self, "transformed_observation", observed)
        object.__setattr__(self, "theta_reference", theta)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "posterior_covariance", posterior_covariance)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "chol", chol)
        object.__setattr__(self, "solve_residual_max", residual_max)

        digest = hashlib.sha256()
        for value in (
            ROUTE_ID,
            GUIDE_CONVENTION_ID,
            self.time_index,
            self.nu,
            transition,
            process,
            observed,
            theta,
        ):
            _hash_value(digest, value)
        object.__setattr__(self, "proposal_id", digest.hexdigest())

    @property
    def dimension(self) -> int:
        return int(self.transition_matrix.shape[0])

    def conditional_mean(self, parent_states: tf.Tensor) -> tf.Tensor:
        parents = tf.ensure_shape(
            tf.convert_to_tensor(parent_states, DTYPE), [None, self.dimension]
        )
        if tf.executing_eagerly():
            return self._compiled_evaluation(int(parents.shape[0]), "mean")(parents)
        prior_mean = tf.linalg.matmul(
            parents, self.transition_matrix, transpose_b=True
        )
        innovation = self.transformed_observation[None, :] - prior_mean
        return prior_mean + tf.linalg.matmul(innovation, self.gain, transpose_b=True)

    def log_density(self, states: tf.Tensor, parent_states: tf.Tensor) -> tf.Tensor:
        values = tf.ensure_shape(
            tf.convert_to_tensor(states, DTYPE), [None, self.dimension]
        )
        if tf.executing_eagerly():
            return self._compiled_evaluation(int(values.shape[0]), "density")(
                values, tf.convert_to_tensor(parent_states, DTYPE))
        means = self.conditional_mean(parent_states)
        centered = values - means
        whitened = tf.transpose(
            tf.linalg.triangular_solve(
                self.chol, tf.transpose(centered), lower=True
            )
        )
        quadratic = tf.reduce_sum(tf.square(whitened), axis=1)
        dimension = tf.cast(self.dimension, DTYPE)
        nu = tf.constant(float(self.nu), DTYPE)
        normalizer = (
            tf.math.lgamma(0.5 * (nu + dimension))
            - tf.math.lgamma(0.5 * nu)
            - 0.5 * dimension * tf.math.log(nu * tf.constant(math.pi, DTYPE))
            - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.chol)))
        )
        return normalizer - 0.5 * (nu + dimension) * tf.math.log1p(quadratic / nu)

    def _compiled_evaluation(self, count, kind):
        key = (self.proposal_id, count, kind)
        if key in _TRANSFORM_CACHE:
            _TRANSFORM_CACHE.move_to_end(key)
            return _TRANSFORM_CACHE[key]
        specification = tf.TensorSpec([count, self.dimension], DTYPE)
        function = self.conditional_mean if kind == "mean" else self.log_density
        signature = [specification] if kind == "mean" else [specification, specification]
        return _cache_program(key, tf.function(function, input_signature=signature,
                                               jit_compile=True, autograph=False))

    def compiled_transform(self, particle_count: int, *, jit_compile: bool = True):
        count = int(particle_count)
        if count < 1:
            raise ValueError("particle_count must be positive")
        key = (self.proposal_id, count, bool(jit_compile))
        cached = _TRANSFORM_CACHE.get(key)
        if cached is not None:
            _TRANSFORM_CACHE.move_to_end(key)
            return cached

        @tf.function(
            input_signature=[
                tf.TensorSpec([count, self.dimension], DTYPE),
                tf.TensorSpec([count, self.dimension], DTYPE),
                tf.TensorSpec([count], DTYPE),
            ],
            jit_compile=bool(jit_compile),
            autograph=False,
        )
        def transform(parent_states, standard_normal, chi_square):
            means = self.conditional_mean(parent_states)
            whitened = standard_normal / tf.sqrt(
                chi_square[:, None] / tf.constant(float(self.nu), DTYPE)
            )
            states = means + tf.einsum("ij,nj->ni", self.chol, whitened)
            return {
                "physical_points": states,
                "physical_log_density": self.log_density(states, parent_states),
                "finite": tf.reduce_all(tf.math.is_finite(states))
                & tf.reduce_all(tf.math.is_finite(chi_square)),
            }

        return _cache_program(key, transform)

    def compiled_seeded_sampler(self, particle_count: int, *, jit_compile: bool = True):
        count = int(particle_count)
        transform = self.compiled_transform(count, jit_compile=jit_compile).python_function
        key = (self.proposal_id, count, bool(jit_compile), "seeded")
        if key in _TRANSFORM_CACHE:
            _TRANSFORM_CACHE.move_to_end(key)
            return _TRANSFORM_CACHE[key]

        @tf.function(input_signature=[tf.TensorSpec([count, self.dimension], DTYPE),
            tf.TensorSpec([2], tf.int64)], jit_compile=bool(jit_compile), autograph=False)
        def sample(parents, seed):
            normal = philox_normal_float64([count, self.dimension], seed)
            chi_square = philox_gamma_float64([count], seed + [0, 1],
                tf.constant(float(self.nu) / 2., DTYPE), tf.constant(.5, DTYPE))
            return transform(parents, normal, chi_square)

        return _cache_program(key, sample)

    def sample_with_seed(
        self,
        parent_states: tf.Tensor,
        particle_count: int,
        seed: tuple[int, int],
        *,
        jit_compile: bool = True,
    ) -> Mapping[str, tf.Tensor]:
        count = int(particle_count)
        parents = tf.ensure_shape(
            tf.convert_to_tensor(parent_states, DTYPE), [count, self.dimension]
        )
        first, second = int(seed[0]), int(seed[1])
        if not -2**63 <= first < 2**63 or not -2**63 <= second < 2**63 - 1:
            raise ValueError("proposal seed and offsets must fit int64")
        return self.compiled_seeded_sampler(count, jit_compile=jit_compile)(
            parents, tf.constant([first, second], tf.int64))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "guide_convention_id": GUIDE_CONVENTION_ID,
            "raw_zero_gain_negative_control_id": RAW_ZERO_GAIN_ID,
            "proposal_id": self.proposal_id,
            "time_index": int(self.time_index),
            "dimension": self.dimension,
            "nu": float(self.nu),
            "theta_reference": self.theta_reference,
            "transformed_observation": self.transformed_observation,
            "log_chi_square_mean": LOG_CHI_SQUARE_MEAN,
            "log_chi_square_variance": LOG_CHI_SQUARE_VARIANCE,
            "gain_solve_residual_max": self.solve_residual_max,
            "proposal_parameter_dependence": "none_after_compilation",
            "full_support": True,
            "score_claim": "proposal_is_frozen_and_excluded_from_runtime_score",
            "status": "scout_not_truth",
        }


def build_c2_transformed_observation_student_proposal(
    *,
    transition_matrix: tf.Tensor,
    process_covariance: tf.Tensor,
    observation: tf.Tensor,
    theta_reference: tf.Tensor,
    nu: float,
    time_index: int,
) -> C2TransformedObservationStudentProposal:
    transformed = transformed_log_square_observation(observation, theta_reference)
    return C2TransformedObservationStudentProposal(
        transition_matrix=transition_matrix,
        process_covariance=process_covariance,
        transformed_observation=transformed,
        theta_reference=theta_reference,
        nu=float(nu),
        time_index=int(time_index),
    )


def _check_spd(matrix: tf.Tensor, name: str) -> None:
    matrix = tf.convert_to_tensor(matrix, DTYPE)
    scale = tf.maximum(tf.reduce_max(tf.abs(matrix)), tf.constant(1.0, DTYPE))
    symmetry = tf.reduce_max(tf.abs(matrix - tf.transpose(matrix)))
    if float(symmetry.numpy()) > 2e-11 * float(scale.numpy()):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = tf.linalg.eigvalsh(0.5 * (matrix + tf.transpose(matrix)))
    if float(tf.reduce_min(eigenvalues).numpy()) <= 2e-12 * float(scale.numpy()):
        raise ValueError(f"{name} must be positive definite")


def _hash_value(digest: object, value: object) -> None:
    if isinstance(value, tf.Tensor):
        digest.update(value.dtype.name.encode("ascii"))
        digest.update(repr(value.shape.as_list()).encode("ascii"))
        digest.update(tf.io.serialize_tensor(value).numpy())
    else:
        digest.update(repr(value).encode("utf-8"))


__all__ = [
    "DTYPE",
    "EULER_GAMMA",
    "GUIDE_CONVENTION_ID",
    "LOG_CHI_SQUARE_MEAN",
    "LOG_CHI_SQUARE_VARIANCE",
    "RAW_ZERO_GAIN_ID",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "C2TransformedObservationStudentProposal",
    "build_c2_transformed_observation_student_proposal",
    "raw_observation_zero_cross_covariance",
    "student_scale_from_covariance",
    "transformed_log_square_observation",
]
