"""GPU-oriented production/default target for batched LEDH-PFPF-OT.

The module path is historical.  Repository governance now treats this
GPU-oriented LEDH-PFPF-OT TF32 route as the default production target for DPF
transport work, while public API exposure, posterior correctness, and HMC
readiness remain separately gated.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

from bayesfilter.highdim.transport_chunk_policy import resolve_transport_chunks

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    AnnealedTransportWarmstartStateTF,
    DEFAULT_STREAMING_CHUNK_SIZE,
    _maybe_stop,
    _clip_transport_upstream_gradient,
    _filterflow_custom_gradient_transport_matrix,
    _filterflow_exact_transport_matrix,
    _filterflow_epsilon_start,
    _filterflow_manual_dense_finite_transport_matrix_stopped_scale_keys,
    _filterflow_manual_streaming_blockwise_vjp_finite_transport_stopped_scale_keys,
    _filterflow_manual_streaming_finite_transport_stopped_scale_keys,
    _filterflow_manual_streaming_finite_transport_total_vjp,
    _filterflow_scale,
    _filterflow_streaming_transport,
    _transport_ad_stop_scale,
    _validate_transport_ad_mode,
)


DEFAULT_DTYPE = tf.float32
DTYPE = DEFAULT_DTYPE
DEFAULT_TF32_MODE = "enabled"
DEFAULT_PRECISION_POLICY = "production_ledh_pfpf_ot_gpu_tf32"
DEFAULT_EXECUTION_TARGET = "gpu"
DEFAULT_ALGORITHM_TARGET = "ledh_pfpf_ot_tf32"
DEFAULT_TARGET_STATUS = "production_default_by_owner_directive"
DEFAULT_ROUTE_ACCEPTANCE = "accepted_default_use_whenever_possible"
DEFAULT_ROUTE_RATIONALE = (
    "streaming GPU TF32 LEDH-PFPF-OT avoids dense transport/history storage "
    "for large-particle DPF transport while preserving explicit reference "
    "and fallback arms"
)
MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE = (
    "manual_dense_finite_sinkhorn_stopped_scale_keys"
)
MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE = (
    "manual_streaming_finite_sinkhorn_stopped_scale_keys"
)
MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE = (
    "manual_streaming_blockwise_vjp_finite_sinkhorn_stopped_scale_keys"
)

SCALAR_PARITY_ATOL = 1.0e-10
SCALAR_PARITY_RTOL = 1.0e-10
TRANSPORT_PARITY_ATOL = 1.0e-8
TRANSPORT_PARITY_RTOL = 1.0e-8

FORBIDDEN_RNG_SOURCE_TOKENS = (
    "tf.random",
    "tensorflow.random",
    "np.random",
    "numpy.random",
    "random.",
)


def _sync_transport_dtype() -> None:
    """Keep imported transport helper globals aligned with this experimental lane."""

    annealed_transport_tf.DTYPE = DTYPE


def _manual_dense_finite_steps(max_iterations: int | tf.Tensor) -> int:
    value = tf.get_static_value(max_iterations)
    if value is None:
        raise ValueError("manual dense finite route requires static max_iterations")
    max_iterations_value = int(value)
    if max_iterations_value <= 0:
        raise ValueError("manual dense finite route requires positive max_iterations")
    # The raw Sinkhorn loop initializes its potentials before the loop and uses
    # `i < max_iter - 1`, so its maximum number of annealing updates is one less
    # than the public iteration cap. The manual JVP must replay that same map.
    return max_iterations_value - 1


def precision_policy_metadata() -> dict[str, Any]:
    """Return the default precision policy for the batched DPF route."""

    return {
        "precision_default_policy": DEFAULT_PRECISION_POLICY,
        "default_execution_target": DEFAULT_EXECUTION_TARGET,
        "default_algorithm_target": DEFAULT_ALGORITHM_TARGET,
        "default_target_status": DEFAULT_TARGET_STATUS,
        "default_route_acceptance": DEFAULT_ROUTE_ACCEPTANCE,
        "default_route_guidance": (
            "use for BayesFilter DPF LEDH-PFPF-OT work whenever GPU execution "
            "and the streaming fixed-branch contract are applicable"
        ),
        "default_route_rationale": DEFAULT_ROUTE_RATIONALE,
        "default_dtype": DEFAULT_DTYPE.name,
        "active_dtype": DTYPE.name,
        "default_tf32_mode": DEFAULT_TF32_MODE,
        "fp64_reference_requires_explicit_dtype": True,
        "scope": "production_ledh_pfpf_ot_gpu_tf32_default_lane",
        "historical_module_path": "experiments/dpf_implementation",
        "public_api_exposure": "separately_gated",
    }

NONCLAIMS = (
    "production/default target by owner directive",
    "public API exposure remains separately gated",
    "no public API claim",
    "no categorical particle-filter gradient claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no statistical superiority claim",
)


@dataclass(frozen=True)
class BatchedLEDHPFPFOTShapeContract:
    """Static tensor dimensions for a fixed-branch batched LEDH-PFPF-OT run."""

    batch_size: int
    time_steps: int
    num_particles: int
    state_dim: int
    observation_dim: int
    parameter_dim: int
    scalar_parity_atol: float = SCALAR_PARITY_ATOL
    scalar_parity_rtol: float = SCALAR_PARITY_RTOL
    transport_parity_atol: float = TRANSPORT_PARITY_ATOL
    transport_parity_rtol: float = TRANSPORT_PARITY_RTOL
    nonclaims: tuple[str, ...] = NONCLAIMS

    @property
    def value_shape(self) -> tuple[int]:
        return (self.batch_size,)

    @property
    def score_shape(self) -> tuple[int, int]:
        return (self.batch_size, self.parameter_dim)

    @property
    def particle_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.num_particles, self.state_dim)

    @property
    def fixed_pre_flow_shape(self) -> tuple[int, int, int, int]:
        return (
            self.batch_size,
            self.time_steps,
            self.num_particles,
            self.state_dim,
        )

    @property
    def fixed_mask_shape(self) -> tuple[int, int]:
        return (self.batch_size, self.time_steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "time_steps": self.time_steps,
            "num_particles": self.num_particles,
            "state_dim": self.state_dim,
            "observation_dim": self.observation_dim,
            "parameter_dim": self.parameter_dim,
            "scalar_parity_atol": self.scalar_parity_atol,
            "scalar_parity_rtol": self.scalar_parity_rtol,
            "transport_parity_atol": self.transport_parity_atol,
            "transport_parity_rtol": self.transport_parity_rtol,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class BatchedLEDHPFPFOTFixedInputs:
    """Deterministic inputs for a fixed-branch batched LEDH-PFPF-OT fixture."""

    theta_batch: tf.Tensor
    observations: tf.Tensor
    initial_particles: tf.Tensor
    pre_flow_particles: tf.Tensor
    fixed_resampling_mask: tf.Tensor
    initial_log_weights: tf.Tensor | None = None
    shape_contract: BatchedLEDHPFPFOTShapeContract = field(init=False)

    def __post_init__(self) -> None:
        theta_batch = _to_float_tensor(self.theta_batch, "theta_batch")
        observations = _to_float_tensor(self.observations, "observations")
        initial_particles = _to_float_tensor(self.initial_particles, "initial_particles")
        pre_flow_particles = _to_float_tensor(self.pre_flow_particles, "pre_flow_particles")
        fixed_resampling_mask = tf.convert_to_tensor(
            self.fixed_resampling_mask,
            dtype=tf.bool,
        )

        _require_static_rank(theta_batch, 2, "theta_batch")
        _require_static_rank(observations, 2, "observations")
        _require_static_rank(initial_particles, 3, "initial_particles")
        _require_static_rank(pre_flow_particles, 4, "pre_flow_particles")
        _require_static_rank(fixed_resampling_mask, 2, "fixed_resampling_mask")

        batch_size, num_particles, state_dim = _static_shape(
            initial_particles,
            "initial_particles",
        )
        theta_batch_size, parameter_dim = _static_shape(theta_batch, "theta_batch")
        time_steps, observation_dim = _static_shape(observations, "observations")
        pre_batch, pre_time, pre_particles, pre_state = _static_shape(
            pre_flow_particles,
            "pre_flow_particles",
        )
        mask_batch, mask_time = _static_shape(
            fixed_resampling_mask,
            "fixed_resampling_mask",
        )

        _require_equal(theta_batch_size, batch_size, "theta_batch batch size")
        _require_equal(pre_batch, batch_size, "pre_flow_particles batch size")
        _require_equal(pre_time, time_steps, "pre_flow_particles time steps")
        _require_equal(pre_particles, num_particles, "pre_flow_particles particle count")
        _require_equal(pre_state, state_dim, "pre_flow_particles state dim")
        _require_equal(mask_batch, batch_size, "fixed_resampling_mask batch size")
        _require_equal(mask_time, time_steps, "fixed_resampling_mask time steps")

        if self.initial_log_weights is None:
            initial_log_weights = uniform_log_weights(batch_size, num_particles)
        else:
            initial_log_weights = _to_float_tensor(
                self.initial_log_weights,
                "initial_log_weights",
            )
            _require_static_rank(initial_log_weights, 2, "initial_log_weights")
            weights_batch, weights_particles = _static_shape(
                initial_log_weights,
                "initial_log_weights",
            )
            _require_equal(weights_batch, batch_size, "initial_log_weights batch size")
            _require_equal(
                weights_particles,
                num_particles,
                "initial_log_weights particle count",
            )

        shape_contract = BatchedLEDHPFPFOTShapeContract(
            batch_size=batch_size,
            time_steps=time_steps,
            num_particles=num_particles,
            state_dim=state_dim,
            observation_dim=observation_dim,
            parameter_dim=parameter_dim,
        )

        object.__setattr__(self, "theta_batch", theta_batch)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "initial_particles", initial_particles)
        object.__setattr__(self, "pre_flow_particles", pre_flow_particles)
        object.__setattr__(self, "fixed_resampling_mask", fixed_resampling_mask)
        object.__setattr__(self, "initial_log_weights", initial_log_weights)
        object.__setattr__(self, "shape_contract", shape_contract)


@dataclass(frozen=True)
class BatchedLEDHFlowTensors:
    """Tensor output contract for one batched LEDH flow step."""

    post_flow_particles: tf.Tensor
    pre_flow_log_density: tf.Tensor
    forward_log_det: tf.Tensor
    local_posterior_means: tf.Tensor
    local_posterior_covariances: tf.Tensor

    def __post_init__(self) -> None:
        post_flow_particles = _to_float_tensor(
            self.post_flow_particles,
            "post_flow_particles",
        )
        pre_flow_log_density = _to_float_tensor(
            self.pre_flow_log_density,
            "pre_flow_log_density",
        )
        forward_log_det = _to_float_tensor(self.forward_log_det, "forward_log_det")
        local_posterior_means = _to_float_tensor(
            self.local_posterior_means,
            "local_posterior_means",
        )
        local_posterior_covariances = _to_float_tensor(
            self.local_posterior_covariances,
            "local_posterior_covariances",
        )

        _require_static_rank(post_flow_particles, 3, "post_flow_particles")
        _require_static_rank(pre_flow_log_density, 2, "pre_flow_log_density")
        _require_static_rank(forward_log_det, 2, "forward_log_det")
        _require_static_rank(local_posterior_means, 3, "local_posterior_means")
        _require_static_rank(
            local_posterior_covariances,
            4,
            "local_posterior_covariances",
        )
        _require_flow_shapes(
            post_flow_particles=post_flow_particles,
            pre_flow_log_density=pre_flow_log_density,
            forward_log_det=forward_log_det,
            local_posterior_means=local_posterior_means,
            local_posterior_covariances=local_posterior_covariances,
        )

        object.__setattr__(self, "post_flow_particles", post_flow_particles)
        object.__setattr__(self, "pre_flow_log_density", pre_flow_log_density)
        object.__setattr__(self, "forward_log_det", forward_log_det)
        object.__setattr__(self, "local_posterior_means", local_posterior_means)
        object.__setattr__(
            self,
            "local_posterior_covariances",
            local_posterior_covariances,
        )


@dataclass(frozen=True)
class BatchedLEDHFlowVJPTensors:
    """Manual cotangents for the linearized LEDH flow primitive."""

    pre_flow_particles: tf.Tensor
    prior_means: tf.Tensor
    observation_jacobian: tf.Tensor
    observation_residual: tf.Tensor
    transition_covariance: tf.Tensor
    observation_covariance: tf.Tensor


@dataclass(frozen=True)
class _BatchedLEDHLinearizedFlowAux:
    """Forward checkpoints used by the manual linearized LEDH flow VJP."""

    x0: tf.Tensor
    prior_means: tf.Tensor
    observation_jacobian: tf.Tensor
    observation_residual: tf.Tensor
    transition_covariance: tf.Tensor
    observation_covariance: tf.Tensor
    transition_covariance_stable: tf.Tensor
    observation_covariance_stable: tf.Tensor
    prior_chol: tf.Tensor
    prior_precision: tf.Tensor
    obs_precision: tf.Tensor
    pseudo_observation: tf.Tensor
    post_precision: tf.Tensor
    post_precision_stable: tf.Tensor
    post_covariance_unstabilized: tf.Tensor
    post_covariance: tf.Tensor
    post_chol: tf.Tensor
    prior_inv: tf.Tensor
    affine_transform: tf.Tensor
    delta: tf.Tensor
    info: tf.Tensor


@dataclass(frozen=True)
class BatchedAnnealedTransportTensors:
    """Tensor output contract for one fixed-mask annealed transport step."""

    particles: tf.Tensor
    log_weights: tf.Tensor
    transport_matrix: tf.Tensor
    max_row_residual: tf.Tensor
    max_column_residual: tf.Tensor

    def __post_init__(self) -> None:
        particles = _to_float_tensor(self.particles, "transported_particles")
        log_weights = _to_float_tensor(self.log_weights, "transported_log_weights")
        transport_matrix = _to_float_tensor(self.transport_matrix, "transport_matrix")
        max_row_residual = _to_float_tensor(self.max_row_residual, "max_row_residual")
        max_column_residual = _to_float_tensor(
            self.max_column_residual,
            "max_column_residual",
        )

        _require_static_rank(particles, 3, "transported_particles")
        _require_static_rank(log_weights, 2, "transported_log_weights")
        _require_static_rank(transport_matrix, 3, "transport_matrix")
        _require_static_rank(max_row_residual, 0, "max_row_residual")
        _require_static_rank(max_column_residual, 0, "max_column_residual")
        batch_size, num_particles, _state_dim = _static_shape(
            particles,
            "transported_particles",
        )
        _require_equal(
            _static_shape(log_weights, "transported_log_weights")[0],
            batch_size,
            "transported_log_weights batch size",
        )
        _require_equal(
            _static_shape(log_weights, "transported_log_weights")[1],
            num_particles,
            "transported_log_weights particle count",
        )
        matrix_shape = _static_shape(transport_matrix, "transport_matrix")
        expected_matrix_shape = (batch_size, num_particles, num_particles)
        sentinel_shape = (batch_size, 0, 0)
        if matrix_shape not in {expected_matrix_shape, sentinel_shape}:
            raise ValueError(
                "transport_matrix shape mismatch: "
                f"got {matrix_shape}, expected {expected_matrix_shape} or {sentinel_shape}"
            )

        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "log_weights", log_weights)
        object.__setattr__(self, "transport_matrix", transport_matrix)
        object.__setattr__(self, "max_row_residual", max_row_residual)
        object.__setattr__(self, "max_column_residual", max_column_residual)


@dataclass(frozen=True)
class BatchedLEDHPFPFOTValueTensors:
    """Batched fixed-branch LEDH-PFPF-OT value-recursion outputs."""

    log_likelihood: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess_by_time: tf.Tensor

    def __post_init__(self) -> None:
        log_likelihood = _to_float_tensor(self.log_likelihood, "log_likelihood")
        filtered_means = _to_float_tensor(self.filtered_means, "filtered_means")
        filtered_variances = _to_float_tensor(
            self.filtered_variances,
            "filtered_variances",
        )
        ess_by_time = _to_float_tensor(self.ess_by_time, "ess_by_time")
        _require_static_rank(log_likelihood, 1, "log_likelihood")
        _require_static_rank(filtered_means, 3, "filtered_means")
        _require_static_rank(filtered_variances, 3, "filtered_variances")
        _require_static_rank(ess_by_time, 2, "ess_by_time")
        time_steps, batch_size, state_dim = _static_shape(
            filtered_means,
            "filtered_means",
        )
        variance_shape = _static_shape(filtered_variances, "filtered_variances")
        expected_variance_shape = (time_steps, batch_size, state_dim)
        if variance_shape != expected_variance_shape:
            raise ValueError(
                "filtered_variances shape mismatch: "
                f"got {variance_shape}, expected {expected_variance_shape}"
            )
        _require_equal(
            _static_shape(log_likelihood, "log_likelihood")[0],
            batch_size,
            "log_likelihood batch size",
        )
        ess_shape = _static_shape(ess_by_time, "ess_by_time")
        expected_ess_shape = (time_steps, batch_size)
        if ess_shape != expected_ess_shape:
            raise ValueError(
                f"ess_by_time shape mismatch: got {ess_shape}, expected {expected_ess_shape}"
            )

        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "filtered_means", filtered_means)
        object.__setattr__(self, "filtered_variances", filtered_variances)
        object.__setattr__(self, "ess_by_time", ess_by_time)


@dataclass(frozen=True)
class BatchedLEDHPFPFOTValueScoreTensors:
    """Batched relaxed-objective value and TensorFlow autodiff score."""

    log_likelihood: tf.Tensor
    score: tf.Tensor

    def __post_init__(self) -> None:
        log_likelihood = _to_float_tensor(self.log_likelihood, "log_likelihood")
        score = _to_float_tensor(self.score, "score")
        _require_static_rank(log_likelihood, 1, "log_likelihood")
        _require_static_rank(score, 2, "score")
        batch_size = _static_shape(log_likelihood, "log_likelihood")[0]
        _require_equal(_static_shape(score, "score")[0], batch_size, "score batch size")

        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class BatchedLEDHPFPFOTCallbacks:
    """Readiness wrapper for deterministic batched model callbacks."""

    ledh_flow: Callable[..., BatchedLEDHFlowTensors]
    transition_log_density: Callable[..., tf.Tensor]
    observation_log_density: Callable[..., tf.Tensor]
    allow_uninspectable_callbacks: bool = False

    def __post_init__(self) -> None:
        for name, callback in (
            ("ledh_flow", self.ledh_flow),
            ("transition_log_density", self.transition_log_density),
            ("observation_log_density", self.observation_log_density),
        ):
            assert_callback_has_no_forbidden_rng(
                callback,
                name=name,
                allow_uninspectable=self.allow_uninspectable_callbacks,
            )


def uniform_log_weights(batch_size: int, num_particles: int) -> tf.Tensor:
    """Return normalized uniform log weights with shape ``[B,N]``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    return tf.fill(
        [batch_size, num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )


def validate_batched_value_tensor(
    value: tf.Tensor,
    contract: BatchedLEDHPFPFOTShapeContract,
) -> tf.Tensor:
    """Validate and return a batched value tensor with shape ``[B]``."""

    value = _to_float_tensor(value, "value")
    _require_static_rank(value, 1, "value")
    _require_equal(_static_shape(value, "value")[0], contract.batch_size, "value batch size")
    return value


def validate_batched_score_tensor(
    score: tf.Tensor,
    contract: BatchedLEDHPFPFOTShapeContract,
) -> tf.Tensor:
    """Validate and return a batched score tensor with shape ``[B,p]``."""

    score = _to_float_tensor(score, "score")
    _require_static_rank(score, 2, "score")
    batch_size, parameter_dim = _static_shape(score, "score")
    _require_equal(batch_size, contract.batch_size, "score batch size")
    _require_equal(parameter_dim, contract.parameter_dim, "score parameter dim")
    return score


def validate_flow_tensors_against_contract(
    flow: BatchedLEDHFlowTensors,
    contract: BatchedLEDHPFPFOTShapeContract,
) -> BatchedLEDHFlowTensors:
    """Fail closed if one-step LEDH flow tensors violate ``[B,N,D]`` shapes."""

    batch_size, num_particles, state_dim = _static_shape(
        flow.post_flow_particles,
        "post_flow_particles",
    )
    _require_equal(batch_size, contract.batch_size, "flow batch size")
    _require_equal(num_particles, contract.num_particles, "flow particle count")
    _require_equal(state_dim, contract.state_dim, "flow state dim")
    return flow


def batched_ledh_flow_core_tf(
    *,
    pre_flow_particles: tf.Tensor,
    ancestors: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    prior_mean_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    jitter: float | tf.Tensor = 1.0e-9,
) -> BatchedLEDHFlowTensors:
    """Vectorized one-step LEDH affine flow over parameter rows and particles."""

    x0 = _to_float_tensor(pre_flow_particles, "pre_flow_particles")
    ancestors = _to_float_tensor(ancestors, "ancestors")
    observation = _to_float_tensor(observation, "observation")
    transition_matrix = _to_float_tensor(transition_matrix, "transition_matrix")
    transition_covariance = _stabilize_batch_covariance(
        transition_covariance,
        jitter,
        "transition_covariance",
    )
    observation_covariance = _stabilize_batch_covariance(
        observation_covariance,
        jitter,
        "observation_covariance",
    )

    _require_static_rank(x0, 3, "pre_flow_particles")
    _require_static_rank(ancestors, 3, "ancestors")
    _require_static_rank(transition_matrix, 3, "transition_matrix")
    _require_static_rank(transition_covariance, 3, "transition_covariance")
    _require_static_rank(observation_covariance, 3, "observation_covariance")

    batch_size, num_particles, state_dim = _static_shape(x0, "pre_flow_particles")
    _require_equal(
        _static_shape(ancestors, "ancestors")[0],
        batch_size,
        "ancestors batch size",
    )
    _require_equal(
        _static_shape(ancestors, "ancestors")[1],
        num_particles,
        "ancestors particle count",
    )
    _require_equal(
        _static_shape(ancestors, "ancestors")[2],
        state_dim,
        "ancestors state dim",
    )
    _require_square_batch(transition_matrix, batch_size, state_dim, "transition_matrix")
    _require_square_batch(
        transition_covariance,
        batch_size,
        state_dim,
        "transition_covariance",
    )
    observation_dim = _static_shape(observation_covariance, "observation_covariance")[1]
    _require_square_batch(
        observation_covariance,
        batch_size,
        observation_dim,
        "observation_covariance",
    )

    if prior_mean_fn is None:
        prior_means = tf.einsum("bnj,bdj->bnd", ancestors, transition_matrix)
    else:
        prior_means = _to_float_tensor(
            prior_mean_fn(ancestors),
            "prior_mean_fn output",
        )
        _require_static_rank(prior_means, 3, "prior_mean_fn output")
        if _static_shape(prior_means, "prior_mean_fn output") != (
            batch_size,
            num_particles,
            state_dim,
        ):
            raise ValueError("prior_mean_fn output shape mismatch")
    pre_flow_log_density = _batched_gaussian_logpdf(
        x0 - prior_means,
        transition_covariance,
    )

    prior_chol = tf.linalg.cholesky(transition_covariance)
    prior_precision = tf.linalg.cholesky_solve(
        prior_chol,
        _tile_eye(batch_size, state_dim),
    )
    obs_chol = tf.linalg.cholesky(observation_covariance)
    obs_precision = tf.linalg.cholesky_solve(
        obs_chol,
        _tile_eye(batch_size, observation_dim),
    )

    h_ref = _to_float_tensor(observation_fn(x0), "observation_fn output")
    h_jac = _to_float_tensor(observation_jacobian_fn(x0), "observation_jacobian_fn output")
    residual = _to_float_tensor(
        observation_residual_fn(h_ref, observation),
        "observation_residual_fn output",
    )
    _require_static_rank(h_ref, 3, "observation_fn output")
    _require_static_rank(h_jac, 4, "observation_jacobian_fn output")
    _require_static_rank(residual, 3, "observation_residual_fn output")
    _require_observation_callback_shapes(
        h_ref=h_ref,
        h_jac=h_jac,
        residual=residual,
        batch_size=batch_size,
        num_particles=num_particles,
        state_dim=state_dim,
        observation_dim=observation_dim,
    )

    pseudo_observation = tf.einsum("bnod,bnd->bno", h_jac, x0) + residual
    post_precision = prior_precision[:, None, :, :] + tf.einsum(
        "bnod,boq,bnqe->bnde",
        h_jac,
        obs_precision,
        h_jac,
    )
    stabilized_post_precision = _stabilize_batch_covariance(
        post_precision,
        jitter,
        "post_precision",
    )
    post_precision_chol = tf.linalg.cholesky(stabilized_post_precision)
    post_covariance = tf.linalg.cholesky_solve(
        post_precision_chol,
        tf.broadcast_to(
            tf.eye(state_dim, dtype=DTYPE),
            tf.shape(stabilized_post_precision),
        ),
    )
    post_covariance = _stabilize_batch_covariance(
        post_covariance,
        jitter,
        "post_covariance",
    )
    info = tf.einsum("bde,bne->bnd", prior_precision, prior_means) + tf.einsum(
        "bnod,boq,bnq->bnd",
        h_jac,
        obs_precision,
        pseudo_observation,
    )
    post_mean = tf.einsum("bnde,bne->bnd", post_covariance, info)
    post_chol = tf.linalg.cholesky(post_covariance)
    prior_inv = tf.linalg.triangular_solve(
        prior_chol,
        _tile_eye(batch_size, state_dim),
    )
    affine_transform = tf.einsum("bnij,bjk->bnik", post_chol, prior_inv)
    post_flow_particles = post_mean + tf.einsum(
        "bnij,bnj->bni",
        affine_transform,
        x0 - prior_means,
    )
    forward_log_det = tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(post_chol)),
        axis=-1,
    ) - tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(prior_chol)),
        axis=-1,
    )[:, None]

    return BatchedLEDHFlowTensors(
        post_flow_particles=post_flow_particles,
        pre_flow_log_density=pre_flow_log_density,
        forward_log_det=forward_log_det,
        local_posterior_means=post_mean,
        local_posterior_covariances=post_covariance,
    )


def _symmetrize_matrix(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _cholesky_vjp(cholesky_factor: tf.Tensor, upstream: tf.Tensor) -> tf.Tensor:
    """Manual VJP for ``tf.linalg.cholesky`` on symmetric positive matrices."""

    chol = _to_float_tensor(cholesky_factor, "cholesky_factor")
    bar_chol = _to_float_tensor(upstream, "cholesky upstream")
    moment = tf.matmul(chol, bar_chol, transpose_a=True)
    lower = tf.linalg.band_part(moment, -1, 0)
    sym_lower = lower + tf.linalg.matrix_transpose(lower) - tf.linalg.diag(
        tf.linalg.diag_part(lower)
    )
    left = tf.linalg.triangular_solve(
        chol,
        sym_lower,
        lower=True,
        adjoint=True,
    )
    right = tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(
            chol,
            tf.linalg.matrix_transpose(left),
            lower=True,
            adjoint=True,
        )
    )
    return 0.5 * right


def _inverse_spd_vjp(inverse: tf.Tensor, upstream: tf.Tensor) -> tf.Tensor:
    """Manual VJP for ``tf.linalg.inv`` at an SPD inverse output."""

    inv = _to_float_tensor(inverse, "inverse")
    bar_inv = _to_float_tensor(upstream, "inverse upstream")
    return -tf.matmul(
        inv,
        tf.matmul(bar_inv, inv, transpose_b=True),
        transpose_a=True,
    )


def _triangular_solve_eye_vjp(
    triangular_factor: tf.Tensor,
    solution: tf.Tensor,
    upstream_solution: tf.Tensor,
) -> tf.Tensor:
    """Manual VJP for ``tf.linalg.triangular_solve(L, I)`` with lower ``L``."""

    factor = _to_float_tensor(triangular_factor, "triangular_factor")
    solve_value = _to_float_tensor(solution, "solution")
    bar_solution = _to_float_tensor(upstream_solution, "solution upstream")
    raw = -tf.matmul(
        tf.linalg.triangular_solve(
            factor,
            bar_solution,
            lower=True,
            adjoint=True,
        ),
        solve_value,
        transpose_b=True,
    )
    return tf.linalg.band_part(raw, -1, 0)


def _batched_gaussian_logpdf_vjp(
    residuals: tf.Tensor,
    covariance: tf.Tensor,
    upstream: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Manual VJP for ``_batched_gaussian_logpdf``.

    Returns cotangents for residuals with shape ``[B,N,D]`` and covariance with
    shape ``[B,D,D]``.  Covariance cotangents are reduced across particles.
    """

    residuals = _to_float_tensor(residuals, "residuals")
    covariance = _to_float_tensor(covariance, "covariance")
    upstream = _to_float_tensor(upstream, "gaussian logpdf upstream")
    _require_static_rank(residuals, 3, "residuals")
    _require_static_rank(covariance, 3, "covariance")
    _require_static_rank(upstream, 2, "gaussian logpdf upstream")
    batch_size, num_particles, dim = _static_shape(residuals, "residuals")
    _require_square_batch(covariance, batch_size, dim, "covariance")
    if _static_shape(upstream, "gaussian logpdf upstream") != (batch_size, num_particles):
        raise ValueError("gaussian logpdf upstream shape mismatch")

    chol = tf.linalg.cholesky(covariance)
    precision = tf.linalg.cholesky_solve(chol, _tile_eye(batch_size, dim))
    solved = tf.einsum("bnd,bde->bne", residuals, precision)
    bar_residuals = -upstream[:, :, None] * solved
    outer = tf.einsum("bni,bnj->bnij", solved, solved)
    bar_covariance = 0.5 * tf.reduce_sum(
        upstream[:, :, None, None] * (outer - precision[:, None, :, :]),
        axis=1,
    )
    return bar_residuals, _symmetrize_matrix(bar_covariance)


def _transition_gaussian_log_density_vjp(
    x_next: tf.Tensor,
    transition_mean: tf.Tensor,
    transition_covariance: tf.Tensor,
    upstream: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Manual VJP for a transition Gaussian log-density residual."""

    x_next = _to_float_tensor(x_next, "x_next")
    transition_mean = _to_float_tensor(transition_mean, "transition_mean")
    if _static_shape(transition_mean, "transition_mean") != _static_shape(
        x_next,
        "x_next",
    ):
        raise ValueError("transition_mean shape mismatch")
    bar_residual, bar_covariance = _batched_gaussian_logpdf_vjp(
        x_next - transition_mean,
        transition_covariance,
        upstream,
    )
    return {
        "x_next": bar_residual,
        "transition_mean": -bar_residual,
        "transition_covariance": bar_covariance,
    }


def _observation_gaussian_log_density_vjp(
    predicted_observation: tf.Tensor,
    observation: tf.Tensor,
    observation_covariance: tf.Tensor,
    upstream: tf.Tensor,
    *,
    residual_convention: str = "model_minus_observation",
) -> dict[str, tf.Tensor]:
    """Manual VJP for observation Gaussian log-density residual conventions."""

    predicted = _to_float_tensor(predicted_observation, "predicted_observation")
    observation = _to_float_tensor(observation, "observation")
    _require_static_rank(predicted, 3, "predicted_observation")
    observation_rank = observation.shape.rank
    if observation_rank == 1:
        observation_broadcast = observation[None, None, :]
    elif observation_rank == 2:
        observation_broadcast = observation[:, None, :]
    else:
        raise ValueError("observation must have static rank 1 or 2")
    if residual_convention == "model_minus_observation":
        residual = predicted - observation_broadcast
    elif residual_convention == "observation_minus_model":
        residual = observation_broadcast - predicted
    else:
        raise ValueError(
            "residual_convention must be 'model_minus_observation' or "
            "'observation_minus_model'"
        )
    bar_residual, bar_covariance = _batched_gaussian_logpdf_vjp(
        residual,
        observation_covariance,
        upstream,
    )
    bar_predicted, bar_observation = _observation_difference_residual_vjp(
        bar_residual,
        convention=residual_convention,
    )
    if observation_rank == 1:
        bar_observation = tf.reduce_sum(bar_observation, axis=0)
    return {
        "predicted_observation": bar_predicted,
        "observation": bar_observation,
        "observation_covariance": bar_covariance,
    }


def _normalize_log_weights_vjp(
    corrected_log_weights: tf.Tensor,
    normalized_log_weight_upstream: tf.Tensor | None = None,
    incremental_upstream: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Manual VJP for log-normalization and likelihood increment."""

    corrected = _to_float_tensor(corrected_log_weights, "corrected_log_weights")
    _require_static_rank(corrected, 2, "corrected_log_weights")
    batch_size, num_particles = _static_shape(corrected, "corrected_log_weights")
    weights, incremental = _normalize_log_weights(corrected)
    if normalized_log_weight_upstream is None:
        bar_normalized = tf.zeros_like(corrected)
    else:
        bar_normalized = _to_float_tensor(
            normalized_log_weight_upstream,
            "normalized_log_weight_upstream",
        )
        if _static_shape(bar_normalized, "normalized_log_weight_upstream") != (
            batch_size,
            num_particles,
        ):
            raise ValueError("normalized_log_weight_upstream shape mismatch")
    if incremental_upstream is None:
        bar_incremental = tf.zeros([batch_size], dtype=DTYPE)
    else:
        bar_incremental = _to_float_tensor(incremental_upstream, "incremental_upstream")
        _require_static_rank(bar_incremental, 1, "incremental_upstream")
        if _static_shape(bar_incremental, "incremental_upstream") != (batch_size,):
            raise ValueError("incremental_upstream shape mismatch")

    bar_corrected = (
        bar_incremental[:, None] * weights
        + bar_normalized
        - weights * tf.reduce_sum(bar_normalized, axis=1, keepdims=True)
    )
    return bar_corrected, weights, incremental


def _floor_log_weights_vjp(
    weights: tf.Tensor,
    floored_log_weight_upstream: tf.Tensor,
    *,
    floor: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Fixed-branch VJP for ``log(max(weights, floor))``.

    The returned first tensor is the cotangent of unfloored normalized log
    weights.  The second tensor is the branch mask used by the VJP.
    """

    weights = _to_float_tensor(weights, "weights")
    upstream = _to_float_tensor(
        floored_log_weight_upstream,
        "floored_log_weight_upstream",
    )
    _require_static_rank(weights, 2, "weights")
    _require_static_rank(upstream, 2, "floored_log_weight_upstream")
    if _static_shape(upstream, "floored_log_weight_upstream") != _static_shape(
        weights,
        "weights",
    ):
        raise ValueError("floored_log_weight_upstream shape mismatch")
    floor_value = _log_weight_floor() if floor is None else tf.cast(floor, DTYPE)
    active = weights > floor_value
    return tf.where(active, upstream, tf.zeros_like(upstream)), active


def _normalize_log_weights_with_floor_vjp(
    corrected_log_weights: tf.Tensor,
    floored_log_weight_upstream: tf.Tensor,
    incremental_upstream: tf.Tensor | None = None,
    *,
    floor: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Manual VJP for normalize, fixed floor branch, and likelihood increment."""

    corrected = _to_float_tensor(corrected_log_weights, "corrected_log_weights")
    weights, incremental = _normalize_log_weights(corrected)
    bar_normalized, floor_active = _floor_log_weights_vjp(
        weights,
        floored_log_weight_upstream,
        floor=floor,
    )
    bar_corrected, _weights, _incremental = _normalize_log_weights_vjp(
        corrected,
        bar_normalized,
        incremental_upstream,
    )
    return bar_corrected, weights, incremental, floor_active


def _log_weight_correction_vjp(
    corrected_log_weight_upstream: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Distribute cotangents through the LEDH log-weight correction identity."""

    bar = _to_float_tensor(
        corrected_log_weight_upstream,
        "corrected_log_weight_upstream",
    )
    _require_static_rank(bar, 2, "corrected_log_weight_upstream")
    return {
        "current_log_weights": bar,
        "transition_log_density": bar,
        "observation_log_density": bar,
        "pre_flow_log_density": -bar,
        "forward_log_det": bar,
    }


def _observation_difference_residual_vjp(
    residual_upstream: tf.Tensor,
    *,
    convention: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Manual VJP for observation residual subtraction conventions."""

    bar_residual = _to_float_tensor(residual_upstream, "residual_upstream")
    if convention == "model_minus_observation":
        return bar_residual, -tf.reduce_sum(bar_residual, axis=1)
    if convention == "observation_minus_model":
        return -bar_residual, tf.reduce_sum(bar_residual, axis=1)
    raise ValueError(
        "convention must be 'model_minus_observation' or 'observation_minus_model'"
    )


def _batched_ledh_linearized_flow_with_aux_tf(
    *,
    pre_flow_particles: tf.Tensor,
    prior_means: tf.Tensor,
    observation_jacobian: tf.Tensor,
    observation_residual: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    jitter: float | tf.Tensor = 1.0e-9,
) -> tuple[BatchedLEDHFlowTensors, _BatchedLEDHLinearizedFlowAux]:
    """Run the matrix-only LEDH flow primitive and retain VJP checkpoints."""

    x0 = _to_float_tensor(pre_flow_particles, "pre_flow_particles")
    prior_means = _to_float_tensor(prior_means, "prior_means")
    h_jac = _to_float_tensor(observation_jacobian, "observation_jacobian")
    residual = _to_float_tensor(observation_residual, "observation_residual")
    transition_covariance = _to_float_tensor(
        transition_covariance,
        "transition_covariance",
    )
    observation_covariance = _to_float_tensor(
        observation_covariance,
        "observation_covariance",
    )
    _require_static_rank(x0, 3, "pre_flow_particles")
    _require_static_rank(prior_means, 3, "prior_means")
    _require_static_rank(h_jac, 4, "observation_jacobian")
    _require_static_rank(residual, 3, "observation_residual")
    _require_static_rank(transition_covariance, 3, "transition_covariance")
    _require_static_rank(observation_covariance, 3, "observation_covariance")
    batch_size, num_particles, state_dim = _static_shape(x0, "pre_flow_particles")
    if _static_shape(prior_means, "prior_means") != (batch_size, num_particles, state_dim):
        raise ValueError("prior_means shape mismatch")
    observation_dim = _static_shape(observation_covariance, "observation_covariance")[1]
    _require_square_batch(
        transition_covariance,
        batch_size,
        state_dim,
        "transition_covariance",
    )
    _require_square_batch(
        observation_covariance,
        batch_size,
        observation_dim,
        "observation_covariance",
    )
    _require_observation_callback_shapes(
        h_ref=residual,
        h_jac=h_jac,
        residual=residual,
        batch_size=batch_size,
        num_particles=num_particles,
        state_dim=state_dim,
        observation_dim=observation_dim,
    )

    transition_covariance_stable = _stabilize_batch_covariance(
        transition_covariance,
        jitter,
        "transition_covariance",
    )
    observation_covariance_stable = _stabilize_batch_covariance(
        observation_covariance,
        jitter,
        "observation_covariance",
    )
    delta = x0 - prior_means
    pre_flow_log_density = _batched_gaussian_logpdf(
        delta,
        transition_covariance_stable,
    )
    prior_chol = tf.linalg.cholesky(transition_covariance_stable)
    prior_precision = tf.linalg.cholesky_solve(
        prior_chol,
        _tile_eye(batch_size, state_dim),
    )
    obs_chol = tf.linalg.cholesky(observation_covariance_stable)
    obs_precision = tf.linalg.cholesky_solve(
        obs_chol,
        _tile_eye(batch_size, observation_dim),
    )
    pseudo_observation = tf.einsum("bnod,bnd->bno", h_jac, x0) + residual
    post_precision = prior_precision[:, None, :, :] + tf.einsum(
        "bnod,boq,bnqe->bnde",
        h_jac,
        obs_precision,
        h_jac,
    )
    post_precision_stable = _stabilize_batch_covariance(
        post_precision,
        jitter,
        "post_precision",
    )
    post_covariance_unstabilized = tf.linalg.inv(post_precision_stable)
    post_covariance = _stabilize_batch_covariance(
        post_covariance_unstabilized,
        jitter,
        "post_covariance",
    )
    info = tf.einsum("bde,bne->bnd", prior_precision, prior_means) + tf.einsum(
        "bnod,boq,bnq->bnd",
        h_jac,
        obs_precision,
        pseudo_observation,
    )
    post_mean = tf.einsum("bnde,bne->bnd", post_covariance, info)
    post_chol = tf.linalg.cholesky(post_covariance)
    prior_inv = tf.linalg.triangular_solve(
        prior_chol,
        _tile_eye(batch_size, state_dim),
    )
    affine_transform = tf.einsum("bnij,bjk->bnik", post_chol, prior_inv)
    post_flow_particles = post_mean + tf.einsum(
        "bnij,bnj->bni",
        affine_transform,
        delta,
    )
    forward_log_det = tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(post_chol)),
        axis=-1,
    ) - tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(prior_chol)),
        axis=-1,
    )[:, None]
    flow = BatchedLEDHFlowTensors(
        post_flow_particles=post_flow_particles,
        pre_flow_log_density=pre_flow_log_density,
        forward_log_det=forward_log_det,
        local_posterior_means=post_mean,
        local_posterior_covariances=post_covariance,
    )
    aux = _BatchedLEDHLinearizedFlowAux(
        x0=x0,
        prior_means=prior_means,
        observation_jacobian=h_jac,
        observation_residual=residual,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        transition_covariance_stable=transition_covariance_stable,
        observation_covariance_stable=observation_covariance_stable,
        prior_chol=prior_chol,
        prior_precision=prior_precision,
        obs_precision=obs_precision,
        pseudo_observation=pseudo_observation,
        post_precision=post_precision,
        post_precision_stable=post_precision_stable,
        post_covariance_unstabilized=post_covariance_unstabilized,
        post_covariance=post_covariance,
        post_chol=post_chol,
        prior_inv=prior_inv,
        affine_transform=affine_transform,
        delta=delta,
        info=info,
    )
    return flow, aux


def _batched_ledh_linearized_flow_vjp(
    aux: _BatchedLEDHLinearizedFlowAux,
    post_flow_particles_upstream: tf.Tensor,
    pre_flow_log_density_upstream: tf.Tensor,
    forward_log_det_upstream: tf.Tensor,
) -> BatchedLEDHFlowVJPTensors:
    """Manual VJP for the matrix-only LEDH affine flow primitive."""

    x0 = aux.x0
    prior_means = aux.prior_means
    h_jac = aux.observation_jacobian
    state_dim = _static_shape(x0, "pre_flow_particles")[2]
    batch_size, num_particles, _state_dim = _static_shape(x0, "pre_flow_particles")

    bar_post = _to_float_tensor(
        post_flow_particles_upstream,
        "post_flow_particles_upstream",
    )
    bar_pre_log = _to_float_tensor(
        pre_flow_log_density_upstream,
        "pre_flow_log_density_upstream",
    )
    bar_logdet = _to_float_tensor(forward_log_det_upstream, "forward_log_det_upstream")
    if _static_shape(bar_post, "post_flow_particles_upstream") != _static_shape(
        x0,
        "pre_flow_particles",
    ):
        raise ValueError("post_flow_particles_upstream shape mismatch")
    if _static_shape(bar_pre_log, "pre_flow_log_density_upstream") != (
        batch_size,
        num_particles,
    ):
        raise ValueError("pre_flow_log_density_upstream shape mismatch")
    if _static_shape(bar_logdet, "forward_log_det_upstream") != (
        batch_size,
        num_particles,
    ):
        raise ValueError("forward_log_det_upstream shape mismatch")

    bar_x0 = tf.zeros_like(x0)
    bar_prior_means = tf.zeros_like(prior_means)
    bar_h_jac = tf.zeros_like(h_jac)
    bar_residual = tf.zeros_like(aux.observation_residual)
    bar_transition_covariance = tf.zeros_like(aux.transition_covariance_stable)
    bar_observation_covariance = tf.zeros_like(aux.observation_covariance_stable)

    bar_delta_from_pre_log, bar_cov_from_pre_log = _batched_gaussian_logpdf_vjp(
        aux.delta,
        aux.transition_covariance_stable,
        bar_pre_log,
    )
    bar_x0 += bar_delta_from_pre_log
    bar_prior_means -= bar_delta_from_pre_log
    bar_transition_covariance += bar_cov_from_pre_log

    bar_post_mean = bar_post
    bar_affine = tf.einsum("bni,bnj->bnij", bar_post, aux.delta)
    bar_delta = tf.einsum("bnij,bni->bnj", aux.affine_transform, bar_post)
    bar_x0 += bar_delta
    bar_prior_means -= bar_delta

    bar_post_chol = tf.einsum("bnik,bjk->bnij", bar_affine, aux.prior_inv)
    bar_prior_inv = tf.einsum("bnij,bnik->bjk", aux.post_chol, bar_affine)

    diag_post = tf.linalg.diag_part(aux.post_chol)
    bar_post_chol += tf.linalg.diag(bar_logdet[:, :, None] / diag_post)
    diag_prior = tf.linalg.diag_part(aux.prior_chol)
    bar_prior_chol = tf.linalg.diag(
        -tf.reduce_sum(bar_logdet, axis=1)[:, None] / diag_prior
    )
    bar_prior_chol += _triangular_solve_eye_vjp(
        aux.prior_chol,
        aux.prior_inv,
        bar_prior_inv,
    )

    bar_post_covariance = _cholesky_vjp(aux.post_chol, bar_post_chol)
    bar_post_covariance += tf.einsum(
        "bni,bnj->bnij",
        bar_post_mean,
        aux.info,
    )
    bar_info = tf.einsum("bnij,bni->bnj", aux.post_covariance, bar_post_mean)

    bar_post_covariance_unstabilized = _symmetrize_matrix(bar_post_covariance)
    bar_post_precision_stable = _inverse_spd_vjp(
        aux.post_covariance_unstabilized,
        bar_post_covariance_unstabilized,
    )
    bar_post_precision = _symmetrize_matrix(bar_post_precision_stable)

    bar_prior_precision = tf.reduce_sum(bar_post_precision, axis=1)
    sh = tf.einsum("boq,bnqd->bnod", aux.obs_precision, h_jac)
    bar_h_jac += tf.matmul(sh, bar_post_precision, transpose_b=True)
    bar_h_jac += tf.einsum(
        "boq,bnqd->bnod",
        tf.linalg.matrix_transpose(aux.obs_precision),
        tf.matmul(h_jac, bar_post_precision),
    )
    bar_obs_precision = tf.einsum(
        "bnod,bnde,bnpe->bop",
        h_jac,
        bar_post_precision,
        h_jac,
    )

    bar_prior_precision += tf.einsum("bnd,bne->bde", bar_info, prior_means)
    bar_prior_means += tf.einsum("bde,bnd->bne", aux.prior_precision, bar_info)

    s_z = tf.einsum("bop,bnp->bno", aux.obs_precision, aux.pseudo_observation)
    bar_h_jac += tf.einsum("bno,bnd->bnod", s_z, bar_info)
    h_bar_info = tf.einsum("bnod,bnd->bno", h_jac, bar_info)
    bar_obs_precision += tf.einsum(
        "bno,bnp->bop",
        h_bar_info,
        aux.pseudo_observation,
    )
    bar_pseudo_observation = tf.einsum(
        "bop,bno->bnp",
        aux.obs_precision,
        h_bar_info,
    )

    bar_h_jac += tf.einsum("bno,bnd->bnod", bar_pseudo_observation, x0)
    bar_x0 += tf.einsum("bnod,bno->bnd", h_jac, bar_pseudo_observation)
    bar_residual += bar_pseudo_observation

    bar_transition_covariance += _inverse_spd_vjp(
        aux.prior_precision,
        bar_prior_precision,
    )
    bar_observation_covariance += _inverse_spd_vjp(
        aux.obs_precision,
        bar_obs_precision,
    )
    bar_transition_covariance += _cholesky_vjp(aux.prior_chol, bar_prior_chol)

    return BatchedLEDHFlowVJPTensors(
        pre_flow_particles=bar_x0,
        prior_means=bar_prior_means,
        observation_jacobian=bar_h_jac,
        observation_residual=bar_residual,
        transition_covariance=_symmetrize_matrix(bar_transition_covariance),
        observation_covariance=_symmetrize_matrix(bar_observation_covariance),
    )


def batched_annealed_transport_core_tf(
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    fixed_resampling_mask: tf.Tensor,
    *,
    epsilon: float | tf.Tensor = 0.5,
    scaling: float | tf.Tensor = 0.9,
    convergence_threshold: float | tf.Tensor = 1.0e-3,
    max_iterations: int | tf.Tensor = 100,
    transport_gradient_mode: str = "filterflow_clipped",
    transport_plan_mode: str = "dense",
    transport_ad_mode: str = "stabilized",
    row_chunk_size: int | None = None,
    col_chunk_size: int | None = None,
    warmstart_state: AnnealedTransportWarmstartStateTF | None = None,
) -> BatchedAnnealedTransportTensors:
    """Apply annealed transport to fixed-mask rows without eager branch logic."""

    _sync_transport_dtype()
    allowed_gradient_modes = {
        "filterflow_clipped",
        "filterflow_custom_op",
        "raw",
        MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
        MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    }
    if transport_gradient_mode not in allowed_gradient_modes:
        raise ValueError(
            "transport_gradient_mode must be 'filterflow_clipped', "
            f"'filterflow_custom_op', 'raw', or "
            f"'{MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE}'/"
            f"'{MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE}'/"
            f"'{MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE}'"
        )
    if transport_plan_mode not in {"dense", "streaming"}:
        raise ValueError("transport_plan_mode must be 'dense' or 'streaming'")
    _validate_transport_ad_mode(transport_ad_mode)
    if transport_gradient_mode == MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE:
        if transport_plan_mode != "dense":
            raise ValueError("manual dense finite route supports dense transport only")
        if warmstart_state is not None:
            raise ValueError("manual dense finite route does not support warmstarts")
        if transport_ad_mode != "stabilized":
            raise ValueError("manual dense finite route requires transport_ad_mode='stabilized'")
        manual_dense_finite_steps = _manual_dense_finite_steps(max_iterations)
    else:
        manual_dense_finite_steps = 0
    if transport_gradient_mode == MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE:
        if transport_plan_mode != "streaming":
            raise ValueError("manual streaming finite route supports streaming transport only")
        if warmstart_state is not None:
            raise ValueError("manual streaming finite route does not support warmstarts")
        if transport_ad_mode not in {"stabilized", "full"}:
            raise ValueError("manual streaming finite route requires transport_ad_mode='stabilized' or 'full'")
        manual_streaming_finite_steps = _manual_dense_finite_steps(max_iterations)
    else:
        manual_streaming_finite_steps = 0
    if transport_gradient_mode == MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE:
        if transport_plan_mode != "streaming":
            raise ValueError("manual streaming blockwise VJP finite route supports streaming transport only")
        if warmstart_state is not None:
            raise ValueError("manual streaming blockwise VJP finite route does not support warmstarts")
        if transport_ad_mode != "stabilized":
            raise ValueError(
                "manual streaming blockwise VJP finite route requires transport_ad_mode='stabilized'"
            )
        manual_streaming_blockwise_vjp_finite_steps = _manual_dense_finite_steps(max_iterations)
    else:
        manual_streaming_blockwise_vjp_finite_steps = 0
    if transport_plan_mode == "streaming" and transport_gradient_mode != "raw":
        if transport_gradient_mode not in {
            MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
        }:
            raise ValueError("streaming transport currently supports raw or manual streaming gradients only")
    x = _to_float_tensor(particles, "particles")
    logw = _to_float_tensor(log_weights, "log_weights")
    mask = tf.reshape(tf.convert_to_tensor(fixed_resampling_mask, dtype=tf.bool), [-1])
    _require_static_rank(x, 3, "particles")
    _require_static_rank(logw, 2, "log_weights")
    _require_static_rank(mask, 1, "fixed_resampling_mask")

    batch_size, num_particles, _state_dim = _static_shape(x, "particles")
    if transport_plan_mode == "streaming":
        chunks = resolve_transport_chunks(
            num_particles,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        row_chunk_size = chunks.row_chunk_size
        col_chunk_size = chunks.col_chunk_size
    _require_equal(
        _static_shape(logw, "log_weights")[0],
        batch_size,
        "log_weights batch size",
    )
    _require_equal(
        _static_shape(logw, "log_weights")[1],
        num_particles,
        "log_weights particle count",
    )
    _require_equal(
        _static_shape(mask, "fixed_resampling_mask")[0],
        batch_size,
        "fixed_resampling_mask batch size",
    )

    center = tf.reduce_mean(x, axis=1, keepdims=True)
    center = _maybe_stop(center, stop=_transport_ad_stop_scale(transport_ad_mode))
    centered = x - center
    scale = _filterflow_scale(x)
    scale_for_division = _maybe_stop(
        scale,
        stop=_transport_ad_stop_scale(transport_ad_mode),
    )
    scaled_x = centered / scale_for_division[:, None, None]
    epsilon_tensor = tf.convert_to_tensor(epsilon, dtype=DTYPE)
    scaling_tensor = tf.convert_to_tensor(scaling, dtype=DTYPE)
    threshold_tensor = tf.convert_to_tensor(convergence_threshold, dtype=DTYPE)
    max_iterations_tensor = tf.convert_to_tensor(max_iterations, dtype=tf.int32)
    num_particles_tensor = tf.shape(x)[1]
    if transport_gradient_mode == MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE:
        if epsilon_tensor.shape.rank is not None and epsilon_tensor.shape.rank != 0:
            raise ValueError("manual streaming finite route supports scalar epsilon only")
        epsilon0 = _filterflow_epsilon_start(scaled_x)
        if transport_ad_mode == "stabilized":
            epsilon0 = tf.stop_gradient(epsilon0)
            transport_fn = _filterflow_manual_streaming_finite_transport_stopped_scale_keys
        else:
            transport_fn = _filterflow_manual_streaming_finite_transport_total_vjp
        (
            transported,
            row_residual,
        ) = transport_fn(
            scaled_x,
            x,
            logw,
            epsilon_tensor,
            epsilon0,
            scaling_tensor,
            steps=manual_streaming_finite_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        transport_matrix = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
        column_residual = tf.constant(0.0, DTYPE)
    elif transport_gradient_mode == MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE:
        if epsilon_tensor.shape.rank is not None and epsilon_tensor.shape.rank != 0:
            raise ValueError("manual streaming blockwise VJP finite route supports scalar epsilon only")
        epsilon0 = tf.stop_gradient(_filterflow_epsilon_start(scaled_x))
        (
            transported,
            row_residual,
        ) = _filterflow_manual_streaming_blockwise_vjp_finite_transport_stopped_scale_keys(
            scaled_x,
            x,
            logw,
            epsilon_tensor,
            epsilon0,
            scaling_tensor,
            steps=manual_streaming_blockwise_vjp_finite_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        transport_matrix = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
        column_residual = tf.constant(0.0, DTYPE)
    elif transport_plan_mode == "streaming":
        transport_matrix = tf.zeros([batch_size, 0, 0], dtype=DTYPE)
        (
            transported,
            _iterations,
            row_residual,
            column_residual,
        ) = _filterflow_streaming_transport(
            scaled_x,
            x,
            logw,
            epsilon_tensor,
            scaling_tensor,
            threshold_tensor,
            max_iterations_tensor,
            num_particles_tensor,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            transport_ad_mode=transport_ad_mode,
            warmstart_state=warmstart_state,
        )
    elif transport_gradient_mode == "filterflow_custom_op":
        transport_matrix, _iterations = _filterflow_custom_gradient_transport_matrix(
            scaled_x,
            logw,
            epsilon_tensor,
            scaling_tensor,
            threshold_tensor,
            max_iterations_tensor,
            num_particles_tensor,
        )
        transported = tf.linalg.matmul(transport_matrix, x)
        row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport_matrix, axis=2) - 1.0))
        source_weights = tf.exp(logw)
        column_mass = tf.reduce_sum(transport_matrix, axis=1)
        column_target = source_weights * tf.cast(num_particles, DTYPE)
        column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    elif transport_gradient_mode == MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE:
        if epsilon_tensor.shape.rank is not None and epsilon_tensor.shape.rank != 0:
            raise ValueError("manual dense finite route supports scalar epsilon only")
        epsilon0 = tf.stop_gradient(_filterflow_epsilon_start(scaled_x))
        transport_matrix = _filterflow_manual_dense_finite_transport_matrix_stopped_scale_keys(
            scaled_x,
            logw,
            epsilon_tensor,
            epsilon0,
            scaling_tensor,
            steps=manual_dense_finite_steps,
        )
        transported = tf.linalg.matmul(transport_matrix, x)
        row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport_matrix, axis=2) - 1.0))
        source_weights = tf.exp(logw)
        column_mass = tf.reduce_sum(transport_matrix, axis=1)
        column_target = source_weights * tf.cast(num_particles, DTYPE)
        column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    else:
        transport_matrix, _iterations = _filterflow_exact_transport_matrix(
            scaled_x,
            logw,
            epsilon_tensor,
            scaling_tensor,
            threshold_tensor,
            max_iterations_tensor,
            num_particles_tensor,
            transport_ad_mode=transport_ad_mode,
            warmstart_state=warmstart_state,
        )
        if transport_gradient_mode == "filterflow_clipped":
            transport_matrix = _clip_transport_upstream_gradient(transport_matrix)
        transported = tf.linalg.matmul(transport_matrix, x)
        row_residual = tf.reduce_max(tf.abs(tf.reduce_sum(transport_matrix, axis=2) - 1.0))
        source_weights = tf.exp(logw)
        column_mass = tf.reduce_sum(transport_matrix, axis=1)
        column_target = source_weights * tf.cast(num_particles, DTYPE)
        column_residual = tf.reduce_max(tf.abs(column_mass - column_target))
    uniform_log = uniform_log_weights(batch_size, num_particles)
    out_particles = tf.where(mask[:, None, None], transported, x)
    out_log_weights = tf.where(mask[:, None], uniform_log, logw)
    if transport_plan_mode == "dense":
        identity_transport = tf.tile(
            tf.eye(num_particles, dtype=DTYPE)[None, :, :],
            [batch_size, 1, 1],
        )
        out_transport = tf.where(mask[:, None, None], transport_matrix, identity_transport)
    else:
        out_transport = transport_matrix
    return BatchedAnnealedTransportTensors(
        particles=out_particles,
        log_weights=out_log_weights,
        transport_matrix=out_transport,
        max_row_residual=tf.cast(row_residual, DTYPE),
        max_column_residual=tf.cast(column_residual, DTYPE),
    )


def batched_ledh_pfpf_ot_value_core_tf(
    *,
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    pre_flow_particles: tf.Tensor,
    fixed_resampling_mask: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_jacobian_fn: Callable[[tf.Tensor], tf.Tensor],
    observation_residual_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    transition_log_density_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    observation_log_density_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    initial_log_weights: tf.Tensor | None = None,
    sinkhorn_epsilon: float | tf.Tensor = 0.5,
    annealed_scaling: float | tf.Tensor = 0.9,
    annealed_convergence_threshold: float | tf.Tensor = 1.0e-3,
    sinkhorn_iterations: int | tf.Tensor = 80,
    ledh_jitter: float | tf.Tensor = 1.0e-9,
    transport_gradient_mode: str = "filterflow_clipped",
    transport_plan_mode: str = "dense",
    transport_ad_mode: str = "stabilized",
    row_chunk_size: int | None = None,
    col_chunk_size: int | None = None,
) -> BatchedLEDHPFPFOTValueTensors:
    """Run fixed-branch batched LEDH-PFPF-OT value recursion."""

    observations = _to_float_tensor(observations, "observations")
    particles = _to_float_tensor(initial_particles, "initial_particles")
    pre_flow_particles = _to_float_tensor(pre_flow_particles, "pre_flow_particles")
    fixed_resampling_mask = tf.convert_to_tensor(fixed_resampling_mask, dtype=tf.bool)
    _require_static_rank(observations, 2, "observations")
    _require_static_rank(particles, 3, "initial_particles")
    _require_static_rank(pre_flow_particles, 4, "pre_flow_particles")
    _require_static_rank(fixed_resampling_mask, 2, "fixed_resampling_mask")

    batch_size, num_particles, state_dim = _static_shape(particles, "initial_particles")
    if transport_plan_mode == "streaming":
        chunks = resolve_transport_chunks(
            num_particles,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        row_chunk_size = chunks.row_chunk_size
        col_chunk_size = chunks.col_chunk_size
    time_steps, _observation_dim = _static_shape(observations, "observations")
    expected_pre_flow_shape = (batch_size, time_steps, num_particles, state_dim)
    pre_flow_shape = _static_shape(pre_flow_particles, "pre_flow_particles")
    if pre_flow_shape != expected_pre_flow_shape:
        raise ValueError(
            f"pre_flow_particles shape mismatch: got {pre_flow_shape}, "
            f"expected {expected_pre_flow_shape}"
        )
    mask_shape = _static_shape(fixed_resampling_mask, "fixed_resampling_mask")
    expected_mask_shape = (batch_size, time_steps)
    if mask_shape != expected_mask_shape:
        raise ValueError(
            f"fixed_resampling_mask shape mismatch: got {mask_shape}, "
            f"expected {expected_mask_shape}"
        )
    if initial_log_weights is None:
        log_weights = uniform_log_weights(batch_size, num_particles)
    else:
        log_weights = _to_float_tensor(initial_log_weights, "initial_log_weights")
        _require_static_rank(log_weights, 2, "initial_log_weights")
        weights_shape = _static_shape(log_weights, "initial_log_weights")
        expected_weights_shape = (batch_size, num_particles)
        if weights_shape != expected_weights_shape:
            raise ValueError(
                f"initial_log_weights shape mismatch: got {weights_shape}, "
                f"expected {expected_weights_shape}"
            )

    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    means = []
    variances = []
    esses = []
    for t in range(time_steps):
        observation = observations[t]
        ancestors = particles
        pre_flow = pre_flow_particles[:, t, :, :]
        flow = batched_ledh_flow_core_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=observation_fn,
            observation_jacobian_fn=observation_jacobian_fn,
            observation_residual_fn=observation_residual_fn,
            jitter=ledh_jitter,
        )
        post_flow = flow.post_flow_particles
        target_transition = _to_float_tensor(
            transition_log_density_fn(post_flow, ancestors, tf.constant(t, dtype=tf.int32)),
            "transition_log_density_fn output",
        )
        target_observation = _to_float_tensor(
            observation_log_density_fn(post_flow, observation, tf.constant(t, dtype=tf.int32)),
            "observation_log_density_fn output",
        )
        _require_static_rank(target_transition, 2, "transition_log_density_fn output")
        _require_static_rank(target_observation, 2, "observation_log_density_fn output")
        expected_log_density_shape = (batch_size, num_particles)
        if _static_shape(target_transition, "transition_log_density_fn output") != (
            expected_log_density_shape
        ):
            raise ValueError("transition_log_density_fn output shape mismatch")
        if _static_shape(target_observation, "observation_log_density_fn output") != (
            expected_log_density_shape
        ):
            raise ValueError("observation_log_density_fn output shape mismatch")

        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = _normalize_log_weights(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        mean, variance = _weighted_mean_and_variance(post_flow, weights)
        means.append(mean)
        variances.append(variance)
        esses.append(ess)

        transported = batched_annealed_transport_core_tf(
            post_flow,
            tf.math.log(tf.maximum(weights, _log_weight_floor())),
            fixed_resampling_mask[:, t],
            epsilon=sinkhorn_epsilon,
            scaling=annealed_scaling,
            convergence_threshold=annealed_convergence_threshold,
            max_iterations=sinkhorn_iterations,
            transport_gradient_mode=transport_gradient_mode,
            transport_plan_mode=transport_plan_mode,
            transport_ad_mode=transport_ad_mode,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        particles = transported.particles
        log_weights = transported.log_weights

    return BatchedLEDHPFPFOTValueTensors(
        log_likelihood=log_likelihood,
        filtered_means=tf.stack(means, axis=0),
        filtered_variances=tf.stack(variances, axis=0),
        ess_by_time=tf.stack(esses, axis=0),
    )


def batched_ledh_pfpf_ot_value_and_score_tf(
    theta_batch: tf.Tensor,
    value_fn: Callable[[tf.Tensor], BatchedLEDHPFPFOTValueTensors | tf.Tensor],
) -> BatchedLEDHPFPFOTValueScoreTensors:
    """Differentiate a fixed relaxed batched LEDH-PFPF-OT objective.

    The returned score is the TensorFlow autodiff gradient of
    ``sum(value_fn(theta_batch))``.  It is not a categorical classical
    particle-filter likelihood score.
    """

    theta = _to_float_tensor(theta_batch, "theta_batch")
    _require_static_rank(theta, 2, "theta_batch")
    batch_size = _static_shape(theta, "theta_batch")[0]

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value_result = value_fn(theta)
        if isinstance(value_result, BatchedLEDHPFPFOTValueTensors):
            value = value_result.log_likelihood
        else:
            value = _to_float_tensor(value_result, "value_fn output")
        _require_static_rank(value, 1, "value_fn output")
        _require_equal(_static_shape(value, "value_fn output")[0], batch_size, "value batch size")
        objective = tf.reduce_sum(value)

    score = tape.gradient(
        objective,
        theta,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    if score is None:
        score = tf.zeros_like(theta)

    return BatchedLEDHPFPFOTValueScoreTensors(
        log_likelihood=value,
        score=score,
    )


def assert_callback_has_no_forbidden_rng(
    callback: Callable[..., Any],
    *,
    name: str,
    allow_uninspectable: bool = False,
) -> None:
    """Reject callbacks whose source contains known RNG calls.

    This is a conservative contract check for deterministic fixtures.  It is
    not a proof that a callback is mathematically deterministic.
    """

    try:
        source = inspect.getsource(callback)
    except (OSError, TypeError) as exc:
        if allow_uninspectable:
            return
        raise ValueError(
            f"{name} callback source is not inspectable; hidden RNG cannot be ruled out"
        ) from exc
    for token in FORBIDDEN_RNG_SOURCE_TOKENS:
        if token in source:
            raise ValueError(f"{name} callback source contains forbidden RNG token {token!r}")


def _to_float_tensor(value: object, name: str) -> tf.Tensor:
    try:
        return tf.convert_to_tensor(value, dtype=DTYPE)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a floating Tensor") from exc


def _log_weight_floor() -> tf.Tensor:
    """Small positive floor that remains representable for the active dtype."""

    if DTYPE == tf.float32:
        return tf.constant(1.0e-30, dtype=DTYPE)
    return tf.constant(1.0e-300, dtype=DTYPE)


def _require_static_rank(tensor: tf.Tensor, rank: int, name: str) -> None:
    if tensor.shape.rank != rank:
        raise ValueError(f"{name} must have static rank {rank}")


def _static_shape(tensor: tf.Tensor, name: str) -> tuple[int, ...]:
    shape = tensor.shape.as_list()
    if any(dim is None for dim in shape):
        raise ValueError(f"{name} must have fully static shape")
    return tuple(int(dim) for dim in shape)


def _require_equal(actual: int, expected: int, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: got {actual}, expected {expected}")


def _require_square_batch(tensor: tf.Tensor, batch_size: int, dim: int, name: str) -> None:
    shape = _static_shape(tensor, name)
    expected = (batch_size, dim, dim)
    if shape != expected:
        raise ValueError(f"{name} shape mismatch: got {shape}, expected {expected}")


def _tile_eye(batch_size: int, dim: int) -> tf.Tensor:
    return tf.tile(tf.eye(dim, dtype=DTYPE)[None, :, :], [batch_size, 1, 1])


def _stabilize_batch_covariance(
    covariance: object,
    jitter: float | tf.Tensor,
    name: str,
) -> tf.Tensor:
    matrix = _to_float_tensor(covariance, name)
    _require_static_rank(matrix, matrix.shape.rank or 0, name)
    if matrix.shape.rank is None or matrix.shape.rank < 2:
        raise ValueError(f"{name} must have rank at least 2")
    shape = matrix.shape.as_list()
    if shape[-1] is None or shape[-2] is None or shape[-1] != shape[-2]:
        raise ValueError(f"{name} must have static square trailing dimensions")
    sym = 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))
    eigvals = tf.linalg.eigvalsh(sym)
    min_eig = tf.reduce_min(eigvals, axis=-1)
    jitter_tensor = tf.convert_to_tensor(jitter, dtype=DTYPE)
    needed = tf.maximum(jitter_tensor - min_eig, tf.zeros_like(min_eig))
    eye = tf.eye(int(shape[-1]), dtype=DTYPE)
    return sym + needed[..., None, None] * eye


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = _to_float_tensor(residuals, "residuals")
    covariance = _to_float_tensor(covariance, "covariance")
    _require_static_rank(residuals, 3, "residuals")
    _require_static_rank(covariance, 3, "covariance")
    batch_size, _num_particles, dim = _static_shape(residuals, "residuals")
    _require_square_batch(covariance, batch_size, dim, "covariance")
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals))
    )
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim_tensor = tf.cast(dim, DTYPE)
    return -0.5 * (
        dim_tensor * tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE))
        + logdet[:, None]
        + quad
    )


def _normalize_log_weights(log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    log_weights = _to_float_tensor(log_weights, "log_weights")
    log_normalizer = tf.reduce_logsumexp(log_weights, axis=1)
    normalized_log_weights = log_weights - log_normalizer[:, None]
    return tf.exp(normalized_log_weights), log_normalizer


def _weighted_mean_and_variance(
    particles: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_sum(particles * weights[:, :, None], axis=1)
    centered = particles - mean[:, None, :]
    variance = tf.reduce_sum(centered * centered * weights[:, :, None], axis=1)
    return mean, variance


def _require_observation_callback_shapes(
    *,
    h_ref: tf.Tensor,
    h_jac: tf.Tensor,
    residual: tf.Tensor,
    batch_size: int,
    num_particles: int,
    state_dim: int,
    observation_dim: int,
) -> None:
    expected_observation_shape = (batch_size, num_particles, observation_dim)
    for name, tensor in (
        ("observation_fn output", h_ref),
        ("observation_residual_fn output", residual),
    ):
        shape = _static_shape(tensor, name)
        if shape != expected_observation_shape:
            raise ValueError(
                f"{name} shape mismatch: got {shape}, expected {expected_observation_shape}"
            )
    jac_shape = _static_shape(h_jac, "observation_jacobian_fn output")
    expected_jac_shape = (batch_size, num_particles, observation_dim, state_dim)
    if jac_shape != expected_jac_shape:
        raise ValueError(
            "observation_jacobian_fn output shape mismatch: "
            f"got {jac_shape}, expected {expected_jac_shape}"
        )


def _require_flow_shapes(
    *,
    post_flow_particles: tf.Tensor,
    pre_flow_log_density: tf.Tensor,
    forward_log_det: tf.Tensor,
    local_posterior_means: tf.Tensor,
    local_posterior_covariances: tf.Tensor,
) -> None:
    batch_size, num_particles, state_dim = _static_shape(
        post_flow_particles,
        "post_flow_particles",
    )
    _require_equal(
        _static_shape(pre_flow_log_density, "pre_flow_log_density")[0],
        batch_size,
        "pre_flow_log_density batch size",
    )
    _require_equal(
        _static_shape(pre_flow_log_density, "pre_flow_log_density")[1],
        num_particles,
        "pre_flow_log_density particle count",
    )
    _require_equal(
        _static_shape(forward_log_det, "forward_log_det")[0],
        batch_size,
        "forward_log_det batch size",
    )
    _require_equal(
        _static_shape(forward_log_det, "forward_log_det")[1],
        num_particles,
        "forward_log_det particle count",
    )
    _require_equal(
        _static_shape(local_posterior_means, "local_posterior_means")[0],
        batch_size,
        "local_posterior_means batch size",
    )
    _require_equal(
        _static_shape(local_posterior_means, "local_posterior_means")[1],
        num_particles,
        "local_posterior_means particle count",
    )
    _require_equal(
        _static_shape(local_posterior_means, "local_posterior_means")[2],
        state_dim,
        "local_posterior_means state dim",
    )
    cov_shape = _static_shape(local_posterior_covariances, "local_posterior_covariances")
    expected_cov_shape = (batch_size, num_particles, state_dim, state_dim)
    if cov_shape != expected_cov_shape:
        raise ValueError(
            "local_posterior_covariances shape mismatch: "
            f"got {cov_shape}, expected {expected_cov_shape}"
        )
