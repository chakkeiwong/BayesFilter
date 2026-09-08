"""C2 adapter for the generic per-ancestor UKF/APF candidate.

The generic sigma-point and sampling primitives live in
``c2_mixture_ukf_apf_tf``.  This module only binds the C2 stochastic-volatility
model to those primitives and prepares a frozen branch for the repository's
exact target/analytical-score evaluator.  The transformed log-square
observation is a proposal coordinate; it never replaces the raw C2 likelihood
in the finite-program numerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_tf import (
    BatchedUKFConfig,
    DTYPE,
    complete_gaussian_mixture_log_density,
    gaussian_log_density,
    make_batched_ukf_kernel,
    make_gaussian_student_defensive_apf_sampler_from_random_inputs,
    make_k_mixture_apf_sampler_from_random_inputs,
    make_k1_apf_sampler_from_random_inputs,
    require_valid_ukf_result,
    student_log_density,
)
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
)
from bayesfilter.highdim.c2_transformed_observation_student_proposal_tf import (
    GUIDE_CONVENTION_ID,
    LOG_CHI_SQUARE_VARIANCE,
    transformed_log_square_observation,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    FrozenProposalAPFProgram,
    PreparedFrozenProposalBranch,
    prepare_frozen_proposal_apf_program,
    prepare_frozen_proposal_branch,
)


ROUTE_ID = "c2_per_ancestor_ukf_apf_k1_frozen_branch_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"
MIXTURE_ROUTE_ID = "c2_per_ancestor_ukf_apf_fixed_mixture_frozen_branch_v1"
MIXTURE_ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"
DEFENSIVE_MIXTURE_ROUTE_ID = (
    "c2_per_ancestor_ukf_apf_smooth_student_defensive_frozen_branch_v1"
)
DEFENSIVE_MIXTURE_ROUTE_CLASSIFICATION = (
    "extension_or_invention_candidate_diagnostic_only"
)


@dataclass(frozen=True)
class C2UKFAPFCompilation:
    """A frozen C2 branch and its UKF/APF construction record."""

    branch: PreparedFrozenProposalBranch
    compiler_id: str
    manifest: Mapping[str, object]
    proposal_diagnostics: tuple[Mapping[str, object], ...]
    program: FrozenProposalAPFProgram = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.branch, PreparedFrozenProposalBranch):
            raise TypeError("branch must be a PreparedFrozenProposalBranch")
        if len(str(self.compiler_id)) != 64:
            raise ValueError("compiler_id must be a SHA-256 digest")
        object.__setattr__(self, "manifest", dict(self.manifest))
        object.__setattr__(
            self,
            "proposal_diagnostics",
            tuple(dict(row) for row in self.proposal_diagnostics),
        )

    def bind_program(
        self, model: C2StochasticVolatilityFrozenAPFModel
    ) -> FrozenProposalAPFProgram:
        """Bind the exact model evaluator to this already-frozen branch."""

        return prepare_frozen_proposal_apf_program(model, self.branch)


def compile_c2_per_ancestor_ukf_apf_k1(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    particle_count: int,
    seed: int,
    jit_compile: bool = True,
    ukf_config: BatchedUKFConfig | None = None,
) -> C2UKFAPFCompilation:
    """Construct a transformed-observation, per-ancestor K=1 APF branch.

    The branch construction is intentionally eager and time-recursive: each
    step uses the exact finite-program prefix weights to form the APF ancestor
    law.  All row-wise sigma-point arithmetic and random sampling are executed
    by fixed-shape TensorFlow functions supplied by the generic module.
    """

    if not tf.executing_eagerly():
        raise RuntimeError("compile the C2 UKF/APF branch before tracing")
    observed = tf.convert_to_tensor(observations, DTYPE)
    theta = tf.ensure_shape(tf.convert_to_tensor(theta_reference, DTYPE), [2])
    if (
        observed.shape.rank != 2
        or not observed.shape.is_fully_defined()
        or observed.shape[0] < 2
        or observed.shape[1] != model.observation_dim()
    ):
        raise ValueError("observations must have static shape [time, state]")
    count = int(particle_count)
    if count < 2:
        raise ValueError("particle_count must be at least two")
    if not isinstance(model, C2StochasticVolatilityFrozenAPFModel):
        raise TypeError("model must be a C2StochasticVolatilityFrozenAPFModel")
    stability = model.stability_diagnostics(theta)
    if float(stability["spectral_radius"].numpy()) >= 1.0:
        raise ValueError("theta_reference is outside the stationary stability domain")

    dimension = model.state_dim()
    transition = model.transition_matrix(theta)
    process_covariance = tf.eye(dimension, dtype=DTYPE) * float(model.sigma) ** 2
    stationary_covariance, _ = model.stationary_covariance_and_derivative(theta)
    initial_cholesky = tf.linalg.cholesky(stationary_covariance)
    initial_normal = tf.random.stateless_normal(
        [count, dimension], [int(seed), 1001], dtype=DTYPE
    )
    initial_states = tf.einsum("ij,nj->ni", initial_cholesky, initial_normal)
    initial_log_q = model.initial_log_density(theta, initial_states)
    initial_covariances = tf.broadcast_to(
        stationary_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )
    process_covariances = tf.broadcast_to(
        process_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )
    observation_covariance = tf.eye(dimension, dtype=DTYPE) * tf.constant(
        LOG_CHI_SQUARE_VARIANCE, DTYPE
    )
    observation_covariances = tf.broadcast_to(
        observation_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )

    def transition_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,bpj->bpi", transition, points)

    # In the transformed guide, z is modelled as x plus log-chi-square noise.
    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.identity(points)

    kernel = make_batched_ukf_kernel(
        batch_size=count,
        state_dim=dimension,
        observation_dim=dimension,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        config=ukf_config,
        jit_compile=bool(jit_compile),
    )
    sampler = make_k1_apf_sampler_from_random_inputs(
        batch_size=count,
        state_dim=dimension,
        jit_compile=bool(jit_compile),
    )

    states = [initial_states]
    covariances = initial_covariances
    ancestors = []
    auxiliary_log_probabilities = []
    transition_log_q = []
    diagnostics = []

    branch = prepare_frozen_proposal_branch(
        observations=observed[:1],
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.zeros([0, count], tf.int32),
        auxiliary_log_probabilities=tf.zeros([0, count], DTYPE),
        transition_log_proposal_density=tf.zeros([0, count], DTYPE),
    )
    exact_prefix = prepare_frozen_proposal_apf_program(model, branch).evaluate(theta)
    if not bool(exact_prefix["finite"].numpy()):
        raise ValueError("initial exact C2 prefix is non-finite")
    log_parent_weights = exact_prefix["final_log_weights"]

    for time_index in range(1, int(observed.shape[0])):
        transformed = transformed_log_square_observation(observed[time_index], theta)
        ukf = kernel(
            states[-1],
            covariances,
            process_covariances,
            transformed,
            observation_covariances,
        )
        require_valid_ukf_result(ukf)
        categorical_uniforms = tf.random.stateless_uniform(
            [count], [int(seed), 5100 + 41 * time_index], dtype=DTYPE
        )
        standard_normal = tf.random.stateless_normal(
            [count, dimension], [int(seed), 5200 + 43 * time_index], dtype=DTYPE
        )
        step = sampler(
            ukf["posterior_mean"],
            ukf["posterior_covariance"],
            ukf["posterior_cholesky"],
            ukf["innovation_log_likelihood"],
            log_parent_weights,
            categorical_uniforms,
            standard_normal,
        )
        if not bool(step["finite"].numpy()):
            raise ValueError(f"non-finite K=1 APF output at time {time_index}")

        states.append(step["samples"])
        covariances = step["next_covariances"]
        ancestors.append(step["ancestor_indices"])
        auxiliary_log_probabilities.append(step["log_ancestor_probabilities"])
        transition_log_q.append(step["complete_log_q"])

        branch = prepare_frozen_proposal_branch(
            observations=observed[: time_index + 1],
            states=tf.stack(states),
            initial_log_proposal_density=initial_log_q,
            ancestors=tf.stack(ancestors),
            auxiliary_log_probabilities=tf.stack(auxiliary_log_probabilities),
            transition_log_proposal_density=tf.stack(transition_log_q),
        )
        exact_prefix = prepare_frozen_proposal_apf_program(model, branch).evaluate(theta)
        if not bool(exact_prefix["finite"].numpy()):
            raise ValueError(f"exact C2 prefix is non-finite at time {time_index}")
        log_parent_weights = exact_prefix["final_log_weights"]

        diagnostics.append(
            {
                "time_index": time_index,
                "transformed_observation": transformed,
                "lookahead_log_likelihood": ukf["innovation_log_likelihood"],
                "ancestor_log_probabilities": step["log_ancestor_probabilities"],
                "selected_mean": step["selected_mean"],
                "selected_cholesky": step["selected_cholesky"],
                "selected_log_q": step["selected_log_q"],
                "proposal_density_recomposition_max_abs": tf.reduce_max(
                    tf.abs(
                        step["complete_log_q"]
                        - gaussian_log_density(
                            step["samples"],
                            step["selected_mean"],
                            step["selected_cholesky"],
                        )
                    )
                ),
                "posterior_mean": ukf["posterior_mean"],
                "posterior_min_eigenvalue": ukf["posterior_min_eigenvalue"],
                "proposal_finite": step["finite"],
                "exact_prefix_finite": exact_prefix["finite"],
            }
        )

    manifest = {
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "guide_convention_id": GUIDE_CONVENTION_ID,
        "particle_count": count,
        "horizon": int(observed.shape[0]),
        "state_dimension": dimension,
        "theta_reference": theta,
        "seed": int(seed),
        "jit_compile_default": True,
        "jit_compile_requested": bool(jit_compile),
        "proposal_family": "per_ancestor_gaussian_ukf_posterior_k1",
        "proposal_density": "complete_conditional_gaussian_density",
        "auxiliary_law": "exact_frozen_prefix_log_weights_plus_ukf_lookahead",
        "covariance_lifecycle": "selected_posterior_covariance_carried_to_next_step",
        "observation_guide": "identity_latent_state_plus_log_chi_square_noise",
        "process_covariance": process_covariance,
        "observation_covariance": observation_covariance,
        "model": model.manifest_payload(),
        "branch_id": branch.branch_id,
        "runtime_branch_parameter_dependence": "none_after_compilation",
        "score_contract": "exact_frozen_branch_analytical_recursive_score",
        "random_inputs": "stateless_uniform_and_normal_frozen_outside_compiled_sampler",
        "exact_pseudo_marginal_claimed": False,
        "status": "candidate_diagnostic_only",
        "nonclaims": (
            "no posterior correctness claim",
            "no low-variance claim",
            "no production readiness claim",
            "no adaptive-total-gradient claim",
        ),
        "stability": stability,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(_jsonable_manifest(manifest), sort_keys=True).encode("utf-8"))
    digest.update(branch.branch_id.encode("ascii"))
    compiler_id = digest.hexdigest()
    return C2UKFAPFCompilation(
        branch=branch,
        compiler_id=compiler_id,
        manifest=manifest,
        proposal_diagnostics=tuple(diagnostics),
    )


def _jsonable_manifest(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable_manifest(value.numpy().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable_manifest(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable_manifest(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest contains a non-finite float")
        return value
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def compile_c2_per_ancestor_ukf_apf_mixture(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    particle_count: int,
    seed: int,
    component_count: int,
    offset: float = 0.35,
    jit_compile: bool = True,
):
    """Construct a fixed-topology K=2 or K=4 per-ancestor UKF/APF branch.

    The component offsets use the first one or two columns of each UKF
    Cholesky factor.  Equal weights and a common covariance
    ``P - delta**2 sum(d_k d_k^T)`` preserve each UKF posterior's mean and
    covariance while making ``delta`` dimensionless.  For ``0 < delta < 1``
    the residual covariance is positive definite; the covariance is still
    checked at every time step.  All random inputs and the component topology
    are frozen before the exact evaluator is bound, so the resulting branch
    has the same analytical-score contract as the K=1 route.
    """

    if not tf.executing_eagerly():
        raise RuntimeError("compile the C2 UKF/APF mixture branch before tracing")
    component_count = int(component_count)
    if component_count not in (2, 4):
        raise ValueError("fixed mixture component_count must be 2 or 4")
    offset = float(offset)
    if not math.isfinite(offset) or not (0.0 < offset < 1.0):
        raise ValueError("mixture offset must satisfy 0 < offset < 1")
    observed = tf.convert_to_tensor(observations, DTYPE)
    theta = tf.ensure_shape(tf.convert_to_tensor(theta_reference, DTYPE), [2])
    if (
        observed.shape.rank != 2
        or not observed.shape.is_fully_defined()
        or observed.shape[0] < 2
        or observed.shape[1] != model.observation_dim()
    ):
        raise ValueError("observations must have static shape [time, state]")
    count = int(particle_count)
    if count < 2:
        raise ValueError("particle_count must be at least two")
    if not isinstance(model, C2StochasticVolatilityFrozenAPFModel):
        raise TypeError("model must be a C2StochasticVolatilityFrozenAPFModel")
    stability = model.stability_diagnostics(theta)
    if float(stability["spectral_radius"].numpy()) >= 1.0:
        raise ValueError("theta_reference is outside the stationary stability domain")

    dimension = model.state_dim()
    if dimension < 2 and component_count == 4:
        raise ValueError("K=4 fixed split requires state dimension at least two")
    transition = model.transition_matrix(theta)
    process_covariance = tf.eye(dimension, dtype=DTYPE) * float(model.sigma) ** 2
    stationary_covariance, _ = model.stationary_covariance_and_derivative(theta)
    initial_cholesky = tf.linalg.cholesky(stationary_covariance)
    initial_normal = tf.random.stateless_normal(
        [count, dimension], [int(seed), 1001], dtype=DTYPE
    )
    initial_states = tf.einsum("ij,nj->ni", initial_cholesky, initial_normal)
    initial_log_q = model.initial_log_density(theta, initial_states)
    initial_covariances = tf.broadcast_to(
        stationary_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )
    process_covariances = tf.broadcast_to(
        process_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )
    observation_covariance = tf.eye(dimension, dtype=DTYPE) * tf.constant(
        LOG_CHI_SQUARE_VARIANCE, DTYPE
    )
    observation_covariances = tf.broadcast_to(
        observation_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )

    def transition_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,bpj->bpi", transition, points)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.identity(points)

    kernel = make_batched_ukf_kernel(
        batch_size=count,
        state_dim=dimension,
        observation_dim=dimension,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        config=None,
        jit_compile=bool(jit_compile),
    )
    sampler = make_k_mixture_apf_sampler_from_random_inputs(
        batch_size=count,
        state_dim=dimension,
        component_count=component_count,
        jit_compile=bool(jit_compile),
    )

    # Fixed sign topology: K=2 uses +/-d_1; K=4 uses all sign combinations of
    # d_1 and d_2.  The directions are Cholesky columns, so their scale is
    # inherited from the current UKF covariance rather than from coordinates.
    direction_one_index = 0
    direction_two_index = 1
    if component_count == 2:
        signs = ((1.0, 0.0), (-1.0, 0.0))
    else:
        signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    sign_tensor = tf.constant(signs, DTYPE)
    component_weights = tf.fill(
        [count, component_count], tf.constant(1.0 / component_count, DTYPE)
    )

    states = [initial_states]
    covariances = initial_covariances
    ancestors = []
    auxiliary_log_probabilities = []
    transition_log_q = []
    diagnostics = []

    branch = prepare_frozen_proposal_branch(
        observations=observed[:1],
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.zeros([0, count], tf.int32),
        auxiliary_log_probabilities=tf.zeros([0, count], DTYPE),
        transition_log_proposal_density=tf.zeros([0, count], DTYPE),
    )
    exact_prefix = prepare_frozen_proposal_apf_program(model, branch).evaluate(theta)
    if not bool(exact_prefix["finite"].numpy()):
        raise ValueError("initial exact C2 prefix is non-finite")
    log_parent_weights = exact_prefix["final_log_weights"]

    for time_index in range(1, int(observed.shape[0])):
        transformed = transformed_log_square_observation(observed[time_index], theta)
        ukf = kernel(
            states[-1],
            covariances,
            process_covariances,
            transformed,
            observation_covariances,
        )
        require_valid_ukf_result(ukf)
        posterior_cholesky = ukf["posterior_cholesky"]
        direction_one = posterior_cholesky[:, :, direction_one_index]
        direction_two = (
            posterior_cholesky[:, :, direction_two_index]
            if dimension >= 2
            else tf.zeros_like(direction_one)
        )
        between_component_covariance = offset**2 * tf.einsum(
            "bi,bj->bij", direction_one, direction_one
        )
        if component_count == 4:
            between_component_covariance = between_component_covariance + offset**2 * tf.einsum(
                "bi,bj->bij", direction_two, direction_two
            )
        component_covariance = 0.5 * (
            ukf["posterior_covariance"] - between_component_covariance
            + tf.linalg.matrix_transpose(
                ukf["posterior_covariance"] - between_component_covariance
            )
        )
        component_minimum = tf.reduce_min(
            tf.linalg.eigvalsh(component_covariance), axis=1
        )
        if not bool(tf.reduce_all(component_minimum > 0.0).numpy()):
            raise ValueError(f"K={component_count} split covariance is not SPD at time {time_index}")
        component_chol_single = tf.linalg.cholesky(component_covariance)
        component_means = ukf["posterior_mean"][:, tf.newaxis, :] + offset * (
            sign_tensor[tf.newaxis, :, 0, tf.newaxis] * direction_one[:, tf.newaxis, :]
            + sign_tensor[tf.newaxis, :, 1, tf.newaxis] * direction_two[:, tf.newaxis, :]
        )
        component_cholesky = tf.broadcast_to(
            component_chol_single[:, tf.newaxis, :, :],
            [count, component_count, dimension, dimension],
        )
        ancestor_uniforms = tf.random.stateless_uniform(
            [count], [int(seed), 6100 + 41 * time_index], dtype=DTYPE
        )
        component_uniforms = tf.random.stateless_uniform(
            [count], [int(seed), 6200 + 43 * time_index], dtype=DTYPE
        )
        standard_normal = tf.random.stateless_normal(
            [count, dimension], [int(seed), 6300 + 47 * time_index], dtype=DTYPE
        )
        step = sampler(
            ukf["posterior_covariance"],
            component_means,
            component_cholesky,
            component_weights,
            ukf["innovation_log_likelihood"],
            log_parent_weights,
            ancestor_uniforms,
            component_uniforms,
            standard_normal,
        )
        if not bool(step["finite"].numpy()):
            raise ValueError(f"non-finite K={component_count} APF output at time {time_index}")
        selected_component_means = tf.gather(component_means, step["ancestor_indices"])
        selected_component_cholesky = tf.gather(
            component_cholesky, step["ancestor_indices"]
        )
        selected_weights = tf.gather(component_weights, step["ancestor_indices"])
        recomposed_mean = tf.reduce_sum(
            selected_weights[:, :, tf.newaxis] * selected_component_means, axis=1
        )
        centered = selected_component_means - recomposed_mean[:, tf.newaxis, :]
        recomposed_covariance = tf.reduce_sum(
            selected_weights[:, :, tf.newaxis, tf.newaxis]
            * (
                tf.matmul(selected_component_cholesky, selected_component_cholesky, transpose_b=True)
                + tf.matmul(centered[:, :, :, tf.newaxis], centered[:, :, tf.newaxis, :])
            ),
            axis=1,
        )
        moment_error = tf.reduce_max(
            tf.abs(recomposed_covariance - tf.gather(ukf["posterior_covariance"], step["ancestor_indices"]))
        )
        reverse = tf.range(component_count - 1, -1, -1)
        permuted_log_q = complete_gaussian_mixture_log_density(
            step["samples"],
            tf.gather(selected_component_means, reverse, axis=1),
            tf.gather(selected_component_cholesky, reverse, axis=1),
            tf.gather(selected_weights, reverse, axis=1),
        )
        label_permutation_error = tf.reduce_max(
            tf.abs(step["complete_log_q"] - permuted_log_q)
        )
        states.append(step["samples"])
        covariances = step["next_covariances"]
        ancestors.append(step["ancestor_indices"])
        auxiliary_log_probabilities.append(step["log_ancestor_probabilities"])
        transition_log_q.append(step["complete_log_q"])

        branch = prepare_frozen_proposal_branch(
            observations=observed[: time_index + 1],
            states=tf.stack(states),
            initial_log_proposal_density=initial_log_q,
            ancestors=tf.stack(ancestors),
            auxiliary_log_probabilities=tf.stack(auxiliary_log_probabilities),
            transition_log_proposal_density=tf.stack(transition_log_q),
        )
        exact_prefix = prepare_frozen_proposal_apf_program(model, branch).evaluate(theta)
        if not bool(exact_prefix["finite"].numpy()):
            raise ValueError(f"exact C2 prefix is non-finite at time {time_index}")
        log_parent_weights = exact_prefix["final_log_weights"]
        diagnostics.append(
            {
                "time_index": time_index,
                "component_count": component_count,
                "offset": tf.constant(offset, DTYPE),
                "transformed_observation": transformed,
                "lookahead_log_likelihood": ukf["innovation_log_likelihood"],
                "ancestor_log_probabilities": step["log_ancestor_probabilities"],
                "component_indices": step["component_indices"],
                "component_minimum_eigenvalue": component_minimum,
                "moment_recomposition_max_abs": moment_error,
                "label_permutation_max_abs": label_permutation_error,
                "proposal_density_recomposition_max_abs": tf.reduce_max(
                    tf.abs(
                        step["complete_log_q"]
                        - complete_gaussian_mixture_log_density(
                            step["samples"],
                            selected_component_means,
                            selected_component_cholesky,
                            selected_weights,
                        )
                    )
                ),
                "posterior_mean": ukf["posterior_mean"],
                "posterior_min_eigenvalue": ukf["posterior_min_eigenvalue"],
                "proposal_finite": step["finite"],
                "exact_prefix_finite": exact_prefix["finite"],
            }
        )

    manifest = {
        "route_id": MIXTURE_ROUTE_ID,
        "route_classification": MIXTURE_ROUTE_CLASSIFICATION,
        "component_count": component_count,
        "offset": offset,
        "offset_topology": "cholesky_column_sign_split_d1_or_d1_d2",
        "component_weights": "equal_fixed",
        "proposal_family": f"per_ancestor_gaussian_ukf_posterior_k{component_count}",
        "proposal_density": "complete_conditional_gaussian_mixture_density",
        "auxiliary_law": "exact_frozen_prefix_log_weights_plus_ukf_lookahead",
        "covariance_lifecycle": "selected_posterior_covariance_carried_to_next_step",
        "observation_guide": "identity_latent_state_plus_log_chi_square_noise",
        "particle_count": count,
        "horizon": int(observed.shape[0]),
        "state_dimension": dimension,
        "theta_reference": theta,
        "seed": int(seed),
        "jit_compile_default": True,
        "jit_compile_requested": bool(jit_compile),
        "process_covariance": process_covariance,
        "observation_covariance": observation_covariance,
        "model": model.manifest_payload(),
        "branch_id": branch.branch_id,
        "runtime_branch_parameter_dependence": "none_after_compilation",
        "score_contract": "exact_frozen_branch_analytical_recursive_score",
        "random_inputs": "stateless_uniform_and_normal_frozen_outside_compiled_sampler",
        "exact_pseudo_marginal_claimed": False,
        "status": "candidate_diagnostic_only",
        "nonclaims": (
            "no posterior correctness claim",
            "no low-variance claim",
            "no production readiness claim",
            "no adaptive-total-gradient claim",
        ),
        "stability": stability,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(_jsonable_manifest(manifest), sort_keys=True).encode("utf-8"))
    digest.update(branch.branch_id.encode("ascii"))
    return C2UKFAPFCompilation(
        branch=branch,
        compiler_id=digest.hexdigest(),
        manifest=manifest,
        proposal_diagnostics=tuple(diagnostics),
    )


def compile_c2_per_ancestor_ukf_apf_defensive_mixture(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    particle_count: int,
    seed: int,
    local_component_count: int = 1,
    offset: float = 0.5,
    nu: float = 8.0,
    epsilon_min: float = 0.05,
    epsilon_max: float = 0.20,
    gate_center: float | None = None,
    gate_temperature: float | None = None,
    jit_compile: bool = True,
    ukf_config: BatchedUKFConfig | None = None,
) -> C2UKFAPFCompilation:
    """Construct a smooth, full-support Student-defensive UKF/APF branch.

    The local proposal is K=1, K=2, or K=4 Gaussian UKF components.  A
    Student component centered at the UKF posterior is mixed in with a
    per-ancestor sigmoid fraction based on the normalized UKF innovation.  All
    proposal parameters and random tensors are frozen before the shared exact
    evaluator is bound.
    """

    if not tf.executing_eagerly():
        raise RuntimeError("compile the C2 defensive branch before tracing")
    local_component_count = int(local_component_count)
    if local_component_count not in (1, 2, 4):
        raise ValueError("local_component_count must be one of 1, 2, or 4")
    nu = float(nu)
    if not math.isfinite(nu) or nu <= 2.0:
        raise ValueError("defensive Student nu must be finite and greater than two")
    epsilon_min = float(epsilon_min)
    epsilon_max = float(epsilon_max)
    if (
        not math.isfinite(epsilon_min)
        or not math.isfinite(epsilon_max)
        or not (0.0 < epsilon_min <= epsilon_max < 1.0)
    ):
        raise ValueError("defensive epsilon bounds must satisfy 0 < min <= max < 1")
    observed = tf.convert_to_tensor(observations, DTYPE)
    theta = tf.ensure_shape(tf.convert_to_tensor(theta_reference, DTYPE), [2])
    if (
        observed.shape.rank != 2
        or not observed.shape.is_fully_defined()
        or observed.shape[0] < 2
        or observed.shape[1] != model.observation_dim()
    ):
        raise ValueError("observations must have static shape [time, state]")
    count = int(particle_count)
    if count < 2:
        raise ValueError("particle_count must be at least two")
    if not isinstance(model, C2StochasticVolatilityFrozenAPFModel):
        raise TypeError("model must be a C2StochasticVolatilityFrozenAPFModel")
    dimension = model.state_dim()
    center = float(dimension if gate_center is None else gate_center)
    temperature = float(2.0 * dimension if gate_temperature is None else gate_temperature)
    if not math.isfinite(center) or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("gate center must be finite and temperature must be positive")
    if local_component_count in (2, 4):
        offset = float(offset)
        if not math.isfinite(offset) or not (0.0 < offset < 1.0):
            raise ValueError("mixture offset must satisfy 0 < offset < 1")

    stability = model.stability_diagnostics(theta)
    if float(stability["spectral_radius"].numpy()) >= 1.0:
        raise ValueError("theta_reference is outside the stationary stability domain")
    transition = model.transition_matrix(theta)
    process_covariance = tf.eye(dimension, dtype=DTYPE) * float(model.sigma) ** 2
    stationary_covariance, _ = model.stationary_covariance_and_derivative(theta)
    initial_cholesky = tf.linalg.cholesky(stationary_covariance)
    initial_normal = tf.random.stateless_normal(
        [count, dimension], [int(seed), 1101], dtype=DTYPE
    )
    initial_states = tf.einsum("ij,nj->ni", initial_cholesky, initial_normal)
    initial_log_q = model.initial_log_density(theta, initial_states)
    initial_covariances = tf.broadcast_to(
        stationary_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )
    process_covariances = tf.broadcast_to(
        process_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )
    observation_covariance = tf.eye(dimension, dtype=DTYPE) * tf.constant(
        LOG_CHI_SQUARE_VARIANCE, DTYPE
    )
    observation_covariances = tf.broadcast_to(
        observation_covariance[tf.newaxis, :, :], [count, dimension, dimension]
    )

    def transition_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,bpj->bpi", transition, points)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.identity(points)

    kernel = make_batched_ukf_kernel(
        batch_size=count,
        state_dim=dimension,
        observation_dim=dimension,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        config=ukf_config,
        jit_compile=bool(jit_compile),
    )
    sampler = make_gaussian_student_defensive_apf_sampler_from_random_inputs(
        batch_size=count,
        state_dim=dimension,
        local_component_count=local_component_count,
        nu=nu,
        jit_compile=bool(jit_compile),
    )
    local_weights = tf.fill(
        [count, local_component_count],
        tf.constant(1.0 / local_component_count, DTYPE),
    )
    if local_component_count == 1:
        signs = ((1.0, 0.0),)
    elif local_component_count == 2:
        signs = ((1.0, 0.0), (-1.0, 0.0))
    else:
        signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    sign_tensor = tf.constant(signs, DTYPE)

    states = [initial_states]
    covariances = initial_covariances
    ancestors = []
    auxiliary_log_probabilities = []
    transition_log_q = []
    diagnostics = []
    branch = prepare_frozen_proposal_branch(
        observations=observed[:1],
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=tf.zeros([0, count], tf.int32),
        auxiliary_log_probabilities=tf.zeros([0, count], DTYPE),
        transition_log_proposal_density=tf.zeros([0, count], DTYPE),
    )
    exact_prefix = prepare_frozen_proposal_apf_program(model, branch).evaluate(theta)
    if not bool(exact_prefix["finite"].numpy()):
        raise ValueError("initial exact C2 prefix is non-finite")
    log_parent_weights = exact_prefix["final_log_weights"]

    for time_index in range(1, int(observed.shape[0])):
        transformed = transformed_log_square_observation(observed[time_index], theta)
        ukf = kernel(
            states[-1],
            covariances,
            process_covariances,
            transformed,
            observation_covariances,
        )
        require_valid_ukf_result(ukf)
        posterior_cholesky = ukf["posterior_cholesky"]
        direction_one = posterior_cholesky[:, :, 0]
        direction_two = (
            posterior_cholesky[:, :, 1]
            if dimension >= 2
            else tf.zeros_like(direction_one)
        )
        between_component_covariance = tf.zeros_like(ukf["posterior_covariance"])
        if local_component_count in (2, 4):
            between_component_covariance = offset**2 * tf.einsum(
                "bi,bj->bij", direction_one, direction_one
            )
        if local_component_count == 4:
            between_component_covariance = between_component_covariance + offset**2 * tf.einsum(
                "bi,bj->bij", direction_two, direction_two
            )
        component_covariance = 0.5 * (
            ukf["posterior_covariance"]
            - between_component_covariance
            + tf.linalg.matrix_transpose(
                ukf["posterior_covariance"] - between_component_covariance
            )
        )
        component_minimum = tf.reduce_min(
            tf.linalg.eigvalsh(component_covariance), axis=1
        )
        if not bool(tf.reduce_all(component_minimum > 0.0).numpy()):
            raise ValueError(
                f"local K={local_component_count} covariance is not SPD at time {time_index}"
            )
        component_chol_single = tf.linalg.cholesky(component_covariance)
        local_means = ukf["posterior_mean"][:, tf.newaxis, :] + offset * (
            sign_tensor[tf.newaxis, :, 0, tf.newaxis] * direction_one[:, tf.newaxis, :]
            + sign_tensor[tf.newaxis, :, 1, tf.newaxis] * direction_two[:, tf.newaxis, :]
        )
        local_cholesky = tf.broadcast_to(
            component_chol_single[:, tf.newaxis, :, :],
            [count, local_component_count, dimension, dimension],
        )
        if local_component_count == 1:
            local_means = ukf["posterior_mean"][:, tf.newaxis, :]
            local_cholesky = posterior_cholesky[:, tf.newaxis, :, :]

        innovation_solution = tf.linalg.cholesky_solve(
            ukf["innovation_cholesky"], ukf["innovation"][:, :, tf.newaxis]
        )[:, :, 0]
        innovation_quadratic = tf.reduce_sum(
            ukf["innovation"] * innovation_solution, axis=1
        )
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * tf.math.sigmoid(
            (innovation_quadratic - tf.constant(center, DTYPE))
            / tf.constant(temperature, DTYPE)
        )
        defensive_scale = (nu - 2.0) / nu * ukf["posterior_covariance"]
        defensive_cholesky = tf.linalg.cholesky(defensive_scale)
        ancestor_uniforms = tf.random.stateless_uniform(
            [count], [int(seed), 7100 + 41 * time_index], dtype=DTYPE
        )
        component_uniforms = tf.random.stateless_uniform(
            [count], [int(seed), 7200 + 43 * time_index], dtype=DTYPE
        )
        gaussian_normal = tf.random.stateless_normal(
            [count, dimension], [int(seed), 7300 + 47 * time_index], dtype=DTYPE
        )
        student_normal = tf.random.stateless_normal(
            [count, dimension], [int(seed), 7400 + 53 * time_index], dtype=DTYPE
        )
        student_chi_square = tf.random.stateless_gamma(
            [count],
            [int(seed), 7500 + 59 * time_index],
            alpha=tf.constant(nu / 2.0, DTYPE),
            beta=tf.constant(0.5, DTYPE),
            dtype=DTYPE,
        )
        step = sampler(
            ukf["posterior_covariance"],
            local_means,
            local_cholesky,
            local_weights,
            ukf["posterior_mean"],
            defensive_cholesky,
            epsilon,
            ukf["innovation_log_likelihood"],
            log_parent_weights,
            ancestor_uniforms,
            component_uniforms,
            gaussian_normal,
            student_normal,
            student_chi_square,
        )
        if not bool(step["finite"].numpy()):
            raise ValueError(f"non-finite defensive APF output at time {time_index}")

        selected_local_means = tf.gather(local_means, step["ancestor_indices"])
        selected_local_cholesky = tf.gather(
            local_cholesky, step["ancestor_indices"]
        )
        selected_local_weights = tf.gather(local_weights, step["ancestor_indices"])
        selected_defensive_mean = tf.gather(
            ukf["posterior_mean"], step["ancestor_indices"]
        )
        selected_defensive_cholesky = tf.gather(
            defensive_cholesky, step["ancestor_indices"]
        )
        selected_epsilon = tf.gather(epsilon, step["ancestor_indices"])
        local_log_q = tf.math.log1p(-selected_epsilon) + complete_gaussian_mixture_log_density(
            step["samples"],
            selected_local_means,
            selected_local_cholesky,
            selected_local_weights,
        )
        independent_log_q = tf.reduce_logsumexp(
            tf.stack(
                [
                    local_log_q,
                    tf.math.log(selected_epsilon)
                    + student_log_density(
                        step["samples"],
                        selected_defensive_mean,
                        selected_defensive_cholesky,
                        nu,
                    ),
                ],
                axis=1,
            ),
            axis=1,
        )
        reversed_local_log_q = tf.math.log1p(-selected_epsilon) + complete_gaussian_mixture_log_density(
            step["samples"],
            tf.reverse(selected_local_means, axis=[1]),
            tf.reverse(selected_local_cholesky, axis=[1]),
            tf.reverse(selected_local_weights, axis=[1]),
        )
        permuted_log_q = tf.reduce_logsumexp(
            tf.stack(
                [
                    reversed_local_log_q,
                    tf.math.log(selected_epsilon)
                    + student_log_density(
                        step["samples"],
                        selected_defensive_mean,
                        selected_defensive_cholesky,
                        nu,
                    ),
                ],
                axis=1,
            ),
            axis=1,
        )
        local_mean_recomposed = tf.reduce_sum(
            local_weights[:, :, tf.newaxis] * local_means, axis=1
        )
        local_centered = local_means - local_mean_recomposed[:, tf.newaxis, :]
        local_covariance_recomposed = tf.reduce_sum(
            local_weights[:, :, tf.newaxis, tf.newaxis]
            * (
                tf.matmul(local_cholesky, local_cholesky, transpose_b=True)
                + tf.matmul(
                    local_centered[:, :, :, tf.newaxis],
                    local_centered[:, :, tf.newaxis, :],
                )
            ),
            axis=1,
        )
        local_moment_error = tf.reduce_max(
            tf.abs(local_covariance_recomposed - ukf["posterior_covariance"])
        )
        branch_states = tf.stack(states + [step["samples"]])
        branch_ancestors = tf.stack(ancestors + [step["ancestor_indices"]])
        branch_auxiliary = tf.stack(
            auxiliary_log_probabilities + [step["log_ancestor_probabilities"]]
        )
        branch_q = tf.stack(transition_log_q + [step["complete_log_q"]])
        branch = prepare_frozen_proposal_branch(
            observations=observed[: time_index + 1],
            states=branch_states,
            initial_log_proposal_density=initial_log_q,
            ancestors=branch_ancestors,
            auxiliary_log_probabilities=branch_auxiliary,
            transition_log_proposal_density=branch_q,
        )
        exact_prefix = prepare_frozen_proposal_apf_program(model, branch).evaluate(theta)
        if not bool(exact_prefix["finite"].numpy()):
            raise ValueError(f"exact C2 prefix is non-finite at time {time_index}")
        states.append(step["samples"])
        covariances = step["next_covariances"]
        ancestors.append(step["ancestor_indices"])
        auxiliary_log_probabilities.append(step["log_ancestor_probabilities"])
        transition_log_q.append(step["complete_log_q"])
        diagnostics.append(
            {
                "time_index": time_index,
                "local_component_count": local_component_count,
                "offset": tf.constant(offset, DTYPE),
                "nu": tf.constant(nu, DTYPE),
                "epsilon_min": tf.constant(epsilon_min, DTYPE),
                "epsilon_max": tf.constant(epsilon_max, DTYPE),
                "gate_center": tf.constant(center, DTYPE),
                "gate_temperature": tf.constant(temperature, DTYPE),
                "epsilon": epsilon,
                "epsilon_spread": tf.reduce_max(epsilon) - tf.reduce_min(epsilon),
                "epsilon_mean": tf.reduce_mean(epsilon),
                "innovation_quadratic": innovation_quadratic,
                "student_selected_fraction": tf.reduce_mean(
                    tf.cast(step["student_selected"], DTYPE)
                ),
                "lookahead_log_likelihood": ukf["innovation_log_likelihood"],
                "ancestor_log_probabilities": step["log_ancestor_probabilities"],
                "posterior_mean": ukf["posterior_mean"],
                "posterior_min_eigenvalue": ukf["posterior_min_eigenvalue"],
                "component_minimum_eigenvalue": component_minimum,
                "local_moment_recomposition_max_abs": local_moment_error,
                "label_permutation_max_abs": tf.reduce_max(
                    tf.abs(step["complete_log_q"] - permuted_log_q)
                ),
                "proposal_density_recomposition_max_abs": tf.reduce_max(
                    tf.abs(step["complete_log_q"] - independent_log_q)
                ),
                "proposal_finite": step["finite"],
                "exact_prefix_finite": exact_prefix["finite"],
            }
        )
        log_parent_weights = exact_prefix["final_log_weights"]

    manifest = {
        "route_id": DEFENSIVE_MIXTURE_ROUTE_ID,
        "route_classification": DEFENSIVE_MIXTURE_ROUTE_CLASSIFICATION,
        "local_component_count": local_component_count,
        "offset": float(offset),
        "offset_topology": "cholesky_column_sign_split_d1_or_d1_d2" if local_component_count > 1 else "single_posterior_gaussian",
        "local_component_weights": "equal_fixed",
        "defensive_component": "multivariate_student_centered_at_ukf_posterior",
        "nu": nu,
        "epsilon_min": epsilon_min,
        "epsilon_max": epsilon_max,
        "gate_center": center,
        "gate_temperature": temperature,
        "gate": "sigmoid_normalized_ukf_innovation_quadratic",
        "proposal_family": f"per_ancestor_gaussian_ukf_k{local_component_count}_student_defensive",
        "proposal_density": "complete_gaussian_student_mixture_density",
        "auxiliary_law": "exact_frozen_prefix_log_weights_plus_ukf_lookahead",
        "covariance_lifecycle": "posterior_covariance_carried_for_next_ukf_step",
        "observation_guide": "identity_latent_state_plus_log_chi_square_noise",
        "particle_count": count,
        "horizon": int(observed.shape[0]),
        "state_dimension": dimension,
        "theta_reference": theta,
        "seed": int(seed),
        "jit_compile_default": True,
        "jit_compile_requested": bool(jit_compile),
        "process_covariance": process_covariance,
        "observation_covariance": observation_covariance,
        "model": model.manifest_payload(),
        "branch_id": branch.branch_id,
        "runtime_branch_parameter_dependence": "none_after_compilation",
        "score_contract": "exact_frozen_branch_analytical_recursive_score",
        "random_inputs": "stateless_uniform_normal_and_gamma_frozen_outside_compiled_sampler",
        "complete_mixture_density": True,
        "full_support": True,
        "exact_pseudo_marginal_claimed": False,
        "status": "candidate_diagnostic_only",
        "nonclaims": (
            "no posterior correctness claim",
            "no low-variance claim",
            "no production readiness claim",
            "no adaptive-total-gradient claim",
        ),
        "stability": stability,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(_jsonable_manifest(manifest), sort_keys=True).encode("utf-8"))
    digest.update(branch.branch_id.encode("ascii"))
    return C2UKFAPFCompilation(
        branch=branch,
        compiler_id=digest.hexdigest(),
        manifest=manifest,
        proposal_diagnostics=tuple(diagnostics),
    )


__all__ = [
    "C2UKFAPFCompilation",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "compile_c2_per_ancestor_ukf_apf_k1",
    "MIXTURE_ROUTE_CLASSIFICATION",
    "MIXTURE_ROUTE_ID",
    "compile_c2_per_ancestor_ukf_apf_mixture",
    "DEFENSIVE_MIXTURE_ROUTE_CLASSIFICATION",
    "DEFENSIVE_MIXTURE_ROUTE_ID",
    "compile_c2_per_ancestor_ukf_apf_defensive_mixture",
]
