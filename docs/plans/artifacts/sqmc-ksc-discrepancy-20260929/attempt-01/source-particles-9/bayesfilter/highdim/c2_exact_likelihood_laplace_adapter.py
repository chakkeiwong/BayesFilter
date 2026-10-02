"""C2 binding for the generic exact-likelihood Laplace APF proposal.

The C2 model supplies analytical state score and curvature. The generic
Laplace kernel constructs a frozen Gaussian proposal for each retained
ancestor from the exact Gaussian transition conditional and the raw
observation likelihood. Importance values and parameter scores are still
computed only by ``FrozenProposalAPFProgram``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_tf import (
    gaussian_log_density,
    make_k1_apf_sampler_from_random_inputs,
)
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
)
from bayesfilter.highdim.exact_likelihood_laplace_apf_tf import (
    DTYPE,
    ROUTE_ID as GENERIC_LAPLACE_ROUTE_ID,
    FixedLaplaceConfig,
    make_fixed_laplace_bank_kernel,
    require_valid_laplace_result,
    schedule_manifest,
    single_start_offsets,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    FrozenProposalAPFProgram,
    PreparedFrozenProposalBranch,
    prepare_frozen_proposal_apf_program,
    prepare_frozen_proposal_branch,
)


ROUTE_ID = "c2_exact_likelihood_laplace_apf_k1_frozen_branch_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"

# Keep the K=1 Laplace and UKF comparator streams paired.  The proposal
# transformations differ, but using the same frozen base variates removes a
# needless time-zero and ancestor-sampling confound from their comparison.
INITIAL_NORMAL_KEY_OFFSET = 1001
CATEGORICAL_UNIFORM_KEY_BASE = 5100
CATEGORICAL_UNIFORM_KEY_STRIDE = 41
STANDARD_NORMAL_KEY_BASE = 5200
STANDARD_NORMAL_KEY_STRIDE = 43


@dataclass(frozen=True)
class C2ExactLikelihoodLaplaceCompilation:
    """One frozen exact-likelihood proposal branch and its provenance."""

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
            self,
            "proposal_diagnostics",
            tuple(dict(row) for row in self.proposal_diagnostics),
        )

    def bind_program(
        self, model: C2StochasticVolatilityFrozenAPFModel
    ) -> FrozenProposalAPFProgram:
        return prepare_frozen_proposal_apf_program(model, self.branch)


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest contains a non-finite float")
        return value
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def compile_c2_exact_likelihood_laplace_apf_k1(
    *,
    model: C2StochasticVolatilityFrozenAPFModel,
    observations: tf.Tensor,
    theta_reference: tf.Tensor,
    particle_count: int,
    seed: int,
    laplace_config: FixedLaplaceConfig,
    jit_compile: bool = True,
) -> C2ExactLikelihoodLaplaceCompilation:
    """Construct a frozen K=1 APF branch from exact likelihood geometry."""

    if not tf.executing_eagerly():
        raise RuntimeError("compile the C2 Laplace branch before tracing")
    if not isinstance(model, C2StochasticVolatilityFrozenAPFModel):
        raise TypeError("model must be a C2StochasticVolatilityFrozenAPFModel")
    if not isinstance(laplace_config, FixedLaplaceConfig):
        raise TypeError("laplace_config must be explicit")
    observed = tf.convert_to_tensor(observations, DTYPE)
    theta = tf.ensure_shape(tf.convert_to_tensor(theta_reference, DTYPE), [2])
    if (
        observed.shape.rank != 2
        or not observed.shape.is_fully_defined()
        or observed.shape[0] < 2
        or observed.shape[1] != model.observation_dim()
    ):
        raise ValueError("observations must have static shape [time, observation]")
    count = int(particle_count)
    if count < 2:
        raise ValueError("particle_count must be at least two")
    stability = model.stability_diagnostics(theta)
    if float(stability["spectral_radius"].numpy()) >= 1.0:
        raise ValueError("theta_reference is outside the stationary stability domain")

    dimension = model.state_dim()
    transition = model.transition_matrix(theta)
    transition_covariance = tf.eye(dimension, dtype=DTYPE) * tf.constant(
        float(model.sigma) ** 2, DTYPE
    )
    transition_covariances = tf.broadcast_to(
        transition_covariance[tf.newaxis, :, :],
        [count, dimension, dimension],
    )
    stationary_covariance, _ = model.stationary_covariance_and_derivative(theta)
    stationary_cholesky = tf.linalg.cholesky(stationary_covariance)

    def log_likelihood(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        return model.observation_log_density(theta, states, observation, 0)

    def state_score(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        return model.observation_log_density_state_score(theta, states, observation)

    def state_negative_hessian(
        states: tf.Tensor, observation: tf.Tensor
    ) -> tf.Tensor:
        return model.observation_log_density_state_negative_hessian(
            theta, states, observation
        )

    laplace_kernel = make_fixed_laplace_bank_kernel(
        batch_size=count,
        state_dim=dimension,
        observation_dim=model.observation_dim(),
        component_offsets=single_start_offsets(dimension),
        log_likelihood_fn=log_likelihood,
        state_score_fn=state_score,
        state_negative_hessian_fn=state_negative_hessian,
        config=laplace_config,
        jit_compile=bool(jit_compile),
    )
    sampler = make_k1_apf_sampler_from_random_inputs(
        batch_size=count,
        state_dim=dimension,
        jit_compile=bool(jit_compile),
    )

    initial_normal = tf.random.stateless_normal(
        [count, dimension], [int(seed), INITIAL_NORMAL_KEY_OFFSET], dtype=DTYPE
    )
    initial_states = tf.einsum(
        "ij,nj->ni", stationary_cholesky, initial_normal
    )
    initial_log_q = model.initial_log_density(theta, initial_states)
    states = [initial_states]
    ancestors: list[tf.Tensor] = []
    auxiliary_rows: list[tf.Tensor] = []
    proposal_rows: list[tf.Tensor] = []
    diagnostics: list[Mapping[str, object]] = []

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
        prior_means = tf.linalg.matmul(states[-1], transition, transpose_b=True)
        laplace = laplace_kernel(
            prior_means,
            transition_covariances,
            observed[time_index],
        )
        require_valid_laplace_result(laplace)
        posterior_means = laplace["component_means"][:, 0, :]
        posterior_covariances = laplace["component_covariances"][:, 0, :, :]
        posterior_cholesky = laplace["component_cholesky"][:, 0, :, :]
        categorical_uniforms = tf.random.stateless_uniform(
            [count],
            [
                int(seed),
                CATEGORICAL_UNIFORM_KEY_BASE
                + CATEGORICAL_UNIFORM_KEY_STRIDE * time_index,
            ],
            dtype=DTYPE,
        )
        standard_normal = tf.random.stateless_normal(
            [count, dimension],
            [
                int(seed),
                STANDARD_NORMAL_KEY_BASE
                + STANDARD_NORMAL_KEY_STRIDE * time_index,
            ],
            dtype=DTYPE,
        )
        step = sampler(
            posterior_means,
            posterior_covariances,
            posterior_cholesky,
            laplace["lookahead_log_likelihood"],
            log_parent_weights,
            categorical_uniforms,
            standard_normal,
        )
        if not bool(step["finite"].numpy()):
            raise ValueError(f"non-finite Laplace APF sample at time {time_index}")

        states.append(step["samples"])
        ancestors.append(step["ancestor_indices"])
        auxiliary_rows.append(step["log_ancestor_probabilities"])
        proposal_rows.append(step["complete_log_q"])
        branch = prepare_frozen_proposal_branch(
            observations=observed[: time_index + 1],
            states=tf.stack(states),
            initial_log_proposal_density=initial_log_q,
            ancestors=tf.stack(ancestors),
            auxiliary_log_probabilities=tf.stack(auxiliary_rows),
            transition_log_proposal_density=tf.stack(proposal_rows),
        )
        exact_prefix = prepare_frozen_proposal_apf_program(model, branch).evaluate(theta)
        if not bool(exact_prefix["finite"].numpy()):
            raise ValueError(f"exact C2 prefix is non-finite at time {time_index}")
        log_parent_weights = exact_prefix["final_log_weights"]
        diagnostics.append(
            {
                "time_index": time_index,
                "prior_mean": prior_means,
                "posterior_mean": posterior_means,
                "posterior_covariance": posterior_covariances,
                "lookahead_log_likelihood": laplace["lookahead_log_likelihood"],
                "ancestor_log_probabilities": step["log_ancestor_probabilities"],
                "relative_stationarity_residual": laplace[
                    "relative_stationarity_residual"
                ][:, 0],
                "minimum_precision_eigenvalue": laplace[
                    "minimum_precision_eigenvalue"
                ][:, 0],
                "minimum_covariance_eigenvalue": laplace[
                    "minimum_covariance_eigenvalue"
                ][:, 0],
                "iteration_step_max_abs": laplace["iteration_step_max_abs"][:, :, 0],
                "iteration_objective_before_step": laplace[
                    "iteration_objective_before_step"
                ][:, :, 0],
                "iteration_objective_after_step": laplace[
                    "iteration_objective_after_step"
                ][:, :, 0],
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
                "proposal_finite": step["finite"],
                "laplace_valid": laplace["valid"],
                "exact_prefix_finite": exact_prefix["finite"],
            }
        )

    manifest = {
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "generic_kernel_route_id": GENERIC_LAPLACE_ROUTE_ID,
        "particle_count": count,
        "horizon": int(observed.shape[0]),
        "state_dimension": dimension,
        "theta_reference": theta,
        "seed": int(seed),
        "jit_compile_default": True,
        "jit_compile_requested": bool(jit_compile),
        "proposal_family": "per_ancestor_exact_likelihood_laplace_k1",
        "proposal_density": "complete_conditional_gaussian_density",
        "transition_geometry": "exact_gaussian_transition_conditional",
        "observation_geometry": "analytical_exact_log_likelihood_state_score_and_curvature",
        "auxiliary_law": "exact_frozen_prefix_log_weights_plus_laplace_evidence",
        "laplace_schedule": schedule_manifest(laplace_config),
        "laplace_kernel_trace_count": int(
            laplace_kernel.experimental_get_tracing_count()
        ),
        "sampler_trace_count": int(sampler.experimental_get_tracing_count()),
        "model": model.manifest_payload(),
        "branch_id": branch.branch_id,
        "runtime_branch_parameter_dependence": "none_after_compilation",
        "score_contract": "exact_frozen_branch_analytical_recursive_score",
        "random_inputs": "stateless_uniform_and_normal_frozen_outside_compiled_sampler",
        "random_key_offsets": {
            "initial_normal": INITIAL_NORMAL_KEY_OFFSET,
            "categorical_uniforms": {
                "base": CATEGORICAL_UNIFORM_KEY_BASE,
                "stride": CATEGORICAL_UNIFORM_KEY_STRIDE,
            },
            "standard_normal": {
                "base": STANDARD_NORMAL_KEY_BASE,
                "stride": STANDARD_NORMAL_KEY_STRIDE,
            },
            "paired_with": "c2_per_ancestor_ukf_apf_k1",
        },
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
    digest.update(json.dumps(_jsonable(manifest), sort_keys=True).encode("utf-8"))
    digest.update(branch.branch_id.encode("ascii"))
    return C2ExactLikelihoodLaplaceCompilation(
        branch=branch,
        compiler_id=digest.hexdigest(),
        manifest=manifest,
        proposal_diagnostics=tuple(diagnostics),
    )


__all__ = [
    "C2ExactLikelihoodLaplaceCompilation",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "compile_c2_exact_likelihood_laplace_apf_k1",
]
