"""C2 adapter for the generic per-ancestor UKF/APF candidate.

The generic sigma-point and sampling primitives live in
``c2_mixture_ukf_apf_tf``.  This module only binds the C2 stochastic-volatility
model to those primitives and prepares a frozen branch for the repository's
exact target/analytical-score evaluator.  The transformed log-square
observation is a proposal coordinate; it never replaces the raw C2 likelihood
in the finite-program numerator.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_tf import DTYPE, BatchedUKFConfig
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
)
from bayesfilter.highdim.c2_transformed_observation_student_proposal_tf import (
    GUIDE_CONVENTION_ID,
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

    A retained fixed-signature TensorFlow owner compiles the complete numerical
    preparation by default, including the exact-prefix weight and covariance
    feedback. Configuration, entry validation and completed artifact records
    remain at the host boundary; no host numerical value feeds the time loop.
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

    from bayesfilter.highdim.c2_ukf_preparation_tf import make_k1_preparation

    dimension = model.state_dim()
    configuration = ukf_config or BatchedUKFConfig()
    key = ("k1", int(observed.shape[0]), count, configuration, bool(jit_compile))
    cache = getattr(model, "_c2_preparation_owners", None)
    if cache is None:
        cache = {}
        object.__setattr__(model, "_c2_preparation_owners", cache)
    if key not in cache:
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        cache[key] = make_k1_preparation(model, int(observed.shape[0]), count,
                                         configuration, bool(jit_compile))
    result = cache[key](observed, theta, tf.convert_to_tensor(int(seed), tf.int64))
    status = int(result["status"].numpy())
    failed_time = int(result["failed_time"].numpy())
    if status:
        errors = {
            1: "observations must contain only finite values",
            2: "states must contain only finite values",
            3: "initial_log_proposal_density must contain only finite values",
            4: "initial exact C2 prefix is non-finite",
            5: f"batched UKF result has {int(result['invalid_rows'].numpy())} invalid row(s)",
            6: f"non-finite K=1 APF output at time {failed_time}",
            7: "auxiliary_log_probabilities must contain only finite values",
            8: "each auxiliary categorical law must be normalized",
            9: f"exact C2 prefix is non-finite at time {failed_time}",
            10: "the transformed C2 guide requires nonzero observations",
            11: "transformed observations must be finite",
        }
        raise ValueError(errors[status])
    branch = prepare_frozen_proposal_branch(observations=observed,
        states=result["states"], initial_log_proposal_density=result["initial_log_proposal_density"],
        ancestors=result["ancestors"], auxiliary_log_probabilities=result["auxiliary_log_probabilities"],
        transition_log_proposal_density=result["transition_log_proposal_density"])
    # Completed output records only: no per-date host value feeds the owner.
    rows = tf.nest.map_structure(tf.unstack, result["diagnostics"])
    diagnostics = tuple({**{name: values[index] for name, values in rows.items()},
                         "time_index": index + 1} for index in range(int(observed.shape[0])-1))
    process_covariance = result["process_covariance"]
    observation_covariance = result["observation_covariance"]

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
    result, branch, diagnostics = _prepare_mixture_branch(
        model, observed, theta, count, seed, jit_compile,
        family="mixture", component_count=component_count, offset=offset)
    process_covariance = result["process_covariance"]
    observation_covariance = result["observation_covariance"]

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
    result, branch, diagnostics = _prepare_mixture_branch(
        model, observed, theta, count, seed, jit_compile,
        family="defensive", component_count=local_component_count, offset=offset,
        nu=nu, epsilon_min=epsilon_min, epsilon_max=epsilon_max, center=center,
        temperature=temperature, config=ukf_config)
    process_covariance = result["process_covariance"]
    observation_covariance = result["observation_covariance"]

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


def _prepare_mixture_branch(model, observed, theta, count, seed, jit_compile, *,
                            family, component_count, offset, nu=8.,
                            epsilon_min=.05, epsilon_max=.20, center=0.,
                            temperature=1., config=None):
    """Retain numerical owners; materialize only completed host records."""
    from bayesfilter.highdim.c2_ukf_preparation_tf import make_ukf_preparation

    configuration = config or BatchedUKFConfig()
    horizon = int(observed.shape[0])
    key = (family, horizon, count, configuration, bool(jit_compile),
           component_count, float(offset), nu, epsilon_min, epsilon_max,
           center, temperature)
    cache = getattr(model, "_c2_preparation_owners", None)
    if cache is None:
        cache = {}
        object.__setattr__(model, "_c2_preparation_owners", cache)
    if key not in cache:
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        cache[key] = make_ukf_preparation(model, horizon, count, configuration,
            bool(jit_compile), family=family, component_count=component_count,
            offset=float(offset), nu=nu, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max, center=center, temperature=temperature)
    result = cache[key](observed, theta, tf.convert_to_tensor(int(seed), tf.int64))
    status = int(result["status"].numpy())
    failed_time = int(result["failed_time"].numpy())
    if status:
        errors = {
            1: "observations must contain only finite values",
            2: "states must contain only finite values",
            3: "initial_log_proposal_density must contain only finite values",
            4: "initial exact C2 prefix is non-finite",
            5: f"batched UKF result has {int(result['invalid_rows'].numpy())} invalid row(s)",
            6: (f"non-finite K={component_count} APF output at time {failed_time}"
                if family == "mixture" else f"non-finite defensive APF output at time {failed_time}"),
            7: "auxiliary_log_probabilities must contain only finite values",
            8: "each auxiliary categorical law must be normalized",
            9: f"exact C2 prefix is non-finite at time {failed_time}",
            10: "the transformed C2 guide requires nonzero observations",
            11: "transformed observations must be finite",
            12: (f"K={component_count} split covariance is not SPD at time {failed_time}"
                 if family == "mixture" else f"local K={component_count} covariance is not SPD at time {failed_time}"),
        }
        raise ValueError(errors[status])
    branch = prepare_frozen_proposal_branch(observations=observed,
        states=result["states"], initial_log_proposal_density=result["initial_log_proposal_density"],
        ancestors=result["ancestors"], auxiliary_log_probabilities=result["auxiliary_log_probabilities"],
        transition_log_proposal_density=result["transition_log_proposal_density"])
    rows = tf.nest.map_structure(tf.unstack, result["diagnostics"])
    component_field = "component_count" if family == "mixture" else "local_component_count"
    diagnostics = tuple({**{name: values[index] for name, values in rows.items()},
                         "time_index": index + 1, component_field: component_count}
                        for index in range(horizon - 1))
    return result, branch, diagnostics
