"""Fixed-branch importance evaluator for a Zhao-Cui-inspired TT proposal.

The Zhao-Cui squared-TT and KR operations are an offline proposal compiler for
this route.  The online scalar and score are a BayesFilter extension, not a
source-faithful implementation of the adaptive Zhao-Cui filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import tensorflow as tf

from bayesfilter.highdim.filtering import HighDimCoordinateMap
from bayesfilter.highdim.transport import FixedTTSIRTTransport


ROUTE_ID = "zhao_cui_frozen_proposal_apf_fixed_branch_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
TARGET_CLASS = "deterministic_fixed_branch_approximate_likelihood"
MEASURE_ID = "full_state_lebesgue_v1"
SCORE_BACKEND_ID = "analytical_parameter_score_no_autodiff_v1"
IDENTITY_ROLE = "reproducibility_fingerprint_not_admission"
TTSIRT_COMPILER_ID = "zhao_cui_fixed_ttsirt_prefix_conditioned_branch_compiler_v2"
TTSIRT_COMPILER_CLASSIFICATION = "extension_or_invention"


@dataclass(frozen=True)
class AlgebraicCoordinateMap:
    """Vectorized source-style algebraic map for unbounded physical states."""

    scales: tf.Tensor

    def __post_init__(self) -> None:
        scales = tf.reshape(tf.convert_to_tensor(self.scales, dtype=tf.float64), [-1])
        if scales.shape[0] is None or int(scales.shape[0]) < 1:
            raise ValueError("scales must contain at least one state dimension")
        if not bool(tf.reduce_all(tf.math.is_finite(scales)).numpy()) or bool(
            tf.reduce_any(scales <= 0.0).numpy()
        ):
            raise ValueError("algebraic map scales must be finite and positive")
        object.__setattr__(self, "scales", scales)

    @property
    def dimension(self) -> int:
        return int(self.scales.shape[0])

    def forward(self, reference_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        reference = tf.convert_to_tensor(reference_points, dtype=tf.float64)
        if reference.shape.rank != 2 or reference.shape[1] != self.dimension:
            raise ValueError("reference_points must have shape [sample, state]")
        clipped = tf.clip_by_value(reference, -1.0 + 1e-12, 1.0 - 1e-12)
        one_minus_square = 1.0 - tf.square(clipped)
        physical = clipped * tf.math.rsqrt(one_minus_square) * self.scales[None, :]
        log_det = tf.reduce_sum(
            -1.5 * tf.math.log(one_minus_square)
            + tf.math.log(self.scales)[None, :],
            axis=1,
        )
        return physical, log_det

    def inverse(self, physical_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        physical = tf.convert_to_tensor(physical_points, dtype=tf.float64)
        if physical.shape.rank != 2 or physical.shape[1] != self.dimension:
            raise ValueError("physical_points must have shape [sample, state]")
        scaled = physical / self.scales[None, :]
        reference = scaled * tf.math.rsqrt(1.0 + tf.square(scaled))
        log_det = tf.reduce_sum(
            -1.5 * tf.math.log1p(tf.square(scaled))
            - tf.math.log(self.scales)[None, :],
            axis=1,
        )
        return reference, log_det

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "AlgebraicCoordinateMap",
            "scales": self.scales,
            "source_formula": "z=(x/scale)/sqrt(1+(x/scale)^2)",
            "source_anchor": "bayesfilter/highdim/bases.py:104-142",
            "measure_role": "reference_to_physical_jacobian_explicit",
        }


class FrozenProposalAPFModel(Protocol):
    """Model contract required by the fixed-branch analytical evaluator."""

    def parameter_dim(self) -> int: ...

    def state_dim(self) -> int: ...

    def observation_dim(self) -> int: ...

    def frozen_apf_measure_id(self) -> str: ...

    def frozen_apf_score_backend_id(self) -> str: ...

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor: ...

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_previous: tf.Tensor,
        x_current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor: ...

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor: ...

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, x0: tf.Tensor
    ) -> tf.Tensor: ...

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        x_previous: tf.Tensor,
        x_current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor: ...

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor: ...

    def manifest_payload(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class PreparedFrozenProposalBranch:
    """Parameter-independent particles, genealogy, and proposal densities."""

    observations: tf.Tensor
    states: tf.Tensor
    initial_log_proposal_density: tf.Tensor
    ancestors: tf.Tensor
    auxiliary_log_probabilities: tf.Tensor
    transition_log_proposal_density: tf.Tensor
    branch_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not tf.executing_eagerly():
            raise RuntimeError("prepare the frozen proposal branch before tracing")

        states = tf.convert_to_tensor(self.states)
        if states.dtype not in (tf.float32, tf.float64):
            raise TypeError("states must use float32 or float64")
        observations = tf.convert_to_tensor(self.observations, dtype=states.dtype)
        initial_log_q = tf.convert_to_tensor(
            self.initial_log_proposal_density, dtype=states.dtype
        )
        ancestors = tf.convert_to_tensor(self.ancestors, dtype=tf.int32)
        auxiliary_log_probabilities = tf.convert_to_tensor(
            self.auxiliary_log_probabilities, dtype=states.dtype
        )
        transition_log_q = tf.convert_to_tensor(
            self.transition_log_proposal_density, dtype=states.dtype
        )

        if states.shape.rank != 3 or not states.shape.is_fully_defined():
            raise ValueError("states must have static shape [time, particle, state]")
        time_steps, particle_count, state_dimension = states.shape.as_list()
        if time_steps < 1 or particle_count < 2 or state_dimension < 1:
            raise ValueError("branch requires time >= 1, particles >= 2, and state >= 1")
        if (
            observations.shape.rank != 2
            or not observations.shape.is_fully_defined()
            or observations.shape[0] != time_steps
            or observations.shape[1] < 1
        ):
            raise ValueError("observations must have static shape [time, observation]")
        if initial_log_q.shape != (particle_count,):
            raise ValueError("initial_log_proposal_density must have shape [particle]")
        transition_shape = (time_steps - 1, particle_count)
        if ancestors.shape != transition_shape:
            raise ValueError("ancestors must have shape [time - 1, particle]")
        if auxiliary_log_probabilities.shape != transition_shape:
            raise ValueError(
                "auxiliary_log_probabilities must have shape [time - 1, particle]"
            )
        if transition_log_q.shape != transition_shape:
            raise ValueError(
                "transition_log_proposal_density must have shape [time - 1, particle]"
            )

        for name, value in (
            ("observations", observations),
            ("states", states),
            ("initial_log_proposal_density", initial_log_q),
            ("auxiliary_log_probabilities", auxiliary_log_probabilities),
            ("transition_log_proposal_density", transition_log_q),
        ):
            _require_all_finite(name, value)
        if time_steps > 1:
            ancestor_min = int(tf.reduce_min(ancestors).numpy())
            ancestor_max = int(tf.reduce_max(ancestors).numpy())
            if ancestor_min < 0 or ancestor_max >= particle_count:
                raise ValueError("ancestor index is outside the previous particle set")
            normalization_error = tf.reduce_max(
                tf.abs(tf.reduce_logsumexp(auxiliary_log_probabilities, axis=1))
            )
            tolerance = 5e-5 if states.dtype == tf.float32 else 1e-10
            if float(normalization_error.numpy()) > tolerance:
                raise ValueError("each auxiliary categorical law must be normalized")

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "initial_log_proposal_density", initial_log_q)
        object.__setattr__(self, "ancestors", ancestors)
        object.__setattr__(
            self, "auxiliary_log_probabilities", auxiliary_log_probabilities
        )
        object.__setattr__(self, "transition_log_proposal_density", transition_log_q)
        object.__setattr__(self, "branch_id", _branch_fingerprint(self))

    @property
    def dtype(self) -> tf.dtypes.DType:
        return self.states.dtype

    @property
    def time_steps(self) -> int:
        return int(self.states.shape[0])

    @property
    def particle_count(self) -> int:
        return int(self.states.shape[1])

    @property
    def state_dimension(self) -> int:
        return int(self.states.shape[2])

    @property
    def observation_dimension(self) -> int:
        return int(self.observations.shape[1])

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "target_class": TARGET_CLASS,
            "measure_id": MEASURE_ID,
            "identity_role": IDENTITY_ROLE,
            "branch_id": self.branch_id,
            "proposal_parameter_dependence": "none",
            "fixed_genealogy": True,
            "pseudo_marginal_exact_target_claimed": False,
            "dtype": self.dtype.name,
            "time_steps": self.time_steps,
            "particle_count": self.particle_count,
            "state_dimension": self.state_dimension,
            "observation_dimension": self.observation_dimension,
        }


@dataclass(frozen=True)
class FrozenTTSIRTProposalCompilation:
    """Offline TTSIRT mechanics compiled into a parameter-independent branch."""

    branch: PreparedFrozenProposalBranch
    compiler_id: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.branch, PreparedFrozenProposalBranch):
            raise TypeError("branch must be a PreparedFrozenProposalBranch")
        if len(str(self.compiler_id)) != 64:
            raise ValueError("compiler_id must be a SHA-256 digest")
        object.__setattr__(self, "manifest", dict(self.manifest))


def compile_fixed_ttsirt_proposal_branch(
    *,
    observations: tf.Tensor,
    initial_transport: FixedTTSIRTTransport,
    transition_transports: Sequence[FixedTTSIRTTransport],
    coordinate_map: HighDimCoordinateMap,
    initial_reference_points: tf.Tensor,
    ancestor_uniforms: tf.Tensor,
    auxiliary_log_probabilities: tf.Tensor,
    transition_reference_points: tf.Tensor,
) -> FrozenTTSIRTProposalCompilation:
    """Compile fixed TTSIRT maps using `(previous, current)` axis ordering.

    Zhao-Cui order the forward-map target as `(current, theta, previous)` and
    integrate from the left for their upper conditional map.  The local
    transport supports natural prefix conditioning, so this compiler stores
    `(previous, current)`.  That reordering is an extension, while freezing
    the source sampling operations is the fixed-HMC adaptation.
    """

    if not tf.executing_eagerly():
        raise RuntimeError("compile the TTSIRT proposal branch before tracing")
    if not isinstance(initial_transport, FixedTTSIRTTransport):
        raise TypeError("initial_transport must be a FixedTTSIRTTransport")
    transports = tuple(transition_transports)
    if any(not isinstance(item, FixedTTSIRTTransport) for item in transports):
        raise TypeError("every transition transport must be a FixedTTSIRTTransport")
    if not callable(getattr(coordinate_map, "forward", None)) or not callable(
        getattr(coordinate_map, "inverse", None)
    ):
        raise TypeError("coordinate_map must implement forward() and inverse()")
    if not callable(getattr(coordinate_map, "manifest_payload", None)):
        raise TypeError("coordinate_map must implement manifest_payload()")

    observations_tensor = tf.convert_to_tensor(observations, dtype=tf.float64)
    initial_reference = tf.convert_to_tensor(
        initial_reference_points, dtype=tf.float64
    )
    ancestor_u = tf.convert_to_tensor(ancestor_uniforms, dtype=tf.float64)
    auxiliary_log = tf.convert_to_tensor(
        auxiliary_log_probabilities, dtype=tf.float64
    )
    transition_reference = tf.convert_to_tensor(
        transition_reference_points, dtype=tf.float64
    )
    if observations_tensor.shape.rank != 2 or not observations_tensor.shape.is_fully_defined():
        raise ValueError("observations must have static shape [time, observation]")
    time_steps = int(observations_tensor.shape[0])
    if time_steps < 1 or len(transports) != time_steps - 1:
        raise ValueError("transition transport count must equal time - 1")
    if initial_reference.shape.rank != 2 or not initial_reference.shape.is_fully_defined():
        raise ValueError("initial_reference_points must have shape [state, particle]")
    state_dimension, particle_count = initial_reference.shape.as_list()
    if state_dimension < 1 or particle_count < 2:
        raise ValueError("TTSIRT branch requires state >= 1 and particles >= 2")
    if initial_transport.dimension != state_dimension:
        raise ValueError("initial transport dimension must equal the state dimension")
    transition_shape = (time_steps - 1, particle_count)
    if ancestor_u.shape != transition_shape or auxiliary_log.shape != transition_shape:
        raise ValueError("ancestor uniforms and auxiliary laws require [time - 1, particle]")
    if transition_reference.shape != (time_steps - 1, state_dimension, particle_count):
        raise ValueError(
            "transition_reference_points require [time - 1, state, particle]"
        )
    if any(transport.dimension != 2 * state_dimension for transport in transports):
        raise ValueError("transition transports require `(previous, current)` dimension 2d")
    _require_all_finite("observations", observations_tensor)
    _require_all_finite("initial_reference_points", initial_reference)
    _require_all_finite("ancestor_uniforms", ancestor_u)
    _require_all_finite("auxiliary_log_probabilities", auxiliary_log)
    _require_all_finite("transition_reference_points", transition_reference)
    if not bool(
        tf.reduce_all((initial_reference >= 0.0) & (initial_reference <= 1.0)).numpy()
        and tf.reduce_all(
            (transition_reference >= 0.0) & (transition_reference <= 1.0)
        ).numpy()
        and tf.reduce_all((ancestor_u >= 0.0) & (ancestor_u < 1.0)).numpy()
    ):
        raise ValueError("reference points require [0,1] and ancestor uniforms [0,1)")
    normalization_error = tf.reduce_max(
        tf.abs(tf.reduce_logsumexp(auxiliary_log, axis=1)
    )) if time_steps > 1 else tf.constant(0.0, tf.float64)
    if float(normalization_error.numpy()) > 1e-10:
        raise ValueError("each auxiliary categorical law must be normalized")

    all_transports = (initial_transport,) + transports
    for transport in all_transports:
        transport_manifest = transport.manifest_payload()
        if transport_manifest.get("source_contract_level") != "fixed_ttsirt":
            raise ValueError("transport must declare the fixed TTSIRT source contract")
        if transport_manifest.get("defensive_mass_positive") is not True:
            raise ValueError("TTSIRT proposal compilation requires positive defensive mass")
        if (
            transport_manifest.get("proposition2_marginal_backend")
            != "paired_core_mass_contraction_prefix_suffix"
        ):
            raise ValueError("TTSIRT conditional proposal requires paired-core marginalization")
        if transport_manifest.get("production_kr_closure") is not False:
            raise ValueError("local grid-CDF TTSIRT transport must remain nonproduction")

    initial_local = initial_transport.inverse_transport(initial_reference)
    initial_physical, initial_forward_log_det = coordinate_map.forward(
        tf.transpose(initial_local)
    )
    initial_log_q = (
        tf.math.log(initial_transport.eval_pdf(initial_local))
        - initial_forward_log_det
    )
    states = [initial_physical]
    ancestor_rows = []
    transition_log_q_rows = []

    for time_index, transport in enumerate(transports, start=1):
        cdf = tf.math.cumsum(tf.exp(auxiliary_log[time_index - 1]))
        cdf = tf.concat([cdf[:-1], tf.ones([1], tf.float64)], axis=0)
        ancestor = tf.searchsorted(
            cdf,
            ancestor_u[time_index - 1],
            side="right",
            out_type=tf.int32,
        )
        parent_physical = tf.gather(states[-1], ancestor)
        parent_local, _ = coordinate_map.inverse(parent_physical)
        current_local = transport.conditional_inverse_transport(
            tf.transpose(parent_local),
            transition_reference[time_index - 1],
        )
        current_physical, current_forward_log_det = coordinate_map.forward(
            tf.transpose(current_local)
        )
        conditional_local_log_q = transport.conditional_proposal_log_density(
            conditioning_points=tf.transpose(parent_local),
            generated_points=current_local,
        )
        transition_log_q_rows.append(
            conditional_local_log_q - current_forward_log_det
        )
        ancestor_rows.append(ancestor)
        states.append(current_physical)

    ancestors_tensor = (
        tf.stack(ancestor_rows)
        if ancestor_rows
        else tf.zeros([0, particle_count], tf.int32)
    )
    transition_log_q_tensor = (
        tf.stack(transition_log_q_rows)
        if transition_log_q_rows
        else tf.zeros([0, particle_count], tf.float64)
    )
    branch = prepare_frozen_proposal_branch(
        observations=observations_tensor,
        states=tf.stack(states),
        initial_log_proposal_density=initial_log_q,
        ancestors=ancestors_tensor,
        auxiliary_log_probabilities=auxiliary_log,
        transition_log_proposal_density=transition_log_q_tensor,
    )
    manifest = {
        "compiler_route_id": TTSIRT_COMPILER_ID,
        "classification": TTSIRT_COMPILER_CLASSIFICATION,
        "classification_correction": (
            "v2 classifies the reordered finite-grid compiler as an extension; "
            "v1 incorrectly classified the whole compiler as a fixed_hmc_adaptation"
        ),
        "axis_order": ("x_previous", "x_current"),
        "axis_order_relation_to_zhao_cui": "reordered_for_local_prefix_conditioning",
        "operation_classifications": {
            "squared_tt_defensive_density": {
                "classification": "source_faithful",
                "paper_anchor": (
                    ".localresources/papers/"
                    "zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539-573"
                ),
                "author_source_anchor": (
                    "third_party/audit/zhao_cui_tensor_ssm_p10/source/"
                    "deep-tensor.dev/src/SIRT.m:74-85"
                ),
                "scope": "operation only; the selected defensive mass is locally tuned",
            },
            "paired_core_prefix_conditional": {
                "classification": "source_faithful",
                "paper_anchor": (
                    ".localresources/papers/"
                    "zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:592-670"
                ),
                "author_source_anchor": (
                    "third_party/audit/zhao_cui_tensor_ssm_p10/source/"
                    "deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43-100"
                ),
                "scope": "generic prefix-conditional TT operation, not filtering axis order",
            },
            "frozen_randomness_and_settings": {
                "classification": "fixed_hmc_adaptation",
                "paper_anchor": (
                    ".localresources/papers/"
                    "zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:890-924"
                ),
                "author_source_anchor": (
                    "third_party/audit/zhao_cui_tensor_ssm_p10/source/"
                    "models/full_sol.m:21-43"
                ),
                "scope": "freezes the cited sampling and correction route",
            },
            "previous_current_prefix_axis_order": {
                "classification": "extension_or_invention",
                "reason": "the Zhao-Cui filtering route uses a different variable order",
            },
            "finite_grid_trapezoid_bisection_inverse": {
                "classification": "extension_or_invention",
                "reason": "not the paper's algebraic CDF or author CDFconstructor route",
            },
            "fixed_branch_apf_value_and_score": {
                "classification": "extension_or_invention",
                "reason": "BayesFilter finite-likelihood and analytical-score program",
            },
        },
        "branch_id": branch.branch_id,
        "coordinate_map": coordinate_map.manifest_payload(),
        "initial_transport": initial_transport.manifest_payload(),
        "transition_transports": tuple(
            transport.manifest_payload() for transport in transports
        ),
        "conditional_density_backend": "joint_eval_pdf_minus_proposition2_prefix_marginal",
        "production_kr_closure": False,
        "nonclaims": (
            "no source-faithful variable-order claim",
            "no source-faithful finite-grid inverse claim",
            "no production KR closure",
            "no randomized-estimator unbiasedness claim from finite grid inversion",
            "no HMC or default-readiness claim",
        ),
    }
    digest = hashlib.sha256()
    _update_hash(digest, manifest)
    _update_hash(digest, initial_reference)
    _update_hash(digest, ancestor_u)
    _update_hash(digest, auxiliary_log)
    _update_hash(digest, transition_reference)
    return FrozenTTSIRTProposalCompilation(
        branch=branch,
        compiler_id=digest.hexdigest(),
        manifest=manifest,
    )


def combine_fixed_ttsirt_block_compilations(
    compilations: Sequence[FrozenTTSIRTProposalCompilation],
    *,
    observation_mode: str = "shared",
) -> FrozenTTSIRTProposalCompilation:
    """Compose independent block proposals under one shared ancestor law."""

    blocks = tuple(compilations)
    if not blocks:
        raise ValueError("at least one TTSIRT block compilation is required")
    if any(not isinstance(item, FrozenTTSIRTProposalCompilation) for item in blocks):
        raise TypeError("every block must be a FrozenTTSIRTProposalCompilation")
    mode = str(observation_mode)
    if mode not in {"shared", "concatenate"}:
        raise ValueError("observation_mode must be 'shared' or 'concatenate'")
    first = blocks[0].branch
    for index, item in enumerate(blocks[1:], start=1):
        branch = item.branch
        if branch.time_steps != first.time_steps:
            raise ValueError(f"block {index} time dimension differs")
        if branch.particle_count != first.particle_count:
            raise ValueError(f"block {index} particle dimension differs")
        if mode == "shared":
            tf.debugging.assert_equal(
                branch.observations,
                first.observations,
                message=f"block {index} observations differ",
            )
        tf.debugging.assert_equal(
            branch.ancestors,
            first.ancestors,
            message=f"block {index} ancestor genealogy differs",
        )
        tf.debugging.assert_equal(
            branch.auxiliary_log_probabilities,
            first.auxiliary_log_probabilities,
            message=f"block {index} auxiliary law differs",
        )

    combined_observations = (
        first.observations
        if mode == "shared"
        else tf.concat([item.branch.observations for item in blocks], axis=1)
    )
    combined_branch = prepare_frozen_proposal_branch(
        observations=combined_observations,
        states=tf.concat([item.branch.states for item in blocks], axis=2),
        initial_log_proposal_density=tf.add_n(
            [item.branch.initial_log_proposal_density for item in blocks]
        ),
        ancestors=first.ancestors,
        auxiliary_log_probabilities=first.auxiliary_log_probabilities,
        transition_log_proposal_density=tf.add_n(
            [item.branch.transition_log_proposal_density for item in blocks]
        ),
    )
    manifest = {
        "compiler_route_id": "zhao_cui_blockwise_fixed_ttsirt_branch_compiler_v1",
        "classification": "extension_or_invention",
        "proposal_factorization": "product_of_block_conditionals_given_shared_ancestor",
        "shared_ancestor_genealogy": True,
        "observation_mode": mode,
        "block_count": len(blocks),
        "block_state_dimensions": tuple(
            item.branch.state_dimension for item in blocks
        ),
        "block_observation_dimensions": tuple(
            item.branch.observation_dimension for item in blocks
        ),
        "block_compiler_ids": tuple(item.compiler_id for item in blocks),
        "branch_id": combined_branch.branch_id,
        "production_kr_closure": False,
        "nonclaims": (
            "no source-faithful block-factorization claim",
            "no claim that the target factorizes across blocks",
            "cross-block mismatch is assessed by importance weights and ESS",
            "no production KR closure",
        ),
    }
    digest = hashlib.sha256()
    _update_hash(digest, manifest)
    return FrozenTTSIRTProposalCompilation(
        branch=combined_branch,
        compiler_id=digest.hexdigest(),
        manifest=manifest,
    )


def prepare_frozen_proposal_branch(
    *,
    observations: tf.Tensor,
    states: tf.Tensor,
    initial_log_proposal_density: tf.Tensor,
    ancestors: tf.Tensor,
    auxiliary_log_probabilities: tf.Tensor,
    transition_log_proposal_density: tf.Tensor,
) -> PreparedFrozenProposalBranch:
    """Issue a repository-computed identity for one realized proposal branch."""

    return PreparedFrozenProposalBranch(
        observations=observations,
        states=states,
        initial_log_proposal_density=initial_log_proposal_density,
        ancestors=ancestors,
        auxiliary_log_probabilities=auxiliary_log_probabilities,
        transition_log_proposal_density=transition_log_proposal_density,
    )


@dataclass(frozen=True)
class FrozenProposalAPFProgram:
    """A model and prepared branch bound to one finite value/score program."""

    model: FrozenProposalAPFModel
    branch: PreparedFrozenProposalBranch
    program_id: str = field(init=False)

    def __post_init__(self) -> None:
        required_methods = (
            "parameter_dim",
            "state_dim",
            "observation_dim",
            "frozen_apf_measure_id",
            "frozen_apf_score_backend_id",
            "initial_log_density",
            "transition_log_density",
            "observation_log_density",
            "initial_log_density_parameter_score",
            "transition_log_density_parameter_score",
            "observation_log_density_parameter_score",
            "manifest_payload",
        )
        for name in required_methods:
            if not callable(getattr(self.model, name, None)):
                raise TypeError(f"model must implement {name}()")
        if int(self.model.parameter_dim()) < 1:
            raise ValueError("fixed-branch score program requires at least one parameter")
        if int(self.model.state_dim()) != self.branch.state_dimension:
            raise ValueError("model and branch state dimensions differ")
        if int(self.model.observation_dim()) != self.branch.observation_dimension:
            raise ValueError("model and branch observation dimensions differ")
        if str(self.model.frozen_apf_measure_id()) != MEASURE_ID:
            raise ValueError(
                "this route supports only full-state nonsingular Lebesgue models; "
                "use an innovation-coordinate adapter for singular dynamics"
            )
        if str(self.model.frozen_apf_score_backend_id()) != SCORE_BACKEND_ID:
            raise ValueError("model must expose the reviewed analytical score backend")
        object.__setattr__(self, "program_id", _program_fingerprint(self.model, self.branch))

    def evaluate(self, theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
        """Evaluate eagerly with the same TensorFlow core used by XLA."""

        parameters = _theta_vector(theta, self.model.parameter_dim(), self.branch.dtype)
        return _evaluate_core(self.model, self.branch, parameters)

    def compiled(
        self, *, jit_compile: bool = True
    ) -> Callable[[tf.Tensor], Mapping[str, tf.Tensor]]:
        """Build the default XLA evaluator; non-JIT is an explicit debug exception."""

        parameter_dimension = int(self.model.parameter_dim())
        dtype = self.branch.dtype

        @tf.function(
            input_signature=[tf.TensorSpec([parameter_dimension], dtype)],
            jit_compile=bool(jit_compile),
            autograph=False,
        )
        def evaluate(theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
            return _evaluate_core(self.model, self.branch, theta)

        return evaluate

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            **self.branch.manifest_payload(),
            "program_id": self.program_id,
            "score_backend_id": SCORE_BACKEND_ID,
            "jit_compile_default": True,
            "finite_scalar": "sum_t(logsumexp(log_importance_weight_t)-log(N))",
            "score_semantics": "analytical_recursive_score_of_same_finite_scalar",
            "model": self.model.manifest_payload(),
        }


def prepare_frozen_proposal_apf_program(
    model: FrozenProposalAPFModel,
    branch: PreparedFrozenProposalBranch,
) -> FrozenProposalAPFProgram:
    """Bind the actual model implementation and branch into a program identity."""

    return FrozenProposalAPFProgram(model=model, branch=branch)


def _evaluate_core(
    model: FrozenProposalAPFModel,
    branch: PreparedFrozenProposalBranch,
    theta: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    dtype = branch.dtype
    particle_count = branch.particle_count
    parameter_dimension = int(model.parameter_dim())
    log_particle_count = tf.math.log(tf.cast(particle_count, dtype))

    initial_state = branch.states[0]
    initial_log_density = _vector(
        model.initial_log_density(theta, initial_state), particle_count, dtype
    )
    observation_log_density = _vector(
        model.observation_log_density(
            theta, initial_state, branch.observations[0], 0
        ),
        particle_count,
        dtype,
    )
    initial_score = _score_matrix(
        model.initial_log_density_parameter_score(theta, initial_state),
        particle_count,
        parameter_dimension,
        dtype,
    )
    observation_score = _score_matrix(
        model.observation_log_density_parameter_score(
            theta, initial_state, branch.observations[0], 0
        ),
        particle_count,
        parameter_dimension,
        dtype,
    )
    log_unnormalized = (
        initial_log_density
        + observation_log_density
        - branch.initial_log_proposal_density
    )
    local_marks = initial_score + observation_score
    log_sum = tf.reduce_logsumexp(log_unnormalized)
    increment = log_sum - log_particle_count
    log_weights = log_unnormalized - log_sum
    normalized_weights = tf.exp(log_weights)
    increment_score = tf.reduce_sum(normalized_weights[:, None] * local_marks, axis=0)
    derivative_log_weights = local_marks - increment_score[None, :]
    total_log_likelihood = increment
    total_score = increment_score
    minimum_ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights)))
    maximum_log_weight_spread = tf.reduce_max(log_unnormalized) - tf.reduce_min(
        log_unnormalized
    )
    finite = _all_finite(
        (
            log_unnormalized,
            local_marks,
            increment,
            increment_score,
            derivative_log_weights,
        )
    )
    increments = [increment]
    increment_scores = [increment_score]

    for time_index in range(1, branch.time_steps):
        ancestors = branch.ancestors[time_index - 1]
        previous_state = tf.gather(branch.states[time_index - 1], ancestors)
        current_state = branch.states[time_index]
        selected_previous_log_weights = tf.gather(log_weights, ancestors)
        selected_previous_marks = tf.gather(derivative_log_weights, ancestors)
        selected_auxiliary_log_probability = tf.gather(
            branch.auxiliary_log_probabilities[time_index - 1], ancestors
        )

        transition_log_density = _vector(
            model.transition_log_density(
                theta, previous_state, current_state, time_index
            ),
            particle_count,
            dtype,
        )
        observation_log_density = _vector(
            model.observation_log_density(
                theta,
                current_state,
                branch.observations[time_index],
                time_index,
            ),
            particle_count,
            dtype,
        )
        transition_score = _score_matrix(
            model.transition_log_density_parameter_score(
                theta, previous_state, current_state, time_index
            ),
            particle_count,
            parameter_dimension,
            dtype,
        )
        observation_score = _score_matrix(
            model.observation_log_density_parameter_score(
                theta,
                current_state,
                branch.observations[time_index],
                time_index,
            ),
            particle_count,
            parameter_dimension,
            dtype,
        )
        log_unnormalized = (
            selected_previous_log_weights
            + transition_log_density
            + observation_log_density
            - selected_auxiliary_log_probability
            - branch.transition_log_proposal_density[time_index - 1]
        )
        local_marks = selected_previous_marks + transition_score + observation_score
        log_sum = tf.reduce_logsumexp(log_unnormalized)
        increment = log_sum - log_particle_count
        log_weights = log_unnormalized - log_sum
        normalized_weights = tf.exp(log_weights)
        increment_score = tf.reduce_sum(
            normalized_weights[:, None] * local_marks, axis=0
        )
        derivative_log_weights = local_marks - increment_score[None, :]
        total_log_likelihood = total_log_likelihood + increment
        total_score = total_score + increment_score
        minimum_ess = tf.minimum(
            minimum_ess,
            tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights))),
        )
        maximum_log_weight_spread = tf.maximum(
            maximum_log_weight_spread,
            tf.reduce_max(log_unnormalized) - tf.reduce_min(log_unnormalized),
        )
        finite = finite & _all_finite(
            (
                log_unnormalized,
                local_marks,
                increment,
                increment_score,
                derivative_log_weights,
            )
        )
        increments.append(increment)
        increment_scores.append(increment_score)

    return {
        "log_likelihood": total_log_likelihood,
        "score": total_score,
        "log_increments": tf.stack(increments),
        "increment_scores": tf.stack(increment_scores),
        "final_log_weights": log_weights,
        "minimum_ess": minimum_ess,
        "maximum_log_weight_spread": maximum_log_weight_spread,
        "finite": finite,
        "particle_count": tf.constant(particle_count, tf.int32),
        "time_steps": tf.constant(branch.time_steps, tf.int32),
    }


def _theta_vector(
    theta: tf.Tensor, parameter_dimension: int, dtype: tf.dtypes.DType
) -> tf.Tensor:
    parameters = tf.convert_to_tensor(theta, dtype=dtype)
    if parameters.shape != (int(parameter_dimension),):
        raise ValueError("theta must have static shape [parameter]")
    return parameters


def _vector(
    value: tf.Tensor, length: int, dtype: tf.dtypes.DType
) -> tf.Tensor:
    return tf.ensure_shape(tf.convert_to_tensor(value, dtype=dtype), [length])


def _score_matrix(
    value: tf.Tensor,
    rows: int,
    columns: int,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    return tf.ensure_shape(
        tf.convert_to_tensor(value, dtype=dtype), [rows, columns]
    )


def _all_finite(values: Sequence[tf.Tensor]) -> tf.Tensor:
    flags = [tf.reduce_all(tf.math.is_finite(value)) for value in values]
    return tf.reduce_all(tf.stack(flags))


def _require_all_finite(name: str, value: tf.Tensor) -> None:
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise ValueError(f"{name} must contain only finite values")


def _branch_fingerprint(branch: PreparedFrozenProposalBranch) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, ROUTE_ID)
    _update_hash(digest, ROUTE_CLASSIFICATION)
    _update_hash(digest, TARGET_CLASS)
    _update_hash(digest, MEASURE_ID)
    for name in (
        "observations",
        "states",
        "initial_log_proposal_density",
        "ancestors",
        "auxiliary_log_probabilities",
        "transition_log_proposal_density",
    ):
        _update_hash(digest, name)
        _update_hash(digest, getattr(branch, name))
    return digest.hexdigest()


def _program_fingerprint(
    model: FrozenProposalAPFModel, branch: PreparedFrozenProposalBranch
) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, ROUTE_ID)
    _update_hash(digest, branch.branch_id)
    _update_hash(digest, type(model).__module__)
    _update_hash(digest, type(model).__qualname__)
    _update_hash(digest, model.manifest_payload())
    _update_hash(digest, _source_digest(Path(__file__)))
    model_source = Path(inspect.getfile(type(model)))
    _update_hash(digest, _source_digest(model_source))
    return digest.hexdigest()


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_hash(digest: object, value: object) -> None:
    if isinstance(value, tf.Tensor):
        digest.update(b"tensor\0")
        digest.update(value.dtype.name.encode("ascii"))
        digest.update(repr(value.shape.as_list()).encode("ascii"))
        digest.update(tf.io.serialize_tensor(value).numpy())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_hash(digest, str(key))
            _update_hash(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        for item in value:
            _update_hash(digest, item)
        return
    if isinstance(value, tf.dtypes.DType):
        _update_hash(digest, value.name)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
        return
    if isinstance(value, int):
        digest.update(b"int\0" + str(value).encode("ascii"))
        return
    if isinstance(value, float):
        digest.update(b"float\0" + value.hex().encode("ascii"))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"string\0" + str(len(encoded)).encode("ascii") + b"\0" + encoded)
        return
    raise TypeError(f"unsupported identity payload type: {type(value).__name__}")
