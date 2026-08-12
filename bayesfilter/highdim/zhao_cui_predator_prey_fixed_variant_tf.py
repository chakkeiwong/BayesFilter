"""Source-order fixed-branch Zhao-Cui-derived predator-prey evaluator.

The squared-TT and conditional KR proposal operations are grounded in Zhao and
Cui (2024), Eq. 13, Proposition 2, and Algorithm 3. The frozen finite APF value
and its analytical score are a BayesFilter extension, not the adaptive author
filter and not a source-faithful assembly of that filter.
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


ROUTE_ID = "zhao_cui_predator_prey_source_order_fixed_branch_extension_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
TARGET_CLASS = "deterministic_source_order_fixed_branch_approximate_likelihood"
MEASURE_ID = "full_state_lebesgue_v1"
SCORE_BACKEND_ID = "analytical_parameter_score_no_autodiff_v1"
EVENT_ORDER = "x0_then_20_transition_then_observe_steps_y1_y20"
TARGET_ID = "zhao_cui_predator_prey_tf_seed81104_x0_then_y1_y20_v1"
TARGET_SEED = 81104
TARGET_HORIZON = 20
TARGET_STATE_SHA256 = (
    "63cc7d7e8e3a251f76ebb607b152b58b59cd8ceda4489057e60070b44ab1d2ec"
)
TARGET_OBSERVATION_SHA256 = (
    "fea0681d43a4bd502d1f5a90e04f58da435c6e891e72d9da4d54f4cf0584f00a"
)
COMPILER_ID = "zhao_cui_predator_prey_source_order_ttsirt_branch_compiler_v1"
COMPILER_CLASSIFICATION = "extension_or_invention"
IDENTITY_ROLE = "reproducibility_fingerprint_not_admission"


class SourceOrderFrozenAPFModel(Protocol):
    """Density and manual-score methods required by the online evaluator."""

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
        time_index: tf.Tensor,
    ) -> tf.Tensor: ...

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor: ...

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, x0: tf.Tensor
    ) -> tf.Tensor: ...

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        x_previous: tf.Tensor,
        x_current: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor: ...

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor: ...

    def manifest_payload(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class PreparedSourceOrderFrozenBranch:
    """Parameter-independent source-order particles and proposal corrections."""

    observations: tf.Tensor
    states: tf.Tensor
    initial_log_proposal_density: tf.Tensor
    ancestors: tf.Tensor
    auxiliary_log_probabilities: tf.Tensor
    transition_log_proposal_density: tf.Tensor
    target_id: str
    event_order: str
    target_seed: int
    target_state_sha256: str
    target_observation_sha256: str
    proposal_compiler_id: str = "manual_prepared_branch_no_compiler"
    branch_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not tf.executing_eagerly():
            raise RuntimeError("prepare the source-order branch before TensorFlow tracing")

        states = tf.convert_to_tensor(self.states)
        if states.dtype not in (tf.float32, tf.float64):
            raise TypeError("states must use float32 or float64")
        observations = tf.convert_to_tensor(self.observations, dtype=states.dtype)
        initial_log_q = tf.convert_to_tensor(
            self.initial_log_proposal_density, dtype=states.dtype
        )
        ancestors = tf.convert_to_tensor(self.ancestors, dtype=tf.int32)
        auxiliary_log = tf.convert_to_tensor(
            self.auxiliary_log_probabilities, dtype=states.dtype
        )
        transition_log_q = tf.convert_to_tensor(
            self.transition_log_proposal_density, dtype=states.dtype
        )

        if states.shape.rank != 3 or not states.shape.is_fully_defined():
            raise ValueError("states must have static shape [T + 1, particle, state]")
        state_rows, particle_count, state_dimension = states.shape.as_list()
        transition_count = state_rows - 1
        if transition_count < 1 or particle_count < 2 or state_dimension < 1:
            raise ValueError("branch requires T >= 1, particles >= 2, and state >= 1")
        if (
            observations.shape.rank != 2
            or not observations.shape.is_fully_defined()
            or observations.shape[0] != transition_count
            or observations.shape[1] < 1
        ):
            raise ValueError("observations must have static shape [T, observation]")
        if initial_log_q.shape != (particle_count,):
            raise ValueError("initial_log_proposal_density must have shape [particle]")
        transition_shape = (transition_count, particle_count)
        if ancestors.shape != transition_shape:
            raise ValueError("ancestors must have shape [T, particle]")
        if auxiliary_log.shape != transition_shape:
            raise ValueError("auxiliary_log_probabilities must have shape [T, particle]")
        if transition_log_q.shape != transition_shape:
            raise ValueError("transition_log_proposal_density must have shape [T, particle]")

        for name, value in (
            ("observations", observations),
            ("states", states),
            ("initial_log_proposal_density", initial_log_q),
            ("auxiliary_log_probabilities", auxiliary_log),
            ("transition_log_proposal_density", transition_log_q),
        ):
            _require_all_finite(name, value)
        ancestor_min = int(tf.reduce_min(ancestors).numpy())
        ancestor_max = int(tf.reduce_max(ancestors).numpy())
        if ancestor_min < 0 or ancestor_max >= particle_count:
            raise ValueError("ancestor index is outside the previous particle set")
        normalization_error = tf.reduce_max(
            tf.abs(tf.reduce_logsumexp(auxiliary_log, axis=1))
        )
        tolerance = 5e-5 if states.dtype == tf.float32 else 1e-10
        if float(normalization_error.numpy()) > tolerance:
            raise ValueError("each auxiliary categorical law must be normalized")

        for name in (
            "target_id",
            "event_order",
            "target_state_sha256",
            "target_observation_sha256",
            "proposal_compiler_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty")
        for name in ("target_state_sha256", "target_observation_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "initial_log_proposal_density", initial_log_q)
        object.__setattr__(self, "ancestors", ancestors)
        object.__setattr__(self, "auxiliary_log_probabilities", auxiliary_log)
        object.__setattr__(self, "transition_log_proposal_density", transition_log_q)
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "event_order", str(self.event_order))
        object.__setattr__(self, "target_seed", int(self.target_seed))
        object.__setattr__(self, "target_state_sha256", str(self.target_state_sha256))
        object.__setattr__(
            self, "target_observation_sha256", str(self.target_observation_sha256)
        )
        object.__setattr__(self, "proposal_compiler_id", str(self.proposal_compiler_id))
        object.__setattr__(self, "branch_id", _branch_fingerprint(self))

    @property
    def dtype(self) -> tf.dtypes.DType:
        return self.states.dtype

    @property
    def transition_count(self) -> int:
        return int(self.observations.shape[0])

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
            "target_id": self.target_id,
            "target_seed": self.target_seed,
            "event_order": self.event_order,
            "target_state_sha256": self.target_state_sha256,
            "target_observation_sha256": self.target_observation_sha256,
            "proposal_compiler_id": self.proposal_compiler_id,
            "proposal_parameter_dependence": "none",
            "fixed_genealogy": True,
            "pseudo_marginal_exact_target_claimed": False,
            "dtype": self.dtype.name,
            "transition_count": self.transition_count,
            "state_rows": self.transition_count + 1,
            "particle_count": self.particle_count,
            "state_dimension": self.state_dimension,
            "observation_dimension": self.observation_dimension,
        }


@dataclass(frozen=True)
class SourceOrderTTSIRTCompilation:
    """Offline source-order TTSIRT operations and their frozen branch."""

    branch: PreparedSourceOrderFrozenBranch
    compiler_id: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.branch, PreparedSourceOrderFrozenBranch):
            raise TypeError("branch must be a PreparedSourceOrderFrozenBranch")
        if len(str(self.compiler_id)) != 64:
            raise ValueError("compiler_id must be a SHA-256 digest")
        object.__setattr__(self, "manifest", dict(self.manifest))


@dataclass(frozen=True)
class SourceOrderFrozenAPFProgram:
    """A model and source-order branch bound to one finite value/score program."""

    model: SourceOrderFrozenAPFModel
    branch: PreparedSourceOrderFrozenBranch
    require_predator_prey_target: bool = False
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
            raise ValueError("source-order score requires at least one parameter")
        if int(self.model.state_dim()) != self.branch.state_dimension:
            raise ValueError("model and branch state dimensions differ")
        if int(self.model.observation_dim()) != self.branch.observation_dimension:
            raise ValueError("model and branch observation dimensions differ")
        if str(self.model.frozen_apf_measure_id()) != MEASURE_ID:
            raise ValueError("source-order APF requires a nonsingular full-state model")
        if str(self.model.frozen_apf_score_backend_id()) != SCORE_BACKEND_ID:
            raise ValueError("model must expose the reviewed analytical score backend")
        if self.require_predator_prey_target:
            _require_sealed_predator_prey_branch(self.branch)
            if (
                int(self.model.parameter_dim()) != 6
                or int(self.model.state_dim()) != 2
                or int(self.model.observation_dim()) != 2
            ):
                raise ValueError("sealed predator-prey program requires dimensions 6/2/2")
        object.__setattr__(
            self,
            "program_id",
            _program_fingerprint(
                self.model, self.branch, self.require_predator_prey_target
            ),
        )

    def evaluate(self, theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
        parameters = _theta_vector(
            theta, int(self.model.parameter_dim()), self.branch.dtype
        )
        return _evaluate_source_order_core(self.model, self.branch, parameters)

    def compiled(
        self, *, jit_compile: bool = True
    ) -> Callable[[tf.Tensor], Mapping[str, tf.Tensor]]:
        parameter_dimension = int(self.model.parameter_dim())
        dtype = self.branch.dtype

        @tf.function(
            input_signature=[tf.TensorSpec([parameter_dimension], dtype)],
            jit_compile=bool(jit_compile),
            autograph=False,
        )
        def evaluate(theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
            return _evaluate_source_order_core(self.model, self.branch, theta)

        return evaluate

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            **self.branch.manifest_payload(),
            "program_id": self.program_id,
            "score_backend_id": SCORE_BACKEND_ID,
            "jit_compile_default": True,
            "time_iteration": "tf.while_loop_graph_native",
            "finite_scalar": "c0_plus_sum_t1_to_T_logsumexp_importance_increment",
            "initial_correction": "log_p_theta_x0_minus_log_q0_no_observation",
            "score_semantics": "analytical_recursive_score_of_same_finite_scalar",
            "runtime_autodiff": False,
            "runtime_finite_difference": False,
            "retained_grid_route": False,
            "sealed_predator_prey_target_required": bool(
                self.require_predator_prey_target
            ),
            "model": self.model.manifest_payload(),
            "nonclaims": (
                "not the exact observed-data likelihood",
                "not a source-faithful assembly of the adaptive Zhao-Cui filter",
                "no pseudo-marginal unbiasedness claim",
                "no posterior or HMC readiness claim",
                "no statistical superiority claim",
            ),
        }


def prepare_source_order_frozen_branch(
    *,
    observations: tf.Tensor,
    states: tf.Tensor,
    initial_log_proposal_density: tf.Tensor,
    ancestors: tf.Tensor,
    auxiliary_log_probabilities: tf.Tensor,
    transition_log_proposal_density: tf.Tensor,
    target_id: str,
    event_order: str,
    target_seed: int,
    target_state_sha256: str,
    target_observation_sha256: str,
    proposal_compiler_id: str = "manual_prepared_branch_no_compiler",
) -> PreparedSourceOrderFrozenBranch:
    """Prepare one generic source-order branch before TensorFlow tracing."""

    return PreparedSourceOrderFrozenBranch(
        observations=observations,
        states=states,
        initial_log_proposal_density=initial_log_proposal_density,
        ancestors=ancestors,
        auxiliary_log_probabilities=auxiliary_log_probabilities,
        transition_log_proposal_density=transition_log_proposal_density,
        target_id=target_id,
        event_order=event_order,
        target_seed=target_seed,
        target_state_sha256=target_state_sha256,
        target_observation_sha256=target_observation_sha256,
        proposal_compiler_id=proposal_compiler_id,
    )


def prepare_predator_prey_source_order_branch(
    *,
    observations: tf.Tensor,
    states: tf.Tensor,
    initial_log_proposal_density: tf.Tensor,
    ancestors: tf.Tensor,
    auxiliary_log_probabilities: tf.Tensor,
    transition_log_proposal_density: tf.Tensor,
) -> PreparedSourceOrderFrozenBranch:
    """Prepare the sealed T20 predator-prey branch with repository-owned identity."""

    branch = prepare_source_order_frozen_branch(
        observations=observations,
        states=states,
        initial_log_proposal_density=initial_log_proposal_density,
        ancestors=ancestors,
        auxiliary_log_probabilities=auxiliary_log_probabilities,
        transition_log_proposal_density=transition_log_proposal_density,
        target_id=TARGET_ID,
        event_order=EVENT_ORDER,
        target_seed=TARGET_SEED,
        target_state_sha256=TARGET_STATE_SHA256,
        target_observation_sha256=TARGET_OBSERVATION_SHA256,
        proposal_compiler_id="manual_sealed_branch_not_admission_eligible",
    )
    _require_sealed_predator_prey_branch(branch)
    return branch


def prepare_source_order_frozen_apf_program(
    model: SourceOrderFrozenAPFModel,
    branch: PreparedSourceOrderFrozenBranch,
) -> SourceOrderFrozenAPFProgram:
    """Bind a generic source-order model and branch for mechanics tests."""

    return SourceOrderFrozenAPFProgram(model=model, branch=branch)


def prepare_predator_prey_fixed_variant_program(
    model: SourceOrderFrozenAPFModel,
    branch: PreparedSourceOrderFrozenBranch,
) -> SourceOrderFrozenAPFProgram:
    """Bind the sealed predator-prey target to its actual model implementation."""

    return SourceOrderFrozenAPFProgram(
        model=model, branch=branch, require_predator_prey_target=True
    )


def compile_source_order_ttsirt_proposal_branch(
    *,
    observations: tf.Tensor,
    initial_transport: FixedTTSIRTTransport,
    transition_transports: Sequence[FixedTTSIRTTransport],
    previous_coordinate_maps: Sequence[HighDimCoordinateMap],
    current_coordinate_maps: Sequence[HighDimCoordinateMap],
    initial_reference_points: tf.Tensor,
    ancestor_uniforms: tf.Tensor,
    auxiliary_log_probabilities: tf.Tensor,
    transition_reference_points: tf.Tensor,
    target_id: str,
    event_order: str,
    target_seed: int,
    target_state_sha256: str,
    target_observation_sha256: str,
    tuning_artifact_id: str = "not_applicable_generic_compiler",
    online_dtype: tf.dtypes.DType = tf.float32,
    inverse_microbatch_size: int | None = None,
) -> SourceOrderTTSIRTCompilation:
    """Compile `(x_previous, x_current)` TTSIRT conditionals for `y1:yT`.

    The local prefix ordering, finite-grid inverse, and assembled APF program
    are extensions. Squared-TT defensive density and paired-core conditionals
    remain individually source-grounded operations.
    """

    if not tf.executing_eagerly():
        raise RuntimeError("compile the TTSIRT branch before TensorFlow tracing")
    if online_dtype not in (tf.float32, tf.float64):
        raise TypeError("online_dtype must be float32 or float64")
    if not isinstance(initial_transport, FixedTTSIRTTransport):
        raise TypeError("initial_transport must be a FixedTTSIRTTransport")
    transports = tuple(transition_transports)
    previous_maps = tuple(previous_coordinate_maps)
    current_maps = tuple(current_coordinate_maps)
    if any(not isinstance(item, FixedTTSIRTTransport) for item in transports):
        raise TypeError("every transition transport must be a FixedTTSIRTTransport")

    observations_tensor = tf.convert_to_tensor(observations, tf.float64)
    initial_reference = tf.convert_to_tensor(initial_reference_points, tf.float64)
    ancestor_u = tf.convert_to_tensor(ancestor_uniforms, tf.float64)
    auxiliary_log = tf.convert_to_tensor(auxiliary_log_probabilities, tf.float64)
    transition_reference = tf.convert_to_tensor(
        transition_reference_points, tf.float64
    )
    if (
        observations_tensor.shape.rank != 2
        or not observations_tensor.shape.is_fully_defined()
    ):
        raise ValueError("observations must have static shape [T, observation]")
    transition_count = int(observations_tensor.shape[0])
    if transition_count < 1 or len(transports) != transition_count:
        raise ValueError("transition transport count must equal T")
    if len(previous_maps) != transition_count or len(current_maps) != transition_count:
        raise ValueError("previous/current coordinate-map counts must equal T")
    if initial_reference.shape.rank != 2 or not initial_reference.shape.is_fully_defined():
        raise ValueError("initial_reference_points must have shape [state, particle]")
    state_dimension, particle_count = initial_reference.shape.as_list()
    if state_dimension < 1 or particle_count < 2:
        raise ValueError("TTSIRT branch requires state >= 1 and particles >= 2")
    if initial_transport.dimension != state_dimension:
        raise ValueError("initial transport dimension must equal state dimension")
    microbatch_size = (
        particle_count
        if inverse_microbatch_size is None
        else int(inverse_microbatch_size)
    )
    if microbatch_size < 1 or microbatch_size > particle_count:
        raise ValueError("inverse_microbatch_size must be in [1, particle_count]")
    expected_shape = (transition_count, particle_count)
    if ancestor_u.shape != expected_shape or auxiliary_log.shape != expected_shape:
        raise ValueError("ancestor uniforms and auxiliary laws require [T, particle]")
    if transition_reference.shape != (
        transition_count,
        state_dimension,
        particle_count,
    ):
        raise ValueError("transition_reference_points require [T, state, particle]")
    if any(item.dimension != 2 * state_dimension for item in transports):
        raise ValueError("transition transports require `(previous,current)` dimension 2d")
    for coordinate_map in previous_maps + current_maps:
        if not callable(getattr(coordinate_map, "forward", None)) or not callable(
            getattr(coordinate_map, "inverse", None)
        ):
            raise TypeError("coordinate maps must implement forward() and inverse()")
        if not callable(getattr(coordinate_map, "manifest_payload", None)):
            raise TypeError("coordinate maps must implement manifest_payload()")
    for name, value in (
        ("observations", observations_tensor),
        ("initial_reference_points", initial_reference),
        ("ancestor_uniforms", ancestor_u),
        ("auxiliary_log_probabilities", auxiliary_log),
        ("transition_reference_points", transition_reference),
    ):
        _require_all_finite(name, value)
    if not bool(
        tf.reduce_all((initial_reference >= 0.0) & (initial_reference <= 1.0)).numpy()
        and tf.reduce_all(
            (transition_reference >= 0.0) & (transition_reference <= 1.0)
        ).numpy()
        and tf.reduce_all((ancestor_u >= 0.0) & (ancestor_u < 1.0)).numpy()
    ):
        raise ValueError("reference points require [0,1] and ancestor uniforms [0,1)")
    normalization_error = tf.reduce_max(
        tf.abs(tf.reduce_logsumexp(auxiliary_log, axis=1))
    )
    if float(normalization_error.numpy()) > 1e-10:
        raise ValueError("each auxiliary categorical law must be normalized")

    all_transports = (initial_transport,) + transports
    for transport in all_transports:
        transport_manifest = transport.manifest_payload()
        if transport_manifest.get("source_contract_level") != "fixed_ttsirt":
            raise ValueError("transport must declare the fixed TTSIRT source contract")
        if transport_manifest.get("defensive_mass_positive") is not True:
            raise ValueError("TTSIRT compilation requires positive defensive mass")
        if (
            transport_manifest.get("proposition2_marginal_backend")
            != "paired_core_mass_contraction_prefix_suffix"
        ):
            raise ValueError("conditional proposal requires paired-core marginalization")
        if transport_manifest.get("production_kr_closure") is not False:
            raise ValueError("finite-grid TTSIRT transport must remain nonproduction")

    # The initial map is the previous-state map used by the first joint target.
    initial_map = previous_maps[0]
    initial_local = _inverse_transport_microbatched(
        initial_transport,
        initial_reference,
        microbatch_size=microbatch_size,
    )
    initial_physical, initial_forward_log_det = initial_map.forward(
        tf.transpose(initial_local)
    )
    initial_log_q = (
        tf.math.log(initial_transport.eval_pdf(initial_local))
        - initial_forward_log_det
    )
    states = [initial_physical]
    ancestor_rows = []
    transition_log_q_rows = []

    for time_index, transport in enumerate(transports):
        cdf = tf.math.cumsum(tf.exp(auxiliary_log[time_index]))
        cdf = tf.concat([cdf[:-1], tf.ones([1], tf.float64)], axis=0)
        ancestor = tf.searchsorted(
            cdf,
            ancestor_u[time_index],
            side="right",
            out_type=tf.int32,
        )
        parent_physical = tf.gather(states[-1], ancestor)
        parent_local, _ = previous_maps[time_index].inverse(parent_physical)
        current_local = _conditional_inverse_transport_microbatched(
            transport,
            tf.transpose(parent_local),
            transition_reference[time_index],
            microbatch_size=microbatch_size,
        )
        current_physical, current_forward_log_det = current_maps[time_index].forward(
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

    compiler_identity_payload = {
        "compiler_route_id": COMPILER_ID,
        "classification": COMPILER_CLASSIFICATION,
        "target_id": target_id,
        "event_order": event_order,
        "target_seed": int(target_seed),
        "target_state_sha256": target_state_sha256,
        "target_observation_sha256": target_observation_sha256,
        "tuning_artifact_id": str(tuning_artifact_id),
        "online_dtype": online_dtype.name,
        "inverse_microbatch_size": microbatch_size,
        "initial_transport": {
            "transport": initial_transport.manifest_payload(),
            "density": initial_transport.density.manifest_payload(),
        },
        "transition_transports": tuple(
            {
                "transport": item.manifest_payload(),
                "density": item.density.manifest_payload(),
            }
            for item in transports
        ),
        "previous_coordinate_maps": tuple(
            item.manifest_payload() for item in previous_maps
        ),
        "current_coordinate_maps": tuple(
            item.manifest_payload() for item in current_maps
        ),
        "source_dependency_sha256": {
            "compiler": _source_digest(Path(__file__)),
            "transport": _source_digest(Path(inspect.getfile(FixedTTSIRTTransport))),
            "squared_tt": _source_digest(Path(inspect.getfile(type(initial_transport.density)))),
        },
    }
    compiler_digest = hashlib.sha256()
    _update_hash(compiler_digest, compiler_identity_payload)
    _update_hash(compiler_digest, observations_tensor)
    _update_hash(compiler_digest, initial_reference)
    _update_hash(compiler_digest, ancestor_u)
    _update_hash(compiler_digest, auxiliary_log)
    _update_hash(compiler_digest, transition_reference)
    compiler_id = compiler_digest.hexdigest()

    branch = prepare_source_order_frozen_branch(
        observations=tf.cast(observations_tensor, online_dtype),
        states=tf.cast(tf.stack(states), online_dtype),
        initial_log_proposal_density=tf.cast(initial_log_q, online_dtype),
        ancestors=tf.stack(ancestor_rows),
        auxiliary_log_probabilities=tf.cast(auxiliary_log, online_dtype),
        transition_log_proposal_density=tf.cast(
            tf.stack(transition_log_q_rows), online_dtype
        ),
        target_id=target_id,
        event_order=event_order,
        target_seed=target_seed,
        target_state_sha256=target_state_sha256,
        target_observation_sha256=target_observation_sha256,
        proposal_compiler_id=compiler_id,
    )
    manifest = {
        "compiler_route_id": COMPILER_ID,
        "classification": COMPILER_CLASSIFICATION,
        "axis_order": ("x_previous", "x_current"),
        "axis_order_relation_to_zhao_cui": "reordered_for_local_prefix_conditioning",
        "source_order": EVENT_ORDER,
        "branch_id": branch.branch_id,
        "online_dtype": online_dtype.name,
        "inverse_microbatch_size": microbatch_size,
        "inverse_microbatching": "deterministic_contiguous_particle_slices",
        "operation_classifications": {
            "squared_tt_defensive_density": {
                "classification": "source_faithful",
                "paper_anchor": (
                    ".localresources/papers/"
                    "zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539-573"
                ),
                "author_source_anchor": (
                    "third_party/audit/zhao_cui_tensor_ssm_p10/source/"
                    "deep-tensor.dev/src/SIRT.m:51-85"
                ),
                "scope": "operation only",
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
                "scope": "generic prefix conditional only",
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
            },
            "previous_current_prefix_axis_order": {
                "classification": "extension_or_invention",
            },
            "time_specific_coordinate_maps": {
                "classification": "extension_or_invention",
            },
            "finite_grid_trapezoid_bisection_inverse": {
                "classification": "extension_or_invention",
            },
            "source_order_fixed_branch_value_and_score": {
                "classification": "extension_or_invention",
            },
        },
        "initial_transport": compiler_identity_payload["initial_transport"],
        "transition_transports": tuple(
            compiler_identity_payload["transition_transports"]
        ),
        "previous_coordinate_maps": tuple(
            item.manifest_payload() for item in previous_maps
        ),
        "current_coordinate_maps": tuple(
            item.manifest_payload() for item in current_maps
        ),
        "conditional_density_backend": (
            "joint_eval_pdf_minus_proposition2_prefix_marginal"
        ),
        "production_kr_closure": False,
        "source_dependency_sha256": compiler_identity_payload[
            "source_dependency_sha256"
        ],
        "nonclaims": (
            "no source-faithful assembled-route claim",
            "no source-faithful variable-order claim",
            "no source-faithful finite-grid inverse claim",
            "no production KR closure",
            "no HMC or default-readiness claim",
        ),
    }
    return SourceOrderTTSIRTCompilation(
        branch=branch, compiler_id=compiler_id, manifest=manifest
    )


def compile_predator_prey_source_order_ttsirt_proposal_branch(
    *,
    observations: tf.Tensor,
    initial_transport: FixedTTSIRTTransport,
    transition_transports: Sequence[FixedTTSIRTTransport],
    previous_coordinate_maps: Sequence[HighDimCoordinateMap],
    current_coordinate_maps: Sequence[HighDimCoordinateMap],
    initial_reference_points: tf.Tensor,
    ancestor_uniforms: tf.Tensor,
    auxiliary_log_probabilities: tf.Tensor,
    transition_reference_points: tf.Tensor,
    tuning_artifact_id: str,
    online_dtype: tf.dtypes.DType = tf.float32,
    inverse_microbatch_size: int | None = None,
) -> SourceOrderTTSIRTCompilation:
    """Compile and seal the T20 predator-prey proposal branch."""

    tuning_id = str(tuning_artifact_id)
    if len(tuning_id) != 64 or any(
        character not in "0123456789abcdef" for character in tuning_id
    ):
        raise ValueError("predator-prey compiler requires a tuning artifact SHA-256")
    compilation = compile_source_order_ttsirt_proposal_branch(
        observations=observations,
        initial_transport=initial_transport,
        transition_transports=transition_transports,
        previous_coordinate_maps=previous_coordinate_maps,
        current_coordinate_maps=current_coordinate_maps,
        initial_reference_points=initial_reference_points,
        ancestor_uniforms=ancestor_uniforms,
        auxiliary_log_probabilities=auxiliary_log_probabilities,
        transition_reference_points=transition_reference_points,
        target_id=TARGET_ID,
        event_order=EVENT_ORDER,
        target_seed=TARGET_SEED,
        target_state_sha256=TARGET_STATE_SHA256,
        target_observation_sha256=TARGET_OBSERVATION_SHA256,
        tuning_artifact_id=tuning_id,
        online_dtype=online_dtype,
        inverse_microbatch_size=inverse_microbatch_size,
    )
    _require_sealed_predator_prey_branch(compilation.branch)
    return compilation


def _inverse_transport_microbatched(
    transport: FixedTTSIRTTransport,
    reference_points: tf.Tensor,
    *,
    microbatch_size: int,
) -> tf.Tensor:
    sample_count = int(reference_points.shape[1])
    batches = []
    for start in range(0, sample_count, int(microbatch_size)):
        stop = min(start + int(microbatch_size), sample_count)
        batches.append(transport.inverse_transport(reference_points[:, start:stop]))
    return tf.concat(batches, axis=1)


def _conditional_inverse_transport_microbatched(
    transport: FixedTTSIRTTransport,
    conditioning_points: tf.Tensor,
    reference_points: tf.Tensor,
    *,
    microbatch_size: int,
) -> tf.Tensor:
    sample_count = int(reference_points.shape[1])
    condition_count = int(conditioning_points.shape[1])
    if condition_count not in (1, sample_count):
        raise ValueError("conditioning point count must be one or particle_count")
    batches = []
    for start in range(0, sample_count, int(microbatch_size)):
        stop = min(start + int(microbatch_size), sample_count)
        condition = (
            conditioning_points
            if condition_count == 1
            else conditioning_points[:, start:stop]
        )
        batches.append(
            transport.conditional_inverse_transport(
                condition,
                reference_points[:, start:stop],
            )
        )
    return tf.concat(batches, axis=1)


def _evaluate_source_order_core(
    model: SourceOrderFrozenAPFModel,
    branch: PreparedSourceOrderFrozenBranch,
    theta: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    dtype = branch.dtype
    particle_count = branch.particle_count
    parameter_dimension = int(model.parameter_dim())
    transition_count = branch.transition_count
    log_particle_count = tf.math.log(tf.cast(particle_count, dtype))

    initial_state = branch.states[0]
    initial_log_density = _vector(
        model.initial_log_density(theta, initial_state), particle_count, dtype
    )
    initial_score = _score_matrix(
        model.initial_log_density_parameter_score(theta, initial_state),
        particle_count,
        parameter_dimension,
        dtype,
    )
    log_unnormalized = initial_log_density - branch.initial_log_proposal_density
    log_sum = tf.reduce_logsumexp(log_unnormalized)
    increment = log_sum - log_particle_count
    log_weights = log_unnormalized - log_sum
    normalized_weights = tf.exp(log_weights)
    increment_score = tf.reduce_sum(
        normalized_weights[:, tf.newaxis] * initial_score, axis=0
    )
    derivative_log_weights = initial_score - increment_score[tf.newaxis, :]
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights)))
    spread = tf.reduce_max(log_unnormalized) - tf.reduce_min(log_unnormalized)
    finite = _all_finite(
        (
            log_unnormalized,
            initial_score,
            increment,
            increment_score,
            derivative_log_weights,
            ess,
            spread,
        )
    )

    increment_array = tf.TensorArray(
        dtype=dtype,
        size=transition_count + 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([]),
    ).write(0, increment)
    score_array = tf.TensorArray(
        dtype=dtype,
        size=transition_count + 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([parameter_dimension]),
    ).write(0, increment_score)
    ess_array = tf.TensorArray(
        dtype=dtype,
        size=transition_count + 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([]),
    ).write(0, ess)
    spread_array = tf.TensorArray(
        dtype=dtype,
        size=transition_count + 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([]),
    ).write(0, spread)
    maximum_weight_array = tf.TensorArray(
        dtype=dtype,
        size=transition_count + 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([]),
    ).write(0, tf.reduce_max(normalized_weights))

    def condition(
        time_index: tf.Tensor,
        *_loop_values: object,
    ) -> tf.Tensor:
        return time_index <= transition_count

    def body(
        time_index: tf.Tensor,
        previous_log_weights: tf.Tensor,
        previous_derivative_log_weights: tf.Tensor,
        total_value: tf.Tensor,
        total_score: tf.Tensor,
        minimum_ess: tf.Tensor,
        maximum_spread: tf.Tensor,
        all_finite: tf.Tensor,
        increments: tf.TensorArray,
        increment_scores: tf.TensorArray,
        ess_values: tf.TensorArray,
        spread_values: tf.TensorArray,
        maximum_weight_values: tf.TensorArray,
    ) -> tuple[object, ...]:
        row = time_index - 1
        ancestor = tf.gather(branch.ancestors, row)
        previous_state = tf.gather(
            tf.gather(branch.states, time_index - 1), ancestor
        )
        current_state = tf.gather(branch.states, time_index)
        selected_previous_log_weights = tf.gather(previous_log_weights, ancestor)
        selected_previous_marks = tf.gather(
            previous_derivative_log_weights, ancestor
        )
        selected_auxiliary_log_probability = tf.gather(
            tf.gather(branch.auxiliary_log_probabilities, row), ancestor
        )
        observation = tf.gather(branch.observations, row)

        transition_log_density = _vector(
            model.transition_log_density(
                theta, previous_state, current_state, time_index
            ),
            particle_count,
            dtype,
        )
        observation_log_density = _vector(
            model.observation_log_density(
                theta, current_state, observation, time_index
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
                theta, current_state, observation, time_index
            ),
            particle_count,
            parameter_dimension,
            dtype,
        )
        current_log_unnormalized = (
            selected_previous_log_weights
            + transition_log_density
            + observation_log_density
            - selected_auxiliary_log_probability
            - tf.gather(branch.transition_log_proposal_density, row)
        )
        local_marks = selected_previous_marks + transition_score + observation_score
        current_log_sum = tf.reduce_logsumexp(current_log_unnormalized)
        current_increment = current_log_sum - log_particle_count
        current_log_weights = current_log_unnormalized - current_log_sum
        current_normalized_weights = tf.exp(current_log_weights)
        current_increment_score = tf.reduce_sum(
            current_normalized_weights[:, tf.newaxis] * local_marks, axis=0
        )
        current_derivative_log_weights = (
            local_marks - current_increment_score[tf.newaxis, :]
        )
        current_ess = tf.math.reciprocal(
            tf.reduce_sum(tf.square(current_normalized_weights))
        )
        current_spread = tf.reduce_max(current_log_unnormalized) - tf.reduce_min(
            current_log_unnormalized
        )
        current_finite = _all_finite(
            (
                current_log_unnormalized,
                local_marks,
                current_increment,
                current_increment_score,
                current_derivative_log_weights,
                current_ess,
                current_spread,
            )
        )
        return (
            time_index + 1,
            current_log_weights,
            current_derivative_log_weights,
            total_value + current_increment,
            total_score + current_increment_score,
            tf.minimum(minimum_ess, current_ess),
            tf.maximum(maximum_spread, current_spread),
            all_finite & current_finite,
            increments.write(time_index, current_increment),
            increment_scores.write(time_index, current_increment_score),
            ess_values.write(time_index, current_ess),
            spread_values.write(time_index, current_spread),
            maximum_weight_values.write(
                time_index, tf.reduce_max(current_normalized_weights)
            ),
        )

    (
        _,
        final_log_weights,
        _,
        total_value,
        total_score,
        minimum_ess,
        maximum_spread,
        finite,
        increment_array,
        score_array,
        ess_array,
        spread_array,
        maximum_weight_array,
    ) = tf.while_loop(
        condition,
        body,
        (
            tf.constant(1, tf.int32),
            log_weights,
            derivative_log_weights,
            increment,
            increment_score,
            ess,
            spread,
            finite,
            increment_array,
            score_array,
            ess_array,
            spread_array,
            maximum_weight_array,
        ),
        parallel_iterations=1,
    )

    return {
        "log_likelihood": total_value,
        "score": total_score,
        "log_increments": increment_array.stack(),
        "increment_scores": score_array.stack(),
        "final_log_weights": final_log_weights,
        "ess_by_time": ess_array.stack(),
        "log_weight_spread_by_time": spread_array.stack(),
        "maximum_normalized_weight_by_time": maximum_weight_array.stack(),
        "minimum_ess": minimum_ess,
        "maximum_log_weight_spread": maximum_spread,
        "finite": finite,
        "particle_count": tf.constant(particle_count, tf.int32),
        "transition_count": tf.constant(transition_count, tf.int32),
    }


def _require_sealed_predator_prey_branch(
    branch: PreparedSourceOrderFrozenBranch,
) -> None:
    expected = {
        "target_id": TARGET_ID,
        "event_order": EVENT_ORDER,
        "target_seed": TARGET_SEED,
        "target_state_sha256": TARGET_STATE_SHA256,
        "target_observation_sha256": TARGET_OBSERVATION_SHA256,
        "transition_count": TARGET_HORIZON,
        "state_dimension": 2,
        "observation_dimension": 2,
    }
    actual = {
        "target_id": branch.target_id,
        "event_order": branch.event_order,
        "target_seed": branch.target_seed,
        "target_state_sha256": branch.target_state_sha256,
        "target_observation_sha256": branch.target_observation_sha256,
        "transition_count": branch.transition_count,
        "state_dimension": branch.state_dimension,
        "observation_dimension": branch.observation_dimension,
    }
    mismatches = [name for name, value in expected.items() if actual[name] != value]
    if mismatches:
        raise ValueError(
            "sealed predator-prey target mismatch: " + ", ".join(mismatches)
        )
    # The dataset factory independently verifies both sealed source hashes.
    # Proposal particles are not the latent truth path, so only observations
    # are expected to appear in the prepared APF branch.
    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
        generate_source_order_predator_prey_dataset_tf,
    )

    _sealed_states, sealed_observations = (
        generate_source_order_predator_prey_dataset_tf()
    )
    expected_observations = tf.cast(sealed_observations, branch.dtype)
    if branch.observations.shape != expected_observations.shape or not bool(
        tf.reduce_all(tf.equal(branch.observations, expected_observations)).numpy()
    ):
        raise ValueError("sealed predator-prey target mismatch: observations")


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
    return tf.ensure_shape(tf.cast(tf.convert_to_tensor(value), dtype), [length])


def _score_matrix(
    value: tf.Tensor,
    rows: int,
    columns: int,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    return tf.ensure_shape(
        tf.cast(tf.convert_to_tensor(value), dtype), [rows, columns]
    )


def _all_finite(values: Sequence[tf.Tensor]) -> tf.Tensor:
    return tf.reduce_all(
        tf.stack([tf.reduce_all(tf.math.is_finite(value)) for value in values])
    )


def _require_all_finite(name: str, value: tf.Tensor) -> None:
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise ValueError(f"{name} must contain only finite values")


def _branch_fingerprint(branch: PreparedSourceOrderFrozenBranch) -> str:
    digest = hashlib.sha256()
    for value in (
        ROUTE_ID,
        ROUTE_CLASSIFICATION,
        TARGET_CLASS,
        MEASURE_ID,
        branch.target_id,
        branch.event_order,
        branch.target_seed,
        branch.target_state_sha256,
        branch.target_observation_sha256,
        branch.proposal_compiler_id,
    ):
        _update_hash(digest, value)
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
    model: SourceOrderFrozenAPFModel,
    branch: PreparedSourceOrderFrozenBranch,
    require_predator_prey_target: bool,
) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, ROUTE_ID)
    _update_hash(digest, branch.branch_id)
    _update_hash(digest, bool(require_predator_prey_target))
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
        digest.update(
            b"string\0" + str(len(encoded)).encode("ascii") + b"\0" + encoded
        )
        return
    raise TypeError(f"unsupported identity payload type: {type(value).__name__}")


__all__ = [
    "COMPILER_CLASSIFICATION",
    "COMPILER_ID",
    "EVENT_ORDER",
    "PreparedSourceOrderFrozenBranch",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "SCORE_BACKEND_ID",
    "SourceOrderFrozenAPFProgram",
    "SourceOrderTTSIRTCompilation",
    "TARGET_HORIZON",
    "TARGET_ID",
    "TARGET_OBSERVATION_SHA256",
    "TARGET_SEED",
    "TARGET_STATE_SHA256",
    "compile_predator_prey_source_order_ttsirt_proposal_branch",
    "compile_source_order_ttsirt_proposal_branch",
    "prepare_predator_prey_fixed_variant_program",
    "prepare_predator_prey_source_order_branch",
    "prepare_source_order_frozen_apf_program",
    "prepare_source_order_frozen_branch",
]
