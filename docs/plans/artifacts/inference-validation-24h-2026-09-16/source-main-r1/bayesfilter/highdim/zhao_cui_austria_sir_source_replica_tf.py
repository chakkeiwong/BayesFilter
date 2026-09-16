"""Author-shaped fixed-parameter Zhao-Cui Austria SIR source replica.

This module binds the paper/author settings to the active observed-data target
and exposes a memory-bounded, serialized TT artifact.  The TensorFlow Adam
training backend is deliberately classified as an extension: it is not a
claim of equivalence to the author's random TT-cross/ALS implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.bases import (
    p85_author_sir_lagrangep_algebraic_product_basis_spec,
)
from bayesfilter.highdim.diagnostics import DensityMeasure, MassMeasure, MeasureConvention
from bayesfilter.highdim.squared_tt import SquaredTTDensity, TensorProductReferenceDensity
from bayesfilter.highdim.source_route import (
    SourceRouteCoordinateFrame,
    _p59_author_sir_deterministic_weighted_resample,
    _p59_author_sir_prior_sample_batch,
    _p59_author_sir_source_push_result,
    _p59_author_sir_source_density_callbacks,
    P63_AUTHOR_SIR_EXPANSION_FACTOR,
    P70_HOLDOUT_REPLAY_NORMALIZED_RESIDUAL_VETO,
    source_route_recenter,
    source_route_shifted_negative_log_target,
)
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_tf import (
    RUNTIME_FP32_OBSERVATION_SHA256,
    SIR_OBSERVATION_SHA256,
    TARGET_ID,
    make_austria_sir_observed_data_target,
)
from bayesfilter.highdim.models import zhao_cui_sir_austria_model


AUTHOR_PAPER_ANCHORS = (
    ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:693-719",
    ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:825-924",
    ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:2249-2305",
)
AUTHOR_CODE_ANCHORS = (
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:14-55",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-136",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43-100",
)
AUTHOR_BASIS_ORDER = 4
AUTHOR_BASIS_NUM_ELEMS = 8
AUTHOR_BASIS_DIM = AUTHOR_BASIS_ORDER * AUTHOR_BASIS_NUM_ELEMS + 1
AUTHOR_ALGEBRAIC_SCALE = 1.0
AUTHOR_MAX_RANK = 40
AUTHOR_INIT_RANK = 20
AUTHOR_KICK_RANK = 5
AUTHOR_MAIN_ALS_SWEEPS = 8
AUTHOR_LOW_ALS_SWEEPS = 2
ARTIFACT_SCHEMA = "bayesfilter.zhao_cui_austria_sir_author_source_replica_t1.v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
FIT_BACKEND = "p86_training_base_optimizer_not_author_tt_cross"
FRAME_ADAPTER_ID = "author_order_block_upper_full_covariance_v1"


def _convention() -> MeasureConvention:
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _tensor_hash(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if isinstance(value, tf.Tensor):
        raw = value.numpy()
        return raw.item() if value.shape.rank == 0 else raw.tolist()
    return value


def _semantic_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


@dataclass(frozen=True)
class AuthorSIRSourceReplicaSpec:
    """Setup-static source settings and bounded local execution controls."""

    basis_order: int = AUTHOR_BASIS_ORDER
    basis_num_elems: int = AUTHOR_BASIS_NUM_ELEMS
    algebraic_scale: float = AUTHOR_ALGEBRAIC_SCALE
    fit_rank: int = 4
    fit_sample_count: int = 256
    holdout_sample_count: int = 128
    train_steps: int = 16
    optimizer_batch_size: int = 64
    learning_rate: float = 3e-4
    l1_weight: float = 0.0
    l2_weight: float = 1e-8
    defensive_tau: float = 1e-8
    cdf_grid_size: int = 33
    cdf_bisection_steps: int = 16
    kr_max_batch_working_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        ints = (
            ("basis_order", self.basis_order, 1),
            ("basis_num_elems", self.basis_num_elems, 1),
            ("fit_rank", self.fit_rank, 1),
            ("fit_sample_count", self.fit_sample_count, 2),
            ("holdout_sample_count", self.holdout_sample_count, 0),
            ("train_steps", self.train_steps, 1),
            ("optimizer_batch_size", self.optimizer_batch_size, 2),
            ("cdf_grid_size", self.cdf_grid_size, 3),
            ("cdf_bisection_steps", self.cdf_bisection_steps, 1),
            ("kr_max_batch_working_bytes", self.kr_max_batch_working_bytes, 1),
        )
        for name, value, minimum in ints:
            if int(value) < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        if self.fit_rank > AUTHOR_MAX_RANK:
            raise ValueError("fit_rank exceeds the author max-rank baseline")
        for name in ("algebraic_scale", "learning_rate", "defensive_tau"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("l1_weight", "l2_weight"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def author_basis_exact(self) -> bool:
        return (
            int(self.basis_order) == AUTHOR_BASIS_ORDER
            and int(self.basis_num_elems) == AUTHOR_BASIS_NUM_ELEMS
            and float(self.algebraic_scale) == AUTHOR_ALGEBRAIC_SCALE
        )

    def payload(self) -> Mapping[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "basis_order",
                "basis_num_elems",
                "algebraic_scale",
                "fit_rank",
                "fit_sample_count",
                "holdout_sample_count",
                "train_steps",
                "optimizer_batch_size",
                "learning_rate",
                "l1_weight",
                "l2_weight",
                "defensive_tau",
                "cdf_grid_size",
                "cdf_bisection_steps",
                "kr_max_batch_working_bytes",
            )
        }

    def memory_forecast(self, *, particle_count: int) -> Mapping[str, int]:
        if int(particle_count) < 2:
            raise ValueError("particle_count must be at least two")
        rank = int(self.fit_rank)
        grid = int(self.cdf_grid_size)
        dim = 36
        core_values = dim * rank * rank * AUTHOR_BASIS_DIM
        scalar_slots = int(particle_count) * grid * (
            2 * dim + 4 + 4 * rank * rank
        ) + int(particle_count) * (3 * grid + dim + 8)
        return {
            "core_values": core_values,
            "core_bytes_fp64": core_values * 8,
            "kr_working_bytes_upper_bound": scalar_slots * 16,
            "configured_kr_budget_bytes": int(self.kr_max_batch_working_bytes),
            "full_grid_retention_forbidden": 1,
        }

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "AuthorSIRSourceReplicaSpec",
            "schema": ARTIFACT_SCHEMA,
            "source_settings": {
                "state_dimension": 18,
                "parameter_dimension": 0,
                "horizon": 20,
                "particle_baseline": 5000,
                "basis": "Lagrangep(4,8)",
                "algebraic_mapping_scale": 1.0,
                "max_rank": AUTHOR_MAX_RANK,
                "init_rank": AUTHOR_INIT_RANK,
                "kick_rank": AUTHOR_KICK_RANK,
                "main_als_sweeps": AUTHOR_MAIN_ALS_SWEEPS,
                "low_als_sweeps": AUTHOR_LOW_ALS_SWEEPS,
            },
            "active_target_id": TARGET_ID,
            "active_observation_sha256": SIR_OBSERVATION_SHA256,
            "runtime_fp32_observation_sha256": RUNTIME_FP32_OBSERVATION_SHA256,
            "event_order": "x0_then_transition_then_y1_y20",
            "author_joint_order": "x_t_then_theta_then_x_previous",
            "runtime_forward_compiler_order": "x_previous_then_x_t",
            "frame_adapter_id": FRAME_ADAPTER_ID,
            "frame_adapter_classification": "fixed_hmc_adaptation",
            "joint_order_adapter_classification": "extension_or_invention",
            "basis_exact_author_settings": self.author_basis_exact,
            "fit_backend": FIT_BACKEND,
            "fit_backend_classification": ROUTE_CLASSIFICATION,
            "paper_anchors": AUTHOR_PAPER_ANCHORS,
            "author_code_anchors": AUTHOR_CODE_ANCHORS,
            "controls": self.payload(),
            "memory_policy": "cores_and_streaming_working_set_only_no_full_tensor_grid",
            "nonclaims": (
                "no author TT-cross/ALS equivalence claim",
                "no exact likelihood or pseudo-marginal claim",
                "no production KR or HMC claim",
            ),
        }


@dataclass(frozen=True)
class FrozenAuthorSIRT1Artifact:
    spec: AuthorSIRSourceReplicaSpec
    frame: SourceRouteCoordinateFrame
    shift_constant: tf.Tensor
    cores: tuple[tf.Tensor, ...]
    target_identity: str
    artifact_id: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        cores = tuple(tf.convert_to_tensor(core, tf.float64) for core in self.cores)
        if len(cores) != 36:
            raise ValueError("author T1 artifact requires 36 cores")
        if any(core.shape.rank != 3 for core in cores):
            raise ValueError("TT cores must be rank-3")
        shift = tf.reshape(tf.convert_to_tensor(self.shift_constant, tf.float64), [])
        if not bool(tf.math.is_finite(shift).numpy()):
            raise ValueError("shift_constant must be finite")
        payload = _artifact_identity_payload(
            self.spec,
            self.frame,
            shift,
            cores,
            self.target_identity,
            self.diagnostics,
        )
        if str(self.artifact_id) != _semantic_hash(payload):
            raise ValueError("caller-stamped author source-replica identity rejected")
        object.__setattr__(self, "cores", cores)
        object.__setattr__(self, "shift_constant", shift)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def basis(self):
        return p85_author_sir_lagrangep_algebraic_product_basis_spec(
            dimension=36,
            convention=_convention(),
            order=self.spec.basis_order,
            num_elems=self.spec.basis_num_elems,
        ).build_product_basis()

    def density(self) -> SquaredTTDensity:
        basis = self.basis()
        ftt = FunctionalTT(tuple(TTCore(core) for core in self.cores), basis, basis.convention)
        defensive = TensorProductReferenceDensity(basis, basis.convention)
        tau = tf.constant(self.spec.defensive_tau, tf.float64)
        floor = tf.constant(1e-12, tf.float64)
        identity = SquaredTTDensity.expected_branch_identity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=floor,
            denominator_floor=floor,
            measure_convention=basis.convention,
        )
        return SquaredTTDensity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=floor,
            denominator_floor=floor,
            measure_convention=basis.convention,
            branch_identity=identity,
        )

    def transport(self) -> FixedTTSIRTTransport:
        return FixedTTSIRTTransport(
            density=self.density(),
            cdf_config=KRCDFConfig(
                grid_size=self.spec.cdf_grid_size,
                bisection_steps=self.spec.cdf_bisection_steps,
                monotonicity_tolerance=1e-12,
                bracket_tolerance=1e-12,
                denominator_floor=1e-12,
                max_floor_count=0,
                max_batch_working_bytes=self.spec.kr_max_batch_working_bytes,
            ),
        )

    def upper_conditional_inverse(
        self,
        conditioning_points: tf.Tensor,
        reference_points: tf.Tensor,
    ) -> tf.Tensor:
        """Generate current-state local coordinates given previous-state local coordinates."""

        return self.transport().conditional_inverse_transport_suffix(
            conditioning_points=conditioning_points,
            reference_points=reference_points,
        )

    def upper_conditional_forward(
        self,
        conditioning_points: tf.Tensor,
        generated_points: tf.Tensor,
    ) -> tf.Tensor:
        return self.transport().conditional_forward_transport_suffix(
            conditioning_points=conditioning_points,
            generated_points=generated_points,
        )

    def t1_algorithm3_diagnostic(
        self,
        *,
        particle_count: int,
        seed: int,
    ) -> Mapping[str, object]:
        """Run a bounded fixed-parameter Algorithm-3 proposal diagnostic."""

        n = int(particle_count)
        if n < 2:
            raise ValueError("particle_count must be at least two")
        model = zhao_cui_sir_austria_model()
        target = make_austria_sir_observed_data_target()
        previous_batch = _p59_author_sir_prior_sample_batch(
            model=model,
            sample_count=n,
            seed=int(seed),
        )
        previous_physical = previous_batch.samples
        g = 18
        previous_local = tf.linalg.triangular_solve(
            self.frame.matrix[g:, g:],
            previous_physical - self.frame.mu[g:, tf.newaxis],
            lower=True,
        )
        reference = tf.random.Generator.from_seed(int(seed) + 1).uniform(
            [g, n],
            minval=tf.constant(1e-6, tf.float64),
            maxval=tf.constant(1.0 - 1e-6, tf.float64),
            dtype=tf.float64,
        )
        current_local = self.upper_conditional_inverse(previous_local, reference)
        reconstructed = self.upper_conditional_forward(
            previous_local,
            current_local,
        )
        physical = (
            self.frame.matrix
            @ tf.concat([current_local, previous_local], axis=0)
            + self.frame.mu[:, tf.newaxis]
        )
        current_physical = physical[:g]
        conditional_local_log_q = (
            self.transport().conditional_forward_log_jacobian_suffix(
                previous_local,
                current_local,
            )
        )
        exact_conditional_local_log_q = (
            self.transport().conditional_proposal_log_density_suffix(
                conditioning_points=previous_local,
                generated_points=current_local,
            )
        )
        current_log_abs_det = tf.linalg.slogdet(self.frame.matrix[:g, :g])[1]
        conditional_physical_log_q = conditional_local_log_q - current_log_abs_det
        theta = tf.zeros([0], tf.float64)
        previous_rows = tf.transpose(previous_physical)
        current_rows = tf.transpose(current_physical)
        transition_log_density = model.transition_log_density(
            theta,
            previous_rows,
            current_rows,
            t=1,
        )
        observation_log_density = model.observation_log_density(
            theta,
            current_rows,
            target.source_observations[0],
            t=1,
        )
        log_weights = (
            transition_log_density
            + observation_log_density
            - conditional_physical_log_q
        )
        normalized = tf.nn.softmax(log_weights)
        ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized)))
        roundtrip = tf.reduce_max(tf.abs(reconstructed - reference))
        finite = tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(current_physical, [-1]),
                        conditional_physical_log_q,
                        log_weights,
                    ],
                    axis=0,
                )
            )
        )
        transport = self.transport()
        memory = max(
            transport.batch_working_set_estimate(axis=axis, sample_count=n)[
                "estimated_bytes"
            ]
            for axis in range(g)
        )
        return {
            "particle_count": n,
            "effective_sample_size": ess,
            "ess_fraction": ess / tf.cast(n, tf.float64),
            "roundtrip_max_abs": roundtrip,
            "finite": finite,
            "log_weight_spread": tf.reduce_max(log_weights) - tf.reduce_min(log_weights),
            "transition_log_density_range": tf.stack(
                [
                    tf.reduce_min(transition_log_density),
                    tf.reduce_max(transition_log_density),
                ]
            ),
            "observation_log_density_range": tf.stack(
                [
                    tf.reduce_min(observation_log_density),
                    tf.reduce_max(observation_log_density),
                ]
            ),
            "numerical_proposal_log_density_range": tf.stack(
                [
                    tf.reduce_min(conditional_physical_log_q),
                    tf.reduce_max(conditional_physical_log_q),
                ]
            ),
            "corrected_log_weight_range": tf.stack(
                [tf.reduce_min(log_weights), tf.reduce_max(log_weights)]
            ),
            "numerical_vs_exact_conditional_log_density_max_abs": tf.reduce_max(
                tf.abs(conditional_local_log_q - exact_conditional_local_log_q)
            ),
            "max_kr_working_bytes": memory,
            "current_block_log_abs_det": current_log_abs_det,
            "frame_adapter_id": FRAME_ADAPTER_ID,
            "conditional_route": "upper_suffix_reverse_numerical_kr",
            "classification": "fixed_hmc_adaptation_plus_extension_or_invention",
        }

    def payload(self) -> Mapping[str, object]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "artifact_id": self.artifact_id,
            "target_identity": self.target_identity,
            "spec": self.spec.manifest_payload(),
            "frame": self.frame.manifest_payload(),
            "shift_constant": self.shift_constant,
            "cores": tuple(
                {
                    "axis": axis,
                    "shape": core.shape.as_list(),
                    "sha256": _tensor_hash(core),
                    "values": core,
                }
                for axis, core in enumerate(self.cores)
            ),
            "diagnostics": self.diagnostics,
        }


def _artifact_identity_payload(spec, frame, shift, cores, target_identity, diagnostics):
    return {
        "schema": ARTIFACT_SCHEMA,
        "spec": spec.manifest_payload(),
        "frame": frame.manifest_payload(),
        "shift_constant": shift,
        "core_hashes": tuple(_tensor_hash(core) for core in cores),
        "target_identity": str(target_identity),
        "diagnostics": diagnostics,
    }


def make_author_order_block_upper_frame(
    frame: SourceRouteCoordinateFrame,
    *,
    generated_dimension: int,
) -> SourceRouteCoordinateFrame:
    """Refactor the full affine covariance to an upper conditional frame.

    For ``physical = mu + U @ local`` with
    ``U = [[A, B], [0, D]]``, the suffix block is independent of the generated
    prefix.  The construction preserves ``U @ U.T = M @ M.T`` from the
    source-shaped frame, while changing only the square root used for the
    fixed-HMC upper conditional adapter.
    """

    if not isinstance(frame, SourceRouteCoordinateFrame):
        raise TypeError("frame must be SourceRouteCoordinateFrame")
    g = int(generated_dimension)
    d = int(frame.dimension)
    if g <= 0 or g >= d:
        raise ValueError("generated_dimension must split the full frame")
    covariance = frame.matrix @ tf.transpose(frame.matrix)
    current = covariance[:g, :g]
    cross = covariance[:g, g:]
    previous = covariance[g:, g:]
    lower_previous = tf.linalg.cholesky(previous)
    upper_right = tf.transpose(
        tf.linalg.triangular_solve(
            lower_previous,
            tf.transpose(cross),
            lower=True,
        )
    )
    conditional_current = current - upper_right @ tf.transpose(upper_right)
    lower_current = tf.linalg.cholesky(conditional_current)
    upper = tf.concat(
        [
            tf.concat([lower_current, upper_right], axis=1),
            tf.concat(
                [
                    tf.zeros([d - g, g], dtype=tf.float64),
                    lower_previous,
                ],
                axis=1,
            ),
        ],
        axis=0,
    )
    if not bool(
        tf.reduce_all(tf.math.is_finite(upper)).numpy()
        and tf.reduce_all(tf.abs(upper[g:, :g]) <= 1e-12).numpy()
    ):
        raise ValueError("block-upper frame construction failed")
    return SourceRouteCoordinateFrame(
        mu=frame.mu,
        matrix=upper,
        expansion_factor=frame.expansion_factor,
    )


def _training_payload(target, *, fit_count: int, holdout_count: int) -> Mapping[str, object]:
    # This mirrors full_sol.m's source push/computeL construction while keeping
    # the active y1 observation and the unbounded AlgebraicMapping coordinates.
    model = zhao_cui_sir_austria_model()
    observation = tf.convert_to_tensor(target.source_observations[0], tf.float64)

    def one_cloud(
        *,
        sample_count: int,
        prior_seed: int,
        process_seed: int,
        frame: SourceRouteCoordinateFrame | None = None,
        shift_constant: tf.Tensor | None = None,
    ):
        previous = _p59_author_sir_prior_sample_batch(
            model=model, sample_count=int(sample_count), seed=int(prior_seed)
        )
        pushed = _p59_author_sir_source_push_result(
            model=model,
            previous_batch=previous,
            observation=observation,
            time_index=1,
            process_noise_seed=int(process_seed),
        )
        if frame is None:
            frame = source_route_recenter(
                samples=pushed.augmented_batch.samples,
                log_weights=pushed.augmented_batch.log_weights,
                expansion_factor=P63_AUTHOR_SIR_EXPANSION_FACTOR,
                covariance_jitter=1e-5,
                use_quantile_scale=True,
            )
            frame = make_author_order_block_upper_frame(
                frame,
                generated_dimension=18,
            )
        resampled, indices = _p59_author_sir_deterministic_weighted_resample(
            samples=pushed.augmented_batch.samples,
            log_weights=pushed.augmented_batch.log_weights,
        )
        local = tf.linalg.solve(
            frame.matrix, resampled - frame.mu[:, tf.newaxis]
        )
        physical = tf.linalg.matmul(frame.matrix, local) + frame.mu[:, tf.newaxis]
        prior, transition, likelihood = _p59_author_sir_source_density_callbacks(
            model, observation
        )
        # The callback consumes the author joint order [x_t, x_previous].
        joint = tf.concat([physical[:18], physical[18:]], axis=0)
        log_target = (
            prior(physical[18:])
            + transition(joint, 1)
            + likelihood(physical[:18], 1)
        )
        negative = -log_target
        local_negative = negative - frame.log_abs_det()
        domain = p85_author_sir_lagrangep_algebraic_product_basis_spec(
            dimension=1, convention=_convention()
        ).build_product_basis().bases[0].domain
        reference = domain.to_reference(tf.transpose(local))
        log_dxdz = tf.reduce_sum(
            domain.reference_to_domain_log_density(reference), axis=1
        )
        shifted_target = local_negative - log_dxdz
        shift = (
            tf.reduce_min(shifted_target)
            if shift_constant is None
            else tf.reshape(tf.convert_to_tensor(shift_constant, tf.float64), [])
        )
        target_values = tf.exp(
            -0.5 * source_route_shifted_negative_log_target(
                negative_log_target=shifted_target,
                shift_constant=shift,
            )
        )
        return {
            "points": tf.transpose(local),
            "target_values": target_values,
            "weights": tf.ones([int(sample_count)], tf.float64),
            "frame": frame,
            "shift_constant": shift,
            "indices": indices,
            "source_push_ess": pushed.diagnostics.effective_sample_size,
        }

    fit_data = one_cloud(
        sample_count=int(fit_count), prior_seed=6301, process_seed=6401
    )
    holdout = (
        one_cloud(
            sample_count=int(holdout_count),
            prior_seed=7301,
            process_seed=7401,
            frame=fit_data["frame"],
            shift_constant=fit_data["shift_constant"],
        )
        if int(holdout_count) > 0
        else None
    )
    return {
        "points": fit_data["points"],
        "target_values": fit_data["target_values"],
        "weights": fit_data["weights"],
        "holdout_points": None if holdout is None else holdout["points"],
        "holdout_values": None if holdout is None else holdout["target_values"],
        "holdout_weights": None if holdout is None else holdout["weights"],
        "fit_data_manifest": {
            "active_observation_sha256": SIR_OBSERVATION_SHA256,
            "runtime_fp32_observation_sha256": RUNTIME_FP32_OBSERVATION_SHA256,
            "source_joint_order": "x_t_theta_x_previous",
            "frame_adapter_id": FRAME_ADAPTER_ID,
            "basis_domain": "author_Lagrangep_4_8_AlgebraicMapping_1",
            "fit_sample_count": int(fit_count),
            "holdout_sample_count": int(holdout_count),
            "fit_source_push_ess": fit_data["source_push_ess"],
            "holdout_source_push_ess": None if holdout is None else holdout["source_push_ess"],
            "fit_resample_indices": fit_data["indices"],
        },
        "frame": fit_data["frame"],
        "shift_constant": fit_data["shift_constant"],
    }


def fit_author_sir_t1_source_replica(
    spec: AuthorSIRSourceReplicaSpec,
    *,
    seed: int = 8615,
) -> FrozenAuthorSIRT1Artifact:
    """Fit and freeze one active-data T1 source-replica candidate."""

    if not isinstance(spec, AuthorSIRSourceReplicaSpec):
        raise TypeError("spec must be AuthorSIRSourceReplicaSpec")
    target = make_austria_sir_observed_data_target()
    if target.manifest["runtime_fp32_observation_sha256"] != RUNTIME_FP32_OBSERVATION_SHA256:
        raise ValueError("active observation identity rejected")
    from scripts.p86_author_lagrangep_phase5_budget_fit import (
        _rank_tuple,
        _run_training_base,
    )

    payload = _training_payload(
        target,
        fit_count=spec.fit_sample_count,
        holdout_count=spec.holdout_sample_count,
    )
    basis = p85_author_sir_lagrangep_algebraic_product_basis_spec(
        dimension=36,
        convention=_convention(),
        order=spec.basis_order,
        num_elems=spec.basis_num_elems,
    ).build_product_basis()
    training = _run_training_base(
        product_basis=basis,
        ranks=_rank_tuple(36, spec.fit_rank),
        batch_payload=payload,
        seed=int(seed),
        learning_rate=spec.learning_rate,
        optimizer_batch_size=spec.optimizer_batch_size,
        prefit_steps=0,
        train_steps=spec.train_steps,
        max_seconds=1800.0,
        l1_weight=spec.l1_weight,
        l2_weight=spec.l2_weight,
        serialize_trained_cores=True,
    )
    cores = tuple(tf.identity(core) for core in training["trainer"].variables)
    diagnostics = {
        "schema": ARTIFACT_SCHEMA,
        "route_classification": ROUTE_CLASSIFICATION,
        "fit_backend": FIT_BACKEND,
        "target_id": TARGET_ID,
        "active_observation_sha256": SIR_OBSERVATION_SHA256,
        "runtime_fp32_observation_sha256": RUNTIME_FP32_OBSERVATION_SHA256,
        "fit_data_manifest": payload["fit_data_manifest"],
        "training_summary": {
            "fit_residual": training["fit_residual"],
            "holdout_residual": training["holdout_residual"],
            "normalizer": training["normalizer"],
            "runtime_seconds": training["runtime_seconds"],
            "trained_core_serialization": training["trained_core_serialization"],
        },
        "source_paper_anchors": AUTHOR_PAPER_ANCHORS,
        "author_code_anchors": AUTHOR_CODE_ANCHORS,
        "nonclaims": spec.manifest_payload()["nonclaims"],
    }
    frame = payload["frame"]
    artifact_payload = _artifact_identity_payload(
        spec,
        frame,
        payload["shift_constant"],
        cores,
        target.target_identity,
        diagnostics,
    )
    return FrozenAuthorSIRT1Artifact(
        spec=spec,
        frame=frame,
        shift_constant=payload["shift_constant"],
        cores=cores,
        target_identity=target.target_identity,
        artifact_id=_semantic_hash(artifact_payload),
        diagnostics=diagnostics,
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "AUTHOR_ALGEBRAIC_SCALE",
    "AUTHOR_BASIS_DIM",
    "AUTHOR_BASIS_NUM_ELEMS",
    "AUTHOR_BASIS_ORDER",
    "AUTHOR_CODE_ANCHORS",
    "AUTHOR_PAPER_ANCHORS",
    "FRAME_ADAPTER_ID",
    "P70_HOLDOUT_REPLAY_NORMALIZED_RESIDUAL_VETO",
    "AuthorSIRSourceReplicaSpec",
    "FrozenAuthorSIRT1Artifact",
    "fit_author_sir_t1_source_replica",
    "make_author_order_block_upper_frame",
]
