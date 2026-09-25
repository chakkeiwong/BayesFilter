"""Active-target Austria SIR adjacent squared-TT proposal fitting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from bayesfilter.highdim.diagnostics import DensityMeasure, MassMeasure, MeasureConvention
from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.stochastic_density_training import (
    P75ObjectiveBatch,
    P75TrainableTTConfig,
    TrainableFunctionalTT,
    make_adam_optimizer,
    terms_payload,
)
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.ukf_initializer import (
    p76_embed_rank_one_with_seeded_channels,
)
from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_tf import (
    EVENT_ORDER,
    RUNTIME_FP32_OBSERVATION_SHA256,
    TARGET_ID,
    make_austria_sir_observed_data_target,
    prepare_austria_sir_source_order_branch,
)
from bayesfilter.highdim.zhao_cui_predator_prey_fixed_variant_tf import (
    SourceOrderTTSIRTCompilation,
    compile_source_order_ttsirt_proposal_branch,
)
from bayesfilter.testing.sir_filter_neutra_target_design_tf import SIR_DATASET_SEED


FIT_DTYPE = tf.float64
FITTER_ID = "zhao_cui_austria_sir_latent_preclip_t1_stochastic_tt_fitter_v1"
ARTIFACT_SCHEMA = "bayesfilter.zhao_cui_austria_sir_t1_frozen_tt.v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
T1_CALIBRATION_SEED_BASE = 973_100
T1_VALIDATION_SEED = 974_100


def _tensor_hash(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        raw = value.numpy()
        return raw.item() if value.shape.rank == 0 else raw.tolist()
    if isinstance(value, tf.dtypes.DType):
        return value.name
    return value


def _convention() -> MeasureConvention:
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="uniform_probability_on_minus_one_one",
    )


def _basis(dimension: int, degree: int) -> ProductBasis:
    return ProductBasis(
        [
            LegendreBasis1D(BoundedInterval(-1.0, 1.0), int(degree))
            for _ in range(int(dimension))
        ],
        _convention(),
    )


@dataclass(frozen=True)
class WhitenedAlgebraicCoordinateMap:
    """Full-support map `x = location + matrix h(u)` with algebraic `h`."""

    locations: tf.Tensor
    matrix: tf.Tensor

    def __post_init__(self) -> None:
        locations = tf.reshape(tf.convert_to_tensor(self.locations, FIT_DTYPE), [-1])
        matrix = tf.convert_to_tensor(self.matrix, FIT_DTYPE)
        dimension = int(locations.shape[0])
        if matrix.shape != (dimension, dimension):
            raise ValueError("matrix must be square and match locations")
        sign, log_abs_det = tf.linalg.slogdet(matrix)
        if not bool(
            tf.reduce_all(tf.math.is_finite(locations)).numpy()
            and tf.reduce_all(tf.math.is_finite(matrix)).numpy()
            and tf.math.is_finite(log_abs_det).numpy()
            and (sign != 0.0).numpy()
        ):
            raise ValueError("coordinate map must be finite and nonsingular")
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "matrix", matrix)

    @property
    def dimension(self) -> int:
        return int(self.locations.shape[0])

    def forward(self, reference_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        reference = tf.convert_to_tensor(reference_points, FIT_DTYPE)
        if reference.shape.rank != 2 or reference.shape[1] != self.dimension:
            raise ValueError("reference_points must have shape [sample,dimension]")
        clipped = tf.clip_by_value(reference, -1.0 + 1e-12, 1.0 - 1e-12)
        whitened = clipped * tf.math.rsqrt(1.0 - tf.square(clipped))
        physical = self.locations[tf.newaxis, :] + tf.linalg.matmul(
            whitened, self.matrix, transpose_b=True
        )
        log_det = self.log_abs_det() - 1.5 * tf.reduce_sum(
            tf.math.log(1.0 - tf.square(clipped)), axis=1
        )
        return physical, log_det

    def inverse(self, physical_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        physical = tf.convert_to_tensor(physical_points, FIT_DTYPE)
        if physical.shape.rank != 2 or physical.shape[1] != self.dimension:
            raise ValueError("physical_points must have shape [sample,dimension]")
        whitened = tf.linalg.matrix_transpose(
            tf.linalg.solve(
                self.matrix,
                tf.linalg.matrix_transpose(physical - self.locations[tf.newaxis, :]),
            )
        )
        reference = whitened * tf.math.rsqrt(1.0 + tf.square(whitened))
        inverse_log_det = -self.log_abs_det() - 1.5 * tf.reduce_sum(
            tf.math.log1p(tf.square(whitened)), axis=1
        )
        return reference, inverse_log_det

    def log_abs_det(self) -> tf.Tensor:
        return tf.linalg.slogdet(self.matrix)[1]

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "WhitenedAlgebraicCoordinateMap",
            "classification": ROUTE_CLASSIFICATION,
            "locations": self.locations,
            "matrix": self.matrix,
            "formula": "x=location+matrix@(u/sqrt(1-u^2))",
            "full_support": True,
        }


@dataclass(frozen=True)
class WhitenedGaussianQuantileCoordinateMap:
    """Full-covariance Gaussian quantile map from ``[-1,1]^d``."""

    locations: tf.Tensor
    matrix: tf.Tensor

    def __post_init__(self) -> None:
        locations = tf.reshape(tf.convert_to_tensor(self.locations, FIT_DTYPE), [-1])
        matrix = tf.convert_to_tensor(self.matrix, FIT_DTYPE)
        dimension = int(locations.shape[0])
        if matrix.shape != (dimension, dimension):
            raise ValueError("matrix must be square and match locations")
        sign, log_abs_det = tf.linalg.slogdet(matrix)
        if not bool(
            tf.reduce_all(tf.math.is_finite(locations)).numpy()
            and tf.reduce_all(tf.math.is_finite(matrix)).numpy()
            and tf.math.is_finite(log_abs_det).numpy()
            and (sign != 0.0).numpy()
        ):
            raise ValueError("coordinate map must be finite and nonsingular")
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "matrix", matrix)

    @property
    def dimension(self) -> int:
        return int(self.locations.shape[0])

    def forward(self, reference_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        reference = tf.convert_to_tensor(reference_points, FIT_DTYPE)
        if reference.shape.rank != 2 or reference.shape[1] != self.dimension:
            raise ValueError("reference_points must have shape [sample,dimension]")
        clipped = tf.clip_by_value(reference, -1.0 + 1e-12, 1.0 - 1e-12)
        whitened = tf.sqrt(tf.constant(2.0, FIT_DTYPE)) * tf.math.erfinv(clipped)
        physical = self.locations[tf.newaxis, :] + tf.linalg.matmul(
            whitened, self.matrix, transpose_b=True
        )
        log_det = self.log_abs_det() + tf.reduce_sum(
            0.5 * tf.math.log(tf.constant(math.pi / 2.0, FIT_DTYPE))
            + 0.5 * tf.square(whitened),
            axis=1,
        )
        return physical, log_det

    def inverse(self, physical_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        physical = tf.convert_to_tensor(physical_points, FIT_DTYPE)
        if physical.shape.rank != 2 or physical.shape[1] != self.dimension:
            raise ValueError("physical_points must have shape [sample,dimension]")
        whitened = tf.linalg.matrix_transpose(
            tf.linalg.solve(
                self.matrix,
                tf.linalg.matrix_transpose(physical - self.locations[tf.newaxis, :]),
            )
        )
        reference = tf.math.erf(whitened / tf.sqrt(tf.constant(2.0, FIT_DTYPE)))
        inverse_log_det = -self.log_abs_det() + tf.reduce_sum(
            0.5 * tf.math.log(tf.constant(2.0 / math.pi, FIT_DTYPE))
            - 0.5 * tf.square(whitened),
            axis=1,
        )
        return reference, inverse_log_det

    def log_abs_det(self) -> tf.Tensor:
        return tf.linalg.slogdet(self.matrix)[1]

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "WhitenedGaussianQuantileCoordinateMap",
            "classification": ROUTE_CLASSIFICATION,
            "locations": self.locations,
            "matrix": self.matrix,
            "formula": "x=location+matrix@(sqrt(2)*erfinv(u))",
            "full_support": True,
        }


@dataclass(frozen=True)
class AustriaSIRT1ProposalSpec:
    degree: int = 2
    rank: int = 4
    batch_size: int = 128
    train_batches: int = 4
    learning_rate: float = 1e-3
    l1_weight: float = 1e-9
    l2_weight: float = 1e-8
    defensive_tau: float = 1e-8
    cdf_grid_size: int = 33
    cdf_bisection_steps: int = 16
    kr_max_batch_working_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.degree < 2 or self.rank < 1:
            raise ValueError("degree must be >=2 and rank must be positive")
        if self.batch_size < 2 or self.train_batches < 1:
            raise ValueError("batch size and train batches must be positive")
        for name in (
            "learning_rate",
            "l1_weight",
            "l2_weight",
            "defensive_tau",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.learning_rate <= 0.0 or self.defensive_tau <= 0.0:
            raise ValueError("learning_rate and defensive_tau must be positive")

    def payload(self) -> Mapping[str, object]:
        return {
            key: getattr(self, key)
            for key in (
                "degree",
                "rank",
                "batch_size",
                "train_batches",
                "learning_rate",
                "l1_weight",
                "l2_weight",
                "defensive_tau",
                "cdf_grid_size",
                "cdf_bisection_steps",
                "kr_max_batch_working_bytes",
            )
        }


@dataclass(frozen=True)
class AustriaSIRSourceOrderT1Guide:
    previous_map: WhitenedGaussianQuantileCoordinateMap
    current_map: WhitenedGaussianQuantileCoordinateMap
    manifest: Mapping[str, object]


def make_source_order_t1_ukf_guide() -> AustriaSIRSourceOrderT1Guide:
    """Build a transition-before-observation UKF geometry guide for active `y1`."""

    target = make_austria_sir_observed_data_target()
    model = latent_preclip_zhao_cui_sir_austria_model().physical_model.base_model
    mean0 = tf.convert_to_tensor(model.initial_mean, FIT_DTYPE)
    covariance0 = tf.convert_to_tensor(model.initial_covariance, FIT_DTYPE)
    dimension = int(mean0.shape[0])
    spread = tf.constant(float(dimension), FIT_DTYPE)
    factor0 = tf.linalg.cholesky(covariance0)
    offsets = tf.concat(
        [
            tf.zeros([1, dimension], FIT_DTYPE),
            tf.sqrt(spread) * tf.eye(dimension, dtype=FIT_DTYPE),
            -tf.sqrt(spread) * tf.eye(dimension, dtype=FIT_DTYPE),
        ],
        axis=0,
    )
    points0 = mean0[tf.newaxis, :] + offsets @ tf.transpose(factor0)
    weights = tf.concat(
        [tf.zeros([1], FIT_DTYPE), tf.fill([2 * dimension], 0.5 / spread)], axis=0
    )
    propagated = model.transition_mean(points0)
    mean1 = tf.reduce_sum(weights[:, tf.newaxis] * propagated, axis=0)
    centered1 = propagated - mean1[tf.newaxis, :]
    covariance1 = (
        tf.einsum("n,ni,nj->ij", weights, centered1, centered1)
        + model.process_covariance
    )
    observed_indices = tf.range(1, dimension, 2, dtype=tf.int32)
    observation_mean = tf.gather(mean1, observed_indices)
    cross_covariance = tf.gather(covariance1, observed_indices, axis=1)
    innovation_covariance = (
        tf.gather(tf.gather(covariance1, observed_indices, axis=0), observed_indices, axis=1)
        + model.observation_covariance
    )
    gain = tf.transpose(
        tf.linalg.solve(innovation_covariance, tf.transpose(cross_covariance))
    )
    residual = target.source_observations[0] - observation_mean
    filtered_mean = mean1 + tf.linalg.matvec(gain, residual)
    filtered_covariance = covariance1 - gain @ innovation_covariance @ tf.transpose(gain)
    filtered_covariance = 0.5 * (filtered_covariance + tf.transpose(filtered_covariance))
    eigenvalues, eigenvectors = tf.linalg.eigh(filtered_covariance)
    floored = tf.maximum(eigenvalues, tf.constant(1e-6, FIT_DTYPE))
    filtered_factor = eigenvectors @ tf.linalg.diag(tf.sqrt(floored))

    manifest = {
        "guide_id": "austria_sir_source_order_ukf_t1_block_diagonal_v1",
        "claim_class": "scout_not_truth",
        "event_order": "x0_then_transition_then_y1",
        "source_observation_sha256": target.manifest["source_observation_sha256"],
        "previous_current_cross_covariance": "dropped_controlled_baseline",
        "previous_mean": mean0,
        "current_mean": filtered_mean,
        "previous_covariance": covariance0,
        "current_covariance": filtered_covariance,
        "current_map_role": "ukf_filtered_gaussian_quantile_geometry",
        "nonclaims": (
            "UKF is geometry only",
            "not target truth",
            "not likelihood or score evidence",
        ),
    }
    return AustriaSIRSourceOrderT1Guide(
        previous_map=WhitenedGaussianQuantileCoordinateMap(mean0, factor0),
        current_map=WhitenedGaussianQuantileCoordinateMap(
            filtered_mean, filtered_factor
        ),
        manifest=MappingProxyType(manifest),
    )


def _training_batch(
    *,
    guide: AustriaSIRSourceOrderT1Guide,
    sample_count: int,
    seed: int,
    label: str,
) -> tuple[P75ObjectiveBatch, Mapping[str, object]]:
    target = make_austria_sir_observed_data_target()
    model = latent_preclip_zhao_cui_sir_austria_model()
    theta = tf.zeros([3], FIT_DTYPE)
    roots = tf.random.experimental.stateless_split(tf.constant([seed, 17], tf.int32), 2)
    initial_noise = tf.random.stateless_normal(
        [sample_count, 18], roots[0], dtype=FIT_DTYPE
    )
    process_noise = tf.random.stateless_normal(
        [sample_count, 18], roots[1], dtype=FIT_DTYPE
    )
    scaled = model.physical_model.scaled_model(theta)
    initial_chol = tf.linalg.cholesky(scaled.initial_covariance)
    z0 = scaled.initial_mean[tf.newaxis, :] + tf.linalg.matmul(
        initial_noise, initial_chol, transpose_b=True
    )
    z1 = model.transition_push_from_standard_normal(theta, z0, process_noise, 1)
    previous_reference, _ = guide.previous_map.inverse(z0)
    current_reference, _ = guide.current_map.inverse(z1)
    points = tf.concat([previous_reference, current_reference], axis=1)
    if not bool(tf.reduce_all(tf.abs(points) < 1.0).numpy()):
        raise ValueError("algebraic inverse must produce interior reference points")
    log_likelihood = model.observation_log_density(
        theta, z1, target.source_observations[0], 1
    )
    shift = tf.reduce_max(log_likelihood)
    importance = tf.exp(log_likelihood - shift)
    batch = P75ObjectiveBatch(
        points=points,
        target_values=tf.ones([sample_count], FIT_DTYPE),
        weights=importance,
        provenance_label=label,
    )
    return batch, {
        "seed": int(seed),
        "sample_count": int(sample_count),
        "point_hash": _tensor_hash(points),
        "observation_hash": _tensor_hash(target.source_observations[:1]),
        "log_likelihood_shift": shift,
        "objective_identity": (
            "samples_from_p_z0_f_z1_given_z0_weighted_by_g_y1_given_z1; "
            "P75 target_values=1 keeps defensive_tau out_of_empirical_target"
        ),
        "importance_ess": 1.0
        / tf.reduce_sum(
            tf.square(tf.nn.softmax(log_likelihood))
        ),
        "minimum_reference": tf.reduce_min(points),
        "maximum_reference": tf.reduce_max(points),
    }


@dataclass(frozen=True)
class FrozenAustriaSIRT1TTArtifact:
    spec: AustriaSIRT1ProposalSpec
    previous_map: WhitenedGaussianQuantileCoordinateMap
    current_map: WhitenedGaussianQuantileCoordinateMap
    cores: tuple[tf.Tensor, ...]
    diagnostics: Mapping[str, object]
    artifact_id: str

    def __post_init__(self) -> None:
        cores = tuple(tf.convert_to_tensor(core, FIT_DTYPE) for core in self.cores)
        if len(cores) != 36:
            raise ValueError("T1 artifact requires 36 TT cores")
        payload = _artifact_identity_payload(
            self.spec, self.previous_map, self.current_map, cores, self.diagnostics
        )
        if str(self.artifact_id) != _semantic_hash(payload):
            raise ValueError("caller-stamped T1 artifact identity rejected")
        object.__setattr__(self, "cores", cores)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def density(self) -> SquaredTTDensity:
        basis = _basis(36, self.spec.degree)
        ftt = FunctionalTT(
            tuple(TTCore(core) for core in self.cores), basis, basis.convention
        )
        defensive = TensorProductReferenceDensity(basis, basis.convention)
        tau = tf.constant(self.spec.defensive_tau, FIT_DTYPE)
        floor = tf.constant(1e-12, FIT_DTYPE)
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

    def initial_transport(self) -> FixedTTSIRTTransport:
        """Return a valid 18D reference proposal for ``z0``.

        The fitted 36D target supplies the conditional proposal.  The initial
        proposal is deliberately separate, constant in the previous-state
        reference coordinates, and is corrected exactly by ``log p-log q``.
        """

        basis = _basis(18, 0)
        ftt = FunctionalTT(
            tuple(TTCore(tf.ones([1, 1, 1], FIT_DTYPE)) for _ in range(18)),
            basis,
            basis.convention,
        )
        defensive = TensorProductReferenceDensity(basis, basis.convention)
        tau = tf.constant(self.spec.defensive_tau, FIT_DTYPE)
        floor = tf.constant(1e-12, FIT_DTYPE)
        identity = SquaredTTDensity.expected_branch_identity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=floor,
            denominator_floor=floor,
            measure_convention=basis.convention,
        )
        density = SquaredTTDensity(
            sqrt_tt=ftt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=floor,
            denominator_floor=floor,
            measure_convention=basis.convention,
            branch_identity=identity,
        )
        return FixedTTSIRTTransport(
            density=density,
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

    def payload(self) -> Mapping[str, object]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "artifact_id": self.artifact_id,
            "spec": self.spec.payload(),
            "previous_map": self.previous_map.manifest_payload(),
            "current_map": self.current_map.manifest_payload(),
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


def _artifact_identity_payload(spec, previous_map, current_map, cores, diagnostics):
    return {
        "schema": ARTIFACT_SCHEMA,
        "fitter_id": FITTER_ID,
        "target_id": TARGET_ID,
        "runtime_observation_sha256": RUNTIME_FP32_OBSERVATION_SHA256,
        "spec": spec.payload(),
        "previous_map": previous_map.manifest_payload(),
        "current_map": current_map.manifest_payload(),
        "core_hashes": tuple(_tensor_hash(core) for core in cores),
        "diagnostics": diagnostics,
    }


def fit_austria_sir_t1_proposal(
    spec: AustriaSIRT1ProposalSpec,
    *,
    seed: int = T1_CALIBRATION_SEED_BASE,
) -> FrozenAustriaSIRT1TTArtifact:
    """Fit one active-observation 36D candidate; this is not tuning selection."""

    guide = make_source_order_t1_ukf_guide()
    basis = _basis(36, spec.degree)
    ranks = tuple([1] + [spec.rank] * 35 + [1])
    coefficients = tuple(
        tf.concat(
            [tf.ones([1], FIT_DTYPE), tf.zeros([spec.degree], FIT_DTYPE)], axis=0
        )
        for _ in range(36)
    )
    initial_cores = p76_embed_rank_one_with_seeded_channels(
        coefficients, ranks=ranks, seed_epsilon=1e-3
    )
    config = P75TrainableTTConfig(
        product_basis=basis,
        ranks=ranks,
        tau=tf.constant(spec.defensive_tau, FIT_DTYPE),
        normalizer_floor=tf.constant(1e-12, FIT_DTYPE),
        denominator_floor=tf.constant(1e-12, FIT_DTYPE),
        l1_weight=tf.constant(spec.l1_weight, FIT_DTYPE),
        l2_weight=tf.constant(spec.l2_weight, FIT_DTYPE),
        learning_rate=spec.learning_rate,
        gradient_clip_norm=100.0,
        seed=int(seed),
        metadata={
            "target": "active_austria_sir_latent_preclip_t1",
            "observation_role": "active_y1",
            "initializer": "constant_reference_density_in_ukf_innovation_frame",
            "ukf_geometry_applied": True,
            "l1_weight_tuned": False,
            "classification": ROUTE_CLASSIFICATION,
        },
    )
    trainer = TrainableFunctionalTT(config, initial_cores=initial_cores)
    baseline_density = trainer.snapshot_density()
    optimizer = make_adam_optimizer(config)
    trace = []
    training_manifests = []
    for batch_index in range(spec.train_batches):
        batch, manifest = _training_batch(
            guide=guide,
            sample_count=spec.batch_size,
            seed=int(seed) + batch_index,
            label=f"austria_sir_t1_calibration_batch_{batch_index}",
        )
        terms = trainer.train_step(batch, optimizer)
        trace.append({"batch": batch_index + 1, "terms": terms_payload(terms)})
        training_manifests.append(manifest)
    validation, validation_manifest = _training_batch(
        guide=guide,
        sample_count=spec.batch_size,
        seed=T1_VALIDATION_SEED,
        label="austria_sir_t1_validation_metric_only",
    )
    alpha = validation.weights * tf.square(validation.target_values)
    alpha = alpha / tf.reduce_sum(alpha)
    baseline_ce = -tf.reduce_sum(alpha * baseline_density.log_density(validation.points))
    density = trainer.snapshot_density()
    trained_ce = -tf.reduce_sum(alpha * density.log_density(validation.points))
    cores = tuple(tf.identity(core) for core in trainer.variables)
    diagnostics = {
        "fitter_id": FITTER_ID,
        "target_id": TARGET_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "source_observation_sha256": make_austria_sir_observed_data_target().manifest[
            "source_observation_sha256"
        ],
        "runtime_observation_sha256": RUNTIME_FP32_OBSERVATION_SHA256,
        "theta_reference": (0.0, 0.0, 0.0),
        "measure": "latent_preclip_z0_z1_lebesgue_v1",
        "sampling_law": "p_z0_times_f_z1_given_z0",
        "importance_factor": "g_y1_given_z1",
        "empirical_target_derivation": (
            "E_{p(z0)f(z1|z0)}[g(y1|z1) log q_U(U)] is the cross-entropy "
            "of the transformed unnormalized p(z0)f(z1|z0)g(y1|z1) target; "
            "coordinate Jacobians are carried by the sampling distribution"
        ),
        "ukf_guide": guide.manifest,
        "training_batches": tuple(training_manifests),
        "validation": validation_manifest,
        "baseline_validation_cross_entropy": baseline_ce,
        "trained_validation_cross_entropy": trained_ce,
        "validation_cross_entropy_delta": trained_ce - baseline_ce,
        "trace": tuple(trace),
        "density_normalizer": density.normalizer(),
        "core_count": len(cores),
        "core_value_count": sum(int(tf.size(core).numpy()) for core in cores),
        "calibration_validation_seed_disjoint": all(
            int(seed) + index != T1_VALIDATION_SEED
            for index in range(spec.train_batches)
        ),
        "nonclaims": (
            "single candidate not selected tuning artifact",
            "T1 only",
            "no proposal-quality or claim-run admission",
        ),
    }
    identity_payload = _artifact_identity_payload(
        spec, guide.previous_map, guide.current_map, cores, diagnostics
    )
    return FrozenAustriaSIRT1TTArtifact(
        spec=spec,
        previous_map=guide.previous_map,
        current_map=guide.current_map,
        cores=cores,
        diagnostics=diagnostics,
        artifact_id=_semantic_hash(identity_payload),
    )


def compile_austria_sir_t1_proposal_branch(
    artifact: FrozenAustriaSIRT1TTArtifact,
    *,
    initial_reference_points: tf.Tensor,
    ancestor_uniforms: tf.Tensor,
    transition_reference_points: tf.Tensor,
    inverse_microbatch_size: int | None = None,
) -> SourceOrderTTSIRTCompilation:
    """Compile and seal one fitted ``y1`` proposal into the Austria APF branch."""

    if not isinstance(artifact, FrozenAustriaSIRT1TTArtifact):
        raise TypeError("artifact must be a FrozenAustriaSIRT1TTArtifact")
    target = make_austria_sir_observed_data_target()
    initial_reference = tf.convert_to_tensor(initial_reference_points, FIT_DTYPE)
    if initial_reference.shape.rank != 2 or initial_reference.shape[0] != 18:
        raise ValueError("initial_reference_points must have shape [18,particle]")
    particle_count = int(initial_reference.shape[1])
    auxiliary_log = tf.fill(
        [1, particle_count], -tf.math.log(tf.cast(particle_count, FIT_DTYPE))
    )
    generic = compile_source_order_ttsirt_proposal_branch(
        observations=target.source_observations[:1],
        initial_transport=artifact.initial_transport(),
        transition_transports=(artifact.transport(),),
        previous_coordinate_maps=(artifact.previous_map,),
        current_coordinate_maps=(artifact.current_map,),
        initial_reference_points=initial_reference,
        ancestor_uniforms=ancestor_uniforms,
        auxiliary_log_probabilities=auxiliary_log,
        transition_reference_points=transition_reference_points,
        target_id=f"{TARGET_ID}_prefix_t1",
        event_order=EVENT_ORDER,
        target_seed=SIR_DATASET_SEED,
        target_state_sha256=_tensor_hash(target.source_states[:2]),
        target_observation_sha256=_tensor_hash(target.observations[:1]),
        tuning_artifact_id=artifact.artifact_id,
        online_dtype=tf.float32,
        inverse_microbatch_size=inverse_microbatch_size,
    )
    sealed = prepare_austria_sir_source_order_branch(
        target=target,
        observations=generic.branch.observations,
        states=generic.branch.states,
        initial_log_proposal_density=generic.branch.initial_log_proposal_density,
        ancestors=generic.branch.ancestors,
        auxiliary_log_probabilities=generic.branch.auxiliary_log_probabilities,
        transition_log_proposal_density=generic.branch.transition_log_proposal_density,
        proposal_compiler_id=generic.compiler_id,
    )
    manifest = dict(generic.manifest)
    manifest.update(
        {
            "austria_t1_artifact_id": artifact.artifact_id,
            "austria_target_identity": target.target_identity,
        "initial_proposal": "exact_gaussian_prior_via_uniform_quantile_map",
            "branch_id": sealed.branch_id,
        }
    )
    return SourceOrderTTSIRTCompilation(
        branch=sealed,
        compiler_id=generic.compiler_id,
        manifest=manifest,
    )


def load_t1_artifact(payload: Mapping[str, object]) -> FrozenAustriaSIRT1TTArtifact:
    if payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported T1 artifact schema")
    spec = AustriaSIRT1ProposalSpec(**dict(payload["spec"]))
    previous = payload["previous_map"]
    current = payload["current_map"]
    cores = tuple(tf.constant(record["values"], FIT_DTYPE) for record in payload["cores"])
    for record, core in zip(payload["cores"], cores):
        if record["shape"] != core.shape.as_list() or record["sha256"] != _tensor_hash(core):
            raise ValueError("serialized TT core identity rejected")
    return FrozenAustriaSIRT1TTArtifact(
        spec=spec,
        previous_map=WhitenedGaussianQuantileCoordinateMap(
            previous["locations"], previous["matrix"]
        ),
        current_map=WhitenedGaussianQuantileCoordinateMap(
            current["locations"], current["matrix"]
        ),
        cores=cores,
        diagnostics=dict(payload["diagnostics"]),
        artifact_id=str(payload["artifact_id"]),
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "AustriaSIRSourceOrderT1Guide",
    "AustriaSIRT1ProposalSpec",
    "FrozenAustriaSIRT1TTArtifact",
    "WhitenedAlgebraicCoordinateMap",
    "WhitenedGaussianQuantileCoordinateMap",
    "compile_austria_sir_t1_proposal_branch",
    "fit_austria_sir_t1_proposal",
    "load_t1_artifact",
    "make_source_order_t1_ukf_guide",
]
