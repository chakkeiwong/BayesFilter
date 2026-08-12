"""Compact external-theta child mechanics for the admitted Lane-B TT parents.

This is an extension/invention mechanics surface. It preserves parent cores and
the defensive squared-TT program at theta zero; it is not a trained parameter
model and does not claim a Zhao-Cui source-faithful parameter score.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import AlgebraicMap
from bayesfilter.highdim.fixed_branch import BranchHash, BranchIdentity, BranchManifest
from bayesfilter.highdim.source_route import SourceRouteCoordinateFrame
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.tt import FunctionalTT, TTCore
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    LaneBT1Artifact,
    lane_b_measure_convention,
    lane_b_product_basis,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (
    LaneBT2Artifact,
    T2_ARTIFACT_SCHEMA,
    T2_IDENTITY_SCHEMA,
    _t2_estimate_from_payload,
    issue_lane_b_t2_identity,
    t2_source_closure,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import tensor_sha256


DTYPE = tf.float64
PARAMETER_DIM = 3
CHILD_SCHEMA = "bayesfilter.zhao_cui_austria_sir_parameter_child.v1"
CHILD_IDENTITY_SCHEMA = "bayesfilter.zhao_cui_austria_sir_parameter_child_identity.v1"
CHILD_CLASSIFICATION = "extension_or_invention"
ROOT = Path(__file__).resolve().parents[2]
SELECTED_T2_IDENTITY = "f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e"
SELECTED_T2_GPU_BOUND_PARENT_VALUE = -31.1290512231882
SELECTED_T2_CLAIM_PATH = Path(
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "attempt-13-selected-untouched-value-claim-xla-repair/result.json"
)
SELECTED_T2_CLAIM_SHA256 = (
    "289565b59455a59e31190a5240ef98cbd885cfe4213677ecde1f22c31e206244"
)


def _as_theta(theta: tf.Tensor) -> tf.Tensor:
    value = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
    tf.debugging.assert_all_finite(value, "theta must be finite")
    return value


def _parent_fields(
    parent: LaneBT1Artifact | LaneBT2Artifact,
) -> tuple[object, tuple[tf.Tensor, ...], tf.Tensor, tf.Tensor, object, str]:
    if isinstance(parent, LaneBT1Artifact):
        settings = parent.settings
        cores = tuple(tf.identity(core) for core in parent.cores)
        frame = parent.frame
        identity = parent.identity.hash.value
        shift = parent.shift_constant
    elif isinstance(parent, LaneBT2Artifact):
        settings = parent.settings
        cores = tuple(tf.identity(core) for core in parent.cores)
        frame = parent.frame
        identity = parent.identity.hash.value
        shift = parent.shift_constant
    else:
        raise TypeError("parameter child parent must be an admitted Lane-B T1/T2 artifact")
    return settings, cores, frame, tf.reshape(tf.convert_to_tensor(shift, DTYPE), []), lane_b_product_basis(
        order=settings.basis_order, num_elems=settings.basis_num_elems
    ), identity


def load_selected_t2_parameter_parent_compat(
    directory: Path,
    *,
    parent_artifact: LaneBT1Artifact,
) -> LaneBT2Artifact:
    """Decode only the admitted GPU-issued T2 parent on a CPU process.

    The v1 identity includes a recomputed parent value and is therefore one-ULP
    backend dependent. This decoder restores the GPU-bound scalar from the
    passed claim artifact while independently checking the CPU parent value.
    """

    claim_path = ROOT / SELECTED_T2_CLAIM_PATH
    if hashlib.sha256(claim_path.read_bytes()).hexdigest() != SELECTED_T2_CLAIM_SHA256:
        raise ValueError("selected T2 claim hash mismatch")
    claim = json.loads(claim_path.read_text())
    if (
        claim.get("status") != "PASS_NEW_FIXED_VARIANT_T1_T2_VALUE_BASELINE"
        or claim.get("artifact_identity") != SELECTED_T2_IDENTITY
        or float(claim.get("parent_t1_value")) != SELECTED_T2_GPU_BOUND_PARENT_VALUE
    ):
        raise ValueError("selected T2 claim payload mismatch")
    cpu_parent_value = float(parent_artifact.value().numpy())
    if abs(cpu_parent_value - SELECTED_T2_GPU_BOUND_PARENT_VALUE) > 4.0 * math.ulp(
        SELECTED_T2_GPU_BOUND_PARENT_VALUE
    ):
        raise ValueError("selected T2 CPU parent value exceeds compatibility tolerance")

    output = Path(directory)
    payload = json.loads((output / "manifest.json").read_text())
    if (
        payload.get("schema_version") != T2_ARTIFACT_SCHEMA
        or payload.get("identity_sha256") != SELECTED_T2_IDENTITY
        or payload.get("parent_t1_identity") != parent_artifact.identity.hash.value
        or payload.get("source_closure") != dict(t2_source_closure())
    ):
        raise ValueError("selected T2 parent manifest mismatch")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("selected T2 parent tensor ledger missing")

    def read_tensor(name: str) -> tf.Tensor:
        row = tensors.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"selected T2 parent tensor missing: {name}")
        serialized = tf.io.read_file((output / str(row["path"])).as_posix())
        if hashlib.sha256(bytes(serialized.numpy())).hexdigest() != row.get("sha256"):
            raise ValueError(f"selected T2 parent tensor hash mismatch: {name}")
        return tf.ensure_shape(
            tf.io.parse_tensor(serialized, out_type=tf.dtypes.as_dtype(str(row["dtype"]))),
            row["shape"],
        )

    settings = __import__(
        "bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf",
        fromlist=["LaneBT1Settings"],
    ).LaneBT1Settings(
        **{
            name: payload["settings"][name]
            for name in __import__(
                "bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf",
                fromlist=["LaneBT1Settings"],
            ).LaneBT1Settings.__dataclass_fields__
        }
    )
    frame = SourceRouteCoordinateFrame(
        mu=read_tensor("frame_mu"),
        matrix=read_tensor("frame_matrix"),
        expansion_factor=settings.expansion_factor,
    )
    cores = tuple(read_tensor(f"core_{axis:02d}") for axis in range(36))
    calibration = _t2_estimate_from_payload(payload["calibration_estimate"])
    validation = _t2_estimate_from_payload(payload["validation_estimate"])
    shift = tf.constant(float(payload["shift_constant"]), DTYPE)
    cpu_identity = issue_lane_b_t2_identity(
        parent_artifact=parent_artifact,
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift,
        calibration_estimate=calibration,
        validation_estimate=validation,
        training_cloud_manifest=payload["training_cloud_manifest"],
        validation_cloud_manifest=payload["validation_cloud_manifest"],
        source_hashes=payload["source_closure"],
    )
    identity_payload = dict(cpu_identity.manifest.payload)
    identity_payload["parent_t1_value"] = SELECTED_T2_GPU_BOUND_PARENT_VALUE
    manifest = BranchManifest(T2_IDENTITY_SCHEMA, identity_payload)
    identity = BranchIdentity(manifest=manifest, hash=BranchHash(SELECTED_T2_IDENTITY))
    # The canonical v1 dataclass reissues its identity in __post_init__, which
    # is intentionally backend-sensitive because it recomputes parent value.
    # Populate this narrowly scoped compatibility instance only after all
    # serialized tensors and the claim-bound identity have been verified.
    artifact = object.__new__(LaneBT2Artifact)
    object.__setattr__(artifact, "parent_artifact", parent_artifact)
    object.__setattr__(artifact, "settings", settings)
    object.__setattr__(artifact, "frame", frame)
    object.__setattr__(artifact, "cores", cores)
    object.__setattr__(artifact, "shift_constant", shift)
    object.__setattr__(artifact, "calibration_estimate", calibration)
    object.__setattr__(artifact, "validation_estimate", validation)
    object.__setattr__(artifact, "training_cloud_manifest", payload["training_cloud_manifest"])
    object.__setattr__(artifact, "validation_cloud_manifest", payload["validation_cloud_manifest"])
    object.__setattr__(artifact, "source_hashes", payload["source_closure"])
    object.__setattr__(artifact, "identity", identity)
    if abs(float(artifact.value().numpy()) - float(claim["artifact_cumulative_value"])) > 5e-13:
        raise ValueError("selected T2 CPU value does not match admitted claim")
    return artifact


def _linear_cores(
    parent_cores: Sequence[tf.Tensor], tangent_cores: Sequence[Sequence[tf.Tensor]], theta: tf.Tensor
) -> tuple[tf.Tensor, ...]:
    parameter = _as_theta(theta)
    if len(parent_cores) != len(tangent_cores):
        raise ValueError("tangent bank must match parent core count")
    result = []
    for axis, (parent, bank) in enumerate(zip(parent_cores, tangent_cores)):
        if len(bank) != PARAMETER_DIM:
            raise ValueError(f"tangent bank axis {axis} must have three coordinates")
        value = tf.identity(tf.convert_to_tensor(parent, DTYPE))
        for index, tangent in enumerate(bank):
            tangent_tensor = tf.convert_to_tensor(tangent, DTYPE)
            if tangent_tensor.shape != value.shape:
                raise ValueError(f"tangent shape mismatch at axis {axis}, parameter {index}")
            value = value + parameter[index] * tangent_tensor
        tf.debugging.assert_all_finite(value, f"conditioned core {axis}")
        result.append(value)
    return tuple(result)


def _density_from_cores(
    *,
    settings: object,
    cores: Sequence[tf.Tensor],
) -> SquaredTTDensity:
    basis = lane_b_product_basis(
        order=settings.basis_order, num_elems=settings.basis_num_elems
    )
    ftt = FunctionalTT(tuple(TTCore(core) for core in cores), basis, lane_b_measure_convention())
    defensive = TensorProductReferenceDensity(basis, lane_b_measure_convention())
    tau = tf.constant(settings.tau, DTYPE)
    normalizer_floor = tf.constant(1e-14, DTYPE)
    denominator_floor = tf.constant(1e-300, DTYPE)
    branch = SquaredTTDensity.expected_branch_identity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=normalizer_floor,
        denominator_floor=denominator_floor,
        measure_convention=lane_b_measure_convention(),
    )
    return SquaredTTDensity(
        sqrt_tt=ftt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=normalizer_floor,
        denominator_floor=denominator_floor,
        measure_convention=lane_b_measure_convention(),
        branch_identity=branch,
    )


def _paired_mass_product(
    *,
    basis: object,
    parent_cores: Sequence[tf.Tensor],
    tangent_cores: Sequence[Sequence[tf.Tensor]],
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    parameter = _as_theta(theta)
    value_vector = tf.ones([1], DTYPE)
    derivative_vector = tf.zeros([PARAMETER_DIM, 1], DTYPE)
    for axis, (parent, bank) in enumerate(zip(parent_cores, tangent_cores)):
        conditioned = parent
        tangents = tuple(tf.convert_to_tensor(item, DTYPE) for item in bank)
        for index in range(PARAMETER_DIM):
            conditioned = conditioned + parameter[index] * tangents[index]
        mass = basis.bases[axis].mass_matrix(lane_b_measure_convention().mass_measure)
        pair = tf.einsum("alb,AmB,lm->aAbB", conditioned, conditioned, mass)
        matrix = tf.reshape(pair, [int(conditioned.shape[0]) ** 2, int(conditioned.shape[2]) ** 2])
        next_value = tf.einsum("a,ab->b", value_vector, matrix)
        derivative_rows = []
        for index in range(PARAMETER_DIM):
            tangent = tangents[index]
            d_pair = tf.einsum("alb,AmB,lm->aAbB", tangent, conditioned, mass)
            d_pair = d_pair + tf.einsum("alb,AmB,lm->aAbB", conditioned, tangent, mass)
            d_matrix = tf.reshape(
                d_pair,
                [int(conditioned.shape[0]) ** 2, int(conditioned.shape[2]) ** 2],
            )
            derivative_rows.append(
                tf.einsum("a,ab->b", value_vector, d_matrix)
                + tf.einsum("a,ab->b", derivative_vector[index], matrix)
            )
        value_vector = next_value
        derivative_vector = tf.stack(derivative_rows, axis=0)
    return tf.reshape(value_vector, []), tf.reshape(derivative_vector, [PARAMETER_DIM])


def _evaluate_amplitude_and_tangents(
    *,
    basis: object,
    cores: Sequence[tf.Tensor],
    tangent_cores: Sequence[Sequence[tf.Tensor]],
    theta: tf.Tensor,
    points: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    values = tf.convert_to_tensor(points, DTYPE)
    parameter = _as_theta(theta)
    sample_count = tf.shape(values)[0]
    vector = tf.ones([sample_count, 1], DTYPE)
    derivative = tf.zeros([PARAMETER_DIM, sample_count, 1], DTYPE)
    for axis, (core, bank) in enumerate(zip(cores, tangent_cores)):
        basis_values = basis.evaluate_axis(axis, values[:, axis])
        conditioned = core
        for index in range(PARAMETER_DIM):
            conditioned = conditioned + parameter[index] * bank[index]
        matrices = tf.einsum("nl,alb->nab", basis_values, conditioned)
        next_vector = tf.einsum("na,nab->nb", vector, matrices)
        next_derivatives = []
        for index in range(PARAMETER_DIM):
            tangent_matrices = tf.einsum("nl,alb->nab", basis_values, bank[index])
            next_derivatives.append(
                tf.einsum("na,nab->nb", vector, tangent_matrices)
                + tf.einsum("na,nab->nb", derivative[index], matrices)
            )
        vector = next_vector
        derivative = tf.stack(next_derivatives, axis=0)
    return tf.reshape(vector, [sample_count]), tf.reshape(
        derivative, [PARAMETER_DIM, sample_count]
    )


@dataclass(frozen=True)
class LaneBParameterChild:
    parent: LaneBT1Artifact | LaneBT2Artifact
    tangent_cores: tuple[tuple[tf.Tensor, ...], ...]
    chart_id: str = "identity_theta_chart_v1"
    child_identity: BranchIdentity | None = None

    def __post_init__(self) -> None:
        _settings, parent_cores, frame, shift, basis, parent_identity = _parent_fields(self.parent)
        if len(self.tangent_cores) != len(parent_cores):
            raise ValueError("parameter child tangent bank must match parent core count")
        normalized = []
        for axis, bank in enumerate(self.tangent_cores):
            if len(bank) != PARAMETER_DIM:
                raise ValueError(f"parameter child tangent bank axis {axis} must have three entries")
            values = tuple(tf.convert_to_tensor(item, DTYPE) for item in bank)
            for index, value in enumerate(values):
                if value.shape != parent_cores[axis].shape:
                    raise ValueError(f"parameter child tangent shape mismatch at axis {axis}, parameter {index}")
                tf.debugging.assert_all_finite(value, "parameter child tangent")
            normalized.append(values)
        if self.chart_id != "identity_theta_chart_v1":
            raise ValueError("unknown parameter child chart")
        object.__setattr__(self, "tangent_cores", tuple(normalized))
        expected = issue_parameter_child_identity(
            parent=self.parent,
            tangent_cores=tuple(normalized),
            chart_id=self.chart_id,
        )
        if self.child_identity is not None and self.child_identity != expected:
            raise ValueError("parameter child identity mismatch")
        object.__setattr__(self, "child_identity", expected)

    @property
    def identity(self) -> BranchIdentity:
        return self.child_identity

    @property
    def parent_cores(self) -> tuple[tf.Tensor, ...]:
        return _parent_fields(self.parent)[1]

    @property
    def settings(self) -> object:
        return _parent_fields(self.parent)[0]

    def conditioned_cores(self, theta: tf.Tensor) -> tuple[tf.Tensor, ...]:
        return _linear_cores(self.parent_cores, self.tangent_cores, theta)

    def density(self, theta: tf.Tensor) -> SquaredTTDensity:
        return _density_from_cores(settings=self.settings, cores=self.conditioned_cores(theta))

    def log_normalizer_and_score(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, derivative = _paired_mass_product(
            basis=lane_b_product_basis(
                order=self.settings.basis_order, num_elems=self.settings.basis_num_elems
            ),
            parent_cores=self.parent_cores,
            tangent_cores=self.tangent_cores,
            theta=theta,
        )
        tau_mass = tf.constant(self.settings.tau, DTYPE)
        normalizer = value + tau_mass
        return tf.math.log(normalizer), derivative / normalizer

    def increment_and_score(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        log_normalizer, score = self.log_normalizer_and_score(theta)
        shift = _parent_fields(self.parent)[3]
        return log_normalizer - shift, score

    def unnormalized_log_density_and_score(
        self, theta: tf.Tensor, points: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return the full defensive-density log value and its manual score."""

        parameter = _as_theta(theta)
        values = tf.convert_to_tensor(points, DTYPE)
        if values.shape.rank != 2 or values.shape[1] != len(self.parent_cores):
            raise ValueError("parameter child points have wrong shape")
        base = self.density(parameter)
        h, dh = _evaluate_amplitude_and_tangents(
            basis=lane_b_product_basis(
                order=self.settings.basis_order,
                num_elems=self.settings.basis_num_elems,
            ),
            cores=self.parent_cores,
            tangent_cores=self.tangent_cores,
            theta=parameter,
            points=values,
        )
        unnormalized = base.unnormalized_density(values)
        score = (
            2.0
            * h[:, tf.newaxis]
            * tf.transpose(dh)
            / unnormalized[:, tf.newaxis]
        )
        return tf.math.log(unnormalized), score

    def point_log_density_and_score(
        self, theta: tf.Tensor, points: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        parameter = _as_theta(theta)
        values = tf.convert_to_tensor(points, DTYPE)
        if values.shape.rank != 2 or values.shape[1] != len(self.parent_cores):
            raise ValueError("parameter child points have wrong shape")
        log_unnormalized, point_score = self.unnormalized_log_density_and_score(
            parameter, values
        )
        log_normalizer, normalizer_score = self.log_normalizer_and_score(parameter)
        return (
            log_unnormalized - log_normalizer,
            point_score - normalizer_score[tf.newaxis, :],
        )

    def prefix_log_marginal_and_score(
        self,
        theta: tf.Tensor,
        local_prefix_points: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return a normalized source-style prefix marginal and total score."""

        parameter = _as_theta(theta)
        points = tf.convert_to_tensor(local_prefix_points, DTYPE)
        if points.shape.rank != 2 or points.shape[1] is None:
            raise ValueError("prefix points must have shape [sample,prefix_dim]")
        prefix_dim = int(points.shape[1])
        if prefix_dim <= 0 or prefix_dim > len(self.parent_cores):
            raise ValueError("prefix dimension is out of range")
        sample_count = tf.shape(points)[0]
        basis = lane_b_product_basis(
            order=self.settings.basis_order, num_elems=self.settings.basis_num_elems
        )
        value_vector = tf.ones([sample_count, 1], DTYPE)
        derivative_vector = tf.zeros([PARAMETER_DIM, sample_count, 1], DTYPE)
        for axis, (parent_core, bank) in enumerate(
            zip(self.parent_cores, self.tangent_cores)
        ):
            conditioned = parent_core
            for index in range(PARAMETER_DIM):
                conditioned = conditioned + parameter[index] * bank[index]
            if axis < prefix_dim:
                evaluated = basis.evaluate_axis(axis, points[:, axis])
                matrix = tf.einsum(
                    "nl,nm,alb,AmB->naAbB",
                    evaluated,
                    evaluated,
                    conditioned,
                    conditioned,
                )
                derivative_matrices = []
                for index in range(PARAMETER_DIM):
                    tangent = bank[index]
                    derivative_matrices.append(
                        tf.einsum(
                            "nl,nm,alb,AmB->naAbB",
                            evaluated,
                            evaluated,
                            tangent,
                            conditioned,
                        )
                        + tf.einsum(
                            "nl,nm,alb,AmB->naAbB",
                            evaluated,
                            evaluated,
                            conditioned,
                            tangent,
                        )
                    )
            else:
                mass = basis.bases[axis].mass_matrix(
                    lane_b_measure_convention().mass_measure
                )
                pair = tf.einsum(
                    "alb,AmB,lm->aAbB", conditioned, conditioned, mass
                )
                matrix = tf.broadcast_to(
                    pair[tf.newaxis, ...],
                    [sample_count, *pair.shape.as_list()],
                )
                derivative_matrices = []
                for index in range(PARAMETER_DIM):
                    tangent = bank[index]
                    derivative_pair = tf.einsum(
                        "alb,AmB,lm->aAbB", tangent, conditioned, mass
                    ) + tf.einsum(
                        "alb,AmB,lm->aAbB", conditioned, tangent, mass
                    )
                    derivative_matrices.append(
                        tf.broadcast_to(
                            derivative_pair[tf.newaxis, ...],
                            [sample_count, *derivative_pair.shape.as_list()],
                        )
                    )
            left_rank = int(conditioned.shape[0]) ** 2
            right_rank = int(conditioned.shape[2]) ** 2
            matrix = tf.reshape(matrix, [sample_count, left_rank, right_rank])
            next_value = tf.einsum("na,nab->nb", value_vector, matrix)
            next_derivatives = []
            for index in range(PARAMETER_DIM):
                derivative_matrix = tf.reshape(
                    derivative_matrices[index],
                    [sample_count, left_rank, right_rank],
                )
                next_derivatives.append(
                    tf.einsum("na,nab->nb", value_vector, derivative_matrix)
                    + tf.einsum("na,nab->nb", derivative_vector[index], matrix)
                )
            value_vector = next_value
            derivative_vector = tf.stack(next_derivatives, axis=0)
        square_numerator = tf.reshape(value_vector, [sample_count])
        square_derivative = tf.reshape(
            derivative_vector, [PARAMETER_DIM, sample_count]
        )
        numerator = square_numerator + tf.constant(self.settings.tau, DTYPE)
        log_normalizer, normalizer_score = self.log_normalizer_and_score(parameter)
        log_marginal = tf.math.log(numerator) - log_normalizer
        score = tf.transpose(square_derivative) / numerator[:, tf.newaxis]
        return log_marginal, score - normalizer_score[tf.newaxis, :]


def issue_parameter_child_identity(
    *,
    parent: LaneBT1Artifact | LaneBT2Artifact,
    tangent_cores: Sequence[Sequence[tf.Tensor]],
    chart_id: str,
) -> BranchIdentity:
    parent_identity = parent.identity.hash.value
    settings, _cores, frame, shift, _basis, _ = _parent_fields(parent)
    payload = {
        "schema": CHILD_IDENTITY_SCHEMA,
        "classification": CHILD_CLASSIFICATION,
        "parent_identity": parent_identity,
        "parent_core_sha256": tuple(tensor_sha256(core) for core in parent.cores),
        "tangent_core_sha256": tuple(
            tuple(tensor_sha256(core) for core in bank) for bank in tangent_cores
        ),
        "chart_id": chart_id,
        "parameter_order": (
            "log_kappa_scale",
            "log_nu_scale",
            "log_obs_noise_scale",
        ),
        "state_only_normalization": True,
        "theta_integration_forbidden": True,
        "settings": settings.manifest_payload(),
        "frame_mu_sha256": tensor_sha256(frame.mu),
        "frame_matrix_sha256": tensor_sha256(frame.matrix),
        "shift": float(shift.numpy()),
        "tau_frozen": float(settings.tau),
        "hmc_authorized": False,
    }
    manifest = BranchManifest(CHILD_IDENTITY_SCHEMA, payload)
    return BranchIdentity(manifest=manifest, hash=manifest.sha256())


__all__ = [
    "CHILD_CLASSIFICATION",
    "CHILD_IDENTITY_SCHEMA",
    "CHILD_SCHEMA",
    "LaneBParameterChild",
    "issue_parameter_child_identity",
    "load_selected_t2_parameter_parent_compat",
]
