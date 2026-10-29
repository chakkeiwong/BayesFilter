"""Centered external-parameter density mechanics for Austria Lane-B.

The admitted fixed TT is an immutable amplitude component. Parameter dependence
is supplied by separate residual TTs with features that vanish at theta zero.
This defines a full finite density family without integrating over theta or
materializing a block-sum TT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.fixed_branch import BranchIdentity, BranchManifest
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import tensor_sha256
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    LaneBT1Artifact,
    lane_b_measure_convention,
    lane_b_product_basis,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import LaneBT2Artifact


DTYPE = tf.float64
PARAMETER_DIM = 3
CENTERED_CHILD_SCHEMA = "bayesfilter.zhao_cui_austria_sir_centered_density.v1"
CENTERED_CHILD_IDENTITY_SCHEMA = (
    "bayesfilter.zhao_cui_austria_sir_centered_density_identity.v1"
)
CENTERED_CHILD_CLASSIFICATION = "extension_or_invention"
CENTERED_BASIS_EVALUATION_ID = "setup_static_cpu_nodes_barycentric_weights_v1"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-zhao-cui-austria-sir-parameter-conditioned-density-jvp-plan-"
    "2026-07-31.md"
)


def _as_theta(theta: tf.Tensor) -> tf.Tensor:
    value = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
    tf.debugging.assert_all_finite(value, "theta must be finite")
    return value


def _parent_fields(
    parent: LaneBT1Artifact | LaneBT2Artifact,
) -> tuple[object, tuple[tf.Tensor, ...], tf.Tensor, str]:
    if not isinstance(parent, (LaneBT1Artifact, LaneBT2Artifact)):
        raise TypeError("centered child parent must be an admitted Lane-B artifact")
    return (
        parent.settings,
        tuple(tf.identity(tf.convert_to_tensor(core, DTYPE)) for core in parent.cores),
        tf.reshape(tf.convert_to_tensor(parent.shift_constant, DTYPE), []),
        parent.identity.hash.value,
    )


@dataclass(frozen=True)
class _CenteredFrozenEvaluationBasis:
    """Freeze setup-static interpolation nodes before any GPU initialization."""

    delegate: object
    local_nodes: tf.Tensor
    barycentric_weights: tf.Tensor

    @property
    def basis_dim(self) -> int:
        return int(self.delegate.basis_dim)

    @property
    def dtype(self) -> tf.DType:
        return self.delegate.dtype

    @property
    def domain(self):
        return self.delegate.domain

    def evaluate(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, DTYPE)
        reference = self.domain.to_reference(values)
        flat = tf.reshape(reference, [-1])
        order = int(self.delegate.delegate.order)
        num_elems = int(self.delegate.delegate.num_elems)
        local_dim = order + 1
        elem_size = tf.constant(2.0 / float(num_elems), DTYPE)
        raw_indices = tf.cast(tf.math.ceil((flat + 1.0) / elem_size), tf.int32) - 1
        element_indices = tf.clip_by_value(raw_indices, 0, num_elems - 1)
        left_edges = -1.0 + tf.cast(element_indices, DTYPE) * elem_size
        local = (flat - left_edges) / elem_size

        differences = local[:, tf.newaxis] - self.local_nodes[tf.newaxis, :]
        exact = tf.abs(differences) <= tf.constant(1e-12, DTYPE)
        safe = tf.where(exact, tf.ones_like(differences), differences)
        terms = self.barycentric_weights[tf.newaxis, :] / safe
        local_values = terms / tf.reduce_sum(terms, axis=1, keepdims=True)
        local_values = tf.where(
            tf.reduce_any(exact, axis=1)[:, tf.newaxis],
            tf.cast(exact, DTYPE),
            local_values,
        )
        inside = tf.logical_and(flat >= -1.0, flat <= 1.0)
        local_values = tf.where(
            inside[:, tf.newaxis], local_values, tf.zeros_like(local_values)
        )

        rows = tf.repeat(tf.range(tf.shape(flat)[0], dtype=tf.int32), local_dim)
        local_columns = tf.range(local_dim, dtype=tf.int32)
        columns = (
            element_indices[:, tf.newaxis] * order
            + local_columns[tf.newaxis, :]
        )
        scattered = tf.scatter_nd(
            tf.stack([rows, tf.reshape(columns, [-1])], axis=1),
            tf.reshape(local_values, [-1]),
            [tf.shape(flat)[0], self.basis_dim],
        )
        return tf.reshape(
            scattered, tf.concat([tf.shape(reference), [self.basis_dim]], axis=0)
        )

    def mass_matrix(self, measure: MassMeasure) -> tf.Tensor:
        return self.delegate.mass_matrix(measure)

    def integral_vector(self, measure: MassMeasure) -> tf.Tensor:
        return self.delegate.integral_vector(measure)

    def manifest_payload(self) -> Mapping[str, object]:
        payload = dict(self.delegate.manifest_payload())
        payload.update(
            {
                "centered_evaluation_id": CENTERED_BASIS_EVALUATION_ID,
                "local_nodes_sha256": tensor_sha256(self.local_nodes),
                "barycentric_weights_sha256": tensor_sha256(
                    self.barycentric_weights
                ),
            }
        )
        return payload


def _freeze_centered_basis_evaluation(basis: object) -> _CenteredFrozenEvaluationBasis:
    delegate = basis.delegate
    order = int(delegate.order)
    num_elems = int(delegate.num_elems)
    elem_size = tf.constant(2.0 / float(num_elems), DTYPE)
    with tf.device("/CPU:0"):
        global_nodes = tf.identity(delegate.reference_nodes)
        local_nodes = (global_nodes[: order + 1] + 1.0) / elem_size
        differences = local_nodes[:, tf.newaxis] - local_nodes[tf.newaxis, :]
        safe = tf.where(
            tf.eye(order + 1, dtype=DTYPE) > 0.0,
            tf.ones_like(differences),
            differences,
        )
        barycentric_weights = tf.math.reciprocal(tf.reduce_prod(safe, axis=1))
    return _CenteredFrozenEvaluationBasis(
        delegate=basis,
        local_nodes=tf.identity(local_nodes),
        barycentric_weights=tf.identity(barycentric_weights),
    )


def centered_lane_b_product_basis(*, order: int, num_elems: int) -> ProductBasis:
    base = lane_b_product_basis(order=int(order), num_elems=int(num_elems))
    return ProductBasis(
        tuple(_freeze_centered_basis_evaluation(basis) for basis in base.bases),
        base.convention,
    )


@dataclass(frozen=True)
class CenteredThetaFeatures:
    """A small explicit polynomial feature map with every feature zero at zero."""

    feature_ids: tuple[str, ...] = ("linear_0", "linear_1", "linear_2")

    def __post_init__(self) -> None:
        identifiers = tuple(str(item) for item in self.feature_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("centered feature ids must be nonempty and unique")
        for identifier in identifiers:
            _parse_feature_id(identifier)
        object.__setattr__(self, "feature_ids", identifiers)

    @property
    def feature_count(self) -> int:
        return len(self.feature_ids)

    def values_and_jacobian(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        parameter = _as_theta(theta)
        values, jacobian = self.batch_values_and_jacobian(parameter[tf.newaxis, :])
        return values[0], jacobian[0]

    def batch_values_and_jacobian(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        parameter = tf.convert_to_tensor(theta, DTYPE)
        if parameter.shape.rank != 2 or parameter.shape[1] != PARAMETER_DIM:
            raise ValueError("batched theta must have shape [batch,3]")
        values = []
        rows = []
        for identifier in self.feature_ids:
            family, left, right = _parse_feature_id(identifier)
            if family == "linear":
                values.append(parameter[:, left])
                rows.append(
                    tf.broadcast_to(
                        tf.one_hot(left, PARAMETER_DIM, dtype=DTYPE)[tf.newaxis, :],
                        [tf.shape(parameter)[0], PARAMETER_DIM],
                    )
                )
            elif family == "quadratic":
                values.append(tf.square(parameter[:, left]))
                rows.append(
                    2.0
                    * parameter[:, left, tf.newaxis]
                    * tf.one_hot(left, PARAMETER_DIM, dtype=DTYPE)[tf.newaxis, :]
                )
            else:
                values.append(parameter[:, left] * parameter[:, right])
                rows.append(
                    parameter[:, right, tf.newaxis]
                    * tf.one_hot(left, PARAMETER_DIM, dtype=DTYPE)[tf.newaxis, :]
                    + parameter[:, left, tf.newaxis]
                    * tf.one_hot(right, PARAMETER_DIM, dtype=DTYPE)[tf.newaxis, :]
                )
        feature_values = tf.stack(values, axis=1)
        jacobian = tf.stack(rows, axis=1)
        tf.debugging.assert_all_finite(feature_values, "centered feature values")
        tf.debugging.assert_all_finite(jacobian, "centered feature Jacobian")
        return feature_values, jacobian

    def augmented_values_and_jacobian(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        values, jacobian = self.values_and_jacobian(theta)
        return (
            tf.concat([tf.ones([1], DTYPE), values], axis=0),
            tf.concat([tf.zeros([1, PARAMETER_DIM], DTYPE), jacobian], axis=0),
        )


def _parse_feature_id(identifier: str) -> tuple[str, int, int]:
    parts = identifier.split("_")
    if len(parts) == 2 and parts[0] in ("linear", "quadratic"):
        index = int(parts[1])
        if index < 0 or index >= PARAMETER_DIM:
            raise ValueError(f"feature coordinate out of range: {identifier}")
        return parts[0], index, index
    if len(parts) == 3 and parts[0] == "interaction":
        left = int(parts[1])
        right = int(parts[2])
        if left < 0 or right >= PARAMETER_DIM or left >= right:
            raise ValueError(f"invalid interaction feature: {identifier}")
        return parts[0], left, right
    raise ValueError(f"unsupported centered feature: {identifier}")


def _validate_component(
    component: Sequence[tf.Tensor],
    *,
    basis_dims: Sequence[int],
    label: str,
) -> tuple[tf.Tensor, ...]:
    cores = tuple(tf.convert_to_tensor(core, DTYPE) for core in component)
    if len(cores) != len(basis_dims):
        raise ValueError(f"{label} must contain one core per state axis")
    for axis, (core, basis_dim) in enumerate(zip(cores, basis_dims)):
        if core.shape.rank != 3 or int(core.shape[1]) != int(basis_dim):
            raise ValueError(f"{label} core {axis} has an invalid shape")
        if axis == 0 and int(core.shape[0]) != 1:
            raise ValueError(f"{label} first core must have left rank one")
        if axis == len(cores) - 1 and int(core.shape[2]) != 1:
            raise ValueError(f"{label} last core must have right rank one")
        if axis and int(cores[axis - 1].shape[2]) != int(core.shape[0]):
            raise ValueError(f"{label} rank mismatch before axis {axis}")
        tf.debugging.assert_all_finite(core, f"{label} core {axis}")
    return cores


def _evaluate_component(
    component: Sequence[tf.Tensor], basis: object, points: tf.Tensor
) -> tf.Tensor:
    values = tf.convert_to_tensor(points, DTYPE)
    sample_count = tf.shape(values)[0]
    vector = tf.ones([sample_count, 1], DTYPE)
    for axis, core in enumerate(component):
        basis_values = basis.evaluate_axis(axis, values[:, axis])
        matrices = tf.einsum("nl,alb->nab", basis_values, core)
        vector = tf.einsum("na,nab->nb", vector, matrices)
    return tf.reshape(vector, [sample_count])


def _cross_mass(
    left: Sequence[tf.Tensor], right: Sequence[tf.Tensor], basis: object
) -> tf.Tensor:
    vector = tf.ones([1], DTYPE)
    active_measure = lane_b_measure_convention().mass_measure
    for axis, (left_core, right_core) in enumerate(zip(left, right)):
        mass = basis.bases[axis].mass_matrix(active_measure)
        paired = tf.einsum("alb,AmB,lm->aAbB", left_core, right_core, mass)
        matrix = tf.reshape(
            paired,
            [
                int(left_core.shape[0]) * int(right_core.shape[0]),
                int(left_core.shape[2]) * int(right_core.shape[2]),
            ],
        )
        vector = tf.einsum("a,ab->b", vector, matrix)
    return tf.reshape(vector, [])


def _cross_prefix_values(
    left: Sequence[tf.Tensor],
    right: Sequence[tf.Tensor],
    basis: object,
    points: tf.Tensor,
) -> tf.Tensor:
    values = tf.convert_to_tensor(points, DTYPE)
    prefix_dim = int(values.shape[1])
    sample_count = tf.shape(values)[0]
    vector = tf.ones([sample_count, 1], DTYPE)
    active_measure = lane_b_measure_convention().mass_measure
    for axis, (left_core, right_core) in enumerate(zip(left, right)):
        if axis < prefix_dim:
            evaluated = basis.evaluate_axis(axis, values[:, axis])
            paired = tf.einsum(
                "nl,nm,alb,AmB->naAbB",
                evaluated,
                evaluated,
                left_core,
                right_core,
            )
        else:
            mass = basis.bases[axis].mass_matrix(active_measure)
            static_pair = tf.einsum(
                "alb,AmB,lm->aAbB", left_core, right_core, mass
            )
            paired = tf.broadcast_to(
                static_pair[tf.newaxis, ...],
                tf.concat(
                    [tf.reshape(sample_count, [1]), tf.shape(static_pair)], axis=0
                ),
            )
        matrix = tf.reshape(
            paired,
            [
                sample_count,
                int(left_core.shape[0]) * int(right_core.shape[0]),
                int(left_core.shape[2]) * int(right_core.shape[2]),
            ],
        )
        vector = tf.einsum("na,nab->nb", vector, matrix)
    return tf.reshape(vector, [sample_count])


@dataclass(frozen=True)
class CenteredDensityMemoryEstimate:
    stored_elements: int
    stored_bytes: int
    point_workspace_elements: int
    pair_workspace_elements: int
    prefix_pair_workspace_elements: int
    component_count: int
    batch_size: int


@dataclass(frozen=True)
class LaneBCenteredResidualChild:
    """A parent-preserving parameter-conditioned squared-TT density family."""

    parent: LaneBT1Artifact | LaneBT2Artifact
    residual_components: tuple[tuple[tf.Tensor, ...], ...]
    features: CenteredThetaFeatures = field(default_factory=CenteredThetaFeatures)
    child_identity: BranchIdentity | None = None
    _components: tuple[tuple[tf.Tensor, ...], ...] = field(init=False, repr=False)
    _gram: tf.Tensor = field(init=False, repr=False)
    _basis: object = field(init=False, repr=False)
    _settings: object = field(init=False, repr=False)
    _shift: tf.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        settings, parent_cores, shift, _parent_identity = _parent_fields(self.parent)
        basis = centered_lane_b_product_basis(
            order=settings.basis_order, num_elems=settings.basis_num_elems
        )
        if len(self.residual_components) != self.features.feature_count:
            raise ValueError("one residual TT is required for each centered feature")
        parent_component = _validate_component(
            parent_cores,
            basis_dims=basis.basis_dim_tuple(),
            label="parent",
        )
        residuals = tuple(
            _validate_component(
                component,
                basis_dims=basis.basis_dim_tuple(),
                label=f"residual {index}",
            )
            for index, component in enumerate(self.residual_components)
        )
        components = (parent_component,) + residuals
        gram_rows = []
        for left in components:
            gram_rows.append(
                tf.stack([_cross_mass(left, right, basis) for right in components])
            )
        gram = tf.stack(gram_rows)
        gram = 0.5 * (gram + tf.transpose(gram))
        tf.debugging.assert_all_finite(gram, "centered component Gram matrix")
        object.__setattr__(self, "residual_components", residuals)
        object.__setattr__(self, "_components", components)
        object.__setattr__(self, "_gram", gram)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_shift", shift)
        expected = issue_centered_child_identity(
            parent=self.parent,
            residual_components=residuals,
            features=self.features,
        )
        if self.child_identity is not None and self.child_identity != expected:
            raise ValueError("centered child identity mismatch")
        object.__setattr__(self, "child_identity", expected)

    @property
    def identity(self) -> BranchIdentity:
        assert self.child_identity is not None
        return self.child_identity

    @property
    def settings(self) -> object:
        return self._settings

    @property
    def components(self) -> tuple[tuple[tf.Tensor, ...], ...]:
        return self._components

    @property
    def gram(self) -> tf.Tensor:
        return self._gram

    def component_values(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, DTYPE)
        if values.shape.rank != 2 or values.shape[1] != len(self.components[0]):
            raise ValueError("centered child points have the wrong shape")
        return tf.stack(
            [
                _evaluate_component(component, self._basis, values)
                for component in self.components
            ],
            axis=1,
        )

    def amplitude_and_jacobian(
        self, theta: tf.Tensor, points: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        weights, feature_jacobian = self.features.augmented_values_and_jacobian(theta)
        component_values = self.component_values(points)
        amplitude = tf.einsum("nc,c->n", component_values, weights)
        derivative = tf.einsum("nc,cp->np", component_values, feature_jacobian)
        return amplitude, derivative

    def log_normalizer_and_score(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        weights, feature_jacobian = self.features.augmented_values_and_jacobian(theta)
        gram_times_weights = tf.linalg.matvec(self.gram, weights)
        square_mass = tf.tensordot(weights, gram_times_weights, axes=1)
        normalizer = square_mass + tf.constant(self.settings.tau, DTYPE)
        derivative = 2.0 * tf.linalg.matvec(
            feature_jacobian, gram_times_weights, transpose_a=True
        )
        tf.debugging.assert_positive(normalizer, "centered child normalizer")
        tf.debugging.assert_all_finite(derivative, "centered child normalizer score")
        return tf.math.log(normalizer), derivative / normalizer

    def increment_and_score(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        log_normalizer, score = self.log_normalizer_and_score(theta)
        return log_normalizer - self._shift, score

    def unnormalized_log_density_and_score(
        self, theta: tf.Tensor, points: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        amplitude, derivative = self.amplitude_and_jacobian(theta, points)
        # Lane-B's defensive reference density is one under its probability measure.
        rho = tf.square(amplitude) + tf.constant(self.settings.tau, DTYPE)
        score = 2.0 * amplitude[:, tf.newaxis] * derivative / rho[:, tf.newaxis]
        tf.debugging.assert_positive(rho, "centered child density")
        return tf.math.log(rho), score

    def point_log_density_and_score(
        self, theta: tf.Tensor, points: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        log_rho, rho_score = self.unnormalized_log_density_and_score(theta, points)
        log_normalizer, normalizer_score = self.log_normalizer_and_score(theta)
        return log_rho - log_normalizer, rho_score - normalizer_score[tf.newaxis, :]

    def prefix_log_marginal_and_score(
        self, theta: tf.Tensor, local_prefix_points: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        points = tf.convert_to_tensor(local_prefix_points, DTYPE)
        if points.shape.rank != 2 or points.shape[1] is None:
            raise ValueError("prefix points must have shape [sample,prefix_dim]")
        prefix_dim = int(points.shape[1])
        if prefix_dim <= 0 or prefix_dim > len(self.components[0]):
            raise ValueError("prefix dimension is out of range")
        cross_rows = []
        for left in self.components:
            cross_rows.append(
                tf.stack(
                    [
                        _cross_prefix_values(left, right, self._basis, points)
                        for right in self.components
                    ],
                    axis=1,
                )
            )
        cross = tf.stack(cross_rows, axis=1)
        cross = 0.5 * (cross + tf.transpose(cross, [0, 2, 1]))
        weights, feature_jacobian = self.features.augmented_values_and_jacobian(theta)
        cross_times_weights = tf.einsum("nij,j->ni", cross, weights)
        square_numerator = tf.einsum("i,ni->n", weights, cross_times_weights)
        numerator = square_numerator + tf.constant(self.settings.tau, DTYPE)
        numerator_derivative = 2.0 * tf.einsum(
            "ip,ni->np", feature_jacobian, cross_times_weights
        )
        log_normalizer, normalizer_score = self.log_normalizer_and_score(theta)
        tf.debugging.assert_positive(numerator, "centered child prefix numerator")
        return (
            tf.math.log(numerator) - log_normalizer,
            numerator_derivative / numerator[:, tf.newaxis]
            - normalizer_score[tf.newaxis, :],
        )

    def memory_estimate(self, *, batch_size: int) -> CenteredDensityMemoryEstimate:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        stored = sum(
            int(tf.size(core)) for component in self.components for core in component
        )
        max_rank = max(
            max(int(core.shape[0]), int(core.shape[2]))
            for component in self.components
            for core in component
        )
        max_basis = max(int(core.shape[1]) for core in self.components[0])
        point_workspace = int(batch_size) * (max_rank * max_rank + max_basis)
        pair_workspace = max_rank**4
        return CenteredDensityMemoryEstimate(
            stored_elements=stored,
            stored_bytes=stored * DTYPE.size,
            point_workspace_elements=point_workspace,
            pair_workspace_elements=pair_workspace,
            prefix_pair_workspace_elements=int(batch_size) * pair_workspace,
            component_count=len(self.components),
            batch_size=int(batch_size),
        )

    def save(self, directory: Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=False)
        tensors: dict[str, Mapping[str, object]] = {}
        for component_index, component in enumerate(self.residual_components):
            for axis, core in enumerate(component):
                name = f"residual_{component_index:02d}_core_{axis:02d}"
                relative = f"{name}.tensor"
                serialized = tf.io.serialize_tensor(core)
                tf.io.write_file((output / relative).as_posix(), serialized)
                tensors[name] = {
                    "path": relative,
                    "shape": core.shape.as_list(),
                    "dtype": core.dtype.name,
                    "sha256": tensor_sha256(core),
                }
        payload = {
            "schema_version": CENTERED_CHILD_SCHEMA,
            "identity_sha256": self.identity.hash.value,
            "identity_manifest": dict(self.identity.manifest.payload),
            "parent_identity": _parent_fields(self.parent)[3],
            "feature_ids": list(self.features.feature_ids),
            "tensors": tensors,
        }
        (output / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )


def issue_centered_child_identity(
    *,
    parent: LaneBT1Artifact | LaneBT2Artifact,
    residual_components: Sequence[Sequence[tf.Tensor]],
    features: CenteredThetaFeatures,
) -> BranchIdentity:
    settings, parent_cores, _shift, parent_identity = _parent_fields(parent)
    frozen_basis = centered_lane_b_product_basis(
        order=settings.basis_order,
        num_elems=settings.basis_num_elems,
    ).bases[0]
    payload = {
        "schema": CENTERED_CHILD_IDENTITY_SCHEMA,
        "classification": CENTERED_CHILD_CLASSIFICATION,
        "plan_path": PLAN_PATH,
        "parent_identity": parent_identity,
        "parent_core_sha256": tuple(tensor_sha256(core) for core in parent_cores),
        "feature_ids": features.feature_ids,
        "feature_centering": "psi_k_zero_at_theta_zero",
        "amplitude_definition": "parent_plus_sum_psi_k_times_residual_tt",
        "density_definition": "amplitude_squared_plus_parent_tau_reference_density",
        "normalizer_definition": "cross_component_gram_contraction_plus_parent_tau",
        "shift_policy": "immutable_parent_shift",
        "theta_integration_forbidden": True,
        "residual_core_sha256": tuple(
            tuple(tensor_sha256(tf.convert_to_tensor(core, DTYPE)) for core in component)
            for component in residual_components
        ),
        "basis_order": int(settings.basis_order),
        "basis_num_elems": int(settings.basis_num_elems),
        "basis_evaluation_id": CENTERED_BASIS_EVALUATION_ID,
        "basis_local_nodes_sha256": tensor_sha256(
            frozen_basis.local_nodes
        ),
        "basis_barycentric_weights_sha256": tensor_sha256(
            frozen_basis.barycentric_weights
        ),
        "basis_constants_replicated_across_axes": True,
        "tau": float(settings.tau),
    }
    manifest = BranchManifest(CENTERED_CHILD_IDENTITY_SCHEMA, payload)
    return BranchIdentity(manifest=manifest, hash=manifest.sha256())


def load_centered_residual_child(
    directory: Path,
    *,
    parent: LaneBT1Artifact | LaneBT2Artifact,
) -> LaneBCenteredResidualChild:
    source = Path(directory)
    payload = json.loads((source / "manifest.json").read_text(encoding="ascii"))
    if payload.get("schema_version") != CENTERED_CHILD_SCHEMA:
        raise ValueError("centered child schema mismatch")
    if payload.get("parent_identity") != _parent_fields(parent)[3]:
        raise ValueError("centered child parent identity mismatch")
    features = CenteredThetaFeatures(tuple(payload.get("feature_ids", ())))
    tensor_rows = payload.get("tensors")
    if not isinstance(tensor_rows, Mapping):
        raise ValueError("centered child tensor ledger missing")
    residuals = []
    axis_count = len(_parent_fields(parent)[1])
    for component_index in range(features.feature_count):
        component = []
        for axis in range(axis_count):
            name = f"residual_{component_index:02d}_core_{axis:02d}"
            row = tensor_rows.get(name)
            if not isinstance(row, Mapping):
                raise ValueError(f"centered child tensor missing: {name}")
            serialized = tf.io.read_file((source / str(row["path"])).as_posix())
            core = tf.ensure_shape(
                tf.io.parse_tensor(serialized, out_type=DTYPE), row["shape"]
            )
            if tensor_sha256(core) != row.get("sha256"):
                raise ValueError(f"centered child tensor hash mismatch: {name}")
            component.append(core)
        residuals.append(tuple(component))
    child = LaneBCenteredResidualChild(
        parent=parent,
        residual_components=tuple(residuals),
        features=features,
    )
    if child.identity.hash.value != payload.get("identity_sha256"):
        raise ValueError("centered child identity mismatch")
    expected_manifest = json.loads(
        json.dumps(dict(child.identity.manifest.payload), sort_keys=True)
    )
    if payload.get("identity_manifest") != expected_manifest:
        raise ValueError("centered child identity manifest mismatch")
    return child


__all__ = [
    "CENTERED_CHILD_CLASSIFICATION",
    "CENTERED_BASIS_EVALUATION_ID",
    "CenteredDensityMemoryEstimate",
    "CenteredThetaFeatures",
    "LaneBCenteredResidualChild",
    "centered_lane_b_product_basis",
    "issue_centered_child_identity",
    "load_centered_residual_child",
]
