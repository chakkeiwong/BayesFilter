"""Squared functional TT density contracts for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import tensorflow as tf

from bayesfilter.highdim.diagnostics import (
    HighDimStatus,
    MassMeasure,
    MeasureConvention,
    assert_density_matches_mass,
    assert_tf_float64,
    freeze_mapping,
)
from bayesfilter.highdim.fixed_branch import BranchIdentity, BranchManifest
from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.tt import FunctionalTT, TTContractedRepresentation


class DefensiveDensityProtocol(Protocol):
    """Analytic nonnegative defensive density used in squared TT mixtures."""

    def log_density(self, points: tf.Tensor) -> tf.Tensor:
        """Return log defensive density at points with shape ``[N,D]``."""

    def normalizer(self, measure: MassMeasure) -> tf.Tensor:
        """Return the defensive mass under the declared measure."""

    def manifest_payload(self) -> Mapping[str, object]:
        """Return deterministic manifest payload fields."""


@dataclass(frozen=True)
class TensorProductReferenceDensity:
    """Constant tensor-product reference density on the product domain."""

    product_basis: object
    measure_convention: MeasureConvention
    floor: tf.Tensor = tf.constant(0.0, dtype=tf.float64)

    def __post_init__(self) -> None:
        assert_density_matches_mass(self.measure_convention)
        floor = tf.convert_to_tensor(self.floor, dtype=tf.float64)
        if floor.shape.rank != 0:
            raise ValueError(f"floor: {HighDimStatus.INVALID_SHAPE.value}")
        object.__setattr__(self, "floor", floor)

    def log_density(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        assert_tf_float64("points", values)
        if values.shape.rank != 2 or values.shape[1] != self.product_basis.dimension:
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        density = tf.ones([tf.shape(values)[0]], dtype=tf.float64) + self.floor
        return tf.math.log(density)

    def normalizer(self, measure: MassMeasure) -> tf.Tensor:
        if measure is MassMeasure.REFERENCE_MEASURE:
            base = tf.constant(1.0, dtype=tf.float64)
        elif measure is MassMeasure.REFERENCE_LEBESGUE:
            base = tf.constant(1.0, dtype=tf.float64)
            for basis in self.product_basis.bases:
                base = base * basis.domain.length
        else:
            raise TypeError("measure must be a MassMeasure")
        return base * (1.0 + self.floor)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "TensorProductReferenceDensity",
            "floor": self.floor,
            "measure_convention": {
                "density_measure": self.measure_convention.density_measure.value,
                "mass_measure": self.measure_convention.mass_measure.value,
                "reference_weight_name": self.measure_convention.reference_weight_name,
            },
        }


@dataclass(frozen=True)
class SquaredTTMarginal:
    """Marginal density metadata produced by a squared TT density."""

    keep_axes: tuple[int, ...]
    contracted_density: "SquaredTTDensity | TTContractedRepresentation"
    normalizer: tf.Tensor
    measure_convention: MeasureConvention
    branch_identity: BranchIdentity
    diagnostics: Mapping[str, object] | None = None
    density: "SquaredTTDensity | None" = None

    def __post_init__(self) -> None:
        normalizer = tf.convert_to_tensor(self.normalizer, dtype=tf.float64)
        if normalizer.shape.rank != 0:
            raise ValueError(f"normalizer: {HighDimStatus.INVALID_SHAPE.value}")
        if not isinstance(self.branch_identity, BranchIdentity):
            raise TypeError("branch_identity must be a BranchIdentity")
        object.__setattr__(self, "normalizer", normalizer)
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))
        if self.density is not None and not isinstance(self.density, SquaredTTDensity):
            raise TypeError("density must be a SquaredTTDensity")

    def normalized_retained_density_values(self, points: tf.Tensor) -> tf.Tensor:
        if self.density is None:
            raise NotImplementedError("marginal density evaluator is unavailable")
        return self.density.normalized_marginal_density_values(self.keep_axes, points)


@dataclass(frozen=True)
class SquaredTTDensity:
    """Nonnegative density defined by ``h(z)^2 + tau q_0(z)``."""

    sqrt_tt: FunctionalTT
    defensive_density: DefensiveDensityProtocol
    tau: tf.Tensor
    normalizer_floor: tf.Tensor
    denominator_floor: tf.Tensor
    measure_convention: MeasureConvention
    branch_identity: BranchIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.sqrt_tt, FunctionalTT):
            raise TypeError("sqrt_tt must be a FunctionalTT")
        if self.measure_convention != self.sqrt_tt.measure_convention:
            raise ValueError(f"SquaredTTDensity: {HighDimStatus.MEASURE_MISMATCH.value}")
        assert_density_matches_mass(self.measure_convention)
        for name in ("tau", "normalizer_floor", "denominator_floor"):
            value = tf.convert_to_tensor(getattr(self, name), dtype=tf.float64)
            if value.shape.rank != 0:
                raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
            if not bool(tf.math.is_finite(value).numpy()):
                raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
            if bool((value < 0.0).numpy()):
                raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
            object.__setattr__(self, name, value)
        expected_manifest = self.expected_manifest(
            sqrt_tt=self.sqrt_tt,
            defensive_density=self.defensive_density,
            tau=self.tau,
            normalizer_floor=self.normalizer_floor,
            denominator_floor=self.denominator_floor,
            measure_convention=self.measure_convention,
        )
        expected_identity = BranchIdentity(
            manifest=expected_manifest,
            hash=expected_manifest.sha256(),
        )
        if self.branch_identity != expected_identity:
            raise ValueError(HighDimStatus.INVALID_BRANCH_MISMATCH.value)

    def unnormalized_density(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        h_value = self.sqrt_tt.evaluate(values)
        q0 = tf.exp(self.defensive_density.log_density(values))
        result = tf.square(h_value) + self.tau * q0
        if not bool(tf.reduce_all(tf.math.is_finite(result)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        return result

    def sqrt_square_normalizer(self) -> tf.Tensor:
        vector = tf.ones([1], dtype=tf.float64)
        active_measure = self.measure_convention.mass_measure
        for axis, core in enumerate(self.sqrt_tt.cores):
            mass = self.sqrt_tt.product_basis.bases[axis].mass_matrix(active_measure)
            paired = tf.einsum("alb,A mB,lm->aAbB", core.values, core.values, mass)
            matrix = tf.reshape(
                paired,
                [core.left_rank * core.left_rank, core.right_rank * core.right_rank],
            )
            vector = tf.einsum("a,ab->b", vector, matrix)
        return tf.reshape(vector, [])

    def normalizer(self) -> tf.Tensor:
        z_h = self.sqrt_square_normalizer()
        z_0 = self.defensive_density.normalizer(self.measure_convention.mass_measure)
        z = z_h + self.tau * z_0
        if not bool(tf.math.is_finite(z).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        if bool((z <= self.normalizer_floor).numpy()):
            raise ValueError(HighDimStatus.NORMALIZER_FLOOR_EXCEEDED.value)
        return z

    def log_density(self, points: tf.Tensor) -> tf.Tensor:
        z = self.normalizer()
        q_u = self.unnormalized_density(points)
        return tf.math.log(q_u) - tf.math.log(z)

    def normalized_retained_density_values(self, keep_axes: Sequence[int], points: tf.Tensor) -> tf.Tensor:
        """Evaluate the normalized retained density for the all-retained case.

        This deliberately narrow helper supports the M2.6b scalar SV gate.  It
        is not a generic replacement for TT marginalization: if any coordinate
        is integrated out, callers must use a separately validated marginal
        contraction path.
        """

        axes = tuple(sorted(set(int(axis) for axis in keep_axes)))
        dimension = len(self.sqrt_tt.cores)
        if axes != tuple(range(dimension)):
            raise NotImplementedError("normalized retained density values currently require all axes retained")
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank != 2 or values.shape[1] != dimension:
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        return tf.exp(self.log_density(values))

    def marginal_density(self, keep_axes: Sequence[int]) -> SquaredTTMarginal:
        axes = tuple(sorted(set(int(axis) for axis in keep_axes)))
        dimension = len(self.sqrt_tt.cores)
        if any(axis < 0 or axis >= dimension for axis in axes):
            raise IndexError("keep_axes contains an out-of-range axis")
        _validate_source_marginal_axes(axes, dimension)
        integrated = tuple(axis for axis in range(dimension) if axis not in axes)
        contracted = self.sqrt_tt.contract_axes(integrated)
        diagnostics = {
            "status": HighDimStatus.OK.value,
            "keep_axes": axes,
            "integrated_axes": integrated,
            "normalizer": self.normalizer(),
            "semantics": "source_style_squared_tt_marginal",
            "source_anchor": "@TTSIRT/marginalise.m mass-matrix recursion",
            "note": "normalized_retained_density_values uses paired-core mass contractions; grid integration is diagnostic only",
        }
        return SquaredTTMarginal(
            keep_axes=axes,
            contracted_density=contracted,
            normalizer=self.normalizer(),
            measure_convention=self.measure_convention,
            branch_identity=contracted.branch_identity,
            diagnostics=diagnostics,
            density=self,
        )

    def normalized_marginal_density_values(
        self,
        keep_axes: Sequence[int],
        points: tf.Tensor,
    ) -> tf.Tensor:
        axes = tuple(sorted(set(int(axis) for axis in keep_axes)))
        dimension = len(self.sqrt_tt.cores)
        if any(axis < 0 or axis >= dimension for axis in axes):
            raise IndexError("keep_axes contains an out-of-range axis")
        _validate_source_marginal_axes(axes, dimension)
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank == 1 and len(axes) == 1:
            values = values[:, tf.newaxis]
        if values.shape.rank != 2 or values.shape[1] != len(axes):
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        numerator = self._source_style_marginal_unnormalized_values(axes, values)
        normalized = numerator / self.normalizer()
        if not bool(tf.reduce_all(tf.math.is_finite(normalized)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        return normalized

    def _source_style_marginal_unnormalized_values(
        self,
        keep_axes: tuple[int, ...],
        points: tf.Tensor,
    ) -> tf.Tensor:
        active_measure = self.measure_convention.mass_measure
        n_points = tf.shape(points)[0]
        vector = tf.ones([n_points, 1], dtype=tf.float64)
        point_axis = {axis: index for index, axis in enumerate(keep_axes)}
        for axis, core in enumerate(self.sqrt_tt.cores):
            if axis in point_axis:
                basis_values = self.sqrt_tt.product_basis.evaluate_axis(
                    axis,
                    points[:, point_axis[axis]],
                )
                matrices = tf.einsum(
                    "nl,nm,alb,AmB->naAbB",
                    basis_values,
                    basis_values,
                    core.values,
                    core.values,
                )
            else:
                paired = tf.einsum(
                    "alb,A mB,lm->aAbB",
                    core.values,
                    core.values,
                    self.sqrt_tt.product_basis.bases[axis].mass_matrix(active_measure),
                )
                matrices = tf.broadcast_to(
                    paired[tf.newaxis, :, :, :, :],
                    [
                        n_points,
                        core.left_rank,
                        core.left_rank,
                        core.right_rank,
                        core.right_rank,
                    ],
                )
            matrices = tf.reshape(
                matrices,
                [
                    n_points,
                    core.left_rank * core.left_rank,
                    core.right_rank * core.right_rank,
                ],
            )
            vector = tf.einsum("na,nab->nb", vector, matrices)
        sqrt_square = tf.reshape(vector, [n_points])
        defensive = self._defensive_marginal_values(keep_axes, points)
        return sqrt_square + self.tau * defensive

    def _defensive_marginal_values(
        self,
        keep_axes: tuple[int, ...],
        points: tf.Tensor,
    ) -> tf.Tensor:
        if keep_axes == tuple(range(len(self.sqrt_tt.cores))):
            return tf.exp(self.defensive_density.log_density(points))
        if isinstance(self.defensive_density, TensorProductReferenceDensity):
            retained_basis = ProductBasis(
                [self.sqrt_tt.product_basis.bases[axis] for axis in keep_axes],
                self.measure_convention,
            )
            retained_reference = TensorProductReferenceDensity(
                retained_basis,
                self.measure_convention,
                floor=self.defensive_density.floor,
            )
            values = tf.exp(retained_reference.log_density(points))
            if self.measure_convention.mass_measure is MassMeasure.REFERENCE_LEBESGUE:
                integrated_axes = tuple(
                    axis
                    for axis in range(len(self.sqrt_tt.cores))
                    if axis not in keep_axes
                )
                volume = tf.constant(1.0, tf.float64)
                for axis in integrated_axes:
                    volume = volume * self.sqrt_tt.product_basis.bases[axis].domain.length
                values = values * volume
            return values
        raise NotImplementedError("source-style defensive marginal requires tensor-product reference density")

    def conditional_density(
        self,
        axis: int,
        prefix: tf.Tensor,
        grid: tf.Tensor,
    ) -> tf.Tensor:
        grid_values = tf.convert_to_tensor(grid, dtype=tf.float64)
        prefix_values = tf.convert_to_tensor(prefix, dtype=tf.float64)
        if grid_values.shape.rank != 1:
            raise ValueError(f"grid: {HighDimStatus.INVALID_SHAPE.value}")
        if prefix_values.shape.rank != 2:
            raise ValueError(f"prefix: {HighDimStatus.INVALID_SHAPE.value}")
        if axis < 0 or axis >= len(self.sqrt_tt.cores):
            raise IndexError("axis out of range")
        if prefix_values.shape[0] != 1 or prefix_values.shape[1] != axis:
            raise ValueError(f"prefix: {HighDimStatus.INVALID_SHAPE.value}")
        values = self._prefix_axis_marginal_values(axis, prefix_values, grid_values)
        integral = _trapezoid_integral(grid_values, values)
        if not bool(tf.math.is_finite(integral).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        if bool((integral <= self.denominator_floor).numpy()):
            raise ValueError(HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED.value)
        return values / integral

    def _prefix_axis_marginal_values(
        self,
        axis: int,
        prefix: tf.Tensor,
        grid: tf.Tensor,
    ) -> tf.Tensor:
        dimension = len(self.sqrt_tt.cores)
        suffix_axes = tuple(range(axis + 1, dimension))
        if not suffix_axes:
            full_points = _points_with_prefix_axis_grid(
                dimension=dimension,
                axis=axis,
                prefix=prefix,
                grid=grid,
                suffix_points=None,
            )
            return tf.exp(self.log_density(full_points))
        suffix_grids = [self._axis_default_grid(suffix_axis) for suffix_axis in suffix_axes]
        suffix_points, suffix_weights = _tensor_product_grid_and_weights(suffix_grids)
        rows = []
        for z_value in tf.unstack(grid):
            full_points = _points_with_prefix_axis_grid(
                dimension=dimension,
                axis=axis,
                prefix=prefix,
                grid=tf.reshape(z_value, [1]),
                suffix_points=suffix_points,
            )
            values = tf.exp(self.log_density(full_points))
            rows.append(tf.reduce_sum(values * suffix_weights))
        return tf.stack(rows)

    def _axis_default_grid(self, axis: int) -> tf.Tensor:
        basis = self.sqrt_tt.product_basis.bases[axis]
        return tf.linspace(basis.domain.left, basis.domain.right, 33)

    def manifest_payload(self) -> Mapping[str, object]:
        return self._manifest_payload_from(
            sqrt_tt=self.sqrt_tt,
            defensive_density=self.defensive_density,
            tau=self.tau,
            normalizer_floor=self.normalizer_floor,
            denominator_floor=self.denominator_floor,
            measure_convention=self.measure_convention,
        )

    @staticmethod
    def _manifest_payload_from(
        sqrt_tt: FunctionalTT,
        defensive_density: DefensiveDensityProtocol,
        tau: tf.Tensor,
        normalizer_floor: tf.Tensor,
        denominator_floor: tf.Tensor,
        measure_convention: MeasureConvention,
    ) -> Mapping[str, object]:
        return {
            "family": "SquaredTTDensity",
            "sqrt_tt": sqrt_tt.manifest_payload(),
            "defensive_density": defensive_density.manifest_payload(),
            "tau": tau,
            "normalizer_floor": normalizer_floor,
            "denominator_floor": denominator_floor,
            "measure_convention": {
                "density_measure": measure_convention.density_measure.value,
                "mass_measure": measure_convention.mass_measure.value,
                "reference_weight_name": measure_convention.reference_weight_name,
            },
        }

    def manifest(self, version: str = "squared_tt_density.v1") -> BranchManifest:
        return BranchManifest(version=version, payload=self.manifest_payload())

    @staticmethod
    def expected_manifest(
        sqrt_tt: FunctionalTT,
        defensive_density: DefensiveDensityProtocol,
        tau: tf.Tensor,
        normalizer_floor: tf.Tensor,
        denominator_floor: tf.Tensor,
        measure_convention: MeasureConvention,
        version: str = "squared_tt_density.v1",
    ) -> BranchManifest:
        return BranchManifest(
            version=version,
            payload=SquaredTTDensity._manifest_payload_from(
                sqrt_tt=sqrt_tt,
                defensive_density=defensive_density,
                tau=tf.convert_to_tensor(tau, dtype=tf.float64),
                normalizer_floor=tf.convert_to_tensor(normalizer_floor, dtype=tf.float64),
                denominator_floor=tf.convert_to_tensor(denominator_floor, dtype=tf.float64),
                measure_convention=measure_convention,
            ),
        )

    @staticmethod
    def expected_branch_identity(
        sqrt_tt: FunctionalTT,
        defensive_density: DefensiveDensityProtocol,
        tau: tf.Tensor,
        normalizer_floor: tf.Tensor,
        denominator_floor: tf.Tensor,
        measure_convention: MeasureConvention,
    ) -> BranchIdentity:
        manifest = SquaredTTDensity.expected_manifest(
            sqrt_tt=sqrt_tt,
            defensive_density=defensive_density,
            tau=tau,
            normalizer_floor=normalizer_floor,
            denominator_floor=denominator_floor,
            measure_convention=measure_convention,
        )
        return BranchIdentity(manifest=manifest, hash=manifest.sha256())


def _points_with_prefix_axis_grid(
    dimension: int,
    axis: int,
    prefix: tf.Tensor,
    grid: tf.Tensor,
    suffix_points: tf.Tensor | None,
) -> tf.Tensor:
    n = tf.shape(suffix_points)[0] if suffix_points is not None else tf.shape(grid)[0]
    columns = []
    for dim in range(dimension):
        if dim < axis:
            columns.append(tf.fill([n], prefix[0, dim]))
        elif dim == axis:
            if suffix_points is None:
                columns.append(grid)
            else:
                columns.append(tf.fill([n], grid[0]))
        else:
            if suffix_points is None:
                columns.append(tf.zeros([n], dtype=tf.float64))
            else:
                columns.append(suffix_points[:, dim - axis - 1])
    return tf.stack(columns, axis=1)


def _validate_source_marginal_axes(axes: tuple[int, ...], dimension: int) -> None:
    if not axes:
        return
    prefix = tuple(range(len(axes)))
    suffix = tuple(range(dimension - len(axes), dimension))
    if axes != prefix and axes != suffix:
        raise NotImplementedError("source-style marginal currently supports retained prefix or suffix axes")


def _trapezoid_integral(grid: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
    widths = grid[1:] - grid[:-1]
    heights = 0.5 * (values[1:] + values[:-1])
    return tf.reduce_sum(widths * heights)


def trapezoid_integral(grid: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
    """Public internal helper for tests and transport grid integrations."""

    return _trapezoid_integral(
        tf.convert_to_tensor(grid, dtype=tf.float64),
        tf.convert_to_tensor(values, dtype=tf.float64),
    )


def _trapezoid_weights(grid: tf.Tensor) -> tf.Tensor:
    widths = grid[1:] - grid[:-1]
    if grid.shape[0] == 2:
        return tf.constant([0.5, 0.5], dtype=tf.float64) * widths[0]
    middle = 0.5 * (widths[:-1] + widths[1:])
    return tf.concat([widths[:1] * 0.5, middle, widths[-1:] * 0.5], axis=0)


def _tensor_product_grid_and_weights(grids: Sequence[tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    meshes = tf.meshgrid(*grids, indexing="ij")
    weight_meshes = tf.meshgrid(*[_trapezoid_weights(grid) for grid in grids], indexing="ij")
    points = tf.stack([tf.reshape(mesh, [-1]) for mesh in meshes], axis=1)
    weights = tf.ones([tf.shape(points)[0]], dtype=tf.float64)
    for weight_mesh in weight_meshes:
        weights = weights * tf.reshape(weight_mesh, [-1])
    return points, weights
