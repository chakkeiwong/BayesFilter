"""KR transport diagnostics and fixed source-style transport contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.diagnostics import HighDimStatus, freeze_mapping
from bayesfilter.highdim.squared_tt import SquaredTTDensity, trapezoid_integral


@dataclass(frozen=True)
class KRCDFConfig:
    """Configuration for deterministic grid CDF and bisection inversion."""

    grid_size: int
    bisection_steps: int
    monotonicity_tolerance: float
    bracket_tolerance: float
    denominator_floor: float
    max_floor_count: int
    dtype: tf.DType = tf.float64

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.bisection_steps <= 0:
            raise ValueError("bisection_steps must be positive")
        if self.dtype != tf.float64:
            raise ValueError("KRCDFConfig requires tf.float64")


@dataclass(frozen=True)
class KRInversionResult:
    """Result for one coordinate inverse-CDF operation."""

    z_value: tf.Tensor
    cdf_value: tf.Tensor
    iterations: int
    status: HighDimStatus
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "z_value",
            tf.convert_to_tensor(self.z_value, dtype=tf.float64),
        )
        object.__setattr__(
            self,
            "cdf_value",
            tf.convert_to_tensor(self.cdf_value, dtype=tf.float64),
        )
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be a HighDimStatus")
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


@dataclass(frozen=True)
class KRTransport:
    """Lower-triangular KR map built from Phase-2 grid conditionals."""

    density: SquaredTTDensity
    coordinate_order: tuple[int, ...]
    cdf_config: KRCDFConfig

    def __init__(
        self,
        density: SquaredTTDensity,
        coordinate_order: Sequence[int],
        cdf_config: KRCDFConfig,
    ) -> None:
        if not isinstance(density, SquaredTTDensity):
            raise TypeError("density must be a SquaredTTDensity")
        order = tuple(int(axis) for axis in coordinate_order)
        dimension = len(density.sqrt_tt.cores)
        if sorted(order) != list(range(dimension)):
            raise ValueError(f"coordinate_order: {HighDimStatus.INVALID_SHAPE.value}")
        if order != tuple(range(dimension)):
            raise NotImplementedError("Phase 2 supports natural coordinate order")
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "coordinate_order", order)
        object.__setattr__(self, "cdf_config", cdf_config)

    def forward(self, z_points: tf.Tensor):
        values = tf.convert_to_tensor(z_points, dtype=tf.float64)
        if values.shape.rank != 2 or values.shape[1] != len(self.coordinate_order):
            raise ValueError(f"z_points: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        rows = []
        log_terms = []
        per_axis_results = []
        for row_index in range(int(values.shape[0])):
            row = values[row_index : row_index + 1, :]
            u_columns = []
            row_log_terms = []
            for axis in self.coordinate_order:
                prefix = row[:, :axis]
                z_value = row[:, axis]
                cdf_value, density_value, status, diagnostics = self._cdf_at(axis, prefix, z_value)
                u_columns.append(tf.reshape(cdf_value, []))
                row_log_terms.append(tf.math.log(tf.reshape(density_value, [])))
                per_axis_results.append(
                    KRInversionResult(
                        z_value=tf.reshape(z_value, []),
                        cdf_value=tf.reshape(cdf_value, []),
                        iterations=0,
                        status=status,
                        diagnostics=diagnostics,
                    )
                )
            rows.append(tf.stack(u_columns))
            log_terms.append(tf.reduce_sum(tf.stack(row_log_terms)))
        return tf.stack(rows, axis=0), tf.stack(log_terms), per_axis_results

    def inverse(self, u_points: tf.Tensor):
        values = tf.convert_to_tensor(u_points, dtype=tf.float64)
        if values.shape.rank != 2 or values.shape[1] != len(self.coordinate_order):
            raise ValueError(f"u_points: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        rows = []
        log_terms = []
        per_axis_results = []
        for row_index in range(int(values.shape[0])):
            current = []
            row_log_terms = []
            for axis in self.coordinate_order:
                if axis != len(current):
                    raise NotImplementedError("Phase 2 supports natural coordinate order")
                prefix = (
                    tf.reshape(tf.stack(current), [1, axis])
                    if current
                    else tf.zeros([1, 0], dtype=tf.float64)
                )
                result, density_value = self._inverse_axis(
                    axis,
                    prefix,
                    tf.reshape(values[row_index, axis], []),
                )
                current.append(tf.reshape(result.z_value, []))
                row_log_terms.append(-tf.math.log(tf.reshape(density_value, [])))
                per_axis_results.append(result)
            rows.append(tf.stack(current))
            log_terms.append(tf.reduce_sum(tf.stack(row_log_terms)))
        return tf.stack(rows, axis=0), tf.stack(log_terms), per_axis_results

    def log_jacobian(self, z_points: tf.Tensor) -> tf.Tensor:
        _, log_det, _ = self.forward(z_points)
        return log_det

    def _axis_grid(self, axis: int) -> tf.Tensor:
        basis = self.density.sqrt_tt.product_basis.bases[axis]
        return tf.linspace(
            basis.domain.left,
            basis.domain.right,
            self.cdf_config.grid_size,
        )

    def _cdf_at(self, axis: int, prefix: tf.Tensor, z_value: tf.Tensor):
        if not bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(z_value))).numpy()):
            return (
                tf.constant(float("nan"), dtype=tf.float64),
                tf.constant(float("nan"), dtype=tf.float64),
                HighDimStatus.NONFINITE_VALUE,
                {"z_value": z_value},
            )
        grid = self._axis_grid(axis)
        conditional = self.density.conditional_density(axis, prefix, grid)
        if not bool(tf.reduce_all(tf.math.is_finite(conditional)).numpy()):
            return (
                tf.constant(float("nan"), dtype=tf.float64),
                tf.constant(float("nan"), dtype=tf.float64),
                HighDimStatus.NONFINITE_VALUE,
                {"reason": "nonfinite conditional"},
            )
        increments = 0.5 * (conditional[1:] + conditional[:-1]) * (grid[1:] - grid[:-1])
        cdf_grid = tf.concat(
            [tf.zeros([1], dtype=tf.float64), tf.cumsum(increments)],
            axis=0,
        )
        total = cdf_grid[-1]
        if not bool(tf.math.is_finite(total).numpy()):
            return (
                tf.constant(float("nan"), dtype=tf.float64),
                tf.constant(float("nan"), dtype=tf.float64),
                HighDimStatus.NONFINITE_VALUE,
                {"total": total},
            )
        if bool((total <= self.cdf_config.denominator_floor).numpy()):
            return (
                tf.constant(float("nan"), dtype=tf.float64),
                tf.constant(float("nan"), dtype=tf.float64),
                HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED,
                {"total": total},
            )
        cdf_grid = cdf_grid / total
        min_increment = tf.reduce_min(cdf_grid[1:] - cdf_grid[:-1])
        if not bool(tf.math.is_finite(min_increment).numpy()):
            return (
                tf.constant(float("nan"), dtype=tf.float64),
                tf.constant(float("nan"), dtype=tf.float64),
                HighDimStatus.NONFINITE_VALUE,
                {"min_increment": min_increment},
            )
        if bool((min_increment < -self.cdf_config.monotonicity_tolerance).numpy()):
            return (
                tf.constant(float("nan"), dtype=tf.float64),
                tf.constant(float("nan"), dtype=tf.float64),
                HighDimStatus.CDF_MONOTONICITY_FAILURE,
                {"min_increment": min_increment},
            )
        z_scalar = tf.reshape(z_value, [])
        cdf_value = _interp_1d(z_scalar, grid, cdf_grid)
        density_value = _interp_1d(z_scalar, grid, conditional)
        return cdf_value, density_value, HighDimStatus.OK, {"min_increment": min_increment}

    def _inverse_axis(self, axis: int, prefix: tf.Tensor, target_u: tf.Tensor):
        target = tf.reshape(tf.convert_to_tensor(target_u, dtype=tf.float64), [])
        if not bool(tf.math.is_finite(target).numpy()):
            result = KRInversionResult(
                z_value=tf.constant(float("nan"), dtype=tf.float64),
                cdf_value=target,
                iterations=0,
                status=HighDimStatus.NONFINITE_VALUE,
                diagnostics={"target": target},
            )
            return result, tf.constant(float("nan"), dtype=tf.float64)
        if bool((target < -self.cdf_config.bracket_tolerance).numpy()) or bool(
            (target > 1.0 + self.cdf_config.bracket_tolerance).numpy()
        ):
            result = KRInversionResult(
                z_value=tf.constant(float("nan"), dtype=tf.float64),
                cdf_value=target,
                iterations=0,
                status=HighDimStatus.INVERSE_BRACKET_FAILURE,
                diagnostics={"target": target},
            )
            return result, tf.constant(float("nan"), dtype=tf.float64)
        grid = self._axis_grid(axis)
        lo = tf.reshape(grid[0], [])
        hi = tf.reshape(grid[-1], [])
        mid = 0.5 * (lo + hi)
        cdf_mid = tf.constant(float("nan"), dtype=tf.float64)
        density_mid = tf.constant(float("nan"), dtype=tf.float64)
        status = HighDimStatus.OK
        diagnostics = {}
        for _ in range(self.cdf_config.bisection_steps):
            mid = 0.5 * (lo + hi)
            cdf_mid, density_mid, status, diagnostics = self._cdf_at(
                axis,
                prefix,
                tf.reshape(mid, [1]),
            )
            if status is not HighDimStatus.OK:
                break
            if bool((cdf_mid < target).numpy()):
                lo = mid
            else:
                hi = mid
        return (
            KRInversionResult(
                z_value=mid,
                cdf_value=cdf_mid,
                iterations=self.cdf_config.bisection_steps,
                status=status,
                diagnostics=diagnostics,
            ),
            density_mid,
        )


def _interp_1d(x: tf.Tensor, grid: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
    clipped = tf.clip_by_value(x, grid[0], grid[-1])
    right = tf.searchsorted(grid, tf.reshape(clipped, [1]), side="right")[0]
    right = tf.clip_by_value(right, 1, tf.shape(grid)[0] - 1)
    left = right - 1
    x0 = tf.gather(grid, left)
    x1 = tf.gather(grid, right)
    y0 = tf.gather(values, left)
    y1 = tf.gather(values, right)
    weight = (clipped - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


@dataclass(frozen=True)
class FixedTTSIRTTransport:
    """Fixed source-style TTSIRT transport over a squared TT density.

    This class implements the reference-coordinate map surface used by the
    source-route protocol.  The one-dimensional CDFs are numerical CDF
    constructors over the source conditional densities; they are not promoted
    from the older diagnostic ``KRTransport`` object.
    """

    density: SquaredTTDensity
    cdf_config: KRCDFConfig
    int_dir: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.density, SquaredTTDensity):
            raise TypeError("density must be a SquaredTTDensity")
        if not isinstance(self.cdf_config, KRCDFConfig):
            raise TypeError("cdf_config must be a KRCDFConfig")
        if int(self.int_dir) != 1:
            raise NotImplementedError("FixedTTSIRTTransport currently supports int_dir > 0")
        object.__setattr__(self, "int_dir", int(self.int_dir))

    @property
    def dimension(self) -> int:
        return len(self.density.sqrt_tt.cores)

    def manifest_payload(self) -> Mapping[str, object]:
        tau = float(self.density.tau.numpy())
        return {
            "family": "FixedTTSIRTTransport",
            "source_contract_level": "fixed_ttsirt",
            "tt_cores_declared": True,
            "defensive_density_declared": True,
            "defensive_mass_positive": tau > 0.0,
            "defensive_tau": tau,
            "defensive_tau_source": "author_executable_ttsirt_default",
            "source_map_semantics": "ttsirt_eval_irt_rt_cirt_reference_style",
            "proposition2_marginal_backend": "paired_core_mass_contraction_prefix_suffix",
            "conditional_cdf_backend": "numerical_grid_trapezoid_bisection",
            "conditional_cdf_route_class": "fixed_hmc_adaptation_diagnostic_approximation",
            "production_kr_closure": False,
            "proposal_density_backend": "eval_pdf_on_local_samples",
            "p83_nonclaims": (
                "no production KR closure",
                "no d18 correctness",
                "no author-scale fit quality",
                "no derivative readiness",
                "no LEDH readiness",
                "no HMC readiness",
            ),
            "int_dir": self.int_dir,
            "dimension": self.dimension,
            "cdf_config": {
                "grid_size": self.cdf_config.grid_size,
                "bisection_steps": self.cdf_config.bisection_steps,
                "monotonicity_tolerance": self.cdf_config.monotonicity_tolerance,
                "bracket_tolerance": self.cdf_config.bracket_tolerance,
                "denominator_floor": self.cdf_config.denominator_floor,
                "max_floor_count": self.cdf_config.max_floor_count,
            },
            "source_anchors": (
                "@TTSIRT/eval_irt_reference.m:15-42",
                "@TTSIRT/eval_rt_reference.m:13-33",
                "@TTSIRT/eval_rt_jac_reference.m:1-190",
                "@TTSIRT/eval_cirt_reference.m:43-100",
                "@TTSIRT/eval_potential_reference.m:10-22",
                "@TTSIRT/TTSIRT.m:185-188",
                "@TTSIRT/marginalise.m:85",
            ),
        }

    def inverse_transport(self, reference_points: tf.Tensor) -> tf.Tensor:
        values = _validate_map_points("reference_points", reference_points, self.dimension)
        if not bool(
            tf.reduce_all((values >= 0.0) & (values <= 1.0)).numpy()
        ):
            raise ValueError(f"reference_points: {HighDimStatus.INVERSE_BRACKET_FAILURE.value}")
        rows = []
        for sample_index in range(int(values.shape[1])):
            current = []
            for axis in range(self.dimension):
                prefix = _prefix_from_current(current)
                result, _ = self._inverse_axis(
                    axis,
                    prefix,
                    tf.reshape(values[axis, sample_index], []),
                )
                if result.status is not HighDimStatus.OK:
                    raise ValueError(result.status.value)
                current.append(tf.reshape(result.z_value, []))
            rows.append(tf.stack(current))
        return tf.transpose(tf.stack(rows, axis=0))

    def forward_transport(self, local_points: tf.Tensor) -> tf.Tensor:
        values = _validate_map_points("local_points", local_points, self.dimension)
        columns = []
        for sample_index in range(int(values.shape[1])):
            sample = values[:, sample_index]
            u_rows = []
            for axis in range(self.dimension):
                prefix = tf.reshape(sample[:axis], [1, axis])
                cdf_value, _, status, diagnostics = self._cdf_at(
                    axis,
                    prefix,
                    tf.reshape(sample[axis], []),
                )
                if status is not HighDimStatus.OK:
                    raise ValueError(f"forward_transport: {status.value}: {diagnostics}")
                u_rows.append(tf.reshape(cdf_value, []))
            columns.append(tf.stack(u_rows))
        return tf.transpose(tf.stack(columns, axis=0))

    def forward_log_jacobian(self, local_points: tf.Tensor) -> tf.Tensor:
        values = _validate_map_points("local_points", local_points, self.dimension)
        terms = []
        for sample_index in range(int(values.shape[1])):
            sample = values[:, sample_index]
            sample_terms = []
            for axis in range(self.dimension):
                prefix = tf.reshape(sample[:axis], [1, axis])
                _, density_value, status, diagnostics = self._cdf_at(
                    axis,
                    prefix,
                    tf.reshape(sample[axis], []),
                )
                if status is not HighDimStatus.OK:
                    raise ValueError(f"forward_log_jacobian: {status.value}: {diagnostics}")
                sample_terms.append(tf.math.log(tf.reshape(density_value, [])))
            terms.append(tf.reduce_sum(tf.stack(sample_terms)))
        return tf.stack(terms)

    def conditional_inverse_transport(
        self,
        conditioning_points: tf.Tensor,
        reference_points: tf.Tensor,
    ) -> tf.Tensor:
        condition = tf.convert_to_tensor(conditioning_points, dtype=tf.float64)
        reference = tf.convert_to_tensor(reference_points, dtype=tf.float64)
        if condition.shape.rank != 2 or reference.shape.rank != 2:
            raise ValueError(f"conditional_inverse_transport: {HighDimStatus.INVALID_SHAPE.value}")
        if int(condition.shape[0]) + int(reference.shape[0]) != self.dimension:
            raise ValueError(f"conditional_inverse_transport: {HighDimStatus.INVALID_SHAPE.value}")
        if int(condition.shape[1]) not in (1, int(reference.shape[1])):
            raise ValueError(f"conditional_inverse_transport: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(
            tf.reduce_all(tf.math.is_finite(condition)).numpy()
            and tf.reduce_all(tf.math.is_finite(reference)).numpy()
        ):
            raise ValueError(f"conditional_inverse_transport: {HighDimStatus.NONFINITE_VALUE.value}")
        if not bool(
            tf.reduce_all((reference >= 0.0) & (reference <= 1.0)).numpy()
        ):
            raise ValueError(
                f"conditional_inverse_transport: {HighDimStatus.INVERSE_BRACKET_FAILURE.value}"
            )
        dx = int(condition.shape[0])
        dr = int(reference.shape[0])
        outputs = []
        for sample_index in range(int(reference.shape[1])):
            cond_index = sample_index if int(condition.shape[1]) > 1 else 0
            current = [
                tf.reshape(value, [])
                for value in tf.unstack(condition[:, cond_index], axis=0)
            ]
            generated = []
            for local_axis in range(dr):
                axis = dx + local_axis
                prefix = _prefix_from_current(current)
                result, _ = self._inverse_axis(
                    axis,
                    prefix,
                    tf.reshape(reference[local_axis, sample_index], []),
                )
                if result.status is not HighDimStatus.OK:
                    raise ValueError(result.status.value)
                value = tf.reshape(result.z_value, [])
                current.append(value)
                generated.append(value)
            outputs.append(tf.stack(generated))
        return tf.transpose(tf.stack(outputs, axis=0))

    def eval_pdf(self, local_points: tf.Tensor) -> tf.Tensor:
        values = _validate_map_points("local_points", local_points, self.dimension)
        reference_density = tf.exp(self.density.log_density(tf.transpose(values)))
        return reference_density * self._reference_measure_density(values)

    def potential(self, local_points: tf.Tensor) -> tf.Tensor:
        return -tf.math.log(self.eval_pdf(local_points))

    def proposal_log_density(
        self,
        *,
        local_points: tf.Tensor,
        reference_points: tf.Tensor,
    ) -> tf.Tensor:
        del reference_points
        return tf.math.log(self.eval_pdf(local_points))

    def conditional_proposal_log_density(
        self,
        *,
        conditioning_points: tf.Tensor,
        generated_points: tf.Tensor,
    ) -> tf.Tensor:
        """Evaluate the suffix density using the Proposition-2 prefix marginal."""

        condition = tf.convert_to_tensor(conditioning_points, dtype=tf.float64)
        generated = tf.convert_to_tensor(generated_points, dtype=tf.float64)
        if condition.shape.rank != 2 or generated.shape.rank != 2:
            raise ValueError(
                f"conditional_proposal_log_density: {HighDimStatus.INVALID_SHAPE.value}"
            )
        conditioning_dimension = int(condition.shape[0])
        if conditioning_dimension < 1 or conditioning_dimension >= self.dimension:
            raise ValueError(
                f"conditional_proposal_log_density: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if conditioning_dimension + int(generated.shape[0]) != self.dimension:
            raise ValueError(
                f"conditional_proposal_log_density: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if int(condition.shape[1]) != int(generated.shape[1]):
            raise ValueError(
                f"conditional_proposal_log_density: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if not bool(
            tf.reduce_all(tf.math.is_finite(condition)).numpy()
            and tf.reduce_all(tf.math.is_finite(generated)).numpy()
        ):
            raise ValueError(
                f"conditional_proposal_log_density: {HighDimStatus.NONFINITE_VALUE.value}"
            )

        joint_points = tf.concat([condition, generated], axis=0)
        joint_log_density = tf.math.log(self.eval_pdf(joint_points))
        prefix_axes = tuple(range(conditioning_dimension))
        prefix_relative_density = self.density.normalized_marginal_density_values(
            prefix_axes,
            tf.transpose(condition),
        )
        prefix_reference_density = tf.ones(
            [tf.shape(condition)[1]], dtype=tf.float64
        )
        for axis in prefix_axes:
            prefix_reference_density = (
                prefix_reference_density
                * self._axis_reference_measure_density(axis)
            )
        prefix_log_density = tf.math.log(
            prefix_relative_density * prefix_reference_density
        )
        result = joint_log_density - prefix_log_density
        if not bool(tf.reduce_all(tf.math.is_finite(result)).numpy()):
            raise ValueError(
                f"conditional_proposal_log_density: {HighDimStatus.NONFINITE_VALUE.value}"
            )
        return result

    def marginalize(self, keep_axes: tuple[int, ...]):
        return self.density.marginal_density(tuple(int(axis) for axis in keep_axes))

    def log_normalizer(self) -> tf.Tensor:
        return tf.math.log(self.density.normalizer())

    def _axis_grid(self, axis: int) -> tf.Tensor:
        basis = self.density.sqrt_tt.product_basis.bases[axis]
        return tf.linspace(
            basis.domain.left,
            basis.domain.right,
            self.cdf_config.grid_size,
        )

    def _cdf_at(self, axis: int, prefix: tf.Tensor, z_value: tf.Tensor):
        if not bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(z_value))).numpy()):
            return (
                tf.constant(float("nan"), dtype=tf.float64),
                tf.constant(float("nan"), dtype=tf.float64),
                HighDimStatus.NONFINITE_VALUE,
                {"z_value": z_value},
            )
        grid = self._axis_grid(axis)
        conditional = self._source_conditional_density(axis, prefix, grid)
        return _cdf_from_conditional_grid(
            grid=grid,
            conditional=conditional,
            z_value=z_value,
            config=self.cdf_config,
        )

    def _inverse_axis(self, axis: int, prefix: tf.Tensor, target_u: tf.Tensor):
        target = tf.reshape(tf.convert_to_tensor(target_u, dtype=tf.float64), [])
        if not bool(tf.math.is_finite(target).numpy()):
            result = KRInversionResult(
                z_value=tf.constant(float("nan"), dtype=tf.float64),
                cdf_value=target,
                iterations=0,
                status=HighDimStatus.NONFINITE_VALUE,
                diagnostics={"target": target},
            )
            return result, tf.constant(float("nan"), dtype=tf.float64)
        if bool((target < -self.cdf_config.bracket_tolerance).numpy()) or bool(
            (target > 1.0 + self.cdf_config.bracket_tolerance).numpy()
        ):
            result = KRInversionResult(
                z_value=tf.constant(float("nan"), dtype=tf.float64),
                cdf_value=target,
                iterations=0,
                status=HighDimStatus.INVERSE_BRACKET_FAILURE,
                diagnostics={"target": target},
            )
            return result, tf.constant(float("nan"), dtype=tf.float64)
        grid = self._axis_grid(axis)
        lo = tf.reshape(grid[0], [])
        hi = tf.reshape(grid[-1], [])
        mid = 0.5 * (lo + hi)
        cdf_mid = tf.constant(float("nan"), dtype=tf.float64)
        density_mid = tf.constant(float("nan"), dtype=tf.float64)
        status = HighDimStatus.OK
        diagnostics = {}
        for _ in range(self.cdf_config.bisection_steps):
            mid = 0.5 * (lo + hi)
            cdf_mid, density_mid, status, diagnostics = self._cdf_at(
                axis,
                _prefix_from_current([]) if prefix.shape[1] == 0 else prefix,
                tf.reshape(mid, []),
            )
            if status is not HighDimStatus.OK:
                break
            if bool((cdf_mid < target).numpy()):
                lo = mid
            else:
                hi = mid
        return (
            KRInversionResult(
                z_value=mid,
                cdf_value=cdf_mid,
                iterations=self.cdf_config.bisection_steps,
                status=status,
                diagnostics=diagnostics,
            ),
            density_mid,
        )

    def _source_conditional_density(
        self,
        axis: int,
        prefix: tf.Tensor,
        grid: tf.Tensor,
    ) -> tf.Tensor:
        if axis < 0 or axis >= self.dimension:
            raise IndexError("axis out of range")
        prefix_values = tf.convert_to_tensor(prefix, dtype=tf.float64)
        if prefix_values.shape.rank != 2 or prefix_values.shape != (1, axis):
            raise ValueError(f"prefix: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(tf.reduce_all(tf.math.is_finite(prefix_values)).numpy()):
            raise ValueError(f"prefix: {HighDimStatus.NONFINITE_VALUE.value}")
        if axis == 0:
            numerator = self.density.normalized_marginal_density_values(
                (0,),
                tf.reshape(grid, [-1, 1]),
            )
            denominator = tf.constant(1.0, dtype=tf.float64)
        else:
            prefix_axis_points = tf.concat(
                [
                    tf.tile(prefix_values, [tf.shape(grid)[0], 1]),
                    tf.reshape(grid, [-1, 1]),
                ],
                axis=1,
            )
            numerator = self.density.normalized_marginal_density_values(
                tuple(range(axis + 1)),
                prefix_axis_points,
            )
            denominator = self.density.normalized_marginal_density_values(
                tuple(range(axis)),
                prefix_values,
            )[0]
        if not bool(tf.math.is_finite(denominator).numpy()) or bool(
            (denominator <= self.density.denominator_floor).numpy()
        ):
            raise ValueError(HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED.value)
        conditional = numerator / denominator
        conditional = conditional * self._axis_reference_measure_density(axis)
        if not bool(tf.reduce_all(tf.math.is_finite(conditional)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        return conditional

    def _axis_reference_measure_density(self, axis: int) -> tf.Tensor:
        basis = self.density.sqrt_tt.product_basis.bases[axis]
        return tf.math.reciprocal(basis.domain.length)

    def _reference_measure_density(self, points: tf.Tensor) -> tf.Tensor:
        density = tf.ones([tf.shape(points)[1]], dtype=tf.float64)
        for axis in range(self.dimension):
            density = density * self._axis_reference_measure_density(axis)
        return density


def _cdf_from_conditional_grid(
    *,
    grid: tf.Tensor,
    conditional: tf.Tensor,
    z_value: tf.Tensor,
    config: KRCDFConfig,
):
    if not bool(tf.reduce_all(tf.math.is_finite(conditional)).numpy()):
        return (
            tf.constant(float("nan"), dtype=tf.float64),
            tf.constant(float("nan"), dtype=tf.float64),
            HighDimStatus.NONFINITE_VALUE,
            {"reason": "nonfinite conditional"},
        )
    increments = 0.5 * (conditional[1:] + conditional[:-1]) * (grid[1:] - grid[:-1])
    cdf_grid = tf.concat(
        [tf.zeros([1], dtype=tf.float64), tf.cumsum(increments)],
        axis=0,
    )
    total = cdf_grid[-1]
    if not bool(tf.math.is_finite(total).numpy()):
        return (
            tf.constant(float("nan"), dtype=tf.float64),
            tf.constant(float("nan"), dtype=tf.float64),
            HighDimStatus.NONFINITE_VALUE,
            {"total": total},
        )
    if bool((total <= config.denominator_floor).numpy()):
        return (
            tf.constant(float("nan"), dtype=tf.float64),
            tf.constant(float("nan"), dtype=tf.float64),
            HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED,
            {"total": total},
        )
    cdf_grid = cdf_grid / total
    min_increment = tf.reduce_min(cdf_grid[1:] - cdf_grid[:-1])
    if not bool(tf.math.is_finite(min_increment).numpy()):
        return (
            tf.constant(float("nan"), dtype=tf.float64),
            tf.constant(float("nan"), dtype=tf.float64),
            HighDimStatus.NONFINITE_VALUE,
            {"min_increment": min_increment},
        )
    if bool((min_increment < -config.monotonicity_tolerance).numpy()):
        return (
            tf.constant(float("nan"), dtype=tf.float64),
            tf.constant(float("nan"), dtype=tf.float64),
            HighDimStatus.CDF_MONOTONICITY_FAILURE,
            {"min_increment": min_increment},
        )
    z_scalar = tf.reshape(z_value, [])
    cdf_value = _interp_1d(z_scalar, grid, cdf_grid)
    density_value = _interp_1d(z_scalar, grid, conditional) / total
    return cdf_value, density_value, HighDimStatus.OK, {"min_increment": min_increment}


def _validate_map_points(name: str, points: tf.Tensor, dimension: int) -> tf.Tensor:
    values = tf.convert_to_tensor(points, dtype=tf.float64)
    if values.shape.rank != 2 or int(values.shape[0]) != int(dimension):
        raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
    return values


def _prefix_from_current(current: Sequence[tf.Tensor]) -> tf.Tensor:
    if not current:
        return tf.zeros([1, 0], dtype=tf.float64)
    return tf.reshape(tf.stack([tf.reshape(value, []) for value in current]), [1, len(current)])
