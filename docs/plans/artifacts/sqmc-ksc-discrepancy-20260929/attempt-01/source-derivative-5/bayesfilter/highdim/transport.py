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
    max_batch_working_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.bisection_steps <= 0:
            raise ValueError("bisection_steps must be positive")
        if self.dtype != tf.float64:
            raise ValueError("KRCDFConfig requires tf.float64")
        if int(self.max_batch_working_bytes) <= 0:
            raise ValueError("max_batch_working_bytes must be positive")


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


def _interp_rows(x: tf.Tensor, grid: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
    """Interpolate one value in every row of a shared-grid table."""

    points = tf.reshape(tf.convert_to_tensor(x, dtype=tf.float64), [-1])
    table = tf.convert_to_tensor(values, dtype=tf.float64)
    if table.shape.rank != 2 or table.shape[0] != points.shape[0]:
        raise ValueError(f"values: {HighDimStatus.INVALID_SHAPE.value}")
    clipped = tf.clip_by_value(points, grid[0], grid[-1])
    right = tf.searchsorted(grid, clipped, side="right")
    right = tf.clip_by_value(right, 1, tf.shape(grid)[0] - 1)
    left = right - 1
    rows = tf.range(tf.shape(points)[0], dtype=tf.int32)
    y0 = tf.gather_nd(table, tf.stack([rows, left], axis=1))
    y1 = tf.gather_nd(table, tf.stack([rows, right], axis=1))
    x0 = tf.gather(grid, left)
    x1 = tf.gather(grid, right)
    weight = (clipped - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


@dataclass(frozen=True)
class FixedTTSIRTTransport:
    """Fixed TTSIRT transport with source-grounded and local diagnostic pieces.

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
            "route_classification": "extension_or_invention",
            "source_contract_level": "fixed_ttsirt",
            "source_contract_level_role": "api_capability_not_source_faithfulness",
            "tt_cores_declared": True,
            "defensive_density_declared": True,
            "defensive_mass_positive": tau > 0.0,
            "defensive_tau": tau,
            "defensive_mass_operation_classification": "source_faithful",
            "defensive_tau_value_classification": "extension_or_invention",
            "defensive_tau_source": "caller_supplied_scope_specific_hypothesis",
            "source_map_semantics": "ttsirt_eval_irt_rt_cirt_reference_style",
            "proposition2_marginal_backend": "paired_core_mass_contraction_prefix_suffix",
            "proposition2_marginal_classification": "source_faithful",
            "conditional_cdf_backend": "numerical_grid_trapezoid_bisection",
            "conditional_cdf_route_class": "extension_or_invention_diagnostic_approximation",
            "upper_suffix_conditional_available": True,
            "upper_suffix_conditional_classification": "extension_or_invention",
            "upper_suffix_conditional_dependency": "generated_axes_reverse_given_fixed_suffix",
            "batched_inverse_grid_reuse": True,
            "batched_inverse_semantics": "same_grid_cdf_and_bisection_as_scalar_route",
            "batched_inverse_grid_reuse_classification": "extension_or_invention",
            "fixed_settings_classification": "fixed_hmc_adaptation",
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
                "max_batch_working_bytes": self.cdf_config.max_batch_working_bytes,
            },
            "paper_anchors": (
                ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:539-573",
                ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:592-670",
                ".localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:807-924",
            ),
            "author_source_anchors": (
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_irt_reference.m:15-42",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_rt_reference.m:13-33",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43-100",
                "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:19-85",
            ),
        }

    def inverse_transport(self, reference_points: tf.Tensor) -> tf.Tensor:
        values = _validate_map_points("reference_points", reference_points, self.dimension)
        if not bool(
            tf.reduce_all((values >= 0.0) & (values <= 1.0)).numpy()
        ):
            raise ValueError(f"reference_points: {HighDimStatus.INVERSE_BRACKET_FAILURE.value}")
        sample_count = int(values.shape[1])
        current = []
        for axis in range(self.dimension):
            prefix = (
                tf.stack(current, axis=1)
                if current
                else tf.zeros([sample_count, 0], dtype=tf.float64)
            )
            current.append(
                self._inverse_axis_batch(axis, prefix, values[axis])
            )
        return tf.stack(current, axis=0)

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
        sample_count = int(reference.shape[1])
        prefixes = (
            tf.transpose(condition)
            if int(condition.shape[1]) > 1
            else tf.tile(tf.transpose(condition), [sample_count, 1])
        )
        generated = []
        for local_axis in range(dr):
            axis = dx + local_axis
            value = self._inverse_axis_batch(
                axis,
                prefixes,
                reference[local_axis],
            )
            generated.append(value)
            prefixes = tf.concat([prefixes, value[:, tf.newaxis]], axis=1)
        return tf.stack(generated, axis=0)

    def conditional_inverse_transport_suffix(
        self,
        conditioning_points: tf.Tensor,
        reference_points: tf.Tensor,
    ) -> tf.Tensor:
        """Invert the upper conditional map for a fixed suffix.

        This is the dependency order used by Zhao-Cui Eq. (20): the supplied
        suffix is held fixed while generated coordinates are inverted from
        the last generated axis back to the first.  It is distinct from the
        natural lower-prefix route exposed by ``conditional_inverse_transport``.
        """

        condition = tf.convert_to_tensor(conditioning_points, dtype=tf.float64)
        reference = tf.convert_to_tensor(reference_points, dtype=tf.float64)
        if condition.shape.rank != 2 or reference.shape.rank != 2:
            raise ValueError(
                f"conditional_inverse_transport_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        generated_dimension = int(reference.shape[0])
        conditioning_dimension = int(condition.shape[0])
        if generated_dimension + conditioning_dimension != self.dimension:
            raise ValueError(
                f"conditional_inverse_transport_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        sample_count = int(reference.shape[1])
        if int(condition.shape[1]) not in (1, sample_count):
            raise ValueError(
                f"conditional_inverse_transport_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if not bool(
            tf.reduce_all(tf.math.is_finite(condition)).numpy()
            and tf.reduce_all(tf.math.is_finite(reference)).numpy()
        ):
            raise ValueError(
                f"conditional_inverse_transport_suffix: {HighDimStatus.NONFINITE_VALUE.value}"
            )
        if not bool(
            tf.reduce_all((reference >= 0.0) & (reference <= 1.0)).numpy()
        ):
            raise ValueError(
                f"conditional_inverse_transport_suffix: {HighDimStatus.INVERSE_BRACKET_FAILURE.value}"
            )
        suffix = (
            tf.transpose(condition)
            if int(condition.shape[1]) > 1
            else tf.tile(tf.transpose(condition), [sample_count, 1])
        )
        generated = tf.TensorArray(tf.float64, size=generated_dimension)
        known = suffix
        for axis in range(generated_dimension - 1, -1, -1):
            value = self._inverse_axis_suffix_batch(
                axis,
                known,
                reference[axis],
            )
            generated = generated.write(axis, value)
            known = tf.concat([value[:, tf.newaxis], known], axis=1)
        return generated.stack()

    def conditional_forward_transport_suffix(
        self,
        conditioning_points: tf.Tensor,
        generated_points: tf.Tensor,
    ) -> tf.Tensor:
        """Evaluate the upper conditional KR map for a fixed suffix."""

        condition = tf.convert_to_tensor(conditioning_points, dtype=tf.float64)
        generated = tf.convert_to_tensor(generated_points, dtype=tf.float64)
        if condition.shape.rank != 2 or generated.shape.rank != 2:
            raise ValueError(
                f"conditional_forward_transport_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        generated_dimension = int(generated.shape[0])
        if generated_dimension + int(condition.shape[0]) != self.dimension:
            raise ValueError(
                f"conditional_forward_transport_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        sample_count = int(generated.shape[1])
        if int(condition.shape[1]) not in (1, sample_count):
            raise ValueError(
                f"conditional_forward_transport_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if int(condition.shape[1]) == 1:
            suffix = tf.tile(tf.transpose(condition), [sample_count, 1])
        else:
            suffix = tf.transpose(condition)
        if not bool(
            tf.reduce_all(tf.math.is_finite(suffix)).numpy()
            and tf.reduce_all(tf.math.is_finite(generated)).numpy()
        ):
            raise ValueError(
                f"conditional_forward_transport_suffix: {HighDimStatus.NONFINITE_VALUE.value}"
            )
        known = suffix
        uniforms = tf.TensorArray(tf.float64, size=generated_dimension)
        for axis in range(generated_dimension - 1, -1, -1):
            cdf, _, status, diagnostics = self._cdf_at_suffix(
                axis,
                known,
                generated[axis],
            )
            if status is not HighDimStatus.OK:
                raise ValueError(
                    f"conditional_forward_transport_suffix: {status.value}: {diagnostics}"
                )
            uniforms = uniforms.write(axis, cdf)
            known = tf.concat([generated[axis][:, tf.newaxis], known], axis=1)
        return uniforms.stack()

    def conditional_forward_log_jacobian_suffix(
        self,
        conditioning_points: tf.Tensor,
        generated_points: tf.Tensor,
    ) -> tf.Tensor:
        """Log density of the same numerical upper conditional map."""

        condition = tf.convert_to_tensor(conditioning_points, dtype=tf.float64)
        generated = tf.convert_to_tensor(generated_points, dtype=tf.float64)
        if condition.shape.rank != 2 or generated.shape.rank != 2:
            raise ValueError(
                f"conditional_forward_log_jacobian_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        generated_dimension = int(generated.shape[0])
        sample_count = int(generated.shape[1])
        if generated_dimension + int(condition.shape[0]) != self.dimension:
            raise ValueError(
                f"conditional_forward_log_jacobian_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if int(condition.shape[1]) not in (1, sample_count):
            raise ValueError(
                f"conditional_forward_log_jacobian_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        known = (
            tf.tile(tf.transpose(condition), [sample_count, 1])
            if int(condition.shape[1]) == 1
            else tf.transpose(condition)
        )
        log_density = tf.zeros([sample_count], dtype=tf.float64)
        for axis in range(generated_dimension - 1, -1, -1):
            _, density, status, diagnostics = self._cdf_at_suffix(
                axis,
                known,
                generated[axis],
            )
            if status is not HighDimStatus.OK:
                raise ValueError(
                    f"conditional_forward_log_jacobian_suffix: {status.value}: {diagnostics}"
                )
            log_density += tf.math.log(density)
            known = tf.concat([generated[axis][:, tf.newaxis], known], axis=1)
        if not bool(tf.reduce_all(tf.math.is_finite(log_density)).numpy()):
            raise ValueError(
                f"conditional_forward_log_jacobian_suffix: {HighDimStatus.NONFINITE_VALUE.value}"
            )
        return log_density

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
                * self._axis_reference_measure_density(axis, condition[axis])
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

    def conditional_proposal_log_density_suffix(
        self,
        *,
        conditioning_points: tf.Tensor,
        generated_points: tf.Tensor,
    ) -> tf.Tensor:
        """Evaluate a suffix-conditioned density for the upper KR route."""

        condition = tf.convert_to_tensor(conditioning_points, dtype=tf.float64)
        generated = tf.convert_to_tensor(generated_points, dtype=tf.float64)
        if condition.shape.rank != 2 or generated.shape.rank != 2:
            raise ValueError(
                f"conditional_proposal_log_density_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        conditioning_dimension = int(condition.shape[0])
        if conditioning_dimension < 1 or conditioning_dimension >= self.dimension:
            raise ValueError(
                f"conditional_proposal_log_density_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if conditioning_dimension + int(generated.shape[0]) != self.dimension:
            raise ValueError(
                f"conditional_proposal_log_density_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if int(condition.shape[1]) != int(generated.shape[1]):
            raise ValueError(
                f"conditional_proposal_log_density_suffix: {HighDimStatus.INVALID_SHAPE.value}"
            )
        joint = tf.concat([generated, condition], axis=0)
        joint_log_density = tf.math.log(self.eval_pdf(joint))
        suffix_axes = tuple(
            range(self.dimension - conditioning_dimension, self.dimension)
        )
        suffix_relative_density = self.density.normalized_marginal_density_values(
            suffix_axes,
            tf.transpose(condition),
        )
        suffix_reference_density = tf.ones(
            [tf.shape(condition)[1]], dtype=tf.float64
        )
        for axis in suffix_axes:
            suffix_reference_density *= self._axis_reference_measure_density(
                axis, condition[axis - suffix_axes[0]]
            )
        result = joint_log_density - tf.math.log(
            suffix_relative_density * suffix_reference_density
        )
        if not bool(tf.reduce_all(tf.math.is_finite(result)).numpy()):
            raise ValueError(
                f"conditional_proposal_log_density_suffix: {HighDimStatus.NONFINITE_VALUE.value}"
            )
        return result

    def marginalize(self, keep_axes: tuple[int, ...]):
        return self.density.marginal_density(tuple(int(axis) for axis in keep_axes))

    def log_normalizer(self) -> tf.Tensor:
        return tf.math.log(self.density.normalizer())

    def _axis_grid(self, axis: int) -> tf.Tensor:
        del axis
        # Every supported basis uses the finite reference interval [-1, 1].
        # Algebraic maps have unbounded physical domains, so CDF construction
        # must stay in this reference coordinate.
        return tf.linspace(
            tf.constant(-1.0, tf.float64),
            tf.constant(1.0, tf.float64),
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
        result = _cdf_from_conditional_grid(
            grid=grid,
            conditional=conditional,
            z_value=self._axis_domain_to_reference(axis, z_value),
            config=self.cdf_config,
        )
        cdf_value, density_value, status, diagnostics = result
        if status is HighDimStatus.OK:
            # The CDF grid is in reference coordinates. Convert its normalized
            # density back to the local physical coordinate for correction.
            density_value = density_value * self._axis_domain_to_reference_jacobian(
                axis, z_value
            )
        return cdf_value, density_value, status, diagnostics

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
            domain_mid = self._axis_reference_to_domain(axis, mid)
            cdf_mid, density_mid, status, diagnostics = self._cdf_at(
                axis,
                _prefix_from_current([]) if prefix.shape[1] == 0 else prefix,
                tf.reshape(domain_mid, []),
            )
            if status is not HighDimStatus.OK:
                break
            if bool((cdf_mid < target).numpy()):
                lo = mid
            else:
                hi = mid
        return (
            KRInversionResult(
                z_value=self._axis_reference_to_domain(axis, mid),
                cdf_value=cdf_mid,
                iterations=self.cdf_config.bisection_steps,
                status=status,
                diagnostics=diagnostics,
            ),
            density_mid,
        )

    def _inverse_axis_batch(
        self,
        axis: int,
        prefixes: tf.Tensor,
        target_u: tf.Tensor,
    ) -> tf.Tensor:
        """Reuse each row's numerical conditional grid across bisection steps."""

        prefix_values = tf.convert_to_tensor(prefixes, dtype=tf.float64)
        targets = tf.reshape(tf.convert_to_tensor(target_u, dtype=tf.float64), [-1])
        if (
            prefix_values.shape.rank != 2
            or prefix_values.shape[1] != axis
            or prefix_values.shape[0] != targets.shape[0]
        ):
            raise ValueError(f"prefixes: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(
            tf.reduce_all(tf.math.is_finite(prefix_values)).numpy()
            and tf.reduce_all(tf.math.is_finite(targets)).numpy()
        ):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        tolerance = tf.cast(self.cdf_config.bracket_tolerance, tf.float64)
        if not bool(
            tf.reduce_all((targets >= -tolerance) & (targets <= 1.0 + tolerance)).numpy()
        ):
            raise ValueError(HighDimStatus.INVERSE_BRACKET_FAILURE.value)

        grid, cdf_grid = self._conditional_cdf_grid_batch(axis, prefix_values)
        lo = tf.fill(tf.shape(targets), grid[0])
        hi = tf.fill(tf.shape(targets), grid[-1])
        mid = 0.5 * (lo + hi)
        for _ in range(self.cdf_config.bisection_steps):
            mid = 0.5 * (lo + hi)
            cdf_mid = _interp_rows(mid, grid, cdf_grid)
            choose_right = cdf_mid < targets
            lo = tf.where(choose_right, mid, lo)
            hi = tf.where(choose_right, hi, mid)
        return self._axis_reference_to_domain(axis, mid)

    def _inverse_axis_suffix_batch(
        self,
        axis: int,
        suffixes: tf.Tensor,
        target_u: tf.Tensor,
    ) -> tf.Tensor:
        suffix_values = tf.convert_to_tensor(suffixes, dtype=tf.float64)
        targets = tf.reshape(tf.convert_to_tensor(target_u, dtype=tf.float64), [-1])
        expected_suffix = self.dimension - axis - 1
        if suffix_values.shape != (targets.shape[0], expected_suffix):
            raise ValueError(
                f"suffixes: {HighDimStatus.INVALID_SHAPE.value}"
            )
        if not bool(
            tf.reduce_all(tf.math.is_finite(suffix_values)).numpy()
            and tf.reduce_all(tf.math.is_finite(targets)).numpy()
        ):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        tolerance = tf.cast(self.cdf_config.bracket_tolerance, tf.float64)
        if not bool(
            tf.reduce_all((targets >= -tolerance) & (targets <= 1.0 + tolerance)).numpy()
        ):
            raise ValueError(HighDimStatus.INVERSE_BRACKET_FAILURE.value)
        grid, cdf_grid, _ = self._suffix_cdf_grid_batch(axis, suffix_values)
        lo = tf.fill(tf.shape(targets), grid[0])
        hi = tf.fill(tf.shape(targets), grid[-1])
        mid = 0.5 * (lo + hi)
        for _ in range(self.cdf_config.bisection_steps):
            mid = 0.5 * (lo + hi)
            cdf_mid = _interp_rows(mid, grid, cdf_grid)
            choose_right = cdf_mid < targets
            lo = tf.where(choose_right, mid, lo)
            hi = tf.where(choose_right, hi, mid)
        return self._axis_reference_to_domain(axis, mid)

    def _cdf_at_suffix(
        self,
        axis: int,
        suffixes: tf.Tensor,
        z_value: tf.Tensor,
    ):
        grid, cdf_grid, conditional_grid = self._suffix_cdf_grid_batch(
            axis, suffixes
        )
        reference_value = self._axis_domain_to_reference(axis, z_value)
        cdf_value = _interp_rows(reference_value, grid, cdf_grid)
        right = tf.searchsorted(grid, tf.reshape(reference_value, [-1]), side="right")
        right = tf.clip_by_value(right, 1, tf.shape(grid)[0] - 1)
        left = right - 1
        rows = tf.range(tf.shape(suffixes)[0])
        y0 = tf.gather_nd(conditional_grid, tf.stack([rows, left], axis=1))
        y1 = tf.gather_nd(conditional_grid, tf.stack([rows, right], axis=1))
        x0 = tf.gather(grid, left)
        x1 = tf.gather(grid, right)
        conditional = y0 + (reference_value - x0) / (x1 - x0) * (y1 - y0)
        density = conditional * self._axis_domain_to_reference_jacobian(axis, z_value)
        return cdf_value, density, HighDimStatus.OK, {}

    def _suffix_cdf_grid_batch(
        self,
        axis: int,
        suffix_values: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        suffix = tf.convert_to_tensor(suffix_values, dtype=tf.float64)
        sample_count = int(suffix.shape[0])
        grid = self._axis_grid(axis)
        working_set = self.batch_working_set_estimate(
            axis=axis,
            sample_count=sample_count,
        )
        if working_set["estimated_bytes"] > self.cdf_config.max_batch_working_bytes:
            raise ValueError(
                "KR batch working set exceeds max_batch_working_bytes: "
                f"estimated={working_set['estimated_bytes']}, "
                f"budget={self.cdf_config.max_batch_working_bytes}, "
                f"axis={axis}, samples={sample_count}, grid={self.cdf_config.grid_size}"
            )
        physical_grid = self._axis_reference_to_domain(axis, grid)
        tiled_grid = tf.broadcast_to(
            physical_grid[tf.newaxis, :, tf.newaxis],
            [sample_count, tf.shape(grid)[0], 1],
        )
        tiled_suffix = tf.broadcast_to(
            suffix[:, tf.newaxis, :],
            [sample_count, tf.shape(grid)[0], tf.shape(suffix)[1]],
        )
        points = tf.reshape(
            tf.concat([tiled_grid, tiled_suffix], axis=2),
            [sample_count * tf.shape(grid)[0], self.dimension - axis],
        )
        retained_axes = tuple(range(axis, self.dimension))
        numerator = tf.reshape(
            self.density.normalized_marginal_density_values(retained_axes, points),
            [sample_count, tf.shape(grid)[0]],
        )
        if axis == self.dimension - 1:
            denominator = tf.ones([sample_count], dtype=tf.float64)
        else:
            denominator = self.density.normalized_marginal_density_values(
                tuple(range(axis + 1, self.dimension)), suffix
            )
        if not bool(
            tf.reduce_all(tf.math.is_finite(denominator)).numpy()
            and tf.reduce_all(denominator > self.density.denominator_floor).numpy()
        ):
            raise ValueError(
                HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED.value
            )
        conditional = numerator / denominator[:, tf.newaxis]
        increments = 0.5 * (conditional[:, 1:] + conditional[:, :-1]) * (
            grid[1:] - grid[:-1]
        )[tf.newaxis, :]
        cdf_grid = tf.concat(
            [
                tf.zeros([sample_count, 1], dtype=tf.float64),
                tf.cumsum(increments, axis=1),
            ],
            axis=1,
        )
        totals = cdf_grid[:, -1]
        if not bool(
            tf.reduce_all(tf.math.is_finite(cdf_grid)).numpy()
            and tf.reduce_all(totals > self.cdf_config.denominator_floor).numpy()
        ):
            raise ValueError(
                HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED.value
            )
        normalized_conditional = conditional / totals[:, tf.newaxis]
        cdf_grid = cdf_grid / totals[:, tf.newaxis]
        if bool(
            (
                tf.reduce_min(cdf_grid[:, 1:] - cdf_grid[:, :-1])
                < -self.cdf_config.monotonicity_tolerance
            ).numpy()
        ):
            raise ValueError(HighDimStatus.CDF_MONOTONICITY_FAILURE.value)
        return grid, cdf_grid, normalized_conditional

    def _conditional_cdf_grid_batch(
        self,
        axis: int,
        prefixes: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        prefix_values = tf.convert_to_tensor(prefixes, dtype=tf.float64)
        sample_count = int(prefix_values.shape[0])
        grid = self._axis_grid(axis)
        grid_size = self.cdf_config.grid_size
        working_set = self.batch_working_set_estimate(
            axis=axis,
            sample_count=sample_count,
        )
        if working_set["estimated_bytes"] > self.cdf_config.max_batch_working_bytes:
            raise ValueError(
                "KR batch working set exceeds max_batch_working_bytes: "
                f"estimated={working_set['estimated_bytes']}, "
                f"budget={self.cdf_config.max_batch_working_bytes}, "
                f"axis={axis}, samples={sample_count}, grid={grid_size}"
            )
        tiled_prefix = tf.repeat(
            prefix_values[:, tf.newaxis, :], repeats=grid_size, axis=1
        )
        physical_grid = self._axis_reference_to_domain(axis, grid)
        tiled_grid = tf.broadcast_to(
            physical_grid[tf.newaxis, :, tf.newaxis],
            [sample_count, grid_size, 1],
        )
        points = tf.reshape(
            tf.concat([tiled_prefix, tiled_grid], axis=2),
            [sample_count * grid_size, axis + 1],
        )
        numerator = tf.reshape(
            self.density.normalized_marginal_density_values(
                tuple(range(axis + 1)), points
            ),
            [sample_count, grid_size],
        )
        if axis == 0:
            denominator = tf.ones([sample_count], dtype=tf.float64)
        else:
            denominator = self.density.normalized_marginal_density_values(
                tuple(range(axis)), prefix_values
            )
        if not bool(
            tf.reduce_all(tf.math.is_finite(denominator)).numpy()
            and tf.reduce_all(denominator > self.density.denominator_floor).numpy()
        ):
            raise ValueError(
                HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED.value
            )
        conditional = (
            numerator
            / denominator[:, tf.newaxis]
        )
        increments = 0.5 * (conditional[:, 1:] + conditional[:, :-1]) * (
            grid[1:] - grid[:-1]
        )[tf.newaxis, :]
        cdf_grid = tf.concat(
            [
                tf.zeros([sample_count, 1], dtype=tf.float64),
                tf.cumsum(increments, axis=1),
            ],
            axis=1,
        )
        totals = cdf_grid[:, -1]
        if not bool(
            tf.reduce_all(tf.math.is_finite(cdf_grid)).numpy()
            and tf.reduce_all(totals > self.cdf_config.denominator_floor).numpy()
        ):
            raise ValueError(
                HighDimStatus.CONDITIONAL_DENOMINATOR_FLOOR_EXCEEDED.value
            )
        cdf_grid = cdf_grid / totals[:, tf.newaxis]
        minimum_increment = tf.reduce_min(cdf_grid[:, 1:] - cdf_grid[:, :-1])
        if not bool(tf.math.is_finite(minimum_increment).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        if bool(
            (minimum_increment < -self.cdf_config.monotonicity_tolerance).numpy()
        ):
            raise ValueError(HighDimStatus.CDF_MONOTONICITY_FAILURE.value)
        return grid, cdf_grid

    def batch_working_set_estimate(
        self,
        *,
        axis: int,
        sample_count: int,
    ) -> Mapping[str, int]:
        """Conservatively estimate the FP64 batched conditional-grid workspace."""

        checked_axis = int(axis)
        checked_samples = int(sample_count)
        if checked_axis < 0 or checked_axis >= self.dimension:
            raise ValueError("axis outside transport dimension")
        if checked_samples < 1:
            raise ValueError("sample_count must be positive")
        ranks = [
            max(int(core.values.shape[0]), int(core.values.shape[2]))
            for core in self.density.sqrt_tt.cores
        ]
        max_rank = max(ranks, default=1)
        grid_size = int(self.cdf_config.grid_size)
        # Account for tiled prefix/points, marginal numerator/CDF buffers, and
        # paired-core rank contractions. The factor of two covers overlapping
        # TensorFlow temporaries during concatenation and contraction.
        active_extent = max(checked_axis + 1, self.dimension - checked_axis)
        scalar_slots = (
            checked_samples
            * grid_size
            * (2 * active_extent + 4 + 4 * max_rank * max_rank)
            + checked_samples * (3 * grid_size + active_extent + 8)
        )
        estimated_bytes = 2 * tf.float64.size * scalar_slots
        return {
            "axis": checked_axis,
            "sample_count": checked_samples,
            "grid_size": grid_size,
            "max_tt_rank": max_rank,
            "estimated_bytes": int(estimated_bytes),
            "budget_bytes": int(self.cdf_config.max_batch_working_bytes),
        }

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
                tf.reshape(self._axis_reference_to_domain(0, grid), [-1, 1]),
            )
            denominator = tf.constant(1.0, dtype=tf.float64)
        else:
            physical_axis = self._axis_reference_to_domain(axis, grid)
            prefix_axis_points = tf.concat(
                [
                    tf.tile(prefix_values, [tf.shape(grid)[0], 1]),
                    tf.reshape(physical_axis, [-1, 1]),
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
        if not bool(tf.reduce_all(tf.math.is_finite(conditional)).numpy()):
            raise ValueError(HighDimStatus.NONFINITE_VALUE.value)
        return conditional

    def _axis_reference_to_domain(self, axis: int, reference: tf.Tensor) -> tf.Tensor:
        basis = self.density.sqrt_tt.product_basis.bases[axis]
        values = tf.convert_to_tensor(reference, tf.float64)
        if hasattr(basis.domain, "from_reference"):
            return basis.domain.from_reference(values)
        return basis.domain.left + 0.5 * (values + 1.0) * basis.domain.length

    def _axis_domain_to_reference(self, axis: int, points: tf.Tensor) -> tf.Tensor:
        basis = self.density.sqrt_tt.product_basis.bases[axis]
        return basis.domain.to_reference(tf.convert_to_tensor(points, tf.float64))

    def _axis_domain_to_reference_jacobian(
        self, axis: int, points: tf.Tensor
    ) -> tf.Tensor:
        basis = self.density.sqrt_tt.product_basis.bases[axis]
        values = tf.convert_to_tensor(points, tf.float64)
        if hasattr(basis.domain, "domain_to_reference_log_density"):
            return tf.exp(basis.domain.domain_to_reference_log_density(values))
        return tf.ones_like(values) * (2.0 / basis.domain.length)

    def _axis_reference_measure_density(
        self, axis: int, points: tf.Tensor
    ) -> tf.Tensor:
        return (
            tf.constant(0.5, tf.float64)
            * self._axis_domain_to_reference_jacobian(axis, points)
        )

    def _reference_measure_density(self, points: tf.Tensor) -> tf.Tensor:
        density = tf.ones([tf.shape(points)[1]], dtype=tf.float64)
        for axis in range(self.dimension):
            density = density * self._axis_reference_measure_density(
                axis, points[axis]
            )
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
