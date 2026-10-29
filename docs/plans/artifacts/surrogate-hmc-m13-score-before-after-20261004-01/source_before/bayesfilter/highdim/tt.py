"""Functional tensor-train algebra for Phase-1 highdim contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import (
    HighDimStatus,
    HighDimValidationResult,
    MassMeasure,
    MeasureConvention,
    assert_density_matches_mass,
    assert_tf_float64,
    freeze_mapping,
)
from bayesfilter.highdim.fixed_branch import BranchIdentity, BranchManifest
from bayesfilter.highdim.validation import ComplexityBudget


@dataclass(frozen=True)
class TTCore:
    """One functional TT core with shape [rank_left, basis_dim, rank_right]."""

    values: tf.Tensor

    def __post_init__(self) -> None:
        values = tf.convert_to_tensor(self.values, dtype=tf.float64)
        if values.shape.rank != 3:
            raise ValueError(f"TTCore: {HighDimStatus.INVALID_SHAPE.value}")
        assert_tf_float64("TTCore.values", values)
        object.__setattr__(self, "values", values)

    @property
    def left_rank(self) -> int:
        return int(self.values.shape[0])

    @property
    def basis_dim(self) -> int:
        return int(self.values.shape[1])

    @property
    def right_rank(self) -> int:
        return int(self.values.shape[2])


@dataclass(frozen=True)
class TTContractedRepresentation:
    """Representation after integrating selected TT axes."""

    kept_axes: tuple[int, ...]
    integrated_axes: tuple[int, ...]
    cores: tuple[TTCore, ...]
    measure_convention: MeasureConvention
    branch_identity: BranchIdentity
    retained_bases: tuple[Mapping[str, object], ...] = ()
    diagnostics: Mapping[str, object] | None = None
    scalar_value: tf.Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))
        object.__setattr__(
            self,
            "retained_bases",
            tuple(freeze_mapping(basis) for basis in self.retained_bases),
        )
        if not isinstance(self.branch_identity, BranchIdentity):
            raise TypeError("branch_identity must be a BranchIdentity")
        if self.scalar_value is not None:
            scalar = tf.convert_to_tensor(self.scalar_value, dtype=tf.float64)
            if scalar.shape.rank != 0:
                raise ValueError(f"scalar_value: {HighDimStatus.INVALID_SHAPE.value}")
            object.__setattr__(self, "scalar_value", scalar)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "kept_axes": self.kept_axes,
            "integrated_axes": self.integrated_axes,
            "measure_convention": _measure_convention_payload(self.measure_convention),
            "retained_bases": self.retained_bases,
            "cores": tuple(core.values for core in self.cores),
            "scalar_value": self.scalar_value,
            "diagnostics": dict(self.diagnostics),
        }

    def manifest(self, version: str = "functional_tt_contracted.v1") -> BranchManifest:
        return BranchManifest(version=version, payload=self.manifest_payload())


@dataclass(frozen=True)
class FunctionalTT:
    """Fixed-rank functional tensor train over a product basis."""

    cores: tuple[TTCore, ...]
    product_basis: ProductBasis
    measure_convention: MeasureConvention
    complexity_budget: ComplexityBudget
    branch_identity: BranchIdentity | None = None

    def __init__(
        self,
        cores: Sequence[TTCore],
        product_basis: ProductBasis,
        measure_convention: MeasureConvention,
        complexity_budget: ComplexityBudget | None = None,
        branch_identity: BranchIdentity | None = None,
    ) -> None:
        object.__setattr__(self, "cores", tuple(cores))
        object.__setattr__(self, "product_basis", product_basis)
        object.__setattr__(self, "measure_convention", measure_convention)
        object.__setattr__(
            self,
            "complexity_budget",
            complexity_budget
            or ComplexityBudget(max_elements=10_000_000, max_bytes=80_000_000),
        )
        object.__setattr__(self, "branch_identity", branch_identity)
        self._validate()

    def _validate(self) -> None:
        if not self.cores:
            raise ValueError("FunctionalTT requires at least one core")
        if not isinstance(self.product_basis, ProductBasis):
            raise TypeError("product_basis must be a ProductBasis")
        if self.product_basis.dimension != len(self.cores):
            raise ValueError(f"FunctionalTT: {HighDimStatus.INVALID_SHAPE.value}")
        if self.measure_convention != self.product_basis.convention:
            raise ValueError(f"FunctionalTT: {HighDimStatus.MEASURE_MISMATCH.value}")
        if not isinstance(self.complexity_budget, ComplexityBudget):
            raise TypeError("complexity_budget must be a ComplexityBudget")
        assert_density_matches_mass(self.measure_convention)
        if self.cores[0].left_rank != 1 or self.cores[-1].right_rank != 1:
            raise ValueError(f"FunctionalTT: {HighDimStatus.INVALID_SHAPE.value}")
        for axis, (core, basis_dim) in enumerate(
            zip(self.cores, self.product_basis.basis_dim_tuple())
        ):
            if core.basis_dim != basis_dim:
                raise ValueError(f"core {axis}: {HighDimStatus.INVALID_SHAPE.value}")
            if axis + 1 < len(self.cores) and core.right_rank != self.cores[axis + 1].left_rank:
                raise ValueError(f"rank {axis}: {HighDimStatus.INVALID_SHAPE.value}")

    def rank_tuple(self) -> tuple[int, ...]:
        return tuple([self.cores[0].left_rank] + [core.right_rank for core in self.cores])

    def basis_dim_tuple(self) -> tuple[int, ...]:
        return tuple(core.basis_dim for core in self.cores)

    def _estimated_evaluate_elements(self, n_points: int) -> int:
        total = 0
        current_rank = 1
        for core in self.cores:
            total += n_points * core.basis_dim
            total += n_points * core.left_rank * core.right_rank
            total += n_points * current_rank
            total += n_points * core.right_rank
            current_rank = core.right_rank
        return total

    def _estimated_contract_elements(self) -> int:
        total = 0
        max_rank = 1
        for core in self.cores:
            total += core.left_rank * core.right_rank
            total += core.left_rank * core.basis_dim * core.right_rank
            max_rank = max(max_rank, core.left_rank, core.right_rank)
        total += max_rank * max_rank * len(self.cores)
        return total

    def _estimated_axis_contraction_elements(
        self,
        integrate_axes: Sequence[int],
    ) -> int:
        axes = set(int(axis) for axis in integrate_axes)
        total = 0
        pending_left = self.cores[0].left_rank
        pending_right = self.cores[0].left_rank
        retained: list[tuple[int, int, int]] = []
        for axis, core in enumerate(self.cores):
            if axis in axes:
                total += core.basis_dim
                total += core.left_rank * core.right_rank
                total += pending_left * pending_right * core.right_rank
                pending_right = core.right_rank
            else:
                total += pending_left * core.basis_dim * core.right_rank
                retained.append((pending_left, core.basis_dim, core.right_rank))
                pending_left = core.right_rank
                pending_right = core.right_rank
        if retained:
            last_left, last_basis, _ = retained[-1]
            total += last_left * last_basis * pending_right
        elif axes:
            total += pending_left * pending_right
        return total

    def evaluate(
        self,
        points: tf.Tensor,
        budget: ComplexityBudget | None = None,
    ) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        assert_tf_float64("points", values)
        if values.shape.rank != 2 or values.shape[1] != len(self.cores):
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        if values.shape[0] is None:
            raise ValueError(f"points: {HighDimStatus.INVALID_SHAPE.value}")
        active_budget = budget or self.complexity_budget
        result = active_budget.validate(
            estimated_elements=self._estimated_evaluate_elements(int(values.shape[0])),
            dtype_size=tf.float64.size,
        )
        if result.status is not HighDimStatus.OK:
            raise ValueError(HighDimStatus.COMPLEXITY_GATE.value)
        n_points = tf.shape(values)[0]
        vector = tf.ones([n_points, 1], dtype=tf.float64)
        for axis, core in enumerate(self.cores):
            basis_values = self.product_basis.evaluate_axis(axis, values[:, axis])
            matrices = tf.einsum("nl,alb->nab", basis_values, core.values)
            vector = tf.einsum("na,nab->nb", vector, matrices)
        return tf.reshape(vector, [n_points])

    def integrate_all(
        self,
        measure: MassMeasure | None = None,
        budget: ComplexityBudget | None = None,
    ) -> tf.Tensor:
        active_budget = budget or self.complexity_budget
        result = active_budget.validate(
            estimated_elements=self._estimated_contract_elements(),
            dtype_size=tf.float64.size,
        )
        if result.status is not HighDimStatus.OK:
            raise ValueError(HighDimStatus.COMPLEXITY_GATE.value)
        active_measure = measure or self.measure_convention.mass_measure
        vector = tf.ones([1], dtype=tf.float64)
        for axis, core in enumerate(self.cores):
            integrals = self.product_basis.bases[axis].integral_vector(active_measure)
            matrix = tf.einsum("l,alb->ab", integrals, core.values)
            vector = tf.einsum("a,ab->b", vector, matrix)
        return tf.reshape(vector, [])

    def contract_axes(
        self,
        integrate_axes: Sequence[int],
        budget: ComplexityBudget | None = None,
    ) -> TTContractedRepresentation:
        axes = tuple(sorted(set(int(axis) for axis in integrate_axes)))
        dimension = len(self.cores)
        if any(axis < 0 or axis >= dimension for axis in axes):
            raise IndexError("integrate_axes contains an out-of-range axis")
        active_budget = budget or self.complexity_budget
        result = active_budget.validate(
            estimated_elements=self._estimated_axis_contraction_elements(axes),
            dtype_size=tf.float64.size,
        )
        if result.status is not HighDimStatus.OK:
            raise ValueError(HighDimStatus.COMPLEXITY_GATE.value)
        kept_axes = tuple(axis for axis in range(dimension) if axis not in axes)
        active_measure = self.measure_convention.mass_measure
        pending = tf.eye(self.cores[0].left_rank, dtype=tf.float64)
        kept_core_values: list[tf.Tensor] = []
        for axis, core in enumerate(self.cores):
            if axis in axes:
                integrals = self.product_basis.bases[axis].integral_vector(active_measure)
                matrix = tf.einsum("l,alb->ab", integrals, core.values)
                pending = tf.matmul(pending, matrix)
            else:
                kept_core_values.append(tf.einsum("ea,alb->elb", pending, core.values))
                pending = tf.eye(core.right_rank, dtype=tf.float64)
        scalar_value = None
        if kept_core_values:
            kept_core_values[-1] = tf.einsum("elb,bc->elc", kept_core_values[-1], pending)
        elif axes:
            scalar_value = tf.reshape(pending, [])
        diagnostics = {
            "status": HighDimStatus.OK.value,
            "integrated_axes": axes,
            "kept_axes": kept_axes,
            "mass_measure": active_measure.value,
            "representation": "scalar" if scalar_value is not None else "tt",
        }
        kept_cores = tuple(TTCore(values) for values in kept_core_values)
        retained_bases = tuple(
            _basis_payload(self.product_basis.bases[axis]) for axis in kept_axes
        )
        draft = TTContractedRepresentation(
            kept_axes=kept_axes,
            integrated_axes=axes,
            cores=kept_cores,
            measure_convention=self.measure_convention,
            branch_identity=_empty_branch_identity(),
            retained_bases=retained_bases,
            diagnostics=diagnostics,
            scalar_value=scalar_value,
        )
        manifest = draft.manifest()
        branch_identity = BranchIdentity(manifest=manifest, hash=manifest.sha256())
        return TTContractedRepresentation(
            kept_axes=kept_axes,
            integrated_axes=axes,
            cores=kept_cores,
            measure_convention=self.measure_convention,
            branch_identity=branch_identity,
            retained_bases=retained_bases,
            diagnostics=diagnostics,
            scalar_value=scalar_value,
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "rank_tuple": self.rank_tuple(),
            "basis_dim_tuple": self.basis_dim_tuple(),
            "complexity_budget": {
                "max_elements": self.complexity_budget.max_elements,
                "max_bytes": self.complexity_budget.max_bytes,
            },
            "measure_convention": _measure_convention_payload(self.measure_convention),
            "bases": tuple(_basis_payload(basis) for basis in self.product_basis.bases),
            "cores": tuple(core.values for core in self.cores),
        }

    def manifest(self, version: str = "functional_tt.v1") -> BranchManifest:
        return BranchManifest(version=version, payload=self.manifest_payload())

    def validate_complexity(
        self,
        budget: ComplexityBudget,
        estimated_elements: int,
        dtype_size: int = 8,
    ) -> HighDimValidationResult:
        return budget.validate(estimated_elements=estimated_elements, dtype_size=dtype_size)


def _measure_convention_payload(convention: MeasureConvention) -> Mapping[str, object]:
    return {
        "density_measure": convention.density_measure.value,
        "mass_measure": convention.mass_measure.value,
        "reference_weight_name": convention.reference_weight_name,
        "physical_coordinate_name": convention.physical_coordinate_name,
        "reference_coordinate_name": convention.reference_coordinate_name,
        "dtype_name": convention.dtype_name,
    }


def _basis_payload(basis) -> Mapping[str, object]:
    if hasattr(basis, "manifest_payload"):
        payload = basis.manifest_payload()
        if not isinstance(payload, Mapping):
            raise TypeError("basis manifest_payload() must return a mapping")
        return payload
    return {
        "family": "LegendreBasis1D",
        "left": basis.domain.left,
        "right": basis.domain.right,
        "dtype": basis.dtype.name,
        "reference_measure": "UniformReferenceMeasure",
        "max_degree": basis.max_degree,
        "normalized": basis.normalized,
    }


def _empty_branch_identity() -> BranchIdentity:
    manifest = BranchManifest(version="empty.v1", payload={"status": "placeholder"})
    return BranchIdentity(manifest=manifest, hash=manifest.sha256())
