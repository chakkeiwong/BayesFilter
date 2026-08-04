"""Basis functions for internal high-dimensional TT contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

import tensorflow as tf

from bayesfilter.highdim.diagnostics import (
    MassMeasure,
    MeasureConvention,
    assert_density_matches_mass,
    assert_tf_float64,
)


P85_AUTHOR_SIR_BASIS_DOMAIN_SOURCE_ANCHORS = (
    "docs/references.bib:703-710",
    "docs/plans/bayesfilter-highdim-nonlinear-filtering-paper-first-scholarship-p18-zhao-cui-source-support-ledger-2026-06-01.md:18-22",
    "docs/plans/bayesfilter-highdim-nonlinear-filtering-paper-first-scholarship-p10-zhao-cui-tt-paper-code-crosswalk-ledger-2026-05-30.md:16-30",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/README.md:28-41",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:43-55",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/ApproxBases.m:70-119",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/Polynomials/Lagrangep.m:12-52",
    "third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/Domains/AlgebraicMapping.m:5-43",
)

P85_LEGACY_DIAGNOSTIC_BASIS_DOMAIN_SOURCE_ANCHORS = (
    "bayesfilter/highdim/source_route.py:2262-2269",
    "bayesfilter/highdim/source_route.py:3421-3427",
    "tests/highdim/test_p59_author_sir_36d_target_fit.py:44-47",
)

P85_XLA_STATIC_FIELDS = (
    "basis_family",
    "basis_dim",
    "legendre_max_degree",
    "lagrangep_order",
    "lagrangep_num_elems",
    "domain_map_family",
    "domain_map_scale",
    "bounded_interval_left",
    "bounded_interval_right",
    "dimension",
    "basis_dim_tuple",
    "measure_convention",
    "dtype",
)

P85_BASIS_CONFIG_NONCLAIMS = (
    "no fitting evidence",
    "no posterior correctness claim",
    "no production readiness claim",
    "no HMC readiness claim",
    "no XLA performance claim",
)

P85_LEGACY_DIAGNOSTIC_CLASSIFICATION = "local_gap"
P85_LEGACY_DIAGNOSTIC_SUBTYPE = "diagnostic_legendre_route"
P85_AUTHOR_SIR_CLASSIFICATION = "source_faithful"
P85_AUTHOR_SIR_SUBTYPE = "sir_config"
P85_AUTHOR_SIR_LAGRANGEP_ORDER = 4
P85_AUTHOR_SIR_LAGRANGEP_NUM_ELEMS = 8
P85_AUTHOR_SIR_DEGREE_COMPARATOR_CLASSIFICATION = "extension_or_invention"
P85_AUTHOR_SIR_DEGREE_COMPARATOR_SUBTYPE = "setup_static_degree_comparator_config"


@dataclass(frozen=True)
class BoundedInterval:
    """Closed one-dimensional interval for bounded polynomial bases."""

    left: tf.Tensor
    right: tf.Tensor
    dtype: tf.DType = tf.float64

    def __init__(self, left, right, dtype: tf.DType = tf.float64) -> None:
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "left", tf.convert_to_tensor(left, dtype=dtype))
        object.__setattr__(self, "right", tf.convert_to_tensor(right, dtype=dtype))
        if self.dtype != tf.float64:
            raise ValueError("Phase-1 highdim bases require tf.float64")
        if not bool((self.right > self.left).numpy()):
            raise ValueError("right must be greater than left")

    @property
    def length(self) -> tf.Tensor:
        return self.right - self.left

    def to_reference(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=self.dtype)
        return 2.0 * (values - self.left) / self.length - 1.0

    def from_reference(self, reference_points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(reference_points, dtype=self.dtype)
        return self.left + 0.5 * (values + 1.0) * self.length

    def domain_to_reference_log_density(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=self.dtype)
        return tf.zeros_like(values) + tf.math.log(2.0 / self.length)

    def reference_to_domain_log_density(self, reference_points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(reference_points, dtype=self.dtype)
        return tf.zeros_like(values) + tf.math.log(self.length / 2.0)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "bounded_interval",
            "left": self.left,
            "right": self.right,
            "dtype": self.dtype.name,
        }


@dataclass(frozen=True)
class AlgebraicMap:
    """Algebraic map between unbounded physical values and reference ``[-1, 1]``."""

    scale: tf.Tensor
    dtype: tf.DType = tf.float64

    def __init__(self, scale=1.0, dtype: tf.DType = tf.float64) -> None:
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "scale", tf.convert_to_tensor(scale, dtype=dtype))
        if self.dtype != tf.float64:
            raise ValueError("P85 algebraic maps require tf.float64")
        if not bool((self.scale > 0.0).numpy()):
            raise ValueError("scale must be positive")

    def to_reference(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=self.dtype) / self.scale
        return values * tf.math.rsqrt(1.0 + tf.square(values))

    def from_reference(self, reference_points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(reference_points, dtype=self.dtype)
        clipped = tf.clip_by_value(values, tf.constant(-1.0 + 1e-12, self.dtype), tf.constant(1.0 - 1e-12, self.dtype))
        return clipped * tf.math.rsqrt(1.0 - tf.square(clipped)) * self.scale

    def domain_to_reference_log_density(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=self.dtype) / self.scale
        return -1.5 * tf.math.log1p(tf.square(values)) - tf.math.log(self.scale)

    def reference_to_domain_log_density(self, reference_points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(reference_points, dtype=self.dtype)
        clipped = tf.clip_by_value(values, tf.constant(-1.0 + 1e-12, self.dtype), tf.constant(1.0 - 1e-12, self.dtype))
        return -1.5 * tf.math.log(1.0 - tf.square(clipped)) + tf.math.log(self.scale)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "algebraic",
            "scale": self.scale,
            "dtype": self.dtype.name,
            "source_formula": "z=(x/scale)/sqrt(1+(x/scale)^2)",
        }


@dataclass(frozen=True)
class DomainMapSpec:
    """Setup-static domain mapping identity for P85 basis manifests."""

    family: str
    classification: str
    classification_subtype: str
    source_anchors: tuple[str, ...]
    left: float | None = None
    right: float | None = None
    scale: float | None = None
    dtype_name: str = "float64"

    def __post_init__(self) -> None:
        family = str(self.family)
        if family not in {"bounded_interval", "algebraic"}:
            raise ValueError("unsupported domain map family")
        if str(self.dtype_name) != "float64":
            raise ValueError("P85 domain map specs require float64")
        if family == "bounded_interval":
            if self.left is None or self.right is None:
                raise ValueError("bounded_interval requires left and right")
            if not float(self.right) > float(self.left):
                raise ValueError("right must be greater than left")
            object.__setattr__(self, "scale", None)
        if family == "algebraic":
            if self.scale is None:
                raise ValueError("algebraic map requires scale")
            if not float(self.scale) > 0.0:
                raise ValueError("scale must be positive")
            object.__setattr__(self, "left", None)
            object.__setattr__(self, "right", None)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "source_anchors", tuple(str(anchor) for anchor in self.source_anchors))

    @classmethod
    def bounded_interval(
        cls,
        left: float,
        right: float,
        *,
        classification: str,
        classification_subtype: str,
        source_anchors: Sequence[str],
    ) -> "DomainMapSpec":
        return cls(
            family="bounded_interval",
            classification=classification,
            classification_subtype=classification_subtype,
            source_anchors=tuple(source_anchors),
            left=float(left),
            right=float(right),
        )

    @classmethod
    def algebraic(
        cls,
        scale: float,
        *,
        classification: str,
        classification_subtype: str,
        source_anchors: Sequence[str],
    ) -> "DomainMapSpec":
        return cls(
            family="algebraic",
            classification=classification,
            classification_subtype=classification_subtype,
            source_anchors=tuple(source_anchors),
            scale=float(scale),
        )

    def build_domain_map(self) -> BoundedInterval | AlgebraicMap:
        if self.family == "bounded_interval":
            return BoundedInterval(float(self.left), float(self.right))
        return AlgebraicMap(float(self.scale))

    def manifest_payload(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "family": self.family,
            "classification": self.classification,
            "classification_subtype": self.classification_subtype,
            "source_anchors": self.source_anchors,
            "dtype": self.dtype_name,
        }
        if self.family == "bounded_interval":
            payload.update({"left": float(self.left), "right": float(self.right)})
        else:
            payload.update({"scale": float(self.scale)})
        return payload


@dataclass(frozen=True)
class UniformReferenceMeasure:
    """Uniform probability reference measure on a bounded interval."""

    domain: BoundedInterval

    def __post_init__(self) -> None:
        if not isinstance(self.domain, BoundedInterval):
            raise TypeError("domain must be a BoundedInterval")


@dataclass(frozen=True)
class LegendreBasis1D:
    """Normalized Legendre basis under the uniform probability measure."""

    domain: BoundedInterval
    max_degree: int
    normalized: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.domain, BoundedInterval):
            raise TypeError("domain must be a BoundedInterval")
        if int(self.max_degree) < 0:
            raise ValueError("max_degree must be nonnegative")
        if not self.normalized:
            raise ValueError("Phase-1 pins normalized Legendre bases only")

    @property
    def basis_dim(self) -> int:
        return int(self.max_degree) + 1

    @property
    def dtype(self) -> tf.DType:
        return self.domain.dtype

    @property
    def reference_measure(self) -> UniformReferenceMeasure:
        return UniformReferenceMeasure(self.domain)

    def evaluate(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        xi = self.domain.to_reference(values)
        polys = _legendre_values(xi, self.max_degree)
        scales = tf.sqrt(
            tf.cast(2 * tf.range(self.basis_dim, dtype=tf.int32) + 1, tf.float64)
        )
        return polys * scales

    def derivative(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        xi = self.domain.to_reference(values)
        derivs = _legendre_reference_derivatives(xi, self.max_degree)
        scales = tf.sqrt(
            tf.cast(2 * tf.range(self.basis_dim, dtype=tf.int32) + 1, tf.float64)
        )
        return derivs * scales * (2.0 / self.domain.length)

    def mass_matrix(self, measure: MassMeasure) -> tf.Tensor:
        if not isinstance(measure, MassMeasure):
            raise TypeError("measure must be a MassMeasure")
        identity = tf.eye(self.basis_dim, dtype=tf.float64)
        if measure is MassMeasure.REFERENCE_MEASURE:
            return identity
        return identity * self.domain.length

    def integral_vector(self, measure: MassMeasure) -> tf.Tensor:
        if not isinstance(measure, MassMeasure):
            raise TypeError("measure must be a MassMeasure")
        first = tf.concat(
            [
                tf.ones([1], dtype=tf.float64),
                tf.zeros([self.basis_dim - 1], dtype=tf.float64),
            ],
            axis=0,
        )
        if measure is MassMeasure.REFERENCE_MEASURE:
            return first
        return first * self.domain.length

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "legendre",
            "basis_dim": self.basis_dim,
            "domain_map": self.domain.manifest_payload(),
            "max_degree": int(self.max_degree),
            "normalized": bool(self.normalized),
            "dtype": self.dtype.name,
            "reference_measure": "UniformReferenceMeasure",
        }


@dataclass(frozen=True)
class LagrangePiecewiseBasis1D:
    """BayesFilter-owned piecewise Lagrange basis matching author cardinality."""

    domain: BoundedInterval | AlgebraicMap
    order: int
    num_elems: int

    def __post_init__(self) -> None:
        if not isinstance(self.domain, (BoundedInterval, AlgebraicMap)):
            raise TypeError("domain must be a BoundedInterval or AlgebraicMap")
        if int(self.order) < 1:
            raise ValueError("order must be positive")
        if int(self.num_elems) < 1:
            raise ValueError("num_elems must be positive")
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "num_elems", int(self.num_elems))

    @property
    def basis_dim(self) -> int:
        return int(self.num_elems) * int(self.order) + 1

    @property
    def dtype(self) -> tf.DType:
        return self.domain.dtype

    @property
    def reference_nodes(self) -> tf.Tensor:
        return _lagrangep_reference_nodes(self.order, self.num_elems)

    def evaluate(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=self.dtype)
        reference = self.domain.to_reference(values)
        flat = tf.reshape(reference, [-1])
        basis = _lagrangep_cardinal_values(flat, self.order, self.num_elems)
        return tf.reshape(basis, tf.concat([tf.shape(reference), [self.basis_dim]], axis=0))

    def derivative(self, points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=self.dtype)
        reference = self.domain.to_reference(values)
        flat = tf.reshape(reference, [-1])
        deriv = _lagrangep_cardinal_derivatives(flat, self.order, self.num_elems)
        if isinstance(self.domain, BoundedInterval):
            chain = tf.ones_like(flat) * (2.0 / self.domain.length)
        else:
            scaled = tf.reshape(values, [-1]) / self.domain.scale
            chain = tf.pow(1.0 + tf.square(scaled), -1.5) / self.domain.scale
        deriv = deriv * chain[:, tf.newaxis]
        return tf.reshape(deriv, tf.concat([tf.shape(reference), [self.basis_dim]], axis=0))

    def mass_matrix(self, measure: MassMeasure) -> tf.Tensor:
        if not isinstance(measure, MassMeasure):
            raise TypeError("measure must be a MassMeasure")
        reference_lebesgue_mass, _ = _lagrangep_reference_mass_and_integral(
            self.order,
            self.num_elems,
        )
        if measure is MassMeasure.REFERENCE_MEASURE:
            return reference_lebesgue_mass / 2.0
        if isinstance(self.domain, BoundedInterval):
            return reference_lebesgue_mass * (self.domain.length / 2.0)
        return reference_lebesgue_mass

    def integral_vector(self, measure: MassMeasure) -> tf.Tensor:
        if not isinstance(measure, MassMeasure):
            raise TypeError("measure must be a MassMeasure")
        _, reference_lebesgue_integral = _lagrangep_reference_mass_and_integral(
            self.order,
            self.num_elems,
        )
        if measure is MassMeasure.REFERENCE_MEASURE:
            return reference_lebesgue_integral / 2.0
        if isinstance(self.domain, BoundedInterval):
            return reference_lebesgue_integral * (self.domain.length / 2.0)
        return reference_lebesgue_integral

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "lagrangep",
            "basis_dim": self.basis_dim,
            "domain_map": self.domain.manifest_payload(),
            "order": int(self.order),
            "num_elems": int(self.num_elems),
            "dtype": self.dtype.name,
            "implementation_note": "BayesFilter-owned P85 cardinal basis; no third-party code copied",
        }


@runtime_checkable
class Basis1DProtocol(Protocol):
    @property
    def basis_dim(self) -> int: ...

    @property
    def dtype(self) -> tf.DType: ...

    def evaluate(self, points: tf.Tensor) -> tf.Tensor: ...

    def mass_matrix(self, measure: MassMeasure) -> tf.Tensor: ...

    def integral_vector(self, measure: MassMeasure) -> tf.Tensor: ...

    def manifest_payload(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class BasisSpec:
    """Setup-static one-dimensional basis identity for P85 manifests."""

    family: str
    domain_map: DomainMapSpec
    classification: str
    classification_subtype: str
    source_anchors: tuple[str, ...]
    max_degree: int | None = None
    normalized: bool | None = None
    order: int | None = None
    num_elems: int | None = None
    dtype_name: str = "float64"

    def __post_init__(self) -> None:
        family = str(self.family)
        if family not in {"legendre", "lagrangep"}:
            raise ValueError("unsupported basis family")
        if str(self.dtype_name) != "float64":
            raise ValueError("P85 basis specs require float64")
        if not isinstance(self.domain_map, DomainMapSpec):
            raise TypeError("domain_map must be a DomainMapSpec")
        if family == "legendre":
            if self.max_degree is None or int(self.max_degree) < 0:
                raise ValueError("legendre requires nonnegative max_degree")
            object.__setattr__(self, "order", None)
            object.__setattr__(self, "num_elems", None)
            object.__setattr__(self, "normalized", True if self.normalized is None else bool(self.normalized))
        if family == "lagrangep":
            if self.order is None or int(self.order) < 1:
                raise ValueError("lagrangep requires positive order")
            if self.num_elems is None or int(self.num_elems) < 1:
                raise ValueError("lagrangep requires positive num_elems")
            object.__setattr__(self, "max_degree", None)
            object.__setattr__(self, "normalized", None)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "source_anchors", tuple(str(anchor) for anchor in self.source_anchors))

    @property
    def basis_dim(self) -> int:
        if self.family == "legendre":
            return int(self.max_degree) + 1
        return int(self.num_elems) * int(self.order) + 1

    def build_basis(self) -> LegendreBasis1D | LagrangePiecewiseBasis1D:
        domain = self.domain_map.build_domain_map()
        if self.family == "legendre":
            if not isinstance(domain, BoundedInterval):
                raise ValueError("legendre Phase-4 builder requires a bounded interval")
            return LegendreBasis1D(domain, int(self.max_degree), bool(self.normalized))
        return LagrangePiecewiseBasis1D(domain, int(self.order), int(self.num_elems))

    def manifest_payload(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "family": self.family,
            "basis_dim": self.basis_dim,
            "domain_map": self.domain_map.manifest_payload(),
            "classification": self.classification,
            "classification_subtype": self.classification_subtype,
            "source_anchors": self.source_anchors,
            "dtype": self.dtype_name,
        }
        if self.family == "legendre":
            payload.update({"max_degree": int(self.max_degree), "normalized": bool(self.normalized)})
        else:
            payload.update({"order": int(self.order), "num_elems": int(self.num_elems)})
        return payload


@dataclass(frozen=True)
class ProductBasis:
    """Tensor-product collection of one-dimensional bases."""

    bases: tuple[Basis1DProtocol, ...]
    convention: MeasureConvention

    def __init__(
        self,
        bases: Sequence[Basis1DProtocol],
        convention: MeasureConvention,
    ) -> None:
        object.__setattr__(self, "bases", tuple(bases))
        object.__setattr__(self, "convention", convention)
        if not self.bases:
            raise ValueError("ProductBasis requires at least one basis")
        for basis in self.bases:
            _validate_basis_protocol(basis)
        assert_density_matches_mass(convention)

    @property
    def dimension(self) -> int:
        return len(self.bases)

    def basis_dim_tuple(self) -> tuple[int, ...]:
        return tuple(basis.basis_dim for basis in self.bases)

    def evaluate_axis(self, axis: int, points: tf.Tensor) -> tf.Tensor:
        if axis < 0 or axis >= self.dimension:
            raise IndexError("axis out of range")
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        assert_tf_float64("points", values)
        return self.bases[axis].evaluate(values)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "ProductBasis",
            "dimension": self.dimension,
            "basis_dim_tuple": self.basis_dim_tuple(),
            "convention": {
                "density_measure": self.convention.density_measure.value,
                "mass_measure": self.convention.mass_measure.value,
                "reference_weight_name": self.convention.reference_weight_name,
                "physical_coordinate_name": self.convention.physical_coordinate_name,
                "reference_coordinate_name": self.convention.reference_coordinate_name,
                "dtype_name": self.convention.dtype_name,
            },
            "bases": tuple(basis.manifest_payload() for basis in self.bases),
        }


@dataclass(frozen=True)
class ProductBasisSpec:
    """Setup-static tensor-product basis identity for P85 manifests."""

    dimension: int
    axis_specs: tuple[BasisSpec, ...]
    convention: MeasureConvention
    classification: str
    classification_subtype: str
    source_anchors: tuple[str, ...]
    replicated: bool = False

    def __post_init__(self) -> None:
        if int(self.dimension) < 1:
            raise ValueError("dimension must be positive")
        if not self.axis_specs:
            raise ValueError("axis_specs must be nonempty")
        if self.replicated:
            if len(self.axis_specs) != 1:
                raise ValueError("replicated ProductBasisSpec requires one axis spec")
        elif len(self.axis_specs) != int(self.dimension):
            raise ValueError("axis_specs length must equal dimension")
        if not isinstance(self.convention, MeasureConvention):
            raise TypeError("convention must be a MeasureConvention")
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(self, "axis_specs", tuple(self.axis_specs))
        object.__setattr__(self, "source_anchors", tuple(str(anchor) for anchor in self.source_anchors))

    @property
    def expanded_axis_specs(self) -> tuple[BasisSpec, ...]:
        if self.replicated:
            return tuple(self.axis_specs[0] for _ in range(self.dimension))
        return self.axis_specs

    def basis_dim_tuple(self) -> tuple[int, ...]:
        return tuple(spec.basis_dim for spec in self.expanded_axis_specs)

    def build_product_basis(self) -> ProductBasis:
        return ProductBasis([spec.build_basis() for spec in self.expanded_axis_specs], self.convention)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "basis_config_version": "basis_config.v1",
            "family": "ProductBasisSpec",
            "dimension": self.dimension,
            "replicated": bool(self.replicated),
            "basis_dim_tuple": self.basis_dim_tuple(),
            "basis_family": tuple(spec.family for spec in self.expanded_axis_specs),
            "domain_map_family": tuple(spec.domain_map.family for spec in self.expanded_axis_specs),
            "classification": self.classification,
            "classification_subtype": self.classification_subtype,
            "source_anchors": self.source_anchors,
            "xla_static_fields": P85_XLA_STATIC_FIELDS,
            "nonclaims": P85_BASIS_CONFIG_NONCLAIMS,
            "axis_specs": tuple(spec.manifest_payload() for spec in self.expanded_axis_specs),
            "convention": {
                "density_measure": self.convention.density_measure.value,
                "mass_measure": self.convention.mass_measure.value,
                "reference_weight_name": self.convention.reference_weight_name,
                "physical_coordinate_name": self.convention.physical_coordinate_name,
                "reference_coordinate_name": self.convention.reference_coordinate_name,
                "dtype_name": self.convention.dtype_name,
            },
        }


def p85_legacy_legendre_product_basis_spec(
    *,
    dimension: int,
    fit_degree: int,
    convention: MeasureConvention,
) -> ProductBasisSpec:
    domain = DomainMapSpec.bounded_interval(
        -1.0,
        1.0,
        classification=P85_LEGACY_DIAGNOSTIC_CLASSIFICATION,
        classification_subtype=P85_LEGACY_DIAGNOSTIC_SUBTYPE,
        source_anchors=P85_LEGACY_DIAGNOSTIC_BASIS_DOMAIN_SOURCE_ANCHORS,
    )
    basis = BasisSpec(
        family="legendre",
        domain_map=domain,
        classification=P85_LEGACY_DIAGNOSTIC_CLASSIFICATION,
        classification_subtype=P85_LEGACY_DIAGNOSTIC_SUBTYPE,
        source_anchors=P85_LEGACY_DIAGNOSTIC_BASIS_DOMAIN_SOURCE_ANCHORS,
        max_degree=int(fit_degree),
        normalized=True,
    )
    return ProductBasisSpec(
        dimension=int(dimension),
        axis_specs=(basis,),
        convention=convention,
        classification=P85_LEGACY_DIAGNOSTIC_CLASSIFICATION,
        classification_subtype=P85_LEGACY_DIAGNOSTIC_SUBTYPE,
        source_anchors=P85_LEGACY_DIAGNOSTIC_BASIS_DOMAIN_SOURCE_ANCHORS,
        replicated=True,
    )


def p85_author_sir_lagrangep_algebraic_product_basis_spec(
    *,
    dimension: int = 36,
    convention: MeasureConvention,
    order: int = P85_AUTHOR_SIR_LAGRANGEP_ORDER,
    num_elems: int = P85_AUTHOR_SIR_LAGRANGEP_NUM_ELEMS,
) -> ProductBasisSpec:
    is_author_default = (
        int(order) == P85_AUTHOR_SIR_LAGRANGEP_ORDER
        and int(num_elems) == P85_AUTHOR_SIR_LAGRANGEP_NUM_ELEMS
    )
    classification = (
        P85_AUTHOR_SIR_CLASSIFICATION
        if is_author_default
        else P85_AUTHOR_SIR_DEGREE_COMPARATOR_CLASSIFICATION
    )
    classification_subtype = (
        P85_AUTHOR_SIR_SUBTYPE
        if is_author_default
        else P85_AUTHOR_SIR_DEGREE_COMPARATOR_SUBTYPE
    )
    domain = DomainMapSpec.algebraic(
        1.0,
        classification=classification,
        classification_subtype=classification_subtype,
        source_anchors=P85_AUTHOR_SIR_BASIS_DOMAIN_SOURCE_ANCHORS,
    )
    basis = BasisSpec(
        family="lagrangep",
        domain_map=domain,
        classification=classification,
        classification_subtype=classification_subtype,
        source_anchors=P85_AUTHOR_SIR_BASIS_DOMAIN_SOURCE_ANCHORS,
        order=int(order),
        num_elems=int(num_elems),
    )
    return ProductBasisSpec(
        dimension=int(dimension),
        axis_specs=(basis,),
        convention=convention,
        classification=classification,
        classification_subtype=classification_subtype,
        source_anchors=P85_AUTHOR_SIR_BASIS_DOMAIN_SOURCE_ANCHORS,
        replicated=True,
    )


def _legendre_values(xi: tf.Tensor, max_degree: int) -> tf.Tensor:
    xi = tf.convert_to_tensor(xi, dtype=tf.float64)
    flat = tf.reshape(xi, [-1])
    values = [tf.ones_like(flat)]
    if max_degree >= 1:
        values.append(flat)
    for n in range(1, max_degree):
        n_float = tf.cast(n, tf.float64)
        next_value = ((2.0 * n_float + 1.0) * flat * values[n] - n_float * values[n - 1]) / (
            n_float + 1.0
        )
        values.append(next_value)
    stacked = tf.stack(values, axis=-1)
    return tf.reshape(stacked, tf.concat([tf.shape(xi), [max_degree + 1]], axis=0))


def _legendre_reference_derivatives(xi: tf.Tensor, max_degree: int) -> tf.Tensor:
    xi = tf.convert_to_tensor(xi, dtype=tf.float64)
    flat = tf.reshape(xi, [-1])
    polys = tf.reshape(_legendre_values(flat, max_degree), [tf.shape(flat)[0], max_degree + 1])
    derivs = [tf.zeros_like(flat)]
    if max_degree >= 1:
        derivs.append(tf.ones_like(flat))
    for n in range(2, max_degree + 1):
        n_float = tf.cast(n, tf.float64)
        deriv = n_float * (polys[:, n - 1] + flat * derivs[n - 1]) - n_float * derivs[n - 2] / (
            n_float - 1.0
        )
        # Simpler and stable for low-degree tests: differentiate recurrence.
        prev_n = tf.cast(n - 1, tf.float64)
        deriv = (
            (2.0 * prev_n + 1.0) * (polys[:, n - 1] + flat * derivs[n - 1])
            - prev_n * derivs[n - 2]
        ) / (prev_n + 1.0)
        derivs.append(deriv)
    stacked = tf.stack(derivs, axis=-1)
    return tf.reshape(stacked, tf.concat([tf.shape(xi), [max_degree + 1]], axis=0))


def _validate_basis_protocol(basis: Basis1DProtocol) -> None:
    required = ("basis_dim", "dtype", "evaluate", "mass_matrix", "integral_vector", "manifest_payload")
    for name in required:
        if not hasattr(basis, name):
            raise TypeError(f"basis must provide {name}")
    if int(basis.basis_dim) < 1:
        raise ValueError("basis_dim must be positive")
    if basis.dtype != tf.float64:
        raise ValueError("ProductBasis requires tf.float64 bases")


def _lagrangep_reference_mass_and_integral(
    order: int,
    num_elems: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    local_dim = int(order) + 1
    basis_dim = int(num_elems) * int(order) + 1
    elem_size = tf.constant(2.0 / float(num_elems), dtype=tf.float64)
    local_mass, local_integral = _lagrange_ref_mass_and_integral(local_dim)
    mass = tf.zeros([basis_dim, basis_dim], dtype=tf.float64)
    integral = tf.zeros([basis_dim], dtype=tf.float64)
    local_cols = tf.range(local_dim, dtype=tf.int32)
    for elem in range(int(num_elems)):
        cols = local_cols + int(elem) * int(order)
        row_cols, col_cols = tf.meshgrid(cols, cols, indexing="ij")
        matrix_indices = tf.stack(
            [tf.reshape(row_cols, [-1]), tf.reshape(col_cols, [-1])],
            axis=1,
        )
        mass = tf.tensor_scatter_nd_add(
            mass,
            matrix_indices,
            tf.reshape(local_mass * elem_size, [-1]),
        )
        integral = tf.tensor_scatter_nd_add(
            integral,
            cols[:, tf.newaxis],
            local_integral * elem_size,
        )
    return 0.5 * (mass + tf.transpose(mass)), integral


def _lagrange_ref_mass_and_integral(num_points: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes = _lagrange_ref_nodes(int(num_points))
    degree = int(num_points) - 1
    powers = tf.cast(tf.range(degree + 1), dtype=tf.float64)
    vandermonde = tf.pow(nodes[:, tf.newaxis], powers[tf.newaxis, :])
    coefficients = tf.linalg.solve(
        vandermonde,
        tf.eye(int(num_points), dtype=tf.float64),
    )
    monomial_integral = 1.0 / tf.cast(tf.range(1, degree + 2), tf.float64)
    integral = tf.einsum("pj,p->j", coefficients, monomial_integral)
    denominator = 1.0 / (
        tf.cast(tf.range(degree + 1)[:, tf.newaxis], tf.float64)
        + tf.cast(tf.range(degree + 1)[tf.newaxis, :], tf.float64)
        + 1.0
    )
    mass = tf.einsum("pi,qj,pq->ij", coefficients, coefficients, denominator)
    return 0.5 * (mass + tf.transpose(mass)), integral


def _lagrange_cardinal_values(points: tf.Tensor, nodes: tf.Tensor) -> tf.Tensor:
    points = tf.convert_to_tensor(points, dtype=tf.float64)
    nodes = tf.convert_to_tensor(nodes, dtype=tf.float64)
    diffs = points[:, tf.newaxis] - nodes[tf.newaxis, :]
    exact = tf.abs(diffs) <= tf.constant(1e-12, dtype=tf.float64)
    safe_diffs = tf.where(exact, tf.ones_like(diffs), diffs)
    weights = _barycentric_weights(nodes)
    terms = weights[tf.newaxis, :] / safe_diffs
    values = terms / tf.reduce_sum(terms, axis=1, keepdims=True)
    exact_any = tf.reduce_any(exact, axis=1)
    exact_values = tf.cast(exact, tf.float64)
    return tf.where(exact_any[:, tf.newaxis], exact_values, values)


def _lagrange_cardinal_derivatives(points: tf.Tensor, nodes: tf.Tensor) -> tf.Tensor:
    points = tf.convert_to_tensor(points, dtype=tf.float64)
    nodes = tf.convert_to_tensor(nodes, dtype=tf.float64)
    diffs = points[:, tf.newaxis] - nodes[tf.newaxis, :]
    exact = tf.abs(diffs) <= tf.constant(1e-12, dtype=tf.float64)
    safe_diffs = tf.where(exact, tf.ones_like(diffs), diffs)
    weights = _barycentric_weights(nodes)
    terms = weights[tf.newaxis, :] / safe_diffs
    squared_terms = weights[tf.newaxis, :] / tf.square(safe_diffs)
    denominator = tf.reduce_sum(terms, axis=1, keepdims=True)
    numerator = tf.reduce_sum(squared_terms, axis=1, keepdims=True)
    derivatives = terms * numerator / tf.square(denominator) - squared_terms / denominator

    node_diffs = nodes[:, tf.newaxis] - nodes[tf.newaxis, :]
    eye = tf.eye(tf.shape(nodes)[0], dtype=tf.float64)
    safe_node_diffs = tf.where(eye > 0.0, tf.ones_like(node_diffs), node_diffs)
    weight_ratios = weights[tf.newaxis, :] / weights[:, tf.newaxis]
    off_diagonal = tf.where(eye > 0.0, tf.zeros_like(node_diffs), weight_ratios / safe_node_diffs)
    diagonal = -tf.reduce_sum(off_diagonal, axis=1)
    derivatives_at_nodes = off_diagonal + tf.linalg.diag(diagonal)

    exact_any = tf.reduce_any(exact, axis=1)
    exact_indices = tf.argmax(tf.cast(exact, tf.int32), axis=1)
    exact_derivatives = tf.gather(derivatives_at_nodes, exact_indices)
    return tf.where(exact_any[:, tf.newaxis], exact_derivatives, derivatives)


def _lagrangep_cardinal_values(points: tf.Tensor, order: int, num_elems: int) -> tf.Tensor:
    flat = tf.reshape(tf.convert_to_tensor(points, dtype=tf.float64), [-1])
    local_nodes = _lagrange_ref_nodes(int(order) + 1)
    local_values = _lagrange_cardinal_values(
        _lagrangep_local_coordinates(flat, int(num_elems)),
        local_nodes,
    )
    inside = _lagrangep_inside_mask(flat)
    local_values = tf.where(inside[:, tf.newaxis], local_values, tf.zeros_like(local_values))
    return _scatter_lagrangep_local_values(local_values, flat, int(order), int(num_elems))


def _lagrangep_cardinal_derivatives(points: tf.Tensor, order: int, num_elems: int) -> tf.Tensor:
    flat = tf.reshape(tf.convert_to_tensor(points, dtype=tf.float64), [-1])
    elem_size = tf.constant(2.0 / float(num_elems), dtype=tf.float64)
    local_nodes = _lagrange_ref_nodes(int(order) + 1)
    local_derivatives = _lagrange_cardinal_derivatives(
        _lagrangep_local_coordinates(flat, int(num_elems)),
        local_nodes,
    )
    reference_derivatives = local_derivatives / elem_size
    inside = _lagrangep_inside_mask(flat)
    reference_derivatives = tf.where(
        inside[:, tf.newaxis],
        reference_derivatives,
        tf.zeros_like(reference_derivatives),
    )
    return _scatter_lagrangep_local_values(reference_derivatives, flat, int(order), int(num_elems))


def _lagrangep_inside_mask(points: tf.Tensor) -> tf.Tensor:
    return tf.logical_and(points >= -1.0, points <= 1.0)


def _lagrangep_element_indices(points: tf.Tensor, num_elems: int) -> tf.Tensor:
    elem_size = tf.constant(2.0 / float(num_elems), dtype=tf.float64)
    raw = tf.cast(tf.math.ceil((points + 1.0) / elem_size), tf.int32) - 1
    return tf.clip_by_value(raw, 0, int(num_elems) - 1)


def _lagrangep_local_coordinates(points: tf.Tensor, num_elems: int) -> tf.Tensor:
    elem_size = tf.constant(2.0 / float(num_elems), dtype=tf.float64)
    element_indices = _lagrangep_element_indices(points, int(num_elems))
    left_edges = -1.0 + tf.cast(element_indices, tf.float64) * elem_size
    return (points - left_edges) / elem_size


def _scatter_lagrangep_local_values(
    local_values: tf.Tensor,
    points: tf.Tensor,
    order: int,
    num_elems: int,
) -> tf.Tensor:
    local_dim = int(order) + 1
    basis_dim = int(num_elems) * int(order) + 1
    element_indices = _lagrangep_element_indices(points, int(num_elems))
    rows = tf.repeat(tf.range(tf.shape(points)[0], dtype=tf.int32), local_dim)
    local_cols = tf.range(local_dim, dtype=tf.int32)
    cols = element_indices[:, tf.newaxis] * int(order) + local_cols[tf.newaxis, :]
    scatter_indices = tf.stack([rows, tf.reshape(cols, [-1])], axis=1)
    return tf.scatter_nd(
        scatter_indices,
        tf.reshape(local_values, [-1]),
        [tf.shape(points)[0], basis_dim],
    )


def _barycentric_weights(nodes: tf.Tensor) -> tf.Tensor:
    nodes = tf.convert_to_tensor(nodes, dtype=tf.float64)
    diffs = nodes[:, tf.newaxis] - nodes[tf.newaxis, :]
    eye = tf.eye(tf.shape(nodes)[0], dtype=tf.float64)
    safe = tf.where(eye > 0.0, tf.ones_like(diffs), diffs)
    return 1.0 / tf.reduce_prod(safe, axis=1)


def _lagrangep_reference_nodes(order: int, num_elems: int) -> tf.Tensor:
    local_nodes = _lagrange_ref_nodes(int(order) + 1)
    grid = tf.linspace(
        tf.constant(-1.0, dtype=tf.float64),
        tf.constant(1.0, dtype=tf.float64),
        int(num_elems) + 1,
    )
    elem_size = tf.constant(2.0 / float(num_elems), dtype=tf.float64)
    segments = []
    for elem in range(int(num_elems)):
        mapped = grid[elem] + local_nodes * elem_size
        if elem:
            mapped = mapped[1:]
        segments.append(mapped)
    return tf.concat(segments, axis=0)


def _lagrange_ref_nodes(num_points: int) -> tf.Tensor:
    if int(num_points) < 2:
        raise ValueError("num_points must be at least 2")
    endpoints = (
        tf.constant([0.0], dtype=tf.float64),
        tf.constant([1.0], dtype=tf.float64),
    )
    if int(num_points) == 2:
        return tf.concat(endpoints, axis=0)
    interior = 0.5 * (_jacobi11_nodes(int(num_points) - 3) + 1.0)
    return tf.concat([endpoints[0], interior, endpoints[1]], axis=0)


def _jacobi11_nodes(order: int) -> tf.Tensor:
    k = tf.cast(tf.range(int(order) + 1), tf.float64)
    a = (2.0 * k + 3.0) * (k + 2.0) / ((k + 1.0) * (k + 3.0))
    b = tf.zeros_like(k)
    c = (k + 2.0) / (k + 3.0)
    diagonal = -b / a
    off_diagonal = tf.sqrt(c[1:] / (a[:-1] * a[1:]))
    jacobi = tf.linalg.diag(diagonal)
    jacobi = jacobi + tf.linalg.diag(off_diagonal, k=1)
    jacobi = jacobi + tf.linalg.diag(off_diagonal, k=-1)
    eigenvalues = tf.linalg.eigvalsh(jacobi)
    return tf.sort(eigenvalues)
