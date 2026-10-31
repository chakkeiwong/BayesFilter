"""Phase-0 audit and reconstruction for the committed P88 fixed variant.

This module deliberately stops at the exact evidence boundary.  P88 serializes
the T1 squared-TT density, but a density payload is not silently promoted to a
complete TTSIRT transport or a T2/T20 sequential filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.bases import (
    p85_author_sir_lagrangep_algebraic_product_basis_spec,
)
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
)
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.tt import FunctionalTT, TTCore


P88_ARTIFACT_RELATIVE_PATH = (
    "docs/plans/bayesfilter-highdim-zhao-cui-p88-phase2-degree-order3-rank4-"
    "lr3e-4-l1-0-fit-2026-06-27.json"
)
P88_ARTIFACT_SHA256 = "ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e"
P88_GIT_INTRODUCTION = "c815edc52162779e969b2982723b2f52770fd849"
P88_FIT_SCRIPT_FIRST_COMMIT = "9bc5a658bfaac29987438a50aea4bf7e9036719f"
P88_STATUS = "P88_PHASE2_DEGREE_ORDER3_RANK4_CANDIDATE_TRAINING_BASE_COMPLETED"
P88_TARGET_ID = "zhao_cui_sir_austria_d18"
P88_TRAINING_BACKEND = "training_base_optimizer"
P88_ROUTE_CLASSIFICATION = "extension_or_invention"
P88_DIMENSION = 36
P88_CORE_COUNT = 36
P88_BASIS_ORDER = 3
P88_BASIS_NUM_ELEMS = 8
P88_BASIS_DIM = 25
P88_RANK = 4
P88_TAU = 1.0e-8
P88_NORMALIZER_FLOOR = 1.0e-12
P88_DENOMINATOR_FLOOR = 1.0e-12
P88_TIME_INDEX = 1
P88_MEASURE_CONVENTION = MeasureConvention(
    density_measure=DensityMeasure.REFERENCE_MEASURE,
    mass_measure=MassMeasure.REFERENCE_MEASURE,
    reference_weight_name="omega",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _json_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _core_values(record: Mapping[str, object], axis: int) -> tf.Tensor:
    if int(record.get("axis", -1)) != int(axis):
        raise ValueError(f"P88 core axis mismatch at axis {axis}")
    if str(record.get("dtype")) != "float64":
        raise ValueError(f"P88 core dtype mismatch at axis {axis}")
    shape = tuple(int(value) for value in record.get("shape", ()))
    values = tf.convert_to_tensor(record.get("values"), dtype=tf.float64)
    if tuple(int(value) for value in values.shape) != shape:
        raise ValueError(f"P88 core shape mismatch at axis {axis}")
    if _sha256_tensor(values) != str(record.get("sha256")):
        raise ValueError(f"P88 core hash mismatch at axis {axis}")
    return values


@dataclass(frozen=True)
class P88Phase0Audit:
    """Exact P88 metadata and density reconstruction result."""

    artifact_path: str
    artifact_sha256: str
    artifact: Mapping[str, object]
    cores: tuple[tf.Tensor, ...]
    density: SquaredTTDensity
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.cores) != P88_CORE_COUNT:
            raise ValueError("P88 audit requires exactly 36 cores")
        if self.artifact_sha256 != P88_ARTIFACT_SHA256:
            raise ValueError("P88 artifact SHA-256 mismatch")
        if not isinstance(self.artifact, Mapping):
            raise TypeError("artifact must be a mapping")
        object.__setattr__(self, "artifact", MappingProxyType(dict(self.artifact)))
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))

    @property
    def status(self) -> str:
        return (
            "BLOCK_FIXED_VARIANT_BASELINE_NOT_RECONSTRUCTIBLE"
            if self.blockers
            else "PASS_FIXED_VARIANT_T1_DENSITY_RECONSTRUCTED"
        )

    @property
    def t2_boundary_status(self) -> str:
        return (
            "BLOCK_T2_BOUNDARY_METADATA_MISSING"
            if "missing_explicit_transport_branch_metadata" in self.blockers
            else "NOT_EVALUATED"
        )

    def manifest_payload(self) -> Mapping[str, object]:
        serialization = _json_mapping(
            self.artifact.get("trained_core_serialization"),
            "trained_core_serialization",
        )
        config = _json_mapping(self.artifact.get("training_config"), "training_config")
        route = _json_mapping(self.artifact.get("route_manifest"), "route_manifest")
        return {
            "schema": "bayesfilter.zhao_cui.austria_sir.phase0_p88_audit.v1",
            "status": self.status,
            "t2_boundary_status": self.t2_boundary_status,
            "blockers": self.blockers,
            "p88_artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "git_introduction": P88_GIT_INTRODUCTION,
            "fit_script_first_commit": P88_FIT_SCRIPT_FIRST_COMMIT,
            "p88_status": self.artifact.get("status"),
            "target_id": route.get("target_id"),
            "time_index": P88_TIME_INDEX,
            "dimension": P88_DIMENSION,
            "core_count": serialization.get("core_count"),
            "basis_order": route.get("basis_order"),
            "basis_num_elems": route.get("basis_num_elems"),
            "basis_dim": tuple(config.get("basis_dim_tuple", ())),
            "ranks": tuple(config.get("ranks", ())),
            "training_backend": self.artifact.get("training_backend"),
            "measure_convention": config.get("measure_convention"),
            "defensive_tau": config.get("defensive_tau"),
            "normalizer_floor": config.get("normalizer_floor"),
            "denominator_floor": config.get("denominator_floor"),
            "density_normalizer": self.density.normalizer(),
            "sqrt_square_normalizer": self.density.sqrt_square_normalizer(),
            "stored_density_normalizer": self.artifact.get("normalizer"),
            "stored_sqrt_square_normalizer": self.artifact.get(
                "sqrt_square_normalizer"
            ),
            "density_branch_hash": self.density.branch_identity.hash.value,
            "source_observation_hash_binding": "absent_in_p88_artifact",
            "transport_branch_metadata": "absent_in_p88_artifact",
            "source_dependency_closure": "absent_in_p88_artifact",
            "source_relation": {
                "p88_trainer": "extension_or_invention",
                "squared_tt_density": "source_faithful_mathematical_operation",
                "t2_transport": "not_checked",
            },
        }


def _validate_p88_metadata(payload: Mapping[str, object]) -> list[str]:
    blockers: list[str] = []
    if payload.get("status") != P88_STATUS:
        blockers.append("p88_status_mismatch")
    route = _json_mapping(payload.get("route_manifest"), "route_manifest")
    if route.get("target_id") != P88_TARGET_ID:
        blockers.append("p88_target_id_mismatch")
    if payload.get("training_backend") != P88_TRAINING_BACKEND:
        blockers.append("p88_training_backend_mismatch")
    if payload.get("time_index", P88_TIME_INDEX) != P88_TIME_INDEX:
        blockers.append("p88_time_index_mismatch")
    serialization = _json_mapping(
        payload.get("trained_core_serialization"),
        "trained_core_serialization",
    )
    if serialization.get("status") != "serialized_with_values":
        blockers.append("serialized_core_values_missing")
    if serialization.get("core_count") != P88_CORE_COUNT:
        blockers.append("serialized_core_count_mismatch")
    config = _json_mapping(payload.get("training_config"), "training_config")
    if config.get("dimension") != P88_DIMENSION:
        blockers.append("p88_dimension_mismatch")
    if route.get("basis_order") != P88_BASIS_ORDER:
        blockers.append("p88_basis_order_mismatch")
    if route.get("basis_num_elems") != P88_BASIS_NUM_ELEMS:
        blockers.append("p88_basis_num_elems_mismatch")
    if tuple(config.get("basis_dim_tuple", ())) != (P88_BASIS_DIM,) * P88_DIMENSION:
        blockers.append("p88_basis_dimension_mismatch")
    expected_ranks = (1,) + (P88_RANK,) * (P88_DIMENSION - 1) + (1,)
    if tuple(config.get("ranks", ())) != expected_ranks:
        blockers.append("p88_rank_tuple_mismatch")
    if float(config.get("defensive_tau", float("nan"))) != P88_TAU:
        blockers.append("p88_defensive_tau_mismatch")
    if float(config.get("normalizer_floor", float("nan"))) != P88_NORMALIZER_FLOOR:
        blockers.append("p88_normalizer_floor_mismatch")
    if float(config.get("denominator_floor", float("nan"))) != P88_DENOMINATOR_FLOOR:
        blockers.append("p88_denominator_floor_mismatch")
    if "transport_manifest" not in payload and "coordinate_frame" not in payload:
        blockers.append("missing_explicit_transport_branch_metadata")
    if not any(
        key in payload
        for key in (
            "source_dependency_closure",
            "callable_dependency_closure",
            "source_file_hashes",
        )
    ):
        blockers.append("missing_source_dependency_closure")
    return blockers


def reconstruct_p88_phase0(
    repository_root: str | Path = ".",
) -> P88Phase0Audit:
    """Load and reconstruct the exact serialized P88 T1 density.

    The function is intentionally fail-closed for missing transport metadata.
    It still returns the reconstructed density so the result can distinguish
    density reconstruction from full fixed-variant baseline admission.
    """

    root = Path(repository_root)
    path = root / P88_ARTIFACT_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact_sha256 = _sha256_file(path)
    if artifact_sha256 != P88_ARTIFACT_SHA256:
        raise ValueError("P88 artifact SHA-256 mismatch")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("P88 artifact must contain a JSON object")
    blockers = _validate_p88_metadata(payload)
    config = _json_mapping(payload.get("training_config"), "training_config")
    serialization = _json_mapping(
        payload.get("trained_core_serialization"),
        "trained_core_serialization",
    )
    records = serialization.get("cores")
    if not isinstance(records, list) or len(records) != P88_CORE_COUNT:
        raise ValueError("P88 serialized cores must be a 36-record list")
    cores = tuple(
        _core_values(_json_mapping(record, "core"), axis)
        for axis, record in enumerate(records)
    )
    basis_spec = p85_author_sir_lagrangep_algebraic_product_basis_spec(
        dimension=P88_DIMENSION,
        convention=P88_MEASURE_CONVENTION,
        order=P88_BASIS_ORDER,
        num_elems=P88_BASIS_NUM_ELEMS,
    )
    product_basis = basis_spec.build_product_basis()
    if product_basis.basis_dim_tuple() != (P88_BASIS_DIM,) * P88_DIMENSION:
        raise ValueError("P88 reconstructed basis dimension mismatch")
    sqrt_tt = FunctionalTT(
        tuple(TTCore(core) for core in cores),
        product_basis,
        P88_MEASURE_CONVENTION,
    )
    defensive = TensorProductReferenceDensity(product_basis, P88_MEASURE_CONVENTION)
    tau = tf.constant(P88_TAU, dtype=tf.float64)
    normalizer_floor = tf.constant(P88_NORMALIZER_FLOOR, dtype=tf.float64)
    denominator_floor = tf.constant(P88_DENOMINATOR_FLOOR, dtype=tf.float64)
    density = SquaredTTDensity(
        sqrt_tt=sqrt_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=normalizer_floor,
        denominator_floor=denominator_floor,
        measure_convention=P88_MEASURE_CONVENTION,
        branch_identity=SquaredTTDensity.expected_branch_identity(
            sqrt_tt=sqrt_tt,
            defensive_density=defensive,
            tau=tau,
            normalizer_floor=normalizer_floor,
            denominator_floor=denominator_floor,
            measure_convention=P88_MEASURE_CONVENTION,
        ),
    )
    tf.debugging.assert_near(
        density.sqrt_square_normalizer(),
        tf.constant(float(payload["sqrt_square_normalizer"]), dtype=tf.float64),
        atol=tf.constant(1.0e-18, dtype=tf.float64),
    )
    tf.debugging.assert_near(
        density.normalizer(),
        tf.constant(float(payload["normalizer"]), dtype=tf.float64),
        atol=tf.constant(1.0e-18, dtype=tf.float64),
    )
    return P88Phase0Audit(
        artifact_path=str(path),
        artifact_sha256=artifact_sha256,
        artifact=payload,
        cores=cores,
        density=density,
        blockers=tuple(blockers),
    )
