"""Exact decoder for Lane-B T1 v1 artifacts with JSON tuple erasure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.source_route import SourceRouteCoordinateFrame
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    ARTIFACT_SCHEMA,
    DTYPE,
    LaneBT1Artifact,
    _estimate_from_payload,
    _settings_from_payload,
    issue_lane_b_t1_identity,
    source_closure,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import SIR_JOINT_DIM


COMPAT_DECODER_ID = "lane_b_t1_v1_json_tuple_restore_exact_identity_v1"


def _cloud_manifest_from_json(payload: Mapping[str, object]) -> Mapping[str, object]:
    restored = dict(payload)
    axis_order = restored.get("joint_axis_order")
    if axis_order is not None:
        if axis_order != ["z1", "z0"]:
            raise ValueError("Lane-B v1 cloud axis order is invalid")
        restored["joint_axis_order"] = ("z1", "z0")
    return restored


def load_lane_b_t1_artifact_v1_compat(directory: Path) -> LaneBT1Artifact:
    """Reload v1 exactly while repairing only JSON's tuple-to-list erasure."""

    output = Path(directory)
    payload = json.loads((output / "manifest.json").read_text())
    if payload.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("Lane-B artifact schema mismatch")
    if payload.get("source_closure") != dict(source_closure()):
        raise ValueError("Lane-B artifact source closure is stale")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("Lane-B artifact tensor ledger missing")

    def read_tensor(name: str) -> tf.Tensor:
        row = tensors.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"Lane-B artifact tensor missing: {name}")
        serialized = tf.io.read_file((output / str(row["path"])).as_posix())
        digest = hashlib.sha256(bytes(serialized.numpy())).hexdigest()
        if digest != row.get("sha256"):
            raise ValueError(f"Lane-B artifact tensor hash mismatch: {name}")
        value = tf.io.parse_tensor(
            serialized,
            out_type=tf.dtypes.as_dtype(str(row["dtype"])),
        )
        return tf.ensure_shape(value, row["shape"])

    settings = _settings_from_payload(payload["settings"])
    frame = SourceRouteCoordinateFrame(
        mu=read_tensor("frame_mu"),
        matrix=read_tensor("frame_matrix"),
        expansion_factor=settings.expansion_factor,
    )
    cores = tuple(read_tensor(f"core_{axis:02d}") for axis in range(SIR_JOINT_DIM))
    reference = read_tensor("frozen_reference_points")
    calibration = _estimate_from_payload(payload["calibration_estimate"])
    validation = _estimate_from_payload(payload["validation_estimate"])
    training_manifest = _cloud_manifest_from_json(payload["training_cloud_manifest"])
    validation_manifest = _cloud_manifest_from_json(payload["validation_cloud_manifest"])
    shift = tf.constant(float(payload["shift_constant"]), DTYPE)
    identity = issue_lane_b_t1_identity(
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift,
        calibration_estimate=calibration,
        validation_estimate=validation,
        frozen_reference_points=reference,
        training_cloud_manifest=training_manifest,
        validation_cloud_manifest=validation_manifest,
        source_hashes=payload["source_closure"],
    )
    if identity.hash.value != payload.get("identity_sha256"):
        raise ValueError("Lane-B artifact manifest identity mismatch after tuple restore")
    return LaneBT1Artifact(
        settings=settings,
        frame=frame,
        cores=cores,
        shift_constant=shift,
        calibration_estimate=calibration,
        validation_estimate=validation,
        frozen_reference_points=reference,
        training_cloud_manifest=training_manifest,
        validation_cloud_manifest=validation_manifest,
        source_hashes=payload["source_closure"],
        identity=identity,
    )


__all__ = ["COMPAT_DECODER_ID", "load_lane_b_t1_artifact_v1_compat"]
