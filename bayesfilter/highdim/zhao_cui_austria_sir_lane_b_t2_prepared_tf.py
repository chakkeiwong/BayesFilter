"""Strict loader for sealed Lane-B T2 proposal clouds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (
    LaneBT2LogNormalizerEstimate,
    LaneBT2ProposalCloud,
    _t2_estimate_from_payload,
    t2_source_closure,
)


SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t2_prepared_cloud.v1"
ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = "docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t2-plan-2026-07-31.md"
PREPARATION_SOURCE_PATHS = (
    "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_artifact_compat.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_t2_prepared_tf.py",
    "scripts/prepare_zhao_cui_austria_sir_lane_b_t2.py",
    PLAN_PATH,
)


def prepared_source_closure() -> Mapping[str, str]:
    result = dict(t2_source_closure())
    for relative_path in PREPARATION_SOURCE_PATHS:
        result[relative_path] = hashlib.sha256(
            (ROOT / relative_path).read_bytes()
        ).hexdigest()
    return {path: result[path] for path in sorted(result)}


def _verify_prepared_source_closure(payload: Mapping[str, object]) -> None:
    run_manifest = payload.get("run_manifest")
    if not isinstance(run_manifest, Mapping):
        raise ValueError("T2 prepared run manifest missing")
    source_hashes = run_manifest.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("T2 prepared source closure missing")
    expected = prepared_source_closure()
    if set(source_hashes) != set(expected):
        raise ValueError("T2 prepared source closure path mismatch")
    for relative_path, current in expected.items():
        if source_hashes.get(relative_path) != current:
            raise ValueError(f"T2 prepared source closure mismatch: {relative_path}")


def load_t2_prepared_cloud(directory: Path) -> tuple[LaneBT2ProposalCloud, Mapping[str, object]]:
    output = Path(directory)
    payload = json.loads((output / "result.json").read_text())
    if payload.get("schema_version") != SCHEMA or payload.get("status") != "PREPARED_T2_CLOUD":
        raise ValueError("T2 prepared-cloud schema/status mismatch")
    _verify_prepared_source_closure(payload)
    if not all(
        bool(payload.get("gates", {}).get(name))
        for name in ("finite", "sample_count", "memory", "effective_sample_size")
    ):
        raise ValueError("T2 prepared cloud failed a hard gate")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("T2 prepared tensor ledger missing")

    def read(name: str) -> tf.Tensor:
        row = tensors.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"T2 prepared tensor missing: {name}")
        serialized = tf.io.read_file((output / str(row["path"])).as_posix())
        digest = hashlib.sha256(bytes(serialized.numpy())).hexdigest()
        if digest != row.get("sha256"):
            raise ValueError(f"T2 prepared tensor hash mismatch: {name}")
        value = tf.io.parse_tensor(
            serialized, out_type=tf.dtypes.as_dtype(str(row["dtype"]))
        )
        return tf.ensure_shape(value, row["shape"])

    manifest = payload["cloud_manifest"]
    cloud = LaneBT2ProposalCloud(
        joint_points=read("joint_points"),
        previous_log_target=read("previous_log_target"),
        previous_log_proposal=read("previous_log_proposal"),
        previous_correction=read("previous_correction"),
        transition_log_density=read("transition_log_density"),
        log_likelihood=read("log_likelihood"),
        reference_uniforms=read("reference_uniforms"),
        reference_seed=int(manifest["reference_seed"]),
        transition_seed=int(manifest["transition_seed"]),
        role=str(manifest["role"]),
    )
    recomputed = cloud.manifest_payload()
    for key, expected in recomputed.items():
        observed = manifest.get(key)
        if key == "joint_axis_order" and observed is not None:
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(f"T2 prepared cloud manifest mismatch: {key}")
    return cloud, payload


def prepared_estimate(payload: Mapping[str, object], *, calibrated: bool) -> LaneBT2LogNormalizerEstimate:
    key = "calibrated_estimate" if calibrated else "zero_shift_estimate"
    row = payload.get(key)
    if not isinstance(row, Mapping):
        raise ValueError(f"T2 prepared estimate missing: {key}")
    return _t2_estimate_from_payload(row)


__all__ = [
    "PLAN_PATH",
    "PREPARATION_SOURCE_PATHS",
    "SCHEMA",
    "load_t2_prepared_cloud",
    "prepared_estimate",
    "prepared_source_closure",
]
