"""Independent sampled Zhao-Cui bounded-feature teacher for Austria T1/T2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_filter import BoundedFeatureShapeTeacher
from bayesfilter.highdim.higher_moment_contract_e import weighted_shape_targets_jvp
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_sampler_tf import (
    LaneBRetainedGridSampler,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_training_jvp_tf import (
    load_t2_training_jvp_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (
    load_t1_training_jvp_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
    load_selected_t2_parameter_parent_compat,
)


DTYPE = tf.float64
ROUTE_ID = "zhao_cui_austria_sir_sampled_bounded_feature_teacher_t1_t2_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
ROOT = Path(__file__).resolve().parents[2]
PARENT_T1_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
PARENT_T2_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "pilot-final-01/t2_p05_r4_b5_lr3e4_l1_1e9/artifact"
)
T1_ISSUER_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-training-jvp-20260806/"
    "attempt-01-current-closure"
)
T2_ISSUER_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-training-jvp-20260806/"
    "attempt-01-current-closure"
)


@dataclass(frozen=True)
class AustriaBoundedTeacherBuild:
    teacher: BoundedFeatureShapeTeacher
    diagnostics: tuple[Mapping[str, tf.Tensor | str | int], ...]
    parent_identities: tuple[str, str]
    child_identities: tuple[str, str]
    route_id: str = ROUTE_ID
    route_classification: str = ROUTE_CLASSIFICATION


def _tensor_sha256(value: tf.Tensor) -> str:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    return hashlib.sha256(bytes(serialized.numpy())).hexdigest()


def save_austria_t1_t2_bounded_teacher(
    build: AustriaBoundedTeacherBuild, directory: Path
) -> Path:
    """Write the fixed teacher tensors and a strict identity manifest."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for name in BoundedFeatureShapeTeacher.__dataclass_fields__:
        value = tf.convert_to_tensor(getattr(build.teacher, name))
        serialized = tf.io.serialize_tensor(value)
        path = output / f"{name}.tensor"
        tf.io.write_file(path.as_posix(), serialized)
        tensors[name] = {
            "path": path.name,
            "sha256": hashlib.sha256(bytes(serialized.numpy())).hexdigest(),
            "dtype": value.dtype.name,
            "shape": value.shape.as_list(),
        }
    payload = {
        "schema_version": "bayesfilter.zhao_cui_austria_bounded_teacher.v1",
        "route_id": build.route_id,
        "route_classification": build.route_classification,
        "parent_identities": list(build.parent_identities),
        "child_identities": list(build.child_identities),
        "diagnostics": [
            {
                key: (
                    value.numpy().item()
                    if isinstance(value, tf.Tensor) and value.shape.rank == 0
                    else value
                )
                for key, value in row.items()
            }
            for row in build.diagnostics
        ],
        "tensors": tensors,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest


def load_austria_t1_t2_bounded_teacher(
    directory: Path,
) -> tuple[BoundedFeatureShapeTeacher, Mapping[str, object]]:
    """Load only a complete sampled teacher artifact with verified tensors."""

    output = Path(directory)
    payload = json.loads((output / "manifest.json").read_text())
    if (
        payload.get("schema_version")
        != "bayesfilter.zhao_cui_austria_bounded_teacher.v1"
        or payload.get("route_id") != ROUTE_ID
        or payload.get("route_classification") != ROUTE_CLASSIFICATION
    ):
        raise ValueError("bounded teacher manifest identity mismatch")
    tensors = payload.get("tensors")
    names = set(BoundedFeatureShapeTeacher.__dataclass_fields__)
    if not isinstance(tensors, dict) or set(tensors) != names:
        raise ValueError("bounded teacher tensor ledger mismatch")
    values = {}
    for name in names:
        row = tensors[name]
        candidate = (output / str(row["path"])).resolve()
        try:
            candidate.relative_to(output.resolve())
        except ValueError as exc:
            raise ValueError("bounded teacher tensor escapes artifact") from exc
        serialized = tf.io.read_file(candidate.as_posix())
        if hashlib.sha256(bytes(serialized.numpy())).hexdigest() != row["sha256"]:
            raise ValueError(f"bounded teacher tensor hash mismatch: {name}")
        values[name] = tf.ensure_shape(
            tf.io.parse_tensor(
                serialized, out_type=tf.dtypes.as_dtype(str(row["dtype"]))
            ),
            row["shape"],
        )
    teacher = BoundedFeatureShapeTeacher(**values)
    return teacher, payload


def _one_time_teacher(
    child: LaneBParameterChild,
    *,
    sample_count: int,
    seed: int,
) -> tuple[dict[str, tf.Tensor], Mapping[str, tf.Tensor | str | int]]:
    uniforms = tf.random.stateless_uniform(
        [18, int(sample_count)],
        seed=tf.constant([int(seed), 1], tf.int32),
        minval=tf.constant(1.0e-6, DTYPE),
        maxval=tf.constant(1.0 - 1.0e-6, DTYPE),
        dtype=DTYPE,
    )
    retained = LaneBRetainedGridSampler(child.parent).inverse(uniforms)
    theta = tf.zeros([3], DTYPE)
    _log_marginal, marginal_score = child.prefix_log_marginal_and_score(
        theta, retained.local_points
    )
    correction = retained.correction_log_weights
    log_weights = correction - tf.reduce_logsumexp(correction)
    weights = tf.exp(log_weights)
    centered_score = marginal_score - tf.reduce_sum(
        weights[:, None] * marginal_score, axis=0, keepdims=True
    )
    weights_tangent = weights[:, None] * centered_score
    bounded = retained.local_points / tf.sqrt(
        1.0 + tf.square(retained.local_points)
    )
    moments = weighted_shape_targets_jvp(
        bounded,
        weights,
        tf.zeros([sample_count, 18, 3], DTYPE),
        weights_tangent,
    )
    off_diagonal = 1.0 - tf.eye(18, dtype=DTYPE)
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(weights)))
    diagnostics: Mapping[str, tf.Tensor | str | int] = {
        "sample_count": int(sample_count),
        "seed": int(seed),
        "effective_sample_size": ess,
        "minimum_log_correction": tf.reduce_min(correction),
        "maximum_log_correction": tf.reduce_max(correction),
        "maximum_absolute_bounded_coordinate": tf.reduce_max(tf.abs(bounded)),
        "maximum_absolute_marginal_score": tf.reduce_max(tf.abs(marginal_score)),
        "parent_identity": child.parent.identity.hash.value,
        "child_identity": child.identity.hash.value,
    }
    return {
        **moments,
        "pairwise_co_skew_mask": off_diagonal,
        "pairwise_co_kurtosis_mask": off_diagonal,
        "frame_mu": child.parent.frame.mu[:18],
        "frame_matrix": child.parent.frame.matrix[:18, :18],
    }, diagnostics


def build_austria_t1_t2_bounded_teacher(
    *,
    sample_count: int = 256,
    seeds: tuple[int, int] = (98501, 98502),
) -> AustriaBoundedTeacherBuild:
    """Strictly load T1/T2 and build an independent sampled TT teacher."""

    if int(sample_count) < 64:
        raise ValueError("bounded teacher requires at least 64 fixed samples")
    parent_t1 = load_lane_b_t1_artifact_v1_compat(PARENT_T1_DIR)
    parent_t2 = load_selected_t2_parameter_parent_compat(
        PARENT_T2_DIR, parent_artifact=parent_t1
    )
    t1_child, _ = load_t1_training_jvp_child(T1_ISSUER_DIR, parent=parent_t1)
    issued_t1_child, t2_child, _ = load_t2_training_jvp_child(
        T2_ISSUER_DIR,
        t1_issuer_directory=T1_ISSUER_DIR,
        parent_t1=parent_t1,
        parent_t2=parent_t2,
    )
    if issued_t1_child.identity != t1_child.identity:
        raise ValueError("T1 child identity differs across strict issuer chains")
    rows = []
    diagnostics = []
    for child, seed in zip((t1_child, t2_child), seeds):
        row, diagnostic = _one_time_teacher(
            child, sample_count=int(sample_count), seed=int(seed)
        )
        rows.append(row)
        diagnostics.append(diagnostic)
    teacher = BoundedFeatureShapeTeacher(
        **{
            name: tf.stack([row[name] for row in rows], axis=0)
            for name in rows[0]
        }
    )
    return AustriaBoundedTeacherBuild(
        teacher=teacher,
        diagnostics=tuple(diagnostics),
        parent_identities=(
            parent_t1.identity.hash.value,
            parent_t2.identity.hash.value,
        ),
        child_identities=(
            t1_child.identity.hash.value,
            t2_child.identity.hash.value,
        ),
    )


__all__ = [
    "AustriaBoundedTeacherBuild",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "build_austria_t1_t2_bounded_teacher",
    "load_austria_t1_t2_bounded_teacher",
    "save_austria_t1_t2_bounded_teacher",
]
