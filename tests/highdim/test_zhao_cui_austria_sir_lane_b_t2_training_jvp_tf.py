from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_training_jvp_tf import (
    REQUIRED_T2_ISSUER_SOURCE_PATHS,
    ROOT,
    T2_OFFLINE_ISSUER_DERIVATIVE,
    T2ReplayCloudInputs,
    T2ReplayInputs,
    _make_t2_compiled_primal,
    _packed_t2_target_log_value,
    _project_tangents_to_scalar_derivative,
    issue_t2_training_jvp_identity_payload,
    load_t2_training_jvp_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_score_tf import (
    physical_z1_to_parent_local_prefix,
    t2_target_log_value_and_manual_score,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (
    FUNCTIONAL_SCREEN_COLUMNS,
    FUNCTIONAL_SCREEN_ORDER,
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_RTOL,
    OFFLINE_ISSUER_DERIVATIVE,
    _semantic_sha256,
    load_t1_training_jvp_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    balanced_initial_cores,
    lane_b_product_basis,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)
from bayesfilter.highdim.zhao_cui_austria_sir_packed_xla_tf import (
    pack_cores,
    pack_tangent_banks,
    packed_core_mask,
    precompute_basis_values,
    precompute_mass_matrices,
    packed_square_mass,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
    load_selected_t2_parameter_parent_compat,
)


DTYPE = tf.float64
PARENT_T1_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
PARENT_T2_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "pilot-final-01/t2_p05_r4_b5_lr3e4_l1_1e9/artifact"
)
T1_ISSUER_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-material-replay-xla-20260802/"
    "t1-material-fd-tangent-issuer-02"
)


def test_t2_radial_projection_enforces_requested_scalar_derivative_under_xla() -> None:
    parent_t1 = load_lane_b_t1_artifact_v1_compat(PARENT_T1_DIR)
    parent_t2 = load_selected_t2_parameter_parent_compat(
        PARENT_T2_DIR, parent_artifact=parent_t1
    )
    basis = lane_b_product_basis(
        order=parent_t2.settings.basis_order,
        num_elems=parent_t2.settings.basis_num_elems,
    )
    parent = pack_cores(parent_t2.cores)
    masses = precompute_mass_matrices(basis)
    raw = tf.random.stateless_normal(
        [3, *parent.shape], [82006, 1], dtype=DTYPE
    ) * packed_core_mask(tuple(core.shape for core in parent_t2.cores))[tf.newaxis, ...]
    requested = tf.constant([-3.25, 0.75, -2.5], DTYPE)
    tau = tf.constant(parent_t2.settings.tau, DTYPE)

    @tf.function(jit_compile=True)
    def compiled():
        corrected, raw_score, correction = _project_tangents_to_scalar_derivative(
            raw, requested, parent, masses, tau
        )
        with tf.GradientTape() as tape:
            tape.watch(parent)
            normalizer = packed_square_mass(parent, masses) + tau
        gradient = tape.gradient(normalizer, parent)
        corrected_score = tf.einsum(
            "pabcd,abcd->p", corrected, gradient
        ) / normalizer
        return corrected_score, raw_score, correction

    corrected_score, raw_score, correction = compiled()
    tf.debugging.assert_near(corrected_score, requested, atol=2e-12, rtol=2e-12)
    tf.debugging.assert_all_finite(raw_score, "raw scalar score")
    tf.debugging.assert_all_finite(correction, "radial correction")


def test_t2_packed_target_matches_manual_origin_and_graph_has_no_callbacks() -> None:
    parent_t1 = load_lane_b_t1_artifact_v1_compat(PARENT_T1_DIR)
    t1_child, _issuer = load_t1_training_jvp_child(
        T1_ISSUER_DIR, parent=parent_t1
    )
    parent_t2 = load_selected_t2_parameter_parent_compat(
        PARENT_T2_DIR, parent_artifact=parent_t1
    )
    settings = parent_t2.settings
    basis = lane_b_product_basis(
        order=settings.basis_order, num_elems=settings.basis_num_elems
    )
    states, observations, _all = generate_sealed_lane_b_dataset()
    joint_row = tf.concat([states[2], states[1]], axis=0)
    training_count = settings.batch_size
    calibration_count = 2

    def cloud(count: int) -> T2ReplayCloudInputs:
        joint = tf.broadcast_to(joint_row[tf.newaxis, :], [count, 36])
        local = tf.zeros([count, 36], DTYPE)
        local_z1 = physical_z1_to_parent_local_prefix(t1_child, joint[:, 18:])
        prefix = tf.concat([local_z1, tf.zeros([count, 18], DTYPE)], axis=1)
        return T2ReplayCloudInputs(
            joint_points=joint,
            local_points=local,
            origin_log_importance_weight=tf.zeros([count], DTYPE),
            origin_target_log_value=tf.zeros([count], DTYPE),
            basis_values=precompute_basis_values(basis, local),
            t1_prefix_basis_values=precompute_basis_values(basis, prefix),
        )

    training = cloud(training_count)
    calibration = cloud(calibration_count)
    initial = balanced_initial_cores(settings, basis)
    inputs = T2ReplayInputs(
        training=training,
        calibration=calibration,
        training_manifest={},
        calibration_manifest={},
        mass_matrices=precompute_mass_matrices(basis),
        initial_packed_cores=pack_cores(initial),
        parent_packed_cores=pack_cores(parent_t2.cores),
        packed_mask=packed_core_mask(tuple(core.shape for core in parent_t2.cores)),
        t1_parent_packed_cores=pack_cores(t1_child.parent_cores),
        t1_packed_tangents=pack_tangent_banks(t1_child.tangent_cores),
        observation=observations[1],
        microbatch_indices=tf.reshape(
            tf.range(training_count, dtype=tf.int32), [1, training_count]
        ),
    )
    origin = tf.zeros([3], DTYPE)
    packed_target = _packed_t2_target_log_value(
        origin, calibration, inputs, tf.constant(settings.tau, DTYPE)
    )
    manual_target = t2_target_log_value_and_manual_score(
        t1_child, origin, calibration.joint_points
    )["log_value"]
    tf.debugging.assert_near(packed_target, manual_target, atol=2e-11, rtol=2e-11)
    concrete = _make_t2_compiled_primal(
        t1_child, parent_t2, inputs
    ).get_concrete_function(tf.TensorSpec([3], DTYPE))
    graph = concrete.graph.as_graph_def()
    op_types = {node.op for node in graph.node}
    for function in graph.library.function:
        op_types.update(node.op for node in function.node_def)
    assert op_types.intersection({"While", "StatelessWhile"})
    assert not op_types.intersection({"PyFunc", "PyFuncStateless", "EagerPyFunc"})


def _write_synthetic_t2_issuer(directory: Path, monkeypatch: pytest.MonkeyPatch):
    parent_t1 = load_lane_b_t1_artifact_v1_compat(PARENT_T1_DIR)
    t1_child = LaneBParameterChild(
        parent_t1,
        tuple(tuple(tf.zeros_like(core) for _ in range(3)) for core in parent_t1.cores),
    )
    t1_issuer = {"issuer_identity": "synthetic-strict-t1-issuer"}
    monkeypatch.setattr(
        "bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_training_jvp_tf."
        "load_t1_training_jvp_child",
        lambda _directory, *, parent: (t1_child, t1_issuer),
    )
    parent_t2 = load_selected_t2_parameter_parent_compat(
        PARENT_T2_DIR, parent_artifact=parent_t1
    )
    banks = tuple(
        tuple(tf.zeros_like(core) for _ in range(3)) for core in parent_t2.cores
    )
    t2_child = LaneBParameterChild(parent_t2, banks)
    tensors = {}
    hashes = {}
    for axis, bank in enumerate(banks):
        for parameter, tangent in enumerate(bank):
            name = f"tangent_{axis:02d}_{parameter}"
            serialized = tf.io.serialize_tensor(tangent)
            path = directory / f"{name}.tensor"
            tf.io.write_file(path.as_posix(), serialized)
            digest = hashlib.sha256(bytes(serialized.numpy())).hexdigest()
            tensors[name] = {
                "path": path.name,
                "sha256": digest,
                "dtype": "float64",
                "shape": tangent.shape.as_list(),
            }
            hashes[name] = digest
    origin = tf.zeros([3], DTYPE)
    t1_value, t1_score = t1_child.increment_and_score(origin)
    increment, increment_score = t2_child.increment_and_score(origin)
    evidence = {
        "functional_replay_metrics": [
            [0.0, 0.0, 0.0] for _ in FUNCTIONAL_SCREEN_ORDER
        ],
        "scalar_absolute_residual": 0.0,
        "scalar_normalized_residual": 0.0,
        "explanatory_maximum_core_residual": 0.0,
        "explanatory_maximum_normalized_core_residual": 0.0,
        "manual_increment": float(increment.numpy()),
        "manual_increment_score": increment_score.numpy().tolist(),
        "manual_cumulative_value": float((t1_value + increment).numpy()),
        "manual_cumulative_score": (t1_score + increment_score).numpy().tolist(),
        "finite_difference_rows": [
            {
                "parameter": parameter,
                "step": 1e-4,
                "finite_difference": 0.0,
                "issued_tangent_score": 0.0,
                "absolute_residual": 0.0,
            }
            for parameter in range(3)
        ],
        "gpu_allocator_peak_bytes": 0,
        "offline_issuer_derivative": T2_OFFLINE_ISSUER_DERIVATIVE,
    }
    identity = issue_t2_training_jvp_identity_payload(
        t1_issuer_identity=t1_issuer["issuer_identity"],
        t1_child_identity=t1_child.identity.hash.value,
        parent_t1_identity=parent_t1.identity.hash.value,
        parent_t2=parent_t2,
        t2_child_identity=t2_child.identity.hash.value,
        tangent_tensor_sha256=hashes,
        evidence=evidence,
    )
    payload = {
        "schema_version": identity["schema_version"],
        "status": "PASS_T1_T2_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT",
        "issuer_identity": _semantic_sha256(identity),
        "issuer_identity_payload": identity,
        "parent_t1_identity": parent_t1.identity.hash.value,
        "parent_t2_identity": parent_t2.identity.hash.value,
        "t1_issuer_identity": t1_issuer["issuer_identity"],
        "t2_child_identity": t2_child.identity.hash.value,
        "functional_replay_metrics": evidence["functional_replay_metrics"],
        "tensors": tensors,
        "hard_gates": {
            "strict_t1_issuer_load": True,
            "strict_t2_prepared_cloud_load": True,
            "material_functional_replay": True,
            "material_scalar_replay": True,
            "manual_issued_tangent_parity": True,
            "independent_step_halving_fd_parity": True,
            "memory_under_6_gib": True,
        },
    }
    (directory / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return parent_t1, parent_t2, payload


def test_t2_issuer_loader_roundtrips_and_rejects_tensor_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_t1, parent_t2, _payload = _write_synthetic_t2_issuer(
        tmp_path, monkeypatch
    )
    _t1_child, t2_child, loaded = load_t2_training_jvp_child(
        tmp_path,
        t1_issuer_directory=tmp_path / "synthetic-t1",
        parent_t1=parent_t1,
        parent_t2=parent_t2,
    )
    assert t2_child.identity.hash.value == loaded["t2_child_identity"]
    path = tmp_path / "tangent_00_0.tensor"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        load_t2_training_jvp_child(
            tmp_path,
            t1_issuer_directory=tmp_path / "synthetic-t1",
            parent_t1=parent_t1,
            parent_t2=parent_t2,
        )


def test_t2_issuer_loader_rejects_self_rehashed_stale_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_t1, parent_t2, payload = _write_synthetic_t2_issuer(
        tmp_path, monkeypatch
    )
    identity = payload["issuer_identity_payload"]
    identity["source_sha256"][REQUIRED_T2_ISSUER_SOURCE_PATHS[0]] = "0" * 64
    payload["issuer_identity"] = _semantic_sha256(identity)
    (tmp_path / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="source closure stale"):
        load_t2_training_jvp_child(
            tmp_path,
            t1_issuer_directory=tmp_path / "synthetic-t1",
            parent_t1=parent_t1,
            parent_t2=parent_t2,
        )
