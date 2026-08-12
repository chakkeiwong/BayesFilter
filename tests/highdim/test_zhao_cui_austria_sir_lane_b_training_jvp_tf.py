from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    FINITE_DIFFERENCE_ATOL,
    FINITE_DIFFERENCE_RTOL,
    FINITE_DIFFERENCE_STEP,
    FUNCTIONAL_SCREEN_COLUMNS,
    FUNCTIONAL_SCREEN_ORDER,
    GPU_MEMORY_LIMIT_MIB,
    ISSUER_ID,
    ISSUER_SCHEMA,
    OFFLINE_ISSUER_DERIVATIVE,
    MEMORY_CAP_BYTES,
    REPLAY_ID,
    REQUIRED_ISSUER_SOURCE_PATHS,
    ROOT,
    RUNTIME_SCORE_BACKEND,
    SHIFT_DERIVATIVE_POLICY,
    TAU_DERIVATIVE_POLICY,
    TANGENT_FINITE_DIFFERENCE_STEP,
    T1ReplayInputs,
    _physical_log_target_and_score,
    _physical_log_target_value_xla,
    _make_t1_compiled_primal,
    _semantic_sha256,
    _training_loss,
    functional_adam_step,
    load_t1_training_jvp_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_packed_xla_tf import (
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_POLICY_ID,
    MATERIAL_REPLAY_RTOL,
    PACKED_XLA_POLICY_ID,
    pack_cores,
    packed_core_mask,
    precompute_mass_matrices,
)
from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    balanced_initial_cores,
    lane_b_product_basis,
)


DTYPE = tf.float64
PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)


def _loss(theta: tf.Tensor, cores: tuple[tf.Tensor, ...]) -> tf.Tensor:
    target = tf.stack([theta[0], 2.0 * theta[0]])
    return tf.reduce_sum(tf.square(cores[0] - target))


def test_functional_adam_matches_keras_one_step() -> None:
    theta = tf.constant([0.2], DTYPE)
    initial = tf.constant([0.5, -0.3], DTYPE)
    variable = tf.Variable(initial)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    with tf.GradientTape() as tape:
        loss = _loss(theta, (variable,))
    gradient = tape.gradient(loss, variable)
    clipped, _ = tf.clip_by_global_norm((gradient,), tf.constant(10.0, DTYPE))
    optimizer.apply_gradients(((clipped[0], variable),))

    observed, momentums, velocities = functional_adam_step(
        theta,
        (initial,),
        (tf.zeros_like(initial),),
        (tf.zeros_like(initial),),
        step=1,
        learning_rate=tf.cast(tf.constant(3e-4, tf.float32), DTYPE),
        gradient_clip_norm=tf.constant(10.0, DTYPE),
        loss_fn=_loss,
    )
    tf.debugging.assert_equal(observed[0], variable)
    tf.debugging.assert_all_finite(momentums[0], "momentum")
    tf.debugging.assert_all_finite(velocities[0], "velocity")


def test_functional_adam_forward_jvp_matches_centered_difference() -> None:
    initial = tf.constant([0.5, -0.3], DTYPE)
    zeros = (tf.zeros_like(initial),)

    def run(theta: tf.Tensor) -> tf.Tensor:
        cores, _, _ = functional_adam_step(
            theta,
            (initial,),
            zeros,
            zeros,
            step=1,
            learning_rate=tf.cast(tf.constant(3e-4, tf.float32), DTYPE),
            gradient_clip_norm=tf.constant(10.0, DTYPE),
            loss_fn=_loss,
        )
        return cores[0]

    theta = tf.constant([0.2], DTYPE)
    with tf.autodiff.ForwardAccumulator(theta, tf.ones_like(theta)) as accumulator:
        value = run(theta)
    tangent = accumulator.jvp(value)
    step = tf.constant(1e-5, DTYPE)
    finite_difference = (run(theta + step) - run(theta - step)) / (2.0 * step)
    tf.debugging.assert_near(tangent, finite_difference, atol=2e-10, rtol=2e-7)


def test_functional_adam_matches_keras_inside_xla() -> None:
    theta = tf.constant([0.2], DTYPE)
    initial = tf.constant([0.5, -0.3], DTYPE)
    variable = tf.Variable(initial)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    optimizer.build((variable,))

    @tf.function(jit_compile=True)
    def keras_step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = _loss(theta, (variable,))
        gradient = tape.gradient(loss, variable)
        clipped, _ = tf.clip_by_global_norm(
            (gradient,), tf.constant(10.0, DTYPE)
        )
        optimizer.apply_gradients(((clipped[0], variable),))
        return tf.identity(variable)

    @tf.function(jit_compile=True)
    def functional_step() -> tf.Tensor:
        observed, _, _ = functional_adam_step(
            theta,
            (initial,),
            (tf.zeros_like(initial),),
            (tf.zeros_like(initial),),
            step=tf.constant(1, tf.int32),
            learning_rate=tf.cast(tf.constant(3e-4, tf.float32), DTYPE),
            gradient_clip_norm=tf.constant(10.0, DTYPE),
            loss_fn=_loss,
        )
        return observed[0]

    tf.debugging.assert_equal(functional_step(), keras_step())


def test_t1_graph_native_target_matches_eager_value_score_and_xla() -> None:
    model = latent_preclip_zhao_cui_sir_austria_model()
    states, observations, _all = generate_sealed_lane_b_dataset()
    points = tf.concat(
        [states[1:3], states[0:2]], axis=1
    )
    theta = tf.constant([0.02, -0.03, 0.01], DTYPE)
    eager_value, eager_score = _physical_log_target_and_score(
        theta,
        points,
        model=model,
        observation=observations[0],
    )
    graph_score_rows = []
    for parameter in range(3):
        with tf.autodiff.ForwardAccumulator(
            theta, tf.one_hot(parameter, 3, dtype=DTYPE)
        ) as accumulator:
            graph_value = _physical_log_target_value_xla(
                theta, points, observations[0]
            )
        graph_score_rows.append(accumulator.jvp(graph_value))
    graph_score = tf.transpose(tf.stack(graph_score_rows))
    tf.debugging.assert_near(graph_value, eager_value, atol=2e-12, rtol=2e-12)
    tf.debugging.assert_near(graph_score, eager_score, atol=2e-10, rtol=2e-10)

    @tf.function(jit_compile=True)
    def compiled_target(theta: tf.Tensor, points: tf.Tensor):
        return _physical_log_target_value_xla(
            theta, points, observations[0]
        )

    value = compiled_target(theta, points)
    tf.debugging.assert_all_finite(value, "compiled T1 target value")
    tf.debugging.assert_near(value, graph_value, atol=2e-12, rtol=2e-12)


def test_t1_real_loss_forward_over_reverse_compiles_with_xla() -> None:
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    settings = parent.settings
    basis = lane_b_product_basis(
        order=settings.basis_order, num_elems=settings.basis_num_elems
    )
    cores = tuple(balanced_initial_cores(settings, basis))
    states, observations, _all = generate_sealed_lane_b_dataset()
    physical = tf.concat([states[1:3], states[0:2]], axis=1)
    local = tf.zeros([2, 36], DTYPE)
    origin = tf.zeros([3], DTYPE)
    origin_log = _physical_log_target_value_xla(
        origin, physical, observations[0]
    )

    @tf.function(jit_compile=True)
    def compiled_step(theta: tf.Tensor):
        def loss_fn(active_theta, active_cores):
            return _training_loss(
                active_theta,
                active_cores,
                physical,
                local,
                tf.ones([2], DTYPE),
                tf.ones([2], DTYPE),
                origin_log,
                tau=tf.constant(settings.tau, DTYPE),
                l1_weight=tf.constant(settings.l1_weight, DTYPE),
                l2_weight=tf.constant(settings.l2_weight, DTYPE),
                basis=basis,
                observation=observations[0],
            )

        updated, _momentums, _velocities = functional_adam_step(
            theta,
            cores,
            tuple(tf.zeros_like(core) for core in cores),
            tuple(tf.zeros_like(core) for core in cores),
            step=tf.constant(1, tf.int32),
            learning_rate=tf.cast(
                tf.constant(settings.learning_rate, tf.float32), DTYPE
            ),
            gradient_clip_norm=tf.constant(settings.gradient_clip_norm, DTYPE),
            loss_fn=loss_fn,
        )
        return updated[0]

    direction = tf.one_hot(0, 3, dtype=DTYPE)
    with tf.autodiff.ForwardAccumulator(origin, direction) as accumulator:
        value = compiled_step(origin)
    tangent = accumulator.jvp(value)
    tf.debugging.assert_all_finite(value, "compiled T1 real-loss Adam value")
    tf.debugging.assert_all_finite(tangent, "compiled T1 real-loss Adam JVP")


def test_t1_full_primal_graph_uses_while_and_has_no_host_callbacks() -> None:
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    basis = lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    initial = balanced_initial_cores(parent.settings, basis)
    population = parent.settings.batch_size
    calibration_count = 2
    inputs = T1ReplayInputs(
        training_physical_points=tf.zeros([population, 36], DTYPE),
        training_local_points=tf.zeros([population, 36], DTYPE),
        training_origin_target_sqrt=tf.ones([population], DTYPE),
        training_integration_weights=tf.ones([population], DTYPE),
        calibration_physical_points=tf.zeros([calibration_count, 36], DTYPE),
        calibration_origin_log_likelihood=tf.zeros([calibration_count], DTYPE),
        training_basis_values=tf.zeros([population, 36, 5], DTYPE),
        calibration_basis_values=tf.zeros([calibration_count, 36, 5], DTYPE),
        mass_matrices=precompute_mass_matrices(basis),
        initial_packed_cores=pack_cores(initial),
        parent_packed_cores=pack_cores(parent.cores),
        packed_mask=packed_core_mask(tuple(core.shape for core in parent.cores)),
        observation=tf.zeros([9], DTYPE),
        training_batch_indices=tf.zeros(
            [parent.settings.train_steps, parent.settings.batch_size], tf.int32
        ),
    )
    concrete = _make_t1_compiled_primal(parent, inputs).get_concrete_function(
        tf.TensorSpec([3], DTYPE)
    )
    graph = concrete.graph.as_graph_def()
    op_types = {node.op for node in graph.node}
    for function in graph.library.function:
        op_types.update(node.op for node in function.node_def)
    assert op_types.intersection({"While", "StatelessWhile"})
    assert not op_types.intersection({"PyFunc", "PyFuncStateless", "EagerPyFunc"})


def _write_synthetic_issuer(directory: Path):
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    banks = tuple(
        tuple(tf.zeros_like(core) for _ in range(3)) for core in parent.cores
    )
    child = LaneBParameterChild(parent, banks)
    tensors = {}
    tensor_hashes = {}
    for axis, bank in enumerate(banks):
        for parameter, tangent in enumerate(bank):
            name = f"tangent_{axis:02d}_{parameter}"
            path = directory / f"{name}.tensor"
            serialized = tf.io.serialize_tensor(tangent)
            tf.io.write_file(path.as_posix(), serialized)
            digest = hashlib.sha256(bytes(serialized.numpy())).hexdigest()
            tensors[name] = {
                "path": path.name,
                "sha256": digest,
                "dtype": "float64",
                "shape": tangent.shape.as_list(),
            }
            tensor_hashes[name] = digest
    source_hashes = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in REQUIRED_ISSUER_SOURCE_PATHS
    }
    identity = {
        "schema_version": ISSUER_SCHEMA,
        "issuer_id": ISSUER_ID,
        "replay_id": REPLAY_ID,
        "classification": "extension_or_invention",
        "material_replay_policy_id": MATERIAL_REPLAY_POLICY_ID,
        "packed_xla_policy_id": PACKED_XLA_POLICY_ID,
        "parent_identity": parent.identity.hash.value,
        "parent_value": float(parent.value().numpy()),
        "child_identity": child.identity.hash.value,
        "training_cloud_manifest": dict(parent.training_cloud_manifest),
        "calibration_estimate": parent.calibration_estimate.manifest_payload(),
        "shift_derivative_policy": SHIFT_DERIVATIVE_POLICY,
        "tau_derivative_policy": TAU_DERIVATIVE_POLICY,
        "optimizer": {
            "family": "keras3_adam_functional_exact_update_order",
            "learning_rate": parent.settings.learning_rate,
            "beta_1": ADAM_BETA_1,
            "beta_2": ADAM_BETA_2,
            "epsilon": ADAM_EPSILON,
            "gradient_clip_norm": parent.settings.gradient_clip_norm,
            "train_steps": parent.settings.train_steps,
            "batch_size": parent.settings.batch_size,
            "jit_compile": True,
            "full_program_control_flow": "tensorflow_while_loop",
            "python_numerical_loops": False,
        },
        "replay_gate": {
            "material_functional_atol": MATERIAL_REPLAY_ATOL,
            "material_functional_rtol": MATERIAL_REPLAY_RTOL,
            "maximum_normalized_functional_residual": 1.0,
            "functional_screen_order": list(FUNCTIONAL_SCREEN_ORDER),
            "functional_screen_columns": list(FUNCTIONAL_SCREEN_COLUMNS),
            "tangent_finite_difference_step": TANGENT_FINITE_DIFFERENCE_STEP,
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
            "finite_difference_atol": FINITE_DIFFERENCE_ATOL,
            "finite_difference_rtol": FINITE_DIFFERENCE_RTOL,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "gpu_memory_limit_mib": GPU_MEMORY_LIMIT_MIB,
        },
        "material_replay_evidence": {
            "functional_replay_metrics": [[0.0, 0.0, 0.0] for _ in FUNCTIONAL_SCREEN_ORDER],
            "scalar_absolute_residual": 0.0,
            "scalar_normalized_residual": 0.0,
            "scalar_log_residual": 0.0,
        },
        "derivative_evidence": {
            "issuer_method": OFFLINE_ISSUER_DERIVATIVE,
            "independent_finite_difference_rows": [
                {
                    "parameter": parameter,
                    "step": FINITE_DIFFERENCE_STEP,
                    "value_plus": 0.0,
                    "value_minus": 0.0,
                    "finite_difference": 0.0,
                    "issued_tangent_score": 0.0,
                    "absolute_residual": 0.0,
                }
                for parameter in range(3)
            ],
        },
        "tangent_tensor_sha256": tensor_hashes,
        "source_sha256": source_hashes,
        "runtime_score_backend": RUNTIME_SCORE_BACKEND,
        "offline_issuer_derivative": OFFLINE_ISSUER_DERIVATIVE,
        "runtime_autodiff": False,
        "runtime_finite_difference": False,
        "hmc_authorized": False,
    }
    payload = {
        "schema_version": ISSUER_SCHEMA,
        "status": "PASS_T1_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT",
        "issuer_identity": _semantic_sha256(identity),
        "issuer_identity_payload": identity,
        "parent_identity": parent.identity.hash.value,
        "child_identity": child.identity.hash.value,
        "manual_value": float(parent.value().numpy()),
        "manual_score": [0.0, 0.0, 0.0],
        "finite_difference_rows": identity["derivative_evidence"][
            "independent_finite_difference_rows"
        ],
        "functional_replay_metrics": [[0.0, 0.0, 0.0] for _ in FUNCTIONAL_SCREEN_ORDER],
        "scalar_absolute_residual": 0.0,
        "scalar_normalized_residual": 0.0,
        "scalar_log_residual": 0.0,
        "explanatory_core_residuals": {
            "maximum_absolute_residual": 0.0,
            "maximum_normalized_residual": 0.0,
            "promotion_role": "explanatory_gauge_diagnostic_only",
        },
        "hard_gates": {
            "training_and_calibration_cloud_hashes": True,
            "material_functional_replay": True,
            "material_scalar_replay": True,
            "manual_issued_tangent_parity": True,
            "independent_step_halving_fd_parity": True,
            "memory_under_6_gib": True,
        },
        "tensors": tensors,
    }
    (directory / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return parent, payload


def test_t1_issuer_loader_roundtrips_and_rejects_tensor_tamper(tmp_path: Path) -> None:
    parent, _payload = _write_synthetic_issuer(tmp_path)
    child, loaded = load_t1_training_jvp_child(tmp_path, parent=parent)
    assert child.identity.hash.value == loaded["child_identity"]
    path = tmp_path / "tangent_00_0.tensor"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        load_t1_training_jvp_child(tmp_path, parent=parent)


def test_t1_issuer_loader_rejects_self_rehashed_stale_source(tmp_path: Path) -> None:
    parent, payload = _write_synthetic_issuer(tmp_path)
    identity = payload["issuer_identity_payload"]
    first = REQUIRED_ISSUER_SOURCE_PATHS[0]
    identity["source_sha256"][first] = "0" * 64
    payload["issuer_identity"] = _semantic_sha256(identity)
    (tmp_path / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="source closure stale"):
        load_t1_training_jvp_child(tmp_path, parent=parent)


def test_t1_issuer_loader_rejects_self_rehashed_failed_material_replay(
    tmp_path: Path,
) -> None:
    parent, payload = _write_synthetic_issuer(tmp_path)
    identity = payload["issuer_identity_payload"]
    identity["material_replay_evidence"]["functional_replay_metrics"][0][1] = 1.01
    payload["functional_replay_metrics"][0][1] = 1.01
    payload["issuer_identity"] = _semantic_sha256(identity)
    (tmp_path / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="material replay evidence failed"):
        load_t1_training_jvp_child(tmp_path, parent=parent)
