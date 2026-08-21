from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_neutra_targets import (
    GenUTControls,
    make_admitted_genut_neutra_target,
    make_genut_neutra_target,
)
from bayesfilter.highdim.genut_shape_lm_tf import GENUT_SHAPE_SOLVER_ID
from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
from bayesfilter.inference.neutra_batching import batch_native_value_status_target_fn


def _controls() -> GenUTControls:
    return GenUTControls(
        epsilon=2.0,
        sinkhorn_steps=2,
        balance_steps=2,
        ridge=1.0e-5,
        higher_moment_correction_steps=1,
        higher_moment_strength=0.05,
        higher_moment_floor=1.0e-5,
        tuning_scope="unit_test",
        tuning_artifact="unit_test",
    )


def test_all_target_factories_bind_data_chart_noise_and_controls() -> None:
    specifications = (
        ("lgssm", 12, 5, 50),
        ("ksc_sv", 12, 2, 1000),
        ("austria_sir", 36, 3, 20),
        ("predator_prey", 12, 6, 20),
    )
    signatures = []
    for model, count, dimension, horizon in specifications:
        target = make_genut_neutra_target(
            model, particle_count=count, controls=_controls()
        )
        assert target.parameter_dim == dimension
        assert target.observations.shape[0] == horizon
        assert len(target.target_signature) == 64
        assert len(target.adapter_signature()) == 64
        signatures.append(target.target_signature)
    assert len(set(signatures)) == len(signatures)


def test_lgssm_target_is_batch_native_and_endpoint_consistent() -> None:
    target = make_genut_neutra_target(
        "lgssm", particle_count=12, controls=_controls()
    )
    theta = tf.zeros([2, target.parameter_dim], tf.float64)
    value, score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    endpoint, endpoint_status = target.batch_value_status(theta)
    assert value.shape == (2,)
    assert score.shape == theta.shape
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
    tf.debugging.assert_equal(
        status["valid_pre_regularized_score"],
        endpoint_status["valid_pre_regularized_score"],
    )
    tf.debugging.assert_equal(value, endpoint)
    binding = bind_batch_native_neutra_target(
        target, target_signature=target.target_signature
    )
    _bound_value, normalized_status = batch_native_value_status_target_fn(binding)(
        theta
    )
    assert not bool(
        tf.reduce_any(
            normalized_status["min_innovation_eigenvalue_available"]
        ).numpy()
    )
    assert binding.payload()["scalar_fallback_used"] is False
    assert binding.payload()["sample_axis_python_loop_used"] is False
    assert binding.payload()["row_mapped_scalar_target_used"] is False


def test_training_method_has_no_sample_mapping_or_python_loop() -> None:
    target = make_genut_neutra_target(
        "lgssm", particle_count=12, controls=_controls()
    )
    source = inspect.getsource(target.neutra_batch_log_prob_and_grad_status)
    assert "map_fn" not in source
    assert "vectorized_map" not in source
    assert "for " not in source
    assert "while " not in source


def test_admitted_factory_rejects_blocked_austria() -> None:
    with pytest.raises(ValueError, match="not admitted"):
        make_admitted_genut_neutra_target("austria_sir")


def test_admitted_factory_requires_bound_arithmetic_scope() -> None:
    tf.config.experimental.enable_tensor_float_32_execution(True)
    with pytest.raises(RuntimeError, match="tf32_enabled=False"):
        make_admitted_genut_neutra_target("lgssm")


def test_repaired_controls_bind_solver_identity_and_scope() -> None:
    controls = GenUTControls(
        epsilon=2.0,
        sinkhorn_steps=2,
        balance_steps=2,
        ridge=1.0e-5,
        higher_moment_lm_damping=1.0e-2,
        higher_moment_lm_scale_floor=1.0e-4,
        higher_moment_trust_radius=0.5,
        tuning_scope="lgssm_T10_N1008_repair_replay",
        tuning_artifact="local_repair_diagnostic_not_admission",
    )
    payload = dict(controls.payload())
    assert payload["higher_moment_solver_id"] == GENUT_SHAPE_SOLVER_ID
    assert payload["tuning_scope"] == "lgssm_T10_N1008_repair_replay"
    assert payload["tuning_artifact"] == "local_repair_diagnostic_not_admission"
