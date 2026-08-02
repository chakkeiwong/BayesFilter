import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_ratio_bridge_tf import (
    EXPECTED_PARENT_VALUE,
    load_admitted_parent,
    local_to_physical,
    parent_measure_ratio_bridge,
    parent_measure_ratio_bridge_autodiff,
)


def test_ratio_bridge_origin_preserves_parent_value() -> None:
    parent = load_admitted_parent()
    local = parent.transport().inverse_transport(parent.frozen_reference_points)
    physical = local_to_physical(local, parent)
    result = parent_measure_ratio_bridge(
        tf.zeros([3], tf.float64), physical, parent=parent
    )
    tf.debugging.assert_near(result["log_value"], tf.constant(EXPECTED_PARENT_VALUE, tf.float64), atol=2e-13)
    tf.debugging.assert_equal(result["log_value_ratio"], tf.constant(0.0, tf.float64))
    assert float(result["effective_sample_size"].numpy()) == 16.0


def test_ratio_bridge_manual_score_matches_autodiff_same_finite_program() -> None:
    parent = load_admitted_parent()
    local = parent.transport().inverse_transport(parent.frozen_reference_points)
    physical = local_to_physical(local, parent)
    theta = tf.constant([0.02, -0.01, 0.03], tf.float64)
    manual = parent_measure_ratio_bridge(theta, physical, parent=parent)
    autodiff = parent_measure_ratio_bridge_autodiff(theta, physical)
    tf.debugging.assert_near(manual["log_value"], autodiff["log_value"], atol=1e-12, rtol=1e-12)
    tf.debugging.assert_near(manual["score"], autodiff["score"], atol=1e-10, rtol=1e-10)
