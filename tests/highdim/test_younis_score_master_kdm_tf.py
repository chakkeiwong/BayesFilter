"""All model/initial/bandwidth terms through the full KDM consumers."""
import pytest
import tensorflow as tf

from bayesfilter.score_study.kdm_adapter_tf import make_kdm_kernel
from tests.highdim.test_younis_score_master_canonical_tf import CONTROLS


@pytest.mark.parametrize("representation", ["integrated_kdm", "resampling_kdm"])
def test_total_kdm_score_matches_its_own_finite_scalar(representation):
    dtype = tf.float64
    theta = tf.constant([.62, -.8, -.6, .9, .25, -.3], dtype)
    observations = tf.constant([[.5], [-.2]], dtype)
    args = (observations,
            tf.random.stateless_normal([8, 2], [3, 7], dtype=dtype),
            tf.random.stateless_normal([2, 8, 2], [3, 8], dtype=dtype),
            tf.random.stateless_normal([8, 2], [3, 9], dtype=dtype),
            tf.random.stateless_uniform([2, 8], [4, 9], dtype=dtype),
            tf.random.stateless_normal([2, 8, 2], [5, 9], dtype=dtype),
            tf.constant(.2, dtype))
    settings = (2, 1, 8, 2, tuple(sorted(CONTROLS.items())), representation)
    anchor = make_kdm_kernel(*settings)
    replay = make_kdm_kernel(*settings, replay=True) if representation == "resampling_kdm" else anchor
    anchor_result = anchor(theta, tf.eye(6, dtype=dtype)[0], *args)
    bank = anchor_result[3:]
    scores, differences = [], []
    for direction in tf.unstack(tf.eye(6, dtype=dtype)):
        value, score, valid, *_ = replay(theta, direction, *args, *bank)
        assert bool(valid)
        tf.debugging.assert_near(value, anchor_result[0], atol=1e-11)
        h = tf.constant(2e-5, dtype)
        plus = replay(theta + h*direction, direction, *args, *bank)[0]
        minus = replay(theta - h*direction, direction, *args, *bank)[0]
        scores.append(score)
        differences.append((plus-minus)/(2*h))
    tf.debugging.assert_near(tf.stack(scores), tf.stack(differences), atol=3e-5, rtol=3e-4)
    assert anchor.experimental_get_tracing_count() == 1
    assert replay.experimental_get_tracing_count() == 1
