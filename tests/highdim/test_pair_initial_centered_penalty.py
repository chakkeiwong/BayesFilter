"""CPU reference checks of the optional SGQF-anchor regression objective."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pytest
import tensorflow as tf
from bayesfilter.highdim import pair_block_tt_tf as pair

D = tf.float64


def test_initial_centered_penalty_matches_one_core_closed_form():
    # Orthogonal features give G=I and an independently known proximal solution.
    features = tf.reshape(2*tf.eye(4, dtype=D), [4, 1, 2, 2])
    target = tf.constant([2., -4., 1., 0.], D)
    initial = (tf.reshape(tf.constant([.2, -.3, .8, .4], D), [1, 2, 2, 1]),)
    rhs = target/2
    for center in ('zero', 'initial'):
        anchor = tf.zeros([4], D) if center == 'zero' else tf.reshape(initial[0], [-1])
        for penalty in (0., .25, 10.):
            cores, info = pair.fit_pair_features(features, target, tf.ones([4], D),
                degree=1, rank=1, sweeps=1, proximal_steps=1, penalty=penalty,
                initial=initial, jit_compile=False, regularization_center=center)
            expected = anchor+tf.sign(rhs-anchor)*tf.maximum(tf.abs(rhs-anchor)-penalty, 0.)
            tf.debugging.assert_near(tf.reshape(cores[0], [-1]), expected, atol=1e-14)
            assert float(info['kkt_residual']) < 1e-14
            assert float(info['minimum_objective_decrease']) >= -1e-14


def test_zero_penalty_and_default_preserve_existing_multicore_fit():
    rows = tf.random.stateless_normal([80, 2, 2], [99, 1], dtype=D)
    feat = pair._pair_features(rows, 1)
    target = tf.exp(.1*rows[:, 0, 0]-.15*rows[:, 1, 1])
    kwargs = dict(degree=1, rank=2, sweeps=2, proximal_steps=8,
                  initial=pair.initial_pair_cores(2, 1, 2), jit_compile=False)
    base, _ = pair.fit_pair_features(feat, target, tf.ones([80], D), **kwargs)
    zero, _ = pair.fit_pair_features(feat, target, tf.ones([80], D), **kwargs,
                                    regularization_center='zero')
    anchored, _ = pair.fit_pair_features(feat, target, tf.ones([80], D), **kwargs,
                                        regularization_center='initial')
    for a, b, c in zip(base, zero, anchored):
        tf.debugging.assert_equal(a, b)
        tf.debugging.assert_near(a, c, atol=2e-12, rtol=2e-12)


def test_unknown_center_fails_before_tracing():
    with pytest.raises(ValueError, match='regularization_center'):
        pair.compiled_pair_fitter(1, regularization_center='unexamined')
