"""Independent diagnostic of the fixture's wrapped angular observation law.

This exercises the existing callback; it is not an admitted runtime endpoint.
"""
import json
import math
from types import SimpleNamespace

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import (
    observation_log_density_tf, range_bearing_observation_tf,
)


def integration_probe(count):
    @tf.function(input_signature=[tf.TensorSpec([], tf.float64)], jit_compile=True)
    def probe(sigma):
        angles = -math.pi + (tf.cast(tf.range(count), tf.float64)+.5)*(2*math.pi/count)
        particles = tf.broadcast_to(tf.constant([[1., 0., 0., 0.]], tf.float64), [count, 4])
        radial = range_bearing_observation_tf(particles)[:, 0]
        observations = tf.stack([radial, angles], axis=1)
        fixture = SimpleNamespace(R=tf.linalg.diag(tf.stack([tf.constant(1., tf.float64), sigma**2])))
        radial_log_density = -.5*tf.math.log(tf.constant(2*math.pi, tf.float64))
        callback = tf.exp(observation_log_density_tf(fixture, particles, observations)-radial_log_density)

        def wrapped(aliases):
            shifts = tf.cast(tf.range(-aliases, aliases+1), tf.float64)*2*math.pi
            residuals = angles[:, None]+shifts[None, :]
            return tf.reduce_sum(tf.exp(-.5*(residuals/sigma)**2), axis=1)/(
                sigma*tf.sqrt(tf.constant(2*math.pi, tf.float64)))

        coarse, fine = wrapped(4), wrapped(8)
        return tf.stack([tf.reduce_sum(callback)*2*math.pi/count,
                         tf.reduce_sum(fine)*2*math.pi/count,
                         tf.reduce_max(tf.abs(coarse-fine)),
                         tf.reduce_max(tf.abs(callback-fine))])
    return probe


def test_wrapped_simulator_law_and_callback_are_different_at_wide_noise():
    results = [integration_probe(n)(tf.constant(math.pi, tf.float64)).numpy().tolist()
               for n in (4096, 8192)]
    expected = math.erf(1/math.sqrt(2))
    for callback_mass, wrapped_mass, alias_change, _ in results:
        assert abs(callback_mass-expected) < 1e-7
        assert abs(wrapped_mass-1.) < 1e-10
        assert alias_change < 1e-12
    assert abs(results[0][0]-results[1][0]) < 1e-7
    assert results[1][0] < .69
    print(json.dumps(dict(sigma=math.pi, expected_callback_mass=expected,
                          midpoint_counts=[4096, 8192], results=results)))


def test_original_narrow_noise_is_a_no_fire_control():
    callback_mass, wrapped_mass, alias_change, maximum_difference = integration_probe(4096)(
        tf.constant(.04, tf.float64)).numpy().tolist()
    assert abs(callback_mass-1.) < 1e-10
    assert abs(wrapped_mass-1.) < 1e-10
    assert alias_change < 1e-12
    assert maximum_difference < 1e-10
    print(json.dumps(dict(sigma=.04, callback_mass=callback_mass,
                          wrapped_mass=wrapped_mass, maximum_difference=maximum_difference)))
