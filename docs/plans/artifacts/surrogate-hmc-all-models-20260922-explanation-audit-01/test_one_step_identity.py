"""CPU-only diagnostic: one-step corrected HMC versus Gaussian proposal MH.

Independent proposal-density calculation; no inference or performance claim.
"""

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce,
    NeuralForceHMCConfig,
    kinetic_energy,
    neural_force_proposal,
)


def test_one_step_nonconservative_field_has_gaussian_mh_correction():
    dtype = tf.float64
    inverse_mass = tf.constant([0.7, 1.6], dtype)
    step = tf.constant(0.23, dtype)
    z = tf.constant([[0.4, -0.7], [1.2, 0.1]], dtype)
    p = tf.constant([[-0.3, 0.9], [0.2, -1.1]], dtype)

    # The rotational component makes this a nonconservative position field.
    def g(q):
        rotation = tf.stack([-q[:, 1], q[:, 0]], axis=-1)
        return 0.3 * q + 0.6 * rotation + tf.constant([0.1, -0.2], dtype)

    def potential(q):
        return tf.reduce_sum(0.5 * tf.square(q) + 0.1 * q**4, axis=-1)

    field = FrozenPositionOnlyForce(g, identity="audit-fixed-rotational-field")
    config = NeuralForceHMCConfig(
        step_size=0.23,
        num_leapfrog_steps=1,
        inverse_mass_diagonal=(0.7, 1.6),
        dtype="float64",
    )
    proposal = neural_force_proposal(z, p, field, config)
    mean_forward = z - 0.5 * step**2 * inverse_mass * g(z)
    predicted_z = mean_forward + step * inverse_mass * p
    tf.debugging.assert_near(proposal.position, predicted_z, atol=1e-13, rtol=1e-13)

    # q(z'|z) has diagonal covariance h^2 M^{-1}; normalizers cancel.
    variance = step**2 * inverse_mass
    mean_reverse = proposal.position - 0.5 * variance * g(proposal.position)
    log_q_forward = -0.5 * tf.reduce_sum(
        tf.square(proposal.position - mean_forward) / variance, axis=-1
    )
    log_q_reverse = -0.5 * tf.reduce_sum(
        tf.square(z - mean_reverse) / variance, axis=-1
    )
    gaussian_mh_ratio = (
        potential(z) - potential(proposal.position) + log_q_reverse - log_q_forward
    )
    hmc_ratio = (
        potential(z) + kinetic_energy(p, config)
        - potential(proposal.position) - kinetic_energy(proposal.momentum, config)
    )
    tf.debugging.assert_near(hmc_ratio, gaussian_mh_ratio, atol=1e-13, rtol=1e-13)
    reversed_proposal = neural_force_proposal(
        proposal.position, proposal.momentum, field, config
    )
    tf.debugging.assert_near(reversed_proposal.position, z, atol=1e-13, rtol=1e-13)
    tf.debugging.assert_near(reversed_proposal.momentum, p, atol=1e-13, rtol=1e-13)
