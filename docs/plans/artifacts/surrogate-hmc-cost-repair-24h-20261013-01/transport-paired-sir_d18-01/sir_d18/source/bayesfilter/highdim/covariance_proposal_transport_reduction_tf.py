"""Candidate shared stable reduction for finite transport logits.

Combines log mass and conditional weights without repeating exponentials.
This module is inactive until endpoint parity and paired GPU costs pass.
"""
import tensorflow as tf


def logsumexp_and_weights(logits, axis=-1):
    maximum = tf.reduce_max(logits, axis, keepdims=True)
    origin = tf.where(tf.math.is_finite(maximum), maximum, tf.zeros_like(maximum))
    exponentials = tf.exp(logits-origin)
    mass = tf.reduce_sum(exponentials, axis, keepdims=True)
    return tf.squeeze(origin+tf.math.log(mass), axis), exponentials/mass
