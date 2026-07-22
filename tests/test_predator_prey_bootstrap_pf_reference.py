from __future__ import annotations

import ast
import inspect
import os
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.testing.predator_prey_bootstrap_pf_reference_tf import (
    predator_prey_bootstrap_pf_reference,
)
from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
    PP_PARAMETER_LOWER,
    PP_PARAMETER_UPPER,
    PP_TRUTH_PHYSICAL,
    generate_frozen_predator_prey_dataset_tf,
)


def _truth_source() -> tf.Tensor:
    probability = (PP_TRUTH_PHYSICAL - PP_PARAMETER_LOWER) / (
        PP_PARAMETER_UPPER - PP_PARAMETER_LOWER
    )
    return (
        tf.sqrt(tf.constant(2.0, tf.float64))
        * tf.math.erfinv(2.0 * probability - 1.0)
    )[None, :]


def test_pf_is_deterministic_finite_and_seed_batched() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    seeds = tf.constant([81400, 81401], tf.int32)
    first = predator_prey_bootstrap_pf_reference(
        _truth_source(), observations=observations[:3], seeds=seeds, num_particles=256
    )
    second = predator_prey_bootstrap_pf_reference(
        _truth_source(), observations=observations[:3], seeds=seeds, num_particles=256
    )
    tf.debugging.assert_equal(first["log_likelihood"], second["log_likelihood"])
    assert bool(tf.reduce_all(first["finite"]).numpy()) is True
    assert bool(tf.reduce_all(first["minimum_ess"] > 0.0).numpy()) is True
    assert first["log_likelihood"].shape == (2,)


def test_inverse_cdf_multinomial_resampling_changes_seed_stream() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    result = predator_prey_bootstrap_pf_reference(
        _truth_source(),
        observations=observations[:3],
        seeds=tf.constant([81400, 81401], tf.int32),
        num_particles=512,
    )
    assert bool(
        tf.not_equal(result["log_likelihood"][0], result["log_likelihood"][1]).numpy()
    ) is True


def test_pf_xla_smoke() -> None:
    _states, observations = generate_frozen_predator_prey_dataset_tf()

    @tf.function(jit_compile=True)
    def compiled(theta):
        result = predator_prey_bootstrap_pf_reference(
            theta,
            observations=observations[:3],
            seeds=tf.constant([81400, 81401], tf.int32),
            num_particles=128,
        )
        return result["log_likelihood"], result["minimum_ess"], result["finite"]

    value, ess, finite = compiled(_truth_source())
    assert value.shape == (2,)
    assert bool(tf.reduce_all(ess > 0.0).numpy()) is True
    assert bool(tf.reduce_all(finite).numpy()) is True


def test_pf_active_path_has_no_python_time_or_particle_loop_or_numpy() -> None:
    source = textwrap.dedent(inspect.getsource(predator_prey_bootstrap_pf_reference))
    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
    assert ".numpy(" not in source
    assert "numpy" not in source
    assert "tf.while_loop" in source
