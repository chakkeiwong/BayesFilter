"""Training-consumer parity for compiled conditional-ratio score preparation."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import (
    zhao_cui_austria_sir_parameter_density_training_tf as candidate,
)
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@pytest.mark.parametrize("rows,samples,seed", [(2, 8, 1729), (4, 16, -17)])
def test_complete_prefix_training_targets_preserve_seeded_estimates(rows, samples, seed):
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    global_score = candidate.RatioScoreEstimate(value=tf.constant(.1, D), score=tf.constant([.2, -.1, .3], D),
        score_standard_error=tf.constant([.02, .03, .01], D), effective_sample_size=tf.constant(6., D))
    mean = candidate._BASE_MODEL.transition_mean(candidate._INITIAL_MEAN[None])[0]
    points = mean[None] + tf.reshape(.2 * tf.sin(tf.cast(tf.range(rows * 18), D)), [rows, 18])
    options = {"prefix_points": points, "global_score": global_score, "sample_count": samples, "seed": seed}
    actual, expected = candidate.estimate_t1_prefix_scores(**options), before.estimate_t1_prefix_scores(**options)
    for result, authority in zip(actual, expected, strict=True):
        for name in ("value", "score", "score_standard_error", "effective_sample_size"):
            np.testing.assert_allclose(getattr(result, name), getattr(authority, name), atol=1e-10, rtol=1e-10)
    seed_tensor = tf.constant([seed, 991], tf.int32)
    noise = candidate._prefix_noise_program(samples)(seed_tensor)
    np.testing.assert_allclose(noise, tf.random.stateless_normal([samples, 18], seed_tensor, dtype=D), atol=1e-14, rtol=1e-13)
    observation = candidate.generate_sealed_lane_b_dataset()[1][0]
    program = candidate._prefix_score_program(rows, samples)
    assert "HloModule" in program.experimental_get_compiler_ir(points, global_score.score,
        global_score.score_standard_error, noise, observation)(stage="hlo")
    assert program.experimental_get_tracing_count() == 1
    _graph(program)
    # A perturbation of one query must not change another query's estimate.
    changed = program(tf.concat([points[:1] + .1, points[1:]], 0), global_score.score,
                      global_score.score_standard_error, noise, observation)
    reference = program(points, global_score.score, global_score.score_standard_error, noise, observation)
    for value, authority in zip(changed, reference, strict=True):
        np.testing.assert_allclose(value[1:], authority[1:], atol=1e-12, rtol=1e-12)


def test_public_prefix_target_has_pure_enclosing_graph_and_xla_boundaries():
    mean = candidate._BASE_MODEL.transition_mean(candidate._INITIAL_MEAN[None])[0]
    points = tf.stack([mean, mean + .1])
    estimate = candidate.RatioScoreEstimate(value=tf.constant(.1, D), score=tf.constant([.2, -.1, .3], D),
        score_standard_error=tf.constant([.02, .03, .01], D), effective_sample_size=tf.constant(6., D))

    def evaluate(points):
        outputs = candidate.estimate_t1_prefix_scores(prefix_points=points, global_score=estimate,
            sample_count=8, seed=1729)
        return tuple((row.value, row.score, row.score_standard_error, row.effective_sample_size) for row in outputs)

    graph = tf.function(evaluate, input_signature=[tf.TensorSpec([2, 18], D)], jit_compile=False, autograph=False)
    compiled = tf.function(evaluate, input_signature=graph.input_signature, jit_compile=True, autograph=False)
    for actual, reference in zip(tf.nest.flatten(compiled(points)), tf.nest.flatten(graph(points)), strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)
    definition = graph.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in definition.library.function)
    _graph(graph)
    assert "HloModule" in compiled.experimental_get_compiler_ir(points)(stage="hlo")
