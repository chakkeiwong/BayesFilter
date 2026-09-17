"""Matched complete forecast/proposal diagnostics, separate from runtime code.

Both isolated source arms receive the same deterministic FP64 tensor inputs.
The C2 extension carries no source-faithfulness or canonical LEDH admission.
"""

FIXTURES = ("ssl_forecast", "complexity_forecast", "hermite_proposal")


def fixture(tf, name, size, jit):
    dtype = tf.float64
    count = 2 * size
    if name == "ssl_forecast":
        from bayesfilter.nonlinear import ssl_lstm_predictive_tf as forecast

        config = forecast.SSLLSTMForecastConfig()
        inputs = (
            tf.tile(tf.constant([[.35, -.08, .65, .05]], dtype), [count, 1]),
            tf.zeros([count, 3], dtype), tf.eye(3, batch_shape=[count], dtype=dtype) * .2,
            tf.reshape(tf.sin(tf.cast(tf.range(count * 2 * 3), dtype)), [count, 2, 3]),
            tf.ones([count, 2, 10, 1], dtype) * .12,
            tf.ones([count, 2, 10, 1], dtype) * -.23,
        )

        def evaluate(*values):
            return forecast._forecast_batch_core(*values, config)

        return evaluate, inputs, {"draws": count, "replications": 2, "horizon": 10,
            "state": 3, "parameters": 4, "role": "conditional_forecast_from_frozen_terminal_states"}

    if name == "complexity_forecast":
        from bayesfilter.nonlinear import ssl_lstm_complexity_predictive_tf as forecast
        from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
            complexity_posterior_target,
        )

        target, horizon = complexity_posterior_target(1, jit_compile=jit), 2 * size
        program = forecast.complexity_forecast_compiled_program(
            target, draw_count=count, replication_count=2, horizon=horizon)
        inputs = (tf.tile(tf.constant([[.35, -.08, .65, .05]], dtype), [count, 1]),
            tf.ones([count, 2, 3], dtype) * .1,
            tf.ones([count, 2, horizon, 1], dtype) * -.15,
            tf.ones([count, 2, horizon], dtype) * .2)
        return program.python_function, inputs, {"draws": count, "replications": 2,
            "horizon": horizon, "filter_horizon": 30, "state": 3, "parameters": 4,
            "role": "complete_terminal_filter_and_conditional_forecast"}

    if name == "hermite_proposal":
        from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
            GaussianHermiteRetainedProposal,
        )

        dimension, particles = 2 * size, 8
        ranks = (1, *([2] * (dimension - 1)), 3)
        cores = tuple(tf.reshape(.2 + .1 * tf.sin(tf.cast(tf.range(ranks[i] * 3 * ranks[i+1]), dtype)),
                                 [ranks[i], 3, ranks[i+1]]) for i in range(dimension))
        gram = tf.linalg.diag(tf.constant([1.2, .8, 1.1], dtype))
        # Independent frozen fixture contraction, outside the measured kernel.
        normalizer = gram
        for core in reversed(cores):
            normalizer = tf.einsum("akb,ckd,bd->ac", core, core, normalizer)
        proposal = GaussianHermiteRetainedProposal(cores, gram, normalizer[0, 0],
            tf.constant(.04, dtype), tf.linspace(tf.constant(-.1, dtype), .2, dimension),
            tf.eye(dimension, dtype=dtype), 5., 1, "a" * 64)
        inputs = (tf.linspace(tf.constant(.1, dtype), .9, particles),
            tf.reshape(tf.linspace(tf.constant(.03, dtype), .97, particles * dimension), [particles, dimension]),
            tf.reshape(tf.sin(tf.cast(tf.range(particles * dimension), dtype)), [particles, dimension]))
        return proposal.sample_physical, inputs, {"particles": particles, "dimension": dimension,
            "degree": 2, "ranks": ranks, "bisection_iterations": 64, "defensive_nu": 5.,
            "role": "extension_or_invention_proposal_only", "canonical_admitted": False}
    raise ValueError(f"Unregistered forecast fixture: {name}")
