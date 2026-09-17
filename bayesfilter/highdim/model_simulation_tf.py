"""Complete XLA model simulators preserving the original Philox call order."""

from collections import OrderedDict

import tensorflow as tf

from bayesfilter.ops.generator_stream_tf import generator_seed_state, normal_call_program

_SIMULATION_CACHE = OrderedDict()


def model_simulation_program(model, count, family, *, jit_compile=True):
    key = (id(model), count, family, bool(jit_compile))
    if key in _SIMULATION_CACHE:
        _SIMULATION_CACHE.move_to_end(key)
        return _SIMULATION_CACHE[key][1]
    n, m = model.state_dim(), model.observation_dim()
    p = model.parameter_dim() if family != "sir" else 0
    generate = normal_call_program(count, n, m, jit_compile=jit_compile).python_function

    @tf.function(input_signature=[tf.TensorSpec([p], tf.float64), tf.TensorSpec([3], tf.uint64)],
                 jit_compile=jit_compile, autograph=False)
    def simulate(theta, seed_state):
        innovations, errors = generate(seed_state)
        if family == "sv":
            parameters = model.physical_parameters(theta)
            gamma, beta = parameters["gamma"], parameters["beta"]
            state = model.sigma / tf.sqrt(1.0-tf.square(gamma)) * innovations[0]

            def observe(state, error):
                return beta * tf.exp(0.5*state) * error

            def transition(state, innovation):
                return gamma * state + model.sigma * innovation
        else:
            initial_chol = tf.linalg.cholesky(model.initial_covariance)
            process_chol = tf.linalg.cholesky(model.process_covariance)
            observation_chol = tf.linalg.cholesky(model.observation_covariance)
            state = model.initial_mean + tf.linalg.matvec(initial_chol, innovations[0])

            def observe(state, error):
                center = model.infectious_components(state)[0] if family == "sir" else state
                return center + tf.linalg.matvec(observation_chol, error)

            def transition(state, innovation):
                mean = model.transition_mean(state)[0] if family == "sir" else model.transition_mean(theta, state)[0]
                following = mean + tf.linalg.matvec(process_chol, innovation)
                return model._apply_process_noise_policy(following[None])[0] if family == "sir" else following

        states = tf.TensorArray(tf.float64, count, element_shape=[n]).write(0, state)
        observations = tf.TensorArray(tf.float64, count, element_shape=[m]).write(0, observe(state, errors[0]))

        def step(index, state, states, observations):
            state = transition(state, innovations[index])
            return index+1, state, states.write(index, state), observations.write(index, observe(state, errors[index]))

        _, _, states, observations = tf.while_loop(lambda index, *_: index < count, step,
            (tf.constant(1), state, states, observations), maximum_iterations=count-1, parallel_iterations=1)
        return states.stack(), observations.stack()

    _SIMULATION_CACHE[key] = (model, simulate)
    if len(_SIMULATION_CACHE) > 16:
        _SIMULATION_CACHE.popitem(last=False)
    return simulate


def simulate_model(model, theta, final_time, seed, family, *, jit_compile=True):
    if int(final_time) < 0:
        raise ValueError("final_time must be nonnegative")
    return model_simulation_program(model, int(final_time)+1, family, jit_compile=jit_compile)(
        tf.convert_to_tensor(theta, tf.float64), generator_seed_state(seed))
