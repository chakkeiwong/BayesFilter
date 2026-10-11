"""Internal stream-preserving batching of independent fixed-horizon HMC trials.

This runner has no tuning authority. It uses the installed TFP seed schedule,
leapfrog integrator, kinetic correction and MH arithmetic. Only target rows are
batched; each trial retains the seed and random array shapes of a four-chain
``ReusableFullChainHMCRunner`` call. There is no pfor or scalar target loop.
"""
from __future__ import annotations

import time


def dependency_paths():
    """Bind the installed TFP internals used by this optional execution path."""
    from tensorflow_probability.python.internal import samplers
    from tensorflow_probability.python.mcmc import hmc, metropolis_hastings, sample
    from tensorflow_probability.python.mcmc.internal import leapfrog_integrator, util
    return tuple(module.__file__ for module in
                 (samplers, hmc, metropolis_hastings, sample, leapfrog_integrator, util))


class ReplicatedTrialBatchRunner:
    """Bound a finite batch shape; host code owns evidence and attempted cost."""

    def __init__(self, adapter, initial_state_template, config, *, batch_size):
        import tensorflow as tf
        from .hmc import ReusableFullChainHMCRunner

        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError('trial batch size must be an integer in [1, 32]')
        if (config.num_burnin_steps or config.tuning_policy.uses_dual_averaging
                or config.require_finite_transitions or config.capture_first_failure
                or config.chain_execution_mode == 'eager'):
            raise ValueError('trial batching requires an unadapted traced fixed-horizon kernel')
        self.reference = ReusableFullChainHMCRunner(adapter, initial_state_template, config,
                                                   dynamic_num_leapfrog_steps=True)
        if len(self.reference.state_shape) != 2:
            raise ValueError('trial batching requires a chain-by-parameter state')
        self.batch_size = batch_size
        self.config = config
        self.shape = (batch_size, *self.reference.state_shape)
        self.dtype = tf.as_dtype(self.reference.state_dtype)
        self._runner = self._build_runner()

    def _build_runner(self):
        import tensorflow as tf
        import tensorflow_probability as tfp
        from tensorflow_probability.python.internal import samplers
        from tensorflow_probability.python.mcmc import hmc as tfp_hmc
        from tensorflow_probability.python.mcmc.internal import leapfrog_integrator, util
        from .hmc_status import cache_hmc_target_status

        batch, chains, dimension = self.shape
        config = self.config
        target = self.reference._target_log_prob
        trace_fn = self.reference._trace_fn

        def map_random_rows(function, rows, signature):
            # map_fn lowers to one while-loop body, never pfor. It maps only
            # RNG operations, not target/gradient evaluation. Event shapes and
            # per-trial seeds remain identical to the sequential TFP call.
            return tf.map_fn(function, rows, fn_output_signature=signature,
                             parallel_iterations=1)

        def compiled(states, seeds, epsilon, leapfrog_steps):
            # TFP sample_chain sanitizes each root with this salt, and splits
            # (step, carry) once per transition. A tensor seed stays stateless.
            seeds = map_random_rows(lambda seed: samplers.sanitize_seed(seed, salt='mcmc.sample_chain'),
                                    seeds, tf.TensorSpec([2], tf.int32))
            state = tf.reshape(states, [batch*chains, dimension])
            base = tfp.mcmc.HamiltonianMonteCarlo(target, epsilon, leapfrog_steps)
            integrator = leapfrog_integrator.SimpleLeapfrogIntegrator(target, [epsilon], leapfrog_steps)

            class IndependentSeedKernel(tfp.mcmc.TransitionKernel):
                @property
                def is_calibrated(self):
                    return True

                def bootstrap_results(self, initial):
                    return base.bootstrap_results(initial)

                def one_step(self, current, previous, seed=None):
                    def random_inputs(trial_seed):
                        proposal_seed, acceptance_seed = samplers.split_seed(trial_seed)
                        momentum_seed = samplers.split_seed(proposal_seed, n=1)[0]
                        return (samplers.normal([chains, dimension], dtype=self_dtype,seed=momentum_seed),
                                samplers.uniform([chains], dtype=self_dtype,seed=acceptance_seed))
                    momenta, uniforms = map_random_rows(random_inputs,seed,
                        (tf.TensorSpec([chains,dimension],self_dtype),tf.TensorSpec([chains],self_dtype)))
                    momentum = tf.reshape(momenta, [batch*chains, dimension])
                    accepted = previous.accepted_results
                    next_momentum, proposal, value, gradient = integrator(
                        [momentum], [current], accepted.target_log_prob, accepted.grads_target_log_prob)
                    correction = tfp_hmc._compute_log_acceptance_correction(
                        [momentum], next_momentum, independent_chain_ndims=1)
                    proposed = previous.proposed_results._replace(
                        log_acceptance_correction=correction, target_log_prob=value,
                        grads_target_log_prob=gradient, initial_momentum=[momentum],
                        final_momentum=next_momentum)
                    ratio = util.safe_sum([value, -accepted.target_log_prob, correction])
                    accepted_mask = tf.math.log(tf.reshape(uniforms, [batch*chains])) < ratio
                    updated = previous._replace(
                        accepted_results=util.choose(accepted_mask, util.strip_seeds(proposed), accepted),
                        is_accepted=accepted_mask, log_accept_ratio=ratio,
                        proposed_state=proposal[0], proposed_results=proposed, extra=[])
                    return util.choose(accepted_mask, proposal[0], current), updated

            self_dtype = states.dtype
            kernel = IndependentSeedKernel()
            if config.capture_candidate_health and config.target_status_trace_policy == 'per_chain_step':
                kernel = cache_hmc_target_status(kernel, self.reference.adapter.target_status_telemetry)
            results = kernel.bootstrap_results(state)
            trace_template = trace_fn(state, results)
            samples = tf.TensorArray(self_dtype, size=config.num_results,
                                     element_shape=[batch*chains, dimension])
            traces = tf.nest.map_structure(lambda value: tf.TensorArray(
                value.dtype, size=config.num_results, element_shape=value.shape), trace_template)

            def body(index, current, previous, carry, sample_array, trace_arrays):
                split = map_random_rows(lambda seed: tf.stack(samplers.split_seed(seed)),carry,
                                        tf.TensorSpec([2,2],tf.int32))
                next_state, next_results = kernel.one_step(current, previous, seed=split[:,0])
                trace = trace_fn(next_state, next_results)
                return (index+1, next_state, next_results, split[:,1],
                    sample_array.write(index, next_state),
                    tf.nest.map_structure(lambda array, value: array.write(index, value), trace_arrays, trace))

            _, _, _, _, samples, traces = tf.while_loop(
                lambda index, *_: index < config.num_results, body,
                (tf.constant(0), state, results, seeds, samples, traces), parallel_iterations=1)

            def grouped(value):
                value = value.stack()
                return tf.reshape(value, [config.num_results, batch, chains, *value.shape[2:]])

            return grouped(samples), tf.nest.map_structure(grouped, traces)

        return tf.function(compiled, input_signature=[
            tf.TensorSpec(self.shape, self.dtype),
            tf.TensorSpec([batch,2], tf.int32),
            tf.TensorSpec([], self.dtype), tf.TensorSpec([], tf.int32)],
            autograph=False, jit_compile=config.use_xla)

    def run(self, *, states, seeds, step_size, num_leapfrog_steps):
        import tensorflow as tf
        states = tf.convert_to_tensor(states, self.dtype)
        seeds = tf.convert_to_tensor(seeds, tf.int32)
        if states.shape != self.shape or seeds.shape != (self.batch_size,2):
            raise ValueError('trial batch state or seed shape mismatch')
        rows = seeds.numpy().tolist()
        if len({tuple(row) for row in rows}) != self.batch_size:
            raise ValueError('independent trials cannot share a root seed')
        epsilon = tf.convert_to_tensor(step_size, self.dtype)
        steps = tf.convert_to_tensor(num_leapfrog_steps, tf.int32)
        if epsilon.shape.rank != 0 or steps.shape.rank != 0:
            raise ValueError('step size and leapfrog count must be scalars')
        tf.debugging.assert_positive(epsilon)
        tf.debugging.assert_all_finite(epsilon, 'nonfinite step size')
        tf.debugging.assert_positive(steps)
        started = time.monotonic()
        samples, trace = self._runner(states, seeds, epsilon, steps)
        samples.numpy()  # Include pending device work in the diagnostic timer.
        return samples, trace, {'sample_chain_call_s':time.monotonic()-started,
            'trial_batch_size':self.batch_size, 'jit_compile':self.config.use_xla,
            'use_xla':self.config.use_xla,
            'seed_layout':'independent_original_tfp_sample_chain_streams_v1',
            'runtime':'tfp_leapfrog_independent_trial_batch',
            'capture_candidate_health':self.config.capture_candidate_health}
