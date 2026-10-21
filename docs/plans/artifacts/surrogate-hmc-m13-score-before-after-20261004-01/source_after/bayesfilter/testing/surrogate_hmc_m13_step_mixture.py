"""Scoped M13 candidate: a fixed mixture using the existing general HMC kernel.

This module supports the verified native TensorFlow tuning/retained lane only.
It changes neither the supplied position-only force nor the MH endpoint.
There is no separate leapfrog implementation here.
"""
from dataclasses import replace
import hashlib
import inspect
import math
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.neural_force_hmc import (
    NeuralForceTransitionKernel, PositionOnlyForceEvaluation,
    TargetPotentialEvaluation, _evaluate_force, _evaluate_endpoint_potential,
)
from bayesfilter.inference.tuning_contract import (
    HMCTuningRunnerBinding, _issue_hmc_tuning_runner_binding,
)


def checked_multipliers(values):
    values = tuple(values)
    if (not values or len(values) > 4 or any(isinstance(v, bool) for v in values)):
        raise ValueError("one to four positive fixed multipliers are required")
    values = tuple(float(v) for v in values)
    if any(not math.isfinite(v) or v <= 0 or v > 1 for v in values):
        raise ValueError("step multipliers must be finite and in (0,1]")
    if len(set(values)) != len(values):
        raise ValueError("duplicate step multipliers are not allowed")
    return values


def draw_scales(seed, chain_count, multipliers, dtype=tf.float64):
    """Independent per-chain choices, disjoint from transition random numbers."""
    choice_seed, transition_seed = tf.unstack(
        tf.random.experimental.stateless_split(tf.convert_to_tensor(seed, tf.int32), 2))
    choices = tf.random.stateless_uniform(
        [chain_count, 1], choice_seed, minval=0, maxval=len(multipliers), dtype=tf.int32)
    return tf.gather(tf.constant(multipliers, dtype), choices), transition_seed


def scaled_step(kernel, current_state, previous_results, scale, seed):
    """Apply the same kernel in a temporary constant diagonal chart q=s*z."""
    dimension = len(kernel.config.inverse_mass_diagonal)

    def force(z):
        value, fallback = _evaluate_force(
            kernel.force, z * scale, dimension=dimension, dtype=z.dtype)
        return PositionOnlyForceEvaluation(value * scale, ~fallback)

    def endpoint(z):
        value, status = _evaluate_endpoint_potential(
            kernel.target, z * scale, dtype=z.dtype, return_status=True)
        return TargetPotentialEvaluation(value, status)

    transformed = NeuralForceTransitionKernel(
        force=replace(kernel.force, function=force,
                      identity=kernel.force.identity + ":constant-step-chart"),
        target=replace(kernel.target, function=endpoint,
                       identity=kernel.target.identity + ":constant-step-chart"),
        config=kernel.config,
    )
    next_z, results = transformed.one_step(
        current_state / scale, previous_results, seed=seed)
    return next_z * scale, results


class FixedStepMixtureKernel(tfp.mcmc.TransitionKernel):
    """An unchanged base step size plus fixed independent multiplier draws."""

    def __init__(self, kernel, multipliers, name="m13_fixed_step_mixture"):
        if not isinstance(kernel, NeuralForceTransitionKernel):
            raise TypeError("mixture requires the general neural-force HMC kernel")
        self._parameters = dict(
            kernel=kernel, multipliers=checked_multipliers(multipliers), name=name)

    @property
    def parameters(self):
        return dict(self._parameters)

    @property
    def is_calibrated(self):
        return True

    def bootstrap_results(self, init_state):
        return self._parameters["kernel"].bootstrap_results(init_state)

    def one_step(self, current_state, previous_kernel_results, seed=None):
        if seed is None:
            raise ValueError("step mixture requires a stateless seed")
        kernel = self._parameters["kernel"]
        multipliers = self._parameters["multipliers"]
        if multipliers == (1.,):
            return kernel.one_step(current_state, previous_kernel_results, seed=seed)
        scales, transition_seed = draw_scales(
            seed, tf.shape(current_state)[0], multipliers, current_state.dtype)
        return scaled_step(kernel, current_state, previous_kernel_results, scales,
                           transition_seed)


def bind_step_mixture(base, multipliers):
    """Repository-owned optional wrapper; the public tuner owns all admission."""
    if not isinstance(base, HMCTuningRunnerBinding):
        raise TypeError("an issued HMC binding is required")
    if base.algorithm_family != "endpoint_corrected_frozen_position_force_hmc":
        raise ValueError("step mixture requires a typed endpoint-corrected force binding")
    multipliers = checked_multipliers(multipliers)
    if multipliers == (1.,):
        return base

    def unsupported_host_runner(*args, **kwargs):
        raise ValueError(
            "M13 fixed step mixture requires VerifiedTensorFlowHMCConfig and its "
            "native tensor lane; the legacy host runner is unsupported")

    def tensor_kernel_factory(*, adapter, step_size, num_leapfrog_steps):
        kernel = base.tensor_kernel_factory(
            adapter=adapter, step_size=step_size, num_leapfrog_steps=num_leapfrog_steps)
        return FixedStepMixtureKernel(kernel, multipliers)

    # Only the repository factory supplies identity. Bind both the actual base
    # dependency closure and this wrapper, including its immutable scale law.
    closure = dict(base.source_dependency_closure)
    path = Path(__file__).resolve()
    root = path.parents[2]
    closure["files"] = tuple(closure["files"]) + (
        dict(path=str(path.relative_to(root)),
             sha256=hashlib.sha256(path.read_bytes()).hexdigest()),)
    closure["fixed_step_mixture"] = dict(
        schema="bayesfilter.m13_fixed_step_mixture.v1",
        multipliers=multipliers, probabilities=tuple(1./len(multipliers) for _ in multipliers),
        choices="independent_per_chain_per_transition_stateless",
        execution_lane="verified_native_tensorflow_only",
        base_binding_hash=base.binding_hash,
        force_changed=False, endpoint_changed=False,
    )
    names = inspect.signature(_issue_hmc_tuning_runner_binding).parameters
    settings = {name: getattr(base, name) for name in names}
    settings.update(
        runner=unsupported_host_runner, tensor_kernel_factory=tensor_kernel_factory,
        runner_identity=__name__ + ".verified_native_tensorflow_step_mixture",
        supported_chain_execution_modes=("tf_function",),
        source_dependency_closure=closure,
    )
    return _issue_hmc_tuning_runner_binding(**settings)
