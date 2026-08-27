"""Focused CPU/reference tests for the corrected neural-force HMC kernel."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    InvalidNeuralForceHMCConfiguration,
    NeuralForceHMCConfig,
    bind_neural_force_hmc_tuning_runner,
    kinetic_energy,
    neural_force_hmc_transition,
    neural_force_proposal,
    run_full_chain_neural_force_hmc,
    sample_neural_force_hmc,
)
from bayesfilter.inference.hmc import FullChainHMCConfig, FullChainHMCRunResult
from bayesfilter.inference.hmc_tuning import HMCTuningPolicy
from bayesfilter.inference.tuning_contract import hmc_tuning_interface_capability


DTYPE = tf.float64


def _config(*, step_size: float = 0.15, steps: int = 4, dimension: int = 2):
    return NeuralForceHMCConfig(
        step_size=step_size,
        num_leapfrog_steps=steps,
        inverse_mass_diagonal=(1.0,) * dimension,
        dtype="float64",
    )


def _gaussian_force(scale: float = 1.0):
    return FrozenPositionOnlyForce(
        function=lambda position: tf.cast(scale, position.dtype) * position,
        identity=f"gaussian-force-{scale}",
    )


def _gaussian_target(dimension: int = 2):
    del dimension
    return FrozenTargetPotential(
        function=lambda position: 0.5 * tf.reduce_sum(tf.square(position), axis=-1),
        identity="standard-gaussian-target",
    )


class _IdentityAffineTransform:
    factor = tf.eye(2, dtype=DTYPE)
    center = tf.zeros((2,), dtype=DTYPE)


class _IdentityAffineAdapter:
    transform = _IdentityAffineTransform()

    @staticmethod
    def latent_to_position(value):
        return tf.convert_to_tensor(value, DTYPE)


def _bound_runner_config(*, target_scope: str = "neural-force-binding-test"):
    return FullChainHMCConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.1,
        num_leapfrog_steps=2,
        seed=(20260828, 31),
        chain_execution_mode="eager",
        target_scope=target_scope,
    )


def test_typed_tuning_binding_validates_identity_and_telemetry():
    binding = bind_neural_force_hmc_tuning_runner(
        force=_gaussian_force(),
        target=_gaussian_target(),
        target_scope="neural-force-binding-test",
    )
    result = binding(
        _IdentityAffineAdapter(),
        tf.zeros((4, 2), DTYPE),
        _bound_runner_config(),
    )

    payload = binding.payload()
    assert len(payload["binding_hash"]) == 64
    assert payload["artifact_authority"] is False
    assert payload["source_dependency_closure"]["files"]
    assert result.metadata["coordinate_route"] == "native_fixed_mass_affine"
    assert result.metadata["force_identity"] == binding.force_identity
    assert result.metadata["target_identity"] == binding.endpoint_target_identity
    capability = hmc_tuning_interface_capability(
        "run_full_chain_neural_force_hmc"
    )
    assert capability.interface_kind == "chain_runner"
    assert capability.artifact_authority is False

    missing = dict(result.diagnostics)
    missing.pop("divergence_count")
    bad_binding = replace(
        binding,
        runner=lambda _adapter, _state, _config: FullChainHMCRunResult(
            samples=result.samples,
            trace=result.trace,
            diagnostics=missing,
            metadata=result.metadata,
        ),
    )
    with pytest.raises(ValueError, match="required telemetry"):
        bad_binding(
            _IdentityAffineAdapter(),
            tf.zeros((4, 2), DTYPE),
            _bound_runner_config(),
        )


def test_typed_tuning_binding_rejects_identity_mass_fallback():
    binding = bind_neural_force_hmc_tuning_runner(
        force=_gaussian_force(),
        target=_gaussian_target(),
        target_scope="neural-force-binding-test",
    )

    with pytest.raises(ValueError, match="native affine"):
        binding(object(), tf.zeros((4, 2), DTYPE), _bound_runner_config())
    with pytest.raises(ValueError, match="target_scope mismatch"):
        binding(
            _IdentityAffineAdapter(),
            tf.zeros((4, 2), DTYPE),
            _bound_runner_config(target_scope="wrong-scope"),
        )


def test_typed_tuning_binding_rejects_coordinate_mismatch():
    transformed_target = FrozenTargetPotential(
        function=lambda position: 0.5 * tf.reduce_sum(
            tf.square(position), axis=-1
        ),
        identity="transformed-gaussian-target",
        coordinate_system="transformed",
        includes_chart_log_jacobian=True,
    )

    with pytest.raises(InvalidNeuralForceHMCConfiguration, match="same coordinate"):
        bind_neural_force_hmc_tuning_runner(
            force=_gaussian_force(),
            target=transformed_target,
            target_scope="neural-force-binding-test",
        )


def test_config_is_immutable_and_validated():
    config = _config()
    with pytest.raises(Exception):
        config.step_size = 0.2
    with pytest.raises(InvalidNeuralForceHMCConfiguration):
        _config(step_size=0.0)
    with pytest.raises(InvalidNeuralForceHMCConfiguration):
        NeuralForceHMCConfig(0.1, 1, (1.0, 0.0))
    with pytest.raises(InvalidNeuralForceHMCConfiguration):
        NeuralForceHMCConfig(0.1, 1, (1.0,), dtype="int32")


def test_proposal_with_momentum_flip_is_an_involution():
    config = _config(step_size=0.17, steps=7)
    position = tf.constant([[0.4, -0.7], [1.2, 0.1]], DTYPE)
    momentum = tf.constant([[-0.3, 0.9], [0.2, -1.1]], DTYPE)
    first = neural_force_proposal(position, momentum, _gaussian_force(0.73), config)
    second = neural_force_proposal(
        first.position, first.momentum, _gaussian_force(0.73), config
    )
    tf.debugging.assert_near(second.position, position, atol=2.0e-12, rtol=2.0e-12)
    tf.debugging.assert_near(second.momentum, momentum, atol=2.0e-12, rtol=2.0e-12)


def test_small_dimensional_proposal_jacobian_has_unit_magnitude():
    config = _config(step_size=0.13, steps=3, dimension=1)
    phase = tf.constant([0.4, -0.8], DTYPE)

    with tf.GradientTape() as tape:
        tape.watch(phase)
        proposal = neural_force_proposal(
            phase[:1][tf.newaxis, :],
            phase[1:][tf.newaxis, :],
            _gaussian_force(0.61),
            config,
        )
        output = tf.concat(
            [tf.reshape(proposal.position, [-1]), tf.reshape(proposal.momentum, [-1])],
            axis=0,
        )
    jacobian = tape.jacobian(output, phase)
    tf.debugging.assert_near(tf.abs(tf.linalg.det(jacobian)), tf.constant(1.0, DTYPE), atol=1e-12)


def test_full_energy_trace_and_cached_endpoint_call_count():
    calls = tf.Variable(0, trainable=False, dtype=tf.int32)

    def counted_target(position):
        calls.assign_add(1)
        return 0.5 * tf.reduce_sum(tf.square(position), axis=-1)

    target = FrozenTargetPotential(counted_target, identity="counted-gaussian")
    config = _config(step_size=0.31, steps=5)
    position = tf.constant([[0.3, -0.2], [1.0, 0.4]], DTYPE)
    current = 0.5 * tf.reduce_sum(tf.square(position), axis=-1)
    result = neural_force_hmc_transition(
        position, current, _gaussian_force(), target, config, seed=(19, 23)
    )
    tf.debugging.assert_equal(calls, 1)
    tf.debugging.assert_equal(result.trace.endpoint_call_count, 1)
    expected = (
        result.trace.final_potential
        + result.trace.final_kinetic
        - result.trace.initial_potential
        - result.trace.initial_kinetic
    )
    tf.debugging.assert_near(result.trace.delta_h, expected, atol=1e-14)
    position_only = result.trace.final_potential - result.trace.initial_potential
    assert bool(tf.reduce_any(tf.abs(position_only - result.trace.delta_h) > 1e-8))


def test_omitted_kinetic_rule_fails_known_swap_counterexample():
    position = tf.constant([0.2, 1.7, -0.4], DTYPE)
    momentum = tf.constant([1.3, -0.1, 0.9], DTYPE)
    correct_delta = 0.5 * tf.square(momentum) + 0.5 * tf.square(position)
    correct_delta -= 0.5 * tf.square(position) + 0.5 * tf.square(momentum)
    position_only_delta = 0.5 * tf.square(momentum) - 0.5 * tf.square(position)
    tf.debugging.assert_near(correct_delta, tf.zeros_like(correct_delta))
    assert bool(tf.reduce_any(tf.abs(position_only_delta) > 0.1))


def test_acceptance_probability_approaches_one_as_step_size_shrinks():
    position = tf.constant(
        [[-1.0, 0.7], [0.3, -0.8], [1.2, 0.4], [-0.5, -1.1]], DTYPE
    )
    current = 0.5 * tf.reduce_sum(tf.square(position), axis=-1)
    target = _gaussian_target()
    large = neural_force_hmc_transition(
        position,
        current,
        _gaussian_force(),
        target,
        _config(step_size=0.4, steps=8),
        seed=(7, 11),
    )
    small = neural_force_hmc_transition(
        position,
        current,
        _gaussian_force(),
        target,
        _config(step_size=0.01, steps=8),
        seed=(7, 11),
    )
    large_probability = tf.reduce_mean(tf.exp(large.trace.log_acceptance_ratio))
    small_probability = tf.reduce_mean(tf.exp(small.trace.log_acceptance_ratio))
    assert float(small_probability) > float(large_probability)
    assert float(small_probability) > 0.9999


def test_native_dual_averaging_preserves_endpoint_only_force_mechanics():
    endpoint_calls = tf.Variable(0, trainable=False, dtype=tf.int32)
    force_calls = tf.Variable(0, trainable=False, dtype=tf.int32)

    def endpoint(position):
        endpoint_calls.assign_add(1)
        return 0.5 * tf.reduce_sum(tf.square(position), axis=-1)

    def force_fn(position):
        force_calls.assign_add(1)
        return tf.convert_to_tensor(position, DTYPE)

    config = FullChainHMCConfig(
        num_results=8,
        num_burnin_steps=16,
        step_size=0.3,
        num_leapfrog_steps=3,
        seed=(20260718, 991),
        chain_execution_mode="eager",
        tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
            num_adaptation_steps=16,
            target_accept_prob=0.70,
            source="test_native_neural_force_dual_averaging",
        ),
    )
    result = run_full_chain_neural_force_hmc(
        object(),
        tf.zeros((4, 2), DTYPE),
        config,
        force=FrozenPositionOnlyForce(force_fn, "counted-native-gaussian-force"),
        target=FrozenTargetPotential(endpoint, "counted-native-gaussian-target"),
    )

    transition_count = config.num_results + config.num_burnin_steps
    assert int(endpoint_calls.numpy()) == transition_count + 1
    assert int(force_calls.numpy()) == transition_count * (
        config.num_leapfrog_steps + 1
    )
    assert result.trace["step_size"].shape == (config.num_results,)
    assert result.trace["divergence"].shape == (config.num_results, 4)
    assert result.trace["force_fallback"].shape == (config.num_results, 4)
    tf.debugging.assert_equal(
        result.diagnostics["divergence_count"], tf.constant(0, tf.int32)
    )
    tf.debugging.assert_equal(
        result.diagnostics["divergence_count_by_chain"], tf.zeros((4,), tf.int32)
    )
    tf.debugging.assert_equal(
        result.diagnostics["force_fallback_count"], tf.constant(0, tf.int32)
    )
    assert float(result.diagnostics["final_step_size"].numpy()) > 0.0
    assert result.diagnostics["acceptance_rate_semantics"] == (
        "mean_metropolis_acceptance_probability"
    )
    tf.debugging.assert_near(
        result.trace["log_accept_ratio"],
        -result.trace["delta_h"],
        atol=1.0e-14,
        rtol=1.0e-14,
    )
    assert result.metadata["endpoint_only_exact_value"] is True
    assert result.metadata["exact_filter_gradient_inside_leapfrog"] is False


def test_biased_position_only_force_preserves_gaussian_moments_after_correction():
    chains = 8
    initial = tf.linspace(tf.constant(-1.0, DTYPE), tf.constant(1.0, DTYPE), chains)
    initial = initial[:, tf.newaxis]
    force = _gaussian_force(0.55)
    target = _gaussian_target(1)
    config = _config(step_size=0.42, steps=5, dimension=1)

    @tf.function(reduce_retracing=True)
    def run(position):
        return sample_neural_force_hmc(
            position,
            0.5 * tf.reduce_sum(tf.square(position), axis=-1),
            force,
            target,
            config,
            num_warmup=2000,
            num_results=8000,
            seed=(2026, 717),
        )

    chain = run(initial)
    draws = chain.positions[2000:, :, 0]
    mean = tf.reduce_mean(draws)
    variance = tf.math.reduce_variance(draws)
    assert abs(float(mean)) < 0.055
    assert abs(float(variance) - 1.0) < 0.09
    assert float(tf.reduce_mean(tf.cast(chain.accepted[2000:], DTYPE))) > 0.35


def test_batch_permutation_replay_shape_and_dtype():
    position = tf.constant([[0.2, -0.1], [1.1, 0.7], [-0.3, 0.8]], DTYPE)
    potential = 0.5 * tf.reduce_sum(tf.square(position), axis=-1)
    config = _config(step_size=0.11, steps=4)
    first = neural_force_hmc_transition(
        position, potential, _gaussian_force(), _gaussian_target(), config, seed=(4, 9)
    )
    replay = neural_force_hmc_transition(
        position, potential, _gaussian_force(), _gaussian_target(), config, seed=(4, 9)
    )
    tf.debugging.assert_equal(first.position, replay.position)
    tf.debugging.assert_equal(first.trace.delta_h, replay.trace.delta_h)
    permutation = tf.constant([2, 0, 1])
    permuted = neural_force_hmc_transition(
        tf.gather(position, permutation),
        tf.gather(potential, permutation),
        _gaussian_force(),
        _gaussian_target(),
        config,
        seed=(4, 9),
    )
    # Stateless randomness is attached to batch slots, so replay is exact and
    # deterministic proposal mechanics commute with batch permutation.
    momentum = tf.constant([[0.4, 0.2], [-0.8, 0.1], [0.3, -0.7]], DTYPE)
    direct = neural_force_proposal(position, momentum, _gaussian_force(), config)
    direct_permuted = neural_force_proposal(
        tf.gather(position, permutation),
        tf.gather(momentum, permutation),
        _gaussian_force(),
        config,
    )
    tf.debugging.assert_near(
        direct_permuted.position, tf.gather(direct.position, permutation), atol=1e-14
    )
    assert first.position.shape == (3, 2)
    assert first.position.dtype == DTYPE
    assert permuted.position.dtype == DTYPE
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        neural_force_proposal(tf.ones([2], DTYPE), tf.ones([2], DTYPE), _gaussian_force(), config)


def test_invalid_force_and_map_apis_are_rejected():
    with pytest.raises(InvalidNeuralForceHMCConfiguration, match="second argument"):
        FrozenPositionOnlyForce(lambda position, momentum: position + momentum, "bad")
    with pytest.raises(InvalidNeuralForceHMCConfiguration, match="momentum-dependent"):
        FrozenPositionOnlyForce(lambda position: position, "bad", momentum_dependent=True)
    with pytest.raises(InvalidNeuralForceHMCConfiguration, match="direct neural"):
        FrozenPositionOnlyForce(lambda position: position, "bad", direct_state_update=True)
    with pytest.raises(InvalidNeuralForceHMCConfiguration, match="symmetric"):
        FrozenPositionOnlyForce(lambda position: position, "bad", symmetric_schedule=False)
    with pytest.raises(InvalidNeuralForceHMCConfiguration, match="frozen"):
        FrozenPositionOnlyForce(lambda position: position, "bad", frozen=False)


def test_nonfinite_force_uses_finite_fallback_and_records_telemetry():
    position = tf.constant([[0.2]], DTYPE)
    config = _config(dimension=1)
    target = _gaussian_target(1)
    bad_force = FrozenPositionOnlyForce(
        lambda value: tf.fill(tf.shape(value), tf.constant(float("nan"), DTYPE)),
        "nan-force",
    )
    result = neural_force_hmc_transition(
        position, tf.constant([0.02], DTYPE), bad_force, target, config, seed=(1, 2)
    )
    tf.debugging.assert_equal(result.trace.force_fallback, [True])
    tf.debugging.assert_equal(result.trace.divergence, [False])
    tf.debugging.assert_all_finite(result.position, "fallback position")
    tf.debugging.assert_all_finite(result.potential, "fallback potential")

    for bad_value in (float("nan"), -float("inf")):
        def bad_target_function(value):
            return tf.fill([tf.shape(value)[0]], tf.constant(bad_value, DTYPE))

        bad_target = FrozenTargetPotential(
            bad_target_function,
            identity=f"bad-target-{bad_value}",
        )
        with pytest.raises(tf.errors.InvalidArgumentError, match="finite or declared"):
            neural_force_hmc_transition(
                position,
                tf.constant([0.02], DTYPE),
                _gaussian_force(),
                bad_target,
                config,
                seed=(1, 2),
            )


def test_arithmetic_overflow_is_rejected_and_reported_as_divergence():
    position = tf.constant([[0.2]], DTYPE)
    config = _config(step_size=1.0, steps=1, dimension=1)
    huge_force = FrozenPositionOnlyForce(
        lambda value: tf.fill(tf.shape(value), tf.constant(1.0e308, DTYPE)),
        "huge-finite-force",
    )
    target = _gaussian_target(1)
    result = neural_force_hmc_transition(
        position,
        tf.constant([0.02], DTYPE),
        huge_force,
        target,
        config,
        seed=(1, 2),
    )
    tf.debugging.assert_equal(result.trace.divergence, [True])
    tf.debugging.assert_equal(result.trace.accepted, [False])
    tf.debugging.assert_equal(result.position, position)
    tf.debugging.assert_all_finite(result.trace.log_acceptance_ratio, "divergence log ratio")


def test_generated_nonfinite_momentum_is_rejected_without_an_assertion():
    position = tf.constant([[0.2]], DTYPE)
    target = _gaussian_target(1)
    force = _gaussian_force()
    # A subnormal inverse mass makes the stateless Gaussian scaling overflow.
    config = NeuralForceHMCConfig(
        step_size=0.1,
        num_leapfrog_steps=1,
        inverse_mass_diagonal=(1.0e-320,),
        dtype="float64",
    )
    result = neural_force_hmc_transition(
        position,
        tf.constant([0.02], DTYPE),
        force,
        target,
        config,
        seed=(3, 5),
    )
    tf.debugging.assert_equal(result.trace.divergence, [True])
    tf.debugging.assert_equal(result.trace.accepted, [False])
    tf.debugging.assert_equal(result.position, position)


def test_declared_positive_infinity_support_boundary_is_ordinary_rejection():
    position = tf.constant([[0.2]], DTYPE)
    target = FrozenTargetPotential(
        lambda value: tf.fill([tf.shape(value)[0]], tf.constant(float("inf"), DTYPE)),
        identity="declared-out-of-support",
    )
    result = neural_force_hmc_transition(
        position,
        tf.constant([0.02], DTYPE),
        _gaussian_force(),
        target,
        _config(dimension=1),
        seed=(1, 2),
    )
    tf.debugging.assert_equal(result.trace.accepted, [False])
    tf.debugging.assert_equal(result.trace.endpoint_out_of_support, [True])
    tf.debugging.assert_equal(result.trace.divergence, [False])
    tf.debugging.assert_equal(result.trace.finite_status, [False])
    tf.debugging.assert_equal(result.position, position)


def test_nonlinear_chart_requires_and_includes_log_jacobian():
    def transform(z):
        return z + 0.25 * tf.pow(z, 3)

    def complete_transformed_potential(z):
        theta = transform(z)
        log_jacobian = tf.reduce_sum(tf.math.log1p(0.75 * tf.square(z)), axis=-1)
        return 0.5 * tf.reduce_sum(tf.square(theta), axis=-1) - log_jacobian

    target = FrozenTargetPotential(
        complete_transformed_potential,
        identity="nonlinear-cubic-chart-complete-target",
        coordinate_system="transformed",
        includes_chart_log_jacobian=True,
    )
    z = tf.constant([[0.4], [-0.9]], DTYPE)
    expected = 0.5 * tf.reduce_sum(tf.square(transform(z)), axis=-1)
    expected -= tf.reduce_sum(tf.math.log1p(0.75 * tf.square(z)), axis=-1)
    tf.debugging.assert_near(target.function(z), expected, atol=1e-14)
    raw_only = 0.5 * tf.reduce_sum(tf.square(transform(z)), axis=-1)
    assert bool(tf.reduce_any(tf.abs(raw_only - expected) > 0.05))
    with pytest.raises(InvalidNeuralForceHMCConfiguration, match="log-Jacobian"):
        FrozenTargetPotential(
            lambda value: 0.5 * tf.reduce_sum(tf.square(transform(value)), axis=-1),
            identity="wrong-raw-only-target",
            coordinate_system="transformed",
            includes_chart_log_jacobian=False,
        )


def test_active_kernel_has_tensorflow_loops_and_no_numpy_or_host_callback():
    source = inspect.getsource(sample_neural_force_hmc)
    proposal_source = inspect.getsource(neural_force_proposal)
    module_source = inspect.getsource(inspect.getmodule(sample_neural_force_hmc))
    assert "tf.while_loop" in source
    assert "tf.while_loop" in proposal_source
    assert "import numpy" not in module_source
    assert "numpy_function" not in module_source
    assert "py_function" not in module_source


def test_warmup_is_retained_in_chain_archive():
    initial = tf.zeros([2, 1], DTYPE)
    chain = sample_neural_force_hmc(
        initial,
        tf.zeros([2], DTYPE),
        _gaussian_force(),
        _gaussian_target(1),
        _config(dimension=1),
        num_warmup=3,
        num_results=5,
        seed=(88, 99),
    )
    assert chain.positions.shape == (8, 2, 1)
    assert chain.divergence.shape == (8, 2)
    assert chain.force_fallback.shape == (8, 2)
    tf.debugging.assert_equal(chain.num_warmup, 3)


def test_kinetic_energy_uses_configured_inverse_mass():
    config = NeuralForceHMCConfig(0.1, 2, (0.5, 2.0), dtype="float64")
    momentum = tf.constant([[2.0, 3.0]], DTYPE)
    tf.debugging.assert_near(kinetic_energy(momentum, config), [10.0])
