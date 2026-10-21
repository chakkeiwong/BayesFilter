"""CPU-only mechanics: compare the mixture to the existing scalar-step kernel."""
from pathlib import Path
from dataclasses import replace
import pytest
import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, NeuralForceHMCConfig,
    NeuralForceTransitionKernel, PositionOnlyForceEvaluation, TargetPotentialEvaluation,
    bind_neural_force_hmc_tuning_runner,
)
from bayesfilter.testing.surrogate_hmc_m13_step_mixture import (
    FixedStepMixtureKernel, bind_step_mixture, checked_multipliers, draw_scales, scaled_step,
)

F64 = tf.float64


def fixture_kernel():
    def force(x):
        # Deliberately not the endpoint gradient: test the actual surrogate contract.
        return PositionOnlyForceEvaluation(x + tf.constant([.4, -.3], F64),
                                            tf.ones(tf.shape(x)[0], tf.bool))
    def target(x):
        value = .5 * tf.reduce_sum(x*x, axis=-1) + .1*x[:, 0]*x[:, 1]
        good = tf.ones(tf.shape(value), tf.bool)
        return TargetPotentialEvaluation(value, dict(value_valid=good, force_valid=good,
            numerical_failure=~good, outside_declared_support=~good))
    return NeuralForceTransitionKernel(
        force=FrozenPositionOnlyForce(force, "diagnostic-biased-force"),
        target=FrozenTargetPotential(target, "diagnostic-correlated-Gaussian"),
        config=NeuralForceHMCConfig(.6, 3, (1., 1.)),
    )


def assert_transition_close(got, expected):
    tf.debugging.assert_near(got[0], expected[0], atol=2e-12, rtol=2e-12)
    for name in ("log_accept_ratio", "delta_h"):
        tf.debugging.assert_near(getattr(got[1], name), getattr(expected[1], name),
                                 atol=2e-12, rtol=2e-12)
    for name in ("is_accepted", "divergence", "force_fallback", "finite_status"):
        tf.debugging.assert_equal(getattr(got[1], name), getattr(expected[1], name))
    for key in got[1].endpoint_status:
        tf.debugging.assert_equal(got[1].endpoint_status[key], expected[1].endpoint_status[key])


@pytest.mark.parametrize("jit", [False, True])
def test_heterogeneous_charts_match_general_kernel_for_each_chain(jit):
    kernel = fixture_kernel()
    start = tf.constant([[-1., .2], [.1, -.5], [1.2, .6], [-.3, -.8]], F64)
    scales = tf.constant([[.125], [1.], [.25], [.5]], F64)
    seed = tf.constant([19, 37], tf.int32)

    @tf.function(input_signature=[tf.TensorSpec([4, 2], F64),
                                  tf.TensorSpec([4, 1], F64),
                                  tf.TensorSpec([2], tf.int32)],
                 jit_compile=jit, autograph=False)
    def mixture(x, s, rng):
        return scaled_step(kernel, x, kernel.bootstrap_results(x), s, rng)

    got = mixture(start, scales, seed)
    for i, multiplier in enumerate((.125, 1., .25, .5)):
        reference = NeuralForceTransitionKernel(
            force=kernel.force, target=kernel.target,
            config=replace(kernel.config, step_size=.6 * multiplier))
        @tf.function(input_signature=[tf.TensorSpec([4, 2], F64),
                                      tf.TensorSpec([2], tf.int32)],
                     jit_compile=jit, autograph=False)
        def direct(x, rng):
            return reference.one_step(x, reference.bootstrap_results(x), seed=rng)
        expected = direct(start, seed)
        tf.debugging.assert_near(got[0][i], expected[0][i], atol=2e-12, rtol=2e-12)
        tf.debugging.assert_near(got[1].delta_h[i], expected[1].delta_h[i],
                                 atol=2e-12, rtol=2e-12)
        for name in ("is_accepted", "divergence", "force_fallback", "finite_status"):
            tf.debugging.assert_equal(getattr(got[1], name)[i], getattr(expected[1], name)[i])
        for key in got[1].endpoint_status:
            tf.debugging.assert_equal(got[1].endpoint_status[key][i], expected[1].endpoint_status[key][i])
    assert mixture.experimental_get_tracing_count() == 1


def test_disabled_path_is_identical_and_mixture_replays():
    kernel = fixture_kernel()
    start = tf.constant([[-1., .2], [.1, -.5], [1.2, .6], [-.3, -.8]], F64)
    state = kernel.bootstrap_results(start)
    seed = tf.constant([7, 11], tf.int32)
    baseline = kernel.one_step(start, state, seed=seed)
    off = FixedStepMixtureKernel(kernel, (1.,)).one_step(start, state, seed=seed)
    for a,b in zip(tf.nest.flatten(baseline), tf.nest.flatten(off)):
        tf.debugging.assert_equal(a,b)
    wrapper = FixedStepMixtureKernel(kernel, (.125, 1.))
    first = wrapper.one_step(start, state, seed=seed)
    second = wrapper.copy().one_step(start, state, seed=seed)
    assert_transition_close(first, second)
    scales, transition_seed = draw_scales(seed, 4, (.125, 1.))
    assert_transition_close(first, scaled_step(kernel, start, state, scales, transition_seed))
    # The adaptable step is retained as the common base, not the last random scale.
    tf.debugging.assert_equal(first[1].step_size, state.step_size)


def test_scale_choices_are_per_chain_and_disjoint_from_transition_seed():
    seed = tf.constant([203, 409], tf.int32)
    scales, transition_seed = draw_scales(seed, 2048, (.125, 1.))
    repeat, seed2 = draw_scales(seed, 2048, (.125, 1.))
    tf.debugging.assert_equal(scales, repeat)
    tf.debugging.assert_equal(transition_seed, seed2)
    fraction = float(tf.reduce_mean(tf.cast(scales == 1., F64)))
    assert .45 < fraction < .55
    assert bool(tf.reduce_any(scales[1:] != scales[:-1]))
    assert not bool(tf.reduce_all(seed == transition_seed))


def test_factory_binds_scale_law_and_rejects_unsupported_host_lane():
    kernel = fixture_kernel()
    base = bind_neural_force_hmc_tuning_runner(
        force=kernel.force, target=kernel.target, target_scope="mixture-mechanics")
    assert bind_step_mixture(base, (1.,)) is base
    mixed = bind_step_mixture(base, (.125, 1.))
    other = bind_step_mixture(base, (.25, 1.))
    assert len({base.binding_hash, mixed.binding_hash, other.binding_hash}) == 3
    assert mixed.force_identity == base.force_identity
    assert mixed.endpoint_target_identity == base.endpoint_target_identity
    assert "fixed_step_mixture" not in base.source_dependency_closure
    policy = mixed.source_dependency_closure["fixed_step_mixture"]
    assert tuple(policy["multipliers"]) == (.125, 1.)
    assert policy["base_binding_hash"] == base.binding_hash
    assert not mixed.artifact_authority
    with pytest.raises(ValueError, match="VerifiedTensorFlowHMCConfig"):
        mixed.runner(None, None, None)


@pytest.mark.parametrize("values", [(), (0., 1.), (-1., 1.), (1.1,), (float("nan"),),
                                    (True,), (.5,.5), (.1,.2,.3,.4,.5)])
def test_invalid_scale_law_fails_closed(values):
    with pytest.raises(ValueError):
        checked_multipliers(values)


@pytest.mark.parametrize("step_cap", [None, 1.])
def test_public_tuner_reload_and_retained_runner_use_mixture(tmp_path, step_cap):
    """Executable native call chain, including artifact reload and retained use."""
    from types import SimpleNamespace
    import json
    from bayesfilter.inference import (
        tune_hmc_kernel, load_verified_tensorflow_hmc_result,
        BoundRetainedHMCArchiveConfig,
        build_retained_bound_hmc_archive_runner_from_tuning_result,
    )
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config
    kernel = fixture_kernel()
    scope = "Gaussian-step-mixture-call-chain-mechanics"
    base = bind_neural_force_hmc_tuning_runner(
        force=kernel.force, target=kernel.target, target_scope=scope)
    binding = bind_step_mixture(base, (.125, 1.))
    adapter = SimpleNamespace(target_scope=scope,
        adapter_signature=lambda: scope + ":" + binding.binding_hash)
    config = tuning_config(scope)
    config = replace(config, nomination=replace(config.nomination,
        mass_window_results=(16, 32), step_adaptation_results=128,
        verification_results=128, seed=(19, 144),
        max_step_size=step_cap, energy_error_max=1000. if step_cap else None), fresh_seed=(19, 145))
    result = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.),
        config=config, parameter_scales=(1., 1.), runner_binding=binding,
        output_dir=tmp_path / "tuning")
    assert result.handoff_eligible
    if step_cap is not None:
        assert float(result.nomination.step_size)<=step_cap
        saved=json.loads(Path(result.nomination.artifact_manifest_path).read_text())
        assert saved["candidate_summary_fields"][-2:]==["max_abs_energy_error","energy_error_breach_count"]
    loaded = load_verified_tensorflow_hmc_result(result.artifact_manifest_path,
        adapter=adapter, runner_binding=binding)
    with pytest.raises(ValueError, match="binding mismatch"):
        load_verified_tensorflow_hmc_result(result.artifact_manifest_path,
            adapter=adapter, runner_binding=base)
    runner = build_retained_bound_hmc_archive_runner_from_tuning_result(
        tuning_result=loaded, runner_binding=binding)
    retained = runner.run(BoundRetainedHMCArchiveConfig(num_results=32,
        seed=(19, 146), output_dir=tmp_path / "retained",
        budget_provenance="CPU-only Gaussian mechanics, no scientific claim"))
    assert retained["health_passed"]
    manifest = json.loads(open(retained["archive_manifest_path"]).read())
    assert manifest["binding_hash"] == binding.binding_hash
