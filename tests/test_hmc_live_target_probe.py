"""Live Model B probes reuse executable graphs, never target answers.

CPU reference checks of graph/eager equality and mutation detection. Actual
GPU HMC stream parity is recorded separately by the scoped release campaign.
"""
import pytest
import tensorflow as tf

from bayesfilter.testing import simple_nonlinear_generic_target_adapter_tf as model
from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload


def encoded(result):
    return tuple(_tensor_payload(value) for value in result)


@pytest.mark.parametrize('backend', ['tf_svd_ukf', 'tf_svd_cubature'])
def test_graph_reads_current_theta_observations_and_model_constants(backend, monkeypatch):
    theta = tf.constant([[.6, .4, .7]], tf.float64)
    data = tf.constant([[.1], [.2], [-.1], [.3]], tf.float64)
    graph = model._eager_likelihood_graph(backend, 1, (4, 1))
    original = None
    for change in ('none', 'theta', 'data', 'alpha', 'observation_sigma'):
        q = theta + tf.constant([[.01, .02, .03]], tf.float64) if change == 'theta' else theta
        y = data + tf.constant(.1, tf.float64) if change == 'data' else data
        with monkeypatch.context() as patch:
            if change == 'alpha':
                patch.setattr(model, 'MODEL_B_ALPHA', tf.constant(.6, tf.float64))
            if change == 'observation_sigma':
                patch.setattr(model, 'MODEL_B_OBSERVATION_SIGMA', tf.constant(.4, tf.float64))
            actual = encoded(model.simple_nonlinear_sigma_point_log_likelihood_and_grad(
                q, backend=backend, observations=y))
            expected = encoded(model._simple_nonlinear_likelihood(
                q, y, model.MODEL_B_ALPHA, model.MODEL_B_OBSERVATION_SIGMA,
                backend=backend, jit_compile=True))
            assert actual == expected
        if change == 'none':
            original = actual
        else:
            assert actual != original, change
    assert graph.experimental_get_tracing_count() == 1
    assert model._eager_likelihood_graph.cache_info().maxsize == 16


def test_native_trace_bypasses_eager_inspection_graph(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('native HMC must preserve its original graph body')
    monkeypatch.setattr(model, '_eager_likelihood_graph', forbidden)
    @tf.function(input_signature=[tf.TensorSpec([1, 3], tf.float64)], jit_compile=True)
    def native(q):
        return model.simple_nonlinear_svd_ukf_log_likelihood_and_grad(
            q, observations=tf.constant([[.1], [.2]], tf.float64))
    values, scores = native(tf.constant([[.6, .4, .7]], tf.float64))
    assert bool(tf.reduce_all(tf.math.is_finite(values)))
    assert bool(tf.reduce_all(tf.math.is_finite(scores)))


@pytest.mark.parametrize('damage', ['prior', 'likelihood', 'alpha', 'observation_sigma',
                                  'start', 'geometry', 'source'])
def test_real_binding_rejects_live_mutations_after_graph_warmup(damage, tmp_path, monkeypatch):
    from tests.test_hmc_acceptance_ssm_recovery import setup
    target, binding, _ = setup('nonlinear')
    binding.validate()  # Reuse the warmed inspection graph before mutation.
    adapter = target._ssm.adapter
    if damage in ('prior', 'likelihood'):
        name = 'prior_log_prob_and_grad' if damage == 'prior' else 'filter_log_likelihood_and_grad'
        original = getattr(adapter, name)
        def changed(q):
            value, score = original(q)
            return value - tf.constant(1., tf.float64), score
        monkeypatch.setattr(adapter, name, changed)
    elif damage == 'alpha':
        monkeypatch.setattr(model, 'MODEL_B_ALPHA', tf.constant(.6, tf.float64))
    elif damage == 'observation_sigma':
        monkeypatch.setattr(model, 'MODEL_B_OBSERVATION_SIGMA', tf.constant(.4, tf.float64))
    elif damage == 'start':
        monkeypatch.setattr(binding, 'initial_active_state', binding.initial_active_state + .1)
    elif damage == 'geometry':
        transform = binding._active_adapter.transform
        object.__setattr__(transform, 'factor', -tf.convert_to_tensor(transform.factor))
    else:
        # Real source-byte mutation while keeping the recorded source digest.
        import hashlib
        source = tmp_path/'dependency.py'
        source.write_text('VERSION = 1\n')
        from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
        binding._spec['source_closure'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        binding.binding_hash = _sha256(binding._spec)
        source.write_text('VERSION = 2\n')
    with pytest.raises(ValueError, match='changed'):
        binding.validate()
