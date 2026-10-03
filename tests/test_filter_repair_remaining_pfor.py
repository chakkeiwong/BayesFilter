"""Fresh independent diagnostic fixtures for non-pfor execution repairs.

No historical LEDH result or frozen pre-invalidation fixture is reused. NumPy
and finite differences are diagnostic only; no pfor oracle is executed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_streaming_tf as streaming
from bayesfilter.highdim import sir_latent_preclip_reference_tf as scout
from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from tests.test_batched_value_score import BatchedQuadraticAdapter

DTYPE = tf.float64


def _save(request, name, report):
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n')
    return directory


def _graph(owner, args, directory, name):
    assert owner.experimental_get_tracing_count() == 1
    graph = owner.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
    assert not {n.op for n in nodes} & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless', 'XlaHostCompute'}
    assert not any('/pfor/' in n.name for n in nodes)
    hlo = owner.experimental_get_compiler_ir(*args)(stage='hlo')
    (directory / f'{name}.hlo.txt').write_text(hlo)
    return {'trace_count': 1, 'hlo_sha256': hashlib.sha256(hlo.encode()).hexdigest(),
            'no_pfor_or_host_callback': True, 'jit_compile': True}


def _contract_inputs():
    rng = np.random.default_rng(81101)
    particles = tf.constant(rng.normal(size=(2, 8, 2)), DTYPE)
    geometry = .2 * particles
    weights = tf.nn.softmax(tf.constant(rng.normal(size=(2, 8)) * .03, DTYPE), axis=1)
    design = tf.broadcast_to(tf.tile(tf.concat([tf.eye(2, dtype=DTYPE), -tf.eye(2, dtype=DTYPE)], 0), [2, 1]) * tf.sqrt(tf.constant(2., DTYPE)), [2, 8, 2])
    base = (geometry, particles, tf.math.log(weights), weights, design,
            tf.constant([.001, .001], DTYPE), tf.constant(1., DTYPE),
            tf.constant([2., 2.], DTYPE), tf.constant(.8, DTYPE))
    d = [np.zeros((*v.shape, 3)) for v in (*base[:6], base[7])]
    d[1][..., 0] = rng.normal(size=(2, 8, 2)) * .03
    for index in (1, 2):
        logits_direction = rng.normal(size=(2, 8)) * .03
        centered = logits_direction - np.sum(weights.numpy() * logits_direction, axis=1, keepdims=True)
        d[2][..., index] = centered
        d[3][..., index] = weights.numpy() * centered
    for index in (0, 1, 4):
        d[index][..., 2] = rng.normal(size=d[index].shape[:-1]) * .03
    d[4][..., 2] -= np.mean(d[4][..., 2], axis=1, keepdims=True)
    d[5][..., 2] = .0001
    d[6][..., 2] = .01
    upstream = tf.constant(rng.normal(size=(2, 8, 2)), DTYPE)
    return base, tuple(tf.constant(v, DTYPE) for v in d), upstream


def test_streaming_total_analytical_direction_loop(request):
    base, directions, upstream = _contract_inputs()
    controls = {'steps': 3, 'balance_steps': 2, 'row_chunk_size': 8, 'col_chunk_size': 8}
    signature = [tf.TensorSpec(v.shape, v.dtype) for v in (*base, *directions, upstream)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def owner(*values):
        inputs, tangent, cotangent = values[:9], values[9:16], values[16]
        fused = streaming._contract_e_streaming_forward_jvp_core(*inputs[:6], *tangent, *inputs[6:], **controls)
        separate = streaming._contract_e_streaming_jvp_core(*inputs[:6], *tangent, *inputs[6:], **controls)
        vjp = streaming._contract_e_streaming_vjp_core(*inputs, cotangent, **controls)
        # Probability weights and log-weights are independent operands here:
        # use transport-only log adjoint plus direct probability-weight adjoint.
        adjoints = (vjp['scaled_geometry'], vjp['source_particles'],
            vjp['normalized_log_weights_transport'], vjp['normalized_weights_probability'],
            vjp['residual_design'], vjp['ridge'], vjp['epsilon0'])
        paired = tf.add_n([tf.reduce_sum(bar[..., None] * delta, axis=tf.range(tf.rank(bar)))
                          for bar, delta in zip(adjoints, tangent, strict=True)])
        primal_pair = tf.reduce_sum(fused['particles_tangent'] * cotangent[..., None], axis=[0, 1, 2])
        return {'value': fused['particles'], 'tangent': fused['particles_tangent'],
            'separate': separate['particles'], 'paired': paired, 'primal_pair': primal_pair,
            'source_direct_pair': tf.reduce_sum(vjp['source_particles_direct'][..., None] * tangent[1], axis=[0, 1, 2]),
            'weight_direct_pair': tf.reduce_sum(vjp['normalized_weights_probability'][..., None] * tangent[3], axis=[0, 1]),
            'finite': fused['reset']['finite'], 'factor_positive': fused['reset']['factor_diagonal_positive'],
            'gap_condition': fused['reset']['gap_condition_proxy'],
            'target_condition': fused['reset']['target_condition_proxy'],
            'injected_condition': fused['reset']['injected_condition_proxy']}

    @tf.function(input_signature=signature[:9], jit_compile=True, autograph=False)
    def forward(*inputs):
        return streaming._contract_e_streaming_forward_core(*inputs, **controls)['particles']

    records = []
    for shift in (0., .01):
        inputs = (base[0], base[1] + shift, *base[2:])
        args = (*inputs, *directions, upstream)
        actual = owner(*args)
        record = {'shift': shift, 'actual': {k: v.numpy().tolist() for k, v in actual.items()}}
        records.append(record)
        directory = _save(request, 'remaining-pfor-contract-e', {'seed': 81101, 'records': records})
        assert bool(tf.reduce_all(actual['finite'] & actual['factor_positive']))
        np.testing.assert_allclose(actual['value'], forward(*inputs), atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(actual['tangent'], actual['separate'], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(actual['paired'], actual['primal_pair'], atol=1e-9, rtol=1e-9)
        assert abs(float(actual['source_direct_pair'][0])) > 1e-8
        assert abs(float(actual['weight_direct_pair'][1])) > 1e-8
        differences = []
        for direction in range(3):
            for step in (1e-3, 5e-4):
                results = []
                for factor in (-2, -1, 1, 2):
                    delta = factor * step
                    changed = [value + delta * tangent[..., direction] for value, tangent in zip(inputs[:6], directions[:6], strict=True)]
                    # Keep the probability/log-weight input identity at finite offsets.
                    changed[2] = tf.math.log(changed[3])
                    changed.extend((inputs[6], inputs[7] + delta * directions[6][..., direction], inputs[8]))
                    results.append(forward(*changed))
                derivative = (results[0] - 8 * results[1] + 8 * results[2] - results[3]) / (12 * step)
                np.testing.assert_allclose(actual['tangent'][..., direction], derivative, atol=2e-6, rtol=2e-6)
                differences.append({'direction': direction, 'step': step,
                    'max_absolute_error': float(tf.reduce_max(tf.abs(actual['tangent'][..., direction] - derivative)))})
        record['finite_differences'] = differences
        np.testing.assert_array_equal(actual['tangent'], owner(*args)['tangent'])
    graph = _graph(owner, args, directory, 'remaining-pfor-contract-e')
    _save(request, 'remaining-pfor-contract-e', {'seed': 81101, 'records': records,
        'graph': graph, 'device': actual['value'].device,
        'shape': {'B': 2, 'N': 8, 'd': 2, 'P': 3}, 'chunk_policy': 'K=N=8',
        'nonclaims': ['Primitive total derivative check, no canonical LEDH, tuning, HMC or training admission.']})


@pytest.mark.parametrize('shape', [(4, 3), (2, 4, 3)])
def test_scalar_transport_fallback_preserves_default_gate(shape, request):
    class ScalarAffine:
        """Fresh scalar-only fixture; exact fixed determinant, no framework slogdet."""
        parameter_dim = 3
        shift = tf.constant([.25, -.5, .75], DTYPE)
        factor = tf.constant([[2., .1, 0.], [0., 1.5, -.2], [.3, 0., 1.25]], DTYPE)

        def manifest_payload(self):
            return {'schema': 'nonpfor_scalar_affine_fixture.v1', 'parameter_dim': 3}

        def forward(self, values):
            return self.shift + tf.linalg.matvec(self.factor, values)

        def log_abs_det_jacobian(self, values):
            return tf.math.log(tf.constant(3.744, DTYPE))

        def pullback_score(self, values, score):
            return tf.linalg.matvec(self.factor, score, transpose_a=True)

        def log_abs_det_jacobian_score(self, values):
            return tf.zeros_like(values)

    transport = ScalarAffine()
    assert not callable(getattr(transport, 'forward_batch', None))
    assert not callable(getattr(transport, 'log_abs_det_jacobian_batch', None))
    with pytest.raises(TypeError, match='forward_batch'):
        FixedTransportValueScoreAdapter(base_adapter=BatchedQuadraticAdapter(), transport=transport,
                                         target_scope='nonpfor_scalar_reference')
    adapter = FixedTransportValueScoreAdapter(base_adapter=BatchedQuadraticAdapter(), transport=transport,
        target_scope='nonpfor_scalar_reference', require_batch_native=False)
    assert adapter.value_score_capability().xla_hmc_ready is False
    @tf.function(input_signature=[tf.TensorSpec(shape, DTYPE)], jit_compile=True, autograph=False)
    def owner(values):
        forward = adapter.latent_to_position(values)
        logdet = adapter.log_abs_det_jacobian(tf.reshape(values, [-1, 3]))
        return forward, logdet
    z = tf.reshape(tf.range(np.prod(shape), dtype=DTYPE), shape) / 20
    records = []
    for shift in (0., .1):
        output, logdet = owner(z + shift)
        expected = transport.shift + tf.linalg.matvec(transport.factor, z + shift)
        np.testing.assert_allclose(output, expected, atol=1e-12, rtol=1e-12)
        # Independent closed-form determinant of the fixed 3x3 fixture.
        determinant = 2 * (1.5 * 1.25) + .1 * (-.2 * .3)
        np.testing.assert_allclose(logdet, np.log(abs(determinant)), atol=1e-12, rtol=1e-12)
        records.append({'shift': shift, 'forward': output.numpy().tolist(), 'logdet': logdet.numpy().tolist()})
    with pytest.raises(ValueError, match='batch-native transport'):
        adapter.log_prob_and_grad_batch(tf.reshape(z, [-1, 3]))
    directory = _save(request, f'remaining-pfor-transport-{len(shape)}', {'records': records})
    graph = _graph(owner, (z,), directory, f'remaining-pfor-transport-{len(shape)}')
    _save(request, f'remaining-pfor-transport-{len(shape)}', {'records': records, 'graph': graph,
        'device': output.device, 'default_batch_native_gate_preserved': True,
        'role': 'explicit scalar-reference fallback, not training or HMC admission'})


def test_reference_scout_uses_nonpfor_jacobian(request, monkeypatch):
    model = scout.reduced_latent_preclip_sir_model()
    theta = tf.constant([.03, -.02, .04], DTYPE)
    original = tf.GradientTape.jacobian
    calls = []
    def observed(tape, target, source, **kwargs):
        assert kwargs.get('experimental_use_pfor') is False
        result = original(tape, target, source, **kwargs)
        columns = []
        for column in range(2):
            direction = tf.one_hot(column, 2, dtype=DTYPE)
            step = 1e-5
            plus = model.physical_model.transition_mean(theta, (source + step * direction)[None])[0]
            minus = model.physical_model.transition_mean(theta, (source - step * direction)[None])[0]
            columns.append((plus - minus) / (2 * step))
        np.testing.assert_allclose(result, tf.stack(columns, axis=1), atol=2e-8, rtol=2e-8)
        calls.append(result.numpy().tolist())
        return result
    monkeypatch.setattr(tf.GradientTape, 'jacobian', observed)
    grids = scout.prepare_reduced_dense_grids(model, theta, time_steps=2, order=3, radius=3.)
    assert len(calls) == 2 and len(grids) == 3
    assert all(bool(tf.reduce_all(tf.math.is_finite(grid.points))) for grid in grids)
    _save(request, 'remaining-pfor-scout', {'jacobians': calls, 'grid_count': len(grids),
        'role': 'independent CPU reference scout only; no runtime admission'})
