"""Compiled numerical mechanics; tiny synthetic fixtures, no CCMA/HMC claims."""
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import _DenseAutoregressiveIAFComponent
from bayesfilter.inference.neutra_scale_aware_training import ScaleAwareUpdatePolicy
from neutra_scale_aware_fixtures import build


def assert_state_close(actual, expected, tolerance=1e-10):
    for left, right in zip((*actual.variables, *actual.optimizer.variables),
                           (*expected.variables, *expected.optimizer.variables), strict=True):
        tf.debugging.assert_near(tf.cast(left, tf.float64), tf.cast(right, tf.float64),
                                 atol=tolerance, rtol=tolerance)
    assert actual.accepted_updates == expected.accepted_updates
    assert actual.active_stages == expected.active_stages


def test_compiled_transactions_match_reference_and_reuse_graphs():
    reference, _ = build(kind='banana', jit=False, stream=200)
    compiled, _ = build(kind='banana', jit=True, stream=200)
    compiled.prepare_training_stream(4)
    replay_identity = id(compiled._kernels._frozen)
    for _ in range(4):
        old = reference.train_step()
        new = compiled.train_step()
        assert old['accepted'] and new['accepted'], (old, new)
        assert old['fraction'] == new['fraction']
        assert new['loss'] == pytest.approx(old['loss'], abs=1e-10)
        assert_state_close(compiled, reference)
        assert id(compiled._kernels._frozen) == replay_identity
    kernels = compiled._kernels
    for function in (kernels.map, kernels.gradient, kernels.aggregate, kernels.snapshot,
                     kernels.proposals[0], kernels.damp, kernels.inverse, kernels.frozen_map, kernels.replay):
        assert function.experimental_get_tracing_count() == 1
        for concrete in function._list_all_concrete_functions():
            graph = concrete.graph.as_graph_def()
            nodes = list(graph.node)+[n for f in graph.library.function for n in f.node_def]
            assert not any(n.op in ('PyFunc', 'EagerPyFunc', 'PyFuncStateless') for n in nodes)
            assert concrete.function_def.attr['_XlaMustCompile'].b


def test_compiled_rejection_restores_all_optimizer_slots_and_variables():
    model, target = build(jit=True, stream=201)
    model.prepare_training_stream(1)
    model.preflight()
    before = model.state_payload()
    # The gradient succeeds; target validation of the proposed map fails.
    target.fail_at_call = target.calls+2
    record = model.train_step()
    after = model.state_payload()
    assert not record['accepted']
    for key in ('variables', 'optimizer_variables', 'active_stages'):
        assert after[key] == before[key]


def test_all_three_compiled_adam_stages_and_independent_releases():
    reference, _ = build(jit=False, stream=202)
    compiled, _ = build(jit=True, stream=202)
    compiled.prepare_training_stream(201)
    for step in range(1, 202):
        old = reference.train_step()
        new = compiled.train_step()
        assert old['accepted'] and new['accepted'], (step, old, new)
        assert old['fraction'] == new['fraction']
        if step in (100, 200):
            assert reference.release_next_stage()['passed']
            assert compiled.release_next_stage()['passed']
            assert_state_close(compiled, reference, tolerance=1e-8)
    assert compiled.active_stages == 3
    assert_state_close(compiled, reference, tolerance=1e-8)
    assert all(f.experimental_get_tracing_count() == 1 for f in compiled._kernels.proposals)


def test_inverse_graph_uses_bounded_tensor_loop_at_dimension_142():
    counts = []
    for dimension in (2, 142):
        component = _DenseAutoregressiveIAFComponent(dim=dimension, hidden_layers=(4, 4),
            activation='tanh', s_max=3., scale_transform='identity',
            weights=(tf.zeros([dimension, 4], tf.float64), tf.zeros([4, 4], tf.float64),
                     tf.zeros([4, 2*dimension], tf.float64)),
            biases=(tf.zeros([4], tf.float64), tf.zeros([4], tf.float64),
                    tf.zeros([2*dimension], tf.float64)))
        inverse = tf.function(component.inverse, autograph=False, jit_compile=True,
            input_signature=[tf.TensorSpec([8, dimension], tf.float64)])
        rows = tf.reshape(tf.cast(tf.range(8*dimension), tf.float64), [8, dimension])/1000
        tf.debugging.assert_equal(inverse(rows), rows)
        graph = inverse.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node)+[n for f in graph.library.function for n in f.node_def]
        assert sum(n.op in ('While', 'StatelessWhile') for n in nodes) == 1
        counts.append(len(nodes))
    assert abs(counts[1]-counts[0]) < 20


def test_compiled_width_exhaustion_preserves_state():
    policy = replace(ScaleAwareUpdatePolicy(), zero_limit=1e-20, rms_limit=1e-20,
                     quantile_limit=1e-20, maximum_limit=1e-20, maximum_halvings=0)
    model, _ = build(jit=True, policy=policy, stream=203)
    model.prepare_training_stream(1)
    model.preflight()
    before = model.state_payload()
    result = model.train_step()
    assert not result['accepted'] and result['reason'] == 'physical_width_budget_exhausted'
    after = model.state_payload()
    assert after['variables'] == before['variables']
    assert after['optimizer_variables'] == before['optimizer_variables']


def test_prepared_stream_preserves_original_draws_and_does_not_mutate_state():
    model, _ = build(jit=True, stream=205)
    before = model.state_payload()
    model.prepare_training_stream(4)
    assert model.state_payload() == before
    for counter, actual in enumerate(model._prepared_training):
        seed = tf.random.experimental.stateless_fold_in(tf.constant(model.training_seed, tf.int32), counter)
        expected = tf.random.stateless_normal([model.policy.training_batch_size, model.dimension], seed, dtype=tf.float64)
        tf.debugging.assert_equal(actual, expected)
