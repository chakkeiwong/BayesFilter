"""Guard invariants and independent restart/chunk/stage regression tests."""

import copy
from dataclasses import replace

import pytest
import tensorflow as tf
from bayesfilter.inference.neutra_scale_aware_training import (
    ScaleAwareReleaseRequired,
    ScaleAwareUpdatePolicy,
    width_displacement_metrics,
)

from tests.neutra_scale_aware_fixtures import build, malformed_batch, take_steps


def effectful(state):
    return {k: state[k] for k in ('variables', 'optimizer_variables', 'accepted_updates', 'active_stages')}


def test_correlated_metric_orientation_units_and_quantile():
    lower = tf.constant([[2., 0.], [.9, .0001]], tf.float64)
    w = tf.constant([[.02, 0.]]+[[.03, .04]]*60+[[.06, .08]]+[[.09, .12]]*3, tf.float64)
    delta = tf.matmul(w, lower, transpose_b=True)
    scales = tf.zeros([65, 6], tf.float64)
    result = width_displacement_metrics(tf.zeros_like(delta), delta, lower, scales, scales)
    assert result['zero'] == pytest.approx(.02)
    assert result['p95'] == pytest.approx(.10)
    assert result['maximum'] == pytest.approx(.15)
    assert result['rms'] == pytest.approx(((60*.05**2+.10**2+3*.15**2)/64)**.5)
    units = tf.constant([1e-4, 1e4], tf.float64)
    other = width_displacement_metrics(tf.zeros_like(delta), delta*units, lower*units[:, None], scales, scales)
    assert other == pytest.approx(result, abs=1e-10)


@pytest.mark.parametrize('field,value', [('release_boundaries', (99, 200)), ('release_boundaries', (100, 199)),
                                      ('maximum_halvings', 31), ('training_batch_size', 1)])
def test_policy_cannot_weaken_required_staging_or_unbound_work(field, value):
    with pytest.raises(ValueError):
        replace(ScaleAwareUpdatePolicy(), **{field: value})


def test_inactive_stages_include_hidden_tensors_and_nonzero_momentum():
    model, _target = build()
    model.preflight()
    for slots in (model.optimizer._momentums, model.optimizer._velocities):
        for slot, stage in zip(slots, model.variable_stages, strict=True):
            if stage > 1:
                slot.assign(tf.ones_like(slot)*.2)
    before = model.state_payload()
    record = model.train_step()
    assert record['accepted']
    after = model.state_payload()
    for i, stage in enumerate(model.variable_stages):
        if stage > 1:
            assert before['variables'][i] == after['variables'][i]
            for slots in (model.optimizer._momentums, model.optimizer._velocities):
                slot_index = next(j for j, v in enumerate(model.optimizer.variables) if v is slots[i])
                assert before['optimizer_variables'][slot_index] == after['optimizer_variables'][slot_index]


@pytest.mark.parametrize('defect', ['signature', 'theta', 'eligibility', 'nan_score'])
def test_external_target_binding_failure_stops_and_rolls_back(defect):
    model, target = build()
    model.preflight()
    z = tf.constant([[.2, -.4], [-.7, .9]], tf.float64)
    theta, _ld = model.forward_and_logdet(z)
    batch = malformed_batch(target.evaluate(theta), defect)
    before = model.state_payload()
    calls = target.calls
    record = model.train_step_with_external_value_score(z, batch)
    assert not record['accepted']
    assert effectful(model.state_payload()) == effectful(before)
    assert target.calls == calls and model.external_target_rows == 2
    with pytest.raises(ValueError, match='stopped'):
        model.train_step()


def test_candidate_exception_restores_and_remains_charged():
    model, target = build()
    model.preflight()
    before = model.state_payload()
    target.fail_at_call = target.calls+2  # Current gradient succeeds; candidate fails.
    record = model.train_step()
    assert not record['accepted'] and 'injected target' in record['reason']
    state = model.state_payload()
    assert effectful(state) == effectful(before)
    assert state['target_rows'] == before['target_rows']+32+97
    assert state['attempted_updates'] == 1 and state['draw_counter'] == 1
    fresh, _ = build()
    fresh.restore_state(state)
    assert fresh.state_payload() == state
    with pytest.raises(ValueError, match='stopped'):
        fresh.train_step()


def test_durable_commit_event_failure_restores_every_effectful_value():
    def sink(event):
        if event['kind'] == 'update_accepted':
            raise OSError('journal unavailable')

    model, _target = build(event_sink=sink)
    model.preflight()
    before = model.state_payload()
    result = model.train_step()
    assert not result['accepted'] and 'journal unavailable' in result['reason']
    assert effectful(before) == effectful(model.state_payload())


def test_budget_exhaustion_is_a_typed_rollback_without_candidate_target():
    policy = replace(ScaleAwareUpdatePolicy(), zero_limit=1e-30, rms_limit=1e-30,
                     quantile_limit=1e-30, maximum_limit=1e-30, maximum_halvings=0)
    model, target = build(policy=policy)
    model.preflight()
    before = model.state_payload()
    calls = target.calls
    result = model.train_step()
    assert result['reason'] == 'physical_width_budget_exhausted' and not result['accepted']
    assert target.calls == calls+1 and effectful(before) == effectful(model.state_payload())


def test_interrupted_restore_reproduces_exact_next_owned_draw_and_update():
    model, _target = build()
    take_steps(model, 2)
    snapshot = model.state_payload()
    fresh, _target2 = build()
    fresh.restore_state(snapshot)
    assert fresh.state_payload() == snapshot
    assert model.train_step() == fresh.train_step()
    assert model.state_payload() == fresh.state_payload()


@pytest.mark.parametrize('defect', ['hash', 'variable_order', 'nonfinite', 'target', 'seed', 'negative_slot'])
def test_invalid_checkpoint_never_partially_restores(defect):
    model, _target = build()
    take_steps(model, 1)
    snapshot = model.state_payload()
    bad = copy.deepcopy(snapshot)
    if defect == 'hash':
        bad['accepted_updates'] += 1
    elif defect == 'variable_order':
        bad['variable_keys'].reverse()
    elif defect == 'nonfinite':
        bad['variables'][1][0] = float('nan')
    elif defect == 'target':
        bad['identity']['initializer']['target_signature'] = '9'*64
    elif defect == 'seed':
        bad['identity']['training_seed'][1] += 1
    else:
        slot = model.optimizer._velocities[0]
        index = next(i for i, var in enumerate(model.optimizer.variables) if var is slot)
        bad['optimizer_variables'][index][0][0] = -1.
    if defect not in ('hash', 'nonfinite'):
        from bayesfilter.inference.neutra_scale_aware_training import _digest
        bad['state_hash'] = _digest({k: v for k, v in bad.items() if k != 'state_hash'})
    with pytest.raises((ValueError, TypeError)):
        model.restore_state(bad)
    assert model.state_payload() == snapshot


def test_fixed_chunk_padding_matches_full_batch_gradient_and_single_adam():
    full, target = build()
    chunked, other = build()
    full.preflight()
    chunked.preflight()
    z = tf.random.stateless_normal([7, 2], [20260915, 8401], dtype=tf.float64)
    theta, _ld = full.forward_and_logdet(z)
    a = full.train_step_with_external_value_score(z, target.evaluate(theta))
    chunks = (z[:4], tf.concat((z[4:], tf.zeros([1, 2], tf.float64)), 0))
    batches = [other.evaluate(chunked.forward_and_logdet(c)[0]) for c in chunks]
    b = chunked.train_step_with_external_value_score_chunks(chunks, batches, (4, 3))
    assert a['accepted'] and b['accepted'] and a['fraction'] == b['fraction']
    assert a['loss'] == pytest.approx(b['loss'], abs=1e-12)
    for x, y in zip(full.variables, chunked.variables, strict=True):
        tf.debugging.assert_near(x, y, atol=1e-12, rtol=1e-12)
    for x, y in zip(full.optimizer.variables, chunked.optimizer.variables, strict=True):
        tf.debugging.assert_near(tf.cast(x, tf.float64), tf.cast(y, tf.float64), atol=1e-12, rtol=1e-12)


def test_healthy_and_damped_adam_moments_match_raw_canonical_gradient():
    for lower in (tf.eye(2, dtype=tf.float64), tf.linalg.diag(tf.constant([1e-6, 1e-7], tf.float64))):
        model, target = build(lower=lower)
        model.preflight()
        z = tf.random.stateless_normal([32, 2], [20260915, 8402], dtype=tf.float64)
        theta, _ld = model.forward_and_logdet(z)
        batch = target.evaluate(theta)
        raw = model._gradient_graph(z, batch.value, batch.score, tf.ones([32], tf.float64))
        clones = [tf.Variable(v.numpy()) for v in model.variables]
        raw_adam = tf.keras.optimizers.Adam(learning_rate=.001, beta_1=.9, beta_2=.999, epsilon=1e-8)
        raw_adam.build(clones)
        raw_adam.apply_gradients([(tf.clip_by_norm(g/32, 10.), v)
                                 for g, v, s in zip(raw[4:], clones, model.variable_stages, strict=True) if s == 1])
        before = [tf.identity(v) for v in model.variables]
        result = model.train_step_with_external_value_score(z, batch)
        assert result['accepted']
        for actual, raw_slot in zip(model.optimizer.variables, raw_adam.variables, strict=True):
            tf.debugging.assert_equal(actual, raw_slot)
        for actual, proposed, old in zip(model.variables, clones, before, strict=True):
            if result['fraction'] == 1:
                tf.debugging.assert_equal(actual, proposed)
            else:
                tf.debugging.assert_equal(actual, old+result['fraction']*(proposed-old))


@pytest.fixture(scope='module')
def boundary_state():
    model, _target = build(stream=50)
    take_steps(model, 99)
    with pytest.raises(ValueError, match='boundary'):
        model.release_next_stage()
    model.train_step()
    return model.state_payload()


def test_release_boundary_preserves_map_and_slots_and_advances_only_one_stage(boundary_state):
    model, _target = build(stream=50)
    model.restore_state(boundary_state)
    with pytest.raises(ScaleAwareReleaseRequired):
        model.train_step()
    before = model.state_payload()
    assert model.release_next_stage()['passed']
    after = model.state_payload()
    assert before['variables'] == after['variables']
    assert before['optimizer_variables'] == after['optimizer_variables']
    assert model.active_stages == 2 and model.accepted_updates == 100
    assert model.train_step()['accepted']


def test_failed_independent_release_stops_without_mutating_map_or_slots(boundary_state):
    model, target = build(stream=50)
    model.restore_state(boundary_state)
    before = model.state_payload()
    target.invalid = True
    result = model.release_next_stage()
    assert not result['passed'] and target.calls == 1
    assert effectful(before) == effectful(model.state_payload())
    with pytest.raises(ValueError, match='stopped'):
        model.train_step()


def test_fixed_shape_cpu_xla_map_and_raw_gradient_match_nonxla():
    ordinary, target = build()
    compiled, _ = build(jit=True)
    z = tf.constant([[.2, -.4], [-.6, .8]], tf.float64)
    theta, _ld = ordinary.forward_and_logdet(z)
    batch = target.evaluate(theta)
    for a, b in zip(ordinary._map_graph(z), compiled._map_graph(z), strict=True):
        if a.dtype == tf.bool:
            tf.debugging.assert_equal(a, b)
        else:
            tf.debugging.assert_near(a, b, atol=1e-10, rtol=1e-10)
    args = (z, batch.value, batch.score, tf.ones([2], tf.float64))
    for a, b in zip(ordinary._gradient_graph(*args), compiled._gradient_graph(*args), strict=True):
        tf.debugging.assert_near(a, b, atol=1e-10, rtol=1e-10)
