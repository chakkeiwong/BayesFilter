"""Endpoint coincidences must not masquerade as immobility in the release profile.

CPU diagnostic references only. The synthetic paths test health semantics, not
MCMC validity. A saved native trial keeps the original failed stream reviewable.
"""
import base64
from dataclasses import replace
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy, evaluate_hmc_trial_health
from bayesfilter.testing.acceptance_release_validation import full_search_configuration


def release_health_policy():
    config = full_search_configuration('k0', seed=(20261002, 2202), wall_seconds=240)
    p = HMCReplicatedAcceptancePolicy.from_payload(config['policy'])
    return HMCAcceptancePolicy(**{name: getattr(p, name) for name in (
        'min_movement_rate', 'max_repeated_state_fraction',
        'min_normalized_return_displacement', 'max_abs_log_accept_energy_proxy')})


def health(samples, *, policy=None, **overrides):
    shape = samples.shape[:2]
    kwargs = dict(samples=samples, log_accept_ratio=tf.fill(shape, tf.constant(-.35, tf.float64)),
                  is_accepted=tf.ones(shape, tf.bool), target_log_prob=tf.zeros(shape, tf.float64),
                  native_divergence_status='available', native_divergence_count=0)
    kwargs.update(overrides)
    return evaluate_hmc_trial_health(**kwargs, policy=policy or release_health_policy())


def moving_return():
    # A nonperiodic deterministic path, deliberately ending exactly where it
    # started. Distances are arbitrary fixture units, not tuning thresholds.
    row = [((i * i * 17 + i * 11) % 101) / 10. for i in range(65)]
    row[-1] = row[0]
    return tf.tile(tf.constant(row, tf.float64)[:, None, None], [1, 4, 1])


def test_moving_path_can_return_to_its_start_without_being_immobile():
    samples = moving_return()
    legacy = health(samples, policy=HMCAcceptancePolicy())
    current = health(samples)
    assert legacy.candidate_promotion_vetoes == ('movement_gate_failed',)
    assert not current.candidate_promotion_vetoes
    assert all(x > .9 for x in current.movement_rate_by_chain)
    assert current.normalized_return_displacement_by_chain == (0.,) * 4
    assert current.acceptance_decision == 'inconclusive_evidence'


def test_saved_original_failed_native_trial_has_a_chance_endpoint_return():
    fixture = json.loads((Path(__file__).parent / 'data/hmc_endpoint_return_trial.json').read_text())
    def decode(payload):
        return tf.io.parse_tensor(base64.b64decode(payload['tensor']),
                                  out_type=tf.dtypes.as_dtype(payload['dtype']))
    prefix = fixture['source_configuration']['policy']['discarded_prefix']
    samples = decode(fixture['samples'])[prefix:]
    trace = {k: decode(v)[prefix:] for k, v in fixture['trace'].items()}
    old = health(samples, policy=HMCAcceptancePolicy(), **trace)
    new = health(samples, **trace)
    assert 'movement_gate_failed' in old.candidate_promotion_vetoes
    assert not new.candidate_promotion_vetoes
    assert new.movement_rate_by_chain == old.movement_rate_by_chain
    assert new.path_return_fraction_by_chain == old.path_return_fraction_by_chain
    assert min(new.movement_rate_by_chain) == pytest.approx(.640625)
    assert new.normalized_return_displacement_by_chain[0] == pytest.approx(5.787317960593729e-5)


@pytest.mark.parametrize('failure', ['stuck', 'frozen_coordinate', 'cycle', 'divergence',
                                    'nonfinite_state', 'nonfinite_target'])
def test_real_health_vetoes_survive_endpoint_reporting_only(failure):
    samples = moving_return()
    overrides = {}
    if failure == 'stuck':
        samples = tf.zeros_like(samples)
    elif failure == 'frozen_coordinate':
        samples = tf.concat([samples, tf.zeros_like(samples)], axis=-1)
    elif failure == 'cycle':
        # Four-step cycle whose final endpoint is different from its first.
        samples = tf.tile(tf.constant([i % 4 for i in range(66)], tf.float64)[:, None, None], [1, 4, 1])
    elif failure == 'divergence':
        overrides['native_divergence_count'] = 1
    elif failure == 'nonfinite_state':
        samples = tf.tensor_scatter_nd_update(samples, [[9, 2, 0]], [float('nan')])
    else:
        overrides['target_log_prob'] = tf.tensor_scatter_nd_update(
            tf.zeros(samples.shape[:2], tf.float64), [[9, 2]], [float('nan')])
    evidence = health(samples, **overrides)
    if failure.startswith('nonfinite'):
        assert evidence.evidence_validity != 'valid'
    else:
        expected = ('path_return_resonance_detected' if failure == 'cycle' else
                    'native_divergence_positive' if failure == 'divergence' else 'movement_gate_failed')
        assert expected in evidence.candidate_promotion_vetoes


def test_explicit_health_profile_has_new_identity_without_changing_legacy_defaults():
    raw = full_search_configuration('k0', seed=(20261002, 2202), wall_seconds=240)['policy']
    p = HMCReplicatedAcceptancePolicy.from_payload(raw)
    historical = replace(p, min_normalized_return_displacement=1.e-4)
    assert p.identity != historical.identity
    assert HMCReplicatedAcceptancePolicy.from_payload(historical.payload()) == historical
    assert HMCAcceptancePolicy().min_normalized_return_displacement == 1.e-4
    assert HMCReplicatedAcceptancePolicy.__dataclass_fields__['min_normalized_return_displacement'].default == 1.e-4
    assert p.min_movement_rate == historical.min_movement_rate == .05
    assert p.max_repeated_state_fraction == historical.max_repeated_state_fraction == .95
