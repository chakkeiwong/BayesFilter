"""Independent normal/mixture/transport oracles; no MacroFinance runtime."""

import math
from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf
from bayesfilter.inference.neutra_training import (
    NeuTraReverseKLTrainer,
    quadratic_anchor_neutra_config,
)
from bayesfilter.inference.neutra_whitening import (
    WhiteningBankLedger,
    WhiteningPolicy,
    WhiteningTargetBatch,
    make_whitening_bank,
    qualify_frozen_whitening_bank,
    summarize_whitening_bank,
    validate_whitening_bank,
)

POLICY = WhiteningPolicy()


def bank(kind='typical', dimension=3, rows=2048, role=9201):
    return make_whitening_bank(kind=kind, dimension=dimension, rows=rows,
                               seed=(20260915, role), role='q0_oracle')


def gaussian_summary(b, shift=0., scale=1., additive=0.):
    theta = shift+scale*b.z
    ell = -.5*tf.reduce_sum(theta**2, axis=1)+int(b.z.shape[1])*math.log(scale)+additive
    score = -scale*theta
    return summarize_whitening_bank(b, ell, score, tf.ones([int(b.z.shape[0])], tf.bool), policy=POLICY)


@pytest.mark.parametrize('kind', ['typical', 'shell', 'tail'])
def test_dimension_142_gaussian_identity_passes_every_applicable_statistic(kind):
    b = bank(kind, dimension=142, role=9201+('typical', 'shell', 'tail').index(kind))
    result = gaussian_summary(b)
    assert result['statistics_passed'] and not result['integrity_checked']
    assert result['score_rms'] == 0 and result['density_rms'] < 1e-12
    if kind == 'shell':
        assert 'importance' not in result and b.log_proposal is None
    elif kind == 'tail':
        assert result['importance']['descriptive_only']
        assert result['importance']['ess_fraction'] < .2
        assert 'ess' not in result['gates'] and 'covariance' not in result['gates']


@pytest.mark.parametrize('shift,scale', [(2., 1.), (0., 2.), (0., .2)])
def test_deliberately_unwhitened_map_fails_expected_gate(shift, scale):
    result = gaussian_summary(bank(), shift, scale)
    assert not result['statistics_passed']
    assert not (result['gates']['score'] and result['gates']['covariance'])


def test_density_constants_cancel_and_rms_averages_coordinates_once():
    b = bank(dimension=142)
    ell = -.5*tf.reduce_sum(b.z**2, axis=1)
    score = -b.z+tf.ones_like(b.z)
    first = summarize_whitening_bank(b, ell, score, tf.ones([2048], tf.bool), policy=POLICY)
    second = summarize_whitening_bank(b, ell+12345., score, tf.ones([2048], tf.bool), policy=POLICY)
    assert first['score_rms'] == pytest.approx(1.)
    assert first['importance']['weighted_score_rms'] == pytest.approx(1.)
    assert first['density_rms'] == pytest.approx(second['density_rms'], abs=1e-10)
    assert not first['gates']['score']


@pytest.mark.parametrize('kind', ['tail', 'mixture'])
def test_correct_proposal_matches_independent_numpy_formula(kind):
    b = bank(kind, dimension=5, role=9204)
    z = b.z.numpy()
    a = -.5*5*math.log(2*math.pi)-.5*np.sum(z*z, axis=1)
    c = -.5*5*math.log(2*math.pi)-5*math.log(2)-np.sum(z*z, axis=1)/8
    expected = c if kind == 'tail' else np.logaddexp(a, c)-math.log(2)
    np.testing.assert_allclose(b.log_proposal.numpy(), expected, rtol=1e-12, atol=1e-12)
    result = gaussian_summary(b)
    if kind == 'mixture':
        assert result['optional_diagnostic'] and not result['statistics_passed']
    assert result['importance']['descriptive_only']


@pytest.mark.parametrize('defect', ['density', 'rows', 'seed', 'tail_as_typical', 'shell_denominator'])
def test_corrupted_or_mislabeled_bank_is_rejected(defect):
    b = bank('shell' if defect == 'shell_denominator' else 'tail')
    if defect == 'density':
        b = replace(b, log_proposal=b.log_proposal+1.)
    elif defect == 'rows':
        b = replace(b, z=tf.reverse(b.z, axis=[0]))
    elif defect == 'seed':
        b = replace(b, seed=(20260915, 9299))
    elif defect == 'tail_as_typical':
        b = replace(b, kind='typical')
    else:
        b = replace(b, log_proposal=tf.zeros([2048], tf.float64))
    with pytest.raises(ValueError):
        validate_whitening_bank(b)


@pytest.mark.parametrize('defect', ['nan_value', 'ineligible_row', 'infinite_score', 'zero_weights'])
def test_invalid_rows_or_weights_cannot_be_dropped(defect):
    b = bank()
    ell = -.5*tf.reduce_sum(b.z**2, axis=1)
    score, valid = -b.z, tf.ones([2048], tf.bool)
    if defect == 'nan_value':
        ell = tf.tensor_scatter_nd_update(ell, [[0]], [float('nan')])
    elif defect == 'ineligible_row':
        valid = tf.tensor_scatter_nd_update(valid, [[0]], [False])
    elif defect == 'infinite_score':
        score = tf.tensor_scatter_nd_update(score, [[0, 0]], [float('inf')])
    else:
        ell = tf.fill([2048], tf.constant(-math.inf, tf.float64))
    with pytest.raises(ValueError):
        summarize_whitening_bank(b, ell, score, valid, policy=POLICY)


def test_concentrated_typical_importance_is_a_nonpass():
    b = bank()
    ell = -.5*tf.reduce_sum(b.z**2, axis=1)
    ell = tf.tensor_scatter_nd_add(ell, [[0]], [100.])
    result = summarize_whitening_bank(b, ell, -b.z, tf.ones([2048], tf.bool), policy=POLICY)
    assert not result['statistics_passed'] and not result['gates']['ess']


def test_ledger_rejects_reuse_across_reload_map_names_and_inspected_roles(tmp_path):
    b = bank()
    ledger = WhiteningBankLedger(tmp_path)
    ledger.reserve(b, map_id='first', target_signature='1'*64, purpose='inspection')
    fresh = WhiteningBankLedger(tmp_path)
    with pytest.raises(ValueError, match='already reserved or inspected'):
        fresh.reserve(replace(b, role='renamed'), map_id='second', target_signature='1'*64)
    with pytest.raises(ValueError, match='already reserved or inspected'):
        fresh.reserve(bank('tail'), map_id='second', target_signature='1'*64)


class GaussianTarget:
    parameter_dim = 2
    parameter_names = ('x', 'y')

    def __init__(self, quartic=0.):
        self.config = self
        self.quartic = quartic
        self.invalid = False

    def target_signature(self):
        return '1'*64

    def adapter_signature(self):
        return '2'*64

    def signature_payload(self):
        return {'parameter_transform': {'orientation': 'identity', 'inverse_orientation': 'identity'}}

    def evaluate(self, theta):
        value = -tf.reduce_sum(.5*theta**2+self.quartic*theta**4, axis=1)
        score = -theta-4*self.quartic*theta**3
        return WhiteningTargetBatch(value, score, tf.fill([int(theta.shape[0])], not self.invalid), self.target_signature())


def payload(scale=1., shift=0., quartic=0.):
    target = GaussianTarget(quartic)
    config = quadratic_anchor_neutra_config(dimension=2, initial_output_shift=[shift, shift],
        initial_output_scale_log=[math.log(scale)]*2, initial_anchor_factor=[[scale, 0.], [0., scale]],
        target_parameter_names=target.parameter_names, target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(), anchor_estimator_signature='3'*64,
        initialization_seed=(20260915, 9205), anchor_release_steps=0, scale_transform='identity', jit_compile=False)
    trainer = NeuTraReverseKLTrainer(target, config)
    return trainer.frozen_transport_payload(transport_id='q0-gaussian-oracle', target_signature=target.target_signature()), target


@pytest.mark.parametrize('kind', ['typical', 'shell', 'tail'])
def test_full_canonical_identity_jacobian_score_curvature_and_statistics(tmp_path, kind):
    artifact, target = payload()
    b = bank(kind, dimension=2, rows=512, role=9201+('typical', 'shell', 'tail').index(kind))
    result = qualify_frozen_whitening_bank(artifact, target.evaluate, b, policy=POLICY,
        ledger=WhiteningBankLedger(tmp_path), curvature_seed=(20260915, 9205), chunk_rows=32)
    assert result['passed'], result
    assert result['integrity_checked'] and result['minimum_jacobian_singular_value'] == pytest.approx(1.)
    assert result['target_rows'] == 2*512+9*2*2*2
    assert all(x['passed'] for x in result['curvature'])


@pytest.mark.parametrize('scale,quartic', [(2., 0.), (1., .2)])
def test_full_unwhitened_or_nonlinear_target_is_rejected(tmp_path, scale, quartic):
    artifact, target = payload(scale=scale, quartic=quartic)
    result = qualify_frozen_whitening_bank(artifact, target.evaluate,
        bank('tail', dimension=2, rows=64, role=9203), policy=POLICY,
        ledger=WhiteningBankLedger(tmp_path), curvature_seed=(20260915, 9205))
    assert result['integrity_checked'], result
    assert not result['passed'] and any(not x['passed'] for x in result['curvature'])


def test_full_invalid_target_stops_and_keeps_bank_consumed(tmp_path):
    artifact, target = payload()
    target.invalid = True
    b = bank('typical', dimension=2, rows=32)
    ledger = WhiteningBankLedger(tmp_path)
    result = qualify_frozen_whitening_bank(artifact, target.evaluate, b,
        policy=POLICY, ledger=ledger, curvature_seed=(20260915, 9205))
    assert not result['passed'] and result['target_calls'] == 1 and result['target_rows'] == 32
    with pytest.raises(ValueError, match='already reserved'):
        qualify_frozen_whitening_bank(artifact, target.evaluate, b,
            policy=POLICY, ledger=ledger, curvature_seed=(20260915, 9205))
