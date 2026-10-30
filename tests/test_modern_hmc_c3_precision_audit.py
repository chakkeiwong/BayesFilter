"""Independent arithmetic and refusal invariants; no statistical promotion."""
import math

import numpy as np
import pytest

from scripts.diagnostics import audit_modern_hmc_c3_precision as audit


def test_independent_batches_equal_direct_loop_and_keep_chain_boundaries():
    x = np.arange(126 * 4 * 2, dtype=float).reshape(126, 4, 2)
    x = np.sin(x / 13) + np.arange(4)[None, :, None]
    out = audit.batch_report(x, 6)
    def direct(b):
        means = np.array([x[j:j+b].mean(axis=0) for j in range(0, len(x)-b+1, b)])
        return b * means.var(axis=0, ddof=1)
    np.testing.assert_allclose(out['lugsail_lrv'], 2*direct(6)-direct(2))
    np.testing.assert_allclose(out['variance'], out['lugsail_lrv'].sum(axis=0)/(16*len(x)))
    shifted = audit.batch_report(x + 37, 6)
    np.testing.assert_allclose(out['lugsail_lrv'], shifted['lugsail_lrv'], atol=1e-12)
    assert not audit.batch_report(x, 12)['available']


def test_zero_variance_is_unavailable_not_success():
    out = audit.batch_report(np.ones((120, 4, 2)), 6)
    assert not out['valid'].any()
    assert audit.native(out)['mcse'] == [None, None]


def test_nonfinite_and_wrong_shapes_rejected():
    with pytest.raises(ValueError, match='finite'):
        audit.batch_report(np.full((120, 4, 2), np.nan), 6)
    with pytest.raises(ValueError, match='finite'):
        audit.batch_report(np.ones((120, 2)), 6)
    with pytest.raises(ValueError, match='arithmetic'):
        audit.close([1.], [1.01], 'fixture')


def test_internal_precision_does_not_imply_reference_agreement():
    ref = dict(sd=[1.], mean=[0.], screen=dict(uncertainty=[.001]))
    out = audit.reference_decomposition(np.array([.06]), np.array([.018]), ref)
    assert .018 <= .02
    assert not out['passed']
    assert out['screen'][0] == pytest.approx(.115)
    assert sum(out[k][0] for k in ('error', 'uncertainty_margin', 'reference_uncertainty')) == pytest.approx(.115)


def test_hidden_common_shift_does_not_change_mcse():
    x = np.random.default_rng(7821).normal(size=(400, 4, 1))
    original = audit.batch_report(x, 20)
    shifted = audit.batch_report(x+5, 20)
    np.testing.assert_allclose(original['mcse'], shifted['mcse'], atol=1e-12)
    ref = dict(sd=[1.], mean=[0.], screen=dict(uncertainty=[.001]))
    assert not audit.reference_decomposition((x+5).mean(axis=(0,1)), shifted['mcse'], ref)['passed']


def test_reference_scale_is_not_silently_repaired():
    with pytest.raises(ValueError, match='invalid reference'):
        audit.reference_decomposition(np.array([0.]), np.array([.01]),
            dict(sd=[0.], mean=[0.], screen=dict(uncertainty=[.001])))


def test_saved_active_coordinates_require_issued_nonidentity_map():
    execution = dict(layers=[dict(kind='fixed_mass', artifact=dict(
        position=[3., -2.], factor=[[2., 0.], [.5, 4.]],
        factor_orientation='row_right_transpose'))])
    active = np.array([[[1., 2.], [-1., 3.]]])
    expected = np.array([[[5., 6.5], [1., 9.5]]])
    np.testing.assert_array_equal(audit.positions_from_active(active, execution), expected)
    with pytest.raises(ValueError, match='arithmetic mismatch'):
        audit.close(active, expected, 'coordinates')
    execution['layers'][0]['artifact']['factor_orientation'] = 'row_right'
    with pytest.raises(ValueError, match='orientation'):
        audit.positions_from_active(active, execution)


def test_execution_hash_changes_with_geometry():
    before = audit.payload_digest(dict(center=[0., 1.], factor=[[1., 0.], [0., 1.]]))
    after = audit.payload_digest(dict(center=[0., 2.], factor=[[1., 0.], [0., 1.]]))
    assert before != after
