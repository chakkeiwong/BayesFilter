"""Independent diagnostic oracles for TFP quantile precision, not calibration."""
import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.stats import beta, norm
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.hmc_precision import quantile_precision


def reference(values, probability):
    """Official TFP ESS plus independently computed NumPy/SciPy order statistics."""
    n = len(values)
    point = np.quantile(values, probability, axis=(0, 1), method='linear')
    split = np.concatenate((values[:n//2], values[-(n//2):]), axis=1)
    ess = tfp.mcmc.effective_sample_size(
        tf.constant((split <= point).astype(float)), cross_chain_dims=1,
        filter_beyond_positive_pairs=True, filter_threshold=None).numpy()
    ordered = np.sort(values.reshape(-1, values.shape[-1]), axis=0)
    se = np.full(values.shape[-1], np.nan)
    for i, effective in enumerate(ess):
        if not np.isfinite(effective) or effective <= 0:
            continue
        bounds = beta.ppf(norm.cdf([-1., 1.]), effective*probability+1,
                          effective*(1-probability)+1)
        lower = max(int(np.floor(bounds[0]*len(ordered))), 1)-1
        upper = min(int(np.ceil(bounds[1]*len(ordered))), len(ordered))-1
        if ordered[upper, i] > ordered[lower, i]:
            se[i] = (ordered[upper, i]-ordered[lower, i])/2
    return point, ess, se


@pytest.mark.parametrize('kind', ['iid','persistent','antithetic','beta','ties','constant'])
@pytest.mark.parametrize('probability', [.05,.5,.95])
def test_quantile_precision_matches_independent_reference(kind, probability):
    rng = np.random.default_rng(92419)
    values = rng.normal(size=(257,4,2))
    if kind in ('persistent','antithetic'):
        rho = .93 if kind=='persistent' else -.85
        values = lfilter([1.], [1.,-rho], rng.normal(size=(1281,4,2)),axis=0)[1024:]
    elif kind=='beta':
        values = rng.beta(.3,4.,size=values.shape)
    elif kind=='ties':
        values = np.repeat(values[::8],8,axis=0)[:257]
    elif kind=='constant':
        values[:] = .1
    point, ess, se = reference(values, probability)
    actual = quantile_precision(tf.constant(values),probability)
    np.testing.assert_allclose(actual['estimate'],point,rtol=2e-13,atol=2e-15)
    np.testing.assert_allclose(actual['indicator_ess'],ess,rtol=2e-11,atol=2e-11,equal_nan=True)
    np.testing.assert_allclose(actual['mcse'],se,rtol=2e-11,atol=2e-15,equal_nan=True)
    np.testing.assert_array_equal(actual['valid'],np.isfinite(se))


def test_tied_cutoff_is_not_moved_across_its_indicator():
    endpoint=np.float64.fromhex('0x1.7bc004f322299p-12')
    values=np.full((257,4,1),endpoint)
    for chain in range(4):
        values[:11+chain,chain]=endpoint/2
        values[-14-chain:,chain]=endpoint*2
    point,ess,se=reference(values,.5)
    actual=quantile_precision(tf.constant(values),.5)
    assert float(actual['estimate'][0])==point[0]
    np.testing.assert_allclose(actual['indicator_ess'],ess,rtol=2e-11)
    np.testing.assert_allclose(actual['mcse'],se,equal_nan=True)


@pytest.mark.parametrize('probability',[.5,.95,.99])
def test_unobserved_event_cannot_supply_quantile_precision(probability):
    actual=quantile_precision(tf.zeros((257,4,1),tf.float64),probability)
    assert not bool(actual['valid'][0])
    assert np.isnan(actual['mcse'][0])
