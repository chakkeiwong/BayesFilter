"""Independent IID/AR(1) uncertainty controls; NumPy is test-reference only."""
import numpy as np
import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_equivalence import _batch_statistics, assess_equivalence


def test_batch_mcse_detects_serial_dependence_against_ar1_long_run_variance():
    rng = np.random.default_rng(831)
    rho = .8
    values = rng.normal(size=(4, 2400, 6))
    for t in range(1,2400):
        values[:,t] = rho*values[:,t-1] + np.sqrt(1-rho*rho)*values[:,t]
    mcse, _ = _batch_statistics(2000)(tf.constant(values[:,400:]))
    expected = np.sqrt((1+rho)/(1-rho)/(4*2000))
    assert np.all(np.max(mcse.numpy(), axis=0) > expected*.7)
    assert np.all(np.max(mcse.numpy(), axis=0) < expected*1.5)
    assert np.all(np.max(mcse.numpy(), axis=0) > 2/np.sqrt(8000))


def test_equivalence_can_pass_reject_shift_and_refuse_stuck_chains():
    independent=tf.constant(np.random.default_rng(83).normal(size=(4,2000,6)))
    values=tf.repeat(independent[:,:,:1],6,axis=2)
    zero=tf.zeros([6],tf.float64)
    margins=tf.fill([6],tf.constant(.15,tf.float64))
    def assess(x):
        return assess_equivalence(x[:,:,:1],x,zero,zero,margins)
    assert assess(values)['status']=='EQUIVALENT'
    single=assess(values)
    double=assess_equivalence(values[:,:,:1],values,zero,zero,margins,comparison_families=2)
    assert all(a>b for a,b in zip(double['upper'],single['upper']))
    assert all(a<b for a,b in zip(double['lower'],single['lower']))
    assert assess(values+.5)['status']=='DISCREPANCY'
    # This fixed IID realization has a large empirical batch-lag diagnostic.
    # It must remain inconclusive despite its known generating distribution.
    assert assess(independent)['status']=='INCONCLUSIVE'
    result=assess(tf.zeros_like(values))
    assert result['status']=='INCONCLUSIVE' and not result['diagnostic_valid']
