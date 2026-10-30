"""CPU-only initialization repair checks; no M13 posterior claim."""
import math
import pytest
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_m13_initialization_repair import geometry_from_warmup,initialization_config
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config

def test_archived_warmup_initialization_keeps_target_and_checks():
    original=tuning_config("diagnostic")
    repaired,metadata=initialization_config(original)
    assert repaired.nomination==original.nomination
    assert repaired.warmup_results==2000 and repaired.verification_results==1000
    assert repaired.energy_error_max==original.energy_error_max
    assert repaired.fresh_seed!=original.fresh_seed
    assert metadata["retained_draws_used"]==0
    assert len(set(repaired.initial_chain_positions))==4
    tf.debugging.assert_near(tf.constant(repaired.initial_chain_positions,tf.float64),
        tf.constant([[-.22102605989967553,-.38583434670768396],[-.22102605989967553,.1458421290084731],
            [.44956990339529534,-.47289501606563383],[.44956990339529534,.05878145965052323]],tf.float64),
        atol=1e-12,rtol=1e-12)

def test_geometry_uses_only_last_half_and_preserves_positive_prior_covariance():
    values=tf.concat([tf.fill([256,4,2],tf.constant(999.,tf.float64)),
                      tf.ones([256,4,2],tf.float64)],0)
    starts,center,covariance=geometry_from_warmup(values)
    tf.debugging.assert_equal(center,tf.ones([2],tf.float64))
    tf.debugging.assert_near(covariance,tf.eye(2,dtype=tf.float64)*.1*math.log(2.)**2)
    assert bool(tf.reduce_all(tf.linalg.eigvalsh(covariance)>0))
    assert len(set(tuple(x) for x in starts.numpy().tolist()))==4

def test_invalid_geometry_rejected():
    with pytest.raises(tf.errors.InvalidArgumentError):
        geometry_from_warmup(tf.fill([512,4,2],tf.constant(float("nan"),tf.float64)))
