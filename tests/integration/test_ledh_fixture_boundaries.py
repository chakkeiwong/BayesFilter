"""Distinguish physical support from floating-point chart failure in real LG1."""
import tensorflow as tf
from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target
from bayesfilter.highdim.ledh_failure_status_tf import FailureCode,FailureStage

def test_sigmoid_roundoff_and_nonfinite_coordinates_are_numerical_exclusions():
    data=dict(model_id='LG1',horizon=10,partition='calibration',attempt=0,
        replicate=0,observations=[[0.]]*10)
    controls=dict(substeps=1,reset_policy='contract_e',reset_sinkhorn_steps=2,
        reset_balance_steps=2,reset_ridge=1e-5,correction_steps=1,
        pairwise_steps=1,correction_trust_radius=.1,coordinate_cap=16.)
    target=make_linear_fixture_target(data,8,controls,dtype=tf.float64)
    values,scores,status=target.compile(4,coordinates='unconstrained')(
        tf.constant([[0.],[1000.],[-1000.],[float('nan')]],tf.float64))
    tf.debugging.assert_equal(status['value_valid'],[True,False,False,False])
    tf.debugging.assert_equal(status['outside_declared_support'],[False]*4)
    tf.debugging.assert_equal(status['value_failure_code'][1:],[int(FailureCode.NONFINITE_INPUT)]*3)
    tf.debugging.assert_equal(status['force_failure_code'][1:],[int(FailureCode.NONFINITE_INPUT)]*3)
    tf.debugging.assert_equal(status['first_failure_stage'][1:],[int(FailureStage.PRIOR)]*3)
    tf.debugging.assert_equal(status['direction_value_valid'][1:],tf.zeros([3,1],tf.bool))
    tf.debugging.assert_equal(status['direction_force_valid'][1:],tf.zeros([3,1],tf.bool))
    tf.debugging.assert_equal(scores[1:],tf.zeros([3,1],tf.float64))
    tf.debugging.assert_equal(values[1:],tf.fill([3],tf.constant(-float('inf'),tf.float64)))
    _,_,physical=target.compile(2)(tf.constant([[.96],[float('nan')]],tf.float64))
    tf.debugging.assert_equal(physical['outside_declared_support'],[True,False])
    tf.debugging.assert_equal(physical['value_failure_code'],
        [int(FailureCode.OUTSIDE_SUPPORT),int(FailureCode.NONFINITE_INPUT)])
