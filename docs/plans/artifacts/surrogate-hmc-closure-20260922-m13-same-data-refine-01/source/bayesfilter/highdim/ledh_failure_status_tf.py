"""Numeric status for LEDH consumers; no caller-issued route/admission identity."""
from enum import IntEnum
import tensorflow as tf


class FailureCode(IntEnum):
    OK = 0
    NONFINITE_INPUT = 1
    FACTORIZATION = 2
    INVALID_TANGENT = 3
    RESET_CORRECTION = 4
    CALLBACK = 5
    NONFINITE_PROPOSAL = 6
    DIRECTION_MISMATCH = 7
    OUTSIDE_SUPPORT = 8


class FailureStage(IntEnum):
    INITIAL = 0
    PREDICT = 1
    FLOW = 2
    WEIGHTS = 3
    UPDATE = 4
    RESET = 5
    CORRECTION = 6
    CALLBACK = 7
    DIRECTIONS = 8
    PRIOR = 9


def initial_status():
    return {
        'value_valid':tf.constant(True), 'force_valid':tf.constant(True),
        'outside_declared_support':tf.constant(False),
        'value_failure_code':tf.constant(0,tf.int32),
        'force_failure_code':tf.constant(0,tf.int32),
        'first_failure_time':tf.constant(-1,tf.int32),
        'first_failure_stage':tf.constant(-1,tf.int32),
        'first_failure_particle':tf.constant(-1,tf.int32),
    }


def record_failure(status, value_valid, force_valid, *, time, stage,
                   value_code=FailureCode.NONFINITE_INPUT,
                   force_code=FailureCode.INVALID_TANGENT, particle=-1):
    """Accumulate failure without ever reviving an invalid trajectory."""
    value_valid = tf.reduce_all(value_valid)
    force_valid = value_valid & tf.reduce_all(force_valid)
    first = (status['first_failure_stage'] < 0) & ~(value_valid & force_valid)
    result = dict(status)
    result['value_valid'] = status['value_valid'] & value_valid
    result['force_valid'] = status['force_valid'] & force_valid
    result['value_failure_code'] = tf.where(status['value_valid'] & ~value_valid, int(value_code), status['value_failure_code'])
    result['force_failure_code'] = tf.where(status['force_valid'] & ~force_valid,
        tf.where(value_valid, int(force_code), int(value_code)), status['force_failure_code'])
    for key, value in [('first_failure_time',time),('first_failure_stage',int(stage)),('first_failure_particle',particle)]:
        result[key] = tf.where(first, tf.cast(value,tf.int32), status[key])
    return result


def first_invalid_particle(valid):
    valid = tf.reshape(valid,[-1])
    return tf.where(tf.reduce_all(valid),-1,tf.argmax(tf.cast(~valid,tf.int32),output_type=tf.int32))
