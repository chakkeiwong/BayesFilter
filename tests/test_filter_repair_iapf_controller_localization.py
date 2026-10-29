"""Diagnostic localization of the preserved CV-threshold decision mismatch."""

import math
from pathlib import Path

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_optimization_barrier

from bayesfilter.score_study.iapf_controller_tf import make_iapf_iteration_decision
from scripts.run_filter_repair_campaign import OUTPUT
from tests.test_filter_repair_fitted_apf_fixed import _host, _save
from tests.test_filter_repair_iapf_controller import FIXTURE, _reference


def test_iapf_controller_arithmetic_localization(request):
    logs = FIXTURE['cases'][-1]['log_values']
    window = logs[-3:]
    shifted = [x-max(window) for x in window]
    values = [math.exp(x) for x in shifted]
    total = sum(values)
    mean = total/3
    squares = [(x-mean)**2 for x in values]
    sum_squares = sum(squares)
    variance = sum_squares/2
    root = math.sqrt(variance)
    reference = {'shifted': shifted, 'values': values, 'total': total, 'mean': mean,
                 'squares': squares, 'sum_squares': sum_squares, 'variance': variance, 'root': root, 'cv': root/mean}

    def factory(dynamic, barriers=False):
        @tf.function(input_signature=[tf.TensorSpec([3], tf.float64),
            tf.TensorSpec([2], tf.float64)], jit_compile=True, autograph=False)
        def evaluate(window, divisors):
            def divide(a, b):
                if barriers:
                    a, b = xla_optimization_barrier(input=[a, b])
                return a/b
            shifted = window-tf.reduce_max(window)
            values = tf.exp(shifted)
            def add(i, total):
                return i+1, total+values[i]
            total = tf.while_loop(lambda i, _: i < 3, add,
                (0, tf.constant(0., tf.float64)), maximum_iterations=3, parallel_iterations=1)[1]
            mean = divide(total, divisors[0] if dynamic else tf.constant(3., tf.float64))
            squares = tf.square(values-mean)
            def add_square(i, total):
                return i+1, total+tf.square(values[i]-mean)
            sum_squares = tf.while_loop(lambda i, _: i < 3, add_square,
                (0, tf.constant(0., tf.float64)), maximum_iterations=3, parallel_iterations=1)[1]
            variance = divide(sum_squares, divisors[1] if dynamic else tf.constant(2., tf.float64))
            root = tf.sqrt(variance)
            return {'shifted': shifted, 'values': values, 'total': total, 'mean': mean,
                    'squares': squares, 'sum_squares': sum_squares, 'variance': variance, 'root': root, 'cv': divide(root, mean)}
        return evaluate
    args = tf.constant(window, tf.float64), tf.constant([3., 2.], tf.float64)
    constant, dynamic = _host(factory(False)(*args)), _host(factory(True)(*args))
    barrier_dynamic = _host(factory(True, True)(*args))
    owner = make_iapf_iteration_decision(8, 2, 64)
    tau = math.nextafter(reference['cv'], math.inf)
    actual = _host(owner(tf.constant(logs+[float('nan')]*4, tf.float64),
        tf.constant([8]*4+[-1]*4, tf.int64), tf.constant(4), tf.constant(tau, tf.float64)))
    expected = _reference()(logs, [8]*4, k=2, tau=tau, max_particles=64)
    @tf.function(input_signature=[tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.float64)],
                 jit_compile=True, autograph=False)
    def scalar_divide(a, b):
        return a/b
    divisions = {}
    for label, a, b in [('mean', total, 3.), ('cv', root, mean)]:
        operands = tf.constant(a, tf.float64), tf.constant(b, tf.float64)
        divisions[label] = {'reference': a/b, 'compiled': float(scalar_divide(*operands))}
    directory = Path(request.config.getoption('xmlpath')).parent
    assert directory.parent == OUTPUT
    (directory/'constant.optimized_hlo.txt').write_text(factory(False).experimental_get_compiler_ir(*args)(stage='optimized_hlo'))
    (directory/'dynamic.optimized_hlo.txt').write_text(factory(True).experimental_get_compiler_ir(*args)(stage='optimized_hlo'))
    full_args = (tf.constant(logs+[float('nan')]*4, tf.float64), tf.constant([8]*4+[-1]*4, tf.int64),
                 tf.constant(4), tf.constant(tau, tf.float64))
    (directory/'barrier-owner.optimized_hlo.txt').write_text(owner.experimental_get_compiler_ir(*full_args)(stage='optimized_hlo'))
    _save(request, 'iapf-controller-localization', {'reference': reference,
        'compiled_constant_divisors': constant, 'compiled_dynamic_divisors': dynamic,
        'compiled_barrier_dynamic_divisors': barrier_dynamic,
        'actual_owner': actual, 'reference_decision': expected,
        'scalar_divisions': divisions,
        'constant_diagnostic_matches_owner_cv': constant['cv'] == actual['cv'],
        'runtime_repaired': False, 'scope': 'Arithmetic diagnosis; no threshold waiver or adaptive admission.'})
    assert reference['cv'] == expected['cv']
