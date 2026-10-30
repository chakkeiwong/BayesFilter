"""Compiled source-route weights and affine-coordinate calculations.

The formulas are unchanged (paper section 4.1; author models/full_sol.m:
32--40, 81--84, 98--99, 123). Frozen affine maps are fixed-HMC adaptations;
this execution layer does not establish the fitting route's source fidelity.
"""

from collections import OrderedDict
from functools import wraps
from inspect import signature

import tensorflow as tf

from bayesfilter.highdim.diagnostics import HighDimStatus
from bayesfilter.ops.fixed_signature_tf import fixed_signature_function
from bayesfilter.ops.slogdet_tf import slogdet_tf

D = tf.float64
_METHOD_PROGRAMS = OrderedDict()


def finite(value, name):
    """Preserve host rejection; invalid inputs remain nonfinite inside XLA."""
    valid = tf.reduce_all(tf.math.is_finite(value))
    if tf.inside_function():
        return tf.where(valid, value, tf.constant(float("nan"), value.dtype))
    if not bool(valid):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
    return value


def compiled(function):
    """Compile a flat tensor calculation while retaining its public signature."""
    program = fixed_signature_function(floating_dtype=D)(function)

    @wraps(function)
    def evaluate(*args, **kwargs):
        if tf.inside_function():
            return function(*args, **kwargs)
        return finite(program(*args, **kwargs), function.__name__)

    evaluate.compiled = program
    return evaluate


def compiled_method(function):
    """Compile complete tensor methods of prepared immutable route objects.

    The bounded cache retains the owning object, preventing identity reuse.
    Query tensors are always arguments with fixed shape/dtype signatures.
    Enclosing graphs inline the numerical method, including callback execution.
    """
    api = signature(function)

    def prepare(owner, *args, **kwargs):
        from bayesfilter.highdim.source_route_runtime_tf import _tensor_program

        bound = api.bind(owner, *args, **kwargs)
        bound.apply_defaults()
        names = tuple(bound.arguments)[1:]
        inputs = tuple(tf.convert_to_tensor(bound.arguments[name], D) for name in names)
        specs = tuple(tf.TensorSpec(value.shape, value.dtype) for value in inputs)
        key = (function, id(owner), specs)
        if key not in _METHOD_PROGRAMS:
            def numerical(*values):
                return function(owner, **dict(zip(names, values, strict=True)))

            program = _tensor_program(numerical, specs, True)
            _METHOD_PROGRAMS[key] = (owner, program)
            if len(_METHOD_PROGRAMS) > 16:
                _METHOD_PROGRAMS.popitem(last=False)
        _METHOD_PROGRAMS.move_to_end(key)
        return _METHOD_PROGRAMS[key][1], inputs

    def program_for(owner, *args, **kwargs):
        return prepare(owner, *args, **kwargs)[0]

    @wraps(function)
    def evaluate(owner, *args, **kwargs):
        if tf.inside_function():
            return function(owner, *args, **kwargs)
        program, inputs = prepare(owner, *args, **kwargs)
        return finite(program(*inputs), function.__name__)

    evaluate.program_for = program_for
    return evaluate


def positive(value, name):
    """Preserve a positive-density veto inside an enclosing compiled target."""
    value = finite(value, name)
    valid = tf.reduce_all(value > 0.0)
    if tf.inside_function():
        return tf.where(valid, value, tf.constant(float("nan"), value.dtype))
    if not bool(valid):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
    return value


@compiled
def log_abs_det(matrix):
    return slogdet_tf(finite(matrix, "matrix"))[1]


@compiled
def physical_points(reference, mu, matrix):
    return tf.linalg.matmul(finite(matrix, "matrix"), finite(reference, "reference_points")) + finite(mu, "mu")[:, None]


@compiled
def reference_log_density(log_physical_density, matrix):
    return finite(log_physical_density, "log_physical_density") + log_abs_det(matrix)


@compiled
def normalizer_increment(log_transport_normalizer, shift_constant):
    return finite(log_transport_normalizer, "log_transport_normalizer") - finite(shift_constant, "shift_constant")
