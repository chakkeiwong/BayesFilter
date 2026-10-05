"""Fixed XLA boundaries whose full pullbacks keep loop tapes internal.

This changes the TensorFlow execution boundary, not the derivative definition.
Captured tensor coefficients and variable resources retain their derivatives.
"""

import tensorflow as tf
from tensorflow.python.eager.polymorphic_function.polymorphic_function import (
    OptionalXlaContext,
)
from tensorflow.python.ops.control_flow_util import GraphOrParentsInXlaContext


def in_xla_context():
    """Whether the enclosing graph will compile its local control flow."""
    return GraphOrParentsInXlaContext(tf.compat.v1.get_default_graph())


def _capture_pullback_coefficient(value, graph):
    # Case/While gradient inputs must be graph tensors. Lift an eager
    # coefficient through the enclosing function before binding the branch.
    with graph.as_default():
        if not tf.is_symbolic_tensor(value) and getattr(graph, "is_control_flow_graph", False):
            value = _capture_pullback_coefficient(value, graph.outer_graph)
        return graph.capture(value)


def tensor_program(function, signature, jit_compile):
    """Keep loop tapes internal without dropping captured tensor derivatives.

    The custom rule is inside the compiled boundary, so both the primal and
    its recomputed complete VJP compile without exporting loop TensorLists.
    Explicitly thread every captured tensor through that rule: a caller's tape
    may watch a target coefficient or frame even though it lives on an object.
    """
    # The returned public boundary encloses the finite program and pullback.
    compile_gradients = bool(jit_compile or in_xla_context())
    raw = tf.function(function, input_signature=signature, jit_compile=False, autograph=False)
    with OptionalXlaContext(compile_gradients):
        concrete = raw.get_concrete_function()
    argument_count = len(signature)
    tensor_indices = tuple(index for index, value in enumerate(concrete.captured_inputs)
                           if value.dtype != tf.resource)
    positions = dict(zip(tensor_indices, range(len(tensor_indices)), strict=True))

    def bound_captures(tensors):
        # Variable handles stay captures and use custom_gradient's variable
        # pullback. Passing the same resource as an argument and capture would
        # duplicate a variable in the XLA cluster.
        return [tensors[positions[index]] if index in positions else value
                for index, value in enumerate(concrete.captured_inputs)]

    captures = tuple(concrete.captured_inputs[index] for index in tensor_indices)
    variables = tuple(concrete.variables)
    output_specs = tuple(tf.TensorSpec(value.shape, value.dtype)
                         for value in tf.nest.flatten(concrete.structured_outputs))
    pullback_specs = (*signature, *(tf.TensorSpec(value.shape, value.dtype) for value in captures),
                      *output_specs)
    input_count = argument_count + len(captures)

    @tf.function(input_signature=pullback_specs, jit_compile=False, autograph=False)
    def derivative(*values):
        arguments, tensor_captures = values[:argument_count], values[argument_count:input_count]
        cotangent = values[input_count:]
        graph = tf.compat.v1.get_default_graph()
        handles = tuple(graph.capture(variable.handle) for variable in variables)
        sources = (*arguments, *tensor_captures, *handles)
        outputs = concrete._call_flat(list(arguments), captured_inputs=bound_captures(tensor_captures))
        # Boolean and integer outputs can be returned by a finite program as
        # status/metadata fields, but TensorFlow has no default derivative for
        # them.  A ``None`` cotangent excludes those leaves while preserving
        # the numeric pullback of every differentiable output.
        gradient_ys = tf.nest.map_structure(
            lambda value, spec: value if spec.dtype.is_floating or spec.dtype.is_complex else None,
            cotangent, output_specs,
        )
        return tf.compat.v1.gradients(tf.nest.flatten(outputs), sources, grad_ys=gradient_ys,
                                      unconnected_gradients=tf.UnconnectedGradients.ZERO)

    derivative_concrete = None

    def complete_derivative():
        nonlocal derivative_concrete
        # A value-only call needs no pullback graph. Build it once on demand
        # under the primal's original context, including when an init_scope
        # paused tape recording during preparation. The VJP still stays inside
        # the compiled boundary and never exports loop TensorLists.
        if derivative_concrete is None:
            with concrete.graph.as_default(), OptionalXlaContext(compile_gradients):
                derivative_concrete = derivative.get_concrete_function()
        return derivative_concrete

    @tf.custom_gradient
    def evaluate(*values):
        graph = tf.compat.v1.get_default_graph()
        values = tuple(graph.capture(value) for value in values)
        arguments, tensor_captures = values[:argument_count], values[argument_count:]
        # Bind primal resources before TensorFlow constructs a Case/While
        # gradient, so that gradient does not introduce eager resource handles.
        primal_captures = {value.ref(): graph.capture(value) for value in concrete.captured_inputs}
        result = concrete._call_flat(list(arguments), captured_inputs=bound_captures(tensor_captures))

        def pullback(*cotangent, variables=None):
            backward = complete_derivative()

            def capture(value):
                if value.ref() in primal_captures:
                    return primal_captures[value.ref()]
                # A nested custom gradient can introduce coefficients unused
                # by its primal. Bind those in the current backward graph.
                return _capture_pullback_coefficient(value, tf.compat.v1.get_default_graph())

            derivative_captures = [capture(value) for value in backward.captured_inputs]
            incoming = tuple(tf.zeros(spec.shape, spec.dtype) if value is None else value
                             for value, spec in zip(cotangent, output_specs, strict=True))
            gradients = backward._call_flat(
                [*values, *incoming], captured_inputs=derivative_captures)
            if variables is None:
                return gradients[:input_count]
            # TensorFlow supplies the same resource variables in its own order.
            by_variable = {variable.ref(): gradient for variable, gradient in
                           zip(concrete.variables, gradients[input_count:], strict=True)}
            return gradients[:input_count], [by_variable[variable.ref()] for variable in variables]

        return result, pullback

    program = tf.function(lambda *arguments: evaluate(*arguments, *captures),
        input_signature=signature, jit_compile=jit_compile, autograph=False)
    program.inline_function = function
    return program



def call_tensor_program(program, arguments, *, jit_compile=True):
    """Call a nested-signature program without exporting TensorList tapes."""
    if tf.inside_function():
        return program.python_function(*arguments)
    if not hasattr(program, "_external_pullback_program"):
        signature = program.input_signature

        def flattened(*values):
            return program.python_function(*tf.nest.pack_sequence_as(signature, values))

        program._external_pullback_program = tensor_program(
            flattened, tf.nest.flatten(signature), jit_compile)
    return program._external_pullback_program(*tf.nest.flatten(arguments))
