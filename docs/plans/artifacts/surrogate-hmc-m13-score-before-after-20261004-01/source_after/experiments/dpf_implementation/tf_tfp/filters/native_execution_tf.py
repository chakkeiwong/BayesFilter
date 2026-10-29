"""Native recurrences and reporting boundaries for diagnostic particle routes."""

import tensorflow as tf
from tensorflow.python.eager import record


def map_fixed_rows(function, elements, output_signature):
    """Native row loop with static dense buffers for nested XLA pullbacks."""
    leaves = tf.nest.flatten(elements)
    count = leaves[0].shape[0]
    if count is None or any(value.shape[0] != count for value in leaves):
        raise ValueError("row mapping requires one shared fixed leading dimension")
    buffers = tf.nest.map_structure(lambda spec: tf.zeros([count, *spec.shape], spec.dtype), output_signature)
    def step(index, buffers):
        row = function(tf.nest.map_structure(lambda value: value[index], elements))
        buffers = tf.nest.map_structure(lambda buffer, value: tf.tensor_scatter_nd_update(
            buffer, tf.reshape(index, [1, 1]), value[None]), buffers, row)
        return index+1, buffers
    return tf.while_loop(lambda index, _: index < count, step, (tf.constant(0), buffers),
                          maximum_iterations=count, parallel_iterations=1)[1]


def tensor_diagnostics(diag, labels):
    """Extract a fixed tensor schema while preserving static report labels."""
    result = {}
    for key, value in diag.items():
        if value is None or isinstance(value, str):
            labels[key] = value
        elif tf.is_tensor(value) and value.dtype == tf.string:
            literal = tf.get_static_value(value)
            if literal is not None and hasattr(literal, "item"):
                literal = literal.item()  # Static text decoding, no numerical operation.
            if not isinstance(literal, bytes):
                raise TypeError("Diagnostic text must be static")
            labels[key] = literal.decode("utf-8")
        else:
            # Original reports were host scalars and never contributed to the
            # value derivative. In particular, zero projection-error norms
            # have undefined AD at zero and must stay outside the score path.
            result[key] = tf.stop_gradient(tf.convert_to_tensor(value))
    return result


def scan_steps(step, state, horizon):
    """Run a fixed-schema step with bounded native date control flow."""
    state, first = step(tf.constant(0), state)
    arrays = tf.nest.map_structure(lambda value: tf.concat([
        value[None], tf.zeros([horizon - 1, *value.shape], value.dtype)], axis=0), first)

    def body(date, state, arrays):
        state, output = step(date, state)
        arrays = tf.nest.map_structure(lambda array, value:
            tf.tensor_scatter_nd_update(array, tf.reshape(date, [1, 1]), value[None]), arrays, output)
        return date + 1, state, arrays

    _, state, arrays = tf.while_loop(lambda date, *_: date < horizon, body,
        (tf.constant(1), state, arrays), maximum_iterations=horizon, parallel_iterations=1)
    return state, arrays


def conditional_step(function, signature, *, fallback, jit_compile=True):
    """Trace a conditional result schema without evaluating its numerical body."""
    compiled = tf.function(function, input_signature=signature, autograph=False)
    result = compiled.get_concrete_function().structured_outputs
    specs = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, value.dtype), result)
    recomputed = compile_recomputed_diagnostic(function, signature, jit_compile=jit_compile)
    skipped = compile_recomputed_diagnostic(lambda *args: fallback(specs, *args),
        signature, jit_compile=jit_compile)
    return _piecewise_recomputed(recomputed, skipped, signature), specs


def _piecewise_recomputed(yes, no, signature):
    """Differentiate each selected branch without saving its loop intermediates."""
    captures = tuple({value.ref(): value for branch in (yes, no)
                      for value in branch.capture_values}.values())
    variables = tuple({value.ref(): value for branch in (yes, no)
                       for value in branch.variables}.values())
    captured_index = {value.ref(): i for i, value in enumerate(captures)}
    variable_index = {value.ref(): i for i, value in enumerate(variables)}
    input_count = len(signature)
    float_positions = tuple(i for i, spec in enumerate((*signature, *captures)) if spec.dtype.is_floating)

    @tf.custom_gradient
    def operation(predicate, *inputs):
        local_handles = tuple(v.handle if tf.executing_eagerly() else
            tf.compat.v1.get_default_graph().capture(v.handle) for v in variables)
        variable_zeros = tuple(tf.zeros_like(v.read_value()) for v in variables)

        def arguments(branch):
            return (*inputs[:input_count], *(inputs[input_count + captured_index[v.ref()]]
                                              for v in branch.capture_values))

        with record.stop_recording():
            result = tf.cond(predicate, lambda: yes.forward(*arguments(yes)),
                             lambda: no.forward(*arguments(no)))
        outputs = tf.nest.flatten(result)

        def grad(*cotangents, variables=None):
            def pullback(branch):
                upstream = tuple(tf.zeros_like(outputs[i]) if cotangents[i] is None else cotangents[i]
                                 for i in branch.output_positions)
                handles = tuple(local_handles[variable_index[v.ref()]] for v in branch.resource_variables)
                derivatives = branch.backward(*arguments(branch), *handles, *upstream)
                result = [tf.zeros_like(inputs[i]) for i in float_positions] + list(variable_zeros)
                for i, value in zip(branch.input_positions, derivatives[:len(branch.input_positions)]):
                    combined = i if i < input_count else input_count + captured_index[branch.capture_values[i-input_count].ref()]
                    result[float_positions.index(combined)] = value
                for variable, value in zip(branch.resource_variables, derivatives[len(branch.input_positions):]):
                    result[len(float_positions) + variable_index[variable.ref()]] = value
                return tuple(result)

            derivatives = tf.cond(predicate, lambda: pullback(yes), lambda: pullback(no))
            by_position = dict(zip(float_positions, derivatives[:len(float_positions)]))
            gradients = (None, *(by_position.get(i) for i in range(len(inputs))))
            if variables is None:
                return gradients
            return gradients, [derivatives[len(float_positions) + variable_index[v.ref()]] for v in variables]
        return result, grad

    return lambda predicate, *inputs: operation(predicate, *inputs, *captures)


def host_scalar(value):
    if not tf.is_tensor(value):
        return value
    return value.numpy().tolist() if tf.executing_eagerly() else value


def host_reports(history, labels):
    """Post-run serialization only; never feeds filtering or gradient control."""
    values = {key: host_scalar(value) for key, value in history.items()}
    count = len(next(iter(values.values())))
    return [{**labels, **{key: value[date] for key, value in values.items()}}
            for date in range(count)]


def compile_recomputed_diagnostic(evaluate, signature, *, jit_compile=True):
    """Keep diagnostic AD loop state inside one compiled VJP.

    Uses TensorFlow's own recompute_grad recording boundary. Unlike calling
    recompute_grad on a compiled forward, differentiation traces the Python
    tensor body inside the compiled backward, so TensorLists do not escape.
    Callback tensor captures are lifted to explicit custom-gradient arguments;
    captured Variables are also registered with the enclosing tape. Both are
    differentiated through the original traced finite program.
    """
    body = tf.function(evaluate, input_signature=signature, autograph=False)
    concrete = body.get_concrete_function()
    structure = concrete.structured_outputs
    flat = tf.nest.flatten(structure)
    positions = tuple(i for i, value in enumerate(flat) if value.dtype.is_floating)
    variable_list = tuple(concrete.variables)
    captures = tuple(concrete.captured_inputs)
    capture_positions = tuple(i for i, value in enumerate(captures) if value.dtype != tf.resource)
    capture_values = tuple(captures[i] for i in capture_positions)
    all_signature = [*signature, *(tf.TensorSpec(value.shape, value.dtype) for value in capture_values)]
    input_positions = tuple(i for i, spec in enumerate(all_signature) if spec.dtype.is_floating)
    resource_positions = tuple(i for i, value in enumerate(captures) if value.dtype == tf.resource)
    resource_values = tuple(captures[i] for i in resource_positions)
    resource_signature = [tf.TensorSpec(value.shape, value.dtype) for value in resource_values]
    resource_by_variable = {v.ref(): next(i for i, handle in enumerate(resource_values)
        if handle is v.handle) for v in variable_list}

    def replay(*args, resources=None):
        bound = list(captures)
        for position, value in zip(capture_positions, args[len(signature):]):
            bound[position] = value
        if resources is not None:
            for position, value in zip(resource_positions, resources):
                bound[position] = value
        # TensorFlow's concrete-call API lets callback captures become inputs
        # without mutating the cached function or replacing its mathematics.
        return concrete._call_flat(list(args[:len(signature)]), captured_inputs=bound)

    forward = tf.function(replay, input_signature=all_signature, jit_compile=jit_compile, autograph=False)

    def backward(*args):
        inputs = args[:len(all_signature)]
        resources = args[len(all_signature):len(all_signature) + len(resource_values)]
        upstream = args[len(all_signature) + len(resource_values):]
        differentiable = tuple(inputs[i] for i in input_positions)
        with tf.GradientTape() as tape:
            tape.watch(differentiable)
            tape.watch(resources)
            outputs = tf.nest.flatten(replay(*inputs, resources=resources))
        return tape.gradient(tuple(outputs[i] for i in positions), (*differentiable, *resources),
                             output_gradients=upstream, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    backward = tf.function(backward, input_signature=[*all_signature, *resource_signature,
        *(tf.TensorSpec(flat[i].shape, flat[i].dtype) for i in positions)],
        jit_compile=jit_compile, autograph=False)

    @tf.custom_gradient
    def operation(*inputs):
        # stop_recording also suppresses automatic variable-access registration.
        # Register before it so custom_gradient can connect its variable VJP.
        for variable in variable_list:
            variable.read_value()
        local_resources = resource_values if tf.executing_eagerly() else tuple(
            tf.compat.v1.get_default_graph().capture(handle) for handle in resource_values)
        with record.stop_recording():
            result = forward(*inputs)
        outputs = tf.nest.flatten(result)

        def grad(*cotangents, variables=None):
            upstream = tuple(tf.zeros_like(outputs[i]) if cotangents[i] is None
                             else cotangents[i] for i in positions)
            derivatives = backward(*inputs, *local_resources, *upstream)
            by_position = dict(zip(input_positions, derivatives[:len(input_positions)]))
            input_gradients = tuple(by_position.get(i) for i in range(len(all_signature)))
            if variables is None:
                return input_gradients
            return input_gradients, [derivatives[len(input_positions) + resource_by_variable[v.ref()]] for v in variables]

        return result, grad

    # Leave the custom-gradient boundary visible to an external diagnostic tape;
    # both numerical forward and backward functions are must-compile by default.
    def call(*inputs):
        return operation(*inputs, *capture_values)

    call.get_concrete_function = lambda: forward.get_concrete_function()
    call.experimental_get_tracing_count = forward.experimental_get_tracing_count
    call.python_function = evaluate
    call.forward = forward
    call.traced_body = concrete
    call.capture_values = capture_values
    call.variables = variable_list
    call.resource_variables = tuple(next(v for v in variable_list if v.handle is handle)
                                    for handle in resource_values)
    call.input_positions, call.output_positions = input_positions, positions
    call.backward = backward
    return call
