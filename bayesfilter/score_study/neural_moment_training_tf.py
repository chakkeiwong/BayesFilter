"""Neural-parameter gradients of the actual corrected LEDH moment loss.

Model scores remain the shared analytical recursion. A TensorFlow loop sums
independent full-consumer gradients before one minibatch optimizer update.
Function bodies are traced into the enclosing training graph so gradient shape
intermediates cannot specialize an XLA executable to the minibatch values.
"""

import tensorflow as tf


def make_neural_moment_trainer(network, endpoint, *, batch_size, learning_rate,
                              jit_compile=True):
    if batch_size < 2 or learning_rate <= 0.:
        raise ValueError("Neural training requires a batch and positive learning rate")
    parameters = network.parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
        beta_1=.9, beta_2=.999, epsilon=1e-8)
    optimizer.build(parameters)
    scalar_signature = list(endpoint.input_signature[:9]) + [endpoint.input_signature[-1]]
    batch_signature = [tf.TensorSpec([batch_size] + item.shape.as_list(), item.dtype)
                       for item in scalar_signature]

    @tf.function(input_signature=scalar_signature, jit_compile=jit_compile, autograph=False)
    def row_gradient(*arguments):
        with tf.GradientTape() as tape:
            coefficients = network.predict.python_function(arguments[5])
            result = endpoint.python_function(*arguments[:-1], *coefficients, arguments[-1])
        gradients = tape.gradient(result["loss"], parameters)
        if any(gradient is None for gradient in gradients):
            raise ValueError("A neural tensor is disconnected from the corrected moment loss")
        valid = result["valid"] & tf.reduce_all(tf.stack([
            tf.reduce_all(tf.math.is_finite(gradient)) for gradient in gradients]))
        return result["loss"], result["blocks"], gradients, valid

    @tf.function(input_signature=batch_signature, jit_compile=jit_compile, autograph=False)
    def update(*arguments):
        def body(index, loss, blocks, gradients, valid):
            row_loss, row_blocks, row_gradients, row_valid = row_gradient.python_function(
                *[argument[index] for argument in arguments])
            return (index + 1, loss + row_loss, blocks + row_blocks,
                    [total + value for total, value in zip(gradients, row_gradients)],
                    valid & row_valid)

        _, loss, blocks, gradients, valid = tf.while_loop(
            lambda index, *_: index < batch_size, body,
            (tf.constant(0), tf.zeros([], parameters[0].dtype),
             tf.zeros([2], parameters[0].dtype), [tf.zeros_like(item) for item in parameters],
             tf.constant(True)), parallel_iterations=1, maximum_iterations=batch_size)
        divisor = tf.cast(batch_size, loss.dtype)
        gradients = [gradient / divisor for gradient in gradients]
        norms = tf.stack([tf.linalg.norm(gradient) for gradient in gradients])

        def apply():
            optimizer.apply_gradients(zip(gradients, parameters))
            return tf.constant(True)

        applied = tf.cond(valid, apply, lambda: tf.constant(False))
        return dict(loss=loss / divisor, blocks=blocks / divisor,
                    gradient_norms=norms, valid=valid, applied=applied,
                    optimizer_step=optimizer.iterations)

    return update, row_gradient, optimizer
