"""Tiny backend localization check; not sampler or speed evidence."""
import os
if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
    raise RuntimeError('memory growth required')
import tensorflow as tf
for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
    assert tf.config.experimental.get_memory_growth(device)
tf.config.set_soft_device_placement(False)
@tf.function(input_signature=[tf.TensorSpec([4], tf.float64)], jit_compile=True, autograph=False)
def compute(x):
    def body(i, x):
        return i+1, tf.cond(i % 2 == 0, lambda: x+.5, lambda: x-.1)
    return tf.while_loop(lambda i, x: i < 4, body, (tf.constant(0), x))[1]
with tf.device('/GPU:0'):
    print('before conditional command-buffer launch', flush=True)
    result = compute(tf.ones([4], tf.float64))
    print('result', result.numpy().tolist(), result.device, flush=True)
