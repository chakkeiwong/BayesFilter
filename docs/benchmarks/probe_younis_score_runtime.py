"""Tiny GPU/XLA compatibility check; no scientific comparison or tuning."""
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bayesfilter.score_study.runtime import configure_runtime, memory_usage

started = time.monotonic()
metadata = configure_runtime(device="GPU", tf32=True, jit_compile=True)
import tensorflow as tf
from bayesfilter.score_study.gaussian_tf import make_gaussian_kernel, make_particle_kernel, parameterized_model

theta = tf.constant([.62, -.8, -.6, .9, .25, -.3], tf.float32)
y = tf.constant([[.5], [-.2], [.8]], tf.float32)
initial = tf.random.stateless_normal([24, 2], [7, 9], dtype=tf.float32)
noise = tf.random.stateless_normal([3, 24, 2], [7, 10], dtype=tf.float32)
records = []
with tf.device('/GPU:0'):
    for kind in ('kalman', 'ukf', 'bootstrap', 'adapted'):
        begin = time.monotonic()
        if kind in ('kalman', 'ukf'):
            kernel = make_gaussian_kernel(2, 1, 6, 'float32', True, kind == 'ukf')
            arguments = (y, *parameterized_model(theta))
        else:
            kernel = make_particle_kernel(2, 1, 24, 3, 'float32', True, kind == 'adapted')
            arguments = (theta, y, initial, noise)
        actual = kernel(*arguments)
        reference = kernel.python_function(*arguments)
        for x, target in zip(actual, reference):
            if x.shape != target.shape or x.dtype != target.dtype:
                raise RuntimeError(f'{kind}: invalid traced result shape/dtype')
            tf.debugging.assert_all_finite(x, kind)
            tf.debugging.assert_near(x, target, rtol=3e-4, atol=3e-5)
        records.append({'kind': kind, 'value': float(actual[0]), 'score': actual[1].numpy().tolist(),
                        'trace_count': kernel.experimental_get_tracing_count(),
                        'device': actual[0].device, 'wall_seconds': time.monotonic() - begin})
metadata.update(memory_usage('GPU'))
payload = {'status': 'pass', 'evidence_class': 'GPU_XLA_mechanics_only', 'runtime': metadata,
           'kernels': records, 'wall_seconds': time.monotonic() - started}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps({'status': 'pass', 'kernels': len(records), 'wall_seconds': payload['wall_seconds']}))
